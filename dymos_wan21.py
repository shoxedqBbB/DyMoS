import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModel

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import (
    WanAttnProcessor, _get_qkv_projections)
from diffusers.utils import export_to_video


DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


class _State:
    def __init__(self, gamma, apply_from_step, apply_until_step, forwards_per_step):
        self.gamma = gamma
        self.apply_from_step = apply_from_step
        self.apply_until_step = apply_until_step
        self.forwards_per_step = forwards_per_step
        self.grid_sizes = None
        self.forward_idx = 0

    def active(self):
        step = self.forward_idx // self.forwards_per_step
        is_cond = (self.forward_idx % self.forwards_per_step == 0)
        return (
            is_cond and self.gamma != 0
            and step >= self.apply_from_step
            and (self.apply_until_step is None or step < self.apply_until_step)
            and self.grid_sizes is not None
        )


def _grid_pre_hook(state, patch_size):
    p_t, p_h, p_w = patch_size

    def hook(module, args, kwargs):
        x = kwargs.get('hidden_states', args[0] if args else None)
        if x is not None:
            _, _, f, h, w = x.shape
            state.grid_sizes = (f // p_t, h // p_h, w // p_w)
    return hook


def _stepped_forward(state, original):
    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        state.forward_idx += 1
        return out
    return wrapper


try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    _flex_attention = torch.compile(_flex_attention, dynamic=False)
    FLEX_AVAILABLE = True
except Exception:
    _flex_attention = None
    FLEX_AVAILABLE = False
USE_FLEX = FLEX_AVAILABLE


def _rope_apply(x, freqs_cos, freqs_sin):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


class _Processor(WanAttnProcessor):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        state = self.state
        apply_bias = (
            encoder_hidden_states is None and attention_mask is None
            and state.active()
        )
        if apply_bias:
            F_, H_, W_ = state.grid_sizes
            HW = H_ * W_
            apply_bias = F_ > 1 and hidden_states.shape[1] == F_ * HW
        if not apply_bias:
            return super().__call__(attn, hidden_states, encoder_hidden_states,
                                    attention_mask, rotary_emb)

        q, k, v = _get_qkv_projections(attn, hidden_states, None)
        q = attn.norm_q(q).unflatten(2, (attn.heads, -1))
        k = attn.norm_k(k).unflatten(2, (attn.heads, -1))
        v = v.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            q = _rope_apply(q, *rotary_emb)
            k = _rope_apply(k, *rotary_emb)
        gamma_v = float(state.gamma)

        if USE_FLEX:
            def f0_mod(score, b_, h, qi, ki):
                return torch.where((qi >= HW) & (ki < HW), score - gamma_v, score)
            q_t, k_t, v_t = (t.transpose(1, 2) for t in (q, k, v))
            out = _flex_attention(q_t.to(v_t.dtype), k_t.to(v_t.dtype), v_t,
                                  score_mod=f0_mod).transpose(1, 2)
        else:
            s = hidden_states.shape[1]
            mask = torch.zeros(s, device=q.device, dtype=q.dtype)
            mask[:HW] = -gamma_v
            attn_mask = mask.view(1, 1, 1, s)
            kw = dict(dropout_p=0.0, is_causal=False, backend=self._attention_backend)
            out_f0 = dispatch_attention_fn(q[:, :HW], k, v, attn_mask=None, **kw)
            out_rest = dispatch_attention_fn(q[:, HW:], k, v, attn_mask=attn_mask, **kw)
            out = torch.cat([out_f0, out_rest], dim=1)

        out = out.flatten(2, 3).type_as(q)
        out = attn.to_out[0](out)
        return attn.to_out[1](out)


def _apply_patches(pipe, state):
    dit = pipe.transformer
    handle = dit.register_forward_pre_hook(
        _grid_pre_hook(state, dit.config.patch_size), with_kwargs=True)
    orig_procs = []
    for block in dit.blocks:
        orig_procs.append((block.attn1, block.attn1.processor))
        block.attn1.processor = _Processor(state)
    orig_forward = dit.forward
    dit.forward = _stepped_forward(state, dit.forward)

    def restore():
        handle.remove()
        for attn, proc in orig_procs:
            attn.processor = proc
        dit.forward = orig_forward
    return restore


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    p.add_argument("--max_area", type=int, default=480 * 832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_inference_steps", type=int, default=40)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--fps", type=int, default=16)

    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--apply_from_step", type=int, default=0)
    p.add_argument("--apply_until_step", type=int, default=10)

    p.add_argument("--cpu_offload", action="store_true", default=False)
    return p.parse_args()


def _build_jobs(args):
    if args.prompt_file:
        with open(args.prompt_file) as f:
            all_lines = [l.strip() for l in f if l.strip()]
        end = len(all_lines) if args.end < 0 else args.end
        lines = all_lines[args.start:end]
        jobs = []
        for i, line in enumerate(lines):
            gidx = args.start + i
            if "||" in line:
                img, pr = [s.strip() for s in line.split("||", 1)]
            else:
                img, pr = args.image, line
            jobs.append((gidx, img, pr))
        return jobs
    assert args.image and args.prompt, "need --image + --prompt (or --prompt_file)"
    return [(0, args.image, args.prompt)]


def _resolve_hw(img, max_area, mod_value):
    aspect = img.height / img.width
    h = round(np.sqrt(max_area * aspect)) // mod_value * mod_value
    w = round(np.sqrt(max_area / aspect)) // mod_value * mod_value
    return max(int(h), mod_value), max(int(w), mod_value)


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    jobs = _build_jobs(args)
    batch_out_dir = args.out_dir or f"output/dymos_wan21/g{args.gamma:+.1f}_to{args.apply_until_step}"
    os.makedirs(batch_out_dir, exist_ok=True)

    image_encoder = CLIPVisionModel.from_pretrained(
        args.ckpt_dir, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(
        args.ckpt_dir, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.ckpt_dir, vae=vae, image_encoder=image_encoder,
        torch_dtype=torch.bfloat16)
    if args.cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

    state = _State(args.gamma, args.apply_from_step, args.apply_until_step,
                   forwards_per_step=2 if args.guidance_scale > 1 else 1)
    restore = _apply_patches(pipe, state)

    ts = datetime.now().strftime("%m%d%H%M")
    try:
        for j, (gidx, img_path, prompt) in enumerate(jobs):
            if not os.path.exists(img_path):
                logging.warning(f"[skip #{gidx}] no image: {img_path}")
                continue
            stub = "_".join(prompt.split())
            stub = "".join(c for c in stub if c.isalnum() or c == "_")[:120]
            if len(jobs) == 1 and args.save_file:
                save_path = args.save_file
            else:
                save_path = os.path.join(batch_out_dir, f"{ts}_{gidx:03d}_{stub}.mp4")
            if os.path.exists(save_path):
                continue
            logging.info(f"[{j+1}/{len(jobs)}] (#{gidx}) {prompt}")
            img = Image.open(img_path).convert("RGB")
            h, w = _resolve_hw(img, args.max_area, mod_value)
            img = img.resize((w, h))
            state.forward_idx = 0
            state.grid_sizes = None
            video = pipe(
                image=img, prompt=prompt, negative_prompt=args.negative_prompt,
                height=h, width=w,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator("cuda").manual_seed(args.seed + gidx),
            ).frames[0]
            export_to_video(video, save_path, fps=args.fps)
    finally:
        restore()


if __name__ == "__main__":
    main()
