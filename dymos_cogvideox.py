import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils import export_to_video, load_image


MODEL_PATH = "THUDM/CogVideoX-5b-I2V"
HEIGHT, WIDTH, NUM_FRAMES = 480, 720, 49


class _State:
    def __init__(self, gamma, apply_from_step, apply_until_step, do_cfg):
        self.gamma = gamma
        self.apply_from_step = apply_from_step
        self.apply_until_step = apply_until_step
        self.do_cfg = do_cfg
        self.latent_frames = None
        self.forward_idx = 0

    def active(self):
        # CFG is batched ([uncond, cond]), so there is one forward per step.
        step = self.forward_idx
        return (
            self.gamma != 0
            and step >= self.apply_from_step
            and (self.apply_until_step is None or step < self.apply_until_step)
            and self.latent_frames is not None
        )


def _frames_pre_hook(state):
    def hook(module, args, kwargs):
        x = kwargs.get('hidden_states', args[0] if args else None)
        if x is not None:
            state.latent_frames = x.shape[1]
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


class _Processor:
    def __init__(self, state):
        self.state = state

    def __call__(self, attn, hidden_states, encoder_hidden_states,
                 attention_mask=None, image_rotary_emb=None):
        text_len = encoder_hidden_states.size(1)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        b, s, _ = hidden_states.shape

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        head_dim = k.shape[-1] // attn.heads
        q = q.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        if image_rotary_emb is not None:
            q[:, :, text_len:] = apply_rotary_emb(q[:, :, text_len:], image_rotary_emb)
            if not attn.is_cross_attention:
                k[:, :, text_len:] = apply_rotary_emb(k[:, :, text_len:], image_rotary_emb)

        state = self.state
        if attention_mask is None and state.active():
            # Sequence layout: [text | frame 0 | frame 1 | ...]
            HW = (s - text_len) // state.latent_frames
            f0_end = text_len + HW
            c0 = b // 2 if state.do_cfg else 0
            gamma_v = float(state.gamma)

            if USE_FLEX:
                def f0_mod(score, b_, h, qi, ki):
                    hit = (b_ >= c0) & (qi >= f0_end) & (ki >= text_len) & (ki < f0_end)
                    return torch.where(hit, score - gamma_v, score)
                out = _flex_attention(q.to(v.dtype), k.to(v.dtype), v, score_mod=f0_mod)
            else:
                mask = torch.zeros(s, device=q.device, dtype=q.dtype)
                mask[text_len:f0_end] = -gamma_v
                attn_mask = mask.view(1, 1, 1, s)
                qc, kc, vc = q[c0:], k[c0:], v[c0:]
                out = torch.cat([
                    F.scaled_dot_product_attention(qc[:, :, :f0_end], kc, vc),
                    F.scaled_dot_product_attention(qc[:, :, f0_end:], kc, vc, attn_mask=attn_mask),
                ], dim=2)
                if c0 > 0:
                    out_uncond = F.scaled_dot_product_attention(q[:c0], k[:c0], v[:c0])
                    out = torch.cat([out_uncond, out], dim=0)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        out = out.transpose(1, 2).reshape(b, -1, attn.heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        encoder_hidden_states, hidden_states = out.split([text_len, s - text_len], dim=1)
        return hidden_states, encoder_hidden_states


def _apply_patches(pipe, state):
    dit = pipe.transformer
    handle = dit.register_forward_pre_hook(_frames_pre_hook(state), with_kwargs=True)
    orig_procs = []
    for block in dit.transformer_blocks:
        orig_procs.append((block.attn1, block.attn1.processor))
        block.attn1.set_processor(_Processor(state))
    orig_forward = dit.forward
    dit.forward = _stepped_forward(state, dit.forward)

    def restore():
        handle.remove()
        for attn, proc in orig_procs:
            attn.set_processor(proc)
        dit.forward = orig_forward
    return restore


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=MODEL_PATH)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--use_dynamic_cfg", action="store_true", default=False)
    p.add_argument("--fps", type=int, default=8)

    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--apply_from_step", type=int, default=0)
    p.add_argument("--apply_until_step", type=int, default=12)
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


def main():
    args = _parse_args()
    jobs = _build_jobs(args)
    batch_out_dir = args.out_dir or f"output/dymos_cogvideox/g{args.gamma:+.1f}_to{args.apply_until_step}"
    os.makedirs(batch_out_dir, exist_ok=True)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16).to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    state = _State(args.gamma, args.apply_from_step, args.apply_until_step,
                   do_cfg=args.guidance_scale > 1)
    restore = _apply_patches(pipe, state)

    ts = datetime.now().strftime("%m%d%H%M")
    try:
        for j, (gidx, img_path, prompt) in enumerate(jobs):
            if not os.path.exists(img_path):
                print(f"[skip #{gidx}] no image: {img_path}")
                continue
            stub = "_".join(prompt.split())
            stub = "".join(c for c in stub if c.isalnum() or c == "_")[:120]
            if len(jobs) == 1 and args.save_file:
                save_path = args.save_file
            else:
                save_path = os.path.join(batch_out_dir, f"{ts}_{gidx:03d}_{stub}.mp4")
            if os.path.exists(save_path):
                continue
            print(f"[{j+1}/{len(jobs)}] (#{gidx}) {prompt}")
            state.forward_idx = 0
            state.latent_frames = None
            image = load_image(img_path).resize((WIDTH, HEIGHT))
            video = pipe(
                prompt=prompt, image=image,
                height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                use_dynamic_cfg=args.use_dynamic_cfg,
                generator=torch.Generator("cpu").manual_seed(args.seed + gidx),
            ).frames[0]
            export_to_video(video, save_path, fps=args.fps)
    finally:
        restore()


if __name__ == "__main__":
    main()
