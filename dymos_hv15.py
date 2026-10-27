import os
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import copy
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import einops
import imageio
import loguru
import torch
import torch.nn.functional as F

from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.models.transformers import hunyuanvideo_1_5_transformer as hv_transformer
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline

parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


class _State:
    def __init__(self, gamma, apply_from_step, apply_until_step):
        self.gamma = gamma
        self.apply_from_step = apply_from_step
        self.apply_until_step = apply_until_step
        self.do_cfg = True
        self.in_dit = False
        self.forward_idx = 0

    def active(self):
        # CFG is batched ([uncond, cond]), so there is one forward per step.
        step = self.forward_idx
        return (
            self.in_dit and self.gamma != 0
            and step >= self.apply_from_step
            and (self.apply_until_step is None or step < self.apply_until_step)
        )


try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    _flex_attention = torch.compile(_flex_attention, dynamic=False)
    FLEX_AVAILABLE = True
except Exception:
    _flex_attention = None
    FLEX_AVAILABLE = False
USE_FLEX = FLEX_AVAILABLE


def _patched_parallel_attention(state, original):
    @torch.compiler.disable
    def patched(q, k, v, img_q_len, img_kv_len, attn_mode=None, text_mask=None,
                attn_param=None, block_idx=None):
        thw = attn_param.get('thw') if isinstance(attn_param, dict) else None
        apply_bias = state.active() and thw is not None and thw[0] > 1
        if apply_bias:
            HW = thw[1] * thw[2]
            L_img = q[0].shape[1]
            apply_bias = L_img == thw[0] * HW
        if not apply_bias:
            return original(q, k, v, img_q_len, img_kv_len, attn_mode=attn_mode,
                            text_mask=text_mask, attn_param=attn_param, block_idx=block_idx)

        # Sequence layout: [frame 0 | frame 1 | ... | text]
        query = torch.cat(q, dim=1).transpose(1, 2)
        key = torch.cat(k, dim=1).transpose(1, 2)
        value = torch.cat(v, dim=1).transpose(1, 2)
        b, _, s, _ = query.shape
        c0 = b // 2 if state.do_cfg else 0
        gamma_v = float(state.gamma)

        # Every query outside frame 0 (video and text) is biased on frame-0 keys.
        if USE_FLEX:
            def f0_mod(score, b_, h, qi, ki):
                hit = (b_ >= c0) & (qi >= HW) & (ki < HW)
                return torch.where(hit, score - gamma_v, score)
            td = value.dtype
            out = _flex_attention(query.to(td), key.to(td), value, score_mod=f0_mod)
        else:
            f0_bias = torch.zeros(b, 1, 1, s, device=query.device, dtype=query.dtype)
            f0_bias[c0:, :, :, :HW] = -gamma_v
            out = torch.cat([
                F.scaled_dot_product_attention(query[:, :, :HW], key, value),
                F.scaled_dot_product_attention(query[:, :, HW:], key, value, attn_mask=f0_bias),
            ], dim=2)

        out = out.transpose(1, 2)
        return out.reshape(b, s, -1)
    return patched


def _apply_patches(pipe, state):
    orig_attn = hv_transformer.parallel_attention
    hv_transformer.parallel_attention = _patched_parallel_attention(state, orig_attn)
    dit = pipe.transformer

    # nn hooks survive diffusers group offloading, which resets module.forward.
    def pre(module, args):
        state.do_cfg = pipe.do_classifier_free_guidance
        state.in_dit = True

    def post(module, args, output):
        state.in_dit = False
        state.forward_idx += 1

    hooks = [dit.register_forward_pre_hook(pre), dit.register_forward_hook(post)]

    def restore():
        hv_transformer.parallel_attention = orig_attn
        for h in hooks:
            h.remove()
    return restore


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected')


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="./ckpts")
    p.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    p.add_argument("--aspect_ratio", type=str, default="16:9")
    p.add_argument("--video_length", type=int, default=81)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--fps", type=int, default=16)

    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--apply_from_step", type=int, default=0)
    p.add_argument("--apply_until_step", type=int, default=10)

    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--cfg_distilled", type=_str2bool, nargs='?', const=True, default=False)
    p.add_argument("--enable_step_distill", type=_str2bool, nargs='?', const=True, default=False)
    p.add_argument("--sparse_attn", type=_str2bool, nargs='?', const=True, default=False)
    p.add_argument("--use_sageattn", type=_str2bool, nargs='?', const=True, default=True)
    p.add_argument("--sage_blocks_range", type=str, default="0-53")
    p.add_argument("--rewrite", type=_str2bool, nargs='?', const=True, default=False)
    p.add_argument("--sr", type=_str2bool, nargs='?', const=True, default=False)
    p.add_argument("--offloading", type=_str2bool, nargs='?', const=True, default=True)
    p.add_argument("--group_offloading", type=_str2bool, nargs='?', const=True, default=None)
    p.add_argument("--overlap_group_offloading", type=_str2bool, nargs='?', const=True, default=True)

    args = p.parse_args()
    for k, v in dict(
        enable_torch_compile=False, enable_cache=False, cache_type="deepcache",
        no_cache_block_id="53", cache_start_step=11, cache_end_step=45,
        total_steps=args.num_inference_steps, cache_step_interval=4,
        use_fp8_gemm=False, quant_type="fp8-per-token-sgl",
        include_patterns="double_blocks",
    ).items():
        setattr(args, k, v)
    return args


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


def _save_video(video, path, fps):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=fps)


def main():
    args = _parse_args()
    assert not parallel_dims.sp_enabled, "sequence parallel is not supported; run on a single GPU"
    if args.sparse_attn and args.use_sageattn:
        raise ValueError("sparse_attn and use_sageattn cannot be enabled simultaneously")

    jobs = _build_jobs(args)
    batch_out_dir = args.out_dir or f"output/dymos_hv15/g{args.gamma:+.1f}_to{args.apply_until_step}"
    os.makedirs(batch_out_dir, exist_ok=True)

    infer_state = initialize_infer_state(args)
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution, "i2v", args.cfg_distilled, args.enable_step_distill, args.sparse_attn)
    transformer_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    enable_offloading = args.offloading
    if args.group_offloading is None:
        enable_group_offloading = HunyuanVideo_1_5_Pipeline.get_offloading_config()['enable_group_offloading']
    else:
        enable_group_offloading = args.group_offloading
    device = torch.device("cpu") if enable_offloading else torch.device("cuda")
    transformer_init_device = torch.device("cpu") if enable_group_offloading else device

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        create_sr_pipeline=args.sr,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
    )
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=args.overlap_group_offloading,
    )
    if args.sr and hasattr(pipe, "sr_pipeline"):
        sr_infer_state = copy.deepcopy(infer_state)
        sr_infer_state.enable_cache = False
        pipe.sr_pipeline.apply_infer_optimization(
            infer_state=sr_infer_state,
            enable_offloading=enable_offloading,
            enable_group_offloading=enable_group_offloading,
            overlap_group_offloading=args.overlap_group_offloading,
        )

    state = _State(args.gamma, args.apply_from_step, args.apply_until_step)
    restore = _apply_patches(pipe, state)

    ts = datetime.now().strftime("%m%d%H%M")
    try:
        for j, (gidx, img_path, prompt) in enumerate(jobs):
            if not (img_path and os.path.exists(img_path)):
                loguru.logger.warning(f"[skip #{gidx}] no image: {img_path}")
                continue
            stub = "_".join(prompt.split())
            stub = "".join(c for c in stub if c.isalnum() or c == "_")[:120]
            if len(jobs) == 1 and args.save_file:
                save_path = args.save_file
            else:
                save_path = os.path.join(batch_out_dir, f"{ts}_{gidx:03d}_{stub}.mp4")
            if os.path.exists(save_path) or glob.glob(
                    os.path.join(batch_out_dir, f"*_{gidx:03d}_{stub}.mp4")):
                continue
            loguru.logger.info(f"[{j+1}/{len(jobs)}] (#{gidx}) {prompt}")
            state.forward_idx = 0
            out = pipe(
                enable_sr=args.sr,
                prompt=prompt,
                aspect_ratio=args.aspect_ratio,
                num_inference_steps=args.num_inference_steps,
                sr_num_inference_steps=None,
                video_length=args.video_length,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                output_type="pt",
                prompt_rewrite=args.rewrite,
                return_pre_sr_video=False,
                reference_image=img_path,
            )
            video = out.sr_videos if (args.sr and hasattr(out, "sr_videos")) else out.videos
            _save_video(video, save_path, fps=args.fps)
    finally:
        restore()


if __name__ == "__main__":
    main()
