import argparse
import logging
import os
import sys
import random
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool


TASK = "i2v-A14B"


class _State:
    def __init__(self, gamma, apply_from_step, apply_until_step):
        self.gamma = gamma
        self.apply_from_step = apply_from_step
        self.apply_until_step = apply_until_step
        self.grid_sizes = None
        self.forward_idx = 0


def _grid_pre_hook(state):
    def hook(module, args, kwargs):
        gs = kwargs.get('grid_sizes', None)
        if gs is None and len(args) >= 4:
            gs = args[3]
        if gs is not None:
            state.grid_sizes = gs.detach().cpu().tolist()
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


def _patched_self_attn(state, attn_module):
    original = attn_module.forward
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim

    def patched(x, seq_lens, grid_sizes, freqs):
        step = state.forward_idx // 2
        is_cond = (state.forward_idx % 2 == 0)

        apply_bias = (
            is_cond and state.gamma != 0
            and step >= state.apply_from_step
            and (state.apply_until_step is None or step < state.apply_until_step)
            and state.grid_sizes is not None
        )
        if not apply_bias:
            return original(x, seq_lens, grid_sizes, freqs)

        b, s, n, d = *x.shape[:2], num_heads, head_dim
        q = attn_module.norm_q(attn_module.q(x)).view(b, s, n, d)
        k = attn_module.norm_k(attn_module.k(x)).view(b, s, n, d)
        v = attn_module.v(x).view(b, s, n, d)
        from wan.modules.model import rope_apply
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        F_, H_, W_ = state.grid_sizes[0]
        HW = H_ * W_
        gamma_v = float(state.gamma)

        if USE_FLEX:
            def f0_mod(score, b_, h, qi, ki):
                return torch.where((qi >= HW) & (ki < HW), score - gamma_v, score)
            target_dtype = v_t.dtype
            q_t_ = q_t.to(target_dtype) if q_t.dtype != target_dtype else q_t
            k_t_ = k_t.to(target_dtype) if k_t.dtype != target_dtype else k_t
            out = _flex_attention(q_t_, k_t_, v_t, score_mod=f0_mod)
        else:
            mask = torch.zeros(s, device=x.device, dtype=q_t.dtype)
            mask[:HW] = -gamma_v
            attn_mask = mask.view(1, 1, 1, s)
            out_f0   = F.scaled_dot_product_attention(q_t[:, :, :HW], k_t, v_t)
            out_rest = F.scaled_dot_product_attention(q_t[:, :, HW:], k_t, v_t, attn_mask=attn_mask)
            out = torch.cat([out_f0, out_rest], dim=2)
            
        out = out.transpose(1, 2).contiguous().flatten(2)
        return attn_module.o(out)

    attn_module.forward = patched
    return original


def _apply_patches(pipeline, state):
    handles, orig_sa, model_origs = [], {}, {}
    for dit in [pipeline.high_noise_model, pipeline.low_noise_model]:
        mid = id(dit)
        handles.append(dit.blocks[0].register_forward_pre_hook(
            _grid_pre_hook(state), with_kwargs=True))
        for i, block in enumerate(dit.blocks):
            orig_sa[(mid, i)] = _patched_self_attn(state, block.self_attn)
        model_origs[mid] = dit.forward
        dit.forward = _stepped_forward(state, dit.forward)

    def restore():
        for h in handles:
            h.remove()
        for dit in [pipeline.high_noise_model, pipeline.low_noise_model]:
            mid = id(dit)
            for i, block in enumerate(dit.blocks):
                if (mid, i) in orig_sa:
                    block.self_attn.forward = orig_sa[(mid, i)]
            if mid in model_origs:
                dit.forward = model_origs[mid]
    return restore


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--size", type=str, default="1280*720",
                   choices=list(SUPPORTED_SIZES[TASK]))
    p.add_argument("--frame_num", type=int, default=None)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--base_seed", type=int, default=-1)

    p.add_argument("--sample_solver", type=str, default="unipc",
                   choices=["unipc", "dpm++"])
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--sample_shift", type=float, default=None)
    p.add_argument("--sample_guide_scale", type=float, default=None)

    p.add_argument("--gamma", type=float, default=0.6)
    p.add_argument("--apply_from_step", type=int, default=0)
    p.add_argument("--apply_until_step", type=int, default=8)

    p.add_argument("--offload_model", type=str2bool, default=None)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=False)
    p.add_argument("--ulysses_size", type=int, default=1)
    p.add_argument("--t5_fsdp", action="store_true", default=False)
    p.add_argument("--dit_fsdp", action="store_true", default=False)

    args = p.parse_args()
    cfg = WAN_CONFIGS[TASK]
    if args.sample_steps is None:      args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:      args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None: args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:         args.frame_num = cfg.frame_num
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


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
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = (world_size == 1)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp)
        assert args.ulysses_size == 1

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size
        init_distributed_group()

    cfg = WAN_CONFIGS[TASK]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0

    jobs = _build_jobs(args)
    batch_out_dir = args.out_dir or f"output/dymos_wan22/g{args.gamma:+.1f}_to{args.apply_until_step}"
    if rank == 0:
        os.makedirs(batch_out_dir, exist_ok=True)

    pipeline = wan.WanI2V(
        config=cfg, checkpoint_dir=args.ckpt_dir,
        device_id=device, rank=rank,
        t5_fsdp=args.t5_fsdp, dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu, convert_model_dtype=args.convert_model_dtype,
    )

    state = _State(args.gamma, args.apply_from_step, args.apply_until_step)
    restore = _apply_patches(pipeline, state)

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
            state.forward_idx = 0
            img = Image.open(img_path).convert("RGB")
            video = pipeline.generate(
                prompt, img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed + gidx,
                offload_model=args.offload_model,
            )
            if rank == 0:
                save_video(tensor=video[None], save_file=save_path,
                           fps=cfg.sample_fps, nrow=1, normalize=True,
                           value_range=(-1, 1))
            del video
            torch.cuda.synchronize()
    finally:
        restore()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
