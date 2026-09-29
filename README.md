# [NeurIPS 2026] Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models

[Wooseok Jeon](https://vvsjeon.github.io)<sup>1\*</sup>, [Seungho Park](https://sh0xed98b8.github.io)<sup>1\*</sup>, [Seunghyun Shin](https://www.seunghyunshin.com)<sup>2</sup>, [Sangeyl Lee](https://scholar.google.com/citations?user=gA4sNvcAAAAJ)<sup>1</sup>, [Hyeonho Jeong](https://hyeonho99.github.io)<sup>3</sup>, [Hae-Gon Jeon](https://sites.google.com/site/hgjeoncv/home)<sup>1</sup>

<sup>1</sup>Yonsei University &nbsp;&nbsp; <sup>2</sup>GIST &nbsp;&nbsp; <sup>3</sup>Adobe Research

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://sh0xed98b8.github.io/DyMoS/) [![arXiv](https://img.shields.io/badge/arXiv-2605.19398-b31b1b.svg)](https://arxiv.org/abs/2605.19398)

![DyMoS teaser](dymos_teaser_hires.gif)

This repository contains the official PyTorch implementation of
**"Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models."**

DyMoS is a **training-free** inference-time method that restores motion while preserving subject identity and the conditioning image in image-to-video models. It biases the self-attention logits of pretrained image-to-video diffusion transformers to weaken the *reference-frame dominance*: during the first few denoising steps, attention from later frames toward the conditioning (first) frame is reduced by a constant log-bias.

We release our implementation on Wan 2.2
(`dymos_wan22.py`). Implementations for other backbones (Wan 2.1, CogVideoX-5B, HunyuanVideo-1.5) will be released soon.

## Environment

`dymos_wan22.py` wraps the official Wan 2.2 I2V-A14B inference pipeline.
Set up the environment according to the upstream Wan 2.2 repository:

- Upstream: https://github.com/Wan-Video/Wan2.2

Make sure the upstream `wan` package is importable from the directory you
run `dymos_wan22.py` in. Additional requirements:

```
torch >= 2.5      # FlexAttention requires recent torch
Pillow, numpy
```

Download the Wan 2.2 I2V-A14B checkpoint and pass its path via
`--ckpt_dir`.

## Usage

Single-prompt generation:

```bash
python dymos_wan22.py \
    --ckpt_dir <path/to/Wan2.2-I2V-A14B> \
    --size 832*480 \
    --image first_frames/cyclists_burning_man.png \
    --prompt "Cyclists cycling at Burning Man festival" \
    --gamma 0.6 --apply_until_step 8 \
    --offload_model True --convert_model_dtype --t5_cpu \
    --out_dir output/wan22
```

Batch over the bundled `prompts/sample.txt` (one `image_path||prompt`
pair per line):

```bash
python dymos_wan22.py \
    --ckpt_dir <path/to/Wan2.2-I2V-A14B> \
    --size 832*480 \
    --prompt_file prompts/sample.txt \
    --gamma 0.6 --apply_until_step 8 \
    --offload_model True --convert_model_dtype --t5_cpu \
    --out_dir output/wan22
```

## Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--ckpt_dir` | str | required | Path to the Wan 2.2 I2V-A14B model checkpoint directory. |
| `--size` | str | `832*480` | Output video resolution in `W*H` format. |
| `--image` | str | — | Path to the conditioning image (single-prompt mode). |
| `--prompt` | str | — | Text prompt describing the desired video content (single-prompt mode). |
| `--prompt_file` | str | — | Path to a batch prompt file; each line must be `image_path\|\|prompt`. |
| `--gamma` | float | `0.6` | DyMoS bias strength. Higher values reduce reference-frame dominance more aggressively. |
| `--apply_until_step` | int | `8` | Number of early denoising steps (`K`) to apply the bias. |
| `--offload_model` | bool | `False` | Offload model weights to CPU between steps to reduce peak GPU memory. |
| `--convert_model_dtype` | flag | — | Convert model weights to BF16 for lower memory usage. |
| `--t5_cpu` | flag | — | Run the T5 text encoder on CPU to save GPU memory. |
| `--out_dir` | str | `output` | Directory where generated videos are saved. |
