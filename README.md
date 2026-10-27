# [NeurIPS 2026] Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models

[Wooseok Jeon](https://vvsjeon.github.io)<sup>1\*</sup>, [Seungho Park](https://shoxedqbbb.github.io/)<sup>1\*</sup>, [Seunghyun Shin](https://www.seunghyunshin.com)<sup>2</sup>, [Sangeyl Lee](https://scholar.google.com/citations?user=gA4sNvcAAAAJ)<sup>1</sup>, [Hyeonho Jeong](https://hyeonho99.github.io)<sup>3</sup>, [Hae-Gon Jeon](https://sites.google.com/site/hgjeoncv/home)<sup>1</sup>

<sup>1</sup>Yonsei University &nbsp;&nbsp; <sup>2</sup>GIST &nbsp;&nbsp; <sup>3</sup>Adobe Research

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://i2v-dymos.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2605.19398-b31b1b.svg)](https://arxiv.org/abs/2605.19398)

![DyMoS teaser](dymos_teaser_hires.gif)

This repository contains the official PyTorch implementation of
**"Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models."**

DyMoS is a **training-free** inference-time method that restores motion while preserving subject identity and the conditioning image in image-to-video models. It biases the self-attention logits of pretrained image-to-video diffusion transformers to weaken the *reference-frame dominance*: during the first few denoising steps, attention from later frames toward the conditioning (first) frame is reduced by a constant log-bias.

We release our implementation on Wan 2.2 (`dymos_wan22.py`), Wan 2.1
(`dymos_wan21.py`), CogVideoX-5B (`dymos_cogvideox.py`), and
HunyuanVideo-1.5 (`dymos_hv15.py`).

## TODO

- [x] Release Wan 2.2 code
- [x] Release CogVideoX-5B, Wan 2.1, and HunyuanVideo-1.5 code

## Default Settings

The bias is applied during the first `K` of the `T` denoising steps, i.e.,
`λ = K / T` in the paper. Each script defaults to the paper setting of its
backbone:

| Backbone | Script | `T` | `--gamma` | `--apply_until_step` (`K`) | `λ = K / T` |
|---|---|---|---|---|---|
| Wan 2.2 I2V-A14B | `dymos_wan22.py` | 40 | 0.6 | 8 | 0.20 |
| Wan 2.1 I2V-14B | `dymos_wan21.py` | 40 | 1.0 | 10 | 0.25 |
| CogVideoX-5B-I2V | `dymos_cogvideox.py` | 50 | 0.8 | 12 | 0.24 |
| HunyuanVideo-1.5 I2V | `dymos_hv15.py` | 50 | 1.0 | 10 | 0.20 |

## Wan 2.2

### Environment

`dymos_wan22.py` wraps the official Wan 2.2 I2V-A14B inference pipeline.
Set up the environment according to the upstream Wan 2.2 repository:

- Upstream: https://github.com/Wan-Video/Wan2.2

Make sure the upstream `wan` package is importable from the directory you
run `dymos_wan22.py` in. Additional requirements:

```
torch >= 2.5      # FlexAttention requires recent torch
Pillow, numpy
```

Download the Wan 2.2 I2V-A14B checkpoint to `./Wan2.2-I2V-A14B` (the default
`--ckpt_dir`).

### Usage

Single-prompt generation:

```bash
python dymos_wan22.py \
    --image first_frames/cyclists_burning_man.png \
    --prompt "Cyclists cycling at Burning Man festival" \
    --gamma 0.6 --apply_until_step 8
```

Batch over the bundled `prompts/sample.txt` (one `image_path||prompt`
pair per line):

```bash
python dymos_wan22.py \
    --prompt_file prompts/sample.txt \
    --gamma 0.6 --apply_until_step 8
```

## Wan 2.1

### Environment

`dymos_wan21.py` wraps the Wan 2.1 I2V-14B pipeline of 🤗 Diffusers
(`WanImageToVideoPipeline`). Requirements:

```
torch >= 2.5      # FlexAttention requires recent torch
diffusers >= 0.35
transformers, accelerate, ftfy
imageio, imageio-ffmpeg, Pillow, numpy
```

The Diffusers-format checkpoint (`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`) is
downloaded from the Hugging Face Hub by default; to use a local copy, pass its
path via `--ckpt_dir`.

### Usage

Single-prompt generation:

```bash
python dymos_wan21.py \
    --image first_frames/cyclists_burning_man.png \
    --prompt "Cyclists cycling at Burning Man festival" \
    --gamma 1.0 --apply_until_step 10
```

Batch over the bundled `prompts/sample.txt` (one `image_path||prompt`
pair per line):

```bash
python dymos_wan21.py \
    --prompt_file prompts/sample.txt \
    --gamma 1.0 --apply_until_step 10
```

## HunyuanVideo-1.5

### Environment

`dymos_hv15.py` wraps the official HunyuanVideo-1.5 I2V inference pipeline.
Set up the environment and download the checkpoints according to the upstream
HunyuanVideo-1.5 repository:

- Upstream: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5

Make sure the upstream `hyvideo` package is importable from the directory you
run `dymos_hv15.py` in. Additional requirements:

```
torch >= 2.5      # FlexAttention requires recent torch
```

Download the checkpoints to `./ckpts` (the default `--model_path`).
`dymos_hv15.py` runs on a single GPU (sequence parallelism is not supported).

### Usage

Single-prompt generation:

```bash
torchrun --nproc_per_node=1 dymos_hv15.py \
    --image first_frames/cyclists_burning_man.png \
    --prompt "Cyclists cycling at Burning Man festival" \
    --gamma 1.0 --apply_until_step 10
```

Batch over the bundled `prompts/sample.txt` (one `image_path||prompt`
pair per line):

```bash
torchrun --nproc_per_node=1 dymos_hv15.py \
    --prompt_file prompts/sample.txt \
    --gamma 1.0 --apply_until_step 10
```

## CogVideoX-5B

### Environment

`dymos_cogvideox.py` wraps the CogVideoX-5B-I2V pipeline of 🤗 Diffusers
(`CogVideoXImageToVideoPipeline`). Requirements:

```
torch >= 2.5      # FlexAttention requires recent torch
diffusers >= 0.35
transformers, accelerate, sentencepiece
imageio, imageio-ffmpeg
```

The checkpoint (`THUDM/CogVideoX-5b-I2V`) is downloaded from the Hugging Face
Hub by default; to use a local copy, pass its path via `--model_path`.
Videos are generated at 720×480 with 49 frames, and the conditioning image is
resized to 720×480.

### Usage

Single-prompt generation:

```bash
python dymos_cogvideox.py \
    --image first_frames/cyclists_burning_man.png \
    --prompt "Cyclists cycling at Burning Man festival" \
    --gamma 0.8 --apply_until_step 12
```

Batch over the bundled `prompts/sample.txt` (one `image_path||prompt`
pair per line):

```bash
python dymos_cogvideox.py \
    --prompt_file prompts/sample.txt \
    --gamma 0.8 --apply_until_step 12
```

## Common Arguments

All scripts share the following arguments. Model-specific options (checkpoint
path, resolution, offloading, etc.) default to the paper settings in each script.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--image` | str | — | Path to the conditioning image (single-prompt mode). |
| `--prompt` | str | — | Text prompt describing the desired video content (single-prompt mode). |
| `--prompt_file` | str | — | Path to a batch prompt file; each line must be `image_path\|\|prompt`. |
| `--gamma` | float | [per backbone](#default-settings) | DyMoS bias strength. Higher values reduce reference-frame dominance more aggressively. |
| `--apply_until_step` | int | [per backbone](#default-settings) | Number of early denoising steps (`K`) to apply the bias. |
| `--num_inference_steps` | int | [per backbone](#default-settings) | Number of denoising steps (`T`). `dymos_wan22.py` uses `--sample_steps` instead. |
| `--out_dir` | str | `output/dymos_<model>/g<γ>_to<K>` | Directory where generated videos are saved. |