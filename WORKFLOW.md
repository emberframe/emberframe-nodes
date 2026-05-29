# Example Workflow

This repository includes sanitized example workflows:

```text
workflows/z-image_5.4_emberframe.json
workflows/z-image_5.1_emberframe_dual_stage_no_pid.json
```

- `z-image_5.4_emberframe.json` demonstrates a Z-Image Base / Z-Image Turbo split-sigma sampling chain using EmberFrame helper nodes and PiD final-latent decode.
- `z-image_5.1_emberframe_dual_stage_no_pid.json` demonstrates the original non-PiD Z-Image Base / Z-Image Turbo split-sigma workflow using EmberFrame prompt and resolution helper nodes.

## What Was Sanitized

- private/custom LoRA filenames were removed and LoRA slots are disabled
- custom merged checkpoint names were replaced with placeholders
- local image inputs were replaced with `example_input.png`
- temporary preview image metadata was removed
- prompt-helper notes were replaced with public-safe example text
- old private wildcard references were replaced with the bundled example wildcard files
- legacy helper-pack node types were replaced with EmberFrame node types where applicable

## Before Running

After loading the workflow in ComfyUI, set these to your local files:

- Z-Image Base diffusion model
- Z-Image Turbo diffusion model
- VAE
- CLIP/text encoder
- PiD checkpoints
- optional ControlNet, depth, upscaler, and SeedVR2 models if you use those sections

If you use the img2img path, replace the placeholder `example_input.png` with your own image.

## Required External Node Packs

The workflow still needs external packs such as:

- `ComfyUI-PiD`
- `rgthree-comfy`
- `ComfyUI-Image-Saver`
- `ComfyUI-Easy-Use`
- `ComfyUI-KJNodes`
- `RES4LYF`
- ControlNet/depth helper nodes used by the optional sections
- SeedVR2 / Ultimate SD Upscale packs used by the optional upscaling sections

Install `emberframe-nodes` once for the EmberFrame helper nodes used by the main sampling and prompt-helper sections.

## PiD Reminder

For Z-Image / Flux PiD, keep PiD scale at:

```text
scale = 4
```

Both the `2k` and `2kto4k` PiD checkpoints are 4x decoders.
