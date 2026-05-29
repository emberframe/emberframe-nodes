# Example Workflow

This repository includes sanitized example workflows:

```text
workflows/z-image_5.0_emberframe.json
workflows/z-image_5.0_emberframe_pid.json
workflows/z-image_5.1_emberframe_pid.json
workflows/z-image_5.2_emberframe_pid.json
```

- `z-image_5.0_emberframe.json` demonstrates the base non-PiD Z-Image Base / Z-Image Turbo split-sigma workflow using EmberFrame prompt and resolution helper nodes.
- `z-image_5.0_emberframe_pid.json` maps to the previous local PiD `v2.0` workflow.
- `z-image_5.1_emberframe_pid.json` maps to the previous local PiD `v2.1` workflow and includes the Impact detailer pass.
- `z-image_5.2_emberframe_pid.json` maps to the previous local PiD `v2.2` workflow and includes the SAM3 detailer pass.

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
- optional Impact detailer detector files for `z-image_5.1_emberframe_pid.json`
- optional SAM3 model access for `z-image_5.2_emberframe_pid.json`
- optional ControlNet, depth, upscaler, and SeedVR2 models if you use those sections

If you use the img2img path, replace the placeholder `example_input.png` with your own image.

## Required External Node Packs

The workflow still needs external packs such as:

- `ComfyUI-PiD`
- `rgthree-comfy`
- `ComfyUI-Image-Saver`
- `ComfyUI-Impact-Pack`
- `ComfyUI-LG_SamplingUtils`
- `ComfyUI_essentials`
- `ComfyUI-Easy-Use`
- `ComfyUI-Impact-Subpack` for the 5.1 Impact detailer pass
- `ComfyUI-TBG-SAM3` for the 5.2 SAM3 detailer pass
- `ComfyUI-KJNodes`
- `RES4LYF`
- ControlNet/depth helper nodes used by the optional sections
- `WAS Node Suite - Revised`
- `comfyui-vrgamedevgirl` for the Film Grain and Sharpen nodes
- SeedVR2 / Ultimate SD Upscale packs used by the optional upscaling sections

Install `emberframe-nodes` once for the EmberFrame helper nodes used by the main sampling and prompt-helper sections.

## PiD Reminder

For Z-Image / Flux PiD, keep PiD scale at:

```text
scale = 4
```

Both the `2k` and `2kto4k` PiD checkpoints are 4x decoders.
