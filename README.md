# EmberFrame Nodes

`emberframe-nodes` is a growing collection of lightweight custom nodes for Comfy workflows.

Version: `1.0.1`

The first release includes the helper nodes created for the advanced Z-Image Base / Z-Image Turbo PiD workflow, but the package name is intentionally broad so more EmberFrame nodes can be added later.

## Nodes

- `EmberFrame SamplerCustomAdvanced Capture`
- `EmberFrame Normalize ZImage/Flux Latent`
- `EmberFrame Attach Sigma To Latent`
- `EmberFrame Latent Capture Inspector`
- `EmberFrame Sequential Wildcard Prompt`
- `EmberFrame Wildcard Rule Builder`
- `EmberFrame Wildcard Config Combiner`
- `EmberFrame Wildcard Prompt Assembler`
- `EmberFrame MP Aspect Resolution Selector`

The pack also includes three small example wildcard text files for testing the prompt nodes:

- `example_camera_angles.txt`
- `example_lighting_styles.txt`
- `example_scene_moods.txt`

## What This Does Not Include

This pack does not include NVIDIA PiD, PiD checkpoints, Z-Image models, VAE files, upscalers, ControlNet models, or the Merserk `ComfyUI-PiD` wrapper.

The example workflow still needs its larger external node packs, including `ComfyUI-PiD`, `rgthree-comfy`, `ComfyUI-Image-Saver`, `ComfyUI-Easy-Use`, `ComfyUI-KJNodes`, `RES4LYF`, ControlNet/Depth nodes, SeedVR2, Ultimate SD Upscale, and the other packs listed inside the workflow note.

This pack replaces the separate lightweight/helper packs:

- `ComfyUI-PiD-AdvancedSampler`
- `ComfyUI-SequentialWildcardPrompt`
- `ComfyUI-WildcardPromptAssembler`
- `ComfyUI-WildcardPromptToolkit`
- `mp_aspect_res_selector`

## Install

Copy or clone this folder into:

```text
ComfyUI/custom_nodes/emberframe-nodes
```

Git clone install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/emberframe/emberframe-nodes.git
```

Restart Comfy. The new nodes appear under:

```text
EmberFrame/PiD
EmberFrame/Prompt
EmberFrame/Resolution
```

## Example Workflows

Sanitized Z-Image example workflows are included:

```text
workflows/z-image_5.4_emberframe.json
workflows/z-image_5.4_emberframe_PiD_Experimental.json
```

See `WORKFLOW.md` before running them. The workflows use placeholder model names, blank disabled LoRA slots, public-safe example prompts, and the bundled example wildcard files.

## Z-Image / Flux PiD Scale Rule

For Z-Image / Flux PiD, treat PiD as a 4x decoder:

```text
scale = 4
```

Both `2k` and `2kto4k` are sr4x checkpoints. `2kto4k` means a higher-resolution 4x decoder, not a 2x decoder.

Known-good examples:

```text
Want about 1088 x 1920 final:
Generate around 272 x 480, then PiD scale = 4, ckpt = 2k.

Want about 2176 x 3840 final:
Generate around 544 x 960, then PiD scale = 4, ckpt = 2kto4k.
```

Using `scale = 1` or `scale = 2` with Z-Image / Flux PiD can produce warped or distorted images.

## Working Final-Latent PiD Path

This is the simplest path that worked well in testing:

```text
ZiB sampler output
  -> ZiT sampler latent_image

ZiT sampler output or denoised_output
  -> EmberFrame Normalize ZImage/Flux Latent
  -> PiD Decode latent

VAE Decode of ZiT output
  -> optional preview / baseline_image
```

Recommended PiD settings:

```text
backbone = zimage
pid_ckpt_type = 2k or 2kto4k
pid_steps = 4
scale = 4
cfg_scale = 1
sigma = 0
```

For final latent use, keep `EmberFrame Normalize ZImage/Flux Latent` set to:

```text
direction = comfy_to_pid
scale_factor = 0.3611
shift_factor = 0.1159
```

## Intermediate Capture Path

The capture node can also capture intermediate sampler state:

```text
ZiT EmberFrame SamplerCustomAdvanced Capture pid_latent
  -> PiD Prepare latent

ZiT EmberFrame SamplerCustomAdvanced Capture pid_sigma
  -> PiD Prepare sigma

PiD Prepare
  -> PiD Sample
  -> PiD Finalize
```

Useful diagnostic settings:

```text
capture_tensor_source = x
sigma_source = next
capture_debug = true
```

If you feed `output`, `denoised_output`, or `capture_tensor_source = final_latent` into PiD, normalize it first with `EmberFrame Normalize ZImage/Flux Latent`. If you feed callback `x` directly, test before adding normalization, because callback `x` is already in the sampler/model latent space.

## Troubleshooting

If the EmberFrame nodes do not appear, restart Comfy and confirm this folder is inside `ComfyUI/custom_nodes/`.

If wildcard dropdowns are empty, confirm the `wildcards/` folder was copied with the node pack. The included wildcard files are exposed with a stable `emberframe-nodes:` prefix even if the installed folder is renamed.

If PiD output is warped or distorted with Z-Image / Flux, confirm `PiD Decode` uses `scale = 4`. The `2k` and `2kto4k` PiD checkpoints are both 4x decoders.

If `PiD Decode`, `PiD Prepare`, `PiD Sample`, or `PiD Finalize` are missing, install the separate `ComfyUI-PiD` node pack. EmberFrame Nodes only provides helper nodes and does not bundle NVIDIA PiD.

## License Notes

This pack is GPL-3.0 because the sampler-capture node mirrors behavior from Comfy's GPL-3.0 sampler implementation.

The Merserk `ComfyUI-PiD` wrapper is MIT licensed and NVIDIA PiD weights/checkpoints may have separate model-card terms. Do not redistribute model weights in this repository.
