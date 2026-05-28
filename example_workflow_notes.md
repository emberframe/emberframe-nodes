# Example Workflow Notes

## Split-Sigma Sampling

```text
Scheduler
  -> SplitSigmas

SplitSigmas high_sigmas
  -> Sigmas Resample
  -> ZIB EmberFrame SamplerCustomAdvanced Capture sigmas

ZIB output
  -> ZIT EmberFrame SamplerCustomAdvanced Capture latent_image

SplitSigmas low_sigmas
  -> ZIT EmberFrame SamplerCustomAdvanced Capture sigmas
```

## PiD Final-Latent Decode

```text
ZIT output or denoised_output
  -> EmberFrame Normalize ZImage/Flux Latent
  -> PiD Decode latent
```

Use:

```text
PiD Decode backbone = zimage
PiD Decode pid_ckpt_type = 2k or 2kto4k
PiD Decode scale = 4
PiD Decode pid_steps = 4
PiD Decode cfg_scale = 1
PiD Decode sigma = 0
```

## PiD Staged Intermediate Decode

```text
ZIT pid_latent
  -> PiD Prepare latent

ZIT pid_sigma
  -> PiD Prepare sigma

PiD Prepare
  -> PiD Sample
  -> PiD Finalize
```

Start diagnostics with:

```text
capture_mode = from_end
capture_step = 1
capture_tensor_source = x
sigma_source = next
capture_debug = true
```

For Z-Image / Flux PiD, keep PiD scale at `4`.
