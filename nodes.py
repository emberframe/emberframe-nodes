from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import comfy.model_management
    import comfy.nested_tensor
    import latent_preview
except Exception:  # pragma: no cover - only available inside ComfyUI
    comfy = None
    latent_preview = None


CAPTURE_MODES = ["from_end", "one_based_step", "sigma_threshold"]
TENSOR_SOURCES = ["x", "x0", "denoised", "final_latent"]
SIGMA_SOURCES = ["current", "next", "previous", "zero"]


def _sigmas_len(sigmas: torch.Tensor) -> int:
    try:
        return int(sigmas.shape[-1])
    except Exception:
        try:
            return len(sigmas)
        except Exception:
            return 0


def _effective_steps(sigmas: torch.Tensor) -> int:
    return max(0, _sigmas_len(sigmas) - 1)


def _sigma_at(sigmas: torch.Tensor, step: int, default: float = 0.0) -> float:
    try:
        return float(sigmas[int(step)].detach().float().cpu().item())
    except Exception:
        return float(default)


def _sigma_at_optional(sigmas: torch.Tensor, step: int):
    try:
        if int(step) < 0 or int(step) >= _sigmas_len(sigmas):
            return None
        return float(sigmas[int(step)].detach().float().cpu().item())
    except Exception:
        return None


def _format_sigma_value(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.9g}"
    except Exception:
        return str(value)


def _select_sigma(sigmas: torch.Tensor, step_index: int, sigma_source: str) -> dict:
    source = str(sigma_source or "current").strip()
    if source not in SIGMA_SOURCES:
        source = "current"

    info = {
        "source": source,
        "previous": _sigma_at_optional(sigmas, int(step_index) - 1),
        "current": _sigma_at_optional(sigmas, int(step_index)),
        "next": _sigma_at_optional(sigmas, int(step_index) + 1),
        "requested_index": None,
        "used_index": None,
        "clamped": False,
        "value": 0.0,
        "note": "",
    }

    if source == "zero":
        info["note"] = "forced zero"
        return info

    total_sigmas = _sigmas_len(sigmas)
    requested = int(step_index) + {"previous": -1, "current": 0, "next": 1}[source]
    info["requested_index"] = requested
    if total_sigmas <= 0:
        info["note"] = "no sigmas available; using 0.0"
        return info

    used = min(max(requested, 0), total_sigmas - 1)
    info["used_index"] = used
    info["clamped"] = used != requested
    info["value"] = _sigma_at(sigmas, used, 0.0)
    if info["clamped"]:
        info["note"] = f"clamped sigma index {requested} to {used}"
    return info


def _clamp_step(step: int, effective_steps: int) -> int:
    if effective_steps <= 0:
        return 0
    return min(max(1, int(step)), int(effective_steps))


def _capture_target(capture_mode: str, capture_step: int, capture_from_end: bool, effective_steps: int):
    mode = str(capture_mode or "from_end").strip()
    if mode not in CAPTURE_MODES:
        mode = "from_end"
    if capture_from_end:
        mode = "from_end"

    if effective_steps <= 0 or mode == "sigma_threshold":
        return mode, None

    requested_step = max(1, int(capture_step))
    if mode == "from_end":
        target = int(effective_steps) - requested_step
    else:
        target = requested_step
    return mode, _clamp_step(target, effective_steps)


def _capture_tensor(x: torch.Tensor, store_latent_on_cpu: bool) -> torch.Tensor:
    out = x.detach()
    if store_latent_on_cpu:
        out = out.to("cpu")
    return out.contiguous()


def _copy_pid_latent(latent: dict, samples: torch.Tensor, sigma: Optional[float] = None) -> dict:
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    if sigma is not None:
        out["pid_sigma"] = float(sigma)
    return out


def _shape_text(samples: torch.Tensor) -> str:
    try:
        return str(list(samples.shape))
    except Exception:
        return str(getattr(samples, "shape", "unknown"))


def _tensor_dtype_text(samples) -> str:
    return str(getattr(samples, "dtype", "unknown"))


def _tensor_device_text(samples) -> str:
    return str(getattr(samples, "device", "unknown"))


def _tensor_for_stats(samples):
    if getattr(samples, "is_nested", False):
        try:
            samples = samples.unbind()[0]
        except Exception:
            pass
    return samples


def _tensor_stats_text(samples) -> str:
    try:
        t = _tensor_for_stats(samples).detach().float()
        if t.numel() == 0:
            return "min=n/a; max=n/a; mean=n/a; std=n/a"
        return (
            f"min={float(t.min().cpu().item()):.9g}; "
            f"max={float(t.max().cpu().item()):.9g}; "
            f"mean={float(t.mean().cpu().item()):.9g}; "
            f"std={float(t.std(unbiased=False).cpu().item()):.9g}"
        )
    except Exception as exc:
        return f"stats_error={exc}"


def _select_callback_tensor(source: str, x0, x):
    source = str(source or "x").strip()
    if source not in TENSOR_SOURCES:
        source = "x"
    if source == "x":
        return source, x
    if source in ("x0", "denoised"):
        return source, x0
    return source, None


def _capture_info_text(captured: dict, capture_debug: bool = False) -> str:
    parts = [
        f"mode={captured.get('mode')}",
        f"requested_step={captured.get('requested_step')}",
        f"target_step={captured.get('target_step')}",
        f"captured_step={captured.get('step')}",
        f"from_end={captured.get('from_end')}",
        f"total_steps={captured.get('total_steps')}",
        f"total_sigmas={captured.get('total_sigmas')}",
        f"callback_index={captured.get('callback_index')}",
        f"tensor_source={captured.get('tensor_source')}",
        f"available_callback_keys={captured.get('available_callback_keys')}",
        f"callback_sigma_available={captured.get('callback_sigma_available')}",
        f"shape={captured.get('shape')}",
        f"dtype={captured.get('dtype')}",
        f"device_before={captured.get('device_before')}",
        f"device_after={captured.get('device_after')}",
        f"sigma_source={captured.get('sigma_source')}",
        f"sigma_previous={_format_sigma_value(captured.get('sigma_previous'))}",
        f"sigma_current={_format_sigma_value(captured.get('sigma_current'))}",
        f"sigma_next={_format_sigma_value(captured.get('sigma_next'))}",
        f"sigma_requested_index={captured.get('sigma_requested_index')}",
        f"sigma_used_index={captured.get('sigma_used_index')}",
        f"sigma_index_clamped={captured.get('sigma_index_clamped')}",
        f"pid_sigma={_format_sigma_value(captured.get('sigma'))}",
        f"fallback_final_latent={captured.get('fallback')}",
        f"moved_to_cpu={captured.get('moved_to_cpu')}",
        f"note={captured.get('note')}",
    ]
    if capture_debug:
        parts.append(_tensor_stats_text(captured.get("samples")))
    return "; ".join(parts)


class PiDAdvancedSamplerCustomAdvancedCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "capture_mode": (CAPTURE_MODES, {"default": "from_end"}),
                "capture_step": ("INT", {"default": 4, "min": 1, "max": 10000, "step": 1}),
                "capture_from_end": ("BOOLEAN", {"default": False}),
                "sigma_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "fail_if_not_captured": ("BOOLEAN", {"default": False}),
                "store_latent_on_cpu": ("BOOLEAN", {"default": True}),
                "print_debug": ("BOOLEAN", {"default": False}),
                "capture_tensor_source": (TENSOR_SOURCES, {"default": "x"}),
                "sigma_source": (SIGMA_SOURCES, {"default": "current"}),
                "capture_debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "FLOAT", "STRING")
    RETURN_NAMES = ("output", "denoised_output", "pid_latent", "pid_sigma", "capture_info")
    FUNCTION = "sample"
    CATEGORY = "EmberFrame/PiD"

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        capture_mode: str = "from_end",
        capture_step: int = 4,
        capture_from_end: bool = False,
        sigma_threshold: float = 0.0,
        fail_if_not_captured: bool = False,
        store_latent_on_cpu: bool = True,
        print_debug: bool = False,
        capture_tensor_source: str = "x",
        sigma_source: str = "current",
        capture_debug: bool = False,
    ):
        if comfy is None:
            raise RuntimeError("PiD Advanced SamplerCustomAdvanced Capture must run inside ComfyUI.")

        # Mirrors ComfyUI's SamplerCustomAdvanced, with only the callback wrapped.
        latent = latent_image
        latent_samples = latent["samples"]
        latent = latent.copy()
        latent_samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent_samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = latent_samples

        noise_mask = latent.get("noise_mask", None)

        effective_steps = _effective_steps(sigmas)
        total_sigmas = _sigmas_len(sigmas)
        resolved_mode, target_step = _capture_target(
            capture_mode,
            capture_step,
            bool(capture_from_end),
            effective_steps,
        )
        threshold = float(sigma_threshold)
        captured = {
            "samples": None,
            "sigma": None,
            "step": None,
            "total_steps": effective_steps,
            "total_sigmas": total_sigmas,
            "callback_index": None,
            "tensor_source": str(capture_tensor_source or "x"),
            "available_callback_keys": "step,x0/denoised,x,total_steps",
            "callback_sigma_available": False,
            "sigma_source": str(sigma_source or "current"),
            "sigma_previous": None,
            "sigma_current": None,
            "sigma_next": None,
            "sigma_requested_index": None,
            "sigma_used_index": None,
            "sigma_index_clamped": False,
            "shape": "unknown",
            "dtype": "unknown",
            "device_before": "unknown",
            "device_after": "unknown",
            "fallback": False,
            "matched": False,
            "deferred_final_latent": False,
            "mode": resolved_mode,
            "requested_step": int(capture_step),
            "target_step": target_step if target_step is not None else "sigma_threshold",
            "from_end": bool(capture_from_end) or resolved_mode == "from_end",
            "moved_to_cpu": bool(store_latent_on_cpu),
            "note": (
                "ComfyUI KSAMPLER forwards callback keys i, denoised, x; "
                "sigma/sigma_hat are not forwarded, so sigma is selected from the external sigmas input."
            ),
        }

        x0_output = {}
        preview_callback = None
        if latent_preview is not None:
            try:
                preview_callback = latent_preview.prepare_callback(
                    guider.model_patcher,
                    effective_steps,
                    x0_output,
                )
            except TypeError:
                preview_callback = latent_preview.prepare_callback(guider.model_patcher, effective_steps)

        def callback(step, x0, x, total_steps):
            if preview_callback is not None:
                preview_callback(step, x0, x, total_steps)

            if captured["matched"]:
                return

            step_index = int(step)
            one_based_step = step_index + 1
            sigma = _sigma_at(sigmas, step_index)
            if resolved_mode == "sigma_threshold":
                should_capture = sigma <= threshold
            else:
                should_capture = target_step is not None and one_based_step == int(target_step)

            if not should_capture:
                return

            selected_source, selected_tensor = _select_callback_tensor(capture_tensor_source, x0, x)
            sigma_info = _select_sigma(sigmas, step_index, sigma_source)
            captured["matched"] = True
            captured["tensor_source"] = selected_source
            captured["sigma"] = float(sigma_info["value"])
            captured["step"] = one_based_step
            captured["total_steps"] = int(total_steps)
            captured["callback_index"] = step_index
            captured["sigma_source"] = sigma_info["source"]
            captured["sigma_previous"] = sigma_info["previous"]
            captured["sigma_current"] = sigma_info["current"]
            captured["sigma_next"] = sigma_info["next"]
            captured["sigma_requested_index"] = sigma_info["requested_index"]
            captured["sigma_used_index"] = sigma_info["used_index"]
            captured["sigma_index_clamped"] = bool(sigma_info["clamped"])
            if sigma_info["note"]:
                captured["note"] = f"{captured['note']} {sigma_info['note']}"

            if selected_source == "final_latent":
                captured["deferred_final_latent"] = True
                captured["shape"] = "deferred until final latent"
                captured["dtype"] = "deferred until final latent"
                captured["device_before"] = "deferred until final latent"
                captured["device_after"] = "deferred until final latent"
                return

            captured["device_before"] = _tensor_device_text(selected_tensor)
            captured["samples"] = _capture_tensor(selected_tensor, bool(store_latent_on_cpu))
            captured["shape"] = _shape_text(captured["samples"])
            captured["dtype"] = _tensor_dtype_text(captured["samples"])
            captured["device_after"] = _tensor_device_text(captured["samples"])

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(
            noise.generate_noise(latent),
            latent_samples,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        if "x0" in x0_output:
            x0_out = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            if samples.is_nested:
                latent_shapes = [x.shape for x in samples.unbind()]
                x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out

        if captured["deferred_final_latent"]:
            captured["device_before"] = _tensor_device_text(samples)
            captured["samples"] = _capture_tensor(samples, bool(store_latent_on_cpu))
            captured["shape"] = _shape_text(captured["samples"])
            captured["dtype"] = _tensor_dtype_text(captured["samples"])
            captured["device_after"] = _tensor_device_text(captured["samples"])

        if captured["samples"] is None:
            if fail_if_not_captured:
                raise RuntimeError(
                    "PiD Advanced SamplerCustomAdvanced Capture did not capture an intermediate latent. "
                    f"mode={resolved_mode}, capture_step={int(capture_step)}, "
                    f"effective_steps={effective_steps}, sigma_threshold={threshold}"
                )
            captured["device_before"] = _tensor_device_text(samples)
            captured["samples"] = _capture_tensor(samples, bool(store_latent_on_cpu))
            captured["sigma"] = 0.0
            captured["step"] = effective_steps
            captured["fallback"] = True
            captured["matched"] = False
            captured["tensor_source"] = "final_latent"
            captured["sigma_source"] = "zero"
            captured["shape"] = _shape_text(captured["samples"])
            captured["dtype"] = _tensor_dtype_text(captured["samples"])
            captured["device_after"] = _tensor_device_text(captured["samples"])
            captured["note"] = f"{captured['note']} fallback used final latent with sigma 0.0"

        pid_sigma = float(captured["sigma"])
        pid_latent = _copy_pid_latent(latent, captured["samples"], pid_sigma)
        capture_info = _capture_info_text(captured, bool(capture_debug))
        if resolved_mode == "sigma_threshold":
            capture_info += f"; sigma_threshold={threshold:.9g}"
        if print_debug or capture_debug:
            print(f"[emberframe-nodes] {capture_info}")

        return (out, out_denoised, pid_latent, pid_sigma, capture_info)

    execute = sample


class PiDAdvancedAttachSigmaToLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "attach"
    CATEGORY = "EmberFrame/PiD"

    def attach(self, latent, sigma: float = 0.0):
        out = latent.copy()
        out["pid_sigma"] = float(sigma)
        return (out,)


class PiDAdvancedNormalizeZImageFluxLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "direction": (["comfy_to_pid", "pid_to_comfy"], {"default": "comfy_to_pid"}),
                "scale_factor": ("FLOAT", {"default": 0.3611, "min": -100.0, "max": 100.0, "step": 0.0001}),
                "shift_factor": ("FLOAT", {"default": 0.1159, "min": -100.0, "max": 100.0, "step": 0.0001}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "normalize"
    CATEGORY = "EmberFrame/PiD"

    def normalize(
        self,
        latent,
        direction: str = "comfy_to_pid",
        scale_factor: float = 0.3611,
        shift_factor: float = 0.1159,
    ):
        if not isinstance(latent, dict):
            raise TypeError(f"Expected LATENT dict, got {type(latent)}")
        samples = latent.get("samples", None)
        if samples is None:
            raise ValueError("LATENT has no samples key.")

        scale = float(scale_factor)
        shift = float(shift_factor)
        if direction == "comfy_to_pid":
            transformed = (samples - shift) * scale
            formula = f"(samples - {shift:.6g}) * {scale:.6g}"
        elif direction == "pid_to_comfy":
            if abs(scale) < 1e-12:
                raise ValueError("scale_factor must be non-zero for pid_to_comfy.")
            transformed = (samples / scale) + shift
            formula = f"(samples / {scale:.6g}) + {shift:.6g}"
        else:
            raise ValueError(f"Unknown direction={direction!r}")

        out = latent.copy()
        out["samples"] = transformed.contiguous()
        info = (
            f"direction={direction}; formula={formula}; "
            f"before_shape={_shape_text(samples)}; before_stats={_tensor_stats_text(samples)}; "
            f"after_shape={_shape_text(out['samples'])}; after_stats={_tensor_stats_text(out['samples'])}; "
            f"embedded_pid_sigma={_format_sigma_value(out.get('pid_sigma', None))}"
        )
        return (out, info)


class PiDAdvancedLatentCaptureInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("capture_info",)
    FUNCTION = "inspect"
    CATEGORY = "EmberFrame/PiD"

    def inspect(self, latent, sigma=None):
        if not isinstance(latent, dict):
            return (f"Expected LATENT dict, got {type(latent)}",)
        samples = latent.get("samples", None)
        if samples is None:
            return ("LATENT has no samples key.",)

        embedded_sigma = latent.get("pid_sigma", None)
        external_sigma = "not connected" if sigma is None else _format_sigma_value(sigma)
        info = (
            f"shape={_shape_text(samples)}; "
            f"dtype={_tensor_dtype_text(samples)}; "
            f"device={_tensor_device_text(samples)}; "
            f"is_nested={bool(getattr(samples, 'is_nested', False))}; "
            f"{_tensor_stats_text(samples)}; "
            f"embedded_pid_sigma={_format_sigma_value(embedded_sigma)}; "
            f"external_sigma={external_sigma}; "
            f"latent_keys={sorted(str(k) for k in latent.keys())}"
        )
        return (info,)


WILDCARD_PLACEHOLDERS = ("{wildcard}", "{{wildcard}}", "__WILDCARD__")
WILDCARD_VALID_MODES = {"fixed", "next", "previous", "randomize", "increment", "decrement"}
LOCAL_WILDCARD_LABEL = "emberframe-nodes"


@dataclass(frozen=True)
class WildcardRule:
    token_name: str
    wildcard_ref: str
    mode: str
    start_line_number: int
    repeat_each_line: int

    @property
    def placeholder(self) -> str:
        name = self.token_name.strip()
        if name.startswith("{") and name.endswith("}"):
            return name
        return "{" + name + "}"


class EmberFrameWildcardBase:
    @staticmethod
    def _normalize_mode(mode: str) -> str:
        m = (mode or "").strip().lower()
        aliases = {
            "increment": "next",
            "decrement": "previous",
        }
        return aliases.get(m, m)

    @staticmethod
    def _load_lines(path: Path) -> List[str]:
        raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        cleaned: List[str] = []
        for line in raw:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            cleaned.append(stripped)
        return cleaned

    @classmethod
    def _resolve_index(
        cls,
        line_count: int,
        mode: str,
        start_line_number: int,
        repeat_each_line: int,
        run_counter: int,
        random_key: str,
    ) -> int:
        if line_count <= 0:
            return 0

        mode = cls._normalize_mode(mode)
        base_idx = max(0, start_line_number - 1)
        block = max(0, int(run_counter)) // max(1, int(repeat_each_line))

        if mode == "fixed":
            idx = base_idx
        elif mode == "next":
            idx = base_idx + block
        elif mode == "previous":
            idx = base_idx - block
        elif mode == "randomize":
            seed_material = f"{random_key}|{block}|{line_count}".encode("utf-8")
            seed_value = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
            idx = random.Random(seed_value).randrange(line_count)
        else:
            idx = base_idx
        return idx % line_count

    @staticmethod
    def _append_prompt_part(base: str, new_part: str) -> str:
        base = (base or "").strip().rstrip(",")
        new_part = (new_part or "").strip().lstrip(",")
        if not base:
            return new_part
        if not new_part:
            return base
        return f"{base}, {new_part}"

    @classmethod
    def _discover_wildcard_files(cls) -> Dict[str, Path]:
        roots = cls._candidate_wildcard_roots()
        local_wildcards = (Path(__file__).resolve().parent / "wildcards").resolve()
        discovered: Dict[str, Path] = {}
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            root_resolved = root.resolve()
            root_label = LOCAL_WILDCARD_LABEL if root_resolved == local_wildcards else cls._label_for_root(root)
            for path in sorted(root.rglob("*.txt"), key=lambda p: str(p).lower()):
                try:
                    rel = path.relative_to(root).as_posix()
                except ValueError:
                    rel = path.name
                discovered[f"{root_label}:{rel}"] = path
        return discovered

    @classmethod
    def _candidate_wildcard_roots(cls) -> List[Path]:
        here = Path(__file__).resolve().parent
        custom_nodes_dir = here.parent
        comfy_root = custom_nodes_dir.parent

        roots: List[Path] = [
            here / "wildcards",
            custom_nodes_dir / "wildcards",
            comfy_root / "wildcards",
        ]
        try:
            for sibling in custom_nodes_dir.iterdir():
                if sibling.is_dir():
                    wc = sibling / "wildcards"
                    if wc.exists() and wc.is_dir():
                        roots.append(wc)
        except OSError:
            pass

        seen = set()
        unique_roots: List[Path] = []
        for root in roots:
            key = str(root.resolve()) if root.exists() else str(root)
            if key in seen:
                continue
            seen.add(key)
            unique_roots.append(root)
        return unique_roots

    @staticmethod
    def _label_for_root(root: Path) -> str:
        parent_name = root.parent.name.strip()
        if parent_name:
            return parent_name
        return root.name.strip() or "wildcards"

    @classmethod
    def _resolve_wildcard_path(cls, wildcard_ref: str, discovered: Dict[str, Path]) -> Path | None:
        ref = wildcard_ref.strip()
        if not ref:
            return None
        if ref in discovered:
            return discovered[ref]

        candidates: List[Tuple[str, Path]] = []
        ref_norm = ref.replace("\\", "/").lower()
        for label, path in discovered.items():
            label_suffix = label.split(":", 1)[1] if ":" in label else label
            suffix_norm = label_suffix.replace("\\", "/").lower()
            name_norm = path.name.lower()
            if ref_norm == suffix_norm or ref_norm == name_norm:
                candidates.append((label, path))

        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) > 1:
            raise ValueError(
                f"Wildcard reference '{wildcard_ref}' is ambiguous. Use one of: "
                + ", ".join(label for label, _ in candidates[:12])
            )

        direct = Path(ref)
        if direct.exists() and direct.is_file():
            return direct
        return None


class EmberFrameSequentialWildcardPrompt(EmberFrameWildcardBase):
    @classmethod
    def INPUT_TYPES(cls):
        discovered = cls._discover_wildcard_files()
        options = ["<none>"] + sorted(discovered.keys(), key=str.lower)
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": "Use {wildcard}, {{wildcard}}, or __WILDCARD__ to place the selected line.",
                    },
                ),
                "wildcard_file": (
                    options,
                    {
                        "default": "<none>",
                        "tooltip": "Wildcard .txt file to load. Blank lines and comments are ignored.",
                    },
                ),
                "line_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt_out", "selected_line", "resolved_line_number", "wildcard_path")
    FUNCTION = "build_prompt"
    CATEGORY = "EmberFrame/Prompt"

    @classmethod
    def IS_CHANGED(cls, prompt, wildcard_file, line_index):
        files = cls._discover_wildcard_files()
        wildcard_path = files.get(wildcard_file)
        mtime = None
        if wildcard_path is not None:
            try:
                mtime = wildcard_path.stat().st_mtime_ns
            except OSError:
                mtime = "missing"
        return (prompt, wildcard_file, line_index, mtime)

    def build_prompt(self, prompt: str, wildcard_file: str, line_index: int):
        files = self._discover_wildcard_files()
        wildcard_path = files.get(wildcard_file)
        selected_line = ""
        resolved_line_number = 0
        wildcard_path_str = ""

        if wildcard_path is not None:
            wildcard_path_str = str(wildcard_path)
            lines = self._load_lines(wildcard_path)
            if lines:
                resolved_idx = int(line_index) % len(lines)
                selected_line = lines[resolved_idx]
                resolved_line_number = resolved_idx + 1

        final_prompt = self._combine_prompt(prompt or "", selected_line)
        return (final_prompt, selected_line, resolved_line_number, wildcard_path_str)

    @classmethod
    def _combine_prompt(cls, prompt: str, selected_line: str) -> str:
        prompt = (prompt or "").strip()
        selected_line = (selected_line or "").strip()

        if not prompt:
            return selected_line
        if not selected_line:
            return prompt

        for token in WILDCARD_PLACEHOLDERS:
            if token in prompt:
                return prompt.replace(token, selected_line)
        return cls._append_prompt_part(prompt, selected_line)


class EmberFrameWildcardRuleBuilder(EmberFrameWildcardBase):
    @classmethod
    def INPUT_TYPES(cls):
        discovered = cls._discover_wildcard_files()
        options = sorted(discovered.keys(), key=str.lower) or ["<no wildcard files found>"]
        return {
            "required": {
                "token_name": ("STRING", {"default": "character"}),
                "wildcard_file": (options, {"default": options[0]}),
                "mode": (["fixed", "next", "previous", "randomize"], {"default": "fixed"}),
                "start_line_number": ("INT", {"default": 1, "min": 1, "max": 2147483647}),
                "repeat_each_line": ("INT", {"default": 1, "min": 1, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("rule_line", "rule_preview")
    FUNCTION = "build_rule"
    CATEGORY = "EmberFrame/Prompt"

    def build_rule(self, token_name: str, wildcard_file: str, mode: str, start_line_number: int, repeat_each_line: int):
        if wildcard_file == "<no wildcard files found>":
            raise ValueError("No wildcard files were discovered. Add txt files to a scanned wildcards folder and restart ComfyUI.")
        rule_line = f"{token_name.strip()} | {wildcard_file} | {mode} | {int(start_line_number)} | {int(repeat_each_line)}"
        rule_preview = (
            f"token: {token_name.strip()}\n"
            f"wildcard_file: {wildcard_file}\n"
            f"mode: {mode}\n"
            f"start_line: {int(start_line_number)}\n"
            f"repeat_each_line: {int(repeat_each_line)}\n\n"
            f"rule_line:\n{rule_line}"
        )
        return (rule_line, rule_preview)


class EmberFrameWildcardConfigCombiner(EmberFrameWildcardBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rule_1": ("STRING", {"multiline": False, "default": ""}),
                "rule_2": ("STRING", {"multiline": False, "default": ""}),
                "rule_3": ("STRING", {"multiline": False, "default": ""}),
                "rule_4": ("STRING", {"multiline": False, "default": ""}),
                "rule_5": ("STRING", {"multiline": False, "default": ""}),
                "rule_6": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("wildcard_config", "rule_count")
    FUNCTION = "combine_rules"
    CATEGORY = "EmberFrame/Prompt"

    def combine_rules(self, rule_1: str, rule_2: str, rule_3: str, rule_4: str, rule_5: str, rule_6: str):
        rules = [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6]
        cleaned = [r.strip() for r in rules if r and r.strip()]
        return ("\n".join(cleaned), len(cleaned))


class EmberFrameWildcardPromptAssembler(EmberFrameWildcardBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "masterpiece, cinematic lighting, {character}, wearing {outfit}, {pose}",
                    },
                ),
                "wildcard_config": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                    },
                ),
                "run_counter": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt_out", "preview_text", "selected_values", "resolved_token_count")
    FUNCTION = "assemble_prompt"
    CATEGORY = "EmberFrame/Prompt"

    @classmethod
    def IS_CHANGED(cls, prompt, wildcard_config, run_counter):
        file_mtimes = []
        discovered = cls._discover_wildcard_files()
        try:
            rules = cls._parse_rules(wildcard_config)
        except Exception:
            rules = []
        for rule in rules:
            path = cls._resolve_wildcard_path(rule.wildcard_ref, discovered)
            if path is not None:
                try:
                    file_mtimes.append((str(path), path.stat().st_mtime_ns))
                except OSError:
                    file_mtimes.append((str(path), "missing"))
        return (prompt, wildcard_config, run_counter, tuple(file_mtimes))

    def assemble_prompt(self, prompt: str, wildcard_config: str, run_counter: int):
        prompt = (prompt or "").strip()
        rules = self._parse_rules(wildcard_config or "")
        discovered = self._discover_wildcard_files()

        resolved_entries = []
        resolved_prompt = prompt

        for rule in rules:
            path = self._resolve_wildcard_path(rule.wildcard_ref, discovered)
            if path is None:
                available_hint = ", ".join(sorted(list(discovered.keys()))[:12])
                raise ValueError(
                    f"Wildcard file not found for '{rule.wildcard_ref}'. "
                    f"Some available entries: {available_hint}"
                )

            lines = self._load_lines(path)
            if not lines:
                raise ValueError(f"Wildcard file '{path}' contains no usable lines.")

            idx = self._resolve_index(
                line_count=len(lines),
                mode=rule.mode,
                start_line_number=rule.start_line_number,
                repeat_each_line=rule.repeat_each_line,
                run_counter=run_counter,
                random_key=f"{rule.token_name}|{rule.wildcard_ref}|{path}",
            )
            selected_line = lines[idx]
            resolved_line_number = idx + 1
            normalized_mode = self._normalize_mode(rule.mode)

            if rule.placeholder in resolved_prompt:
                resolved_prompt = resolved_prompt.replace(rule.placeholder, selected_line)
            else:
                resolved_prompt = self._append_prompt_part(resolved_prompt, selected_line)

            resolved_entries.append({
                "token_name": rule.token_name,
                "placeholder": rule.placeholder,
                "wildcard_ref": rule.wildcard_ref,
                "wildcard_path": str(path),
                "mode": normalized_mode,
                "start_line_number": rule.start_line_number,
                "repeat_each_line": rule.repeat_each_line,
                "resolved_line_number": resolved_line_number,
                "selected_line": selected_line,
            })

        selected_values = self._build_selected_values(resolved_entries)
        preview_text = self._build_preview(prompt, resolved_prompt, run_counter, resolved_entries)
        return (resolved_prompt, preview_text, selected_values, len(resolved_entries))

    @classmethod
    def _parse_rules(cls, config_text: str) -> List[WildcardRule]:
        rules: List[WildcardRule] = []
        for line_no, raw_line in enumerate(config_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid wildcard_config line {line_no}: expected 5 fields "
                    f"'token | file | mode | start_line | repeat_each_line' but got: {raw_line}"
                )

            token_name, wildcard_ref, mode, start_line, repeat_each_line = parts
            mode_normalized = cls._normalize_mode(mode)
            if mode_normalized not in {"fixed", "next", "previous", "randomize"}:
                raise ValueError(f"Invalid mode '{mode}' on line {line_no}.")
            try:
                start_line_number = int(start_line)
                repeat_count = int(repeat_each_line)
            except ValueError:
                raise ValueError(f"Invalid numeric value on line {line_no}.")

            if start_line_number < 1:
                raise ValueError(f"start_line must be >= 1 on line {line_no}.")
            if repeat_count < 1:
                raise ValueError(f"repeat_each_line must be >= 1 on line {line_no}.")
            if not token_name:
                raise ValueError(f"token may not be blank on line {line_no}.")
            if not wildcard_ref:
                raise ValueError(f"file may not be blank on line {line_no}.")

            rules.append(WildcardRule(token_name, wildcard_ref, mode_normalized, start_line_number, repeat_count))
        return rules

    @staticmethod
    def _build_selected_values(entries: List[Dict[str, str]]) -> str:
        lines = []
        for item in entries:
            lines.append(
                f"{item['placeholder']} -> line {item['resolved_line_number']} "
                f"from {item['wildcard_ref']} :: {item['selected_line']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_preview(original_prompt: str, final_prompt: str, run_counter: int, entries: List[Dict[str, str]]) -> str:
        lines = [f"run_counter: {run_counter}", "", "original_prompt:", original_prompt, "", "resolved_values:"]
        if entries:
            for item in entries:
                lines.extend([
                    f"- token: {item['token_name']}",
                    f"  placeholder: {item['placeholder']}",
                    f"  wildcard_ref: {item['wildcard_ref']}",
                    f"  wildcard_path: {item['wildcard_path']}",
                    f"  mode: {item['mode']}",
                    f"  start_line: {item['start_line_number']}",
                    f"  repeat_each_line: {item['repeat_each_line']}",
                    f"  resolved_line: {item['resolved_line_number']}",
                    f"  selected_line: {item['selected_line']}",
                    "",
                ])
        else:
            lines.append("(no wildcard rules resolved)")
            lines.append("")
        lines.extend(["final_prompt:", final_prompt])
        return "\n".join(lines).rstrip()


class EmberFrameMPAspectResolutionSelector64:
    ASPECTS = [
        ("1:1", (1, 1)),
        ("2:1", (2, 1)),
        ("3:2", (3, 2)),
        ("4:3", (4, 3)),
        ("5:4", (5, 4)),
        ("16:9", (16, 9)),
        ("21:9", (21, 9)),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        aspect_names = [a[0] for a in cls.ASPECTS]
        return {
            "required": {
                "aspect": (aspect_names, {"default": "16:9"}),
                "orientation": (["auto", "portrait", "landscape"], {"default": "auto"}),
                "megapixels": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 4.0, "step": 0.25}),
                "align": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8}),
                "priority": (["balanced", "area", "aspect"], {"default": "balanced"}),
                "search_steps": ("INT", {"default": 24, "min": 4, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "actual_megapixels")
    FUNCTION = "pick"
    CATEGORY = "EmberFrame/Resolution"

    def pick(self, aspect, orientation, megapixels, align, priority, search_steps):
        w_ratio, h_ratio = dict(self.ASPECTS)[aspect]

        if orientation == "portrait" and w_ratio > h_ratio:
            w_ratio, h_ratio = h_ratio, w_ratio
        elif orientation == "landscape" and h_ratio > w_ratio:
            w_ratio, h_ratio = h_ratio, w_ratio

        target_area = int(round(float(megapixels) * 1_000_000))
        k = math.sqrt(target_area / (w_ratio * h_ratio))
        ideal_w = w_ratio * k

        def snap(x):
            return max(int(align), int(round(x / int(align))) * int(align))

        base_w = snap(ideal_w)
        if priority == "area":
            w_area, w_aspect = 1.0, 0.15
        elif priority == "aspect":
            w_area, w_aspect = 0.35, 1.0
        else:
            w_area, w_aspect = 1.0, 0.6

        target_ratio = w_ratio / h_ratio
        best = None
        best_score = float("inf")

        for i in range(-int(search_steps), int(search_steps) + 1):
            cand_w = base_w + i * int(align)
            if cand_w < int(align):
                continue
            cand_h = snap(cand_w / target_ratio)
            area = cand_w * cand_h
            ratio = cand_w / cand_h
            area_err = abs(area - target_area) / target_area
            aspect_err = abs(ratio - target_ratio) / target_ratio
            score = w_area * area_err + w_aspect * aspect_err
            if score < best_score or (best is not None and abs(score - best_score) < 1e-12 and area < best[2]):
                best_score = score
                best = (cand_w, cand_h, area)

        width, height, area = best
        return (int(width), int(height), float(area / 1_000_000.0))


NODE_CLASS_MAPPINGS = {
    "EmberFrameSamplerCustomAdvancedCapture": PiDAdvancedSamplerCustomAdvancedCapture,
    "EmberFrameAttachSigmaToLatent": PiDAdvancedAttachSigmaToLatent,
    "EmberFrameNormalizeZImageFluxLatent": PiDAdvancedNormalizeZImageFluxLatent,
    "EmberFrameLatentCaptureInspector": PiDAdvancedLatentCaptureInspector,
    "EmberFrameSequentialWildcardPrompt": EmberFrameSequentialWildcardPrompt,
    "EmberFrameWildcardRuleBuilder": EmberFrameWildcardRuleBuilder,
    "EmberFrameWildcardConfigCombiner": EmberFrameWildcardConfigCombiner,
    "EmberFrameWildcardPromptAssembler": EmberFrameWildcardPromptAssembler,
    "EmberFrameMPAspectResolutionSelector64": EmberFrameMPAspectResolutionSelector64,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmberFrameSamplerCustomAdvancedCapture": "EmberFrame SamplerCustomAdvanced Capture",
    "EmberFrameAttachSigmaToLatent": "EmberFrame Attach Sigma To Latent",
    "EmberFrameNormalizeZImageFluxLatent": "EmberFrame Normalize ZImage/Flux Latent",
    "EmberFrameLatentCaptureInspector": "EmberFrame Latent Capture Inspector",
    "EmberFrameSequentialWildcardPrompt": "EmberFrame Sequential Wildcard Prompt",
    "EmberFrameWildcardRuleBuilder": "EmberFrame Wildcard Rule Builder",
    "EmberFrameWildcardConfigCombiner": "EmberFrame Wildcard Config Combiner",
    "EmberFrameWildcardPromptAssembler": "EmberFrame Wildcard Prompt Assembler",
    "EmberFrameMPAspectResolutionSelector64": "EmberFrame MP Aspect Resolution Selector",
}
