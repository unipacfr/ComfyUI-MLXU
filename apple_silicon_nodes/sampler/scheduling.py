"""Sigma scheduling for diffusion models.

Adapted from DiffusionKit's ModelSamplingDiscreteFlow.
Provides sigma/timestep conversion, noise scaling, and schedule generation
for FLUX and discrete flow models.
"""

from __future__ import annotations

import math
from typing import Any


class SDXLSampling:
    """Discrete DDPM/EPS sigma schedule for SDXL.

    Fundamentally different from FLUX/Krea2's flow-matching schedule:
    SDXL is trained on 1000 DISCRETE diffusion steps with a fixed
    beta/alpha_cumprod schedule, not a continuous sigma in [0,1]. Matches
    `comfy/model_sampling.py::ModelSamplingDiscrete` + `EPS` exactly:

        betas = linspace(sqrt(linear_start), sqrt(linear_end), 1000) ** 2
        alphas_cumprod = cumprod(1 - betas)
        sigmas[i] = sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i])

    `sigma()`/`timestep()` convert between this discrete 1000-step table and
    a continuous sigma value (needed because our Euler loop can use fewer
    than 1000 steps): `sigma(t)` interpolates LINEARLY IN LOG-SIGMA SPACE
    between the two nearest discrete bins (matching comfy's own
    `ModelSamplingDiscrete.sigma`); `timestep(sigma)` finds the nearest
    discrete bin by log-sigma distance (matching `.timestep`'s
    `argmin(|log_sigma - log_sigmas|)` — nearest-neighbor, NOT interpolated).

    `calculate_input`/`calculate_denoised` are `EPS`'s Karras preconditioning
    (`comfy/model_sampling.py::EPS`): the UNet expects `x / sqrt(sigma^2+1)`
    as input (not raw `x`), and predicts noise (`eps`), converted to the
    denoised sample via `x - eps*sigma` — NOT FLUX's flow-matching formulas.
    """

    def __init__(self, linear_start: float = 0.00085, linear_end: float = 0.012,
                 num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        self.sigma_data = 1.0

        sqrt_start, sqrt_end = math.sqrt(linear_start), math.sqrt(linear_end)
        betas = [
            (sqrt_start + (sqrt_end - sqrt_start) * i / (num_timesteps - 1)) ** 2
            for i in range(num_timesteps)
        ]
        alphas_cumprod: list[float] = []
        acc = 1.0
        for b in betas:
            acc *= (1.0 - b)
            alphas_cumprod.append(acc)
        self._sigmas = [math.sqrt((1.0 - ac) / ac) for ac in alphas_cumprod]
        self._log_sigmas = [math.log(s) for s in self._sigmas]

    @property
    def sigma_min(self) -> float:
        return self._sigmas[0]

    @property
    def sigma_max(self) -> float:
        return self._sigmas[-1]

    @property
    def sigmas(self) -> list[float]:
        """Public alias of the discrete table, for the generic scheduler
        functions in calculate_sigmas() below (simple/beta need direct
        index access, matching comfy's `model_sampling.sigmas`)."""
        return self._sigmas

    def sigma(self, timestep: float) -> float:
        """Continuous sigma for a (possibly fractional) discrete timestep index."""
        t = max(0.0, min(timestep, self.num_timesteps - 1))
        low = int(math.floor(t))
        high = min(low + 1, self.num_timesteps - 1)
        w = t - low
        log_sigma = (1 - w) * self._log_sigmas[low] + w * self._log_sigmas[high]
        return math.exp(log_sigma)

    def timestep(self, sigma: float) -> float:
        """Nearest discrete timestep index for a continuous sigma (log-sigma distance)."""
        log_sigma = math.log(max(sigma, 1e-10))
        best_i, best_d = 0, float("inf")
        for i, ls in enumerate(self._log_sigmas):
            d = abs(log_sigma - ls)
            if d < best_d:
                best_d = d
                best_i = i
        return float(best_i)

    def calculate_input(self, sigma: float, x: Any) -> Any:
        """UNet input preconditioning: x / sqrt(sigma^2 + sigma_data^2)."""
        return x / math.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def calculate_denoised(self, sigma: float, model_output: Any, model_input: Any) -> Any:
        """EPS denoise: x0 = x - eps * sigma."""
        return model_input - model_output * sigma


def generate_sigmas_sdxl(steps: int) -> list[float]:
    """Generate an SDXL sigma schedule ('normal'-scheduler style, matching
    comfy's `normal_scheduler`): linearly spaced discrete timestep indices
    (from the last index down to 0), each converted through
    `SDXLSampling.sigma()` (log-sigma interpolation), with an explicit 0.0
    appended at the end — same "always end at exact 0.0" convention as
    `generate_sigmas()` above.

    NOT merged into `generate_sigmas()`'s dispatcher: SDXL's denoising loop
    needs a genuinely different per-step shape (two-pass CFG, EPS
    preconditioning) that the flow-matching Euler loop doesn't, so it has
    its own dedicated entry point.

    Returns:
        List of sigma values with length steps + 1.
    """
    sampling = SDXLSampling()
    start = float(sampling.num_timesteps - 1)
    if steps <= 1:
        timesteps = [start]
    else:
        timesteps = [start - i * start / (steps - 1) for i in range(steps)]
    sigmas = [sampling.sigma(t) for t in timesteps]
    sigmas.append(0.0)
    return sigmas


def flux_time_shift(mu: float, sigma: float, t: float) -> float:
    """Matches comfy.model_sampling.flux_time_shift exactly."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def time_snr_shift(shift: float, t: float) -> float:
    """Matches comfy.model_sampling.time_snr_shift exactly.

    Used by `ModelSamplingDiscreteFlow` (Lumina2/Z-Image's sampling class) —
    a FIXED shift, unlike FLUX-dev's `flux_time_shift` which takes a
    resolution-dependent `mu`. Z-Image's `sampling_settings = {"shift": 3.0,
    "multiplier": 1.0}` (comfy/supported_models.py::ZImage) — multiplier=1.0
    means `timestep(sigma)=sigma` directly (no discrete-table lookup needed
    for our continuous-sigma Euler loop).
    """
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


def _discrete_flow_fixed_shift_sigmas(shift: float, steps: int) -> list[float]:
    """Sigma schedule for a `ModelSamplingDiscreteFlow`-backed model with
    multiplier=1.0 (Z-Image, Anima) -- matches `comfy.samplers.normal_scheduler`
    run on a real `ModelSamplingDiscreteFlow` instance exactly.

    Same double-application shape as `_flux_fixed_shift_sigmas` (its
    ModelSamplingFlux analog): `ModelSamplingDiscreteFlow.timestep(sigma) ==
    sigma * multiplier` is also the identity when multiplier=1.0, so the
    linspace endpoint is itself a sigma value (`sigma_min = time_snr_shift(shift,
    1/1000)`, using the class's own default `timesteps=1000`), which then gets
    run back through `time_snr_shift` again per grid point via `.sigma()`.
    Verified against a real `ModelSamplingDiscreteFlow(shift=3.0,
    multiplier=1.0)` run through `comfy.samplers.calculate_sigmas(..., "normal",
    ...)` -- see tests/test_scheduling_anima.py's header for the exact command.
    """
    sigma_min = time_snr_shift(shift, 1.0 / 1000.0)
    if steps > 1:
        timesteps = [1.0 + (sigma_min - 1.0) * i / (steps - 1) for i in range(steps)]
    else:
        timesteps = [1.0]
    sigmas = [time_snr_shift(shift, t) for t in timesteps]
    sigmas.append(0.0)
    return sigmas


def _flux_fixed_shift_sigmas(shift: float, steps: int) -> list[float]:
    """Sigma schedule for a `ModelSamplingFlux`-backed model with a fixed
    (non-resolution-dependent) shift -- matches `comfy.samplers.
    normal_scheduler` run on a real `ModelSamplingFlux` instance exactly.

    `ModelSamplingFlux.timestep()` is the identity function, so the
    timestep grid is `linspace(1.0, sigma_min, steps)` (uniform in
    TIMESTEP space, not in sigma or in a plain `1 - i/steps` grid), where
    `sigma_min` is the model's precomputed near-zero floor
    (`flux_time_shift(shift, 1.0, 1/10000)`). Each grid point is then
    mapped through `flux_time_shift`, and an explicit 0.0 is appended.
    """
    sigma_min = flux_time_shift(shift, 1.0, 1.0 / 10000.0)
    if steps > 1:
        timesteps = [1.0 + (sigma_min - 1.0) * i / (steps - 1) for i in range(steps)]
    else:
        timesteps = [1.0]
    sigmas = [flux_time_shift(shift, 1.0, t) for t in timesteps]
    sigmas.append(0.0)
    return sigmas


def flux_resolution_shift(width: int, height: int,
                           max_shift: float = 1.15, base_shift: float = 0.5) -> float:
    """Resolution-dependent shift (mu), matching ComfyUI's ModelSamplingFlux node.

    Interpolates linearly between base_shift (at a 256-token image, i.e.
    16x16 latent) and max_shift (at a 4096-token image), based on the FLUX
    packed token count (width*height / (8*8*2*2) = width*height/256).
    """
    x1, x2 = 256, 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    tokens = width * height / (8 * 8 * 2 * 2)
    return tokens * mm + b


def generate_sigmas(
    steps: int,
    model_type: str,
    width: int = 1024,
    height: int = 1024,
) -> list[float]:
    """Generate sigma schedule for a given model type.

    FLUX dev uses ComfyUI's resolution-dependent flux_time_shift (CONST
    sampling: sigma(t) = exp(mu) / (exp(mu) + (1/t - 1))), applied to a
    linear t=1..~0 schedule (matching normal_scheduler + ModelSamplingFlux).
    FLUX schnell uses uniform steps (shift=1, no resolution dependence).
    Krea2 uses a linear flow-matching schedule from 1.0 to 0.0.

    Args:
        steps: Number of denoising steps.
        model_type: "dev", "schnell", "krea2", or "krea2_turbo".
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        List of sigma values with length steps + 1. Always ends at exactly 0.0.
    """
    if model_type == "schnell":
        return [1.0 - i / steps for i in range(steps + 1)]

    if model_type in ("krea2", "krea2_turbo"):
        # Krea2 registers as `model_type=ModelType.FLUX` (comfy/model_base.py
        # ::Krea2.__init__ default), which comfy's model_sampling() factory
        # (comfy/model_base.py:127) maps to `ModelSamplingFlux` -- NOT
        # `ModelSamplingDiscreteFlow`. ModelSamplingFlux.sigma() calls
        # `flux_time_shift(shift, 1.0, t)` (the SAME formula FLUX.1-dev uses,
        # just with a fixed shift=1.15 instead of a resolution-dependent mu),
        # not `time_snr_shift` -- confirmed by reading comfy/model_sampling.py
        # directly and executing both formulas: they diverge sharply away
        # from the t=0/t=1 endpoints (e.g. t=0.5, shift=1.15: flux_time_shift
        # =0.760 vs time_snr_shift=0.535). Using time_snr_shift here was a
        # real, previously-uncaught bug.
        #
        # ModelSamplingFlux.timestep() is also the IDENTITY function, so
        # comfy's `normal_scheduler` (comfy/samplers.py:671) builds its
        # timestep grid via `linspace(1.0, sigma_min, steps)` -- NOT the
        # uniform `1 - i/steps` grid used elsewhere in this file -- where
        # `sigma_min = flux_time_shift(shift, 1.0, 1/10000)` (the model's
        # precomputed near-zero floor, ~0.000316 for shift=1.15), then maps
        # each grid point through sigma() and appends an explicit 0.0.
        # Verified end-to-end against `comfy.samplers.normal_scheduler` run
        # on a real `ModelSamplingFlux` instance: exact match to 1e-6.
        shift = 1.15
        sigmas = _flux_fixed_shift_sigmas(shift, steps)
        return sigmas

    if model_type == "qwen_image21":
        # Registers as ModelType.FLUX in ComfyUI (comfy/model_base.py::QwenImage21),
        # same family as Krea2 above -- flux_time_shift/ModelSamplingFlux, NOT
        # time_snr_shift/ModelSamplingDiscreteFlow like Flux2/Z-Image below, despite
        # sharing their "fixed shift, not resolution-dependent" simplicity. Fixed
        # shift=0.69 (comfy/supported_models.py::QwenImage21.sampling_settings).
        shift = 0.69
        sigmas = _flux_fixed_shift_sigmas(shift, steps)
        return sigmas

    if model_type in ("zimage", "zimage_turbo", "anima"):
        # Flow matching with a FIXED shift (not resolution-dependent like
        # FLUX-dev's mu) — comfy/supported_models.py::ZImage.sampling_settings.
        # Anima: ModelSamplingDiscreteFlow shift=3.0, multiplier=1.0 (comfy/
        # supported_models.py::Anima) -- same curve as Z-Image. Verified
        # end-to-end against a real ModelSamplingDiscreteFlow instance (see
        # tests/test_scheduling_anima.py): the naive `time_snr_shift(shift,
        # 1-i/steps)` grid this branch used before was NOT what comfy's
        # normal_scheduler produces (it undershoots how close to 0 the
        # timestep grid gets) -- `_discrete_flow_fixed_shift_sigmas` matches.
        shift = 3.0
        sigmas = _discrete_flow_fixed_shift_sigmas(shift, steps)
        return sigmas

    if model_type == "flux2":
        # Flow matching with a FIXED shift of 2.02 (not resolution-dependent
        # like FLUX-dev's mu) — comfy/supported_models.py::Flux2.sampling_settings
        # = {"shift": 2.02}, no base_shift/max_shift interpolation, confirmed
        # by reading the real source rather than assuming parity with
        # FLUX.1-dev's resolution-dependent schedule.
        shift = 2.02
        sigmas = [time_snr_shift(shift, 1.0 - i / steps) for i in range(steps)]
        sigmas.append(0.0)
        return sigmas

    mu = flux_resolution_shift(width, height)
    # normal_scheduler linspaces from timestep(sigma_max)=1.0 down to
    # timestep(sigma_min)~=0, but since sigma_min for ModelSamplingFlux is
    # ModelSamplingFlux.sigma(1/10000) which is not exactly 0, ComfyUI adds
    # one extra step and appends 0.0 explicitly instead. Do the same: sample
    # `steps` points in (0, 1], shift each through flux_time_shift, then
    # append the exact zero endpoint — guarantees the schedule always
    # reaches 0 regardless of resolution (fixes the old formula's bug where
    # sub-1024x1024 images left a nonzero residual at the final step).
    sigmas: list[float] = []
    for i in range(steps):
        t = 1.0 - i / steps
        sigmas.append(flux_time_shift(mu, 1.0, t))
    sigmas.append(0.0)
    return sigmas


# ── Generic multi-scheduler support ─────────────────────────────────
#
# Everything above implements only the "normal" scheduler shape per model
# family. calculate_sigmas() below adds ComfyUI's other scheduler shapes
# (simple, karras, sgm_uniform, beta) as an additive layer: scheduler_name=
# "normal" still delegates straight to generate_sigmas()/generate_sigmas_sdxl()
# above (zero risk to the already-verified default path, see canon on the
# krea2 time_snr_shift/flux_time_shift mixup); the others build a small
# model_sampling object and port comfy/samplers.py's generic scheduler
# functions exactly, verified numerically against a real ModelSamplingFlux
# instance (comfy/model_sampling.py) run through comfy.samplers.calculate_sigmas.


def _flow_shift_fn(model_type: str, width: int = 1024, height: int = 1024):
    """Same per-family shift closure generate_sigmas() already uses, factored
    out for reuse by the non-"normal" schedulers below. Not merged into
    generate_sigmas() itself to avoid touching its already-verified body.
    """
    if model_type == "schnell":
        return lambda t: t
    if model_type in ("krea2", "krea2_turbo"):
        return lambda t: flux_time_shift(1.15, 1.0, t)
    if model_type == "qwen_image21":
        return lambda t: flux_time_shift(0.69, 1.0, t)
    if model_type in ("zimage", "zimage_turbo", "anima"):
        return lambda t: time_snr_shift(3.0, t)
    if model_type == "flux2":
        return lambda t: time_snr_shift(2.02, t)
    mu = flux_resolution_shift(width, height)
    return lambda t: flux_time_shift(mu, 1.0, t)


class _FlowModelSampling:
    """Minimal comfy `ModelSamplingFlux`-equivalent for the generic scheduler
    functions below: `timestep(sigma) == sigma` (identity) is comfy's own
    convention for this class (comfy/model_sampling.py::ModelSamplingFlux),
    not a physical round-trip -- ported as-is, verified numerically (see
    calculate_sigmas() docstring) rather than derived by inspection alone.
    """

    def __init__(self, sigma_fn, num_steps: int = 10000):
        self.num_steps = num_steps
        self.sigmas = [sigma_fn((i + 1) / num_steps) for i in range(num_steps)]
        self._sigma_fn = sigma_fn

    @property
    def sigma_min(self) -> float:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> float:
        return self.sigmas[-1]

    def sigma(self, t: float) -> float:
        return self._sigma_fn(t)

    def timestep(self, sigma: float) -> float:
        return sigma


def _simple_scheduler(model_sampling: Any, steps: int) -> list[float]:
    """Matches comfy/samplers.py::simple_scheduler exactly."""
    table = model_sampling.sigmas
    n = len(table)
    ss = n / steps
    sigmas = [table[-(1 + int(x * ss))] for x in range(steps)]
    sigmas.append(0.0)
    return sigmas


def _sgm_uniform_scheduler(model_sampling: Any, steps: int) -> list[float]:
    """Matches comfy/samplers.py::normal_scheduler(sgm=True) exactly."""
    start = model_sampling.timestep(model_sampling.sigma_max)
    end = model_sampling.timestep(model_sampling.sigma_min)
    n = steps + 1
    timesteps = [start + (end - start) * i / (n - 1) for i in range(n)][:-1]
    sigmas = [model_sampling.sigma(ts) for ts in timesteps]
    sigmas.append(0.0)
    return sigmas


def _karras_scheduler(sigma_min: float, sigma_max: float, steps: int, rho: float = 7.0) -> list[float]:
    """Matches k_diffusion/sampling.py::get_sigmas_karras exactly."""
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    denom = (steps - 1) if steps > 1 else 1
    sigmas = [
        (max_inv_rho + (i / denom) * (min_inv_rho - max_inv_rho)) ** rho
        for i in range(steps)
    ]
    sigmas.append(0.0)
    return sigmas


def _beta_scheduler(model_sampling: Any, steps: int, alpha: float = 0.6, beta: float = 0.6) -> list[float]:
    """Matches comfy/samplers.py::beta_scheduler exactly."""
    import scipy.stats

    table = model_sampling.sigmas
    total_timesteps = len(table) - 1
    percents = [1.0 - i / steps for i in range(steps)]
    indices = [round(float(scipy.stats.beta.ppf(p, alpha, beta)) * total_timesteps) for p in percents]

    sigmas: list[float] = []
    last_t = -1
    for t in indices:
        if t != last_t:
            sigmas.append(table[int(t)])
        last_t = t
    sigmas.append(0.0)
    return sigmas


SCHEDULER_NAMES = ["normal", "simple", "karras", "sgm_uniform", "beta"]


def calculate_sigmas(
    model_type: str,
    scheduler_name: str,
    steps: int,
    width: int = 1024,
    height: int = 1024,
) -> list[float]:
    """Generic scheduler dispatcher (mirrors comfy/samplers.py::calculate_sigmas
    + SCHEDULER_HANDLERS). "normal" delegates straight to the already-verified
    generate_sigmas()/generate_sigmas_sdxl() above; the other scheduler shapes
    build a small model_sampling object and run comfy's generic scheduler
    algorithms against it.

    Verified against a real `comfy.model_sampling.ModelSamplingFlux(shift=1.15)`
    (the krea2 family) run through `comfy.samplers.calculate_sigmas`: karras,
    simple, sgm_uniform and beta all match to the displayed precision at
    steps=8.
    """
    if scheduler_name == "normal":
        if model_type == "sdxl":
            return generate_sigmas_sdxl(steps)
        return generate_sigmas(steps, model_type, width, height)

    model_sampling: Any = (
        SDXLSampling() if model_type == "sdxl"
        else _FlowModelSampling(_flow_shift_fn(model_type, width, height))
    )

    if scheduler_name == "karras":
        return _karras_scheduler(model_sampling.sigma_min, model_sampling.sigma_max, steps)
    if scheduler_name == "simple":
        return _simple_scheduler(model_sampling, steps)
    if scheduler_name == "sgm_uniform":
        return _sgm_uniform_scheduler(model_sampling, steps)
    if scheduler_name == "beta":
        return _beta_scheduler(model_sampling, steps)

    raise ValueError(f"ASDX: unknown scheduler_name {scheduler_name!r}")
