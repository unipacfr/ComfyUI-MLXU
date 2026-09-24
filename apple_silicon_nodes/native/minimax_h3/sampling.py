"""Flow-matching `res_multistep` sampling loop for MiniMax H3's dual
video+audio latent.

Deliberately NOT wired into `apple_silicon_nodes/sampler/core.py`'s shared
`_SamplerCore` (2000+ lines, used live by every other family: teacache,
kontext, LoRA schedules, identity edit, CFG variants). None of
that applies to MiniMax H3's single conditional forward pass over two
latent streams with no CFG (the reference workflow uses `BasicGuider`, not
`CFGGuider`) -- reusing the shared class would mean carrying its full
complexity for zero benefit, and touching a class every other family
depends on live for a genuinely different shape of problem is a
disproportionate risk. This module is self-contained instead.

The sigma-schedule and solver formulas below are deliberately NOT imported
from `sampler/scheduling.py`/`sampler/solvers.py` even though they compute
the same thing -- `native/` has no dependency on `sampler/` anywhere else in
this project (the reverse is the only direction: `sampler/` orchestrates
`native/` models), and this is a self-contained scalar formula, not a module
worth inverting that layering for. Verified to match:

- `minimax_h3_sigma_schedule` reproduces `comfy/samplers.py::simple_scheduler`
  exactly (the scheduler the real ComfyUI MiniMax H3 workflow uses:
  `BasicScheduler(scheduler="simple")`), itself built on
  `comfy/model_sampling.py::ModelSamplingDiscreteFlow` (`sigma(t) =
  time_snr_shift(shift, t/1000)`, a 1000-entry precomputed table, `multiplier
  =1000` default) -- matches `sampler/scheduling.py::time_snr_shift` bit for
  bit (same formula, ported there from the same real comfy source).
- `_res_multistep_step` ports `comfy/k_diffusion/sampling.py::res_multistep`
  with `eta=0., cfg_pp=False` (i.e. exactly what `sample_res_multistep`
  calls) -- the solver the real MiniMax H3 ComfyUI template selects via
  `KSamplerSelect(sampler_name="res_multistep")`. With `eta=0`,
  `get_ancestral_step` always returns `(sigma_to, 0.)`, so the ancestral
  noise-injection branch never fires and is omitted here. Verified
  numerically against the real torch implementation (exact match to
  float64 precision) with a synthetic denoiser and a monotonic decreasing
  schedule before being ported. Second-order once warmed up (from
  https://arxiv.org/pdf/2308.02157); the first step of each stream is a
  plain Euler step because there is no `old_denoised` yet to build the
  multistep estimate from -- this replaces the previous plain first-order
  Euler solver used for every step, which needed far more steps than this
  project's default to converge to a comparably clean result (undersampling
  with a first-order solver is what produced visibly grainy video at low
  step counts).
"""

from __future__ import annotations

import math

import mlx.core as mx

from .condition import ConditionPayload, default_noise, prepare_condition
from .model import MiniMaxH3Model, time_shift_sigma


def _time_snr_shift(shift: float, t: float) -> float:
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


def minimax_h3_sigma_schedule(shift: float, steps: int) -> list[float]:
    """Video-stream sigma schedule, matching ComfyUI's `simple_scheduler`
    run against a `ModelSamplingDiscreteFlow`-shaped model with this `shift`."""
    table = [_time_snr_shift(shift, (i + 1) / 1000.0) for i in range(1000)]
    stride = len(table) / steps
    sigmas = [table[-(1 + int(x * stride))] for x in range(steps)]
    sigmas.append(0.0)
    return sigmas


def _t_fn(sigma: float) -> float:
    return -math.log(sigma)


def _phi1(t: float) -> float:
    return math.expm1(t) / t


def _phi2(t: float) -> float:
    return (_phi1(t) - 1.0) / t


class _ResMultistepState:
    """Per-stream state `res_multistep` carries across steps (video and
    audio each need their own -- they run independent schedules)."""

    def __init__(self) -> None:
        self.old_sigma_down: float | None = None
        self.old_denoised: mx.array | None = None


def _res_multistep_step(
    state: _ResMultistepState,
    x: mx.array,
    sigma: float,
    sigma_prev: float | None,
    sigma_next: float,
    denoised: mx.array,
) -> mx.array:
    """One `res_multistep` (eta=0, cfg_pp=False) step -- see module docstring."""
    sigma_down = sigma_next  # eta=0: get_ancestral_step returns (sigma_to, 0.)

    if sigma_down == 0.0 or state.old_denoised is None:
        d = (x - denoised) / sigma
        x_next = x + d * (sigma_down - sigma)
    else:
        t = _t_fn(sigma)
        t_old = _t_fn(state.old_sigma_down)
        t_next = _t_fn(sigma_down)
        t_prev = _t_fn(sigma_prev)
        h = t_next - t
        c2 = (t_prev - t_old) / h

        phi1_val, phi2_val = _phi1(-h), _phi2(-h)
        ratio = 0.0 if c2 == 0.0 else phi2_val / c2
        b1 = phi1_val - ratio
        b2 = ratio
        if math.isnan(b1):
            b1 = 0.0
        if math.isnan(b2):
            b2 = 0.0

        x_next = math.exp(-h) * x + h * (b1 * denoised + b2 * state.old_denoised)

    state.old_denoised = denoised
    state.old_sigma_down = sigma_down
    return x_next


def run_minimax_h3_sampling(
    model: MiniMaxH3Model,
    video_latent: mx.array,
    audio_latent: mx.array,
    context: mx.array,
    steps: int,
    payload: ConditionPayload | None = None,
    noise_fn=None,
) -> tuple[mx.array, mx.array]:
    """Denoise `video_latent`/`context`-conditioned pure Gaussian noise into
    a finished video+audio latent pair. `video_latent`/`audio_latent` are the
    starting noise (e.g. from `ASDX_MiniMaxH3EmptyLatentAV` scaled by noise --
    the caller is responsible for providing actual noise, not zeros; an
    all-zero start never moves under this ODE since `denoised = x` at
    `sigma=0` trivially and every step's `d` depends on `x` having signal).

    The audio stream runs on its own schedule (`config.sigma_shift_audio`,
    derived from the video sigma via `time_shift_sigma` -- the exact
    relationship `MiniMaxH3Model.__call__` uses internally for `t_a`), so it
    gets its own `res_multistep` step each iteration -- independent state
    from, but computed in the same forward pass as, the video stream's
    step.

    `payload`: keyframe/reference conditions, prepared once for the whole run
    (the augmentation noise is identical at every step). `noise_fn` overrides
    the augmentation noise source (defaults to `default_noise`)."""
    cfg = model.config
    sigmas = minimax_h3_sigma_schedule(cfg.sigma_shift_video, steps)
    cond = (
        prepare_condition(payload, cfg.patch_size, noise_fn or default_noise) if payload is not None else None
    )

    video_state, audio_state = _ResMultistepState(), _ResMultistepState()
    video, audio = video_latent, audio_latent
    for t in range(steps):
        sigma_v, sigma_v_next = sigmas[t], sigmas[t + 1]
        sigma_v_prev = sigmas[t - 1] if t >= 1 else None
        sigma_a = time_shift_sigma(sigma_v, cfg.sigma_shift_video, cfg.sigma_shift_audio)
        sigma_a_next = time_shift_sigma(sigma_v_next, cfg.sigma_shift_video, cfg.sigma_shift_audio)
        sigma_a_prev = (
            time_shift_sigma(sigma_v_prev, cfg.sigma_shift_video, cfg.sigma_shift_audio)
            if sigma_v_prev is not None
            else None
        )

        if cond is None:
            video_v, audio_v = model(video, audio, context, sigma_v=sigma_v)
        else:
            video_v, audio_v = model(video, audio, context, sigma_v=sigma_v, cond=cond)
        video_denoised = video - video_v * sigma_v
        audio_denoised = audio - audio_v * sigma_a

        video = _res_multistep_step(video_state, video, sigma_v, sigma_v_prev, sigma_v_next, video_denoised)
        audio = _res_multistep_step(audio_state, audio, sigma_a, sigma_a_prev, sigma_a_next, audio_denoised)
        mx.eval(video, audio)

    return video, audio
