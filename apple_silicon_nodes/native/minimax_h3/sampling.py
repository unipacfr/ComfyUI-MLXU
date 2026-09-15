"""Flow-matching Euler sampling loop for MiniMax H3's dual video+audio latent.

Deliberately NOT wired into `apple_silicon_nodes/sampler/core.py`'s shared
`_SamplerCore` (2000+ lines, used live by every other family: teacache,
kontext, LoRA schedules, identity edit, ControlNet, CFG variants). None of
that applies to MiniMax H3's single conditional forward pass over two
latent streams with no CFG (the reference workflow uses `BasicGuider`, not
`CFGGuider`) -- reusing the shared class would mean carrying its full
complexity for zero benefit, and touching a class every other family
depends on live for a genuinely different shape of problem is a
disproportionate risk. This module is self-contained instead.

The sigma-schedule and Euler-step formulas below are deliberately NOT
imported from `sampler/scheduling.py`/`sampler/solvers.py` even though they
compute the same thing -- `native/` has no dependency on `sampler/` anywhere
else in this project (the reverse is the only direction: `sampler/`
orchestrates `native/` models), and this is a 3-line formula, not a module
worth inverting that layering for. Verified to match:

- `minimax_h3_sigma_schedule` reproduces `comfy/samplers.py::simple_scheduler`
  exactly (the scheduler the real ComfyUI MiniMax H3 workflow uses:
  `BasicScheduler(scheduler="simple")`), itself built on
  `comfy/model_sampling.py::ModelSamplingDiscreteFlow` (`sigma(t) =
  time_snr_shift(shift, t/1000)`, a 1000-entry precomputed table, `multiplier
  =1000` default) -- matches `sampler/scheduling.py::time_snr_shift` bit for
  bit (same formula, ported there from the same real comfy source).
- The per-step update (`denoised = x - v*sigma`, `x_next = x + (x-denoised)/
  sigma * (sigma_next-sigma)`) matches `comfy.model_sampling.CONST.
  calculate_denoised` and `sampler/solvers.py::step_euler`/`_to_d` exactly
  (k_diffusion's standard Euler step, `to_d` reducing to the raw model
  output since `denoised = x - v*sigma` everywhere in this project).
"""

from __future__ import annotations

import mlx.core as mx

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


def _euler_step(x: mx.array, sigma: float, sigma_next: float, denoised: mx.array) -> mx.array:
    if sigma == 0.0:
        return denoised
    d = (x - denoised) / sigma
    return x + d * (sigma_next - sigma)


def run_minimax_h3_sampling(
    model: MiniMaxH3Model,
    video_latent: mx.array,
    audio_latent: mx.array,
    context: mx.array,
    steps: int,
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
    gets its own Euler step each iteration -- independent from, but computed
    in the same forward pass as, the video stream's step."""
    cfg = model.config
    sigmas = minimax_h3_sigma_schedule(cfg.sigma_shift_video, steps)

    video, audio = video_latent, audio_latent
    for t in range(steps):
        sigma_v, sigma_v_next = sigmas[t], sigmas[t + 1]
        sigma_a = time_shift_sigma(sigma_v, cfg.sigma_shift_video, cfg.sigma_shift_audio)
        sigma_a_next = time_shift_sigma(sigma_v_next, cfg.sigma_shift_video, cfg.sigma_shift_audio)

        video_v, audio_v = model(video, audio, context, sigma_v=sigma_v)
        video_denoised = video - video_v * sigma_v
        audio_denoised = audio - audio_v * sigma_a

        video = _euler_step(video, sigma_v, sigma_v_next, video_denoised)
        audio = _euler_step(audio, sigma_a, sigma_a_next, audio_denoised)
        mx.eval(video, audio)

    return video, audio
