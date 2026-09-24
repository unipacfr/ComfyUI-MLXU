"""Generic ODE/SDE solver steps for the MLX-native sampler.

Each `step_*` function advances `x` by one sigma step given a `denoised`
estimate (`x - model_output * sigma`, the same formula for every model
family in this project -- flow-matching's CONST.calculate_denoised and
SDXL's EPS.calculate_denoised are both `model_input - model_output*sigma`,
verified against comfy/model_sampling.py). Multi-eval solvers (currently only
`dpmpp_2s_ancestral`) call back into the model via the `model_call` closure
each loop in `core.py` provides; multistep solvers (`dpmpp_2m`,
`dpmpp_2m_sde`, `deis`) carry history across steps in `state`.

Ported term-by-term from `comfy/k_diffusion/sampling.py` (real ComfyUI
install, not reconstructed from memory) -- see per-function docstrings for
exact source line references.

Flow-matching vs EPS/discrete: comfy dispatches `euler_ancestral` and
`dpmpp_2s_ancestral` to an "_RF" (rectified-flow) variant when the model's
`model_sampling` is `comfy.model_sampling.CONST`. This project has no CONST
class; the equivalent split is `is_flow_matching` (True for dev/schnell/
krea2*/zimage*/flux2, False for sdxl) -- the same partition `core.py::run()`
already uses to dispatch between denoising loops.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import mlx.core as mx

SAMPLER_NAMES = [
    "euler",
    "euler_a",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2s_ancestral",
    "ddim",
    "deis",
]

# Solvers whose skip heuristic (reuse last step's noise_pred) is only valid
# for TeaCache/SeaCache: single model call per step, no cross-step state.
STATELESS_SINGLE_EVAL_SAMPLERS = {"euler", "ddim"}

ModelCall = Callable[[mx.array, float], mx.array]  # (x, sigma) -> denoised


def _to_d(x: mx.array, sigma: float, denoised: mx.array) -> mx.array:
    """Converts a denoiser output to a Karras ODE derivative.

    Matches k_diffusion/sampling.py:63-65::to_d. For every family in this
    project, denoised = x - noise_pred*sigma, so `d` reduces exactly to
    `noise_pred` -- this is a reformulation of the existing Euler update,
    not a new computation.
    """
    return (x - denoised) / sigma


def _noise_like(x: mx.array, seed: int, step_index: int) -> mx.array:
    """Fresh standard-normal noise for ancestral/SDE steps.

    ComfyUI uses a `BrownianTreeNoiseSampler` (torchsde) for SDE solvers and
    a plain per-call `torch.randn` for ancestral ones -- MLX has no
    torchsde equivalent. This uses independent per-step Gaussian noise
    (k_diffusion/sampling.py:78-80::default_noise_sampler's CPU fallback
    shape), a documented, mathematically valid but less temporally-
    correlated simplification for the SDE solvers specifically.
    """
    # mx.random.key() only accepts [0, 2**64-1]; the node's seed input goes
    # up to 2**64-1 itself, so the raw derivation below can overflow past
    # that for large seeds -- wrap it back into range instead of crashing.
    derived = ((seed + 1) * 1_000_003 + step_index) % (2 ** 64)
    key = mx.random.key(derived)
    return mx.random.normal(x.shape, dtype=mx.float32, key=key).astype(x.dtype)


def _get_ancestral_step(sigma_from: float, sigma_to: float, eta: float = 1.0) -> tuple[float, float]:
    """Matches k_diffusion/sampling.py:68-75::get_ancestral_step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def step_euler(
    x: mx.array,
    sigma: float,
    sigma_next: float,
    denoised: mx.array,
    state: dict[str, Any],
    **_: Any,
) -> tuple[mx.array, dict[str, Any]]:
    """Euler method. Matches k_diffusion/sampling.py's Euler update
    (`x = x + to_d(x,sigma,denoised) * dt`); `ddim` aliases this directly
    (comfy/samplers.py:1393-1394 -- the two only differ in inpaint noise
    handling, outside this node's current inpaint path).
    """
    d = _to_d(x, sigma, denoised)
    return x + d * (sigma_next - sigma), state


def step_euler_a(
    x: mx.array,
    sigma: float,
    sigma_next: float,
    denoised: mx.array,
    state: dict[str, Any],
    *,
    seed: int,
    step_index: int,
    is_flow_matching: bool,
    eta: float = 1.0,
    s_noise: float = 1.0,
    **_: Any,
) -> tuple[mx.array, dict[str, Any]]:
    """Ancestral Euler. Flow-matching models use the RF (rectified-flow)
    variant (k_diffusion/sampling.py:240-266::sample_euler_ancestral_RF,
    sigma in [0,1], alpha=1-sigma); SDXL/EPS uses the standard variant
    (216-237::sample_euler_ancestral). Backbone (eta=0, no noise injection)
    verified against a real comfy run for both branches.
    """
    if is_flow_matching:
        if sigma_next == 0:
            return denoised, state
        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
        sigma_down = sigma_next * downstep_ratio
        alpha_ip1 = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        x_new = (sigma_down / sigma) * x + (1.0 - sigma_down / sigma) * denoised
        if eta > 0:
            renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
            x_new = (alpha_ip1 / alpha_down) * x_new + _noise_like(x, seed, step_index) * s_noise * renoise_coeff
        return x_new, state

    sigma_down, sigma_up = _get_ancestral_step(sigma, sigma_next, eta=eta)
    if sigma_down == 0:
        return denoised, state
    d = _to_d(x, sigma, denoised)
    x_new = x + d * (sigma_down - sigma)
    if sigma_next > 0:
        x_new = x_new + _noise_like(x, seed, step_index) * s_noise * sigma_up
    return x_new, state


def step_dpmpp_2m(
    x: mx.array,
    sigma: float,
    sigma_next: float,
    denoised: mx.array,
    state: dict[str, Any],
    **_: Any,
) -> tuple[mx.array, dict[str, Any]]:
    """DPM-Solver++(2M), deterministic multistep. Matches k_diffusion/
    sampling.py:795-818::sample_dpmpp_2m exactly (no RF/standard split --
    comfy's own dpmpp_2m has no CONST isinstance check, it works directly
    off sigma ratios regardless of model family). `sigma_fn(t)/sigma_fn(t')
    == sigma/sigma'` and `(-h).expm1() == sigma'/sigma - 1` by construction
    of `sigma_fn(t)=exp(-t)`, so this is written in plain sigma ratios
    rather than round-tripping through log/exp. Verified bit-exact against
    a real comfy `sample_dpmpp_2m` run on a synthetic deterministic model.
    """
    old_denoised = state.get("old_denoised")
    old_sigma = state.get("old_sigma")
    ratio = sigma_next / sigma if sigma_next > 0 else 0.0

    if old_denoised is None or sigma_next == 0:
        x_new = ratio * x - (ratio - 1.0) * denoised
    else:
        h = math.log(sigma / sigma_next)
        h_last = math.log(old_sigma / sigma)
        r = h_last / h
        denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
        x_new = ratio * x - (ratio - 1.0) * denoised_d

    return x_new, {"old_denoised": denoised, "old_sigma": sigma}


def _half_log_snr(sigma: float, is_flow_matching: bool) -> float:
    """Matches k_diffusion/sampling.py:152-157::sigma_to_half_log_snr.

    Flow-matching sigma=1.0 (this project's schedules start exactly there)
    is a singularity for `log((1-sigma)/sigma)`; comfy avoids it by nudging
    `sigmas[0]` via `model_sampling.percent_to_sigma(1e-4)`
    (k_diffusion/sampling.py:168-176::offset_first_sigma_for_snr) before the
    solver loop even starts. Clamping locally here to `1 - 1e-4` is a close
    numerical stand-in for that offset, not a byte-identical port of it.
    """
    if is_flow_matching:
        s = min(sigma, 1.0 - 1e-4)
        return math.log((1.0 - s) / s)
    return -math.log(sigma)



def step_dpmpp_2m_sde(
    x: mx.array,
    sigma: float,
    sigma_next: float,
    denoised: mx.array,
    state: dict[str, Any],
    *,
    seed: int,
    step_index: int,
    is_flow_matching: bool,
    eta: float = 1.0,
    s_noise: float = 1.0,
    solver_type: str = "midpoint",
    **_: Any,
) -> tuple[mx.array, dict[str, Any]]:
    """DPM-Solver++(2M) SDE. Matches k_diffusion/sampling.py:821-876::
    sample_dpmpp_2m_sde (`solver_type="midpoint"`, comfy's own default).
    Noise: independent per-step Gaussian rather than comfy's
    `BrownianTreeNoiseSampler` (torchsde has no MLX equivalent) -- see
    `_noise_like`'s docstring; not bit-reproducible against a comfy run,
    only the deterministic backbone (`eta=0`) is.
    """
    old_denoised = state.get("old_denoised")
    h_last = state.get("h_last")

    if sigma_next == 0:
        return denoised, {"old_denoised": denoised, "h_last": h_last}

    lambda_s = _half_log_snr(sigma, is_flow_matching)
    lambda_t = _half_log_snr(sigma_next, is_flow_matching)
    h = lambda_t - lambda_s
    h_eta = h * (eta + 1.0)
    alpha_t = sigma_next * math.exp(lambda_t)

    x_new = (sigma_next / sigma) * math.exp(-h * eta) * x + alpha_t * (-math.expm1(-h_eta)) * denoised

    if old_denoised is not None and h_last is not None:
        r = h_last / h
        if solver_type == "heun":
            x_new = x_new + alpha_t * ((-math.expm1(-h_eta)) / (-h_eta) + 1.0) * (1.0 / r) * (denoised - old_denoised)
        else:  # midpoint
            x_new = x_new + 0.5 * alpha_t * (-math.expm1(-h_eta)) * (1.0 / r) * (denoised - old_denoised)

    if eta > 0 and s_noise > 0:
        noise_scale = sigma_next * ((1.0 - math.exp(-2.0 * h * eta)) ** 0.5) * s_noise
        x_new = x_new + _noise_like(x, seed, step_index) * noise_scale

    return x_new, {"old_denoised": denoised, "h_last": h}


def step_dpmpp_2s_ancestral(
    x: mx.array,
    sigma: float,
    sigma_next: float,
    denoised: mx.array,
    state: dict[str, Any],
    *,
    seed: int,
    step_index: int,
    is_flow_matching: bool,
    model_call: ModelCall,
    eta: float = 1.0,
    s_noise: float = 1.0,
    **_: Any,
) -> tuple[mx.array, dict[str, Any]]:
    """Ancestral DPM-Solver++(2S), second-order, one extra model call per
    step. RF branch for flow-matching (k_diffusion/sampling.py:686-734::
    sample_dpmpp_2s_ancestral_RF), standard branch otherwise (648-682).
    Backbone (eta=0) verified against a real comfy run for both branches.
    """
    if is_flow_matching:
        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
        sigma_down = sigma_next * downstep_ratio
        alpha_ip1 = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down

        if sigma_next == 0:
            x_new = denoised
        else:
            if sigma >= 1.0:
                sigma_s = 0.9999
            else:
                t_i = math.log((1.0 - sigma) / sigma)
                t_down = math.log((1.0 - sigma_down) / sigma_down)
                s = t_i + 0.5 * (t_down - t_i)
                sigma_s = 1.0 / (math.exp(s) + 1.0)
            sigma_s_ratio = sigma_s / sigma
            u = sigma_s_ratio * x + (1.0 - sigma_s_ratio) * denoised
            denoised_2 = model_call(u, sigma_s)
            sigma_down_ratio = sigma_down / sigma
            x_new = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * denoised_2

        if sigma_next > 0 and eta > 0:
            renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
            x_new = (alpha_ip1 / alpha_down) * x_new + _noise_like(x, seed, step_index) * s_noise * renoise_coeff
        return x_new, state

    sigma_down, sigma_up = _get_ancestral_step(sigma, sigma_next, eta=eta)
    if sigma_down == 0:
        d = _to_d(x, sigma, denoised)
        x_new = x + d * (sigma_down - sigma)
    else:
        sigma_s_mid = (sigma * sigma_down) ** 0.5  # geometric mean: log-space midpoint (r=1/2)
        x_2 = (sigma_s_mid / sigma) * x - (sigma_s_mid / sigma - 1.0) * denoised
        denoised_2 = model_call(x_2, sigma_s_mid)
        x_new = (sigma_down / sigma) * x - (sigma_down / sigma - 1.0) * denoised_2
    if sigma_next > 0:
        x_new = x_new + _noise_like(x, seed, step_index) * s_noise * sigma_up
    return x_new, state


def _deis_coeffs_order1(t_cur: float, t_next: float, t_prev1: float) -> tuple[float, float]:
    """order=1 branch of k_diffusion/deis.py::get_deis_coeff_list (deis_mode='rhoab')."""
    coeff_cur = ((t_next - t_prev1) ** 2 - (t_cur - t_prev1) ** 2) / (2.0 * (t_cur - t_prev1))
    coeff_prev1 = (t_next - t_cur) ** 2 / (2.0 * (t_prev1 - t_cur))
    return coeff_cur, coeff_prev1


def _deis_def_integral_2(a: float, b: float, start: float, end: float, c: float) -> float:
    """k_diffusion/deis.py::get_def_intergral_2 (deis_mode='rhoab', order=2)."""
    coeff = (end ** 3 - start ** 3) / 3.0 - (end ** 2 - start ** 2) * (a + b) / 2.0 + (end - start) * a * b
    return coeff / ((c - a) * (c - b))


def step_deis(
    x: mx.array,
    sigma: float,
    sigma_next: float,
    denoised: mx.array,
    state: dict[str, Any],
    *,
    step_index: int,
    **_: Any,
) -> tuple[mx.array, dict[str, Any]]:
    """DEIS multistep, capped at 3 terms (current + 2 history points).

    Ported from `k_diffusion/deis.py::get_deis_coeff_list(deis_mode='rhoab')`
    (order-1 and order-2 branches only), verified numerically against a real
    comfy `get_deis_coeff_list` run. `deis_mode='tab'` (comfy's actual
    default) needs autograd through a numerical integral and was not ported;
    `'rhoab'` beyond 3 terms was also not ported -- executing real comfy's
    own `sample_deis(..., deis_mode='rhoab', max_order=3)` at `step_index>=
    max_order` raises `ValueError: too many values to unpack`, a genuine
    upstream bug in the coefficient-count vs. consumption contract between
    `get_deis_coeff_list` and `sample_deis`. Capping at 3 terms sidesteps
    that bug entirely rather than reproducing it.
    """
    d = _to_d(x, sigma, denoised)
    history: list[tuple[float, mx.array]] = state.get("history", [])

    order = min(step_index + 1, 3)
    if sigma_next <= 0:
        order = 1

    if order == 1:
        x_new = x + (sigma_next - sigma) * d
    elif order == 2:
        t_prev1, d_prev1 = history[-1]
        coeff_cur, coeff_prev1 = _deis_coeffs_order1(sigma, sigma_next, t_prev1)
        x_new = x + coeff_cur * d + coeff_prev1 * d_prev1
    else:
        t_prev1, d_prev1 = history[-1]
        t_prev2, d_prev2 = history[-2]
        coeff_cur = _deis_def_integral_2(t_prev1, t_prev2, sigma, sigma_next, sigma)
        coeff_prev1 = _deis_def_integral_2(sigma, t_prev2, sigma, sigma_next, t_prev1)
        coeff_prev2 = _deis_def_integral_2(sigma, t_prev1, sigma, sigma_next, t_prev2)
        x_new = x + coeff_cur * d + coeff_prev1 * d_prev1 + coeff_prev2 * d_prev2

    new_history = (history + [(sigma, d)])[-2:]
    return x_new, {"history": new_history}


_STEP_FNS: dict[str, Callable[..., tuple[mx.array, dict[str, Any]]]] = {
    "euler": step_euler,
    "ddim": step_euler,
    "euler_a": step_euler_a,
    "dpmpp_2m": step_dpmpp_2m,
    "dpmpp_2m_sde": step_dpmpp_2m_sde,
    "dpmpp_2s_ancestral": step_dpmpp_2s_ancestral,
    "deis": step_deis,
}


def step(
    sampler_name: str,
    *,
    x: mx.array,
    sigma: float,
    sigma_next: float,
    denoised: mx.array,
    state: dict[str, Any],
    seed: int,
    step_index: int,
    is_flow_matching: bool,
    model_call: ModelCall | None = None,
    **extra: Any,
) -> tuple[mx.array, dict[str, Any]]:
    """Advance `x` by one sigma step using `sampler_name`.

    `state` is a per-run dict (start as `{}`), threaded back in every call,
    used by multistep solvers to carry `old_denoised`/`h_last` etc.
    `model_call(x_at, sigma_at) -> denoised_at` is required only by solvers
    that evaluate the model more than once per step (`dpmpp_2s_ancestral`).
    `extra` passes through solver-specific knobs (`eta`, `s_noise`,
    `solver_type`) that default sensibly in each `step_*` and aren't part of
    this node's current public interface.
    """
    fn = _STEP_FNS.get(sampler_name)
    if fn is None:
        raise ValueError(f"ASDX: unknown sampler_name {sampler_name!r}")
    return fn(
        x=x,
        sigma=sigma,
        sigma_next=sigma_next,
        denoised=denoised,
        state=state,
        seed=seed,
        step_index=step_index,
        is_flow_matching=is_flow_matching,
        model_call=model_call,
        **extra,
    )
