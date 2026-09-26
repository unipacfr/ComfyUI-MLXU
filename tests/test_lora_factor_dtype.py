"""LoRA factors keep their file dtype; the forward residual runs in MLX's
natural promoted dtype (mlx-gen `adapters.rs` convention), and every
full-size materialization used by merge paths stays float32.

Numeric comparisons run on the CPU stream (canon "MLX GPU fp32 matmul is not
exact: parity tests run on the CPU stream"). Mutation evidence: restoring the
old `x.astype(a.dtype)` cast in `AdaptableLinear.__call__` fails
`test_f32_activations_bf16_factors_exact` (f32 x truncated to bf16) and
`test_f16_factors_bf16_activations_stay_finite` (bf16 x cast to f16 overflows);
restoring the bf16->f32 upcast at load fails `test_factors_keep_file_dtype`."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
lora_mod = load_node_module("lora")
ASDX_LoraLoader = lora_mod.ASDX_LoraLoader
AdaptableLinear = lora_mod.AdaptableLinear

IN, OUT, RANK = 32, 24, 4
KEY = "blocks.0.attn.wq.weight"


@pytest.fixture(autouse=True)
def _cpu_stream():
    with mx.stream(mx.cpu):
        yield


def _factors(dtype):
    mx.random.seed(1)
    a = (mx.random.normal((RANK, IN)) * 0.1).astype(dtype)
    b = (mx.random.normal((OUT, RANK)) * 0.1).astype(dtype)
    return a, b


def _write(tmp_path, dtype):
    a, b = _factors(dtype)
    path = tmp_path / f"lora_{dtype}.safetensors".replace("mlx.core.", "")
    mx.save_safetensors(str(path), {
        f"diffusion_model.{KEY[:-7]}.lora_A.weight": a,
        f"diffusion_model.{KEY[:-7]}.lora_B.weight": b,
        f"diffusion_model.{KEY[:-7]}.alpha": mx.array(2.0),
    })
    return path, a, b


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_factors_keep_file_dtype(tmp_path, dtype):
    path, a, b = _write(tmp_path, dtype)
    lora = ASDX_LoraLoader._load_lora_file(path)
    la, lb = lora.factors[KEY]
    assert la.dtype == dtype and lb.dtype == dtype
    assert mx.array_equal(la, a) and mx.array_equal(lb, b)
    assert lora.alpha == 2.0 and isinstance(lora.alpha, float)


def test_pt_file_still_loads(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "lora.pt"
    torch.save({f"{KEY[:-7]}.lora_A.weight": torch.ones(RANK, IN),
                f"{KEY[:-7]}.lora_B.weight": torch.ones(OUT, RANK)}, path)
    la, lb = ASDX_LoraLoader._load_lora_file(path).factors[KEY]
    assert la.dtype == mx.float32 and lb.shape == (OUT, RANK)


def _layer(dtype):
    mx.random.seed(2)
    lin = nn.Linear(IN, OUT)
    lin.update({"weight": lin.weight.astype(dtype), "bias": lin.bias.astype(dtype)})
    return AdaptableLinear.from_linear(lin)


def test_bf16_activations_bf16_factors_run_bf16(tmp_path):
    a, b = _factors(mx.bfloat16)
    layer = _layer(mx.bfloat16)
    layer._lora_factors.append((a, b, 0.75))
    x = mx.random.normal((3, IN)).astype(mx.bfloat16)
    y = layer(x)
    base = nn.Linear.__call__(layer, x)
    ref = base + (0.75 * ((x @ a.T) @ b.T)).astype(mx.bfloat16)
    assert y.dtype == mx.bfloat16
    assert mx.array_equal(y, ref)


def test_f32_activations_bf16_factors_exact():
    a, b = _factors(mx.bfloat16)
    layer = _layer(mx.float32)
    layer._lora_factors.append((a, b, 0.75))
    x = mx.random.normal((3, IN))
    y = layer(x)
    ref = nn.Linear.__call__(layer, x) + 0.75 * ((x @ a.astype(mx.float32).T) @ b.astype(mx.float32).T)
    assert y.dtype == mx.float32
    assert mx.array_equal(y, ref)


def test_f16_factors_bf16_activations_stay_finite():
    # bf16 x f16 promotes to float32 in MLX, so activations beyond f16's
    # 65504 range must not overflow the residual.
    a, b = _factors(mx.float16)
    layer = _layer(mx.bfloat16)
    layer._lora_factors.append((a, b, 1.0))
    x = mx.full((2, IN), 1e5, dtype=mx.bfloat16)
    y = layer(x)
    assert y.dtype == mx.bfloat16
    assert bool(mx.all(mx.isfinite(y)))


def test_delta_from_factors_returns_f32():
    a, b = _factors(mx.bfloat16)
    d = lora_mod._delta_from_factors(a, b)
    assert d.dtype == mx.float32
    assert mx.array_equal(d, b.astype(mx.float32) @ a.astype(mx.float32))
    a4 = a.reshape(RANK, IN // 4, 2, 2)
    b4 = b.reshape(OUT, RANK, 1, 1)
    assert lora_mod._delta_from_factors(a4, b4).dtype == mx.float32


def test_lokr_and_loha_materialize_f32():
    mx.random.seed(3)
    w1 = mx.random.normal((2, 2)).astype(mx.bfloat16)
    w2 = mx.random.normal((3, 4)).astype(mx.bfloat16)
    d = lora_mod._delta_from_lokr(w1, w2)
    assert d.dtype == mx.float32
    f1, f2 = w1.astype(mx.float32), w2.astype(mx.float32)
    ref = mx.concatenate([mx.concatenate([f1[i, j] * f2 for j in range(2)], axis=1)
                          for i in range(2)], axis=0)   # explicit kron(w1, w2)
    assert mx.array_equal(d, ref)
    p = [mx.random.normal(s).astype(mx.bfloat16) for s in ((OUT, RANK), (RANK, IN))] * 2
    d = lora_mod._delta_from_loha(*p, 0.5)
    assert d.dtype == mx.float32
    f = [t.astype(mx.float32) for t in p]
    assert mx.array_equal(d, ((f[0] @ f[1]) * (f[2] @ f[3])) * 0.5)


def test_merge_delta_bf16_raw_delta_matches_f32_path():
    # A raw `.diff` delta now stays bf16 from the file; the merge must still
    # scale it in float32 (the old path's precision) before narrowing.
    delta = (mx.random.normal((OUT, IN)) * 0.1).astype(mx.bfloat16)
    layer = _layer(mx.bfloat16)
    ref = layer.weight + (0.3 * delta.astype(mx.float32)).astype(mx.bfloat16)
    layer.merge_delta(delta, 0.3)
    assert mx.array_equal(layer.weight, ref)


def test_alpha_and_rank_follow_file_order(tmp_path):
    # The first `.alpha` and first factor rank set the file-level scale, so
    # the load must iterate in the file's tensor data order (what the old
    # `safetensors.torch.load_file` returned), never `mx.load`'s hash order.
    tensors = {}
    for i, (alpha, rank) in enumerate([(16.0, 32), (32.0, 8), (4.0, 16), (8.0, 4),
                                        (2.0, 64), (64.0, 2), (1.0, 1), (12.0, 12)]):
        stem = f"lora_unet_blocks_{i}_{'qkvmozxa'[i]}"
        tensors[f"{stem}.lora_down.weight"] = mx.zeros((rank, IN))
        tensors[f"{stem}.lora_up.weight"] = mx.zeros((OUT, rank))
        tensors[f"{stem}.alpha"] = mx.array(alpha)
    path = tmp_path / "multi.safetensors"
    mx.save_safetensors(str(path), tensors)
    hdr = lora_mod.read_safetensors_header(path).tensors
    file_order = sorted(hdr, key=lambda k: hdr[k].data_offsets)
    assert list(mx.load(str(path))) != file_order, "setup: hash order must differ from file order"
    alpha = float(tensors[next(k for k in file_order if k.endswith(".alpha"))])
    # kohya pairs are registered when their `.lora_up` key is reached.
    rank = tensors[next(k for k in file_order if k.endswith(".lora_up.weight"))].shape[1]
    lora = ASDX_LoraLoader._load_lora_file(path)
    assert (lora.alpha, lora.rank) == (alpha, rank)
    assert lora_mod.base_lora_scale(lora.alpha, lora.rank) == alpha / rank


def test_f16_activations_bf16_lokr_factors_promote_to_f32():
    # Krea2 regression: f16 activations x bf16 LoKr factors must promote to
    # float32 inside `_kron_matmul`, matching the low-rank pair path's rule,
    # not truncate x down to bf16 first (the old `x.astype(w1.dtype)` cast).
    mx.random.seed(4)
    w1 = (mx.random.normal((4, 4)) * 0.1).astype(mx.bfloat16)
    w2 = (mx.random.normal((6, IN // 4)) * 0.1).astype(mx.bfloat16)
    layer = _layer(mx.float16)
    layer._lokr_factors.append((w1, w2, 0.5))
    x = mx.random.normal((3, IN)).astype(mx.float16)
    y = layer(x)
    assert y.dtype == mx.float16

    base = nn.Linear.__call__(layer, x)
    # No-cast reference (what `__call__` must produce now).
    no_cast_ref = base + (0.5 * lora_mod._kron_matmul(x, w1, w2)).astype(mx.float16)
    assert mx.array_equal(y, no_cast_ref)

    # Mutation evidence: the old `x.astype(w1.dtype)` cast truncates x to
    # bf16 before the kron contraction, which changes the actual layer
    # output on these magnitudes -- so `y` must NOT match that reference.
    old_cast_ref = base + (0.5 * lora_mod._kron_matmul(x.astype(w1.dtype), w1, w2)).astype(mx.float16)
    assert not mx.array_equal(y, old_cast_ref)


def test_bf16_activations_bf16_lokr_factors_stay_bf16():
    mx.random.seed(5)
    w1 = (mx.random.normal((4, 4)) * 0.1).astype(mx.bfloat16)
    w2 = (mx.random.normal((6, IN // 4)) * 0.1).astype(mx.bfloat16)
    layer = _layer(mx.bfloat16)
    layer._lokr_factors.append((w1, w2, 0.5))
    x = mx.random.normal((3, IN)).astype(mx.bfloat16)
    y = layer(x)
    assert y.dtype == mx.bfloat16
    ref = nn.Linear.__call__(layer, x) + (0.5 * lora_mod._kron_matmul(x, w1, w2)).astype(mx.bfloat16)
    assert mx.array_equal(y, ref)


def test_lokr_factors_survive_file_overwrite(tmp_path):
    # Final-review fix: `mx.load` is lazy, so LoKr factors must be forced
    # into the eval at the end of `_load_lora_file` alongside deltas/factors
    # -- otherwise the file stays memory-mapped open past load, and
    # overwriting it in place before the first sampling step would surface
    # as garbage values instead of the load-time data.
    mx.random.seed(6)
    w1 = mx.random.normal((2, 2)).astype(mx.bfloat16)
    w2 = mx.random.normal((3, 4)).astype(mx.bfloat16)
    stem = "lora_unet_blocks_0_attn_wq"
    path = tmp_path / "lokr.safetensors"
    mx.save_safetensors(str(path), {f"{stem}.lokr_w1": w1, f"{stem}.lokr_w2": w2})

    lora = ASDX_LoraLoader._load_lora_file(path)
    key = next(iter(lora.lokr_factors))
    l1, l2 = lora.lokr_factors[key]
    before = (mx.array(l1), mx.array(l2))

    # Overwrite the file with different data; a still-lazy array would read
    # this back on its first real use.
    mx.save_safetensors(str(path), {
        f"{stem}.lokr_w1": mx.zeros((2, 2), dtype=mx.bfloat16),
        f"{stem}.lokr_w2": mx.zeros((3, 4), dtype=mx.bfloat16),
    })
    assert mx.array_equal(l1, before[0]) and mx.array_equal(l2, before[1])


def test_f64_alpha_falls_back_to_torch(tmp_path):
    # Some older trainers write an F64 `.alpha` scalar; `mx.load` raises an
    # opaque "[safetensor] unsupported dtype F64" on it, so any file with an
    # F64 tensor must fall back to the torch->numpy route instead.
    pytest.importorskip("torch")
    import safetensors.torch
    import torch

    a, b = _factors(mx.bfloat16)
    stem = KEY[:-len(".weight")]
    path = tmp_path / "f64_alpha.safetensors"
    safetensors.torch.save_file({
        f"diffusion_model.{stem}.lora_A.weight": torch.from_numpy(np.array(a.astype(mx.float32))).to(torch.bfloat16),
        f"diffusion_model.{stem}.lora_B.weight": torch.from_numpy(np.array(b.astype(mx.float32))).to(torch.bfloat16),
        f"diffusion_model.{stem}.alpha": torch.tensor(2.0, dtype=torch.float64),
    }, str(path))

    lora = ASDX_LoraLoader._load_lora_file(path)
    la, lb = lora.factors[KEY]
    assert lora.alpha == 2.0 and isinstance(lora.alpha, float)
    # The torch fallback route upcasts bf16 -> float32 before `.numpy()`
    # (numpy has no bf16), same as the unchanged .pt/.bin path.
    assert la.dtype == mx.float32 and lb.dtype == mx.float32
    assert mx.allclose(la, a, atol=1e-2) and mx.allclose(lb, b, atol=1e-2)


def test_lokr_scale_and_rebuild_single_rounding(tmp_path):
    # The baked scale (`w2 * lycoris_scale`) and the low-rank rebuild
    # (`a @ b`) must each round to bf16 exactly once, in float32, not twice
    # (once implicitly in a bf16 matmul, once when the scale is applied).
    mx.random.seed(7)
    rank = 3  # non-power-of-two, to make double-rounding detectable
    stem = "lora_unet_blocks_0_attn_wk"
    w1 = mx.random.normal((2, 2)).astype(mx.bfloat16)
    w2_a = (mx.random.normal((4, rank)) * 0.1).astype(mx.bfloat16)
    w2_b = (mx.random.normal((rank, 5)) * 0.1).astype(mx.bfloat16)
    alpha = 7.0  # alpha / rank is not exactly representable in bf16
    path = tmp_path / "lokr_lowrank.safetensors"
    mx.save_safetensors(str(path), {
        f"{stem}.lokr_w1": w1,
        f"{stem}.lokr_w2_a": w2_a,
        f"{stem}.lokr_w2_b": w2_b,
        f"{stem}.alpha": mx.array(alpha),
    })

    lora = ASDX_LoraLoader._load_lora_file(path)
    key = next(iter(lora.lokr_factors))
    _, stored_w2 = lora.lokr_factors[key]

    # Rebuild (a @ b) rounds to bf16 once (its own single rounding), then the
    # scale bake takes that bf16 value back through float32 and rounds once
    # more -- two independent single-rounded steps, matching what the fixed
    # `_lokr_factor_pair` (rebuild) and `_load_lora_file` (scale bake) do.
    rebuilt_bf16 = (w2_a.astype(mx.float32) @ w2_b.astype(mx.float32)).astype(mx.bfloat16)
    scale = alpha / rank
    expected = (rebuilt_bf16.astype(mx.float32) * scale).astype(mx.bfloat16)
    assert mx.array_equal(stored_w2, expected)

    # Mutation evidence: doing the scale multiply directly in bf16 (the old,
    # double-rounded path) must NOT match -- otherwise this test could not
    # tell the fix apart from the bug.
    old_bf16_path = (rebuilt_bf16 * mx.array(scale, dtype=mx.bfloat16))
    assert not mx.array_equal(stored_w2, old_bf16_path)


def test_f8_tensor_refused(tmp_path):
    # A real F8_E4M3 tensor, which `mx.load` would hand back as uint8 bytes.
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8")
    from safetensors.torch import save_file
    path = tmp_path / "f8.safetensors"
    save_file({f"{KEY[:-7]}.lora_A.weight": torch.ones(RANK, IN).to(torch.float8_e4m3fn),
               f"{KEY[:-7]}.lora_B.weight": torch.ones(OUT, RANK)}, str(path))
    with pytest.raises(RuntimeError, match="F8_E4M3"):
        ASDX_LoraLoader._load_lora_file(path)
