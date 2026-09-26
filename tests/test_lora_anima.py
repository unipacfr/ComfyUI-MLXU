"""Anima LoRA: residual attach over the live module tree, strict routing.

Every numeric comparison runs on the CPU stream (canon "MLX GPU fp32 matmul
is not exact: parity tests run on the CPU stream") against a reference model
whose target `.weight` was replaced by `W + scale * B @ A` by hand, and the
comparison itself is proven by mutation (canon "A parity test must be proven
by mutation")."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
lora_mod = load_node_module("lora")
anima_mod = load_node_module("native.anima.model")
anima_config_mod = load_node_module("native.anima.config")

ASDX_LoraLoader = lora_mod.ASDX_LoraLoader
_CFG = anima_config_mod.AnimaConfig(dtype="float32", model_channels=256, num_blocks=2, num_heads=2)
_RANK = 4

TARGETS = [
    "blocks.0.self_attn.q_proj",
    "blocks.1.cross_attn.v_proj",
    "blocks.0.mlp.layer1",
    "blocks.0.adaln_modulation_self_attn.1",
    "blocks.1.adaln_modulation_mlp.2",
    "final_layer.adaln_modulation.2",
    "llm_adapter.blocks.0.self_attn.q_proj",
    "llm_adapter.blocks.1.mlp.0",
]


@pytest.fixture(autouse=True)
def _cpu_stream():
    with mx.stream(mx.cpu):
        yield


@pytest.fixture(scope="module")
def base():
    mx.random.seed(0)
    with mx.stream(mx.cpu):
        model = anima_mod.AnimaTransformer(_CFG)
        mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(1)
    return (
        mx.array(rng.standard_normal((1, 16, 4, 4)).astype(np.float32)),
        mx.array([0.5], dtype=mx.float32),
        mx.array(rng.standard_normal((1, 6, _CFG.adapter_dim)).astype(np.float32)),
        mx.array(rng.integers(0, 100, (1, 5)).astype(np.int32)),
    )


def _forward(model, inputs):
    x, t, qwen, ids = inputs
    out = model(x, t, model.encode_context(qwen, ids))
    mx.eval(out)
    return np.array(out)


def _weight(model, name):
    return dict(tree_flatten(model.parameters()))[f"{name}.weight"]


def _factors(model, name, seed):
    out_dim, in_dim = _weight(model, name).shape
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((_RANK, in_dim)) * 0.1).astype(np.float32)
    b = (rng.standard_normal((out_dim, _RANK)) * 0.1).astype(np.float32)
    return a, b


def _write(tmp_path, fname, tensors):
    path = tmp_path / fname
    save_file(tensors, str(path))
    return path


def _kohya(name, a, b, alpha):
    stem = "lora_unet_" + name.replace(".", "_")
    return {f"{stem}.lora_down.weight": a, f"{stem}.lora_up.weight": b,
            f"{stem}.alpha": np.array(alpha, dtype=np.float32)}


def _peft(name, a, b):
    return {f"diffusion_model.{name}.lora_A.weight": a, f"diffusion_model.{name}.lora_B.weight": b}


def _load(path, strength=1.0):
    lora = ASDX_LoraLoader._load_lora_file(path)
    lora.scale = lora_mod.base_lora_scale(lora.alpha, lora.rank) * strength
    return lora


def _reference(base, merges):
    """Fresh model with base's parameters, each `name` weight += scale*B@A."""
    ref = anima_mod.AnimaTransformer(_CFG)
    ref.update(base.parameters())
    flat = dict(tree_flatten(base.parameters()))
    updates = {}
    for name, a, b, scale in merges:
        key = f"{name}.weight"
        w = updates.get(key, flat[key])
        updates[key] = w + scale * (mx.array(b) @ mx.array(a))
    ref.update(tree_unflatten(list(updates.items())))
    return ref


def _attached(capsys):
    m = re.search(r"attached (\d+)/(\d+) adapters", capsys.readouterr().out)
    assert m, "no attach log line"
    return int(m.group(1)), int(m.group(2))


@pytest.mark.parametrize("dialect", ["kohya", "peft"])
@pytest.mark.parametrize("name", TARGETS)
def test_each_target_class_matches_merged_reference(base, inputs, tmp_path, capsys, name, dialect):
    a, b = _factors(base, name, seed=TARGETS.index(name))
    tensors = _kohya(name, a, b, alpha=2.0) if dialect == "kohya" else _peft(name, a, b)
    lora = _load(_write(tmp_path, "t.safetensors", tensors))
    scale = (2.0 / _RANK) if dialect == "kohya" else 1.0
    assert lora.scale == pytest.approx(scale)

    new = ASDX_LoraLoader._apply_lora_to_transformer(base, lora, _CFG)
    assert _attached(capsys) == (1, 1)
    np.testing.assert_allclose(
        _forward(new, inputs), _forward(_reference(base, [(name, a, b, scale)]), inputs), atol=1e-5
    )


def test_mutation_breaks_reference(base, inputs, tmp_path):
    name = "blocks.0.adaln_modulation_self_attn.1"
    a, b = _factors(base, name, seed=3)
    lora = _load(_write(tmp_path, "t.safetensors", _kohya(name, a, b, alpha=2.0)))
    new = ASDX_LoraLoader._apply_lora_to_transformer(base, lora, _CFG)
    ref = _reference(base, [(name, a, b, 0.5)])
    got, want = _forward(new, inputs), _forward(ref, inputs)
    np.testing.assert_allclose(got, want, atol=1e-5)
    ref.update(tree_unflatten([(f"{name}.weight", _weight(ref, name) * 1.001)]))
    assert not np.allclose(got, _forward(ref, inputs), atol=1e-5)
    # And the LoRA itself is visible: the base output differs.
    assert not np.allclose(got, _forward(base, inputs), atol=1e-5)


def test_base_model_untouched(base, tmp_path):
    before = [(k, np.array(v)) for k, v in tree_flatten(base.parameters())]
    tensors = {}
    for i, name in enumerate(TARGETS):
        tensors.update(_kohya(name, *_factors(base, name, seed=i), alpha=2.0))
    lora = _load(_write(tmp_path, "t.safetensors", tensors))
    new = ASDX_LoraLoader._apply_lora_to_transformer(base, lora, _CFG)
    assert new is not base
    after = tree_flatten(base.parameters())
    assert [k for k, _ in before] == [k for k, _ in after]
    for (k, v0), (_, v1) in zip(before, after):
        assert np.array_equal(v0, np.array(v1)), k
    assert not list(lora_mod._iter_adaptable_leaves(base))
    assert len(list(lora_mod._iter_adaptable_leaves(new))) == len(TARGETS)


def test_two_loras_stack_without_double_apply(base, inputs, tmp_path):
    name = "blocks.0.self_attn.q_proj"
    other = "blocks.1.adaln_modulation_mlp.2"
    a1, b1 = _factors(base, name, seed=10)
    a2, b2 = _factors(base, name, seed=11)
    a3, b3 = _factors(base, other, seed=12)
    lora1 = _load(_write(tmp_path, "l1.safetensors", _kohya(name, a1, b1, alpha=2.0)), strength=0.8)
    lora2 = _load(_write(tmp_path, "l2.safetensors",
                         {**_peft(name, a2, b2), **_peft(other, a3, b3)}), strength=1.3)

    one = ASDX_LoraLoader._apply_lora_to_transformer(base, lora1, _CFG)
    two = ASDX_LoraLoader._apply_lora_to_transformer(one, lora2, _CFG)
    s1 = 0.5 * 0.8
    ref_two = _reference(base, [(name, a1, b1, s1), (name, a2, b2, 1.3), (other, a3, b3, 1.3)])
    np.testing.assert_allclose(_forward(two, inputs), _forward(ref_two, inputs), atol=1e-5)
    # LoRA 1's own result is not mutated by stacking LoRA 2 on top of it.
    np.testing.assert_allclose(
        _forward(one, inputs), _forward(_reference(base, [(name, a1, b1, s1)]), inputs), atol=1e-5
    )


def test_schedule_rescale(base, inputs, tmp_path):
    name = "blocks.1.adaln_modulation_mlp.2"
    a, b = _factors(base, name, seed=20)
    lora = _load(_write(tmp_path, "t.safetensors", _kohya(name, a, b, alpha=2.0)))
    lora.scale = 0.3
    first = ASDX_LoraLoader._apply_lora_to_transformer(base, lora, _CFG)
    lora.scale = 0.45  # incremental, as sampler/core.py::_update_lora_schedule sets it
    second = ASDX_LoraLoader._apply_lora_to_transformer(first, lora, _CFG)
    assert second is first  # `_rescale_attached_lora` fast path, no re-attach
    np.testing.assert_allclose(
        _forward(second, inputs), _forward(_reference(base, [(name, a, b, 0.75)]), inputs), atol=1e-5
    )


def test_unrouted_key_raises(base, tmp_path):
    name = "blocks.0.self_attn.q_proj"
    a, b = _factors(base, name, seed=30)
    tensors = {**_kohya(name, a, b, alpha=2.0),
               **_kohya("blocks.0.nonexistent", a, b, alpha=2.0)}
    lora = _load(_write(tmp_path, "bogus.safetensors", tensors))
    with pytest.raises(RuntimeError, match=r"bogus.*1 .*lora_unet_blocks_0_nonexistent"):
        ASDX_LoraLoader._apply_lora_to_transformer(base, lora, _CFG)


def test_text_encoder_keys_are_not_unrouted(base, tmp_path, capsys):
    name = "blocks.0.self_attn.q_proj"
    a, b = _factors(base, name, seed=31)
    te_a = np.zeros((_RANK, 8), dtype=np.float32)
    te_b = np.zeros((8, _RANK), dtype=np.float32)
    tensors = {**_kohya(name, a, b, alpha=2.0),
               **_kohya("te_layers.0.q", te_a, te_b, alpha=2.0),  # lora_unet_te_... is NOT TE
               }
    lora = _load(_write(tmp_path, "t.safetensors", tensors))
    with pytest.raises(RuntimeError, match="unrouted"):
        ASDX_LoraLoader._apply_lora_to_transformer(base, lora, _CFG)

    tensors = {**_kohya(name, a, b, alpha=2.0),
               "lora_te_layers_0_q.lora_down.weight": te_a, "lora_te_layers_0_q.lora_up.weight": te_b,
               "text_encoders.qwen3_06b.transformer.model.layers.0.q.lora_A.weight": te_a,
               "text_encoders.qwen3_06b.transformer.model.layers.0.q.lora_B.weight": te_b}
    lora = _load(_write(tmp_path, "t2.safetensors", tensors))
    ASDX_LoraLoader._apply_lora_to_transformer(base, lora, _CFG)
    assert _attached(capsys) == (1, 1)


def test_dora_scale_refused(base, tmp_path):
    name = "blocks.0.self_attn.q_proj"
    a, b = _factors(base, name, seed=40)
    tensors = {**_kohya(name, a, b, alpha=2.0),
               f"lora_unet_blocks_0_self_attn_q_proj.dora_scale": np.ones((1,), dtype=np.float32)}
    path = _write(tmp_path, "dora.safetensors", tensors)
    with pytest.raises(RuntimeError, match="DoRA"):
        ASDX_LoraLoader._load_lora_file(path)


def _header_file(tmp_path, fname, keys):
    return _write(tmp_path, fname, {k: np.zeros((1, 1), dtype=np.float32) for k in keys})


def test_detect_lora_family_anima(tmp_path):
    cases = {
        "a": ["lora_unet_blocks_0_self_attn_output_proj.lora_down.weight",
              "lora_unet_blocks_0_mlp_layer1.lora_down.weight"],
        "b": ["lora_unet_blocks_3_adaln_modulation_self_attn_1.lora_down.weight"],
        "c": ["diffusion_model.blocks.0.adaln_modulation_mlp.2.lora_A.weight",
              "diffusion_model.blocks.0.self_attn.output_proj.lora_A.weight"],
        "adapter": ["diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight"],
    }
    for tag, keys in cases.items():
        assert lora_mod.detect_lora_family(_header_file(tmp_path, f"{tag}.safetensors", keys)).family == "anima", tag
    krea2 = _header_file(tmp_path, "k.safetensors", ["diffusion_model.blocks.0.attn.wq.lora_A.weight"])
    assert lora_mod.detect_lora_family(krea2).family == "krea2"


def test_foreign_lora_on_anima_refused(tmp_path):
    krea2 = _header_file(tmp_path, "k.safetensors", ["diffusion_model.blocks.0.attn.wq.lora_A.weight"])
    anima_model = {"capability": SimpleNamespace(family="anima"), "config": _CFG}
    with pytest.raises(ValueError, match="krea2 LoRA"):
        lora_mod._check_lora_compatibility(krea2, anima_model)
    anima = _header_file(tmp_path, "a.safetensors", ["lora_unet_blocks_0_self_attn_output_proj.lora_down.weight"])
    krea2_model = {"capability": SimpleNamespace(family="krea2"), "config": None}
    with pytest.raises(ValueError, match="anima LoRA"):
        lora_mod._check_lora_compatibility(anima, krea2_model)
    lora_mod._check_lora_compatibility(anima, anima_model)


def test_unsupported_loha_tucker_refused_on_anima(base, tmp_path):
    """A LoHa target in the Tucker/CP variant is dropped (never populated
    into `loha_factors`) by `_load_lora_file`, so the strict-routing
    `present - consumed` diff in `_apply_lora_residual_to_anima` can't see it
    on its own -- it must be refused via `lora.unsupported_loha` instead."""
    stem = "lora_unet_blocks_0_self_attn_q_proj"
    rank = 4
    rng = np.random.default_rng(50)
    tensors = {
        f"{stem}.hada_w1_a": rng.standard_normal((32, rank)).astype(np.float32),
        f"{stem}.hada_w1_b": rng.standard_normal((rank, 32)).astype(np.float32),
        f"{stem}.hada_w2_a": rng.standard_normal((32, rank)).astype(np.float32),
        f"{stem}.hada_w2_b": rng.standard_normal((rank, 32)).astype(np.float32),
        f"{stem}.hada_t1": rng.standard_normal((rank, rank, rank)).astype(np.float32),
    }
    path = _write(tmp_path, "loha_tucker.safetensors", tensors)
    lora = ASDX_LoraLoader._load_lora_file(path)
    assert lora.unsupported_loha == [stem]
    assert not lora.loha_factors

    with pytest.raises(RuntimeError, match="unsupported LoKr/LoHa"):
        ASDX_LoraLoader._apply_lora_to_transformer(base, lora, _CFG)
