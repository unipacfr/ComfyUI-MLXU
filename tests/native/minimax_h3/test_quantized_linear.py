"""Tests for MiniMax H3's requantize-to-MLX-native-format helper."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from support.minimax_h3_module_loader import load_native_module

qlin = load_native_module("minimax_h3.quantized_linear")


def test_requantize_round_trip_is_reasonably_accurate():
    mx.random.seed(0)
    w = mx.random.normal((128, 64))  # last dim divisible by group_size=64
    w_q, scales, biases = qlin.requantize(w, group_size=64, bits=4)
    back = qlin.dequantize_for_check(w_q, scales, biases, group_size=64, bits=4)

    assert back.shape == w.shape
    # 4-bit affine quantization is lossy by design -- check it's in the
    # right ballpark (not, say, garbage from a wrong group_size/bits mixup)
    # rather than requiring near-exact recovery.
    rel_error = float(mx.mean(mx.abs(back - w)).item()) / float(mx.mean(mx.abs(w)).item())
    assert rel_error < 0.2


def test_requantize_produces_smaller_storage_than_dense():
    # 4-bit packs 8 values per uint32 -- the quantized weight array itself
    # should have far fewer elements than the dense one it replaces.
    w = mx.random.normal((256, 128))
    w_q, scales, biases = qlin.requantize(w, group_size=64, bits=4)
    assert w_q.size < w.size


def test_requantize_matches_mx_quantize_directly():
    # Not testing MLX's own quantize/dequantize correctness (that's MLX's
    # job) -- just that this thin wrapper doesn't reorder/misname arguments.
    w = mx.random.normal((64, 64))
    mine = qlin.requantize(w, group_size=32, bits=8)
    direct = mx.quantize(w, group_size=32, bits=8)
    for a, b in zip(mine, direct):
        assert np.array_equal(np.array(a), np.array(b))


def test_default_group_size_and_bits_are_reasonable_for_a_large_model():
    # 4-bit at group_size 64 is the standard mlx-lm-style choice for large
    # weight-only quantization -- pin these as an explicit regression check
    # rather than leaving the defaults free to drift silently.
    assert qlin.DEFAULT_GROUP_SIZE == 64
    assert qlin.DEFAULT_BITS == 4


def test_quantized_weights_load_into_nn_quantized_linear_and_match_dense():
    # Confirms the actual integration this module exists for: assign
    # requantize()'s output into an nn.QuantizedLinear's own attributes and
    # get a forward pass result close to the equivalent dense nn.Linear.
    import mlx.nn as nn

    in_dims, out_dims = 64, 32
    mx.random.seed(1)
    dense_weight = mx.random.normal((out_dims, in_dims))
    dense_bias = mx.random.normal((out_dims,))

    w_q, scales, biases = qlin.requantize(dense_weight, group_size=64, bits=4)

    qlayer = nn.QuantizedLinear(in_dims, out_dims, bias=True, group_size=64, bits=4)
    qlayer.weight = w_q
    qlayer.scales = scales
    qlayer.biases = biases
    qlayer.bias = dense_bias

    dense = nn.Linear(in_dims, out_dims, bias=True)
    dense.weight = dense_weight
    dense.bias = dense_bias

    x = mx.random.normal((3, in_dims))
    q_out = qlayer(x)
    dense_out = dense(x)

    rel_error = float(mx.mean(mx.abs(q_out - dense_out)).item()) / float(mx.mean(mx.abs(dense_out)).item())
    assert rel_error < 0.2
