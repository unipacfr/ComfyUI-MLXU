"""Anima's sigma schedule: ModelSamplingDiscreteFlow (shift=3.0, multiplier=1.0),
same curve family as Z-Image (comfy/supported_models.py::Anima).

Reference values generated once against the real ComfyUI install (not importable
here -- see tests/support/comfyui_reference_loader.py's module docstring on why the
comfy stub and a genuine `comfy` import can't coexist in one process) via:

    cd /Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI && PYTHONPATH=/Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI \
    /Volumes/X10Pro/ComfyUI/MBP2026/ComfyUI/.venv/bin/python -c "
import comfy.model_sampling, comfy.samplers
ms = comfy.model_sampling.ModelSamplingDiscreteFlow()
ms.set_parameters(shift=3.0, multiplier=1.0)
print(comfy.samplers.calculate_sigmas(ms, 'normal', 30).tolist())
print(comfy.samplers.calculate_sigmas(ms, 'simple', 20).tolist())
print(comfy.samplers.calculate_sigmas(ms, 'karras', 20).tolist())
print(comfy.samplers.calculate_sigmas(ms, 'sgm_uniform', 20).tolist())
print(comfy.samplers.calculate_sigmas(ms, 'simple', 30).tolist())
"

(fix round 1, F1: the non-"normal" schedulers were verified against a
`_FlowModelSampling` built with num_steps=10000 -- ModelSamplingFlux's default
timesteps, not ModelSamplingDiscreteFlow's 1000 -- which silently used the wrong
sigma_min for karras/sgm_uniform/simple/beta. karras_20/sgm_uniform_20/simple_30
below catch that; simple_20 alone was passing by coincidence.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.comfy_stub import install_comfy_stubs, load_node_module

install_comfy_stubs()
scheduling = load_node_module("sampler.scheduling")

_NORMAL_30 = [1.0, 0.9882712960243225, 0.9759791493415833, 0.9630820155143738, 0.9495339393615723, 0.9352845549583435, 0.9202777743339539, 0.9044515490531921, 0.8877370357513428, 0.8700571656227112, 0.8513259887695312, 0.8314467072486877, 0.8103105425834656, 0.7877942323684692, 0.7637580037117004, 0.7380427718162537, 0.7104662656784058, 0.6808186769485474, 0.648857593536377, 0.6143014430999756, 0.576820433139801, 0.5360270142555237, 0.4914618730545044, 0.44257643818855286, 0.3887104094028473, 0.3290618658065796, 0.26264688372612, 0.18824484944343567, 0.10432276129722595, 0.008928571827709675, 0.0]

_SIMPLE_20 = [1.0, 0.9827585816383362, 0.964285671710968, 0.944444477558136, 0.9230769872665405, 0.8999999761581421, 0.8749999403953552, 0.8478260636329651, 0.8181818127632141, 0.7857143878936768, 0.75, 0.7105262875556946, 0.6666666865348816, 0.6176469922065735, 0.5625, 0.5, 0.4285714626312256, 0.3461538851261139, 0.25, 0.13636364042758942, 0.0]

_KARRAS_20 = [1.0, 0.8098191022872925, 0.651522696018219, 0.5205249190330505, 0.4127846956253052, 0.32475361227989197, 0.2533307671546936, 0.19581833481788635, 0.1498812586069107, 0.1135089322924614, 0.08498074859380722, 0.06283295899629593, 0.045829176902770996, 0.0329325832426548, 0.023280777037143707, 0.01616274192929268, 0.01099800132215023, 0.007317767478525639, 0.004747945349663496, 0.002994012786075473, 0.0]

_SGM_UNIFORM_20 = [1.0, 0.9828119874000549, 0.9644002318382263, 0.9446291923522949, 0.9233424663543701, 0.900359034538269, 0.8754674196243286, 0.848419725894928, 0.8189233541488647, 0.7866297364234924, 0.7511211037635803, 0.7118924260139465, 0.668326735496521, 0.6196626424789429, 0.564949631690979, 0.5029850602149963, 0.4322250783443451, 0.35065382719039917, 0.2555886507034302, 0.14337937533855438, 0.0]

_SIMPLE_30 = [1.0, 0.988752543926239, 0.9769874811172485, 0.964285671710968, 0.9513532519340515, 0.937781035900116, 0.9230769872665405, 0.9080505967140198, 0.8922204971313477, 0.8749999403953552, 0.8573263883590698, 0.8386242985725403, 0.8181818127632141, 0.7970947027206421, 0.7746615409851074, 0.75, 0.7244054079055786, 0.697002112865448, 0.6666666865348816, 0.6349481344223022, 0.6007193922996521, 0.5625, 0.5221642851829529, 0.47820165753364563, 0.4285714626312256, 0.37556222081184387, 0.3170347213745117, 0.25, 0.17724867165088654, 0.09550562500953674, 0.0]


def test_anima_normal_matches_comfy_reference():
    got = scheduling.calculate_sigmas("anima", "normal", 30)
    np.testing.assert_allclose(got, _NORMAL_30, atol=1e-5)


def test_anima_simple_matches_comfy_reference():
    got = scheduling.calculate_sigmas("anima", "simple", 20)
    np.testing.assert_allclose(got, _SIMPLE_20, atol=1e-5)


def test_anima_karras_matches_comfy_reference():
    got = scheduling.calculate_sigmas("anima", "karras", 20)
    np.testing.assert_allclose(got, _KARRAS_20, atol=1e-5)


def test_anima_sgm_uniform_matches_comfy_reference():
    got = scheduling.calculate_sigmas("anima", "sgm_uniform", 20)
    np.testing.assert_allclose(got, _SGM_UNIFORM_20, atol=1e-5)


def test_anima_simple_30_matches_comfy_reference():
    got = scheduling.calculate_sigmas("anima", "simple", 30)
    np.testing.assert_allclose(got, _SIMPLE_30, atol=1e-5)


def test_zimage_normal_matches_anima_reference():
    """Z-Image (multiplier 1.0, shift 3.0) shares the same ModelSamplingDiscreteFlow
    curve family as Anima -- pin it to the same reference list so a later split of
    the shared branch cannot silently revert Z-Image."""
    got = scheduling.calculate_sigmas("zimage", "normal", 30)
    np.testing.assert_allclose(got, _NORMAL_30, atol=1e-5)
