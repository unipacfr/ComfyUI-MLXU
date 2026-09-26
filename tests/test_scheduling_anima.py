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
"
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


def test_anima_normal_matches_comfy_reference():
    got = scheduling.calculate_sigmas("anima", "normal", 30)
    np.testing.assert_allclose(got, _NORMAL_30, atol=1e-5)


def test_anima_simple_matches_comfy_reference():
    got = scheduling.calculate_sigmas("anima", "simple", 20)
    np.testing.assert_allclose(got, _SIMPLE_20, atol=1e-5)
