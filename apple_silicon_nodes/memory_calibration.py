"""Two-tier memory-fit prediction: measured evidence, heuristic fallback.

Answers "how much unified memory will (family, quant_format, precision,
low_memory_mode) need to load?" for a pre-load gate. Two statuses are kept
strictly distinct and never blended into one confidence score:

  - "heuristic" -- derived from the checkpoint's on-disk size x a safety
    margin. Always available, the default for any key with no evidence.
  - "measured" -- derived from real `mx.get_peak_memory()` observations
    captured during actual runs, promoted only after MIN_CONSISTENT_RUNS
    mutually-consistent observations exist for that exact key. A single
    successful run is never enough (see `record_observation`/
    `scripts/promote_memory_evidence.py`).

This mirrors the doctrine in SceneWorks's `sceneworks-memory-adapter` crate:
"Backend binaries deliberately return `gated` until every required
measurement ... has actually executed. A successful model call is not
silently promoted into a complete calibration record." Cold start here means
every key begins "heuristic" -- that is the correct, honest starting state,
not a gap to paper over with an invented multiplier.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MIN_CONSISTENT_RUNS = 3          # a single success never promotes to "measured"
CONSISTENCY_TOLERANCE = 0.15     # accepted runs' max/min must stay within this of the median
OS_RESERVE_GB = 4.0              # margin reserved for the OS/ComfyUI, never offered to the model
_DEFAULT_MULTIPLIER = 2.3        # weights + activations + MLX overhead, rough starting point
_FAMILY_HEURISTIC_MULTIPLIER: dict[str, float] = {
    # Populated as real measurements justify a per-family override; until
    # then every family shares the default above.
    #
    # MiniMax H3 requantizes every big linear to MLX 4-bit at load (see
    # native/minimax_h3/quantized_linear.py), so resident size depends on the
    # parameter count, not on the source file's quant format. Measured on the
    # real checkpoints (2026-09-19, per-tensor eval loader): DiT peak 14.1GB
    # from a 20GB INT8-ConvRot safetensors, 13.4GB from the 13.9GB Q5_0 GGUF;
    # text encoder peak 16.8GB from the 26GB INT8 safetensors, 16.2GB from
    # the 18GB Q4_K_M GGUF. 1.0x + OS_RESERVE_GB covers every one of them.
    "minimax_h3_dit": 1.0,
    "minimax_h3_text_encoder": 1.0,
}

_MODULE_DIR = Path(__file__).parent
_EVIDENCE_PATH = _MODULE_DIR / "memory_evidence.json"
_STAGING_PATH = _MODULE_DIR / ".memory_evidence_staging.jsonl"

EvidenceStatus = Literal["measured", "heuristic"]


@dataclass(frozen=True)
class LoadShape:
    family: str
    quant_format: str
    precision: str
    low_memory_mode: bool
    file_size_bytes: int

    def key(self) -> str:
        return f"{self.family}:{self.precision}:{self.quant_format}:{self.low_memory_mode}"


@dataclass(frozen=True)
class MemoryEstimate:
    predicted_peak_bytes: int
    status: EvidenceStatus
    basis: str
    sample_count: int = 0

    @property
    def predicted_peak_gb(self) -> float:
        return self.predicted_peak_bytes / (1024 ** 3)


@dataclass(frozen=True)
class MemoryRun:
    captured_at: str
    peak_bytes: int
    disk_size_bytes: int
    host_memory_bytes: int


def _load_evidence_store() -> dict:
    if not _EVIDENCE_PATH.exists():
        return {"schema_version": 1, "entries": {}}
    with open(_EVIDENCE_PATH) as f:
        return json.load(f)


def _save_evidence_store(store: dict) -> None:
    """Atomic write (temp file + rename) so the versioned JSON is never left
    half-written if the process is interrupted mid-save."""
    tmp_path = _EVIDENCE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(store, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(_EVIDENCE_PATH)


def predict_peak_memory(shape: LoadShape) -> MemoryEstimate:
    """Look up a "measured" verdict for `shape`'s exact key; fall back to the
    disk-size heuristic when no promoted evidence exists yet. Never
    interpolates between dissimilar shapes -- an approximate-but-honest
    heuristic beats a false precision borrowed from a non-comparable entry.
    """
    store = _load_evidence_store()
    entry = store.get("entries", {}).get(shape.key())
    if entry is not None and entry.get("status") == "measured":
        return MemoryEstimate(
            predicted_peak_bytes=int(entry["peak_bytes"]),
            status="measured",
            basis=f"measured:n={entry['sample_count']}",
            sample_count=entry["sample_count"],
        )

    multiplier = _FAMILY_HEURISTIC_MULTIPLIER.get(shape.family, _DEFAULT_MULTIPLIER)
    predicted = int(shape.file_size_bytes * multiplier + OS_RESERVE_GB * (1024 ** 3))
    return MemoryEstimate(
        predicted_peak_bytes=predicted,
        status="heuristic",
        basis=f"disk_size_heuristic:x{multiplier}+{OS_RESERVE_GB}GB",
        sample_count=0,
    )


def available_unified_memory_bytes() -> tuple[int, int]:
    """Returns (total_bytes, estimated_available_bytes) for this macOS host.

    `hw.memsize` (total) is exact and never wrong. "Available" has no
    reliable definition on unified memory -- `vm_stat`'s free+purgeable pages
    undercount what is actually reclaimable, so treat the second value as an
    estimate only, never as a hard ceiling. On any read failure, degrade to
    (total, total - OS_RESERVE_GB) rather than raising -- a gate that can't
    read `sysctl`/`vm_stat` should still let loading proceed with a
    conservative estimate, not block outright.
    """
    try:
        total = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        ).stdout.strip())
    except Exception as e:
        print(f"[ASDX] memory_calibration: sysctl hw.memsize failed ({e}), gate degraded")
        return (0, 0)

    try:
        vm_stat_out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=True
        ).stdout
        page_size = 4096
        first_line = vm_stat_out.splitlines()[0]
        if "page size of" in first_line:
            page_size = int(first_line.split("page size of")[1].split()[0])

        pages: dict[str, int] = {}
        for line in vm_stat_out.splitlines()[1:]:
            if ":" not in line:
                continue
            label, _, value = line.partition(":")
            value = value.strip().rstrip(".")
            if value.isdigit():
                pages[label.strip()] = int(value)

        free_pages = pages.get("Pages free", 0) + pages.get("Pages purgeable", 0)
        available = free_pages * page_size
        return (total, available)
    except Exception as e:
        print(f"[ASDX] memory_calibration: vm_stat read failed ({e}), gate degraded "
              f"to total-minus-reserve")
        return (total, max(0, total - int(OS_RESERVE_GB * (1024 ** 3))))


def check_fits_or_warn(shape: LoadShape) -> MemoryEstimate:
    """Two-level threshold, because "available" memory on unified memory
    macOS is only ever an estimate -- and the heuristic side of `estimate`
    is an unvalidated starting multiplier (see `predict_peak_memory`), so a
    "heuristic" prediction overshooting the "available" estimate is TWO
    uncertain numbers disagreeing, not proof the load will fail:

      - Refuse (raise) only when predicted > total physical memory -- that
        ceiling is exact (`sysctl hw.memsize`), so a refusal here is always
        justified regardless of low_memory_mode.
      - Warn only, NEVER block, when predicted > estimated available but
        <= total. An earlier version of this gate also hard-refused past a
        20% overshoot here when low_memory_mode was off; real usage showed
        that combining an unvalidated heuristic multiplier with vm_stat's
        under-counted "available" (it misses reclaimable purgeable/cached
        pages) produced confident-looking refusals of loads that likely
        would have fit -- exactly the false-confidence failure mode the
        "heuristic vs measured" status distinction exists to prevent
        elsewhere in this module. Only a hard, exact ceiling gets to raise.
    """
    estimate = predict_peak_memory(shape)
    total_bytes, available_bytes = available_unified_memory_bytes()

    print(f"[ASDX] Memory estimate: {estimate.predicted_peak_gb:.1f}GB "
          f"({estimate.status}, {estimate.basis}) vs "
          f"{available_bytes / (1024**3):.1f}GB available / {total_bytes / (1024**3):.1f}GB total")

    if total_bytes and estimate.predicted_peak_bytes > total_bytes:
        raise ValueError(
            f"ASDX: refusing to load -- estimated peak "
            f"{estimate.predicted_peak_gb:.1f}GB ({estimate.status}) exceeds this "
            f"machine's total memory ({total_bytes / (1024**3):.1f}GB). "
            f"This checkpoint cannot fit regardless of low_memory_mode."
        )

    if available_bytes and estimate.predicted_peak_bytes > available_bytes:
        overshoot = (estimate.predicted_peak_bytes - available_bytes) / available_bytes
        print(f"[ASDX] Warning: estimated peak ({estimate.status}) may exceed "
              f"currently available memory ({available_bytes / (1024**3):.1f}GB) "
              f"by {overshoot*100:.0f}% -- proceeding "
              f"(low_memory_mode={shape.low_memory_mode}). This is an estimate, "
              f"not a guarantee; if the load actually OOMs, that's real signal "
              f"worth recording via ASDX_RECORD_MEMORY_EVIDENCE=1.")

    return estimate


def record_observation(shape: LoadShape, peak_bytes: int) -> None:
    """Append one raw observation to the local (git-ignored) staging file.

    Opt-in only, via ASDX_RECORD_MEMORY_EVIDENCE=1 -- capturing on every
    ordinary run would record noise from whatever else is competing for
    memory on a shared machine (the exact situation documented for the
    Krea2 OOM: 78% of memory already held by other processes at capture
    time). This never writes to the versioned `memory_evidence.json`
    directly; promoting staged observations into it is a deliberate step,
    see `scripts/promote_memory_evidence.py`.
    """
    if os.environ.get("ASDX_RECORD_MEMORY_EVIDENCE") != "1":
        return
    total_bytes, _ = available_unified_memory_bytes()
    run = MemoryRun(
        captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        peak_bytes=peak_bytes,
        disk_size_bytes=shape.file_size_bytes,
        host_memory_bytes=total_bytes,
    )
    with open(_STAGING_PATH, "a") as f:
        f.write(json.dumps({"key": shape.key(), "run": asdict(run)}) + "\n")
    print(f"[ASDX] Recorded memory observation for '{shape.key()}': "
          f"{peak_bytes / (1024**3):.1f}GB (staged, not yet promoted)")
