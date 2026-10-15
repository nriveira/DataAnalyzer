"""Spike sorting with MountainSort5, saved beside the recording.

Detection (:mod:`spikes`) says *when* something crossed threshold on a channel; sorting says
*which cell* each spike came from, by clustering the waveform shapes seen across neighbouring
contacts. This module drives MountainSort5 through spikeinterface, on the electrodes of the open
stream and with the probe geometry this project already knows, and stores the result in the same
place as everything else::

    <recording>/dataanalyzer/[<stream>/]sorting/units.npz   spike times, unit of each spike, templates
    <recording>/dataanalyzer/[<stream>/]sorting/meta.json   parameters, per-unit metrics, timing

The recording is wrapped without copying: the stream's ``continuous.dat`` is memory-mapped by
spikeinterface with the same gain, then band-passed and whitened for the sorter. Scheme 1 sorts a
recording in one pass and suits short ones; scheme 2 trains on a stretch and classifies the rest,
which is what long recordings need (``scheme="auto"`` picks by duration).

Per unit we keep what is needed to judge it: the template on every channel, the peak channel and
its depth, peak-to-peak amplitude, SNR against that channel's noise, firing rate, and the share of
inter-spike intervals below the refractory period.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .detect import results_dir
from .io import TimeSeries
from .spikes import noise_mad

SCHEME1_MAX_S = 300.0        # longer than this, scheme 2 (train on a stretch, then classify) is the sane choice
TEMPLATE_PRE_MS = 1.0
TEMPLATE_POST_MS = 2.0
MAX_SNIPPETS = 400           # per unit, for the template and its amplitude


def available() -> tuple[bool, str]:
    """(is MountainSort5 usable here, message)."""
    try:
        import mountainsort5  # noqa: F401
        import spikeinterface  # noqa: F401
    except ImportError as e:
        return False, f"{e}. Install with: uv pip install spikeinterface mountainsort5"
    return True, "mountainsort5 via spikeinterface"


def _recording(ts: TimeSeries, channels: list[str], locations: dict | None):
    """The open stream as a spikeinterface recording: the same file, memory-mapped, with this
    project's channel names, gains and probe positions, sliced to ``channels``."""
    import spikeinterface.full as si
    path = getattr(ts.data, "filename", None)
    if path is None:
        raise ValueError("sorting needs a memory-mapped recording (an Open Ephys stream)")
    scale = np.asarray(ts.scale if ts.scale is not None else np.ones(ts.n_channels), dtype=float)
    gains = scale.copy()
    rec = si.BinaryRecordingExtractor(file_paths=[str(path)], sampling_frequency=float(ts.fs),
                                      num_channels=ts.n_channels, dtype="int16",
                                      channel_ids=list(ts.names), gain_to_uV=gains,
                                      offset_to_uV=np.zeros(ts.n_channels))
    rec = si.ChannelSliceRecording(rec, channel_ids=list(channels))     # spikeinterface 0.104: no Recording.channel_slice
    xy = np.array([[0.0, float(i) * 25.0] for i in range(len(channels))])
    if locations:
        xy = np.array([locations.get(c, [0.0, float(i) * 25.0]) for i, c in enumerate(channels)], dtype=float)
    rec.set_channel_locations(xy)
    return rec


def contact_locations(layout: dict | None, channels: list[str]) -> dict:
    """{channel: [x_um, y_um]} from the probe layout, for the sorter's idea of which contacts
    are neighbours."""
    if not layout or not layout.get("contacts"):
        return {}
    return {c["name"]: [float(c["x_um"]), float(c["y_um"])] for c in layout["contacts"] if c["name"] in set(channels)}


def _templates(ts: TimeSeries, channels: list[str], times: np.ndarray, band: tuple[float, float],
               pre_ms: float = TEMPLATE_PRE_MS, post_ms: float = TEMPLATE_POST_MS, max_n: int = MAX_SNIPPETS):
    """(t_ms, template (n_channels, n_samples)) from the recording itself, band-passed like the sorter."""
    from .analysis import bandpass
    fs = float(ts.fs)
    n_pre, n_post, pad = int(round(pre_ms / 1000 * fs)), int(round(post_ms / 1000 * fs)), int(round(0.01 * fs))
    pick = times if times.size <= max_n else times[np.unique(np.linspace(0, times.size - 1, max_n).round().astype(int))]
    acc, used = None, 0
    for t0 in pick:
        i = int(np.searchsorted(ts.t, t0, "left"))
        lo, hi = i - n_pre - pad, i + n_post + 1 + pad
        if lo < 0 or hi > ts.n:
            continue
        blk = ts.block(lo, hi, channels).T
        blk = np.vstack([bandpass(r, fs, band[0] or None, band[1] or None) for r in blk])[:, pad:blk.shape[1] - pad]
        acc = blk if acc is None else acc + blk
        used += 1
    t_ms = np.arange(-n_pre, n_post + 1) / fs * 1000.0
    return t_ms, (acc / max(used, 1) if acc is not None else np.zeros((len(channels), t_ms.size))), used


def sort_mountainsort(ts: TimeSeries, layout: dict | None = None, channels: list[str] | None = None,
                      band: tuple[float, float] = (300.0, 6000.0), threshold_sd: float = 5.5,
                      scheme: str = "auto", whiten: bool = True, train_s: float = 300.0,
                      npca_per_channel: int = 3, snippet_mask_radius_um: float = 60.0,
                      progress: Callable[[float, str], None] | None = None) -> dict:
    """Sort the open stream with MountainSort5. Returns a result dict; see :func:`save`.

    ``channels`` defaults to the electrodes the probe places, minus the ones channel quality
    flagged (``layout["bad"]``). ``scheme`` is ``"auto"`` (1 for recordings up to five minutes,
    2 above), ``"1"`` or ``"2"``."""
    ok, msg = available()
    if not ok:
        raise ImportError(msg)
    import spikeinterface.full as si
    import mountainsort5 as ms5
    from mountainsort5.util import create_cached_recording

    started = time.perf_counter()
    bad = set((layout or {}).get("bad") or [])
    if channels is None:
        placed = [c["name"] for c in (layout or {}).get("contacts", [])] or ts.headstage_channels()
        channels = [c for c in placed if c not in bad] or ts.headstage_channels()
    if len(channels) < 2:
        raise ValueError("sorting needs at least two electrodes")
    which = "2" if (scheme == "2" or (scheme == "auto" and ts.duration > SCHEME1_MAX_S)) else "1"
    if progress:
        progress(0.05, f"preparing {len(channels)} channels for scheme {which}")

    rec = _recording(ts, channels, contact_locations(layout, channels))
    pre = si.bandpass_filter(rec, freq_min=band[0], freq_max=band[1])
    if whiten:
        pre = si.whiten(pre, dtype="float32")
    if progress:
        progress(0.15, f"MountainSort5 scheme {which}")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="dataanalyzer_ms5_") as tmp:
        if which == "1":
            sorting = ms5.sorting_scheme1(recording=pre, sorting_parameters=ms5.Scheme1SortingParameters(
                detect_threshold=threshold_sd, detect_sign=-1, npca_per_channel=npca_per_channel,
                snippet_mask_radius=snippet_mask_radius_um))
        else:
            cached = create_cached_recording(pre, folder=tmp)
            # scheme 2 names its parameters differently from scheme 1: the clustering ones are
            # phase1_*, and the two channel radii have no defaults
            sorting = ms5.sorting_scheme2(recording=cached, sorting_parameters=ms5.Scheme2SortingParameters(
                phase1_detect_threshold=threshold_sd, detect_threshold=threshold_sd, detect_sign=-1,
                phase1_detect_channel_radius=None, detect_channel_radius=None,
                phase1_npca_per_channel=npca_per_channel,
                snippet_mask_radius=snippet_mask_radius_um,
                training_duration_sec=min(train_s, max(ts.duration / 2, 60.0))))
        unit_ids = [int(u) for u in sorting.unit_ids]
        trains = {u: np.asarray(sorting.get_unit_spike_train(u), dtype=np.int64) for u in unit_ids}

    if progress:
        progress(0.7, f"{len(unit_ids)} units: templates and metrics")
    noise = np.array([noise_mad(_band(ts, c, band)) for c in channels])
    times_all, units_all, rows, templates = [], [], [], []
    t_ms = None
    for n_done, u in enumerate(unit_ids):
        t = ts.t[np.clip(trains[u], 0, ts.n - 1)]
        times_all.append(t)
        units_all.append(np.full(t.size, u, dtype=np.int32))
        t_ms, tpl, used = _templates(ts, channels, t, band)
        templates.append(tpl)
        p2p = np.ptp(tpl, axis=1)
        j = int(np.argmax(p2p))
        isi = np.diff(np.sort(t)) * 1000.0 if t.size > 1 else np.empty(0)
        rows.append({"unit": u, "n_spikes": int(t.size), "rate_hz": float(t.size / max(ts.duration, 1e-9)),
                     "peak_channel": channels[j], "peak_index": j, "amp_uV": float(p2p[j]),
                     "snr": float(p2p[j] / (noise[j] or 1.0)), "trough_uV": float(tpl[j].min()),
                     "isi_violations": int((isi < 1.0).sum()), "isi_violation_frac": float((isi < 1.0).mean()) if isi.size else 0.0,
                     "n_template": int(used)})
        if progress:
            progress(0.7 + 0.25 * (n_done + 1) / max(len(unit_ids), 1), f"unit {u}")
    order = np.argsort(np.concatenate(times_all)) if times_all else np.empty(0, dtype=int)
    return {
        "times": (np.concatenate(times_all)[order] if times_all else np.empty(0)),
        "units": (np.concatenate(units_all)[order] if units_all else np.empty(0, dtype=np.int32)),
        "unit_ids": unit_ids, "channel_names": channels,
        "templates": np.array(templates) if templates else np.zeros((0, len(channels), 1)),
        "template_t_ms": (t_ms if t_ms is not None else np.zeros(1)),
        "noise_sd": noise.tolist(),
        "meta": {"sorter": "mountainsort5", "scheme": which, "band_hz": list(band), "threshold_sd": threshold_sd,
                 "whiten": bool(whiten), "npca_per_channel": npca_per_channel,
                 "snippet_mask_radius_um": snippet_mask_radius_um, "fs": float(ts.fs), "recording": ts.source,
                 "stream": ts.stream, "n_units": len(unit_ids), "n_spikes": int(sum(t.size for t in times_all)),
                 "excluded_channels": sorted(bad), "units": rows, "duration_s": float(ts.duration),
                 "elapsed_s": round(time.perf_counter() - started, 1),
                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }


def _band(ts: TimeSeries, channel: str, band: tuple[float, float], seconds: float = 2.0) -> np.ndarray:
    """A short band-passed stretch from the middle of the recording, for the noise estimate."""
    from .analysis import bandpass
    n = int(min(ts.n, seconds * float(ts.fs)))
    lo = max(0, ts.n // 2 - n // 2)
    return bandpass(ts.block(lo, lo + n, [channel])[:, 0], float(ts.fs), band[0] or None, band[1] or None)


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def sorting_dir(ts: TimeSeries) -> Path:
    return results_dir(ts, "sorting")


def save(ts: TimeSeries, result: dict) -> Path:
    d = sorting_dir(ts)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "units.npz", times=result["times"], units=result["units"],
                        unit_ids=np.array(result["unit_ids"], dtype=np.int64),
                        channel_names=np.array(result["channel_names"]), templates=result["templates"],
                        template_t_ms=result["template_t_ms"], noise_sd=np.array(result["noise_sd"]))
    (d / "meta.json").write_text(json.dumps(result["meta"], indent=1))
    return d / "units.npz"


def load(ts: TimeSeries) -> dict | None:
    d = sorting_dir(ts)
    if not (d / "units.npz").exists():
        return None
    with np.load(d / "units.npz") as z:
        out = {"times": z["times"], "units": z["units"], "unit_ids": [int(u) for u in z["unit_ids"]],
               "channel_names": [str(s) for s in z["channel_names"]], "templates": z["templates"],
               "template_t_ms": z["template_t_ms"], "noise_sd": z["noise_sd"].tolist()}
    try:
        out["meta"] = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        out["meta"] = {}
    return out


def summary(result: dict, duration: float | None = None, bins: int = 600) -> dict:
    """Units with their metrics, their templates on the channels that carry them, and a
    unit x time raster, for the viewer."""
    meta = result.get("meta", {})
    dur = duration or meta.get("duration_s") or 1.0
    ids, t, u = result["unit_ids"], result["times"], result["units"]
    edges = np.linspace(float(t.min()) if t.size else 0.0, float(t.max()) if t.size else dur, bins + 1)
    grid = np.zeros((len(ids), bins), dtype=np.int32)
    if t.size:
        col = np.clip(np.searchsorted(edges, t, "right") - 1, 0, bins - 1)
        row = {v: i for i, v in enumerate(ids)}
        np.add.at(grid, (np.array([row[int(x)] for x in u]), col), 1)
    return {"unit_ids": ids, "channel_names": result["channel_names"], "units": meta.get("units", []),
            "template_t_ms": np.round(result["template_t_ms"], 4).tolist(),
            "noise_sd": result["noise_sd"], "meta": {k: v for k, v in meta.items() if k != "units"},
            "raster": {"t": np.round(edges[:-1] + np.diff(edges) / 2, 3).tolist(), "counts": grid.tolist(),
                       "bin_s": float(edges[1] - edges[0]) if bins else 0.0},
            "n_spikes": int(t.size), "duration_s": float(dur)}


def unit_detail(result: dict, unit: int, max_channels: int = 8) -> dict:
    """One unit: its template on the channels that carry it, its ISI histogram and rate over time."""
    ids = result["unit_ids"]
    if unit not in ids:
        raise KeyError(f"no unit {unit}; have {ids}")
    i = ids.index(unit)
    tpl = np.asarray(result["templates"][i])
    names = result["channel_names"]
    p2p = np.ptp(tpl, axis=1)
    keep = np.argsort(p2p)[::-1][:max_channels]
    keep = np.sort(keep)
    t = result["times"][result["units"] == unit]
    isi = np.diff(np.sort(t)) * 1000.0 if t.size > 1 else np.empty(0)
    edges = np.linspace(0, float(np.percentile(isi, 99)) if isi.size else 50.0, 61)
    row = next((r for r in result.get("meta", {}).get("units", []) if r.get("unit") == unit), {})
    return {"unit": unit, "n_spikes": int(t.size), "t_ms": np.round(result["template_t_ms"], 4).tolist(),
            "channels": [names[j] for j in keep], "templates": np.round(tpl[keep], 2).tolist(),
            "noise_sd": [float(result["noise_sd"][j]) for j in keep], "metrics": row,
            "isi_hist": {"edges": np.round(edges, 3).tolist(),
                         "counts": np.histogram(isi, edges)[0].tolist() if isi.size else [],
                         "violations": int((isi < 1.0).sum()) if isi.size else 0},
            "first_times": np.round(t[:2000], 4).tolist()}
