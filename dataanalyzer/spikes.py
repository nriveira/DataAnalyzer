"""Spike detection infrastructure: threshold crossings on high-passed channels, saved beside the recording.

    python detect_spikes.py ExampleSession                   # all uV channels, defaults
    python detect_spikes.py ExampleSession --threshold 5 --band 300 6000 --channels CH20,CH13

Results go to ``<recording>/dataanalyzer/spikes/spikes.npz`` (times in s, channel
index, amplitude in µV) plus ``meta.json`` with the parameters, and the viewer
draws them as ticks on each lane. Processing is chunked so a 1 GB recording
never has to fit in memory as floats.

Backends: ``numpy`` (always available: Butterworth or FFT band-pass, MAD noise
estimate, negative-peak threshold, per-channel refractory) and
``spikeinterface`` (used only if installed; ``detect_peaks`` with the same
band and threshold). Sorting proper is out of scope here; the saved arrays are
meant to feed spikeinterface / Kilosort pipelines in the sibling projects.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis import bandpass
from .detect import results_dir
from .io import TimeSeries

try:
    from scipy import signal as _sig
except Exception:  # pragma: no cover
    _sig = None


def noise_mad(x: np.ndarray) -> float:
    """Robust noise SD (Quiroga et al. 2004): median(|x|) / 0.6745."""
    return float(np.median(np.abs(x)) / 0.6745) or 1.0


def detect_channel(y: np.ndarray, fs: float, threshold_sd: float, refractory_ms: float,
                   noise: float | None = None, polarity: str = "neg") -> tuple[np.ndarray, np.ndarray]:
    """Peak sample indices and amplitudes of threshold crossings on one filtered channel."""
    noise = noise or noise_mad(y)
    thr = threshold_sd * noise
    sig = -y if polarity == "neg" else (np.abs(y) if polarity == "both" else y)
    above = sig > thr
    if not above.any():
        return np.empty(0, dtype=np.int64), np.empty(0)
    d = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, above.size)
    peaks = np.array([s + int(np.argmax(sig[s:e])) for s, e in zip(starts, ends)], dtype=np.int64)
    if refractory_ms > 0 and peaks.size > 1:
        keep, last, ref = [], -np.inf, refractory_ms / 1000.0 * fs
        for p in peaks:
            if p - last >= ref:
                keep.append(p)
                last = p
        peaks = np.array(keep, dtype=np.int64)
    return peaks, y[peaks]


def detect_spikes(ts: TimeSeries, channels: list[str] | None = None, threshold_sd: float = 4.5,
                  band: tuple[float, float] = (300.0, 6000.0), refractory_ms: float = 1.0,
                  polarity: str = "neg", chunk_s: float = 20.0, progress: Callable[[float, str], None] | None = None,
                  backend: str = "numpy") -> dict:
    """Detect spikes on ``channels`` (default: all µV channels). Returns a result dict; see :func:`save`."""
    fs = float(ts.fs)
    if channels is None:
        channels = ts.headstage_channels() or list(ts.names)
    if backend == "spikeinterface":
        return _detect_spikeinterface(ts, channels, threshold_sd, band, refractory_ms, polarity, progress)

    started = time.perf_counter()
    n, k = ts.n, len(channels)
    chunk = int(chunk_s * fs)
    pad = int(0.05 * fs)                      # filter edge context, discarded
    times, chans, amps = [], [], []
    noise = np.zeros(k)
    # noise from the middle of the recording, so a quiet start does not set the threshold
    mid = max(0, n // 2 - chunk // 2)
    blk = ts.block(mid, min(n, mid + chunk), channels)
    for j in range(k):
        noise[j] = noise_mad(bandpass(blk[:, j], fs, band[0], band[1]))
    last_peak = np.full(k, -np.inf)
    n_chunks = math.ceil(n / chunk)
    for ci in range(n_chunks):
        lo, hi = ci * chunk, min(n, (ci + 1) * chunk)
        plo, phi = max(0, lo - pad), min(n, hi + pad)
        blk = ts.block(plo, phi, channels)
        for j in range(k):
            filt = bandpass(blk[:, j], fs, band[0], band[1])[lo - plo:hi - plo]
            pk, amp = detect_channel(filt, fs, threshold_sd, refractory_ms, noise[j], polarity)
            pk = pk + lo
            if pk.size and refractory_ms > 0:
                keep = (pk - last_peak[j]) >= refractory_ms / 1000.0 * fs
                pk, amp = pk[keep], amp[keep]
            if pk.size:
                last_peak[j] = pk[-1]
                times.append(ts.t[pk]); chans.append(np.full(pk.size, j, dtype=np.int32)); amps.append(amp)
        if progress:
            progress((ci + 1) / n_chunks, f"chunk {ci + 1}/{n_chunks}")
    t_all = np.concatenate(times) if times else np.empty(0)
    c_all = np.concatenate(chans) if chans else np.empty(0, dtype=np.int32)
    a_all = np.concatenate(amps) if amps else np.empty(0)
    order = np.argsort(t_all, kind="stable")
    return {
        "times": t_all[order], "channels": c_all[order], "amplitudes": a_all[order],
        "channel_names": channels, "noise_sd": noise.tolist(),
        "meta": {"backend": "numpy", "threshold_sd": threshold_sd, "band_hz": list(band), "refractory_ms": refractory_ms,
                 "polarity": polarity, "chunk_s": chunk_s, "fs": fs, "recording": ts.source,
                 "n_spikes": int(t_all.size), "elapsed_s": round(time.perf_counter() - started, 1),
                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }


def _detect_spikeinterface(ts, channels, threshold_sd, band, refractory_ms, polarity, progress) -> dict:
    try:
        import spikeinterface.full as si
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    except ImportError as e:
        raise ImportError("spikeinterface is not installed; use backend='numpy' or `uv pip install spikeinterface[full]`") from e
    if not ts.lazy:
        raise ValueError("the spikeinterface backend needs an Open Ephys recording folder")
    rec = si.read_openephys(str(Path(ts.source).parent.parent.parent), stream_name=None)
    rec = rec.channel_slice([rec.channel_ids[ts.index(c)] for c in channels])
    rec = si.bandpass_filter(rec, freq_min=band[0], freq_max=band[1])
    peaks = detect_peaks(rec, method="locally_exclusive", detect_threshold=threshold_sd,
                         peak_sign={"neg": "neg", "pos": "pos", "both": "both"}[polarity],
                         exclude_sweep_ms=refractory_ms)
    t = ts.t[peaks["sample_index"]]
    return {"times": t, "channels": peaks["channel_index"].astype(np.int32), "amplitudes": peaks["amplitude"],
            "channel_names": channels, "noise_sd": [],
            "meta": {"backend": "spikeinterface", "threshold_sd": threshold_sd, "band_hz": list(band),
                     "refractory_ms": refractory_ms, "polarity": polarity, "fs": ts.fs, "recording": ts.source,
                     "n_spikes": int(t.size), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}}


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def spikes_dir(ts: TimeSeries) -> Path:
    return results_dir(ts, "spikes")


def save(ts: TimeSeries, result: dict) -> Path:
    d = spikes_dir(ts)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "spikes.npz", times=result["times"], channels=result["channels"],
                        amplitudes=result["amplitudes"], channel_names=np.array(result["channel_names"]),
                        noise_sd=np.array(result["noise_sd"]))
    (d / "meta.json").write_text(json.dumps(result["meta"], indent=1))
    return d / "spikes.npz"


def load(ts: TimeSeries) -> dict | None:
    d = spikes_dir(ts)
    if not (d / "spikes.npz").exists():
        return None
    with np.load(d / "spikes.npz") as z:
        out = {"times": z["times"], "channels": z["channels"], "amplitudes": z["amplitudes"],
               "channel_names": [str(s) for s in z["channel_names"]], "noise_sd": z["noise_sd"].tolist()}
    try:
        out["meta"] = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        out["meta"] = {}
    return out


def overview(result: dict, duration: float, bins: int = 600) -> dict:
    """Per-channel counts and rates, and a channel x time raster of counts for the whole session."""
    names = result["channel_names"]
    t, ch = result["times"], result["channels"]
    counts = np.bincount(ch, minlength=len(names)) if t.size else np.zeros(len(names), dtype=int)
    t0 = float(t.min()) if t.size else 0.0
    t1 = float(t.max()) if t.size else max(duration, 1.0)
    edges = np.linspace(t0, t1 if t1 > t0 else t0 + 1.0, bins + 1)
    grid = np.zeros((len(names), bins), dtype=np.int32)
    if t.size:
        col = np.clip(np.searchsorted(edges, t, "right") - 1, 0, bins - 1)
        np.add.at(grid, (ch, col), 1)
    amp = {}
    for j, nm in enumerate(names):
        a = np.abs(result["amplitudes"][ch == j]) if t.size else np.empty(0)
        amp[nm] = [round(float(v), 1) for v in np.percentile(a, [25, 50, 75, 99])] if a.size else None
    return {"names": names, "counts": counts.tolist(),
            "rate_hz": [round(c / max(duration, 1e-9), 3) for c in counts.tolist()],
            "noise_sd": result.get("noise_sd") or [], "amp_percentiles": amp,
            "raster": {"t": np.round(edges[:-1] + np.diff(edges) / 2, 3).tolist(), "counts": grid.tolist(),
                       "bin_s": float(edges[1] - edges[0])},
            "meta": result.get("meta", {}), "n_spikes": int(t.size), "duration_s": float(duration)}


def waveforms(ts: TimeSeries, result: dict, channel: str, pre_ms: float = 1.0, post_ms: float = 2.0,
              max_n: int = 300, max_draw: int = 60, isi_bins: int = 60) -> dict:
    """What the detected spikes on one channel look like: the mean waveform with its SD, a sample of
    single waveforms, and the amplitude and inter-spike-interval distributions.

    Waveforms are cut from the recording and band-passed exactly as detection did (a wide window is
    filtered and then trimmed, so the filter's edges never reach the spike)."""
    names = result["channel_names"]
    if channel not in names:
        raise KeyError(f"{channel!r} has no detected spikes; pick one of {names}")
    j = names.index(channel)
    m = result["channels"] == j
    times, amps = result["times"][m], result["amplitudes"][m]
    meta = result.get("meta", {})
    band = tuple(meta.get("band_hz", (300.0, 6000.0)))
    fs = float(ts.fs)
    n_pre, n_post, pad = int(round(pre_ms / 1000 * fs)), int(round(post_ms / 1000 * fs)), int(round(0.01 * fs))
    pick = np.arange(times.size) if times.size <= max_n else np.unique(np.linspace(0, times.size - 1, max_n).round().astype(int))
    rows = []
    for idx in pick:
        i = int(np.searchsorted(ts.t, times[idx], "left"))
        lo, hi = i - n_pre - pad, i + n_post + 1 + pad
        if lo < 0 or hi > ts.n:
            continue
        y = ts.block(lo, hi, [channel])[:, 0]
        rows.append(bandpass(y, fs, band[0] or None, band[1] or None)[pad:len(y) - pad])
    W = np.vstack(rows) if rows else np.zeros((0, n_pre + n_post + 1))
    t_ms = (np.arange(-n_pre, n_post + 1) / fs * 1000.0)
    isi = np.diff(np.sort(times)) * 1000.0 if times.size > 1 else np.empty(0)
    isi_edges = np.linspace(0, float(np.percentile(isi, 99)) if isi.size else 100.0, isi_bins + 1)
    a = np.abs(amps)
    a_edges = np.linspace(0, float(np.percentile(a, 99.5)) if a.size else 1.0, 41)
    noise = (result.get("noise_sd") or [None] * len(names))[j] if result.get("noise_sd") else None
    return {"channel": channel, "n_spikes": int(times.size), "n_waveforms": int(W.shape[0]),
            "t_ms": np.round(t_ms, 4).tolist(),
            "mean": np.round(W.mean(axis=0), 2).tolist() if W.size else [],
            "sd": np.round(W.std(axis=0), 2).tolist() if W.size else [],
            "waveforms": np.round(W[:: max(1, W.shape[0] // max_draw)][:max_draw], 2).tolist() if W.size else [],
            "amp_hist": {"edges": np.round(a_edges, 2).tolist(), "counts": np.histogram(a, a_edges)[0].tolist() if a.size else []},
            "isi_hist": {"edges": np.round(isi_edges, 3).tolist(), "counts": np.histogram(isi, isi_edges)[0].tolist() if isi.size else [],
                         "violations": int((isi < float(meta.get("refractory_ms", 1.0))).sum()) if isi.size else 0},
            "noise_sd": noise, "threshold_uV": (noise * float(meta.get("threshold_sd", 0))) if noise else None,
            "band_hz": list(band), "refractory_ms": meta.get("refractory_ms")}


def summary(result: dict) -> dict:
    counts = np.bincount(result["channels"], minlength=len(result["channel_names"])) if result["times"].size else \
        np.zeros(len(result["channel_names"]), dtype=int)
    return {"n_spikes": int(result["times"].size), "channel_names": result["channel_names"],
            "counts": counts.tolist(), "meta": result.get("meta", {})}
