"""Dentate spikes: large, brief positive LFP transients of the dentate gyrus.

Dentate spikes (DS) are 20-40 ms positive deflections generated in the hilus / granule layer when
entorhinal input arrives during immobility and slow-wave sleep -- the dentate counterpart of the
sharp-wave ripple, and a sign that the probe reaches the dentate gyrus. They are detected on the
wide-band LFP rather than in a narrow band, because their shape, not a rhythm, is what marks them:

    the LFP is filtered 1-200 Hz (Butterworth, order 4) and peaks exceeding six times the median
    absolute amplitude of that filtered signal are taken, keeping events at least 50 ms apart.

Which contact to use matters: the amplitude profile across depth is how DS are told from a sharp
wave (DS are *positive* and peak in the dentate, a sharp wave is negative and peaks in radiatum),
and how DS1 and DS2 are told apart (the depth of the maximum). With ``channel="auto"`` the contact
with the largest such peaks is used, and the amplitude of every event on every electrode is
averaged into ``depth_profile``, so the Events tab's depth panel and this profile together say
whether the detection really sits in the dentate.
"""

from __future__ import annotations

import numpy as np

from .analysis import Param, bandpass
from .detect import detector, motion_z
from .io import TimeSeries
from .swr import decimated

RATE_HZ = 1250.0          # the LFP rate this is written for; 1-200 Hz needs no more


def filtered_lfp(ts: TimeSeries, channels: list[str], low_hz: float, high_hz: float, order: int,
                 rate_hz: float = RATE_HZ, progress=None):
    """(t, X filtered 1-200 Hz, fs) for ``channels``, decimated first."""
    t, X, fsd = decimated(ts, channels, rate_hz, progress=progress)
    out = np.vstack([bandpass(np.asarray(row, dtype=float), fsd, low_hz or None, high_hz or None, order=int(order))
                     for row in X])
    return t, out, fsd


def threshold_of(y: np.ndarray, factor: float = 6.0) -> float:
    """``factor`` x the median absolute amplitude of the filtered LFP, the paper's threshold."""
    return float(factor * np.median(np.abs(y)))


def peaks_above(y: np.ndarray, t: np.ndarray, threshold: float, refractory_ms: float = 50.0,
                sign: int = 1) -> list[dict]:
    """Local maxima of ``sign * y`` over ``threshold``, at least ``refractory_ms`` apart (the
    largest of a close group wins). Each event spans where the signal is above half its peak."""
    v = y * sign
    above = v > threshold
    if not above.any():
        return []
    d = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, above.size)
    cand = []
    for a, b in zip(starts, ends):
        k = a + int(np.argmax(v[a:b]))
        half = v[k] / 2.0
        lo = k
        while lo > 0 and v[lo - 1] > half:
            lo -= 1
        hi = k
        while hi < v.size - 1 and v[hi + 1] > half:
            hi += 1
        cand.append({"start": float(t[lo]), "end": float(t[hi]), "peak": float(t[k]),
                     "score": float(v[k] / threshold), "amp_uV": float(v[k]),
                     "duration": float(t[hi] - t[lo])})
    cand.sort(key=lambda e: -e["score"])          # the largest of a close group wins
    kept: list[dict] = []
    gap = refractory_ms / 1000.0
    for e in cand:
        if all(abs(e["peak"] - k["peak"]) >= gap for k in kept):
            kept.append(e)
    kept.sort(key=lambda e: e["peak"])
    return kept


def dentate_features(ts, channel, stats, low_hz, high_hz, order, **_):
    y = bandpass(ts.channel(channel), float(ts.fs), low_hz or None, high_hz or None, order=int(order))
    return {"raw": ts.channel(channel), "LFP 1-200 Hz": y}


def dentate_thresholds(params: dict) -> dict:
    return {}                                     # the threshold is in µV and lives in the result's stats


@detector("Dentate spikes (wide-band positive peaks)", [
    Param("low_hz", "float", 1.0, min=0.0, max=20.0, step=0.5, help="LFP band"),
    Param("high_hz", "float", 200.0, min=50.0, max=600.0, step=10.0),
    Param("order", "int", 4, min=1, max=8, help="Butterworth order (scipy only)"),
    Param("factor", "float", 6.0, min=1.0, max=20.0, step=0.5,
          help="threshold = this many times the median absolute amplitude of the filtered LFP"),
    Param("refractory_ms", "float", 50.0, min=0.0, max=1000.0, step=5.0, help="events closer than this: the largest wins"),
    Param("max_ms", "float", 0.0, min=0.0, max=500.0, step=5.0, help="reject events wider than this at half height; 0 = no limit"),
    Param("polarity", "choice", "positive", choices=["positive", "negative"],
          help="dentate spikes are positive; the negative setting finds sharp waves instead"),
    Param("rate_hz", "float", RATE_HZ, min=500.0, max=5000.0, step=50.0, help="the signal is decimated to about this first"),
    Param("gate_movement", "bool", False, help="reject events during movement (needs aux channels)"),
    Param("move_z", "float", 2.0, min=0.0, max=20.0, step=0.1),
], features=dentate_features, thresholds=dentate_thresholds)
def dentate_spikes(ts: TimeSeries, channel: str, layout=None, progress=None, **p):
    """Dentate spikes: peaks of the 1-200 Hz LFP over six times its median absolute amplitude, at
    least 50 ms apart. See the module docstring.

    With ``channel="auto"`` the contact with the largest such peaks is used. Every event's
    amplitude on every electrode is averaged into ``depth_profile``: dentate spikes are positive
    and peak in the dentate, so a profile that peaks elsewhere, or that is flat, says the events
    are something else."""
    low_hz, high_hz, order = p["low_hz"], p["high_hz"], int(p["order"])
    sign = -1 if p.get("polarity") == "negative" else 1
    names = ts.headstage_channels() or list(ts.names)
    bad = set((layout or {}).get("bad") or [])
    good = [n for n in names if n not in bad] or names
    if progress:
        progress(0.1, f"filtering {len(good)} electrodes {low_hz}-{high_hz} Hz")
    t, X, fsd = filtered_lfp(ts, good, low_hz, high_hz, order, float(p["rate_hz"]), progress=progress)
    thresholds = np.array([threshold_of(row, p["factor"]) for row in X])
    if channel in ("", "auto", None) or channel not in good:
        if progress:
            progress(0.6, "choosing a channel")
        per_channel = [peaks_above(X[i], t, thresholds[i], p["refractory_ms"], sign) for i in range(len(good))]
        mean_amp = np.array([np.mean([e["amp_uV"] for e in ev]) if ev else 0.0 for ev in per_channel])
        channel = good[int(np.argmax(mean_amp))] if mean_amp.max() > 0 else good[int(np.argmax(np.percentile(sign * X, 99.9, axis=1)))]
    i = good.index(channel)
    if progress:
        progress(0.75, f"peaks on {channel}")
    thr = float(thresholds[i])
    kept = peaks_above(X[i], t, thr, p["refractory_ms"], sign)
    rejected = []
    if p.get("max_ms"):
        wide = [e for e in kept if e["duration"] * 1000.0 > p["max_ms"]]
        kept = [e for e in kept if e["duration"] * 1000.0 <= p["max_ms"]]
        rejected += [{**e, "reason": "duration"} for e in wide]
    if p.get("gate_movement"):
        mz = motion_z(ts, 1.0, 0.5)
        if mz is not None:
            mt, mv, _ = mz
            still = []
            for e in kept:
                m = float(np.max(np.interp([e["start"], e["peak"], e["end"]], mt, mv)))
                e["move_score"] = m
                (rejected if m > p["move_z"] else still).append({**e, "reason": "movement"} if m > p["move_z"] else e)
            kept = still
    profile = depth_profile(t, X, good, kept)
    for e in kept:
        e["max_channel"] = profile["max_channel_per_event"].get(round(e["peak"], 4), channel)
    rejected.sort(key=lambda e: e["peak"])
    extra = {"channel": channel, "rejected": rejected, "n_rejected": len(rejected),
             "threshold_uV": thr, "median_abs_uV": thr / max(p["factor"], 1e-9),
             "depth_profile": {k: v for k, v in profile.items() if k != "max_channel_per_event"},
             "scan": _scan(t, sign * X[i] / (thr or 1.0))}
    return kept, {"centre": 0.0, "scale": thr or 1.0, "robust": 0.0, "threshold_uV": thr}, extra


def depth_profile(t: np.ndarray, X: np.ndarray, names: list[str], events: list[dict]) -> dict:
    """The mean amplitude of the detected events on every electrode: where in depth they are
    largest, which is what says whether they come from the dentate."""
    if not events:
        return {"names": names, "mean_uV": [0.0] * len(names), "max_channel": None, "max_channel_per_event": {}}
    idx = np.clip(np.searchsorted(t, [e["peak"] for e in events]), 0, t.size - 1)
    amps = X[:, idx]                                   # (channels, events) at each peak
    per_event = {round(float(t[j]), 4): names[int(np.argmax(np.abs(amps[:, k])))] for k, j in enumerate(idx)}
    mean = amps.mean(axis=1)
    return {"names": names, "mean_uV": np.round(mean, 2).tolist(),
            "max_channel": names[int(np.argmax(np.abs(mean)))], "max_channel_per_event": per_event}


def _scan(t: np.ndarray, y: np.ndarray, points: int = 6000) -> dict:
    """The whole-session signal in units of the threshold, max-pooled for the SWR-style scan."""
    from .swr import scan_trace
    return scan_trace(t, y, np.zeros_like(y), points)
