"""Stimulation artefact detection: find stimulus times from the recording itself, for procedures
that were not accompanied by a TTL pulse.

An electrical or optical stimulus leaves a fast transient on (nearly) every electrode at the same
sample, which is what separates it from spikes (one or a few contacts) and from slower, probe-wide
events. For every sample:

1. **Reference.** The median across electrodes is subtracted (``reference="median"``). Mains
   interference on an unreferenced recording is a sharp transient shared by every channel, 120
   times a second, and would otherwise swamp the detection; a stimulus artefact survives because
   its size differs from contact to contact. On a recording that was already common-average
   referenced this changes little.
2. **Slope.** |first difference| per channel, in robust SDs of that channel (median / MAD of the
   difference, estimated from stretches spread over the whole recording): artefacts are edges.
3. **Agreement across channels.** A sample is a hit when at least ``min_channels`` (a fraction)
   of the electrodes exceed ``threshold_sd`` at once. Its *score* is the median z across electrodes.
4. **Events.** Hits closer than ``merge_ms`` are one event (a biphasic pulse and its ringing),
   kept when the peak score reaches ``score_min``. Episodes longer than ``max_ms`` (movement,
   chewing, connector noise) are kept in the result as rejected with reason ``duration``.
   ``refractory_ms`` drops events that follow an accepted one too closely, and ``train_gap_ms``
   numbers pulses within trains (``train``, ``pulse``).

Each event carries ``channels_frac`` (the largest share of electrodes over threshold at one
sample; with the median reference the electrodes whose artefact is closest to the median size
lose it, so a real stimulus typically reads 0.7-1.0 rather than exactly 1), ``amp_uV`` (median
peak-to-peak across electrodes of the signal *as recorded*) and ``max_channel``. The result
is an ordinary event set: browse it in the Events tab, align windows on its ``start``, and use the
triggered average to look at the response.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .analysis import Param
from .detect import detector
from .io import TimeSeries

CHUNK_S = 10.0
STAT_STRETCHES = 8
STAT_STRETCH_S = 2.0


def _referenced(X: np.ndarray, reference: str) -> np.ndarray:
    return X - np.median(X, axis=1, keepdims=True) if reference == "median" and X.shape[1] > 2 else X


def slope_scales(ts: TimeSeries, names: list[str], reference: str) -> np.ndarray:
    """Robust SD of the first difference per channel, the median over stretches spread across
    the recording so that a stimulation epoch in one of them does not set the scale."""
    fs = float(ts.fs)
    n_seg = int(min(STAT_STRETCH_S * fs, ts.n))
    starts = np.linspace(0, max(ts.n - n_seg, 0), STAT_STRETCHES).astype(int)
    mads = []
    for a in starts:
        d = np.diff(_referenced(ts.block(int(a), int(a) + n_seg, names), reference), axis=0)
        mads.append(1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0))
    mad = np.median(np.vstack(mads), axis=0)
    mad[mad <= 0] = np.median(mad[mad > 0]) if (mad > 0).any() else 1.0
    return mad


def slope_z(ts: TimeSeries, lo: int, hi: int, names: list[str], mad: np.ndarray, reference: str):
    """(|dV/dt| z per channel for samples lo+1..hi-1, the referenced block)."""
    X = _referenced(ts.block(lo, hi, names), reference)
    return np.abs(np.diff(X, axis=0)) / mad, X


def _stats_mad(stats: dict, n: int) -> np.ndarray | None:
    try:
        return np.array([stats[f"mad_{j}"] for j in range(n)], dtype=float)
    except KeyError:
        return None


def stim_features(ts, channel, stats, threshold_sd=6.0, reference="median", **_):
    names = ts.headstage_channels() or list(ts.names)
    mad = _stats_mad(stats, len(names))
    if mad is None:
        mad = slope_scales(ts, names, reference)
    z, _ = slope_z(ts, 0, ts.n, names, mad, reference)
    z = np.vstack([z[:1], z])                                # back to one value per sample
    out = {"raw": ts.channel(channel)} if channel in ts.names else {}
    out["artefact score z"] = np.median(z, axis=1)
    out["channels over threshold %"] = 100.0 * (z > threshold_sd).mean(axis=1)
    return out


def stim_thresholds(params: dict) -> dict:
    return {"artefact score z": {"score_min": params["score_min"]},
            "channels over threshold %": {"min_channels": 100.0 * params["min_channels"]}}


@detector("Stimulation artefacts (all channels)", [
    Param("threshold_sd", "float", 6.0, min=2.0, max=50.0, step=0.5, help="a channel shows the artefact when its |slope| exceeds this many robust SDs"),
    Param("min_channels", "float", 0.5, min=0.1, max=1.0, step=0.05, help="fraction of electrodes that must show it at the same sample"),
    Param("score_min", "float", 10.0, min=2.0, max=200.0, step=1.0, help="keep events whose peak score (median slope z across electrodes) reaches this"),
    Param("merge_ms", "float", 20.0, min=0.0, max=500.0, step=1.0, help="hits closer than this are one event (pulse phases, ringing)"),
    Param("max_ms", "float", 100.0, min=0.0, max=5000.0, step=10.0, help="longer episodes are rejected as movement / noise; 0 = no limit"),
    Param("refractory_ms", "float", 0.0, min=0.0, max=10000.0, step=1.0, help="drop events closer than this to the previous accepted one; 0 = off"),
    Param("train_gap_ms", "float", 0.0, min=0.0, max=60000.0, step=10.0, help="pulses closer than this belong to one train (adds train and pulse numbers); 0 = off"),
    Param("reference", "choice", "median", choices=["median", "none"], help="subtract the median across electrodes first (needed on unreferenced recordings with mains interference)"),
], features=stim_features, thresholds=stim_thresholds)
def stim_artifacts(ts: TimeSeries, channel: str, threshold_sd, min_channels, score_min, merge_ms, max_ms, refractory_ms,
                   train_gap_ms, reference, layout=None, progress: Callable[[float, str], None] | None = None):
    """Stimulus times from the artefact they leave on every electrode: a fast edge present on at
    least `min_channels` of the electrodes at the same sample. The channel choice only sets what
    the Events tab shows ('auto' = the electrode with the largest artefact). See the module docstring."""
    names = ts.headstage_channels() or list(ts.names)
    if len(names) < 2:
        raise ValueError("artefact detection needs at least two electrodes to check agreement across channels")
    fs = float(ts.fs)
    if progress:
        progress(0.05, "estimating slope noise")
    mad = slope_scales(ts, names, reference)
    chunk = int(CHUNK_S * fs)
    hit_i, hit_score, hit_frac = [], [], []
    for lo in range(0, ts.n - 1, chunk):
        hi = min(lo + chunk + 1, ts.n)
        z, _ = slope_z(ts, lo, hi, names, mad, reference)
        frac = (z > threshold_sd).mean(axis=1)
        idx = np.flatnonzero(frac >= min_channels)
        if idx.size:
            hit_i.append(lo + 1 + idx)
            hit_score.append(np.median(z[idx], axis=1))
            hit_frac.append(frac[idx])
        if progress:
            progress(0.1 + 0.75 * hi / ts.n, "scanning")
    stats = {f"mad_{j}": float(m) for j, m in enumerate(mad)}
    extra = {"rejected": [], "n_rejected": 0, "n_candidates": 0, "n_rejected_duration": 0, "n_rejected_refractory": 0,
             "reference": reference, "n_channels": len(names), "channel_requested": channel, "n_trains": None}
    if not hit_i:
        extra["channel"] = channel if channel in ts.names else names[0]
        return [], stats, extra
    hit_i, hit_score, hit_frac = np.concatenate(hit_i), np.concatenate(hit_score), np.concatenate(hit_frac)
    gap = max(int(round(merge_ms / 1000.0 * fs)), 1)
    breaks = np.flatnonzero(np.diff(hit_i) > gap)
    firsts, lasts = np.r_[0, breaks + 1], np.r_[breaks, hit_i.size - 1]
    pad = max(int(round(0.001 * fs)), 1)
    events, rejected, amp_by_channel = [], [], np.zeros(len(names))
    for a, b in zip(firsts, lasts):
        k = a + int(np.argmax(hit_score[a:b + 1]))
        if hit_score[k] < score_min:
            continue
        i0, i1 = int(hit_i[a]), int(hit_i[b])
        p2p = np.ptp(ts.block(max(i0 - pad, 0), min(i1 + pad + 1, ts.n), names), axis=0)    # as recorded, not referenced
        e = {"start": float(ts.t[i0]), "end": float(ts.t[i1]), "peak": float(ts.t[int(hit_i[k])]), "score": float(hit_score[k]),
             "duration": float(ts.t[i1] - ts.t[i0]), "channels_frac": float(hit_frac[a:b + 1].max()),
             "amp_uV": float(np.median(p2p)), "max_channel": names[int(np.argmax(p2p))]}
        if max_ms > 0 and e["duration"] * 1000.0 > max_ms:
            e["reason"] = "duration"
        elif refractory_ms > 0 and events and e["start"] - events[-1]["start"] < refractory_ms / 1000.0:
            e["reason"] = "refractory"
        if "reason" in e:
            rejected.append(e)
        else:
            events.append(e)
            amp_by_channel += p2p
    if train_gap_ms > 0 and events:
        train, pulse = 0, 0
        for j, e in enumerate(events):
            if j and e["start"] - events[j - 1]["start"] > train_gap_ms / 1000.0:
                train, pulse = train + 1, 0
            e["train"], e["pulse"] = float(train), float(pulse)
            pulse += 1
        extra["n_trains"] = train + 1
    intervals = np.diff([e["start"] for e in events])
    extra.update({"rejected": rejected, "n_rejected": len(rejected), "n_candidates": len(events) + len(rejected),
                  "n_rejected_duration": sum(e["reason"] == "duration" for e in rejected),
                  "n_rejected_refractory": sum(e["reason"] == "refractory" for e in rejected),
                  "median_interval_ms": float(np.median(intervals) * 1000.0) if intervals.size else None,
                  "channel": channel if channel in ts.names else names[int(np.argmax(amp_by_channel))]})
    return events, stats, extra
