"""Event-triggered averages: one channel cut out around every event of a set (a detector's events
or recorded TTL pulses), aligned on the event's start, end or peak.

Returns the mean, its standard deviation and a sample of the single trials, so the viewer can draw
the raw trials overlaid with the average on top. Nothing is assumed about what the events mean; for
a stimulus delivered at a pulse edge, align on ``start`` (rising edge) or ``end`` (falling edge).
"""

from __future__ import annotations

import numpy as np

from .analysis import bandpass
from .csd import EDGE_PAD_S, _decimate_rows_minmax
from .io import TimeSeries

ALIGN = ("start", "end", "peak")
NORMS = ("none", "baseline", "baseline_z", "session_z")
NORM_UNITS = {"baseline_z": "z (baseline)", "session_z": "z (session)"}
SESSION_STRETCHES = 30
SESSION_STRETCH_S = 2.0


def session_stats(ts: TimeSeries, channel: str, band: tuple[float, float] | None = None) -> tuple[float, float]:
    """(mean, SD) of ``channel`` over the whole session, from stretches spread evenly across it (the
    whole channel when it is short), band-passed like the trials so the units match."""
    fs = float(ts.fs)
    filt = bool(band and (band[0] or band[1]))
    pad = int(round(EDGE_PAD_S * fs)) if filt else 0
    n_seg = int(SESSION_STRETCH_S * fs) + 2 * pad
    if ts.n <= n_seg * SESSION_STRETCHES:
        starts, n_seg = [0], ts.n
    else:
        starts = np.linspace(0, ts.n - n_seg, SESSION_STRETCHES).astype(int)
    parts = []
    for a in starts:
        y = ts.block(int(a), int(a) + n_seg, [channel])[:, 0]
        if filt:
            y = bandpass(y, fs, band[0] or None, band[1] or None)
        parts.append(y[pad:len(y) - pad] if pad and len(y) > 2 * pad else y)
    y = np.concatenate(parts)
    return float(y.mean()), float(y.std()) or 1.0


def triggered_average(ts: TimeSeries, result: dict, channel: str, pre_ms: float = 250.0, post_ms: float = 250.0,
                      align: str = "start", band: tuple[float, float] | None = None, baseline: bool | None = None,
                      max_trials: int = 100, max_average: int = 1000, max_points: int = 2000,
                      which: str = "accepted", norm: str = "baseline", base_ms: float = 100.0) -> dict:
    """``channel`` around each event of ``result``: mean, SD and up to ``max_trials`` single trials.

    ``band`` (low, high) band-passes each trial (0 / None on a side = open; None = as recorded).
    ``norm`` is applied to every trial before averaging, so the mean, the SD band and the
    overlaid trials share one unit:

    - ``"none"``: as recorded (µV);
    - ``"baseline"``: minus the trial's own mean over the ``base_ms`` before the event;
    - ``"baseline_z"``: that, divided by the trial's SD over the same ``base_ms`` (response in units
      of what the channel was doing just before each event);
    - ``"session_z"``: minus the channel's mean over the whole session, divided by its SD over the
      whole session (:func:`session_stats`): a response is large or small relative to everything
      the channel does, and comparable across channels and sessions.

    ``baseline=False`` is the older spelling of ``norm="none"``. At most ``max_average`` events, evenly spread over the set, enter the mean; events
    too close to the ends of the recording for a full window are skipped."""
    if channel not in ts.names:
        raise KeyError(f"unknown channel {channel!r}")
    if align not in ALIGN:
        raise ValueError(f"align must be one of {ALIGN}")
    if baseline is False:
        norm = "none"
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}")
    events = result.get("rejected", []) if which == "rejected" else result["events"]
    if not events:
        raise ValueError("no events in this set")
    fs = float(ts.fs)
    n_pre, n_post = int(round(pre_ms / 1000.0 * fs)), int(round(post_ms / 1000.0 * fs))
    filt = bool(band and (band[0] or band[1]))
    n_pad = int(round(EDGE_PAD_S * fs)) if filt else 0
    centres = np.array([e.get(align, e["peak"]) for e in events], dtype=float)
    pick = np.arange(centres.size) if centres.size <= max_average else np.unique(np.linspace(0, centres.size - 1, max_average).round().astype(int))
    keep_every = max(1, int(np.ceil(pick.size / max(max_trials, 1))))
    total = n_pre + n_post + 1
    n_base = int(min(max(round(base_ms / 1000.0 * fs), 2), n_pre)) if n_pre >= 2 else 0
    if norm in ("baseline", "baseline_z") and not n_base:
        raise ValueError("baseline normalisation needs time before the event (pre_ms > 0)")
    ref = session_stats(ts, channel, band) if norm == "session_z" else None
    acc, acc2, trials, trial_index, used = np.zeros(total), np.zeros(total), [], [], 0
    for k, idx in enumerate(pick):
        ic = int(np.searchsorted(ts.t, centres[idx], "left"))
        lo, hi = ic - n_pre - n_pad, ic + n_post + 1 + n_pad
        if lo < 0 or hi > ts.n:
            continue
        y = ts.block(lo, hi, [channel])[:, 0]
        if filt:
            y = bandpass(y, fs, band[0] or None, band[1] or None)
        if n_pad:
            y = y[n_pad:-n_pad]
        if norm in ("baseline", "baseline_z"):
            base = y[n_pre - n_base:n_pre]
            y = y - base.mean()
            if norm == "baseline_z":
                y = y / (float(base.std()) or 1.0)
        elif ref is not None:
            y = (y - ref[0]) / ref[1]
        acc += y
        acc2 += y * y
        used += 1
        if k % keep_every == 0 and len(trials) < max_trials:
            trials.append(y)
            trial_index.append(int(idx))
    if not used:
        raise ValueError("no event leaves room for the window inside the recording")
    mean = acc / used
    sd = np.sqrt(np.maximum(acc2 / used - mean ** 2, 0.0))
    t_ms = np.arange(-n_pre, n_post + 1) / fs * 1000.0
    t_d, rows = _decimate_rows_minmax(t_ms, np.vstack([mean, sd, *trials]), max_points)
    intervals = np.diff(np.sort(centres))
    return {"det": result["name"], "label": result["label"], "channel": channel, "unit": NORM_UNITS.get(norm, ts.unit(channel)),
            "align": align, "norm": norm, "base_ms": n_base / fs * 1000.0 if norm in ("baseline", "baseline_z") else None,
            "session_mean": ref[0] if ref else None, "session_sd": ref[1] if ref else None,
            "set": which, "n_events": len(events), "n_averaged": used, "n_trials": len(trials), "trial_index": trial_index,
            "pre_ms": pre_ms, "post_ms": post_ms, "band": list(band) if filt else None, "baseline": norm != "none",
            "median_interval_ms": float(np.median(intervals) * 1000.0) if intervals.size else None,
            "t_ms": np.round(t_d, 4).tolist(), "mean": np.round(rows[0], 3).tolist(), "sd": np.round(rows[1], 3).tolist(),
            "trials": np.round(rows[2:], 2).tolist()}
