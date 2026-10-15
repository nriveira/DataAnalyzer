"""Analysis time ranges: the parts of a recording that event analyses may use.

A recording can contain stretches where the headstage was unplugged, the cable was moving, or
acquisition misbehaved. The Events tab restricts every event set to the *included* ranges saved
here: events must lie entirely inside one of them. No ranges means the whole recording.

Saved per recording (and stream, like other results) in ``dataanalyzer/…/sections/sections.json``
as ``{"include": [[t0, t1], ...]}`` in seconds on the recording's clock, so scripts can use the
same selection (:func:`load`, :func:`contains`).

:func:`activity` summarises the headstage signal over the whole session from the cached overview
(median peak-to-peak across electrodes per bucket), which is where unplugged stretches show up:
flat when the inputs float to a rail or read zero, huge when a connector is loose.
:func:`suggest` proposes include ranges that avoid both.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import TimeSeries

ACTIVITY_POINTS = 2000


def sections_path(ts: TimeSeries) -> Path:
    from .detect import results_dir
    return results_dir(ts, "sections") / "sections.json"


def normalise(ranges, t0: float | None = None, t1: float | None = None) -> list[list[float]]:
    """Sorted, clipped to [t0, t1], overlapping or touching ranges merged, empty ones dropped."""
    out: list[list[float]] = []
    for a, b in sorted((float(min(r)), float(max(r))) for r in ranges or [] if len(r) == 2):
        if t0 is not None:
            a = max(a, t0)
        if t1 is not None:
            b = min(b, t1)
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def load(ts: TimeSeries) -> list[list[float]]:
    p = sections_path(ts)
    try:
        return normalise(json.loads(p.read_text()).get("include", []))
    except (OSError, ValueError, AttributeError):
        return []


def save(ts: TimeSeries, ranges) -> list[list[float]]:
    """Save the included ranges (normalised to the recording); an empty list deletes the file."""
    ranges = normalise(ranges, float(ts.t[0]), float(ts.t[-1])) if ts.n else []
    p = sections_path(ts)
    if not ranges:
        if p.exists():
            p.unlink()
        return []
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"include": ranges}, indent=1))
    return ranges


def subtract(ranges, cut, t0: float, t1: float) -> list[list[float]]:
    """``ranges`` minus the stretches ``cut``. With no ranges the whole recording is included, so
    the result is the recording minus ``cut``: excluding a stretch works from either start."""
    keep = normalise(ranges, t0, t1) or [[t0, t1]]
    for a, b in normalise(cut, t0, t1):
        out = []
        for lo, hi in keep:
            if b <= lo or a >= hi:
                out.append([lo, hi])
                continue
            if lo < a:
                out.append([lo, a])
            if b < hi:
                out.append([b, hi])
        keep = out
    return normalise(keep, t0, t1)


def contains(ranges, start: float, end: float | None = None) -> bool:
    """True when [start, end] lies inside one included range (always, when there are none)."""
    if not ranges:
        return True
    end = start if end is None else end
    return any(a <= start and end <= b for a, b in ranges)


def event_in(ranges, e: dict) -> bool:
    return contains(ranges, e["start"], e.get("end", e["start"]))


def activity(ts: TimeSeries, points: int = ACTIVITY_POINTS) -> dict:
    """Median peak-to-peak across electrodes per overview bucket, reduced to ``points`` bins as
    (low, high) pairs: what the headstage signal looked like over the whole session."""
    names = ts.headstage_channels() or list(ts.names)
    idx = [ts.index(n) for n in names]
    stride, centres, mins, maxs = ts.overview()
    amp = np.median(maxs[:, idx] - mins[:, idx], axis=1) if centres.size else np.empty(0)
    if not amp.size:
        return {"t": [], "lo": [], "hi": [], "median": 0.0, "bucket_s": 0.0}
    step = max(1, int(np.ceil(amp.size / points)))
    nb = amp.size // step
    cut = nb * step
    groups = amp[:cut].reshape(nb, step) if nb else amp[None, :]
    tc = centres[:cut].reshape(nb, step).mean(axis=1) if nb else centres[:1]
    return {"t": np.round(tc, 3).tolist(), "lo": np.round(groups.min(axis=1), 1).tolist(),
            "hi": np.round(groups.max(axis=1), 1).tolist(), "median": float(np.median(amp)),
            "bucket_s": float(stride / float(ts.fs)) if ts.fs else 0.0, "unit": ts.unit(names[0]) if names else ""}


def suggest(ts: TimeSeries, flat_frac: float = 0.2, loud_factor: float = 8.0, bridge_s: float = 1.0,
            min_s: float = 5.0, min_bad_s: float = 0.5) -> dict:
    """Include ranges that avoid stretches where the median peak-to-peak across electrodes is
    below ``flat_frac`` of its session median (flat: unplugged, floating to a rail) or above
    ``loud_factor`` times it (loose connector, saturation). Bad stretches closer than ``bridge_s``
    are joined, and only stretches lasting ``min_bad_s`` or more are excluded: a stimulation
    artefact or a single large transient is an event, not a bad stretch of recording. Included
    ranges shorter than ``min_s`` are dropped. Nothing is saved."""
    names = ts.headstage_channels() or list(ts.names)
    idx = [ts.index(n) for n in names]
    stride, centres, mins, maxs = ts.overview()
    if not centres.size:
        return {"include": [], "excluded_s": 0.0, "reasons": []}
    amp = np.median(maxs[:, idx] - mins[:, idx], axis=1)
    med = float(np.median(amp)) or 1.0
    bad = (amp < flat_frac * med) | (amp > loud_factor * med)
    half = 0.5 * stride / float(ts.fs)
    t0, t1 = float(ts.t[0]), float(ts.t[-1])
    runs, reasons = [], []
    i = 0
    while i < bad.size:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < bad.size and bad[j + 1]:
            j += 1
        a, b = float(centres[i]) - half, float(centres[j]) + half
        if runs and a - runs[-1][1] < bridge_s:
            runs[-1][1] = b
        else:
            runs.append([a, b])
        i = j + 1
    runs = [r for r in runs if r[1] - r[0] >= min_bad_s]
    for a, b in runs:
        seg = amp[(centres >= a) & (centres <= b)]
        kind = "flat" if seg.size and np.median(seg) < med else "very large"
        reasons.append({"start": max(a, t0), "end": min(b, t1), "kind": kind})
    include, cur = [], t0
    for a, b in runs:
        if a - cur >= min_s:
            include.append([cur, a])
        cur = max(cur, b)
    if t1 - cur >= min_s:
        include.append([cur, t1])
    include = normalise(include, t0, t1)
    kept = sum(b - a for a, b in include)
    return {"include": include, "excluded_s": float((t1 - t0) - kept), "reasons": reasons,
            "median_p2p": med, "flat_below": flat_frac * med, "loud_above": loud_factor * med}
