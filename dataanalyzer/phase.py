"""Where in an ongoing rhythm each spike falls: phase locking inside detected epochs.

The question this answers is "during the high-theta epochs, do these cells fire at a particular
part of the cycle?". It is written for theta (:func:`detect.theta` supplies the epochs) but nothing
here is theta-specific: any event set marks the stretches to use, and the band is a parameter, so
the same code answers it for any rhythm and any event set.

How it works: the reference channel is decimated to ~1 kHz (a cycle of a few Hz needs no more),
band-passed, and turned into an analytic signal, whose angle is the instantaneous phase --
**0 deg = the peak of the band-passed signal, 180 deg = its trough** (the polarity is the recorded
one; what a peak means anatomically depends on the electrode). Every spike inside one of the
epochs is given the phase of the sample nearest to it, and each group of spikes (a channel of the
threshold detection, or a sorted unit) gets a phase histogram and the circular statistics that say
whether it is concentrated: the mean angle, the mean vector length (0 = uniform, 1 = all spikes at
one phase) and the Rayleigh test against uniformity.

Averaging raw traces over whole epochs cancels the rhythm, because the epochs start at arbitrary
points of the cycle. :func:`cycle_times` finds every cycle's peak (or trough) inside the epochs and
:func:`cycle_average` averages on those, so the cycles add up instead of cancelling and a spike
rate can be read against the waveform it is locked to. The same choice sets what 0 deg means
(:func:`rotate`), so the histograms and the average agree.

Phases are only meaningful while the rhythm is actually present, which is what restricting to
epochs is for; ``amp`` keeps the band amplitude at each spike so a weak-cycle spike can be told
from a strong one.

:func:`activity` is the other half of the picture: the theta / delta power ratio over the whole
session, the usual proxy for how active the animal is -- high while it runs or explores, low
while it sits still, grooms or sleeps -- with the accelerometer beside it when the recording has
one, so the two can be read against each other.
"""

from __future__ import annotations

import numpy as np

from . import sections as _sec
from .analysis import bandpass
from .io import TimeSeries

ALIGN = ("peak", "trough")    # which part of the cycle counts as 0 deg and is averaged on
TARGET_HZ = 1000.0            # the reference signal is decimated to about this before filtering
DEFAULT_BAND = (5.0, 10.0)    # theta
BINS = 18                     # 20 deg phase bins
TRACE_POINTS = 4000           # per trace in an epoch view


def _fast_len(n: int) -> int:
    try:
        from scipy.fft import next_fast_len
        return int(next_fast_len(n))
    except ImportError:
        return 1 << int(np.ceil(np.log2(max(n, 1))))


def analytic(y: np.ndarray) -> np.ndarray:
    """The analytic signal of ``y`` (a Hilbert transform that does not need scipy)."""
    n = y.size
    if n == 0:
        return np.empty(0, dtype=complex)
    nf = _fast_len(n)
    h = np.zeros(nf)
    h[0] = 1.0
    if nf % 2 == 0:
        h[nf // 2] = 1.0
        h[1:nf // 2] = 2.0
    else:
        h[1:(nf + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(y, nf) * h)[:n]


def band_phase(ts: TimeSeries, channel: str, band=DEFAULT_BAND, target_hz: float = TARGET_HZ,
               progress=None) -> dict:
    """The reference channel over the whole recording, decimated, band-passed and turned into
    instantaneous phase: ``{t, lfp, filtered, phase (deg), amp, fs}``."""
    from .swr import decimated
    t, X, fsd = decimated(ts, [channel], target_hz, progress=progress)
    y = np.asarray(X[0], dtype=float)
    filt = bandpass(y, fsd, band[0] or None, band[1] or None)
    a = analytic(filt)
    return {"channel": channel, "band": [float(band[0]), float(band[1])], "fs": float(fsd),
            "t": np.asarray(t, dtype=float), "lfp": y, "filtered": filt,
            "phase": np.mod(np.degrees(np.angle(a)), 360.0), "amp": np.abs(a)}


def event_ranges(events, sections=None) -> np.ndarray:
    """(n, 2) start/end of the events that lie inside the selected time ranges, sorted and merged."""
    keep = [[float(e["start"]), float(e.get("end", e["start"]))] for e in events
            if sections is None or _sec.event_in(sections, e)]
    merged = _sec.normalise([r for r in keep if r[1] > r[0]])
    return np.array(merged, dtype=float).reshape(-1, 2)


def in_ranges(times: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Index of the range each time falls in, -1 outside. ``ranges`` must be sorted and disjoint."""
    times = np.asarray(times, dtype=float)
    if not len(ranges) or not times.size:
        return np.full(times.size, -1, dtype=int)
    i = np.searchsorted(ranges[:, 0], times, "right") - 1
    ok = (i >= 0) & (times <= ranges[np.clip(i, 0, None), 1])
    return np.where(ok, i, -1)


def phase_at(ph: dict, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(phase in degrees, band amplitude) of the sample nearest each time."""
    t = ph["t"]
    times = np.asarray(times, dtype=float)
    if not t.size or not times.size:
        return np.empty(0), np.empty(0)
    j = np.clip(np.searchsorted(t, times), 1, t.size - 1)
    j = np.where(times - t[j - 1] <= t[j] - times, j - 1, j)
    return ph["phase"][j], ph["amp"][j]


def circular_stats(deg: np.ndarray) -> dict:
    """Mean angle, mean vector length and the Rayleigh test of ``deg`` (degrees).

    ``mvl`` is 0 for spikes spread evenly over the cycle and 1 for spikes all at one phase;
    ``p`` is the chance of seeing an ``mvl`` this large from a uniform distribution."""
    deg = np.asarray(deg, dtype=float)
    n = int(deg.size)
    if n == 0:
        return {"n": 0, "mean_deg": None, "mvl": 0.0, "rayleigh_z": 0.0, "p": 1.0}
    r = np.exp(1j * np.radians(deg)).mean()
    mvl = float(abs(r))
    R = n * mvl
    p = float(np.exp(np.sqrt(1.0 + 4.0 * n + 4.0 * (n * n - R * R)) - (1.0 + 2.0 * n)))
    return {"n": n, "mean_deg": float(np.mod(np.degrees(np.angle(r)), 360.0)), "mvl": mvl,
            "rayleigh_z": float(R * R / n), "p": min(1.0, p)}


def histogram(deg: np.ndarray, bins: int = BINS) -> list[int]:
    return np.histogram(np.asarray(deg, dtype=float), bins=bins, range=(0.0, 360.0))[0].tolist()


def spike_source(ts: TimeSeries, source: str = "spikes") -> dict:
    """The spikes to look at: ``{times, group (index per spike), names, labels, kind}`` from the
    threshold detection (one group per channel) or from the sorting (one group per unit)."""
    if source == "units":
        from . import sorting
        r = sorting.load(ts)
        if not r:
            raise FileNotFoundError("no sorted units for this recording yet — sort it in the Spikes tab")
        ids = [int(u) for u in r["unit_ids"]]
        at = {u: i for i, u in enumerate(ids)}
        rows = {int(x.get("unit", -1)): x for x in (r.get("meta", {}).get("units") or [])}
        return {"times": np.asarray(r["times"], dtype=float),
                "group": np.array([at[int(u)] for u in r["units"]], dtype=int),
                "names": [f"unit {u}" for u in ids],
                "labels": [f"unit {u} · {rows.get(u, {}).get('peak_channel', '?')}" for u in ids],
                "kind": "units"}
    from . import spikes
    r = spikes.load(ts)
    if not r:
        raise FileNotFoundError("no spikes detected for this recording yet — run the detection in the Spikes tab")
    names = list(r["channel_names"])
    return {"times": np.asarray(r["times"], dtype=float), "group": np.asarray(r["channels"], dtype=int),
            "names": names, "labels": names, "kind": "spikes"}


def phase_table(ph: dict, ranges: np.ndarray, src: dict, bins: int = BINS, min_spikes: int = 1) -> dict:
    """Phase histogram and circular statistics per group, over the spikes inside ``ranges``."""
    times, group = src["times"], src["group"]
    which = in_ranges(times, ranges)
    inside = which >= 0
    deg, amp = phase_at(ph, times[inside])
    g = group[inside]
    total_s = float(np.sum(ranges[:, 1] - ranges[:, 0])) if len(ranges) else 0.0
    groups = []
    for i, name in enumerate(src["names"]):
        d = deg[g == i]
        st = circular_stats(d)
        if st["n"] < min_spikes:
            continue
        groups.append({"index": i, "name": name, "label": src["labels"][i], **st,
                       "counts": histogram(d, bins), "rate_hz": st["n"] / total_s if total_s else 0.0,
                       "amp_median": float(np.median(amp[g == i])) if st["n"] else 0.0,
                       "n_all": int((group == i).sum())})
    edges = np.linspace(0.0, 360.0, bins + 1)
    return {"groups": groups, "edges": edges.tolist(), "bins": bins, "n_in_epochs": int(inside.sum()),
            "n_spikes": int(times.size), "epochs_s": total_s, "n_epochs": int(len(ranges)),
            "kind": src["kind"], "channel": ph["channel"], "band": ph["band"], "fs": ph["fs"]}


def epoch_view(ph: dict, ranges: np.ndarray, index: int, src: dict, pad_s: float = 0.25,
               points: int = TRACE_POINTS) -> dict:
    """One epoch: the reference channel, its band-passed version and phase, and the spikes of every
    group inside it (with the phase of each), for drawing."""
    if not len(ranges):
        raise IndexError("no epochs")
    index = int(np.clip(index, 0, len(ranges) - 1))
    t0, t1 = float(ranges[index, 0]), float(ranges[index, 1])
    t = ph["t"]
    a, b = np.searchsorted(t, [t0 - pad_s, t1 + pad_s])
    a, b = int(max(a, 0)), int(min(b, t.size))
    step = max(1, int(np.ceil((b - a) / max(points, 1))))
    sl = slice(a, b, step)
    times, group = src["times"], src["group"]
    lo, hi = np.searchsorted(times, [t0, t1])
    deg, amp = phase_at(ph, times[lo:hi])
    return {"index": index, "n": int(len(ranges)), "t0": t0, "t1": t1,
            "t": np.round(t[sl], 4).tolist(), "lfp": np.round(ph["lfp"][sl], 2).tolist(),
            "filtered": np.round(ph["filtered"][sl], 2).tolist(),
            "phase": np.round(ph["phase"][sl], 1).tolist(),
            "spikes": [{"group": int(gi), "t": np.round(times[lo:hi][group[lo:hi] == gi], 4).tolist(),
                        "phase": np.round(deg[group[lo:hi] == gi], 1).tolist()}
                       for gi in np.unique(group[lo:hi])],
            "unit": "", "channel": ph["channel"], "band": ph["band"]}


def activity(ts: TimeSeries, channel: str, theta=DEFAULT_BAND, delta=(1.0, 4.0), smooth_s: float = 2.0,
             points: int = 3000, with_motion: bool = True) -> dict:
    """The theta / delta power ratio over the whole recording, as an activity level.

    Both bands are measured on one decimated pass over ``channel`` (envelope squared, smoothed
    over ``smooth_s``), and the ratio is reduced to ``points`` bins by their **mean** -- an
    activity level is what the animal was doing over that stretch, not the largest thing that
    happened in it. High theta with little delta is the running / exploring state; the reverse is
    sitting still, grooming or sleeping, which is when sharp-wave ripples happen.

    ``motion`` is the accelerometer over the same bins (robust z, from :func:`detect.motion_signal`)
    when the recording has aux channels, so a theta state can be checked against actual movement.
    """
    from .analysis import bandpass, envelope, moving_average
    from .detect import motion_signal, zscore
    from .swr import decimated
    t, X, fsd = decimated(ts, [channel], TARGET_HZ)
    y = np.asarray(X[0], dtype=float)
    win = max(1, int(round(smooth_s * fsd)))
    power = lambda b: moving_average(envelope(bandpass(y, fsd, b[0], b[1]), fsd) ** 2, win)
    th, de = power(theta), power(delta)
    ratio = th / np.maximum(de, np.percentile(de, 1) or 1e-9)
    step = max(1, int(np.ceil(ratio.size / max(points, 1))))
    nb = ratio.size // step
    if nb == 0:
        return {"t": [], "ratio": [], "theta": [], "delta": [], "bin_s": 0.0, "median": 0.0, "motion": None}
    cut = nb * step
    mean_of = lambda v: np.round(v[:cut].reshape(nb, step).mean(axis=1), 3)
    tc = t[:cut].reshape(nb, step).mean(axis=1)
    out = {"channel": channel, "theta_band": list(theta), "delta_band": list(delta), "smooth_s": float(smooth_s),
           "t": np.round(tc, 3).tolist(), "ratio": mean_of(ratio).tolist(),
           "theta": mean_of(np.sqrt(th)).tolist(), "delta": mean_of(np.sqrt(de)).tolist(),
           "bin_s": float(step / fsd), "median": float(np.median(ratio)), "motion": None}
    if with_motion:
        m = motion_signal(ts)
        if m is not None:
            mz, _ = zscore(m[1], robust=True)
            out["motion"] = np.round(np.interp(tc, m[0], mz), 2).tolist()
    return out


def rotate(ph: dict, align: str = "peak") -> dict:
    """``ph`` with 0 deg at the peak (the default) or at the trough of the band-passed signal."""
    if align not in ALIGN:
        raise ValueError(f"align must be one of {ALIGN}")
    if align == "peak":
        return ph
    return {**ph, "phase": np.mod(ph["phase"] - 180.0, 360.0), "align": align}


def cycle_times(ph: dict, ranges: np.ndarray, align: str = "peak") -> np.ndarray:
    """The time of every cycle's peak (or trough) that falls inside ``ranges``.

    A cycle is marked where the instantaneous phase passes the aligning angle, interpolated between
    the two samples either side, so the markers do not jitter by a sample. ``ph`` may already have
    been turned by :func:`rotate`; its own ``align`` says which angle is which, so asking for peaks
    gives peaks either way."""
    target = 0.0 if align == ph.get("align", "peak") else 180.0
    d = np.mod(ph["phase"] - target + 180.0, 360.0) - 180.0      # signed distance to the angle, rising through 0
    up = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    if not up.size:
        return np.empty(0)
    t, frac = ph["t"], d[up] / (d[up] - d[up + 1])
    times = t[up] + frac * (t[up + 1] - t[up])
    return times[in_ranges(times, ranges) >= 0]


def cycle_average(ph: dict, ranges: np.ndarray, src: dict | None = None, align: str = "peak",
                  pre_ms: float | None = None, post_ms: float | None = None, max_cycles: int = 4000,
                  bins: int = 40) -> dict:
    """The reference channel averaged on every cycle ``align`` point inside the epochs, with each
    group's spike rate on the same axis.

    The window defaults to three quarters of a cycle either side (one and a half cycles), from the
    centre of the band. ``mean``/``sd`` are the unfiltered signal, ``band`` the band-passed one;
    spike rates are spikes per second per cycle, so they can be compared between groups."""
    period_ms = 1000.0 / max((ph["band"][0] + ph["band"][1]) / 2.0, 1e-9)
    pre = period_ms * 0.75 if pre_ms is None else float(pre_ms)
    post = period_ms * 0.75 if post_ms is None else float(post_ms)
    fs, t = ph["fs"], ph["t"]
    n_pre, n_post = int(round(pre / 1000.0 * fs)), int(round(post / 1000.0 * fs))
    all_cycles = cycle_times(ph, ranges, align)
    cyc = all_cycles
    if cyc.size > max_cycles:                                     # evenly spread over the session, not the first ones
        cyc = cyc[np.unique(np.linspace(0, cyc.size - 1, max_cycles).round().astype(int))]
    t_ms = np.arange(-n_pre, n_post + 1) / fs * 1000.0
    out = {"t_ms": np.round(t_ms, 3).tolist(), "align": align, "n_cycles": int(all_cycles.size),
           "n_averaged": 0, "pre_ms": pre, "post_ms": post, "period_ms": period_ms,
           "channel": ph["channel"], "band": ph["band"], "mean": [], "sd": [], "band_mean": [], "groups": []}
    if not cyc.size:
        return out
    idx = np.searchsorted(t, cyc)
    ok = (idx - n_pre >= 0) & (idx + n_post + 1 <= t.size)
    idx = idx[ok]
    if not idx.size:
        return out
    take = idx[:, None] + np.arange(-n_pre, n_post + 1)[None, :]
    seg = ph["lfp"][take]
    seg = seg - seg.mean(axis=1, keepdims=True)                   # each cycle on its own baseline
    band = ph["filtered"][take]
    out["n_averaged"] = int(idx.size)
    out["mean"] = np.round(seg.mean(axis=0), 3).tolist()
    out["sd"] = np.round(seg.std(axis=0), 3).tolist()
    out["band_mean"] = np.round(band.mean(axis=0), 3).tolist()
    if src is None:
        return out
    # every spike of a group, as an offset to the nearest cycle markers (a window can span two)
    edges = np.linspace(-pre, post, bins + 1)
    bin_s = (edges[1] - edges[0]) / 1000.0
    used = t[idx]
    times, group = src["times"], src["group"]
    keep = in_ranges(times, ranges) >= 0
    times, group = times[keep], group[keep]
    j = np.clip(np.searchsorted(used, times) - 1, 0, max(used.size - 1, 0))
    offs, gs = [], []
    for k in (0, 1):
        jj = np.clip(j + k, 0, used.size - 1)
        off = (times - used[jj]) * 1000.0
        inside = (off >= -pre) & (off <= post)
        offs.append(off[inside]); gs.append(group[inside])
    off = np.concatenate(offs) if offs else np.empty(0)
    gg = np.concatenate(gs) if gs else np.empty(0, dtype=int)
    for i, name in enumerate(src["names"]):
        counts = np.histogram(off[gg == i], edges)[0]
        if not counts.sum():
            continue
        out["groups"].append({"index": i, "name": name, "label": src["labels"][i], "n": int(counts.sum()),
                              "rate": np.round(counts / max(out["n_averaged"] * bin_s, 1e-9), 3).tolist()})
    out["edges"] = np.round(edges, 3).tolist()
    return out


def coherence_profile(ts: TimeSeries, ranges: np.ndarray, band=DEFAULT_BAND, reference: str | None = None,
                      layout: dict | None = None, target_hz: float = TARGET_HZ, min_plv: float = 0.5,
                      progress=None) -> dict:
    """Theta phase across the probe: how tightly each contact follows a reference contact, and at
    what phase, as a depth profile.

    Theta is coherent over the whole hippocampus but **reverses in phase across the pyramidal
    layer**: contacts above and below it ride the same cycle half a period apart. That makes the
    depth of the reversal a landmark that does not depend on the CSD or on catching ripples, which
    is why it is worth measuring when the recording has clear theta.

    Per contact, over the samples inside ``ranges`` only (phase means nothing where the rhythm is
    absent): ``plv`` = the phase-locking value against the reference (``|mean e^{i(phi-phi_ref)}|``,
    1 = a fixed phase relationship, 0 = none), ``phase_deg`` = that fixed phase in -180..180, and
    ``amp_uV`` = the median band amplitude. ``step_deg`` is the phase change from the contact above,
    and ``reversal`` reports the largest such step (the candidate layer boundary) when it exceeds
    ``step_min_deg``; contacts whose ``plv`` is below ``min_plv`` are left out of that search, since
    a contact that does not follow the rhythm has no meaningful phase.

    The reference is the contact with the largest theta amplitude unless one is named.
    """
    from .swr import decimated
    ax = depth_order(ts, layout)
    names = ax["names"]
    if len(names) < 2:
        raise ValueError("need at least two electrodes")
    if progress:
        progress(0.1, f"reading {len(names)} channels")
    t, X, fsd = decimated(ts, names, target_hz, progress=progress)
    mask = in_ranges(t, ranges) >= 0 if len(ranges) else np.ones(t.size, bool)
    if mask.sum() < fsd:                       # under a second of signal: nothing to measure
        raise ValueError("the chosen epochs hold too little signal for a phase measurement")
    ph = np.empty((len(names), int(mask.sum())))
    amp = np.empty_like(ph)
    for i in range(len(names)):
        a = analytic(bandpass(np.asarray(X[i], dtype=float), fsd, band[0] or None, band[1] or None))
        ph[i] = np.angle(a[mask])
        amp[i] = np.abs(a[mask])
    med_amp = np.median(amp, axis=1)
    if reference in (None, "", "auto") or reference not in names:
        reference = names[int(np.argmax(med_amp))]
    ref = names.index(reference)
    z = np.exp(1j * (ph - ph[ref]))
    r = z.mean(axis=1)
    phase_deg = np.degrees(np.angle(r))
    rows = []
    for i, nm in enumerate(names):
        rows.append({"name": nm, "depth_um": float(ax["depth_um"][i]), "plv": float(abs(r[i])),
                     "phase_deg": float(phase_deg[i]), "amp_uV": float(med_amp[i]),
                     "bad": nm in (ax.get("bad") or []), "reference": nm == reference})
    for i in range(1, len(rows)):              # phase change from the contact above, wrapped
        rows[i]["step_deg"] = float((phase_deg[i] - phase_deg[i - 1] + 180.0) % 360.0 - 180.0)
    rows[0]["step_deg"] = 0.0
    return {"channel": reference, "band": [float(band[0]), float(band[1])], "rows": rows,
            "n_samples": int(mask.sum()), "seconds": float(mask.sum() / fsd), "fs": float(fsd),
            "geometry": bool(ax["geometry"]), "spacing_um": float(ax["spacing_um"]),
            "min_plv": float(min_plv), **reversal(rows, min_plv)}


def depth_order(ts: TimeSeries, layout: dict | None) -> dict:
    """The electrodes shallow -> deep (the probe's order when there is geometry)."""
    from .csd import depth_axis
    return depth_axis(layout, ts.headstage_channels() or list(ts.names))


def reversal(rows: list[dict], min_plv: float = 0.5, step_min_deg: float = 60.0) -> dict:
    """The largest phase step between neighbouring contacts that both follow the rhythm: where
    theta reverses, i.e. the candidate pyramidal layer."""
    best = None
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if a["plv"] < min_plv or b["plv"] < min_plv or a["bad"] or b["bad"]:
            continue
        if best is None or abs(b["step_deg"]) > abs(best[1]["step_deg"]):
            best = (a, b)
    if best is None or abs(best[1]["step_deg"]) < step_min_deg:
        return {"reversal": None, "reversal_note": "no phase reversal: no neighbouring pair with a "
                f"step over {step_min_deg:.0f}° and a phase-locking value over {min_plv}"}
    a, b = best
    return {"reversal": {"above": a["name"], "below": b["name"], "step_deg": b["step_deg"],
                         "depth_um": (a["depth_um"] + b["depth_um"]) / 2.0,
                         "between": [a["depth_um"], b["depth_um"]]},
            "reversal_note": f"theta turns by {b['step_deg']:.0f}° between {a['name']} and {b['name']}"}
