"""Current source density around detected events, on the probe's depth axis.

``event_view`` assembles everything the event browser shows for one event (or
the average over all events): the LFP across depth-ordered channels, its CSD,
and the detection features recomputed on the same window with the statistics
saved by the detector, so the traces match what the detector saw.

CSD follows the standard second spatial derivative (Nicholson & Freeman 1975)
with Vaknin edge padding and an optional 3-point Hamming smoothing across
channels. Sign convention: sinks negative. Units are µV/mm² (conductivity is
not applied). The A1x32-Poly2's two staggered columns are treated as one
linear array at 25 µm effective spacing, as in ../Neuronexus.
"""

from __future__ import annotations

import numpy as np

from .analysis import Param, Result, Trace, analysis, bandpass, decimate_for
from .channels import interpolate_bad
from .io import TimeSeries

EDGE_PAD_S = 0.25


def depth_axis(layout: dict | None, headstage: list[str], channel: str | None = None) -> dict:
    """Channels shallow -> deep on one shank (the detection channel's shank, else the first).

    ``headstage`` is ``ts.headstage_channels()``: the fallback, in recorded order with unit
    spacing, when no probe geometry is available."""
    if layout and layout.get("contacts"):
        contacts = layout["contacts"]
        by_name = {c["name"]: c for c in contacts}
        shank = by_name[channel]["shank"] if channel in by_name else contacts[0]["shank"]
        col = sorted((c for c in contacts if c["shank"] == shank), key=lambda c: (-c["y_um"], c["x_um"]))
        ys = np.array([c["y_um"] for c in col])
        spacing = float(np.median(np.abs(np.diff(ys)))) if ys.size > 1 else 1.0
        bad = [n for n in (layout.get("bad") or []) if n in {c["name"] for c in col}]
        return {"names": [c["name"] for c in col], "depth_um": ys.tolist(), "spacing_um": spacing,
                "shank": shank, "source": layout.get("name"), "geometry": True, "bad": bad}
    bad = [n for n in ((layout or {}).get("bad") or []) if n in headstage]
    return {"names": headstage, "depth_um": list(range(len(headstage), 0, -1)), "spacing_um": 1.0,
            "shank": 0, "source": None, "geometry": False, "bad": bad}


def csd(phi: np.ndarray, spacing_um: float, smooth: bool = True) -> np.ndarray:
    """phi: (n_channels, n_samples) in µV, shallow -> deep. Returns µV/mm², sinks negative."""
    if phi.shape[0] < 3:
        return np.zeros_like(phi)
    if smooth:
        padded = np.vstack([phi[:1], phi, phi[-1:]])
        phi = 0.23 * padded[:-2] + 0.54 * padded[1:-1] + 0.23 * padded[2:]
    padded = np.vstack([phi[:1], phi, phi[-1:]])           # Vaknin: duplicate edges
    d2 = padded[:-2] - 2 * padded[1:-1] + padded[2:]
    return -d2 / (spacing_um ** 2) * 1e6


def _window(ts: TimeSeries, names: list[str], centre: float, pre_s: float, post_s: float,
            band: tuple[float, float] | None, with_raw: bool = False):
    """(t_rel, (n_channels, n_samples)) around ``centre``, always the same sample count for a given
    pre/post so events can be averaged; filtered with edge padding when band is given.
    ``with_raw`` appends the unfiltered samples of the same window: (t_rel, filtered, raw)."""
    fs = float(ts.fs)
    n_pre, n_post, n_pad = int(round(pre_s * fs)), int(round(post_s * fs)), int(round(EDGE_PAD_S * fs)) if band else 0
    ic = int(np.searchsorted(ts.t, centre, "left"))
    lo, hi = ic - n_pre - n_pad, ic + n_post + 1 + n_pad
    clo, chi = max(lo, 0), min(hi, ts.n)
    blk = ts.block(clo, chi, names).T
    if clo != lo or chi != hi:                                  # near the recording edge: pad by edge value
        blk = np.pad(blk, ((0, 0), (clo - lo, hi - chi)), mode="edge")
    raw = blk[:, n_pad:-n_pad].copy() if (with_raw and n_pad) else (blk.copy() if with_raw else None)
    if band and (band[0] or band[1]):
        blk = np.vstack([bandpass(row, fs, band[0] or None, band[1] or None) for row in blk])
    if n_pad:
        blk = blk[:, n_pad:-n_pad]
    t = (np.arange(-n_pre, n_post + 1)) / fs
    return (t, blk, raw) if with_raw else (t, blk)


def feature_window(ts: TimeSeries, detector, result: dict, centre: float, pre_s: float, post_s: float,
                   layout: dict | None = None) -> dict:
    """Detection features on the same fixed-length sample grid as ``_window``."""
    fs = float(ts.fs)
    n_pre, n_post, n_pad = int(round(pre_s * fs)), int(round(post_s * fs)), int(round(EDGE_PAD_S * fs))
    ic = int(np.searchsorted(ts.t, centre, "left"))
    lo, hi = max(ic - n_pre - n_pad, 0), min(ic + n_post + 1 + n_pad, ts.n)
    w = ts.window(ts.t[lo], ts.t[hi - 1])
    kw = {"layout": layout} if detector.accepts("layout", features=True) else {}
    f = detector.features(w, result["channel"], result.get("stats", {}), **result["params"], **kw)
    out = {}
    for k, v in f.items():
        v = np.asarray(v, dtype=float)
        start = (ic - n_pre) - lo
        seg = v[max(start, 0):start + n_pre + n_post + 1]
        if seg.size != n_pre + n_post + 1:
            seg = np.pad(seg, (max(-start, 0), n_pre + n_post + 1 - seg.size - max(-start, 0)), mode="edge")
        out[k] = seg
    return out


def _decimate_rows_minmax(t: np.ndarray, rows: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Like ``_decimate_rows`` but keeps each block's minimum and maximum (in the order they occur),
    so unfiltered traces keep their spikes and fast transients. All rows share the returned time axis."""
    if t.size <= max_points:
        return t, rows
    step = int(np.ceil(t.size / (max_points / 2)))
    nb = t.size // step
    blocks = rows[:, :nb * step].reshape(rows.shape[0], nb, step)
    lo, hi = blocks.min(axis=2), blocks.max(axis=2)
    min_first = blocks.argmin(axis=2) < blocks.argmax(axis=2)
    out = np.empty((rows.shape[0], nb * 2))
    out[:, 0::2] = np.where(min_first, lo, hi)
    out[:, 1::2] = np.where(min_first, hi, lo)
    tb = t[:nb * step].reshape(nb, step)
    tt = np.empty(nb * 2)
    tt[0::2], tt[1::2] = tb[:, step // 4], tb[:, (3 * step) // 4]
    return tt, out


def _decimate_rows(t: np.ndarray, rows: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if t.size <= max_points:
        return t, rows
    step = int(np.ceil(t.size / max_points))
    return t[::step], rows[:, ::step]


def event_view(ts: TimeSeries, layout: dict | None, result: dict, detector, index: int | None,
               pre_ms: float = 100.0, post_ms: float = 100.0, lfp_band: tuple[float, float] = (1.0, 300.0),
               smooth: bool = True, max_points: int = 1200, max_average: int = 500, align: str = "peak",
               with_features: bool = True, extras: list[str] | None = None) -> dict:
    """Everything for one event (``index``) or the aligned average of all events (``index=None``).
    ``align`` is the event time the window is centred on: "peak", "start" or "end" (for TTL pulses
    the rising and falling edge). ``extras`` are channels to cut out unchanged beside the
    electrodes — the detector's own recorded output (``RIP_FEAT`` and its thresholds ``RIP_ON`` /
    ``RIP_OFF``), so the Events tab can show what the online detection was actually looking at."""
    at = lambda e: e.get(align, e["peak"])
    feats_on = with_features and detector is not None and detector.features is not None   # the viewer no longer draws them
    events = result["events"]
    if not events:
        raise ValueError("no events in this result")
    channel = result["channel"]
    axis = depth_axis(layout, ts.headstage_channels(), channel)
    names = axis["names"]
    pre_s, post_s = pre_ms / 1000.0, post_ms / 1000.0

    extra_ts, extra_names = extras if extras else (None, [])
    extra_names = [c for c in extra_names if extra_ts is not None and c in extra_ts.names]
    t_rel_x = None
    if index is None:
        sel = events if len(events) <= max_average else events[:: int(np.ceil(len(events) / max_average))]
        acc_lfp = acc_raw = acc_extra = None
        feats_acc: dict[str, np.ndarray] = {}
        n_used = 0
        for e in sel:
            t_rel, blk, rw = _window(ts, names, at(e), pre_s, post_s, lfp_band, with_raw=True)
            acc_lfp = blk if acc_lfp is None else acc_lfp + blk
            acc_raw = rw if acc_raw is None else acc_raw + rw
            if extra_names:
                t_rel_x, ex = _window(extra_ts, extra_names, at(e), pre_s, post_s, None)
                acc_extra = ex if acc_extra is None else acc_extra + ex
            if feats_on:
                for k, v in feature_window(ts, detector, result, at(e), pre_s, post_s, layout).items():
                    feats_acc[k] = v if k not in feats_acc else feats_acc[k] + v
            n_used += 1
        lfp = acc_lfp / max(n_used, 1)
        raw = acc_raw / max(n_used, 1)
        extra = acc_extra / max(n_used, 1) if acc_extra is not None else None
        feats = {k: v / max(n_used, 1) for k, v in feats_acc.items()}
        event = {"peak": None, "centre": None, "n_averaged": n_used}
    else:
        index = int(index) % len(events)
        e = events[index]
        t_rel, lfp, raw = _window(ts, names, at(e), pre_s, post_s, lfp_band, with_raw=True)
        t_rel_x, extra = _window(extra_ts, extra_names, at(e), pre_s, post_s, None) if extra_names else (None, None)
        feats = feature_window(ts, detector, result, at(e), pre_s, post_s, layout) if feats_on else {}
        event = {**{k: e[k] for k in ("start", "end", "peak", "score", "duration", "ref_score", "move_score", "sw_score",
                                       "global_score", "line_frac", "channels_frac", "amp_uV", "max_channel", "train", "pulse", "online_feature", "online_threshold", "reason") if k in e}, "index": index, "centre": at(e)}

    lfp_i = interpolate_bad(lfp, names, axis.get("bad"))
    c = csd(lfp_i, axis["spacing_um"], smooth) if axis["geometry"] or lfp.shape[0] >= 3 else np.zeros_like(lfp)
    t_ms = t_rel * 1000.0
    t_d, lfp_d = _decimate_rows(t_ms, lfp, max_points)
    _, csd_d = _decimate_rows(t_ms, c, max_points)
    t_raw, raw_d = _decimate_rows_minmax(t_ms, raw, 2 * max_points)
    t_ex, extra_d = _decimate_rows_minmax(t_rel_x * 1000.0, extra, max_points) if extra is not None else (np.empty(0), np.empty((0, 0)))
    feat_out = {}
    for k, v in feats.items():
        _, vd = _decimate_rows(t_ms, np.asarray(v)[None, :], max_points)
        feat_out[k] = vd[0].tolist()
    vmax = float(np.percentile(np.abs(csd_d), 99)) if csd_d.size else 1.0
    return {
        "det": result["name"], "label": result["label"], "channel": channel, "n": len(events),
        "index": index, "average": index is None, "event": event, "align": align,
        "t_ms": t_d.tolist(),
        "lfp": {"names": names, "depth_um": axis["depth_um"], "data": lfp_d.tolist(), "unit": "uV",
                "band": list(lfp_band)},
        # the unfiltered samples of the same window (min/max pairs, so nothing fast is lost); the CSD uses "lfp"
        "raw": {"names": names, "depth_um": axis["depth_um"], "t_ms": np.round(t_raw, 4).tolist(),
                "data": np.round(raw_d, 2).tolist(), "unit": "uV"},
        # channels the recording carries beside the electrodes: what the online detector computed and thresholded
        "extras": {"names": extra_names, "t_ms": np.round(t_ex, 4).tolist(), "data": np.round(extra_d, 3).tolist(),
                   "units": [extra_ts.unit(c) for c in extra_names] if extra_names else [],
                   "source": (extra_ts.stream if extra_ts is not None and extra_ts is not ts else "")},
        "csd": {"names": names, "depth_um": axis["depth_um"], "data": csd_d.tolist(), "vmax": vmax or 1.0,
                "spacing_um": axis["spacing_um"], "unit": "uV/mm^2", "geometry": axis["geometry"],
                "probe": axis["source"], "shank": axis["shank"], "bad": axis.get("bad", [])},
        "set": result.get("set", "accepted"), "n_rejected": len(result.get("rejected", [])),
        "features": {"traces": feat_out, "params": result["params"], "stats": result.get("stats", {}),
                     "trace_thresholds": detector.thresholds(result["params"]) if detector.thresholds else None,
                     "thresholds": {k: v for k, v in result["params"].items() if k.startswith("z_")},
                     "ref_thresholds": {k: v for k, v in result["params"].items() if k.startswith("ref_z")},
                     "move_thresholds": {k: v for k, v in result["params"].items() if k == "move_z"}},
    }


# ---------------------------------------------------------------------------
# continuous CSD for the Browse tab
# ---------------------------------------------------------------------------

@analysis("CSD map (all channels)", [
    Param("lfp_low", "float", 1.0, min=0.0, step=0.5, help="LFP band before the spatial derivative; 0 disables"),
    Param("lfp_high", "float", 300.0, min=5.0, step=10.0, help="also sets the time resolution (signal decimated to ~4x)"),
    Param("smooth", "bool", True, help="3-point Hamming across channels"),
    Param("show_lfp", "bool", False, help="overlay the LFP traces (one per row) on the map"),
], multichannel=True)
def csd_map(ts: TimeSeries, channel: str, lfp_low, lfp_high, smooth, show_lfp, layout=None) -> Result:
    """Current source density over depth for the visible window, on the shank of the chosen
    channel (or the first). Sinks blue, sources red. Needs probe geometry for real spacing."""
    fs = float(ts.fs)
    axis = depth_axis(layout, ts.headstage_channels(), channel if channel in ts.names else None)
    names = axis["names"]
    if len(names) < 3:
        raise ValueError("CSD needs at least three channels on one shank")
    blk = ts.block(0, ts.n, names).T
    band = (lfp_low or None, lfp_high or None)
    rows, fsd = [], fs
    for r in blk:
        rf = bandpass(r, fs, band[0], band[1]) if (band[0] or band[1]) else r
        rd, fsd = decimate_for(rf, fs, lfp_high)
        rows.append(rd)
    lfp = np.vstack(rows)
    t = ts.t[0] + np.arange(lfp.shape[1]) / fsd
    c = csd(interpolate_bad(lfp, names, axis.get("bad")), axis["spacing_um"], smooth)
    depth = np.asarray(axis["depth_um"], dtype=float)          # shallow -> deep, i.e. decreasing
    order = np.argsort(depth)                                   # image rows must increase with the row axis
    bad = axis.get("bad") or []
    traces = [Trace("CSD" + (f" (interpolated: {', '.join(bad)})" if bad else ""), t, None, "image", f=depth[order], z=c[order],
                    zlabel="µV/mm²" if axis["geometry"] else "a.u.", row_labels=[names[i] for i in order], cmap="diverging")]
    if show_lfp:
        span = float(np.median(np.ptp(lfp, axis=1))) or 1.0
        step = float(np.median(np.abs(np.diff(depth)))) if depth.size > 1 else 1.0
        for i in order:
            traces.append(Trace(names[i], t, depth[i] + (lfp[i] - lfp[i].mean()) / span * step * 0.8))
    return Result(traces, ylabel="height above tip (µm)" if axis["geometry"] else "channel (recorded order)")


# ---------------------------------------------------------------------------
# hippocampal layer estimation from event-triggered CSD strength profiles
# ---------------------------------------------------------------------------
# After ../Neuronexus/openephys/oe/csd.py::identify_layers, with one change: the ripple-band
# CSD source is measured per event and then averaged, because ripples are not phase-locked
# across events and would cancel in an averaged LFP. Convention: depth axis increases towards
# the surface (height above tip), so radiatum is *below* pyramidale and oriens *above* it.

import json as _json
from datetime import datetime as _dt, timezone as _tz

LAYER_ORDER = ("oriens", "pyramidale", "radiatum")


def _extent(profile: np.ndarray, centre: int, sign: float, frac: float = 0.5) -> tuple[int, int]:
    """Contiguous index range around ``centre`` where sign*profile >= frac * its peak."""
    p = sign * profile
    thr = frac * p[centre]
    lo = hi = centre
    while lo > 0 and p[lo - 1] >= thr:
        lo -= 1
    while hi < p.size - 1 and p[hi + 1] >= thr:
        hi += 1
    return lo, hi


def layer_estimate(ts: TimeSeries, layout: dict | None, result: dict, pre_ms: float = 250.0, post_ms: float = 100.0,
                   ripple_band: tuple[float, float] = (120.0, 200.0), win_ms: float = 10.0,
                   baseline_ms: tuple[float, float] = (-250.0, -150.0), smooth: bool = True,
                   max_events: int = 200) -> dict:
    """Pyramidale, radiatum and oriens from the CSD around the events of a detection result."""
    events = result["events"]
    if not events:
        raise ValueError("the detection has no events to average")
    axis = depth_axis(layout, ts.headstage_channels(), result["channel"])
    names, depth = axis["names"], np.asarray(axis["depth_um"], dtype=float)   # shallow -> deep
    if len(names) < 5:
        raise ValueError("layer estimation needs at least five channels on one shank")
    fs = float(ts.fs)
    pre_s, post_s = pre_ms / 1000.0, post_ms / 1000.0
    sel = events if len(events) <= max_events else events[:: int(np.ceil(len(events) / max_events))]
    src_acc = pow_acc = sink_acc = rip_acc = None
    n_used = 0
    for e in sel:
        t, wide = _window(ts, names, e["peak"], pre_s, post_s, None)
        t_ms = t * 1000.0
        near = np.abs(t_ms) <= win_ms
        base = (t_ms >= baseline_ms[0]) & (t_ms <= baseline_ms[1])
        if not base.any():
            base = t_ms < t_ms.min() + 0.2 * (t_ms.max() - t_ms.min())
        wide = interpolate_bad(wide, names, axis.get("bad"))
        band = np.vstack([bandpass(row, fs, ripple_band[0], ripple_band[1]) for row in wide])
        csd_b = csd(band, axis["spacing_um"], smooth)
        csd_w = csd(wide, axis["spacing_um"], smooth)
        src = csd_b[:, near].max(axis=1)                       # strongest ripple-band source per channel
        pw = np.sqrt((csd_b[:, near] ** 2).mean(axis=1))       # ripple-band CSD power
        sink = csd_w[:, near].mean(axis=1) - csd_w[:, base].mean(axis=1)   # wideband CSD, baseline-subtracted
        rip = (band[:, near] ** 2).mean(axis=1)                # ripple-band LFP power (cross-check)
        src_acc = src if src_acc is None else src_acc + src
        pow_acc = pw if pow_acc is None else pow_acc + pw
        sink_acc = sink if sink_acc is None else sink_acc + sink
        rip_acc = rip if rip_acc is None else rip_acc + rip
        n_used += 1
    src, pw, sink, rip = (a / n_used for a in (src_acc, pow_acc, sink_acc, rip_acc))

    pyr = int(np.argmax(src))
    deep = np.arange(pyr + 1, len(names))       # larger index = deeper (smaller height above tip)
    shallow = np.arange(0, pyr)
    rad = int(deep[np.argmin(sink[deep])]) if deep.size else None
    ori = int(shallow[np.argmin(sink[shallow])]) if shallow.size else None

    def layer(idx, profile, sign):
        if idx is None:
            return None
        lo, hi = _extent(profile, idx, sign)
        return {"channel": names[idx], "index": idx, "depth_um": float(depth[idx]),
                "extent_um": [float(min(depth[lo], depth[hi])), float(max(depth[lo], depth[hi]))],
                "channels": names[lo:hi + 1], "strength": float(profile[idx])}

    layers = {"pyramidale": layer(pyr, src, +1.0), "radiatum": layer(rad, sink, -1.0), "oriens": layer(ori, sink, -1.0)}
    margin = float((src[pyr] - np.median(src)) / (src.std() + 1e-12))
    rip_peak = int(np.argmax(rip))
    return {
        "detection": result["name"], "detection_label": result["label"], "detection_channel": result["channel"],
        "n_events": n_used, "shank": axis["shank"], "geometry": axis["geometry"], "probe": axis["source"],
        "bad_channels": axis.get("bad", []),
        "params": {"pre_ms": pre_ms, "post_ms": post_ms, "ripple_band": list(ripple_band), "win_ms": win_ms,
                   "baseline_ms": list(baseline_ms), "smooth": smooth},
        "layers": layers,
        "profiles": {"names": names, "depth_um": depth.tolist(), "ripple_source": src.tolist(),
                     "ripple_csd_power": pw.tolist(), "wide_sink": sink.tolist(), "ripple_lfp_power": rip.tolist()},
        "diagnostics": {"pyr_margin": margin, "ripple_power_peak_channel": names[rip_peak],
                        "ripple_power_peak_depth_um": float(depth[rip_peak]),
                        "agree_pyr_ripple_power": abs(rip_peak - pyr) <= 1,
                        "note": "pyramidale = strongest ripple-band CSD source near the peak; radiatum / oriens = "
                                "strongest wideband sinks deeper / shallower than it. Low pyr_margin (< 1.5) or "
                                "disagreement with the ripple-power peak means the profile is flat: do not trust it."},
        "created": _dt.now(_tz.utc).isoformat(timespec="seconds"),
    }


def layers_path(ts: TimeSeries):
    from .detect import results_dir
    return results_dir(ts, "layers") / "layers.json"


def save_layers(ts: TimeSeries, est: dict):
    p = layers_path(ts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(est, indent=1))
    est["path"] = str(p)
    return p


def load_layers(ts: TimeSeries) -> dict | None:
    p = layers_path(ts)
    if not p.exists():
        return None
    try:
        est = _json.loads(p.read_text())
        est["path"] = str(p)
        return est
    except (OSError, ValueError):
        return None
