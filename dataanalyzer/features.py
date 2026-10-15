"""Per-event features for detected SWRs, so rejection thresholds can be chosen from the data.

For each event of a detection result: peak ripple frequency and cycle count, the frequency and
share of what the event *adds* to the spectrum over its surroundings, ripple-band power and its
ratio to a 300-500 Hz control band, the radiatum sharp-wave (CSD sink and LFP
deflection, using the layer estimate when there is one), a spike-rate z-score from saved
spikes, and the movement / reference scores the detector recorded. Results are written back
into the detection's JSON file (``event["features"]``) and can be exported as CSV or turned
into a saved rejection stage (``apply_filter``).

When the detection has a ``line_hz`` parameter (the laminar detector), mains harmonics are notched
out of the ripple and control bands before anything is measured: a steady 180 Hz line carrying
most of the band's power otherwise sets the zero-crossing frequency of every event, real or not.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Callable

import numpy as np

from .analysis import bandpass, envelope, remove_line
from .channels import interpolate_bad
from .csd import csd, depth_axis
from .io import TimeSeries

COLUMNS = [
    ("duration_ms", "duration (ms)"),
    ("score", "peak envelope z"),
    ("peak_hz", "peak frequency (Hz)"),
    ("burst_hz", "frequency the event adds over its flanks (Hz)"),
    ("burst_frac", "share of the event's band power added over its flanks"),
    ("n_cycles", "cycles"),
    ("ripple_db", "ripple power (dB)"),
    ("control_ratio_db", "ripple / 300-500 Hz (dB)"),
    ("sw_csd", "radiatum CSD sink (µV/mm²)"),
    ("sw_lfp", "radiatum LFP deflection (µV)"),
    ("mua_z", "spike-rate z"),
    ("move_z", "movement z"),
    ("ref_z", "reference z"),
    ("sw_z", "sharp wave z (deep minus shallow LFP)"),
    ("global_z", "probe-wide ripple z"),
    ("line_frac", "share of ripple-band power on mains harmonics"),
    ("channels_pct", "electrodes showing the artefact (%)"),
    ("amp_uV", "artefact peak-to-peak, median across electrodes (µV)"),
]
PRE_S, POST_S, PAD_S = 0.1, 0.1, 0.25


def _zero_crossing_freq(x: np.ndarray, fs: float) -> float:
    s = np.signbit(x)
    n = int(np.count_nonzero(s[1:] != s[:-1]))
    return n / 2.0 / (x.size / fs) if x.size > 1 else float("nan")


BURST_HALF_S = 0.05     # the event window for the excess spectrum: ±50 ms around the peak ...
BURST_FLANK_S = 0.10    # ... against the mean of the 100 ms on either side of it


def excess_spectrum(y: np.ndarray, k: int, fs: float, low: float, high: float, line_hz: float = 0.0,
                    half_s: float = BURST_HALF_S, flank_s: float = BURST_FLANK_S) -> tuple[float | None, float | None]:
    """(frequency, share) of what the samples around ``y[k]`` add to the spectrum over their flanks.

    The power spectrum of ±``half_s`` around sample ``k`` minus the mean spectrum of the ``flank_s``
    either side, between ``low`` and ``high`` Hz, bins within 6 Hz of a mains harmonic excluded.
    Frequency = where that excess peaks; share = excess / event power in the band. Anything steady
    (hum, a background rhythm) cancels, so this is the event's own frequency. (None, None) when
    the window does not fit inside ``y``."""
    h, fl = int(round(half_s * fs)), int(round(flank_s * fs))
    if k - h - fl < 0 or k + h + fl > y.size or h < 4:
        return None, None
    nfft = 1 << max(12, int(np.ceil(np.log2(4 * h))))
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)

    def power(seg):
        seg = (seg - seg.mean()) * np.hanning(seg.size)
        return np.abs(np.fft.rfft(seg, n=nfft)) ** 2 / seg.size

    ev = power(y[k - h:k + h])
    base = 0.5 * (power(y[k - h - fl:k - h]) + power(y[k + h:k + h + fl]))
    keep = (f >= low) & (f <= high)
    if line_hz and line_hz > 0:
        keep &= np.abs(f - np.round(f / line_hz) * line_hz) > 6.0
    if not keep.any():
        return None, None
    ex = np.clip(ev - base, 0.0, None)[keep]
    total = float(ev[keep].sum())
    if total <= 0 or not ex.any():
        return None, 0.0
    return float(f[keep][int(np.argmax(ex))]), float(ex.sum() / total)


def event_features(ts: TimeSeries, layout: dict | None, result: dict, layers: dict | None = None,
                   spikes: dict | None = None, progress: Callable[[float, str], None] | None = None) -> dict:
    fs = float(ts.fs)
    params = result["params"]
    channel = result["channel"]
    low, high = float(params.get("low_hz", 150.0)), float(params.get("high_hz", 250.0))
    line_hz = float(params.get("line_hz") or 0.0)
    clean = (lambda v, a, b: remove_line(v, fs, line_hz, a, b)) if line_hz > 0 else (lambda v, a, b: v)
    axis = depth_axis(layout, ts.headstage_channels(), channel)
    names = axis["names"]
    rad = None
    if layers and layers.get("layers", {}).get("radiatum") and layers["layers"]["radiatum"]["channel"] in names:
        rad = layers["layers"]["radiatum"]["channel"]
    have_csd = len(names) >= 3
    control_ok = fs > 1100.0
    spike_t = np.asarray(spikes["times"]) if spikes and spikes.get("times") is not None and len(spikes["times"]) else None
    base_rate = spike_t.size / max(ts.duration, 1e-9) if spike_t is not None else None

    groups = [("accepted", result["events"]), ("rejected", result.get("rejected", []))]
    total = sum(len(g) for _, g in groups) or 1
    done = 0
    for _, events in groups:
        for e in events:
            f: dict[str, float | None] = {}
            pk = e["peak"]
            lo = int(np.searchsorted(ts.t, pk - PRE_S - PAD_S, "left")); hi = int(np.searchsorted(ts.t, pk + POST_S + PAD_S, "right"))
            lo, hi = max(lo, 0), min(hi, ts.n)
            t = ts.t[lo:hi]
            in_ev = (t >= e["start"]) & (t <= e["end"])
            near = np.abs(t - pk) <= 0.02
            base = (t >= pk - 0.1) & (t <= pk - 0.05)
            y = ts.block(lo, hi, [channel])[:, 0]        # only the event's samples, not the whole channel
            band = bandpass(clean(y, low, high), fs, low, high)
            f["duration_ms"] = 1000.0 * (e["end"] - e["start"])
            f["score"] = e.get("score")
            seg = band[in_ev] if in_ev.sum() > 4 else band[near]
            f["peak_hz"] = _zero_crossing_freq(seg, fs)
            f["n_cycles"] = f["peak_hz"] * (e["end"] - e["start"]) if f["peak_hz"] == f["peak_hz"] else None
            f["burst_hz"], f["burst_frac"] = excess_spectrum(y, int(np.searchsorted(t, pk, "left")), fs, low, high, line_hz)
            rp = float(np.mean(seg ** 2)) if seg.size else float("nan")
            f["ripple_db"] = 10 * math.log10(rp + 1e-30)
            if control_ok:
                ctrl = bandpass(clean(y, 300.0, min(500.0, fs / 2.5)), fs, 300.0, min(500.0, fs / 2.5))
                cseg = ctrl[in_ev] if in_ev.sum() > 4 else ctrl[near]
                f["control_ratio_db"] = f["ripple_db"] - 10 * math.log10(float(np.mean(cseg ** 2)) + 1e-30)
            else:
                f["control_ratio_db"] = None
            if have_csd:
                blk = ts.block(lo, hi, names).T
                lfp = np.vstack([bandpass(r, fs, 1.0, 50.0) for r in blk])
                lfp = interpolate_bad(lfp, names, axis.get("bad"))
                c = csd(lfp, axis["spacing_um"], True)
                if rad is not None:
                    i = names.index(rad)
                    f["sw_csd"] = float(c[i, near].min() - c[i, base].mean()) if base.any() else float(c[i, near].min())
                    f["sw_lfp"] = float(lfp[i, near].min() - lfp[i, base].mean()) if base.any() else float(lfp[i, near].min())
                else:   # no layer estimate: strongest sink anywhere on the shank
                    prof = c[:, near].min(axis=1) - (c[:, base].mean(axis=1) if base.any() else 0)
                    i = int(np.argmin(prof))
                    f["sw_csd"] = float(prof[i])
                    f["sw_lfp"] = float(lfp[i, near].min() - (lfp[i, base].mean() if base.any() else 0))
            else:
                f["sw_csd"] = f["sw_lfp"] = None
            if spike_t is not None:
                n = int(np.searchsorted(spike_t, e["end"], "right") - np.searchsorted(spike_t, e["start"], "left"))
                expected = base_rate * (e["end"] - e["start"])
                f["mua_z"] = (n - expected) / math.sqrt(expected) if expected > 0 else None
            else:
                f["mua_z"] = None
            f["move_z"] = e.get("move_score")
            f["ref_z"] = e.get("ref_score")
            f["sw_z"] = e.get("sw_score")
            f["global_z"] = e.get("global_score")
            f["line_frac"] = e.get("line_frac")
            f["channels_pct"] = 100.0 * e["channels_frac"] if e.get("channels_frac") is not None else None
            f["amp_uV"] = e.get("amp_uV")
            e["features"] = {k: (None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)) for k, v in f.items()}
            done += 1
            if progress and done % 10 == 0:
                progress(done / total, f"event {done}/{total}")
    result["feature_columns"] = [{"key": k, "label": lb} for k, lb in COLUMNS]
    result["feature_meta"] = {"radiatum": rad, "spikes": spike_t is not None, "control_band": control_ok,
                              "bad_channels": axis.get("bad", []), "n": total}
    return result


def table(result: dict, which: str = "all") -> dict:
    rows = []
    for setname, events in (("accepted", result["events"]), ("rejected", result.get("rejected", []))):
        if which not in ("all", setname):
            continue
        for i, e in enumerate(events):
            rows.append({"index": i, "set": setname, "reason": e.get("reason"), "start": e["start"], "end": e["end"], "peak": e["peak"],
                         **(e.get("features") or {})})
    return {"det": result["name"], "columns": result.get("feature_columns", [{"key": k, "label": lb} for k, lb in COLUMNS]),
            "meta": result.get("feature_meta"), "rows": rows, "computed": any("features" in e for e in result["events"])}


def to_csv(result: dict, include=None) -> str:
    """All events with their features; with ``include`` (time ranges, see :mod:`sections`) an
    ``in_range`` column says which ones the Events tab uses."""
    from .sections import event_in
    tb = table(result, "all")
    for r in tb["rows"]:
        r["in_range"] = int(event_in(include, r))
    keys = ["set", "index", "reason", "start", "end", "peak", "in_range"] + [c["key"] for c in tb["columns"]]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for r in tb["rows"]:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    return buf.getvalue()


def apply_filter(result: dict, rules: dict[str, dict]) -> dict:
    """Move accepted events failing any rule {feature: {min, max}} to rejected with reason
    'filter:<feature>'. Returns counts. Features must have been computed."""
    kept, moved = [], 0
    for e in result["events"]:
        f = e.get("features") or {}
        failed = None
        for key, rule in rules.items():
            v = f.get(key)
            if v is None:
                continue
            if rule.get("min") is not None and v < float(rule["min"]):
                failed = key; break
            if rule.get("max") is not None and v > float(rule["max"]):
                failed = key; break
        if failed:
            e["reason"] = f"filter:{failed}"
            result.setdefault("rejected", []).append(e)
            moved += 1
        else:
            kept.append(e)
    result["events"] = kept
    result["rejected"].sort(key=lambda e: e["peak"])
    result["n_events"], result["n_rejected"] = len(kept), len(result["rejected"])
    result["filter_rules"] = rules
    return {"moved": moved, "n_events": len(kept), "n_rejected": len(result["rejected"])}


def restore_filtered(result: dict) -> dict:
    back = [e for e in result.get("rejected", []) if str(e.get("reason", "")).startswith("filter:")]
    result["rejected"] = [e for e in result.get("rejected", []) if not str(e.get("reason", "")).startswith("filter:")]
    for e in back:
        e.pop("reason", None)
    result["events"] = sorted(result["events"] + back, key=lambda e: e["peak"])
    result["n_events"], result["n_rejected"] = len(result["events"]), len(result["rejected"])
    result.pop("filter_rules", None)
    return {"restored": len(back), "n_events": len(result["events"]), "n_rejected": len(result["rejected"])}
