"""Whole-recording event detection, saved beside the recording.

Detectors find *epochs* (start, end, peak, score) on one channel across the
entire recording, unlike analyses, which redraw the visible window. Results
are written to ``<recording>/dataanalyzer/events/<detector>.json`` so they
load instantly next time and can be consumed by other scripts (CSD, spike
sorting). Register a detector with :func:`detector`, the same way analyses use
``@analysis``; the viewer builds its parameter form from the ``Param`` list.

Built-ins: ``theta`` (high theta-power epochs) and ``swr`` (putative sharp-wave
ripples). Both are envelope z-score detectors and are deliberately simple;
tune the thresholds in the viewer and inspect the marked epochs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .analysis import Param, bandpass, envelope, moving_average
from .io import TimeSeries

RESULTS_DIRNAME = "dataanalyzer"


@dataclass
class Detector:
    name: str
    label: str
    fn: Callable[..., Any]                       # -> events, or (events, stats)
    params: list[Param] = field(default_factory=list)
    doc: str = ""
    features: Callable[..., dict] | None = None  # (ts_window, channel, stats, **params) -> {name: array}
    thresholds: Callable[[dict], dict] | None = None   # params -> {feature trace name: {label: value}} for the Events tab
    category: str = "events"                     # "swr": run from the viewer's SWR tab; anything else from the Events tab

    def accepts(self, arg: str, features: bool = False) -> bool:
        fn = self.features if features else self.fn
        return fn is not None and arg in fn.__code__.co_varnames[:fn.__code__.co_argcount + fn.__code__.co_kwonlyargcount]

    def schema(self) -> dict:
        from dataclasses import asdict
        return {"name": self.name, "label": self.label, "doc": self.doc, "params": [asdict(p) for p in self.params],
                "auto_channel": self.accepts("layout"), "category": self.category}

    def coerce(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {p.name: p.coerce(raw.get(p.name)) for p in self.params}


DETECTORS: dict[str, Detector] = {}


def detector(label: str, params: list[Param] | None = None, features: Callable[..., dict] | None = None,
             thresholds: Callable[[dict], dict] | None = None, category: str = "events"):
    """Decorator registering ``fn(ts, channel, **params) -> events``, ``(events, stats)`` or
    ``(events, stats, extra)``.

    Each event dict has ``start``, ``end``, ``peak`` (seconds) and ``score``. ``stats`` (e.g. the
    mean/std used for z-scoring) is saved with the result so the same signal can be recomputed on
    a short window later by ``features``. ``extra`` is merged into the saved result (e.g. a list of
    ``rejected`` events). A ``Param`` of kind ``"channel"`` is offered as a channel picker; the value
    ``"auto"`` is resolved by the server to the channel farthest from the detection channel.
    A function with a ``layout`` keyword receives the probe layout and may be run with channel
    ``"auto"``; it then reports the channel it chose as ``extra["channel"]``."""
    def wrap(fn):
        DETECTORS[fn.__name__] = Detector(fn.__name__, label, fn, params or [], (fn.__doc__ or "").strip(), features, thresholds, category)
        return fn
    return wrap


def schemas() -> list[dict]:
    return [d.schema() for d in DETECTORS.values()]


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def results_dir(ts: TimeSeries, kind: str = "events") -> Path:
    """``<recording>/dataanalyzer/<kind>`` for a recording folder; next to the file otherwise."""
    src = Path(ts.source) if ts.source and ts.source != "synthetic" else Path.cwd() / "synthetic"
    base = src / RESULTS_DIRNAME if src.is_dir() else src.parent / f"{src.stem}_dataanalyzer"
    if ts.results_subdir:                   # a recording with several streams keeps one set per stream
        base = base / ts.results_subdir
    return base / kind


def pulse_settings_path(ts: TimeSeries) -> Path:
    return results_dir(ts, "pulses") / "pulses.json"


def load_pulse_settings(ts: TimeSeries) -> dict:
    """How the TTL pulses of this recording are built: ``{"merge_ms": ...}``. Saved because the
    pulse sets are rebuilt from the recording on every load."""
    try:
        d = json.loads(pulse_settings_path(ts).read_text())
        return {"merge_ms": max(0.0, float(d.get("merge_ms", 0.0)))}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"merge_ms": 0.0}


def save_pulse_settings(ts: TimeSeries, merge_ms: float) -> dict:
    merge_ms = max(0.0, float(merge_ms))
    p = pulse_settings_path(ts)
    if not merge_ms:
        p.unlink(missing_ok=True)
        return {"merge_ms": 0.0}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"merge_ms": merge_ms}, indent=1))
    return {"merge_ms": merge_ms}


def save_result(ts: TimeSeries, result: dict) -> Path:
    d = results_dir(ts)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{result['name']}.json"
    path.write_text(json.dumps(result, indent=1))
    return path


def load_results(ts: TimeSeries) -> dict[str, dict]:
    d = results_dir(ts)
    out = {}
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            try:
                r = json.loads(p.read_text())
                if "events" in r and "name" in r:
                    out[r["name"]] = r
            except (OSError, ValueError):
                continue
    return out


def delete_result(ts: TimeSeries, name: str) -> bool:
    p = results_dir(ts) / f"{name}.json"
    if p.exists():
        p.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def result_name(detector_name: str, tag: str | None = None) -> str:
    """Saved name of a run: the detector's name, or ``<detector>__<tag>`` so several runs of one
    detector (e.g. different bands) can be kept side by side. The tag is reduced to [A-Za-z0-9._-]."""
    import re
    slug = re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9._-]+", "-", (tag or "").strip())).strip("-.")
    return f"{detector_name}__{slug}" if slug else detector_name


def detector_of(result: dict) -> Detector | None:
    """The detector that produced a saved result (older files have no ``detector`` field: their
    name is the detector's)."""
    return DETECTORS.get(result.get("detector") or result.get("name", ""))


def run(ts: TimeSeries, name: str, channel: str, raw_params: dict | None = None,
        progress: Callable[[float, str], None] | None = None, save: bool = True, layout: dict | None = None,
        tag: str | None = None) -> dict:
    """Run detector ``name`` over the whole recording. ``tag`` saves it as ``<name>__<tag>``
    (see :func:`result_name`) instead of replacing the detector's default result."""
    det = DETECTORS[name]
    params = det.coerce(raw_params or {})
    started = time.perf_counter()
    if progress:
        progress(0.05, f"reading {channel}")
    kw = dict(params)
    if det.accepts("progress"):
        kw["progress"] = progress
    if det.accepts("layout"):
        kw["layout"] = layout
    out = det.fn(ts, channel, **kw)
    if isinstance(out, tuple):
        events, stats = out[0], out[1]
        extra = out[2] if len(out) > 2 else {}
    else:
        events, stats, extra = out, {}, {}
    stats = {k: float(v) for k, v in stats.items()}
    events = [{k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in e.items()}
              for e in events]
    result = {
        "name": result_name(name, tag), "detector": name, "category": det.category, "tag": (tag or "").strip() or None,
        "label": det.label + (f" · {tag.strip()}" if tag and tag.strip() else ""), "channel": channel, "params": params,
        "n_events": len(events), "events": events, "stats": stats,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "recording": ts.source, "fs": ts.fs, "t0": float(ts.t[0]), "t1": float(ts.t[-1]),
    }
    for e in extra.get("rejected", []):
        for k, v in list(e.items()):
            if isinstance(v, (np.floating, np.integer)):
                e[k] = float(v)
    result.update(extra)
    if save:
        result["path"] = str(save_result(ts, result))
    if progress:
        progress(1.0, f"{len(events)} events")
    return result


def scan_trace(t: np.ndarray, z: np.ndarray, points: int = 6000) -> dict:
    """The whole-session detection signal reduced to ``points`` bins by their maximum (so no
    crossing is lost), small enough to save with the result and draw in one go."""
    n = int(np.asarray(z).size)
    step = max(1, int(np.ceil(n / points)))
    nb = n // step
    if nb == 0:
        return {"t": [], "z": [], "bin_s": 0.0}
    cut = nb * step
    return {"t": np.round(np.asarray(t)[:cut:step], 4).tolist(),
            "z": np.round(np.asarray(z)[:cut].reshape(nb, step).max(axis=1), 3).tolist(),
            "bin_s": float(step * (t[1] - t[0])) if n > 1 else 0.0}


def zscore(x: np.ndarray, robust: bool = False) -> tuple[np.ndarray, dict]:
    """z-score with (mean, std), or robustly with (median, 1.4826*MAD) so long epochs do not inflate the scale."""
    if robust:
        centre = float(np.median(x))
        scale = float(1.4826 * np.median(np.abs(x - centre))) or float(x.std()) or 1.0
    else:
        centre, scale = float(x.mean()), float(x.std()) or 1.0
    return (x - centre) / scale, {"centre": centre, "scale": scale, "robust": float(robust)}


def _epochs(z: np.ndarray, t: np.ndarray, enter: float, peak_min: float, min_s: float, max_s: float,
            merge_s: float) -> list[dict]:
    """Contiguous runs of z > enter, merged across short gaps, filtered by peak and duration."""
    above = z > enter
    if not above.any():
        return []
    d = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, above.size)
    # merge gaps shorter than merge_s
    merged = []
    for s, e in zip(starts, ends):
        if merged and t[s] - t[merged[-1][1] - 1] < merge_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        dur = t[e - 1] - t[s]
        if dur < min_s or (max_s > 0 and dur > max_s):
            continue
        k = s + int(np.argmax(z[s:e]))
        if z[k] < peak_min:
            continue
        out.append({"start": t[s], "end": t[e - 1], "peak": t[k], "score": z[k], "duration": dur})
    return out


# ---------------------------------------------------------------------------
# built-in detectors
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# movement from the headstage accelerometer (Intan aux inputs, recorded in mV)
# ---------------------------------------------------------------------------

MOTION_RATE_HZ = 1000.0


def aux_channels(ts: TimeSeries) -> list[str]:
    return ts.aux_channels()


def motion_signal(ts: TimeSeries, hp_hz: float = 1.0, smooth_s: float = 0.5,
                  channels: list[str] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """(t, motion) at ~1 kHz: RMS of the high-passed acceleration across all aux axes, smoothed.

    Gravity and slow posture changes are removed by the high-pass, so the signal is near zero
    when the animal is still and rises with any movement. Units are those of the aux channels:
    mV (the Intan accelerometer is ~340 mV/g) or raw ADC counts for writers that store them as
    "a.u."; the detectors z-score it robustly, so the unit does not matter to them. Returns None
    when the recording has no aux channels."""
    channels = channels or aux_channels(ts)
    if not channels:
        return None
    fs = float(ts.fs)
    step = max(1, int(fs // MOTION_RATE_HZ))
    fsd = fs / step
    acc = None
    for c in channels:
        y = ts.channel(c)
        yd = bandpass(y, fs, None, MOTION_RATE_HZ / 2.5)[::step] if step > 1 else y
        hp = bandpass(yd, fsd, hp_hz, None, order=2)
        acc = hp ** 2 if acc is None else acc + hp ** 2
    motion = np.sqrt(moving_average(acc, int(round(smooth_s * fsd))))
    return ts.t[::step], motion


def motion_z(ts: TimeSeries, hp_hz: float, smooth_s: float, stats: dict | None = None):
    """Robust z of the motion signal (median / MAD, so rest is ~0). Returns (t, z, stats) or None."""
    ms = motion_signal(ts, hp_hz, smooth_s)
    if ms is None:
        return None
    t, m = ms
    if stats is None or "move_centre" not in stats:
        z, st = zscore(m, robust=True)
        return t, z, {"move_centre": st["centre"], "move_scale": st["scale"]}
    return t, (m - stats["move_centre"]) / stats["move_scale"], {}


THETA_RATE_HZ = 1000.0


def _theta_power(ts, channel, low_hz, high_hz, smooth_s, ratio_delta):
    """(t, raw, power) with the power computed at ~1 kHz: theta needs no more, and it makes a
    30 kHz recording ~30x cheaper to process."""
    fs = float(ts.fs)
    y = ts.channel(channel)
    step = max(1, int(fs // THETA_RATE_HZ))
    if step > 1:
        yd = bandpass(y, fs, None, THETA_RATE_HZ / 2.5)[::step]     # anti-alias, then decimate
    else:
        yd = y
    fsd = fs / step
    p = moving_average(envelope(bandpass(yd, fsd, low_hz, high_hz), fsd) ** 2, int(round(smooth_s * fsd)))
    if ratio_delta:
        d = moving_average(envelope(bandpass(yd, fsd, 1.0, 4.0), fsd) ** 2, int(round(smooth_s * fsd)))
        p = p / np.maximum(d, np.percentile(d, 1))
    return ts.t[::step], y, p


def theta_features(ts, channel, stats, low_hz, high_hz, smooth_s, ratio_delta, **_):
    td, y, p = _theta_power(ts, channel, low_hz, high_hz, smooth_s, ratio_delta)
    z = np.interp(ts.t, td, (p - stats["centre"]) / stats["scale"])   # back to the window's sample rate
    return {"raw": y, "theta band": bandpass(y, float(ts.fs), low_hz, high_hz), "power z": z}


@detector("High theta power epochs", [
    Param("low_hz", "float", 5.0, min=1.0, max=20.0, step=0.5, help="theta band"),
    Param("high_hz", "float", 10.0, min=2.0, max=30.0, step=0.5),
    Param("smooth_s", "float", 1.0, min=0.05, max=10.0, step=0.05, help="power smoothing window"),
    Param("z_enter", "float", 1.0, min=0.0, max=8.0, step=0.1, help="epoch while robust-z theta power exceeds this"),
    Param("z_peak", "float", 2.0, min=0.0, max=12.0, step=0.1, help="and its peak exceeds this"),
    Param("min_s", "float", 2.0, min=0.0, max=60.0, step=0.5, help="minimum epoch length"),
    Param("merge_s", "float", 1.0, min=0.0, max=10.0, step=0.1, help="merge epochs closer than this"),
    Param("ratio_delta", "bool", False, help="use theta / delta (1-4 Hz) power ratio instead of theta power"),
], features=theta_features)
def theta(ts: TimeSeries, channel: str, low_hz, high_hz, smooth_s, z_enter, z_peak, min_s, merge_s, ratio_delta,
          progress=None):
    """Epochs of elevated theta-band power on one channel; z-scored robustly (median / MAD) over the recording."""
    if progress:
        progress(0.2, "filtering")
    td, _, p = _theta_power(ts, channel, low_hz, high_hz, smooth_s, ratio_delta)
    if progress:
        progress(0.7, "thresholding")
    z, stats = zscore(p, robust=True)
    return (_epochs(z, td, z_enter, z_peak, min_s, 0.0, merge_s), stats,
            {"scan": scan_trace(td, z), "thresholds": {"z_enter": z_enter, "z_peak": z_peak}})


def _ripple_z(ts, channel, low_hz, high_hz, smooth_ms, stats=None, prefix=""):
    fs = float(ts.fs)
    filt = bandpass(ts.channel(channel), fs, low_hz, high_hz)
    env = envelope(filt, fs, smooth_ms / 1000.0)
    if stats is None:
        z, st = zscore(env)
        return filt, z, {prefix + k: v for k, v in st.items()}
    return filt, (env - stats[prefix + "centre"]) / stats[prefix + "scale"], {}


def swr_features(ts, channel, stats, low_hz, high_hz, smooth_ms, ref_channel="none", gate_movement=True,
                 move_hp_hz=1.0, move_smooth_s=0.5, **_):
    filt, z, _ = _ripple_z(ts, channel, low_hz, high_hz, smooth_ms, stats)
    out = {"raw": ts.channel(channel), "ripple band": filt, "envelope z": z}
    if ref_channel not in ("", "none", "auto") and "ref_centre" in stats and ref_channel in ts.names:
        _, zr, _ = _ripple_z(ts, ref_channel, low_hz, high_hz, smooth_ms, stats, prefix="ref_")
        out[f"ref {ref_channel} envelope z"] = zr
    if gate_movement and "move_centre" in stats:
        mz = motion_z(ts, move_hp_hz, move_smooth_s, stats)
        if mz is not None:
            out["movement z"] = np.interp(ts.t, mz[0], mz[1])
    if "RIP_FEAT" in ts.names:                  # the feature a detector recorded with the data, for comparison
        out["RIP_FEAT (recorded)"] = ts.channel("RIP_FEAT")
    return out


@detector("Putative sharp-wave ripples", [
    Param("low_hz", "float", 150.0, min=80.0, max=300.0, step=5.0, help="ripple band"),
    Param("high_hz", "float", 250.0, min=100.0, max=400.0, step=5.0),
    Param("smooth_ms", "float", 5.0, min=0.0, max=50.0, step=0.5, help="envelope smoothing"),
    Param("z_enter", "float", 2.0, min=0.0, max=8.0, step=0.1, help="event while z-scored envelope exceeds this"),
    Param("z_peak", "float", 4.0, min=0.0, max=12.0, step=0.1, help="and its peak exceeds this"),
    Param("min_ms", "float", 20.0, min=0.0, max=200.0, step=1.0),
    Param("max_ms", "float", 250.0, min=0.0, max=1000.0, step=5.0, help="0 = no maximum"),
    Param("merge_ms", "float", 10.0, min=0.0, max=100.0, step=1.0),
    Param("ref_channel", "channel", "auto",
          help="reference channel far from the detection site; 'auto' = farthest contact on the probe, 'none' disables"),
    Param("ref_z", "float", 2.0, min=0.0, max=10.0, step=0.1,
          help="reject an event if the reference channel's own ripple-band envelope z exceeds this during the event"),
    Param("gate_movement", "bool", True, help="only keep events during immobility (needs accelerometer / aux channels)"),
    Param("move_z", "float", 2.0, min=0.0, max=20.0, step=0.1, help="moving when robust-z movement exceeds this"),
    Param("move_hp_hz", "float", 1.0, min=0.1, max=20.0, step=0.1, help="accelerometer high-pass (removes gravity)"),
    Param("move_smooth_s", "float", 0.5, min=0.05, max=5.0, step=0.05, help="movement smoothing window"),
], features=swr_features, category="swr")
def swr(ts: TimeSeries, channel: str, low_hz, high_hz, smooth_ms, z_enter, z_peak, min_ms, max_ms, merge_ms,
        ref_channel, ref_z, gate_movement, move_z, move_hp_hz, move_smooth_s, progress=None):
    """Ripple-band envelope crossings on one channel, then two rejection stages: events that a
    spatially distant reference channel also shows (broadband noise, EMG), and events during
    movement (accelerometer). Rejected events are kept in the result with their reason."""
    if progress:
        progress(0.2, f"filtering {channel}")
    _, z, stats = _ripple_z(ts, channel, low_hz, high_hz, smooth_ms)
    if progress:
        progress(0.5, "thresholding")
    events = _epochs(z, ts.t, z_enter, z_peak, min_ms / 1000.0, max_ms / 1000.0, merge_ms / 1000.0)
    extra = {"ref_channel": None, "rejected": [], "n_rejected": 0, "n_candidates": len(events),
             "movement_gate": False, "n_rejected_reference": 0, "n_rejected_movement": 0}
    rejected: list[dict] = []
    if not (ref_channel in ("", "none", "auto") or ref_channel == channel or ref_channel not in ts.names):
        if progress:
            progress(0.6, f"filtering reference {ref_channel}")
        _, zr, rstats = _ripple_z(ts, ref_channel, low_hz, high_hz, smooth_ms, prefix="ref_")
        stats.update(rstats)
        kept = []
        for e in events:
            lo = int(np.searchsorted(ts.t, e["start"], "left"))
            hi = int(np.searchsorted(ts.t, e["end"], "right"))
            e["ref_score"] = float(zr[lo:hi].max()) if hi > lo else 0.0
            if e["ref_score"] > ref_z:
                e["reason"] = "reference"
                rejected.append(e)
            else:
                kept.append(e)
        events = kept
        extra["ref_channel"] = ref_channel
    if gate_movement:
        mz = motion_z(ts, move_hp_hz, move_smooth_s)
        if mz is not None:
            if progress:
                progress(0.8, "movement gate")
            tm, zm, mstats = mz
            stats.update(mstats)
            kept = []
            for e in events:
                e["move_score"] = float(np.interp(e["peak"], tm, zm))
                if e["move_score"] > move_z:
                    e["reason"] = "movement"
                    rejected.append(e)
                else:
                    kept.append(e)
            events = kept
            extra["movement_gate"] = True
    rejected.sort(key=lambda e: e["peak"])
    extra.update({"rejected": rejected, "n_rejected": len(rejected),
                  "n_rejected_reference": sum(e.get("reason") == "reference" for e in rejected),
                  "n_rejected_movement": sum(e.get("reason") == "movement" for e in rejected)})
    return events, stats, extra


# ---------------------------------------------------------------------------
# TTL pulses recorded with the data, and comparison between event sets
# ---------------------------------------------------------------------------

PULSE_PREFIX = "pulses_"
ONLINE_PREFIX = PULSE_PREFIX            # older name, kept for callers


def pulse_features(ts, channel, stats, **_):
    """What the Events tab shows under a recorded pulse: the chosen channel as recorded, and every
    channel a processor added to the stream (e.g. a detector's feature and thresholds)."""
    out = {}
    if channel in ts.names:
        out["raw"] = ts.channel(channel)
    for c in ts.channels_of("other"):
        out[f"{c} (recorded)"] = ts.channel(c)
    return out


PULSES = Detector("pulses", "TTL pulses (recorded)", lambda *a, **k: [], [], "", pulse_features)
ONLINE = PULSES


def is_online(result: dict) -> bool:
    """True for event sets that are part of the recording (TTL pulses), not computed by a detector."""
    return bool(result.get("online"))


def _own_channel(ts: TimeSeries, stream: str | None, index_1based: int | None) -> str | None:
    """Name, in the loaded stream, of channel ``index_1based`` of ``stream`` (same headstage input)."""
    from .io import input_from_name
    names = (ts.recording_meta.get("stream_channels") or {}).get(stream or "", [])
    if not index_1based or not 1 <= index_1based <= len(names):
        return None
    name = names[index_1based - 1]
    if stream in (None, "", ts.stream) and (name in ts.names or not name):
        return name if name in ts.names else (ts.names[index_1based - 1] if index_1based <= len(ts.names) else None)
    inp = input_from_name(name)
    for nm, info in zip(ts.names, ts.channel_info or []):
        if inp is not None and info.get("kind") == "headstage" and info.get("input") == inp:
            return nm
    return None


def merge_pulses(events: list[dict], merge_ms: float) -> list[dict]:
    """Pulses separated by at most ``merge_ms`` become one event, from the first rising edge to
    the last falling edge.

    Some stimulators fire on both edges of a trigger pulse, so one stimulus is recorded as two
    pulses a fraction of a millisecond apart and every count, average and comparison sees it
    twice. The gap measured is the next rising edge minus the previous falling edge, so pulses
    that overlap or whose edges nearly touch merge whatever their width. The merged event keeps
    ``n_merged`` (how many pulses it holds) and ``merged_ms`` (the span of their rising edges)."""
    if merge_ms <= 0 or len(events) < 2:
        return events
    gap = merge_ms / 1000.0
    out = [dict(events[0], n_merged=1, merged_ms=0.0)]
    for e in events[1:]:
        last = out[-1]
        if e["start"] - last["end"] <= gap:
            last["end"] = max(last["end"], e["end"])
            last["duration"] = last["end"] - last["start"]
            last["n_merged"] += 1
            last["merged_ms"] = (e["start"] - last["start"]) * 1000.0
            last["score"] = max(last.get("score", 0.0), e.get("score", 0.0))
        else:
            out.append(dict(e, n_merged=1, merged_ms=0.0))
    return out


def online_detections(ts: TimeSeries, merge_ms: float = 0.0) -> dict[str, dict]:
    """Every TTL line of the recording as an event set, in the same shape as saved detector
    results so the viewer can show, browse, average and compare them. What a pulse *means* is
    up to the experiment (a detector firing, a stimulus, a camera frame...), so nothing here
    assumes it. They are rebuilt from the recording on every load, never saved.

    A pulse runs from the rising edge (``start``, also ``peak``) to the next falling edge
    (``end``); views can align on either. When the stream recorded a processor's feature and
    threshold (RIP_FEAT / RIP_ON), ``score`` is their ratio at the rising edge.
    ``merge_ms`` joins pulses that nearly touch into one event (see :func:`merge_pulses`)."""
    out: dict[str, dict] = {}
    procs = ts.recording_meta.get("processors") or {}
    for line in ts.ttl:
        on, off = np.sort(line["on"]), np.sort(line["off"])
        if not on.size:
            continue
        own = line.get("stream") in (None, "", ts.stream)
        feat = ts.channel("RIP_FEAT") if own and "RIP_FEAT" in ts.names else None
        thr = ts.channel("RIP_ON") if own and "RIP_ON" in ts.names else None
        events = []
        for a in on:
            k = int(np.searchsorted(off, a, "right"))
            b = float(off[k]) if k < off.size else float(a)
            e = {"start": float(a), "end": b, "peak": float(a), "score": 0.0, "duration": b - float(a)}
            if feat is not None:
                i = min(int(np.searchsorted(ts.t, a, "left")), ts.n - 1)
                e["online_feature"] = float(feat[i])
                if thr is not None and thr[i] > 0:
                    e["online_threshold"] = float(thr[i])
                    e["score"] = float(feat[i] / thr[i])
            events.append(e)
        n_raw = len(events)
        events = merge_pulses(events, merge_ms)
        source = str(line.get("source") or "")
        params = dict((procs.get(source) or {}).get(line.get("stream") or "", {}))
        try:
            idx = int(float(params.get("Ripple_Input", "")))
        except ValueError:
            idx = None
        channel = _own_channel(ts, line.get("stream"), idx) or (ts.headstage_channels() or ts.names)[0]
        stream = str(line.get("stream") or ts.stream or "")
        name = PULSE_PREFIX + "".join(ch if ch.isalnum() else "_" for ch in f"{stream}_TTL{line['line']}").strip("_")
        where = " · ".join(x for x in (source, stream if not own else "") if x)
        widths = np.array([e["duration"] for e in events])
        out[name] = {"name": name, "label": f"TTL{line['line']} pulses" + (f" · {where}" if where else ""), "online": True,
                     "kind": "pulses", "channel": channel, "params": params, "n_events": len(events), "events": events,
                     "n_pulses": n_raw, "merge_ms": float(merge_ms),
                     "stats": {}, "source": source, "stream": stream, "line": line["line"],
                     "pulse_width_ms": float(np.median(widths) * 1000.0) if widths.size else None,
                     "created": "recorded", "recording": ts.source, "fs": ts.fs,
                     "t0": float(ts.t[0]) if ts.n else 0.0, "t1": float(ts.t[-1]) if ts.n else 0.0}
    return out


def compare(a: dict, b: dict, tol_ms: float = 50.0, include_rejected: bool = True, causal: bool = False) -> dict:
    """Match the events of set ``b`` (e.g. recorded TTL pulses) against ``a`` (e.g. a detector's events).

    A ``b`` event matches an ``a`` event when their extents come within ``tol_ms`` of each other;
    matching is one-to-one in time order. With ``include_rejected`` the events ``a`` rejected
    (reference, movement, filters) are candidates too, so a ``b`` event that falls on one is
    reported as ``rejected:<reason>`` rather than as unmatched. ``latency_ms`` is b.start - a.start,
    how long after a's onset the b event came, summarised as mean, SD, median and range.

    With ``causal`` a b event may only match an a event that *precedes* it by at most ``tol_ms``
    (0 <= b.start - a.start <= tol_ms). That is what a trigger and the response it caused need: the
    pulse marking the stimulus reaching the tissue against the detection pulse that triggered it."""
    tol = tol_ms / 1000.0
    # "_i" (set by a time-range restriction) is the event's index in the full set, so rows point at the right event
    cand = [dict(e, _set="accepted", _i=e.get("_i", i)) for i, e in enumerate(a["events"])]
    if include_rejected:
        cand += [dict(e, _set="rejected", _i=e.get("_i", i)) for i, e in enumerate(a.get("rejected", []))]
    cand.sort(key=lambda e: e["start"])
    used: set[int] = set()
    rows_b = []
    for j, eb in enumerate(sorted(b["events"], key=lambda e: e["start"])):
        best = None
        for k, ea in enumerate(cand):
            if ea["start"] - tol > eb["end"]:
                break
            if k in used or ea["end"] + tol < eb["start"]:
                continue
            lag = eb["start"] - ea["start"]
            if causal and not (0.0 <= lag <= tol):
                continue
            gap = abs(lag) if causal else abs(eb["start"] - ea["peak"])
            if best is None or gap < best[0]:
                best = (gap, k)
        row = {"index": eb.get("_i", j), "start": eb["start"], "end": eb["end"], "peak": eb["peak"], "score": eb.get("score"), "status": "only_b"}
        if best is not None:
            k = best[1]; used.add(k); ea = cand[k]
            row.update({"status": "matched" if ea["_set"] == "accepted" else f"rejected:{ea.get('reason', '?')}",
                        "a_index": ea["_i"], "a_set": ea["_set"], "a_peak": ea["peak"], "a_score": ea.get("score"),
                        "latency_ms": (eb["start"] - ea["start"]) * 1000.0,
                        "latency_peak_ms": (eb["start"] - ea["peak"]) * 1000.0})
        rows_b.append(row)
    matched_a = {(cand[k]["_set"], cand[k]["_i"]) for k in used}
    rows_a = [{"index": e.get("_i", i), "start": e["start"], "end": e["end"], "peak": e["peak"], "score": e.get("score"),
               "status": "matched" if ("accepted", e.get("_i", i)) in matched_a else "only_a"} for i, e in enumerate(a["events"])]
    lat = np.array([r["latency_ms"] for r in rows_b if r["status"] == "matched"])
    n_match = int(lat.size)
    n_rej = sum(r["status"].startswith("rejected") for r in rows_b)
    dur = max(float(a.get("t1", 0)) - float(a.get("t0", 0)), 0.0)
    return {"a": a["name"], "b": b["name"], "a_label": a["label"], "b_label": b["label"], "tol_ms": tol_ms,
            "a_channel": a.get("channel"), "b_channel": b.get("channel"),
            "n_a": len(a["events"]), "n_b": len(b["events"]), "n_matched": n_match, "n_b_on_rejected": n_rej,
            "n_only_a": len(a["events"]) - n_match, "n_only_b": len(b["events"]) - n_match - n_rej,
            "b_confirmed_frac": n_match / len(b["events"]) if b["events"] else None,     # of b's events, how many a agrees with
            "a_caught_frac": n_match / len(a["events"]) if a["events"] else None,        # of a's events, how many b caught
            "causal": bool(causal),
            "latency_ms": {"mean": float(lat.mean()), "sd": float(lat.std(ddof=1)) if n_match > 1 else 0.0,
                           "median": float(np.median(lat)), "min": float(lat.min()), "max": float(lat.max()),
                           "n": n_match, "values": np.round(lat, 4).tolist()} if n_match else None,
            "duration_s": dur, "rows_a": rows_a, "rows_b": rows_b}


def immobility_features(ts, channel, stats, move_hp_hz, move_smooth_s, **_):
    mz = motion_z(ts, move_hp_hz, move_smooth_s, stats)
    out = {c: ts.channel(c) for c in aux_channels(ts)}
    if mz is not None:
        out["movement z"] = np.interp(ts.t, mz[0], mz[1])
    return out


@detector("Immobility epochs (accelerometer)", [
    Param("move_z", "float", 2.0, min=0.0, max=20.0, step=0.1, help="still while robust-z movement stays below this"),
    Param("move_hp_hz", "float", 1.0, min=0.1, max=20.0, step=0.1, help="accelerometer high-pass (removes gravity)"),
    Param("move_smooth_s", "float", 0.5, min=0.05, max=5.0, step=0.05),
    Param("min_s", "float", 1.0, min=0.0, max=60.0, step=0.5, help="minimum epoch length"),
    Param("merge_s", "float", 0.5, min=0.0, max=10.0, step=0.1, help="bridge brief twitches shorter than this"),
], features=immobility_features)
def immobility(ts: TimeSeries, channel: str, move_z, move_hp_hz, move_smooth_s, min_s, merge_s, progress=None):
    """Epochs where the headstage accelerometer (all aux channels) shows no movement. The channel
    argument is ignored. Score is how far below the threshold the movement stayed."""
    mz = motion_z(ts, move_hp_hz, move_smooth_s)
    if mz is None:
        raise ValueError("no accelerometer (aux) channels in this recording")
    tm, zm, stats = mz
    # reuse the epoch finder on the negated signal: still == -z above -move_z
    events = _epochs(-zm, tm, -move_z, -move_z, min_s, 0.0, merge_s)
    for e in events:
        e["score"] = float(-e["score"])       # back to movement z at the calmest point
    return events, stats


# ---------------------------------------------------------------------------
# high theta from a multitaper spectrogram, thresholded at a percentile
# ---------------------------------------------------------------------------

def _decimate_to(ts: TimeSeries, channel: str, rate_hz: float) -> tuple[np.ndarray, float]:
    """One channel, anti-aliased and decimated to about ``rate_hz``."""
    fs = float(ts.fs)
    step = max(1, int(fs // rate_hz))
    y = ts.channel(channel)
    return (bandpass(y, fs, None, rate_hz / 2.5)[::step] if step > 1 else y), fs / step


def theta_mt_power(ts: TimeSeries, channel: str, low_hz: float, high_hz: float, win_s: float,
                   step_s: float, nw: float, smooth_s: float, rate_hz: float):
    """(bin centres, theta power, fs) from a multitaper spectrogram of one channel."""
    from .analysis import band_power, multitaper_spectrogram
    y, fsd = _decimate_to(ts, channel, rate_hz)
    tc, f, spec = multitaper_spectrogram(y, fsd, win_s, step_s, nw)
    p = band_power(f, spec, low_hz, high_hz)
    n = int(round(smooth_s / max(step_s, 1e-9))) if smooth_s else 0
    if n > 1 and p.size:
        p = moving_average(p, n)
    return tc + float(ts.t[0]), p, fsd


def theta_mt_features(ts, channel, stats, low_hz, high_hz, win_s, step_s, nw, smooth_s, rate_hz, **_):
    span = float(ts.t[-1] - ts.t[0])
    win = min(win_s, max(span / 3.0, 0.5))                 # the event window is usually shorter than 4 s
    tc, p, _ = theta_mt_power(ts, channel, low_hz, high_hz, win, min(step_s, win / 2), nw, 0.0, rate_hz)
    y = ts.channel(channel)
    power = np.interp(ts.t, tc, p / (stats["scale"] or 1.0)) if tc.size else np.zeros(ts.n)
    return {"raw": y, "theta band": bandpass(y, float(ts.fs), low_hz, high_hz),
            "theta power (x threshold)": power}


def theta_mt_thresholds(params: dict) -> dict:
    return {"theta power (x threshold)": {"high": 1.0}}


@detector("High theta (multitaper, 90th percentile)", [
    Param("low_hz", "float", 5.0, min=1.0, max=20.0, step=0.5, help="theta band"),
    Param("high_hz", "float", 12.0, min=2.0, max=30.0, step=0.5),
    Param("win_s", "float", 4.0, min=0.5, max=30.0, step=0.5, help="spectrogram window"),
    Param("step_s", "float", 2.0, min=0.1, max=10.0, step=0.1, help="and how far it moves each time"),
    Param("nw", "float", 3.0, min=1.0, max=10.0, step=0.5, help="time-bandwidth product; 2*nw-1 Slepian tapers are averaged"),
    Param("smooth_s", "float", 4.0, min=0.0, max=60.0, step=1.0, help="moving average over the power series"),
    Param("percentile", "float", 90.0, min=50.0, max=99.9, step=1.0,
          help="bins above this percentile of the session's own theta power are high theta"),
    Param("min_s", "float", 0.0, min=0.0, max=60.0, step=0.5, help="drop epochs shorter than this"),
    Param("merge_s", "float", 0.0, min=0.0, max=60.0, step=0.5, help="join epochs closer than this"),
    Param("rate_hz", "float", 1250.0, min=200.0, max=5000.0, step=50.0, help="the signal is decimated to about this first"),
], features=theta_mt_features, thresholds=theta_mt_thresholds)
def theta_mt(ts: TimeSeries, channel: str, layout=None, progress=None, **p):
    """High-theta periods from a multitaper spectrogram, thresholded at a percentile of the session.

    The LFP of one reference channel is taken at ~1250 Hz; a multitaper spectrogram (4 s window,
    2 s step, 2*nw-1 Slepian tapers) gives the theta (5-12 Hz) power of every bin; the series is
    smoothed and the bins above the **90th percentile of the whole session** are high theta.

    A percentile threshold labels a fixed share of the session whatever its state, which is the
    point: it asks "when was theta strongest *here*", not "was there theta at all". The reference
    should be a contact where theta is large and steady -- stratum oriens in the usual recipe;
    ``channel="auto"`` picks the contact with the most theta power, which is normally that one.
    """
    low_hz, high_hz = p["low_hz"], p["high_hz"]
    names = ts.headstage_channels() or list(ts.names)
    good = [n for n in names if n not in set((layout or {}).get("bad") or [])] or names
    if channel in ("", "auto", None) or channel not in ts.names:
        if progress:
            progress(0.1, f"choosing a reference from {len(good)} electrodes")
        power = []
        for n in good:
            y, fsd = _decimate_to(ts, n, p["rate_hz"])
            power.append(float(np.median(envelope(bandpass(y, fsd, low_hz, high_hz), fsd, 0.5))))
        channel = good[int(np.argmax(power))]
    if progress:
        progress(0.6, f"multitaper spectrogram on {channel}")
    tc, pw, _ = theta_mt_power(ts, channel, low_hz, high_hz, p["win_s"], p["step_s"], p["nw"],
                               p["smooth_s"], p["rate_hz"])
    if not pw.size:
        return [], {"centre": 0.0, "scale": 1.0, "robust": 0.0}, {"channel": channel}
    thr = float(np.percentile(pw, p["percentile"]))
    half = p["step_s"] / 2.0
    above = pw > thr
    runs: list[list[int]] = []
    for i in np.flatnonzero(above):
        if runs and tc[i] - tc[runs[-1][1]] <= p["step_s"] + p["merge_s"] + 1e-9:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    events = []
    for a, b in runs:
        start, end = float(tc[a] - half), float(tc[b] + half)
        if end - start < p["min_s"]:
            continue
        k = a + int(np.argmax(pw[a:b + 1]))
        events.append({"start": start, "end": end, "peak": float(tc[k]),
                       "score": float(pw[k] / (thr or 1.0)), "duration": end - start,
                       "power": float(pw[k])})
    if progress:
        progress(1.0, f"{len(events)} epochs")
    stats = {"centre": 0.0, "scale": thr or 1.0, "robust": 0.0}
    extra = {"channel": channel, "threshold_power": thr, "percentile": float(p["percentile"]),
             "thresholds": {"high": 1.0}, "n_bins": int(pw.size),
             "scan": scan_trace(tc, pw / (thr or 1.0))}
    return events, stats, extra
