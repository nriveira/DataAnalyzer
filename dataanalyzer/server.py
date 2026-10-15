"""Lightweight HTTP server exposing a TimeSeries and the analysis registry.

Pure standard library. Endpoints (all JSON unless noted):

    GET  /                      the viewer page
    GET  /api/meta              channel names, sample rate, analysis schemas
    GET  /api/analyze           ?analysis=&channel=&t0=&t1=&points=&<param>=...
    GET  /api/multi             same, but channels=all | kind:headstage | unit:uV | a,b,c  -> one lane per channel;
                                order=recorded (default) | depth (shallow to deep, needs a probe)
    GET  /api/probe             probe geometry joined with the recorded channels
    GET  /api/detections        detector schemas, saved detection results (plus the recording's TTL pulse sets), spike summary
    GET  /api/sections          included time ranges of the open recording, and the headstage amplitude over the session
    POST /api/sections          {"include": [[t0, t1], ...]} save (empty = whole recording) | {"exclude": [[t0, t1], ...]}
                                cut those stretches out of what is included | {"clear": true} | {"suggest": true, ...}
                                event averages, triggered averages, compare and the Events tab use only events inside them
    GET  /api/swr/scan          ?det=  one SWR run: whole-session scan trace, events + features, depth profile, thresholds
    GET  /api/swr/sessions      every SWR run saved under the root, per recording (read from the result files)
    POST /api/swr/batch         {"name","params","tag","channel","only","min_s","stream"} run an SWR detector on every recording
    GET  /api/compare           ?a=swr&b=pulses_...&tol_ms=50&rejected=1&causal=0  match event set b against a: counts, per-event status,
                                and the latency b - a (mean, SD, median, range); causal=1 only matches a b event that follows its a event
    GET  /api/triggered         ?det=&channel=&pre_ms=250&post_ms=250&align=start|end|peak&low=&high=&max_trials=100
                                &norm=none|baseline|baseline_z|session_z&base_ms=100
                                event-triggered average of one channel: mean, SD and a sample of the single trials
    GET  /api/event             ... &align=peak|start|end  centre the event window on that time; the payload carries "extras",
                                the recording's non-electrode channels (a detector's recorded feature and thresholds)
    POST /api/detect            {"name","channel","params"[,"tag"]} run a detector as a job, saved as <name>__<tag> when tagged;
                                {"name","remove":true} deletes a saved result by its saved name
    POST /api/spikes/run        {"channels","threshold_sd","band","refractory_ms","polarity"} spike detection job
    GET  /api/spikes            ?t0=&t1=&max=  spike times per channel in the window
    POST /api/sort              {"threshold_sd","band","scheme":"auto|1|2","whiten"} sort this stream with MountainSort5 (a job)
    GET  /api/units             sorted units: metrics, templates, a unit x time raster
    GET  /api/unit              ?unit=  one unit: template on the channels that carry it, ISI histogram
    GET  /api/spikes/overview   counts, rates and a channel x time raster for the whole recording
    GET  /api/spikes/channel    ?channel=&pre_ms=1&post_ms=2&max_n=300  mean waveform + SD, single waveforms,
                                amplitude and ISI histograms for one channel
    GET  /api/phase             ?det=theta&ref=auto&low_hz=5&high_hz=10&source=spikes|units&bins=18&align=peak|trough  phase histogram
                                and circular statistics per channel / unit, over the spikes inside that event set's epochs
    GET  /api/phase/epoch       ?det=&index=&...  one epoch: reference LFP, its band-passed cycle and phase, spikes in it
    GET  /api/phase/cycle       ?det=&align=peak|trough&pre_ms=&post_ms=  the reference channel averaged on every cycle
                                peak (or trough) inside the epochs, with each group's spike rate on the same axis
    GET  /api/theta/coherence   ?det=theta&ref=auto&low_hz=5&high_hz=10&min_plv=0.5  theta phase and phase-locking
                                value of every electrode against one contact, by depth, with the reversal (layer landmark)
    GET  /api/theta/activity    ?ref=&low_hz=&high_hz=&delta_low=1&delta_high=4&smooth_s=2  theta/delta ratio over the
                                whole session (activity level), with the accelerometer beside it when there is one
    GET  /api/jobs              progress of running / finished jobs
    GET  /api/event             ?det=swr&index=3|avg&pre_ms=&post_ms=&lfp_low=&lfp_high=&smooth=  LFP, CSD, features
    GET  /api/layers            saved hippocampal layer estimate (or null)
    GET  /api/channels          channel quality stats, automatic flags, manual overrides, effective bad list
    POST /api/features          {"det": "swr"} compute per-event features as a job
    GET  /api/features          ?det=&set=all|accepted|rejected  feature table;  /api/features.csv?det=  CSV download
    POST /api/filter            {"det", "rules": {feature: {min, max}}} save failing events as rejected; {"det", "restore": true}
    POST /api/pulses            {"merge_ms": 1} rebuild the recorded TTL pulse sets with pulses that nearly touch merged
                                into one event (a stimulator firing on both edges records each stimulus twice)
    POST /api/channels          {"check": true} recompute; {"bad": [...]} / {"good": [...]} manual overrides; {"reset": true}
    POST /api/layers            {"det": "swr", ...params} estimate layers from that detection's CSD; {"remove": true}
    GET  /api/events            ?t0=&t1=&points=   recorded event times (TTL, messages)
    GET  /api/live?since=N      traces published by running code, if newer than N
    POST /api/live              publish a trace {"name","x","y","kind"} from another process
    GET  /api/sessions          Open Ephys sessions and data files found under the data root
    POST /api/pick              show the native folder dialog, set the root, open what it holds
    POST /api/root              {"path": ...} set the root without a dialog (same auto-open)
    POST /api/open              {"path": ..., "stream": ...} switch to a session under the root
    POST /api/reload            re-import plugin analysis files and drop the result cache

Traces are decimated with min/max buckets so the browser never receives more
than ``points`` samples per trace while peaks are preserved.

Use :func:`serve` to block, or :func:`serve_in_background` from an analysis
script and publish intermediate results with ``server.store.publish``.
"""

from __future__ import annotations

import json
import math
import os
import threading
import traceback
import uuid
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import analysis as _an
from . import channels as _chan
from . import csd as _csd
from . import detect as _det
from . import features as _feat
from . import phase as _phase
from . import sections as _sec
from . import sorting as _srt
from . import spikes as _spk
from . import dentate as _dentate  # noqa: F401  (registers the dentate-spike detector)
from . import stim as _stim  # noqa: F401  (registers the stimulation-artefact detector)
from . import swr as _swr  # noqa: F401  (registers the laminar SWR detector)
from . import triggered as _trig
from .dialog import available as dialog_available, pick_folder, remember_root
from .io import TimeSeries, find_extra_channels, load
from .probe import Probe, auto_probe, load_probe

VIEWER_HTML = Path(__file__).with_name("viewer") / "index.html"


def _code_mtime() -> float:
    """Newest modification time of the package's Python files."""
    return max((p.stat().st_mtime for p in Path(__file__).parent.glob("*.py")), default=0.0)


CODE_MTIME_AT_START = _code_mtime()   # the viewer page is re-read on every load, the Python code is not
DEFAULT_POINTS = 3000
IMAGE_MAX_COLS = 1200
IMAGE_MAX_ROWS = 160
WINDOW_PAD_S = 1.0        # seconds of context computed either side of the visible window
WINDOW_PAD_FRAC = 0.1     # or this fraction of the window, whichever is larger
CACHE_SIZE = 256
MULTI_MAX_SAMPLES = 3_000_000   # per-channel window cap for non-raw analyses across many channels
DATA_SUFFIXES = (".csv", ".tsv", ".npy", ".npz")
SCAN_MAX_DEPTH = 6


# ---------------------------------------------------------------------------
# Decimation and JSON helpers
# ---------------------------------------------------------------------------

def decimate_minmax(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the min and max of each bucket so spikes survive downsampling. NaN gaps are preserved."""
    n = x.size
    if n <= max_points or max_points < 4:
        return x, y
    stride = int(math.ceil(n / (max_points // 2)))
    n_buckets = int(math.ceil(n / stride))
    pad = stride * n_buckets - n                       # always < stride, so no bucket is empty
    ypad = np.concatenate([y, np.full(pad, np.nan)]).reshape(n_buckets, stride)
    nan = np.isnan(ypad)
    lo = np.argmin(np.where(nan, np.inf, ypad), axis=1) + np.arange(n_buckets) * stride
    hi = np.argmax(np.where(nan, -np.inf, ypad), axis=1) + np.arange(n_buckets) * stride
    idx = np.unique(np.concatenate([lo, hi]))
    idx = idx[idx < n]
    return x[idx], y[idx]


def _jsonable(arr: np.ndarray | None) -> list | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    out = a.tolist()
    if not np.all(np.isfinite(a)):
        out = [v if math.isfinite(v) else None for v in out]
    return out


def result_to_json(res: _an.Result, t0: float | None, t1: float | None, points: int) -> dict:
    traces = []
    for tr in res.traces:
        x, y = np.asarray(tr.x, dtype=float), tr.y
        if res.domain == "time" and (t0 is not None or t1 is not None):
            lo = 0 if t0 is None else int(np.searchsorted(x, t0, "left"))
            hi = x.size if t1 is None else int(np.searchsorted(x, t1, "right"))
            x = x[lo:hi]
            y = None if y is None else np.asarray(y, dtype=float)[lo:hi]
        if tr.kind == "image":
            z = np.asarray(tr.z, dtype=float)[:, lo:hi] if (res.domain == "time" and (t0 is not None or t1 is not None)) else np.asarray(tr.z, dtype=float)
            cstep = int(math.ceil(x.size / min(points, IMAGE_MAX_COLS))) if x.size > min(points, IMAGE_MAX_COLS) else 1
            f = np.asarray(tr.f, dtype=float)
            rstep = int(math.ceil(f.size / IMAGE_MAX_ROWS)) if f.size > IMAGE_MAX_ROWS else 1
            diverging = tr.cmap == "diverging"
            if cstep > 1:   # block-max keeps short bursts visible; signed maps take the block mean
                nc = z.shape[1] // cstep * cstep
                blocks = z[:, :nc].reshape(z.shape[0], -1, cstep)
                zc = blocks.mean(axis=2) if diverging else blocks.max(axis=2)
                x = x[cstep // 2:nc:cstep]
                z = zc[:, :x.size]
            labels = tr.row_labels
            if rstep > 1:
                z, f = z[::rstep], f[::rstep]
                labels = labels[::rstep] if labels else None
            finite = z[np.isfinite(z)]
            if diverging:
                v = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
                zmin, zmax = -v, v
            else:
                zmin, zmax = (float(np.percentile(finite, 1)), float(np.percentile(finite, 99))) if finite.size else (0.0, 1.0)
            traces.append({"name": tr.name, "kind": "image", "x": _jsonable(x), "f": _jsonable(f),
                           "z": [_jsonable(row) for row in np.round(z, 3)], "zmin": zmin, "zmax": zmax if zmax > zmin else zmin + 1,
                           "zlabel": tr.zlabel, "log_f": bool(tr.log_f), "row_labels": labels, "cmap": tr.cmap})
            continue
        if y is not None:
            x, y = decimate_minmax(x, np.asarray(y, dtype=float), points)
        elif x.size > points:
            x = x[:: int(math.ceil(x.size / points))]
        traces.append({"name": tr.name, "kind": tr.kind, "x": _jsonable(x), "y": _jsonable(y)})
    return {"domain": res.domain, "xlabel": res.xlabel, "ylabel": res.ylabel, "traces": traces}


# ---------------------------------------------------------------------------
# Live store: traces pushed by running analysis code
# ---------------------------------------------------------------------------

class LiveStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._traces: dict[str, dict] = {}
        self.version = 0

    def publish(self, name: str, x, y=None, kind: str = "line", **meta) -> int:
        """Add or replace a named trace. ``x`` is time in seconds; ``y`` may be None for events."""
        x = np.asarray(x, dtype=float).reshape(-1)
        y = None if y is None else np.asarray(y, dtype=float).reshape(-1)
        if y is not None and y.size != x.size:
            raise ValueError("x and y must have the same length")
        with self._lock:
            self.version += 1
            self._traces[name] = {"name": name, "kind": kind, "x": x, "y": y,
                                  "version": self.version, "meta": meta}
            return self.version

    def remove(self, name: str) -> None:
        with self._lock:
            if self._traces.pop(name, None) is not None:
                self.version += 1

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()
            self.version += 1

    def snapshot(self, since: int, t0: float | None, t1: float | None, points: int) -> dict:
        with self._lock:
            if self.version <= since:
                return {"version": self.version, "changed": False, "traces": []}
            res = _an.Result([_an.Trace(d["name"], d["x"], d["y"], d["kind"]) for d in self._traces.values()])
            metas = [d["meta"] for d in self._traces.values()]
        payload = result_to_json(res, t0, t1, points)
        for tr, m in zip(payload["traces"], metas):
            tr["meta"] = m
        return {"version": self.version, "changed": True, "traces": payload["traces"]}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DataServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], ts: TimeSeries | None, store: LiveStore | None = None,
                 root: str | Path | None = None, plugins: list[str | Path] | None = None,
                 probe: str | Path | Probe | None = None) -> None:
        super().__init__(address, _Handler)
        self.ts = ts
        self.probe_spec = probe                      # explicit choice, applied to every session
        self.probe: Probe | None = None
        self.detections: dict[str, dict] = {}
        self.spikes: dict | None = None
        self.jobs: dict[str, dict] = {}
        self._apply_probe()
        self._load_saved()
        self.store = store or LiveStore()
        self.root = Path(root).resolve() if root else None
        self.plugins = [Path(p).resolve() for p in (plugins or [])]
        self.session: dict | None = None
        self._cache_lock = threading.Lock()
        self._cache: dict[tuple, _an.Result] = {}
        self._sessions: list[dict] | None = None
        self.reload_plugins()

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"

    def set_data(self, ts: TimeSeries, session: dict | None = None) -> None:
        with self._cache_lock:
            self.ts = ts
            self.session = session
            self._cache.clear()
        self._apply_probe()
        self._load_saved()

    def _load_saved(self) -> None:
        """Detection and spike results saved beside the recording by earlier runs."""
        self.detections = _det.load_results(self.ts) if self.ts is not None else {}
        if self.ts is not None:             # detections made during acquisition (Ripple Detector TTL), rebuilt from the recording
            self.detections = {k: v for k, v in self.detections.items() if not k.startswith(_det.ONLINE_PREFIX)}
            self.detections.update(_det.online_detections(self.ts))
        self.spikes = _spk.load(self.ts) if self.ts is not None else None
        self.sorting = _srt.load(self.ts) if self.ts is not None else None
        self.layers = _csd.load_layers(self.ts) if self.ts is not None else None
        self.channels = _chan.load(self.ts) if self.ts is not None else None
        self.sections = _sec.load(self.ts) if self.ts is not None else []
        self._extras = None                  # (TimeSeries, names) of a processor's recorded channels, found on demand
        self._phase = None                   # (key, band_phase result): the reference cycle, kept because it costs a pass over the recording
        self._activity = None                # (key, theta/delta ratio over the session)
        self._coherence = None               # (key, theta phase across the probe)
        self.sections_rev = getattr(self, "sections_rev", 0) + 1

    # -- spike phase within detected epochs (Theta tab) --------------------------------
    def band_phase(self, channel: str, band: tuple[float, float]) -> dict:
        """The reference channel's instantaneous phase, cached: one decimated pass over the
        recording per (channel, band)."""
        key = (id(self.ts), channel, round(band[0], 3), round(band[1], 3))
        if not self._phase or self._phase[0] != key:
            self._phase = (key, _phase.band_phase(self.ts, channel, band))
        return self._phase[1]

    def _phase_inputs(self, q: dict):
        """(phase signal, epoch ranges, spike source, the detection) for a phase request."""
        name = q.get("det") or ""
        if name not in self.detections:
            raise KeyError(f"no event set {name!r}; run a detector first")
        r = self.detections[name]
        evs = r["events"] if q.get("set", "accepted") != "rejected" else r.get("rejected", [])
        ranges = _phase.event_ranges(evs, self.sections or None)
        ref = q.get("ref") or ""
        if ref in ("", "auto"):
            ref = r.get("channel") if r.get("channel") in self.ts.names else (self.ts.headstage_channels() or self.ts.names)[0]
        if ref not in self.ts.names:
            raise KeyError(f"no channel {ref!r}")
        band = (float(q.get("low_hz", _phase.DEFAULT_BAND[0])), float(q.get("high_hz", _phase.DEFAULT_BAND[1])))
        try:                      # the epochs, the scan and the cycle average do not need spikes
            src = _phase.spike_source(self.ts, q.get("source", "spikes"))
        except FileNotFoundError as e:
            src = {"times": np.empty(0), "group": np.empty(0, dtype=int), "names": [], "labels": [],
                   "kind": q.get("source", "spikes"), "note": str(e)}
        ph = _phase.rotate(self.band_phase(ref, band), q.get("align", "peak"))
        return ph, ranges, src, r

    def phase_json(self, q: dict) -> dict:
        ph, ranges, src, r = self._phase_inputs(q)
        out = _phase.phase_table(ph, ranges, src, bins=int(q.get("bins", _phase.BINS)))
        out.update({"spikes_note": src.get("note"),
                    "det": r["name"], "det_label": r["label"], "det_channel": r.get("channel"),
                    "epochs": np.round(ranges, 4).tolist(), "scan": r.get("scan"),
                    "scan_thresholds": r.get("thresholds"), "source": src["kind"],
                    "t0": float(self.ts.t[0]), "t1": float(self.ts.t[-1]),
                    "restricted": bool(self.sections)})
        return out

    def phase_cycle(self, q: dict) -> dict:
        """The reference channel averaged on every cycle peak (or trough) inside the epochs: the
        raw average over whole epochs cancels, because they start at arbitrary phases."""
        ph, ranges, src, r = self._phase_inputs(q)
        num = lambda k: float(q[k]) if q.get(k) not in (None, "") else None
        out = _phase.cycle_average(ph, ranges, src, align=q.get("align", "peak"),
                                   pre_ms=num("pre_ms"), post_ms=num("post_ms"))
        out["det"] = r["name"]
        return out

    def phase_coherence(self, q: dict) -> dict:
        """Theta phase across the probe against one contact, as a depth profile: the reversal is a
        layer landmark. Cached: it band-passes every electrode over the whole recording."""
        name = q.get("det") or ""
        r = self.detections.get(name) or {}
        evs = r.get("events") or []
        ranges = _phase.event_ranges(evs, self.sections or None) if evs else np.empty((0, 2))
        band = (float(q.get("low_hz", _phase.DEFAULT_BAND[0])), float(q.get("high_hz", _phase.DEFAULT_BAND[1])))
        ref = q.get("ref") or "auto"
        min_plv = float(q.get("min_plv", 0.5))
        key = (id(self.ts), name, band, ref, min_plv, self.sections_rev, len(ranges))
        if not getattr(self, "_coherence", None) or self._coherence[0] != key:
            out = _phase.coherence_profile(self.ts, ranges, band, reference=None if ref == "auto" else ref,
                                           layout=self.probe_layout(), min_plv=min_plv)
            out.update({"det": name, "det_label": r.get("label"), "n_epochs": int(len(ranges)),
                        "layers": self.layers})
            self._coherence = (key, out)
        return self._coherence[1]

    def activity_json(self, q: dict) -> dict:
        """The theta / delta ratio over the session (cached: it costs a pass over the recording)."""
        ref = q.get("ref") or ""
        if ref in ("", "auto"):
            name = q.get("det") or ""
            r = self.detections.get(name) or {}
            ref = r.get("channel") if r.get("channel") in self.ts.names else (self.ts.headstage_channels() or self.ts.names)[0]
        if ref not in self.ts.names:
            raise KeyError(f"no channel {ref!r}")
        band = (float(q.get("low_hz", _phase.DEFAULT_BAND[0])), float(q.get("high_hz", _phase.DEFAULT_BAND[1])))
        delta = (float(q.get("delta_low", 1.0)), float(q.get("delta_high", 4.0)))
        smooth = float(q.get("smooth_s", 2.0))
        key = (id(self.ts), ref, band, delta, smooth)
        if not getattr(self, "_activity", None) or self._activity[0] != key:
            self._activity = (key, _phase.activity(self.ts, ref, band, delta, smooth))
        return self._activity[1]

    def phase_epoch(self, q: dict) -> dict:
        ph, ranges, src, r = self._phase_inputs(q)
        out = _phase.epoch_view(ph, ranges, int(q.get("index", 0)), src,
                                pad_s=float(q.get("pad_s", 0.25)))
        out["labels"] = src["labels"]
        return out

    # -- analysis time ranges (Events tab) ---------------------------------------------
    def restrict(self, result: dict) -> dict:
        """``result`` with only the events inside the included time ranges (unchanged when there are
        none). Each kept event carries ``_i``, its index in the full set."""
        if not self.sections:
            return result
        pick = lambda evs: [dict(e, _i=i) for i, e in enumerate(evs) if _sec.event_in(self.sections, e)]
        ev, rej = pick(result["events"]), pick(result.get("rejected", []))
        return {**result, "events": ev, "rejected": rej, "n_excluded": len(result["events"]) - len(ev),
                "n_rejected_excluded": len(result.get("rejected", [])) - len(rej)}

    def sections_json(self) -> dict:
        ts = self.ts
        return {"include": self.sections, "t0": float(ts.t[0]), "t1": float(ts.t[-1]), "activity": _sec.activity(ts),
                "included_s": float(sum(b - a for a, b in self.sections)) if self.sections else float(ts.t[-1] - ts.t[0])}

    def update_sections(self, body: dict) -> dict:
        if body.get("suggest"):
            return {"suggested": _sec.suggest(self.ts, float(body.get("flat_frac", 0.2)), float(body.get("loud_factor", 8.0)),
                                               float(body.get("bridge_s", 1.0)), float(body.get("min_s", 5.0)),
                                               float(body.get("min_bad_s", 0.5)))}
        if body.get("exclude"):          # take a stretch out of what is currently included
            ranges = _sec.subtract(self.sections, body["exclude"], float(self.ts.t[0]), float(self.ts.t[-1]))
        else:
            ranges = [] if body.get("clear") else body.get("include", [])
        self.sections = _sec.save(self.ts, ranges)
        self.sections_rev += 1
        return self.sections_json()

    # -- jobs (long-running detections) ------------------------------------------
    def start_job(self, kind: str, fn) -> str:
        """Run ``fn(progress)`` on a thread; poll /api/jobs. ``progress(frac, message)`` updates status."""
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "kind": kind, "status": "running", "progress": 0.0, "message": "starting",
               "started": time.time()}
        self.jobs[jid] = job

        def progress(frac, msg=""):
            job["progress"] = float(frac)
            job["message"] = msg

        def run():
            try:
                job["result"] = fn(progress)
                job["status"] = "done"
                job["progress"] = 1.0
            except Exception as e:  # report to the viewer rather than dying silently
                job["status"] = "error"
                job["message"] = f"{type(e).__name__}: {e}"
                job["trace"] = traceback.format_exc()
            job["finished"] = time.time()

        threading.Thread(target=run, daemon=True, name=f"job-{kind}").start()
        return jid

    def jobs_json(self) -> dict:
        now = time.time()
        for jid in [j for j, job in self.jobs.items() if job.get("finished") and now - job["finished"] > 300]:
            del self.jobs[jid]
        return {"jobs": [{k: v for k, v in job.items() if k != "trace"} for job in self.jobs.values()]}   # results are small summaries

    def farthest_channel(self, channel: str) -> str | None:
        """The headstage channel farthest from ``channel``: by probe geometry when available, else by
        recorded position (which is depth order for a mapped recording)."""
        lay = self.probe_layout()
        if lay and lay.get("contacts"):
            pos = {c["name"]: (c["x_um"], c["y_um"], c["shank"]) for c in lay["contacts"] if c["name"] not in set(lay.get("bad") or [])}
            if channel in pos:
                x0, y0, sh0 = pos[channel]
                same = [(n, (x - x0) ** 2 + (y - y0) ** 2) for n, (x, y, sh) in pos.items() if sh == sh0 and n != channel]
                if same:
                    return max(same, key=lambda p: p[1])[0]
        names = self.ts.headstage_channels()
        if channel not in names or len(names) < 2:
            return None
        i = names.index(channel)
        return max((n for n in names if n != channel), key=lambda n: abs(names.index(n) - i))

    def resolve_channel_params(self, det: _det.Detector, channel: str, params: dict) -> dict:
        out = dict(params)
        for p in det.params:
            if p.kind != "channel":
                continue
            v = str(out.get(p.name, p.default)).strip()
            if v == "auto":
                v = self.farthest_channel(channel) or "none"
            elif v not in ("", "none") and v not in self.ts.names:
                raise KeyError(f"{p.name}: unknown channel {v!r}")
            out[p.name] = v
        return out

    def run_detection(self, name: str, channel: str, params: dict, tag: str | None = None) -> str:
        if name not in _det.DETECTORS:
            raise KeyError(f"unknown detector {name!r}")
        det = _det.DETECTORS[name]
        if channel == "auto" and not det.accepts("layout"):
            raise ValueError(f"{det.label} needs a channel; only the all-channel detectors can choose one")
        if channel != "auto" and channel not in self.ts.names:
            raise KeyError(f"unknown channel {channel!r}")
        ts, layout = self.ts, self.probe_layout()
        params = self.resolve_channel_params(det, channel, params)

        def fn(progress):
            res = _det.run(ts, name, channel, params, progress, layout=layout, tag=tag)
            if det.category == "swr":            # the SWR tab summarises what the events add; the Events tab filters on it
                progress(0.9, "event features")
                _feat.event_features(ts, layout, res, self.layers, self.spikes)
                res["path"] = str(_det.save_result(ts, res))
            if self.ts is ts:                    # the viewer may have switched recordings meanwhile: the file is saved beside
                self.detections[res["name"]] = res   # the right one, but only the recording it ran on lists it
            return {"name": res["name"], "n_events": res["n_events"], "path": res.get("path")}
        return self.start_job(f"detect:{name}", fn)

    def run_spikes(self, body: dict) -> str:
        ts = self.ts
        channels = body.get("channels")
        if isinstance(channels, str):
            channels = self.resolve_channels(channels)
        band = tuple(float(v) for v in body.get("band", (300.0, 6000.0)))

        def fn(progress):
            res = _spk.detect_spikes(ts, channels, float(body.get("threshold_sd", 4.5)), band,
                                     float(body.get("refractory_ms", 1.0)), body.get("polarity", "neg"),
                                     float(body.get("chunk_s", 20.0)), progress, backend=body.get("backend", "numpy"))
            path = _spk.save(ts, res)
            self.spikes = res
            return {"n_spikes": int(res["times"].size), "path": str(path)}
        return self.start_job("spikes", fn)

    def run_sorting(self, body: dict) -> str:
        """MountainSort5 on this stream's electrodes, saved beside the recording (a job)."""
        ok, msg = _srt.available()
        if not ok:
            raise ValueError(msg)
        ts, layout = self.ts, self.probe_layout()
        band = tuple(float(v) for v in body.get("band", (300.0, 6000.0)))
        channels = self.resolve_channels(body["channels"]) if isinstance(body.get("channels"), str) else body.get("channels")

        def fn(progress):
            res = _srt.sort_mountainsort(ts, layout, channels, band, float(body.get("threshold_sd", 5.5)),
                                         str(body.get("scheme", "auto")), body.get("whiten", True) not in (False, "0", "false"),
                                         float(body.get("train_s", 300.0)), progress=progress)
            path = _srt.save(ts, res)
            if self.ts is ts:
                self.sorting = res
            return {"units": len(res["unit_ids"]), "spikes": res["meta"]["n_spikes"], "scheme": res["meta"]["scheme"],
                    "path": str(path)}
        return self.start_job("sorting (mountainsort5)", fn)

    def run_features(self, name: str) -> str:
        if name not in self.detections:
            raise KeyError(f"no saved detection {name!r}")
        ts, result, layout, layers, spikes = self.ts, self.detections[name], self.probe_layout(), self.layers, self.spikes

        def fn(progress):
            _feat.event_features(ts, layout, result, layers, spikes, progress)
            if not _det.is_online(result):      # online detections are rebuilt from the recording, never saved
                _det.save_result(ts, result)
            return {"name": name, "n": result["feature_meta"]["n"]}
        return self.start_job(f"features:{name}", fn)

    def compare(self, q: dict[str, str]) -> dict:
        """Match event set ``b`` (default: the first recorded pulse set) against ``a`` (default: swr)."""
        online = [k for k, r in self.detections.items() if _det.is_online(r)]
        a = q.get("a") or ("swr" if "swr" in self.detections else "")
        b = q.get("b") or (online[0] if online else "")
        for k in (a, b):
            if k not in self.detections:
                raise KeyError(f"no detection {k!r}; run or pick one of {sorted(self.detections)}")
        out = _det.compare(self.restrict(self.detections[a]), self.restrict(self.detections[b]), float(q.get("tol_ms", 50.0)),
                           q.get("rejected", "1") not in ("0", "false"), causal=q.get("causal") in ("1", "true"))
        out["restricted"] = bool(self.sections)
        return out

    # -- SWR tab ------------------------------------------------------------------
    def swr_scan(self, name: str) -> dict:
        """One SWR run for the SWR tab: its whole-session scan trace, events with the features that
        say what they add, the depth profile the ripple channel was chosen from, and thresholds."""
        if name not in self.detections:
            raise KeyError(f"no saved result {name!r}")
        r = self.detections[name]
        det = _det.detector_of(r)
        keep = ("start", "end", "peak", "score", "duration", "reason", "sw_score", "global_score", "line_frac", "move_score", "ref_score")
        ev = lambda e: {**{k: e[k] for k in keep if k in e},
                        **{k: (e.get("features") or {}).get(k) for k in ("burst_hz", "burst_frac", "peak_hz")}}
        lay = self.probe_layout() or {}
        return {"name": name, "label": r["label"], "detector": r.get("detector") or name, "channel": r["channel"],
                "params": r["params"], "created": r.get("created"), "t0": r.get("t0"), "t1": r.get("t1"),
                "thresholds": det.thresholds(r["params"]) if det and det.thresholds else {},
                "scan": r.get("scan"), "events": [ev(e) for e in r["events"]], "rejected": [ev(e) for e in r.get("rejected", [])],
                "has_features": any("features" in e for e in r["events"] + r.get("rejected", [])),
                "ripple_profile": r.get("ripple_profile"), "pool_channels": r.get("pool_channels"),
                "sw_deep_channels": r.get("sw_deep_channels"), "sw_shallow_channels": r.get("sw_shallow_channels"),
                "layer_confidence": r.get("layer_confidence"), "layer_confident": r.get("layer_confident"),
                "bad": lay.get("bad") or [], "counts": {k: r.get(k) for k in r if k.startswith("n_")}}

    def swr_sessions(self) -> dict:
        """Every SWR run saved under the data root, per recording, read from the result files
        (no recording is opened), for the SWR tab's all-sessions table."""
        rows = []
        for s in self.list_sessions():
            if s["kind"] != "openephys":
                continue
            rec = (self.root / s["path"]).parent
            runs = []
            for p in sorted((rec / _det.RESULTS_DIRNAME).glob("**/events/*.json")):
                try:
                    r = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue
                det = _det.detector_of(r)
                if r.get("category", det.category if det else "") != "swr":
                    continue
                dur = max(float(r.get("t1", 0)) - float(r.get("t0", 0)), 1e-9)
                sub = p.parent.parent.name
                runs.append({"name": r["name"], "label": r.get("label"), "stream": None if sub == _det.RESULTS_DIRNAME else sub,
                             "n_events": r.get("n_events"), "n_rejected": r.get("n_rejected"), "rate_per_min": 60.0 * r.get("n_events", 0) / dur,
                             "channel": r.get("channel"), "layer_confidence": r.get("layer_confidence"),
                             "layer_confident": r.get("layer_confident"), "n_with_sharp_wave": r.get("n_with_sharp_wave"),
                             "band": [r["params"].get("low_hz"), r["params"].get("high_hz")], "created": r.get("created")})
            rows.append({"path": s["path"], "label": s["label"], "streams": s["streams"], "duration_s": s.get("duration_s"), "runs": runs,
                         "open": bool(self.session and self.session.get("path") == s["path"])})
        return {"root": str(self.root) if self.root else None, "sessions": rows}

    def run_swr_batch(self, body: dict) -> str:
        """Run an SWR detector on every recording under the root (optionally only paths containing
        ``only`` and at least ``min_s`` long), each on its broadband stream (or ``stream``), with
        features, saved beside each recording. Runs as one job; the open session is refreshed."""
        name = body.get("name", "swr_laminar")
        det = _det.DETECTORS.get(name)
        if det is None or det.category != "swr":
            raise KeyError(f"{name!r} is not an SWR detector")
        params, tag, channel = body.get("params") or {}, body.get("tag"), body.get("channel") or "auto"
        only, min_s, want = body.get("only") or "", float(body.get("min_s") or 0), (body.get("stream") or "").lower()
        todo = [s for s in self.list_sessions(rescan=True) if s["kind"] == "openephys" and only in s["path"]
                and (s.get("duration_s") is None or s["duration_s"] >= min_s)]
        root = self.root

        def pick(streams):
            if want:
                return next((x for x in streams if want in x.lower()), None)
            return next((x for x in streams if "broadband" in x.lower()), streams[0] if len(streams) > 1 else None)

        def fn(progress):
            done = []
            for i, s in enumerate(todo):
                progress(i / max(len(todo), 1), f"{s['label']} ({i + 1}/{len(todo)})")
                try:
                    ts = load(root / s["path"], stream=pick(s["streams"]))
                    probe = self.probe_spec if isinstance(self.probe_spec, Probe) else (load_probe(self.probe_spec) if self.probe_spec
                             else auto_probe(ts.source, len(ts.headstage_channels()) or ts.n_channels))
                    lay = probe.layout(ts.names, ts.channel_info, ts.units) if probe else {"contacts": []}
                    lay["bad"] = _chan.effective_bad(_chan.load(ts))
                    res = _det.run(ts, name, channel if channel == "auto" or channel in ts.names else "auto", params,
                                   save=False, layout=lay, tag=tag)
                    _feat.event_features(ts, lay, res, _csd.load_layers(ts), _spk.load(ts))
                    res["path"] = str(_det.save_result(ts, res))
                    if self.session and self.session.get("path") == s["path"] and self.ts is not None and self.ts.stream == ts.stream:
                        self.detections[res["name"]] = res
                    done.append({"session": s["label"], "name": res["name"], "n_events": res["n_events"], "channel": res["channel"]})
                except Exception as e:  # one bad recording does not stop the batch
                    done.append({"session": s["label"], "error": f"{type(e).__name__}: {e}"})
            return {"sessions": len(todo), "ok": sum("error" not in d for d in done), "errors": [d for d in done if "error" in d][:5]}
        return self.start_job(f"swr batch ({len(todo)} recordings)", fn)

    def detections_json(self) -> dict:
        results = {}
        for k, r in self.detections.items():
            results[k] = {kk: vv for kk, vv in r.items() if kk not in ("events", "rejected", "scan")}
            results[k].setdefault("category", (_det.detector_of(r).category if _det.detector_of(r) else "events"))
            results[k]["has_features"] = any("features" in e for e in r["events"]) or any("features" in e for e in r.get("rejected", []))
        sorter_ok, sorter_msg = _srt.available()
        return {"detectors": _det.schemas(), "results": results,
                "pulses": getattr(self, "pulses", {"merge_ms": 0.0}),
                "spikes": _spk.summary(self.spikes) if self.spikes else None,
                "sorting": {"n_units": len(self.sorting["unit_ids"]), "meta": {k: v for k, v in self.sorting["meta"].items() if k != "units"}}
                           if self.sorting else None,
                "sorter": {"available": sorter_ok, "message": sorter_msg}}

    def spikes_json(self, t0: float | None, t1: float | None, max_per_channel: int) -> dict:
        if not self.spikes:
            return {"channels": {}, "n": 0}
        t, ch = self.spikes["times"], self.spikes["channels"]
        lo = 0 if t0 is None else int(np.searchsorted(t, t0, "left"))
        hi = t.size if t1 is None else int(np.searchsorted(t, t1, "right"))
        out = {}
        for j, name in enumerate(self.spikes["channel_names"]):
            m = ch[lo:hi] == j
            times = t[lo:hi][m]
            if times.size > max_per_channel:
                times = times[:: int(math.ceil(times.size / max_per_channel))]
            if times.size:
                out[name] = _jsonable(times)
        return {"channels": out, "n": int(hi - lo)}

    def estimate_layers(self, body: dict) -> dict:
        name = body.get("det", "swr")
        if name not in self.detections:
            raise KeyError(f"no saved detection {name!r}; run it first")
        est = _csd.layer_estimate(self.ts, self.probe_layout(), self.detections[name],
                                  float(body.get("pre_ms", 250.0)), float(body.get("post_ms", 100.0)),
                                  (float(body.get("ripple_low", 120.0)), float(body.get("ripple_high", 200.0))),
                                  float(body.get("win_ms", 10.0)), smooth=body.get("smooth", True) not in (False, "0", "false"))
        _csd.save_layers(self.ts, est)
        self.layers = est
        return est

    def event_view(self, q: dict[str, str]) -> dict:
        name = q.get("det", "swr")
        if name not in self.detections:
            raise KeyError(f"no saved detection {name!r}; run it first")
        result = self.detections[name]
        det = _det.ONLINE if _det.is_online(result) else _det.detector_of(result)
        if det is None:
            raise KeyError(f"detector {name!r} is not registered")
        idx_raw = q.get("index", "0")
        index = None if idx_raw in ("avg", "average", "mean", "") else int(idx_raw)
        if q.get("set") == "rejected":
            result = {**result, "events": result.get("rejected", []), "rejected": [], "set": "rejected"}
        band = (float(q.get("lfp_low", 1.0)), float(q.get("lfp_high", 300.0)))
        align = q.get("align", "peak") if q.get("align", "peak") in _trig.ALIGN else "peak"
        if index is None:                   # the average uses only events inside the included time ranges
            result = self.restrict(result)
        key = ("event", name, index, q.get("pre_ms"), q.get("post_ms"), band, q.get("smooth"), result.get("created"), q.get("set"), align, q.get("features"),
               self.sections_rev if index is None else 0)
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = _csd.event_view(self.ts, self.probe_layout(), result, det, index,
                              float(q.get("pre_ms", 100.0)), float(q.get("post_ms", 100.0)), band,
                              q.get("smooth", "1") not in ("0", "false"), align=align,
                              with_features=q.get("features", "1") not in ("0", "false"),
                              extras=self.extras_source())
        with self._cache_lock:
            if len(self._cache) > CACHE_SIZE:
                self._cache.clear()
            self._cache[key] = out
        return out

    def extras_source(self):
        """(TimeSeries, names) for the channels a processor recorded beside the electrodes, from
        this stream or wherever else in the session they are (loaded once, then kept)."""
        if self._extras is None:
            found = find_extra_channels(self.ts) if self.ts is not None else None
            if not found:
                self._extras = (None, [])
            else:
                path, stream, names = found
                same = path == str(self.ts.source) and stream == self.ts.stream
                try:                      # the panel is a bonus: never fail an event view over it
                    self._extras = (self.ts if same else load(path, stream=stream), names[:6])
                except Exception as e:
                    print(f"[extras] cannot load {stream} of {path}: {e}")
                    self._extras = (None, [])
        return self._extras

    def triggered(self, q: dict[str, str]) -> dict:
        """Event-triggered average of one channel with a sample of the single trials."""
        name = q.get("det", "")
        if name not in self.detections:
            raise KeyError(f"no event set {name!r}; pick one of {sorted(self.detections)}")
        result = self.restrict(self.detections[name])
        channel = q.get("channel") or result.get("channel") or self.ts.names[0]
        band = (float(q.get("low", 0) or 0), float(q.get("high", 0) or 0))
        norm = q.get("norm") or ("baseline" if q.get("baseline", "1") not in ("0", "false") else "none")
        args = (name, channel, float(q.get("pre_ms", 250.0)), float(q.get("post_ms", 250.0)), q.get("align", "start"), band,
                norm, int(q.get("max_trials", 100)), q.get("set", "accepted"), result.get("created"), float(q.get("base_ms", 100.0)), self.sections_rev)
        key = ("triggered",) + args
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        started = time.perf_counter()
        out = _trig.triggered_average(self.ts, result, channel, args[2], args[3], args[4], band if any(band) else None,
                                      max_trials=args[7], which=args[8], norm=norm, base_ms=args[10])
        out["compute_ms"] = round((time.perf_counter() - started) * 1000, 1)
        with self._cache_lock:
            if len(self._cache) > CACHE_SIZE:
                self._cache.clear()
            self._cache[key] = out
        return out

    def _apply_probe(self) -> None:
        if isinstance(self.probe_spec, Probe):
            self.probe = self.probe_spec
        elif self.probe_spec:
            self.probe = load_probe(self.probe_spec)
        elif self.ts is not None:
            n_headstage = len(self.ts.headstage_channels()) or self.ts.n_channels
            self.probe = auto_probe(self.ts.source if self.ts.lazy else None, n_headstage)
        else:
            self.probe = None

    @property
    def bad_channels(self) -> list[str]:
        return _chan.effective_bad(self.channels)

    def probe_layout(self) -> dict | None:
        if self.ts is None:
            return None
        if self.probe is None:
            return {"name": None, "contacts": [], "unmapped": list(self.ts.names), "bad": self.bad_channels}
        lay = self.probe.layout(self.ts.names, self.ts.channel_info, self.ts.units)
        lay["bad"] = self.bad_channels
        return lay

    def channels_json(self) -> dict:
        st = self.channels or {}
        return {"quality": st.get("quality"), "manual": st.get("manual", {"bad": [], "good": []}), "bad": self.bad_channels}

    def set_pulse_merge(self, merge_ms: float) -> dict:
        """Rebuild the recorded TTL pulse sets, merging pulses that nearly touch. Saved with the
        recording, because the sets themselves are rebuilt from the files on every load."""
        self.pulses = _det.save_pulse_settings(self.ts, merge_ms)
        for name in [n for n in self.detections if n.startswith(_det.ONLINE_PREFIX)]:
            self.detections.pop(name)
        self.detections.update(_det.online_detections(self.ts, self.pulses["merge_ms"]))
        return {"merge_ms": self.pulses["merge_ms"],
                "sets": {n: {"n_events": r["n_events"], "n_pulses": r.get("n_pulses")}
                         for n, r in self.detections.items() if n.startswith(_det.ONLINE_PREFIX)}}

    def update_channels(self, body: dict) -> dict:
        st = self.channels or {"quality": None, "manual": {"bad": [], "good": []}}
        if body.get("reset"):
            st = {"quality": None, "manual": {"bad": [], "good": []}}
        if body.get("check"):
            lay = self.probe.layout(self.ts.names, self.ts.channel_info, self.ts.units) if self.probe else None
            st["quality"] = _chan.channel_quality(self.ts, lay, float(body.get("sample_s", 30.0)),
                                                  float(body.get("noise_z", 2.5)), float(body.get("corr_min", 0.35)))
        man = st.setdefault("manual", {"bad": [], "good": []})
        for nm in body.get("bad", []):
            if nm not in self.ts.names:
                raise KeyError(f"unknown channel {nm!r}")
            if nm not in man["bad"]:
                man["bad"].append(nm)
            man["good"] = [g for g in man["good"] if g != nm]
        for nm in body.get("good", []):
            if nm not in man["good"]:
                man["good"].append(nm)
            man["bad"] = [b for b in man["bad"] if b != nm]
        self.channels = st
        _chan.save(self.ts, st)
        with self._cache_lock:
            self._cache.clear()
        return self.channels_json()


    def depth_order(self, names: list[str]) -> list[str]:
        """Shallow to deep (top of the screen = top of the probe), by shank then y; unmapped last."""
        lay = self.probe_layout()
        if not lay:
            return names
        pos = {c["name"]: (c["shank"], -c["y_um"], c["x_um"]) for c in lay["contacts"]}
        mapped = [n for n in names if n in pos]
        return sorted(mapped, key=lambda n: pos[n]) + [n for n in names if n not in pos]

    def meta(self) -> dict:
        return {
            "data": self.ts.summary() if self.ts is not None else None,
            "session": self.session,
            "root": str(self.root) if self.root else None,
            "dialog": dialog_available(),
            "probe": self.probe.name if self.probe else None,
            "plugins": [str(p) for p in self.plugins],
            "analyses": _an.schemas(),
            "detectors": _det.schemas(),
            # the page is served fresh from disk while this process keeps the code it started with:
            # tell the viewer when they no longer match, so it can ask for a restart
            "code_stale": _code_mtime() > CODE_MTIME_AT_START + 1.0,
        }

    # -- session browsing ---------------------------------------------------
    def list_sessions(self, rescan: bool = False) -> list[dict]:
        """Open Ephys recordings (one entry per structure.oebin) and loose data files under root."""
        if self.root is None:
            return []
        if self._sessions is not None and not rescan:
            return self._sessions
        found: list[dict] = []
        root_depth = len(self.root.parts)
        for dirpath, dirnames, filenames in os.walk(self.root):
            depth = len(Path(dirpath).parts) - root_depth
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            if depth >= SCAN_MAX_DEPTH:
                dirnames[:] = []
            if "structure.oebin" in filenames:
                dirnames[:] = []            # a recording's own files are not separate sessions
                filenames = ["structure.oebin"]
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                rel = str(p.relative_to(self.root))
                if fn == "structure.oebin":
                    duration = None
                    try:
                        meta = json.loads(p.read_text())
                        # a stream that recorded no samples cannot be opened, so do not offer it
                        # (the GLANCE IMU stream is written empty when the IMU is not streaming)
                        recorded = [c for c in meta.get("continuous", [])
                                    if (p.parent / "continuous" / c.get("folder_name", "") / "continuous.dat").exists()
                                    and (p.parent / "continuous" / c.get("folder_name", "") / "continuous.dat").stat().st_size]
                        meta["continuous"] = recorded or meta.get("continuous", [])
                        streams = [c.get("stream_name") or c.get("folder_name", "") for c in meta["continuous"]]
                        c0 = meta["continuous"][0]      # length from the file size: no need to open the recording
                        size = (p.parent / "continuous" / c0["folder_name"] / "continuous.dat").stat().st_size
                        duration = size / 2 / int(c0["num_channels"]) / float(c0["sample_rate"])
                    except Exception:
                        streams = streams if "streams" in locals() else []
                    label = str(p.parent.relative_to(self.root))
                    node = p.parent.parent.parent
                    if p.parent.name == "recording1" and p.parent.parent.name == "experiment1" and node != self.root \
                            and len(list(node.glob("experiment*/recording*/structure.oebin"))) == 1:
                        label = str(node.relative_to(self.root)) if self.root in node.parents else label
                    # a session written by several Record Nodes is one recording with different processing:
                    # group them under the session folder, the node is part of the stream choice
                    node_name, session = "", label
                    if p.parent.name.startswith("recording") and p.parent.parent.name.startswith("experiment") and node.name.startswith("Record Node"):
                        node_name = node.name
                        session = str(node.parent.relative_to(self.root)) if (self.root in node.parent.parents or node.parent == self.root) else label
                    found.append({"path": rel, "kind": "openephys", "label": self.root.name if label == "." else label,
                                  "streams": streams, "duration_s": duration,
                                  "session": self.root.name if session == "." else session, "node": node_name})
                elif p.suffix.lower() in DATA_SUFFIXES:
                    found.append({"path": rel, "kind": "file", "label": rel, "streams": [], "session": rel, "node": ""})
        self._sessions = found
        return found

    def set_root(self, path: str | Path, auto_open: bool = True) -> dict:
        """Point the browser at ``path``: a single recording or a folder of them. Opens the first."""
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"{path} is not a directory")
        self.root = root
        self._sessions = None
        remember_root(root)
        sessions = self.list_sessions(rescan=True)
        opened = None
        if auto_open and sessions:
            first = next((s for s in sessions if s["kind"] == "openephys"), sessions[0])
            stream = first["streams"][0] if len(first["streams"]) > 1 else None
            self.open_session(first["path"], stream)
            opened = self.session
        out = self.meta()
        out["sessions"] = sessions
        out["opened"] = opened
        return out

    def pick_root(self) -> dict:
        chosen = pick_folder(initial=self.root)
        if not chosen:
            from . import dialog as _dialog
            last = _dialog.remembered_root()
            return {"cancelled": True, "root": str(self.root) if self.root else None,
                    "error": _dialog.LAST_ERROR, "last_root": str(last) if last else None}
        out = self.set_root(chosen)
        out["cancelled"] = False
        return out

    def open_session(self, rel_path: str, stream: str | None = None, fs: float | None = None) -> dict:
        if self.root is None:
            raise ValueError("server has no data root; start with --root")
        target = (self.root / rel_path).resolve()
        if self.root not in target.parents and target != self.root:
            raise ValueError("path is outside the data root")
        if not target.exists():
            raise ValueError(f"{rel_path} does not exist")
        ts = load(target, fs=fs, stream=stream)
        self.set_data(ts, {"path": rel_path, "stream": stream})
        return self.meta()

    def reload_plugins(self) -> dict:
        loaded: dict[str, list[str]] = {}
        for p in self.plugins:
            loaded[str(p)] = _an.load_plugin(p)
        with self._cache_lock:
            self._cache.clear()
        return {"loaded": loaded, "analyses": _an.schemas()}

    def analyze(self, name: str, channel: str, raw_params: dict,
                t0: float | None = None, t1: float | None = None) -> _an.Result:
        """Run an analysis on the visible window (padded to hide filter edges) with a small cache."""
        an = _an.REGISTRY[name]
        params = an.coerce(raw_params)
        ts = self.ts
        lo, hi = 0, ts.n
        if t0 is not None or t1 is not None:
            a = ts.t[0] if t0 is None else t0
            b = ts.t[-1] if t1 is None else t1
            pad = max(WINDOW_PAD_S, WINDOW_PAD_FRAC * (b - a))
            lo = int(np.searchsorted(ts.t, a - pad, "left"))
            hi = int(np.searchsorted(ts.t, b + pad, "right"))
        key = (name, channel, tuple(sorted(params.items())), lo, hi)
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        sub = ts if (lo, hi) == (0, ts.n) else ts.window(ts.t[lo], ts.t[min(hi, ts.n) - 1])
        if an.wants_layout:
            params = {**params, "layout": self.probe_layout()}
        res = an.fn(sub, channel, **params)
        with self._cache_lock:
            if len(self._cache) > CACHE_SIZE:
                self._cache.clear()
            self._cache[key] = res
        return res

    def resolve_channels(self, spec: str) -> list[str]:
        ts = self.ts
        spec = (spec or "all").strip()
        if spec in ("all", "*"):
            return list(ts.names)
        if spec.startswith("unit:"):
            unit = spec[5:]
            names = [c for c, u in zip(ts.names, ts.units) if u == unit]
            if not names:
                raise ValueError(f"no channels with unit {unit!r}")
            return names
        if spec.startswith("kind:"):            # kind:headstage | kind:aux | kind:other
            names = ts.channels_of(spec[5:])
            if not names:
                raise ValueError(f"no {spec[5:]} channels")
            return names
        names = [c.strip() for c in spec.split(",") if c.strip()]
        for c in names:
            if c not in ts.names:
                raise KeyError(f"unknown channel {c!r}")
        return names

    def multi_json(self, name: str, spec: str, raw_params: dict, t0: float | None, t1: float | None,
                   points: int, order: str = "recorded") -> dict:
        """One lane per channel. Raw comes from the overview or a single block read; other
        analyses run per channel on the window (capped so the request stays interactive)."""
        ts = self.ts
        an = _an.REGISTRY[name]
        names = self.resolve_channels(spec)
        if order == "depth":
            names = self.depth_order(names)
        lo = 0 if t0 is None else int(np.searchsorted(ts.t, t0, "left"))
        hi = ts.n if t1 is None else int(np.searchsorted(ts.t, t1, "right"))
        nsamp = max(0, hi - lo)
        lanes = []
        meta = {"domain": "time", "xlabel": "time (s)", "ylabel": ""}
        if name == "raw":
            stride = ts.overview_stride()
            coarse = stride > 1 and nsamp / max(points, 1) >= 2 * stride
            if coarse:
                stride, centres, mins, maxs = ts.overview()     # built once, on first coarse request
                b0, b1 = lo // stride, int(math.ceil(hi / stride))
                x2 = np.repeat(centres[b0:b1], 2)
                for c in names:
                    i = ts.index(c)
                    y2 = np.column_stack([mins[b0:b1, i], maxs[b0:b1, i]]).ravel()
                    x, y = decimate_minmax(x2, y2, points)
                    lanes.append({"name": c, "unit": ts.unit(c), "traces": [{"name": c, "kind": "line", "x": _jsonable(x), "y": _jsonable(y)}]})
            else:
                blk = ts.block(lo, hi, names)
                x = ts.t[lo:hi]
                for j, c in enumerate(names):
                    xd, yd = decimate_minmax(x, blk[:, j], points)
                    lanes.append({"name": c, "unit": ts.unit(c), "traces": [{"name": c, "kind": "line", "x": _jsonable(xd), "y": _jsonable(yd)}]})
            meta["source"] = "overview" if coarse else "block"
        elif an.multichannel:
            res = self.analyze(name, names[0], raw_params, t0, t1)
            payload = result_to_json(res, t0, t1, points)
            meta = {"domain": payload["domain"], "xlabel": payload["xlabel"], "ylabel": payload["ylabel"]}
            lanes.append({"name": an.label, "unit": "", "traces": payload["traces"]})
        else:
            if nsamp > MULTI_MAX_SAMPLES and len(names) > 1:
                limit = MULTI_MAX_SAMPLES / (ts.fs or 1.0)
                raise ValueError(f"'{an.label}' on {len(names)} channels needs a window under {limit:.0f} s; zoom in or pick one channel")
            for c in names:
                res = self.analyze(name, c, raw_params, t0, t1)
                payload = result_to_json(res, t0, t1, points)
                meta = {"domain": payload["domain"], "xlabel": payload["xlabel"], "ylabel": payload["ylabel"]}
                lanes.append({"name": c, "unit": ts.unit(c), "traces": payload["traces"]})
        return {**meta, "channels": lanes, "params": an.coerce(raw_params)}

    def events_json(self, t0: float | None, t1: float | None, points: int) -> dict:
        out = {}
        for name, times in self.ts.events.items():
            lo = 0 if t0 is None else int(np.searchsorted(times, t0, "left"))
            hi = times.size if t1 is None else int(np.searchsorted(times, t1, "right"))
            x = times[lo:hi]
            labels = self.ts.event_labels.get(name)
            labels = labels[lo:hi] if labels is not None else None
            if x.size > points:
                step = int(math.ceil(x.size / points))
                x = x[::step]
                labels = labels[::step] if labels is not None else None
            out[name] = {"x": _jsonable(x), "labels": labels, "total": int(hi - lo)}
        dets = {}
        for name, r in self.detections.items():
            evs = r["events"]
            sel = [e for e in evs if (t1 is None or e["start"] <= t1) and (t0 is None or e["end"] >= t0)]
            if len(sel) > points:
                sel = sel[:: int(math.ceil(len(sel) / points))]
            rej = r.get("rejected", [])
            rsel = [e for e in rej if (t1 is None or e["start"] <= t1) and (t0 is None or e["end"] >= t0)]
            dets[name] = {"label": r["label"], "channel": r["channel"], "total": len(evs),
                          "events": [{k: e[k] for k in ("start", "end", "peak", "score", "ref_score", "move_score", "sw_score", "channels_frac", "amp_uV") if k in e} for e in sel],
                          "peaks": [e["peak"] for e in evs],   # all peaks, for prev/next navigation
                          "ref_channel": r.get("ref_channel"), "n_rejected": len(rej),
                          "rejected": [{k: e[k] for k in ("start", "end", "peak", "score", "ref_score", "move_score", "reason") if k in e} for e in rsel[:points]],
                          "rejected_peaks": [e["peak"] for e in rej],
                          # indices outside the included time ranges: the Events tab skips them
                          "excluded": [i for i, e in enumerate(evs) if not _sec.event_in(self.sections, e)],
                          "rejected_excluded": [i for i, e in enumerate(rej) if not _sec.event_in(self.sections, e)]}
        return {"events": out, "detections": dets}


class _Handler(BaseHTTPRequestHandler):
    server: DataServer

    def log_message(self, fmt, *args):  # quieter than the default
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    # -- helpers ------------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"))

    def _error(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    @staticmethod
    def _query(url: str) -> dict[str, str]:
        return {k: v[-1] for k, v in parse_qs(urlparse(url).query).items()}

    @staticmethod
    def _optfloat(q: dict, key: str) -> float | None:
        v = q.get(key)
        return None if v in (None, "") else float(v)

    # -- routes -------------------------------------------------------------
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        q = self._query(self.path)
        try:
            if path in ("/", "/index.html"):
                self._send(200, VIEWER_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/meta":
                self._json(self.server.meta())
            elif path == "/api/sessions":
                self._json({"root": str(self.server.root) if self.server.root else None,
                            "sessions": self.server.list_sessions(rescan=q.get("rescan") == "1")})
            elif path == "/api/analyze":
                self._analyze(q)
            elif path == "/api/multi":
                self._multi(q)
            elif path == "/api/event":
                self._require_data()
                started = time.perf_counter()
                out = self.server.event_view(q)
                out["compute_ms"] = round((time.perf_counter() - started) * 1000, 1)
                self._json(out)
            elif path == "/api/layers":
                self._json({"layers": self.server.layers})
            elif path == "/api/channels":
                self._require_data()
                self._json(self.server.channels_json())
            elif path in ("/api/features", "/api/features.csv"):
                self._require_data()
                name = q.get("det", "swr")
                if name not in self.server.detections:
                    raise KeyError(f"no saved detection {name!r}")
                res = self.server.detections[name]
                if path.endswith(".csv"):
                    body = _feat.to_csv(res, self.server.sections).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="{name}_events.csv"')
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    tb = _feat.table(res, q.get("set", "all"))
                    for row in tb["rows"]:
                        row["in_range"] = _sec.event_in(self.server.sections, row)
                    self._json(tb)
            elif path == "/api/detections":
                self._json(self.server.detections_json())
            elif path == "/api/triggered":
                self._require_data()
                self._json(self.server.triggered(q))
            elif path == "/api/sections":
                self._require_data()
                self._json(self.server.sections_json())
            elif path == "/api/units":
                self._require_data()
                ok, msg = _srt.available()
                if not self.server.sorting:
                    raise KeyError(("no units sorted for this recording yet" if ok else msg))
                self._json(_srt.summary(self.server.sorting, self.server.ts.duration))
            elif path == "/api/unit":
                self._require_data()
                if not self.server.sorting:
                    raise KeyError("no units sorted for this recording yet")
                self._json(_srt.unit_detail(self.server.sorting, int(q.get("unit", "0"))))
            elif path == "/api/spikes/overview":
                self._require_data()
                if not self.server.spikes:
                    raise KeyError("no spikes detected for this recording yet")
                self._json(_spk.overview(self.server.spikes, self.server.ts.duration))
            elif path == "/api/spikes/channel":
                self._require_data()
                if not self.server.spikes:
                    raise KeyError("no spikes detected for this recording yet")
                self._json(_spk.waveforms(self.server.ts, self.server.spikes, q.get("channel", ""),
                                          float(q.get("pre_ms", 1.0)), float(q.get("post_ms", 2.0)),
                                          int(q.get("max_n", 300))))
            elif path == "/api/phase":
                self._require_data()
                self._json(self.server.phase_json(q))
            elif path == "/api/theta/coherence":
                self._require_data()
                self._json(self.server.phase_coherence(q))
            elif path == "/api/theta/activity":
                self._require_data()
                self._json(self.server.activity_json(q))
            elif path == "/api/phase/cycle":
                self._require_data()
                self._json(self.server.phase_cycle(q))
            elif path == "/api/phase/epoch":
                self._require_data()
                self._json(self.server.phase_epoch(q))
            elif path == "/api/swr/scan":
                self._require_data()
                self._json(self.server.swr_scan(q.get("det", "swr_laminar")))
            elif path == "/api/swr/sessions":
                self._json(self.server.swr_sessions())
            elif path == "/api/compare":
                self._require_data()
                self._json(self.server.compare(q))
            elif path == "/api/jobs":
                self._json(self.server.jobs_json())
            elif path == "/api/spikes":
                self._require_data()
                self._json(self.server.spikes_json(self._optfloat(q, "t0"), self._optfloat(q, "t1"), int(q.get("max", 5000))))
            elif path == "/api/probe":
                self._require_data()
                lay = self.server.probe_layout()
                self._json(lay if lay and lay.get("contacts") else {"name": None, "contacts": [], "unmapped": list(self.server.ts.names), "bad": self.server.bad_channels,
                                            "notes": "no probe geometry: pass --probe FILE or a built-in name, "
                                                     "or put probe.json / probe.csv next to the recording"})
            elif path == "/api/events":
                self._require_data()
                self._json(self.server.events_json(self._optfloat(q, "t0"), self._optfloat(q, "t1"),
                                                   int(q.get("points", DEFAULT_POINTS))))
            elif path == "/api/live":
                snap = self.server.store.snapshot(int(q.get("since", -1)), self._optfloat(q, "t0"),
                                                 self._optfloat(q, "t1"), int(q.get("points", DEFAULT_POINTS)))
                self._json(snap)
            else:
                self._error(404, f"no route {path}")
        except (KeyError, ValueError, FileNotFoundError) as e:
            self._error(400, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _require_data(self) -> TimeSeries:
        if self.server.ts is None:
            raise ValueError("no data loaded; open a session first")
        return self.server.ts

    def _analyze(self, q: dict[str, str]) -> None:
        name = q.get("analysis", "raw")
        if name not in _an.REGISTRY:
            raise KeyError(f"unknown analysis {name!r}")
        ts = self._require_data()
        channel = q.get("channel") or ts.names[0]
        if channel not in ts.names:
            raise KeyError(f"unknown channel {channel!r}")
        t0, t1 = self._optfloat(q, "t0"), self._optfloat(q, "t1")
        points = int(q.get("points", DEFAULT_POINTS))
        reserved = {"analysis", "channel", "t0", "t1", "points"}
        params = {k: v for k, v in q.items() if k not in reserved}
        started = time.perf_counter()
        res = self.server.analyze(name, channel, params, t0, t1)
        payload = result_to_json(res, t0, t1, points)
        payload["params"] = _an.REGISTRY[name].coerce(params)
        payload["compute_ms"] = round((time.perf_counter() - started) * 1000, 1)
        payload["unit"] = self.server.ts.unit(channel)
        self._json(payload)

    def _multi(self, q: dict[str, str]) -> None:
        name = q.get("analysis", "raw")
        if name not in _an.REGISTRY:
            raise KeyError(f"unknown analysis {name!r}")
        self._require_data()
        t0, t1 = self._optfloat(q, "t0"), self._optfloat(q, "t1")
        points = int(q.get("points", DEFAULT_POINTS))
        reserved = {"analysis", "channel", "channels", "t0", "t1", "points", "order"}
        params = {k: v for k, v in q.items() if k not in reserved}
        started = time.perf_counter()
        payload = self.server.multi_json(name, q.get("channels", "all"), params, t0, t1, points,
                                         q.get("order", "recorded"))
        payload["order"] = q.get("order", "recorded")
        payload["compute_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self._json(payload)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/open":
                self._json(self.server.open_session(body["path"], body.get("stream") or None, body.get("fs")))
            elif path == "/api/pick":
                self._json(self.server.pick_root())
            elif path == "/api/root":
                self._json(self.server.set_root(body["path"], auto_open=body.get("auto_open", True)))
            elif path == "/api/reload":
                self._json(self.server.reload_plugins())
            elif path == "/api/detect":
                self._require_data()
                if body.get("remove"):
                    if _det.is_online(self.server.detections.get(body["name"]) or {}):
                        raise ValueError("TTL pulses are part of the recording and cannot be deleted")
                    self.server.detections.pop(body["name"], None)
                    self._json({"removed": _det.delete_result(self.server.ts, body["name"])})
                else:
                    self._json({"job": self.server.run_detection(body["name"], body.get("channel") or self.server.ts.names[0],
                                                                 body.get("params") or {}, body.get("tag"))})
            elif path == "/api/pulses":
                self._require_data()
                self._json(self.server.set_pulse_merge(float(body.get("merge_ms", 0) or 0)))
            elif path == "/api/channels":
                self._require_data()
                self._json(self.server.update_channels(body))
            elif path == "/api/layers":
                self._require_data()
                if body.get("remove"):
                    p = _csd.layers_path(self.server.ts)
                    removed = p.exists()
                    if removed:
                        p.unlink()
                    self.server.layers = None
                    self._json({"removed": removed})
                else:
                    started = time.perf_counter()
                    est = self.server.estimate_layers(body)
                    self._json({"layers": est, "compute_ms": round((time.perf_counter() - started) * 1000, 1)})
            elif path == "/api/features":
                self._require_data()
                self._json({"job": self.server.run_features(body.get("det", "swr"))})
            elif path == "/api/filter":
                self._require_data()
                name = body.get("det", "swr")
                if name not in self.server.detections:
                    raise KeyError(f"no saved detection {name!r}")
                res = self.server.detections[name]
                out = _feat.restore_filtered(res) if body.get("restore") else _feat.apply_filter(res, body.get("rules") or {})
                if not _det.is_online(res):
                    _det.save_result(self.server.ts, res)
                self._json(out)
            elif path == "/api/sections":
                self._require_data()
                self._json(self.server.update_sections(body))
            elif path == "/api/swr/batch":
                self._json({"job": self.server.run_swr_batch(body)})
            elif path == "/api/sort":
                self._require_data()
                self._json({"job": self.server.run_sorting(body)})
            elif path == "/api/spikes/run":
                self._require_data()
                self._json({"job": self.server.run_spikes(body)})
            elif path == "/api/live":
                if body.get("clear"):
                    self.server.store.clear()
                    self._json({"version": self.server.store.version})
                    return
                if body.get("remove"):
                    self.server.store.remove(body["remove"])
                    self._json({"version": self.server.store.version})
                    return
                v = self.server.store.publish(body["name"], body["x"], body.get("y"),
                                              body.get("kind", "line"), **body.get("meta", {}))
                self._json({"version": v})
            else:
                self._error(404, f"no route {path}")
        except (KeyError, ValueError, TypeError, OSError, ImportError, SyntaxError) as e:
            self._error(400, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def serve_in_background(ts: TimeSeries | None, host: str = "127.0.0.1", port: int = 8000,
                        store: LiveStore | None = None, open_browser: bool = False,
                        root: str | Path | None = None, plugins: list[str | Path] | None = None,
                        probe: str | Path | Probe | None = None) -> DataServer:
    """Start serving ``ts`` on a daemon thread and return the server (use ``.store.publish``)."""
    server = DataServer((host, port), ts, store, root=root, plugins=plugins, probe=probe)
    threading.Thread(target=server.serve_forever, daemon=True, name="dataanalyzer-http").start()
    if open_browser:
        webbrowser.open(server.url)
    return server


def serve(ts_or_server: TimeSeries | DataServer | None, host: str = "127.0.0.1", port: int = 8000,
          open_browser: bool = True, root: str | Path | None = None,
          plugins: list[str | Path] | None = None) -> None:
    """Serve until Ctrl-C. Accepts a TimeSeries (or None) or an already configured DataServer."""
    if isinstance(ts_or_server, DataServer):
        server = ts_or_server
    else:
        server = DataServer((host, port), ts_or_server, root=root, plugins=plugins)
    print(f"DataAnalyzer viewer at {server.url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
