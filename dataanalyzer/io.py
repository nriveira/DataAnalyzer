"""Time series container and loaders.

Everything downstream works on a :class:`TimeSeries`: a time vector in seconds
plus an ``(n, k)`` array-like of channels. ``data`` may be a read-only
``np.memmap`` of raw integers (this is how Open Ephys recordings are held, a
1 GB file is never read whole) with a per-channel ``scale`` applied lazily by
:meth:`TimeSeries.channel`. Analyses should always go through ``channel()``
or ``window()`` rather than touching ``data`` directly.

Loaders: Open Ephys binary sessions (directory containing ``structure.oebin``),
CSV/TSV with a header row, ``.npy`` and ``.npz``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TIME_COLUMN_NAMES = ("t", "time", "timestamp", "timestamps", "seconds", "sec")
HEADSTAGE_UNITS = ("uV", "µV", "")
OVERVIEW_BUCKETS = 20_000       # target bucket count for the all-channel min/max overview
OVERVIEW_CHUNK_ROWS = 1 << 20   # rows read per pass when building it (~70 MB for 35 int16 channels)


@dataclass
class TimeSeries:
    t: np.ndarray                                   # (n,) seconds, increasing
    data: np.ndarray                                # (n, k) samples; may be an int memmap
    names: list[str] = field(default_factory=list)
    fs: float | None = None                         # nominal sample rate, Hz
    source: str = ""
    scale: np.ndarray | None = None                 # (k,) multiplier applied by channel()
    units: list[str] = field(default_factory=list)
    events: dict[str, np.ndarray] = field(default_factory=dict)        # name -> times (s)
    event_labels: dict[str, list[str]] = field(default_factory=dict)   # name -> per-event text
    channel_info: list[dict] | None = field(default=None, repr=False, compare=False)   # per-channel provenance
    results_subdir: str = ""                        # stream folder under <recording>/dataanalyzer/ when a recording has several streams
    stream: str = ""                                # Open Ephys stream name
    ttl: list[dict] = field(default_factory=list, repr=False, compare=False)       # TTL lines as on/off times, see _oe_events
    format_notes: dict = field(default_factory=dict, repr=False, compare=False)   # how the loader read time and channel inputs
    recording_meta: dict = field(default_factory=dict, repr=False, compare=False)  # {"processors", "stream_channels"} of an Open Ephys recording
    _overview: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        if not isinstance(self.data, np.memmap):
            self.data = np.asarray(self.data)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        if self.data.shape[0] != self.t.shape[0]:
            raise ValueError(f"data has {self.data.shape[0]} rows but t has {self.t.shape[0]} samples")
        k = self.data.shape[1]
        if not self.names:
            self.names = [f"ch{i}" for i in range(k)]
        if len(self.names) != k:
            raise ValueError("names length does not match number of channels")
        if len(set(self.names)) != k:
            raise ValueError("channel names must be unique")
        if self.scale is not None:
            self.scale = np.asarray(self.scale, dtype=float).reshape(-1)
            if self.scale.size != k:
                raise ValueError("scale length does not match number of channels")
        if not self.units:
            self.units = [""] * k
        if self.fs is None and len(self.t) > 1:
            dt = np.diff(self.t)
            if dt.min() > 0 and np.allclose(dt, dt[0], rtol=1e-3, atol=0):
                self.fs = float(1.0 / dt[0])

    # -- basic properties ---------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.t.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n else 0.0

    @property
    def lazy(self) -> bool:
        return isinstance(self.data, np.memmap)

    def index(self, name: str | int) -> int:
        return name if isinstance(name, int) else self.names.index(name)

    def channel(self, name: str | int) -> np.ndarray:
        """One channel as a float64 array in physical units (a copy, safe to modify)."""
        idx = self.index(name)
        col = np.array(self.data[:, idx], dtype=float)
        if self.scale is not None:
            col *= self.scale[idx]
        return col

    def unit(self, name: str | int) -> str:
        return self.units[self.index(name)]

    def kinds(self) -> list[str]:
        """'headstage' | 'aux' | 'other' per channel. Loaders that know (Open Ephys) record it in
        ``channel_info``; otherwise it is inferred from the units (µV or none = electrode, mV = aux)."""
        out = []
        for i, u in enumerate(self.units):
            k = (self.channel_info[i].get("kind") if self.channel_info and i < len(self.channel_info) else None)
            out.append(k or ("headstage" if u in HEADSTAGE_UNITS else "aux" if u == "mV" else "other"))
        return out

    def channels_of(self, kind: str) -> list[str]:
        return [n for n, k in zip(self.names, self.kinds()) if k == kind]

    def headstage_channels(self) -> list[str]:
        """Electrode channels, in recorded order (excludes aux inputs and processor outputs such as RIP_FEAT)."""
        return self.channels_of("headstage")

    def aux_channels(self) -> list[str]:
        return self.channels_of("aux")

    def block(self, lo: int, hi: int, names: list[str | int] | None = None) -> np.ndarray:
        """Samples ``lo:hi`` for several channels at once as an (hi-lo, m) float array in physical units."""
        idxs = list(range(self.n_channels)) if names is None else [self.index(c) for c in names]
        rows = self.data[lo:hi]
        out = np.array(rows[:, idxs] if len(idxs) != self.n_channels else rows, dtype=float)
        if self.scale is not None:
            out *= self.scale[idxs]
        return out

    def overview_stride(self) -> int:
        """Samples per overview bucket (without building the overview)."""
        if self._overview is not None:
            return self._overview[0]
        return max(1, math.ceil(self.n / OVERVIEW_BUCKETS))

    def overview(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Per-bucket (stride, t_centre, mins, maxs) for every channel, built once in chunks.

        ``mins``/``maxs`` are (n_buckets, k) in physical units. This is what lets the viewer
        show the whole recording for all channels without reading it into RAM as floats.
        """
        if self._overview is None:
            n, k = self.n, self.n_channels
            stride = self.overview_stride()
            nb = math.ceil(n / stride) if n else 0
            mins = np.empty((nb, k)); maxs = np.empty((nb, k))
            per_chunk = max(1, OVERVIEW_CHUNK_ROWS // stride)
            for b0 in range(0, nb, per_chunk):
                b1 = min(nb, b0 + per_chunk)
                blk = np.asarray(self.data[b0 * stride:min(n, b1 * stride)])
                full = (blk.shape[0] // stride) * stride
                if full:
                    r = blk[:full].reshape(-1, stride, k)
                    mins[b0:b0 + full // stride] = r.min(axis=1)
                    maxs[b0:b0 + full // stride] = r.max(axis=1)
                if full < blk.shape[0]:
                    mins[b1 - 1] = blk[full:].min(axis=0)
                    maxs[b1 - 1] = blk[full:].max(axis=0)
            if self.scale is not None:
                mins *= self.scale; maxs *= self.scale
                flip = self.scale < 0
                if flip.any():
                    mins[:, flip], maxs[:, flip] = maxs[:, flip].copy(), mins[:, flip].copy()
            centres = self.t[np.minimum(np.arange(nb) * stride + stride // 2, max(n - 1, 0))] if n else np.empty(0)
            self._overview = (stride, centres, mins, maxs)
        return self._overview

    def window(self, t0: float | None = None, t1: float | None = None) -> TimeSeries:
        """Samples with ``t0 <= t <= t1`` as a new TimeSeries sharing memory with this one."""
        lo = 0 if t0 is None else int(np.searchsorted(self.t, t0, side="left"))
        hi = self.n if t1 is None else int(np.searchsorted(self.t, t1, side="right"))
        lo, hi = min(lo, self.n), max(min(hi, self.n), min(lo, self.n))
        ev, lab = {}, {}
        for name, times in self.events.items():
            m = np.ones(times.size, dtype=bool)
            if t0 is not None:
                m &= times >= t0
            if t1 is not None:
                m &= times <= t1
            ev[name] = times[m]
            if name in self.event_labels:
                lab[name] = [s for s, keep in zip(self.event_labels[name], m) if keep]
        return TimeSeries(self.t[lo:hi], self.data[lo:hi], list(self.names), self.fs, self.source,
                          self.scale, list(self.units), ev, lab, self.channel_info, self.results_subdir)

    def summary(self) -> dict:
        return {
            "source": self.source,
            "n": self.n,
            "n_channels": self.n_channels,
            "fs": self.fs,
            "t0": float(self.t[0]) if self.n else 0.0,
            "t1": float(self.t[-1]) if self.n else 0.0,
            "duration": self.duration,
            "channels": list(self.names),
            "units": list(self.units),
            "lazy": self.lazy,
            "channel_info": self.channel_info,
            "kinds": self.kinds(),
            "stream": self.stream,
            "format": self.format_notes,
            "signal_chain": self.recording_meta.get("signal_chain") or [],
            "events": {k: int(v.size) for k, v in self.events.items()},
        }


# ---------------------------------------------------------------------------
# Generic loaders
# ---------------------------------------------------------------------------

def load(path: str | Path, fs: float | None = None, time_column: str | None = None,
         stream: str | None = None) -> TimeSeries:
    """Load by type: a directory or ``structure.oebin`` is Open Ephys, else by extension."""
    path = Path(path)
    if path.is_dir() or path.name == "structure.oebin":
        return load_openephys(path, stream=stream)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        return load_csv(path, fs=fs, time_column=time_column)
    if suffix == ".npy":
        return load_npy(path, fs=fs)
    if suffix == ".npz":
        return load_npz(path, fs=fs)
    raise ValueError(f"unsupported input: {path}")


def load_csv(path: str | Path, fs: float | None = None, time_column: str | None = None) -> TimeSeries:
    """CSV/TSV with a header row. The time column is auto-detected by name unless given."""
    path = Path(path)
    delimiter = {".tsv": "\t", ".csv": ","}.get(path.suffix.lower())   # None = any whitespace
    table = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=float, encoding="utf-8")
    if table.dtype.names is None:
        raise ValueError(f"{path} has no header row; cannot name channels")
    if table.ndim == 0:  # single-row file
        table = table.reshape(1)
    cols = list(table.dtype.names)

    if time_column is None:
        time_column = next((c for c in cols if c.lower() in TIME_COLUMN_NAMES), None)
    if time_column is not None:
        if time_column not in cols:
            raise ValueError(f"time column {time_column!r} not in {cols}")
        t = table[time_column]
        cols.remove(time_column)
    else:
        if fs is None:
            raise ValueError("no time column found; pass fs")
        t = np.arange(len(table)) / fs

    data = np.column_stack([table[c] for c in cols]) if cols else np.empty((len(t), 0))
    return TimeSeries(t, data, cols, fs, source=str(path))


def load_npy(path: str | Path, fs: float | None = None, names: list[str] | None = None) -> TimeSeries:
    """``.npy`` holding ``(n,)``, ``(n, k)`` or ``(k, n)``; the longer axis is taken as time."""
    if fs is None:
        raise ValueError("fs is required for .npy input")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = np.ascontiguousarray(arr.T)
    t = np.arange(arr.shape[0]) / fs
    return TimeSeries(t, arr, names or [], fs, source=str(path))


def load_npz(path: str | Path, fs: float | None = None) -> TimeSeries:
    """``.npz`` with keys ``data`` and optionally ``t``, ``names``, ``fs``."""
    with np.load(path, allow_pickle=False) as z:
        data = z["data"]
        names = [str(s) for s in z["names"]] if "names" in z else []
        fs = float(z["fs"]) if "fs" in z else fs
        if "t" in z:
            t = z["t"]
        else:
            if fs is None:
                raise ValueError(".npz has no 't' array; pass fs")
            t = np.arange(data.shape[0]) / fs
    return TimeSeries(t, data, names, fs, source=str(path))


# ---------------------------------------------------------------------------
# Open Ephys binary format
# ---------------------------------------------------------------------------

def find_oebin(path: str | Path) -> Path:
    path = Path(path)
    if path.name == "structure.oebin":
        return path
    found = sorted(path.rglob("structure.oebin"))
    if not found:
        raise FileNotFoundError(f"no structure.oebin under {path}")
    return found[0]


TIME_CHECK_CHUNK = 1 << 22


def _increasing(a: np.ndarray) -> bool:
    """Strictly increasing and finite, checked in chunks so a memmap is never read whole."""
    for lo in range(0, a.shape[0], TIME_CHECK_CHUNK):
        blk = np.asarray(a[max(lo - 1, 0):lo + TIME_CHECK_CHUNK], dtype=float)
        if not np.isfinite(blk).all() or (blk.size > 1 and not (np.diff(blk) > 0).all()):
            return False
    return True


def _oe_time_axis(folder: Path, fs: float, n: int, tick_rates: tuple[float, ...] = ()) -> tuple[np.ndarray, str]:
    """(seconds on the acquisition clock, where they came from).

    ``timestamps.npy`` is used only when it is a believable clock for the whole stream: strictly
    increasing and spanning (n-1)/fs, in seconds or in clock ticks — of this stream's rate or of
    one of ``tick_rates`` (the recording's other streams: the GLANCE writer stores the LFP stream's
    timestamps as 30 kHz acquisition ticks *per buffer*, and its LFP ``sample_numbers.npy`` can start
    from an unrelated count, 267 s off in ``SalineLatencyTest1014/Minimum``; the axis is then the
    sample numbers shifted onto the ticks at the buffer starts). Otherwise (the GLANCE writer
    restarts the broadband timestamps every buffer) the time axis is ``sample_numbers.npy / fs``,
    then ``arange / fs``."""
    sn_path, ts_path = folder / "sample_numbers.npy", folder / "timestamps.npy"
    sn = np.load(sn_path, mmap_mode="r") if sn_path.exists() else None
    ts = np.load(ts_path, mmap_mode="r") if ts_path.exists() else None
    expect = (n - 1) / fs
    if ts is not None and ts.shape[0] == n and n > 1 and _increasing(ts):
        span = float(ts[-1]) - float(ts[0])
        if abs(span - expect) <= max(0.01 * expect, 2.0 / fs):
            return np.asarray(ts, dtype=float), "timestamps"
        if abs(span / fs - expect) <= max(0.01 * expect, 2.0 / fs):     # tick counts written as timestamps
            return np.asarray(ts, dtype=float) / fs, "timestamps (ticks)"
        for rate in tick_rates:                                          # ticks of another stream's clock
            if rate != fs and abs(span / rate - expect) <= max(0.01 * expect, 2.0 / fs):
                tk = np.asarray(ts, dtype=float)
                if sn is None or sn.shape[0] != n:
                    return tk / rate, f"timestamps (ticks at {rate:g} Hz)"
                # the GLANCE LFP ticks are per buffer (every sample of a buffer carries its first
                # sample's tick, plus a negligible fraction), so they set only the offset: the
                # spacing comes from the sample numbers, the offset from the buffer starts
                s = np.asarray(sn, dtype=float) / fs
                start = np.r_[True, np.diff(tk) > 0.5]
                offset = float(np.median(tk[start] / rate - s[start]))
                return s + offset, f"sample_numbers + offset from timestamps (ticks at {rate:g} Hz)"
    if sn is not None and sn.shape[0] == n:
        return np.asarray(sn, dtype=float) / fs, "sample_numbers"
    return np.arange(n) / fs, "sample index"


def _oe_settings(rec_dir: Path):
    """Root element of the settings.xml beside the Record Node (searched upward), or None."""
    import xml.etree.ElementTree as ET
    for parent in [rec_dir, *rec_dir.parents[:3]]:
        cand = parent / "settings.xml"
        if cand.exists():
            try:
                return ET.parse(cand).getroot()
            except ET.ParseError:
                return None
    return None


def oe_processor_params(rec_dir: Path) -> dict[str, dict[str, dict]]:
    """{processor name: {stream name: parameters}} from settings.xml, e.g. the online Ripple
    Detector's thresholds and input channel. Empty when there is no settings.xml."""
    root = _oe_settings(Path(rec_dir))
    out: dict[str, dict[str, dict]] = {}
    if root is None:
        return out
    for proc in root.iter("PROCESSOR"):
        name = proc.attrib.get("name", "").split("/")[-1].strip()
        for st in proc.findall("STREAM"):
            par = st.find("PARAMETERS")
            if par is not None and par.attrib and st.attrib.get("name"):
                out.setdefault(name, {})[st.attrib["name"]] = dict(par.attrib)
    return out


def _ranges(nums: list[int]) -> str:
    """[1, 2, 3, 7] -> '1-3, 7'."""
    out, nums = [], sorted(set(nums))
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def _mask_text(value: str | None, n_channels: int | None) -> str:
    """Open Ephys saves a channel-mask parameter as the channels that are *excluded* (1-based):
    in the 2026-09-18 recordings ``affected="33,34,35"`` went with a common average applied to
    channels 1-32, and an empty value with all channels. Returns the selected channels as text."""
    try:
        excluded = [int(v) for v in (value or "").replace(" ", "").split(",") if v]
    except ValueError:
        return str(value)
    if not n_channels:
        return f"all but {_ranges(excluded)}" if excluded else "all"
    selected = [c for c in range(1, n_channels + 1) if c not in excluded]
    return "none" if not selected else ("all" if len(selected) == n_channels else _ranges(selected))


def _oe_describe(name: str, par: dict, n_in: int | None, n_out: int | None, custom) -> tuple[str, bool]:
    """(one-line summary of what a processor did to the loaded stream, does it alter the data)."""
    low = name.lower()
    on = par.get("enable_stream", "1") != "0"
    off = "" if on else " (disabled for this stream)"
    if "notch" in low or "bandpass" in low or "filter" in low and "low_cut" in par:
        return f"{par.get('low_cut', '?')}-{par.get('high_cut', '?')} Hz on ch {_mask_text(par.get('channels'), n_in)}{off}", on
    if "common avg" in low or "car" == low:
        return (f"ch {_mask_text(par.get('affected'), n_in)} minus the mean of ch {_mask_text(par.get('reference'), n_in)}"
                f" x {float(par.get('gain', 100)) / 100:g}{off}"), on
    if "channel map" in low:
        order = None
        if custom is not None:
            for st in custom.findall("STREAM"):
                chs = [int(c.attrib["index"]) for c in st.findall("CH") if c.attrib.get("enabled", "1") == "1"]
                if n_out and len(chs) == n_out:
                    order = chs
                    break
        if order is None:
            return "channel order unknown" + off, on
        moved = order != sorted(order)
        return ("reordered: recorded position 1 = input " + str(order[0] + 1) if moved else "identity (order unchanged)") + off, on and moved
    if "ripple" in low:
        added = (n_out - n_in) if n_in and n_out else 0
        meth = {"0": "RMS", "1": "Hilbert", "2": "other"}.get(par.get("method", ""), par.get("method", "?"))
        return (f"input ch {par.get('Ripple_Input', '?')}, on > {par.get('ripple_std', '?')} SD, off < {par.get('ripple_std_off', '?')} SD, "
                f"TTL line {int(float(par.get('Ripple_Out', 0))) + 1}, method {meth}"
                + (f"; adds {added} channels (RIP_FEAT, RIP_ON, RIP_OFF)" if added > 0 else "") + off), added > 0
    if "record" in low:
        return "writes the files being viewed", False
    if n_in and n_out and n_in != n_out:
        return f"{n_in} -> {n_out} channels", True
    return "", False


def oe_signal_chain(rec_dir: Path, stream: str | None = None) -> list[dict]:
    """The processors the loaded stream passed through, source first, ending at the Record Node
    that wrote ``rec_dir``; processors on other Splitter branches are listed after it with
    ``on_path`` false. Each entry: name, node_id, on_path, recorder (this Record Node), channels
    in/out, summary, alters (changes the samples or channels), params.

    settings.xml is flat: a Splitter's first branch follows it directly and ``<SWITCH number="n"/>``
    starts branch n, so the path to a Record Node is the trunk plus that node's own branch."""
    root = _oe_settings(Path(rec_dir))
    if root is None:
        return []
    node_id = None
    for part in Path(rec_dir).parts:
        m = re.fullmatch(r"Record Node (\d+)", part)
        if m:
            node_id = m.group(1)
    base = re.sub(r"-[A-Z]$", "", stream or "")
    out: list[dict] = []
    for chain in root.iter("SIGNALCHAIN"):
        branch, split = 0, False
        for el in chain:
            if el.tag == "SWITCH":
                branch = int(el.attrib.get("number", branch + 1))
                continue
            if el.tag != "PROCESSOR":
                continue
            name = el.attrib.get("name", "").split("/")[-1].strip()
            streams = {st.attrib.get("name", ""): st for st in el.findall("STREAM")}
            st = streams.get(stream or "") or streams.get(base) or (next(iter(streams.values())) if len(streams) == 1 else None)
            par = dict(st.find("PARAMETERS").attrib) if st is not None and st.find("PARAMETERS") is not None else {}
            n_out = int(st.attrib["channel_count"]) if st is not None and st.attrib.get("channel_count") else None
            out.append({"name": name, "node_id": el.attrib.get("nodeId"), "branch": branch if split else None,
                        "n_out": n_out, "params": par, "_custom": el.find("CUSTOM_PARAMETERS"),
                        "_has_stream": st is not None or not streams})
            if "splitter" in name.lower() or "merger" in name.lower():
                split = True
    mine = next((p for p in out if "record" in p["name"].lower() and p["node_id"] == node_id), None)
    my_branch = mine["branch"] if mine else None
    seen_recorder, n_prev, bandpassed = False, None, False
    for p in out:
        p["recorder"] = p is mine
        p["on_path"] = bool(p["_has_stream"] and (p["branch"] is None or my_branch is None or p["branch"] == my_branch)
                            and not seen_recorder and not ("record" in p["name"].lower() and p is not mine and mine is not None))
        if p["on_path"]:
            p["n_in"] = n_prev
            p["summary"], p["alters"] = _oe_describe(p["name"], p["params"], n_prev, p["n_out"], p["_custom"])
            n_prev = p["n_out"] or n_prev
            low = p["name"].lower()
            if "filter" in low and "notch" not in low and p["params"].get("enable_stream", "1") != "0":
                bandpassed = True
            if "ripple" in low and not bandpassed:
                p["warning"] = "no band-pass filter upstream: this detector thresholds the wideband signal, not the ripple band"
        else:
            p["n_in"], p["summary"], p["alters"] = None, "other branch" if p["branch"] is not None else "", False
        seen_recorder = seen_recorder or p is mine
        del p["_custom"], p["_has_stream"]
    return out


def _oe_channel_map(rec_dir: Path, stream_name: str | None, nch: int) -> list[int] | None:
    """Headstage input index for each recorded position, from the Channel Map in settings.xml.

    Open Ephys records channels downstream of the Channel Map, so ``continuous.dat`` is
    already in mapped order; the map tells us which physical input each position came from.
    Returns None when there is no settings.xml, no Channel Map, or the counts do not match.
    """
    root = _oe_settings(rec_dir)
    if root is None:
        return None
    for proc in root.iter("PROCESSOR"):
        if "channel map" not in proc.attrib.get("name", "").lower():
            continue
        custom = proc.find("CUSTOM_PARAMETERS")
        if custom is None:
            continue
        streams = list(custom.findall("STREAM"))
        chosen = None
        for st in streams:
            if stream_name and st.attrib.get("name") == stream_name:
                chosen = st
                break
        if chosen is None:
            for st in streams:
                enabled = [c for c in st.findall("CH") if c.attrib.get("enabled", "1") == "1"]
                if len(enabled) == nch:
                    chosen = st
                    break
        if chosen is None:
            continue
        order = [int(c.attrib["index"]) for c in chosen.findall("CH") if c.attrib.get("enabled", "1") == "1"]
        if len(order) == nch:
            return order
    return None


def input_from_name(name: str) -> int | None:
    """Headstage input number (1-based) carried by a channel name, else None.

    'CH20', 'A_CH20', 'LFP_A_CH20' -> 20; 'A-019' -> 20 (Intan RHX is 0-based)."""
    nm = name.strip()
    m = re.fullmatch(r"(?:.*[_\-\s])?CH(\d+)", nm, re.I)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"[A-D]-(\d{3})", nm, re.I)
    if m:
        return int(m.group(1)) + 1
    return None


def _oe_channel_kind(name: str, units: str, description: str = "") -> str:
    """'headstage' (an electrode), 'aux' (accelerometer / ADC) or 'other' (a processor's output,
    e.g. the Ripple Detector's RIP_FEAT, which is written in µV but is not an electrode)."""
    nm, desc = name.strip().upper(), description.lower()
    if "AUX" in nm or "ADC" in nm or "aux" in desc or units == "mV":
        return "aux"
    if units in HEADSTAGE_UNITS and (not nm or input_from_name(nm) is not None):
        return "headstage"
    return "other"


def _oe_channel_names(chans: list[dict], channel_map: list[int] | None = None) -> list[str]:
    """Blank names become CH<n>: the headstage input number (1-based) via the channel map
    when available, else the recorded position."""
    names: list[str] = []
    for i, ch in enumerate(chans):
        nm = str(ch.get("channel_name", "")).strip()
        if not nm:
            nm = f"CH{(channel_map[i] if channel_map else i) + 1}"
        base, k = nm, 2
        while nm in names:
            nm, k = f"{base}_{k}", k + 1
        names.append(nm)
    return names


def _oe_event_name(em: dict, cont: dict, line: int | None = None, state: str = "", node: str = "") -> str:
    """'TTL1 on' for the loaded stream's own source; events of another processor, stream or Record
    Node say so ('Ripple Detector TTL2 on', 'Ripple Detector LFPStream-A TTL1 on',
    'Ripple Detector TTL3 on (Record Node 121)')."""
    parts = []
    src = str(em.get("source_processor") or "").strip()
    if src and src != str(cont.get("source_processor_name") or "").strip():
        parts.append(src)
    stream = em.get("stream_name")
    if stream and stream != cont.get("stream_name"):
        parts.append(str(stream))
    parts.append(f"TTL{line} {state}".strip() if line is not None else str(em.get("channel_name", em["folder_name"])).strip())
    return " ".join(parts) + (f" ({node})" if node else "")


def _oe_events(rec_dir: Path, event_meta: list[dict], cont: dict, t: np.ndarray, cont_sn: np.ndarray | None,
               stream_rates: dict[str, float], node: str = "", ttl_only: bool = False,
               clocks: dict | None = None) -> tuple[dict, dict, list[dict]]:
    """(events, labels, ttl) for every event channel of the recording, on the loaded stream's clock.

    Event times come from ``sample_numbers.npy``: looked up in the loaded stream's own sample
    numbers ``cont_sn`` for its own events (never as an offset from the first one: a stream that
    dropped samples has gaps in its sample numbers, and every event after a gap would land late),
    looked up the same way in another stream's sample numbers and time axis when ``clocks`` has
    them (stream name -> callable returning (sample_numbers, t, fs)), divided by the source
    stream's rate otherwise. ``timestamps.npy`` is the last
    resort, because some writers fill it with per-buffer values. ``ttl`` lists each TTL line as
    {name, source, stream, line, on, off} for pairing into epochs (see ``detect.online_detections``)."""
    events: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    ttl: list[dict] = []
    own = cont.get("stream_name")
    fs_own = float(cont["sample_rate"])
    for em in event_meta:
        folder = rec_dir / "events" / em["folder_name"]
        ts_path, sn_path = folder / "timestamps.npy", folder / "sample_numbers.npy"
        if not ts_path.exists() and not sn_path.exists():
            continue
        sn = np.load(sn_path) if sn_path.exists() else None
        size = sn.size if sn is not None else np.load(ts_path, mmap_mode="r").shape[0]
        if size == 0:
            continue
        stream = em.get("stream_name")
        fs_ev = float(em.get("sample_rate") or 0) or stream_rates.get(stream or "", 0.0)
        if sn is not None and cont_sn is not None and len(cont_sn) and not node and stream in (None, own) and fs_ev in (0.0, fs_own):
            sn = sn.astype(np.int64)
            # the recorded sample at or before each event (the first one for events before it), then
            # whatever is left over at the stream's rate: exact for recorded samples, and an event
            # inside a dropped stretch or outside the recording is placed by its distance from it
            k = np.clip(np.searchsorted(cont_sn, sn, "right") - 1, 0, t.size - 1)
            times = t[k] + (sn - np.asarray(cont_sn[k], dtype=np.int64)) / fs_own
        elif sn is not None and clocks and stream in clocks and clocks[stream]() is not None:
            c_sn, c_t, c_fs = clocks[stream]()
            sn = sn.astype(np.int64)
            k = np.clip(np.searchsorted(c_sn, sn, "right") - 1, 0, c_t.size - 1)
            times = c_t[k] + (sn - np.asarray(c_sn[k], dtype=np.int64)) / c_fs
        elif sn is not None and fs_ev:
            times = sn / fs_ev
        else:
            times = np.load(ts_path)
        times = np.asarray(times, dtype=float)
        if ttl_only and not (folder / "states.npy").exists():
            continue
        if em.get("type") == "string" or (folder / "text.npy").exists():
            text = [s.decode(errors="replace") if isinstance(s, bytes) else str(s) for s in np.load(folder / "text.npy")] \
                if (folder / "text.npy").exists() else []
            key = "messages" + (f" ({node})" if node else "")
            events[key] = times
            labels[key] = text if len(text) == times.size else [""] * times.size
            continue
        states_path = folder / "states.npy"
        if states_path.exists():
            states = np.load(states_path).astype(int)
            for line in np.unique(np.abs(states)):
                on, off = times[states == line], times[states == -line]
                if on.size:
                    events[_oe_event_name(em, cont, line, "on", node)] = on
                if off.size:
                    events[_oe_event_name(em, cont, line, "off", node)] = off
                ttl.append({"name": _oe_event_name(em, cont, line, "", node), "source": str(em.get("source_processor") or ""),
                            "stream": stream, "line": int(line), "on": on, "off": off, "node": node})
        else:
            events[_oe_event_name(em, cont, None, "", node)] = times
    return events, labels, ttl


def _same_line(a: dict, b: dict, tol_s: float = 0.002) -> bool:
    """Two TTL lines carrying the same pulses: the same source and line number, the same number of
    rising edges, and every edge within ``tol_s`` (one physical input recorded by two nodes)."""
    if a["source"] != b["source"] or a["line"] != b["line"] or a["on"].size != b["on"].size:
        return False
    return not a["on"].size or float(np.max(np.abs(np.sort(a["on"]) - np.sort(b["on"])))) <= tol_s


def _oe_sibling_events(rec_dir: Path, cont: dict, t: np.ndarray, own: list[dict]) -> tuple[dict, list[dict]]:
    """(events, ttl) recorded by the *other* Record Nodes of the same session.

    One acquisition is usually written by several Record Nodes (one per branch of a Splitter), and a
    processor's TTL output is only saved by the node downstream of it — so a line can be missing
    from the recording being viewed while a sibling has it. All nodes see the same acquisition
    clock, so those events belong here too: their times are ``sample_numbers.npy`` over the source
    stream's rate, the same absolute seconds this recording uses. A sibling is used only when one
    of its streams starts within a second of this one, and a line it shares with this recording
    (same source, line and edges) is not added twice. Only TTL lines are taken, not messages."""
    events: dict[str, np.ndarray] = {}
    ttl: list[dict] = []
    if len(rec_dir.parents) < 3 or not t.size:
        return events, ttl
    node_dir, session_dir = rec_dir.parents[1], rec_dir.parents[2]
    for ob in sorted(session_dir.glob("*/experiment*/recording*/structure.oebin")):
        other = ob.parent
        if other == rec_dir or ob.parents[2] == node_dir:
            continue
        try:
            meta = json.loads(ob.read_text())
        except (OSError, ValueError):
            continue
        conts = meta.get("continuous", [])
        starts = []
        for c in conts:
            p = other / "continuous" / c["folder_name"] / "sample_numbers.npy"
            try:
                starts.append(float(np.load(p, mmap_mode="r")[0]) / float(c["sample_rate"]))
            except (OSError, ValueError, IndexError, KeyError):
                pass
        if not any(abs(s - float(t[0])) < 1.0 for s in starts):     # another acquisition: not the same clock
            continue
        rates = {c.get("stream_name", ""): float(c["sample_rate"]) for c in conts}
        ev, _, tl = _oe_events(other, meta.get("events", []), cont, t, None, rates,
                               node=ob.parents[2].name, ttl_only=True)
        for line in tl:
            if any(_same_line(line, x) for x in own + ttl):
                continue
            ttl.append(line)
            for state in ("on", "off"):
                key = f"{line['name']} {state}" if not line["name"].endswith(")") else line["name"].replace(" (", f" {state} (")
                if key in ev:
                    events[key] = ev[key]
    return events, ttl


def find_extra_channels(ts: TimeSeries) -> tuple[str, str, list[str]] | None:
    """(recording path, stream, channel names) of the channels a processor added beside the
    electrodes — the Ripple Detector's ``RIP_FEAT`` and its thresholds — looking in the open
    stream first, then the recording's other streams, then the other Record Nodes of the session.

    In the 2026-09-23 recordings they are in the LFP stream of the node downstream of the
    detector, so a view of the broadband stream has to fetch them from there."""
    own = ts.channels_of("other")
    if own:
        return str(ts.source), ts.stream, own
    is_extra = lambda nm: bool(nm) and input_from_name(nm) is None and "AUX" not in nm.upper() and "ADC" not in nm.upper()
    # the processor writes them next to the electrodes it read, so a stream without electrodes
    # (the GLANCE IMU stream: quaternion, acceleration, gyroscope, and often empty) is not it
    has_electrodes = lambda names: any(input_from_name(n) is not None for n in names)
    for stream, names in (ts.recording_meta.get("stream_channels") or {}).items():
        if stream != ts.stream and has_electrodes(names) and any(is_extra(n) for n in names):
            return str(ts.source), stream, [n for n in names if is_extra(n)]
    rec_dir = Path(ts.source)
    if len(rec_dir.parents) < 3:
        return None
    node_dir, session_dir = rec_dir.parents[1], rec_dir.parents[2]
    for ob in sorted(session_dir.glob("*/experiment*/recording*/structure.oebin")):
        if ob.parent == rec_dir or ob.parents[2] == node_dir:
            continue
        try:
            meta = json.loads(ob.read_text())
        except (OSError, ValueError):
            continue
        for c in meta.get("continuous", []):
            names = [str(ch.get("channel_name", "")).strip() for ch in c.get("channels", [])]
            if not (has_electrodes(names) and any(is_extra(n) for n in names)):
                continue
            dat = ob.parent / "continuous" / c.get("folder_name", "") / "continuous.dat"
            if not dat.exists() or dat.stat().st_size == 0:
                continue
            return str(ob.parent), str(c.get("stream_name") or c["folder_name"]).strip("/"), [n for n in names if is_extra(n)]
    return None


def duplicate_channels(ts: TimeSeries, names: list[str] | None = None, sample_s: float = 1.0) -> list[list[str]]:
    """Groups of channels whose samples are bit-identical over a stretch from the middle of the
    recording. Seen in the GLANCE 3 kHz LFP stream, where inputs (1,2), (3,4)... are copies of each
    other, so such a stream holds half as many signals as it has channels."""
    names = names if names is not None else ts.headstage_channels()
    if len(names) < 2 or ts.n < 2:
        return []
    n = int(min(ts.n, max(sample_s * float(ts.fs or 1.0), 100)))
    lo = (ts.n - n) // 2
    blk = np.asarray(ts.data[lo:lo + n][:, [ts.index(c) for c in names]])
    groups: dict[bytes, list[str]] = {}
    for j, nm in enumerate(names):
        col = np.ascontiguousarray(blk[:, j])
        if col.min() != col.max():                  # flat (disconnected) channels are not "copies"
            groups.setdefault(col.tobytes(), []).append(nm)
    return [g for g in groups.values() if len(g) > 1]


def load_openephys(path: str | Path, stream: str | None = None) -> TimeSeries:
    """Memory-map one continuous stream of an Open Ephys binary recording.

    ``path`` may be the session root, the recording folder, or ``structure.oebin``.
    ``stream`` selects by stream or folder name when a recording has several;
    the first stream is used otherwise.

    Two layouts are handled by inspection, not by a format switch: recordings with blank channel
    names, mV aux channels and a valid ``timestamps.npy`` (names and headstage inputs then come
    from the Channel Map in settings.xml), and GLANCE recordings with named channels
    (``A_CH20``), ``a.u.`` aux channels, extra processor outputs (``RIP_FEAT``...), several
    streams per Record Node and an unusable ``timestamps.npy``. ``ts.format_notes`` says which
    choices were made.
    """
    oebin = find_oebin(path)
    rec_dir = oebin.parent
    meta = json.loads(oebin.read_text())
    conts = meta.get("continuous", [])
    if not conts:
        raise ValueError(f"{oebin} lists no continuous streams")
    if stream is not None:
        matches = [c for c in conts if stream in (c.get("stream_name", ""), c.get("folder_name", ""))
                   or stream in c.get("folder_name", "")]
        if not matches:
            raise ValueError(f"stream {stream!r} not in {[c.get('folder_name') for c in conts]}")
        cont = matches[0]
    else:
        cont = conts[0]

    fs = float(cont["sample_rate"])
    nch = int(cont["num_channels"])
    folder = rec_dir / "continuous" / cont["folder_name"]
    dat = folder / "continuous.dat"
    if not dat.exists() or dat.stat().st_size == 0:
        raise ValueError(f"stream {cont.get('stream_name') or cont['folder_name']!r} recorded no samples "
                         f"({dat.name} is empty): choose another stream")
    data = np.memmap(dat, dtype=np.int16, mode="r").reshape(-1, nch)
    n = data.shape[0]
    chans = cont.get("channels", [])
    if len(chans) != nch:
        chans = [{} for _ in range(nch)]
    raw_names = [str(c.get("channel_name", "")).strip() for c in chans]
    # The Channel Map in settings.xml is only consulted for channels whose name does not say which
    # input they hold (its streams are matched by channel count, so a name is the better witness).
    channel_map = _oe_channel_map(rec_dir, cont.get("stream_name"), nch) if not all(raw_names) else None
    names = _oe_channel_names(chans, channel_map)
    scale = np.array([float(c.get("bit_volts", 1.0)) for c in chans])
    units = [str(c.get("units", "")).strip() for c in chans]
    rates = {c.get("stream_name", ""): float(c["sample_rate"]) for c in conts}
    tick_rates = tuple(sorted(set(rates.values()) - {fs}, reverse=True))
    t, time_source = _oe_time_axis(folder, fs, n, tick_rates)
    sn_path = folder / "sample_numbers.npy"
    cont_sn = np.load(sn_path, mmap_mode="r") if sn_path.exists() and n else None
    if cont_sn is not None and cont_sn.shape[0] != n:
        cont_sn = None

    def stream_clock(c: dict):
        """(sample_numbers, t, fs) of another stream, built on first use: its events go through its own time axis."""
        memo: list = []
        def get():
            if not memo:
                f = rec_dir / "continuous" / c["folder_name"]
                p = f / "sample_numbers.npy"
                try:
                    c_sn = np.load(p, mmap_mode="r")
                    c_fs = float(c["sample_rate"])
                    c_t, _ = _oe_time_axis(f, c_fs, c_sn.shape[0], tuple(sorted(set(rates.values()) - {c_fs}, reverse=True)))
                    memo.append((c_sn, c_t, c_fs) if c_sn.shape[0] else None)
                except (OSError, ValueError, KeyError):
                    memo.append(None)
            return memo[0]
        return get
    clocks = {c.get("stream_name", ""): stream_clock(c) for c in conts if c is not cont}
    events, labels, ttl = _oe_events(rec_dir, meta.get("events", []), cont, t, cont_sn, rates, clocks=clocks)
    sib_ev, sib_ttl = _oe_sibling_events(rec_dir, cont, t, ttl)     # lines only this session's other Record Nodes recorded
    events.update(sib_ev); ttl += sib_ttl
    ts = TimeSeries(t, data, names, fs, source=str(rec_dir), scale=scale, units=units,
                    events=events, event_labels=labels)
    info = []
    for i, ch in enumerate(chans):
        kind = _oe_channel_kind(raw_names[i], units[i], str(ch.get("description", "")))
        if raw_names[i]:
            inp = input_from_name(raw_names[i]) if kind == "headstage" else None
        else:
            inp = (channel_map[i] + 1) if channel_map else None
        info.append({"recorded": i + 1, "input": inp, "kind": kind})
    ts.channel_info = info
    if len(conts) > 1:                  # keep each stream's saved results apart
        ts.results_subdir = str(cont.get("stream_name") or cont["folder_name"]).strip("/")
    ts.ttl = ttl
    ts.recording_meta = {"processors": oe_processor_params(rec_dir),
                         "signal_chain": oe_signal_chain(rec_dir, cont.get("stream_name")),
                         "stream_channels": {c.get("stream_name", ""): [str(ch.get("channel_name", "")).strip() for ch in c.get("channels", [])]
                                             for c in conts}}
    ts.stream = str(cont.get("stream_name") or "")
    ts.format_notes = {"time": time_source, "inputs": "channel names" if all(raw_names) else
                       ("channel map (settings.xml)" if channel_map else "recorded position"),
                       "streams": [c.get("stream_name") or c.get("folder_name") for c in conts]}
    dup = duplicate_channels(ts)
    if dup:
        ts.format_notes["warnings"] = [f"{sum(len(g) for g in dup)} headstage channels are exact copies of another "
                                       f"({', '.join('='.join(g) for g in dup[:4])}{'…' if len(dup) > 4 else ''}): "
                                       f"this stream holds {len(ts.headstage_channels()) - sum(len(g) - 1 for g in dup)} distinct signals"]
        ts.format_notes["duplicates"] = dup
    return ts


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def synthetic(duration: float = 20.0, fs: float = 1000.0, seed: int = 0) -> TimeSeries:
    """Demo signal: a few channels with structure at different scales, for trying the viewer."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    slow = np.sin(2 * np.pi * 0.5 * t)
    tone = 0.5 * np.sin(2 * np.pi * 40 * t)
    burst_env = np.exp(-((t % 4.0 - 2.0) ** 2) / 0.05)
    burst = burst_env * np.sin(2 * np.pi * 150 * t)
    noise = 0.3 * rng.standard_normal(t.size)
    steps = np.cumsum(rng.standard_normal(t.size)) / np.sqrt(fs)
    data = np.column_stack([slow + tone + burst + noise, burst + 0.2 * noise, steps])
    burst_times = np.arange(2.0, duration, 4.0)
    return TimeSeries(t, data, ["mixed", "bursts", "random_walk"], fs, source="synthetic",
                      units=["uV", "uV", "mm"], events={"burst centre": burst_times})
