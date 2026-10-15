"""Laminar sharp-wave ripple detection using every electrode and whole-session statistics.

The single-channel ``swr`` detector in :mod:`detect` needs the user to know which channel sits
in stratum pyramidale. This one reads all headstage channels (decimated to ~1.5 kHz in chunks,
so a 30 kHz recording is never held in memory), and for each session:

0. **Mains harmonics.** A 60 Hz notch upstream leaves 180 Hz, which sits mid ripple band and in
   the 2026-09-18 sessions carried 91-99 % of the band's power, differently on every contact (so a
   common average does not remove it). ``line_hz`` notches the harmonics inside the band first.
1. **Ripple channel.** Ripple-band (default 120-250 Hz) envelope per channel, z-scored robustly
   (median / MAD over the whole session). Artefacts and EMG raise the envelope on every channel
   at once, ripples only near pyramidale, so the *local* envelope is the channel's z minus the
   median z across the probe. The ripple channel is the one whose local envelope has the
   heaviest upper tail (``channel="auto"``), or the one the user picked.
2. **Candidates.** The local envelope averaged over the ripple channel and its neighbours within
   ``pool_um``, z-scored again, thresholded as epochs (enter / peak / duration / merge).
3. **Sharp wave.** A bipolar derivation that survives common-average referencing and common
   noise: the 2-40 Hz LFP of the contacts ``sw_near_um``..``sw_far_um`` deeper than the ripple
   channel (radiatum side) minus the contacts the same distance shallower (oriens side). A sharp
   wave is a negative deflection of that difference; ``sw_score`` is its robust z (sign flipped,
   so positive = sharp wave) within ±``sw_win_ms`` of the ripple peak.
4. **Rejection**, each kept in the result with its reason: ``global`` (the probe-wide median
   envelope is itself high: noise, EMG, chewing), ``line_noise`` (the hum's amplitude wanders, and
   those bursts survive a steady notch: more than ``line_max`` of the event's ripple-band power
   lies within 6 Hz of a mains harmonic, measured against the flanking baseline), ``movement`` (accelerometer), and
   ``no_sharp_wave`` when ``require_sw`` is on. With it off the score is stored for filtering in
   the Events tab.

Thresholds are in robust SDs of an envelope, whose distribution is skewed: in plain noise the
local z crosses 5 for 20 ms about 0.3 times a minute. Short or quiet sessions therefore always
yield a few "events"; trust ``layer_confident``, the sharp-wave score and the Events tab, not the
count. ``layer_confidence`` is how far the ripple layer stands out of the depth profile (in robust
SDs, after a neighbour median so one noisy contact cannot win); below ``LAYER_CONFIDENCE_MIN`` the
automatic channel is a guess and the result says so.

Without probe geometry the recorded order is taken as the depth order with unit spacing, and the
``*_um`` distances are read as numbers of channels / 25.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .analysis import Param, bandpass, envelope, moving_average, remove_line
from .detect import _epochs, detector, motion_z, zscore
from .io import TimeSeries

TARGET_HZ = 1500.0          # ripple band tops out at 250 Hz; 6x is comfortable for the envelope
CHUNK_S = 20.0
PAD_S = 0.5
ASSUMED_PITCH_UM = 25.0     # spacing assumed when there is no probe geometry
LAYER_CONFIDENCE_MIN = 4.0  # below this the automatic ripple channel is a guess; pick the channel by hand


def decimated(ts: TimeSeries, names: list[str], target_hz: float = TARGET_HZ, t0: float | None = None,
              t1: float | None = None, progress: Callable[[float, str], None] | None = None,
              span: tuple[float, float] = (0.0, 1.0)) -> tuple[np.ndarray, np.ndarray, float]:
    """(t, X, fs_d): channels ``names`` low-passed and decimated to about ``target_hz``.

    ``X`` is (n_channels, n) float32. Read in ``CHUNK_S`` pieces with ``PAD_S`` of context either
    side so the zero-phase filter's edges fall outside what is kept."""
    fs = float(ts.fs)
    step = max(1, int(fs // target_hz))
    lo = 0 if t0 is None else int(np.searchsorted(ts.t, t0, "left"))
    hi = ts.n if t1 is None else int(np.searchsorted(ts.t, t1, "right"))
    lo -= lo % step                                  # keep the decimation grid fixed whatever the window
    idx = np.arange(lo, hi, step)
    out = np.empty((len(names), idx.size), dtype=np.float32)
    if step == 1:
        out[:] = ts.block(lo, hi, names).T
        return ts.t[lo:hi], out, fs
    chunk, pad = int(CHUNK_S * fs) // step * step, int(PAD_S * fs) // step * step
    done = 0
    for c0 in range(lo, hi, chunk):
        c1 = min(c0 + chunk, hi)
        a, b = max(c0 - pad, 0), min(c1 + pad, ts.n)
        a += (lo - a) % step                         # stay on the decimation grid
        blk = ts.block(a, b, names).T
        for j in range(blk.shape[0]):
            blk[j] = bandpass(blk[j], fs, None, fs / step / 2.5)
        keep = blk[:, (c0 - a):(c1 - a):step]
        out[:, done:done + keep.shape[1]] = keep
        done += keep.shape[1]
        if progress:
            progress(span[0] + (span[1] - span[0]) * (c1 - lo) / max(hi - lo, 1), "reading and decimating")
    return ts.t[idx], out[:, :done], fs / step


def depth_order(ts: TimeSeries, layout: dict | None, channel: str | None = None) -> tuple[list[str], np.ndarray, bool]:
    """(names shallow -> deep, depth in µm increasing downward, has_geometry), bad channels dropped."""
    bad = set((layout or {}).get("bad") or [])
    if layout and layout.get("contacts"):
        contacts = layout["contacts"]
        by_name = {c["name"]: c for c in contacts}
        shank = by_name[channel]["shank"] if channel in by_name else contacts[0]["shank"]
        col = sorted((c for c in contacts if c["shank"] == shank and c["name"] not in bad), key=lambda c: (-c["y_um"], c["x_um"]))
        top = max(c["y_um"] for c in col) if col else 0.0
        return [c["name"] for c in col], np.array([top - c["y_um"] for c in col], dtype=float), True
    names = [n for n in ts.headstage_channels() if n not in bad]
    return names, np.arange(len(names), dtype=float) * ASSUMED_PITCH_UM, False


def _robust(x: np.ndarray, centre: float | None = None, scale: float | None = None) -> tuple[np.ndarray, float, float]:
    if centre is None:
        centre = float(np.median(x))
        scale = float(1.4826 * np.median(np.abs(x - centre))) or float(np.std(x)) or 1.0
    return (x - centre) / scale, centre, scale


def ripple_band(y: np.ndarray, fs: float, low_hz: float, high_hz: float, line_hz: float = 60.0) -> np.ndarray:
    """Ripple band with the mains harmonics inside it notched out (180 Hz is mid-band)."""
    return bandpass(remove_line(np.asarray(y, dtype=float), fs, line_hz, low_hz, high_hz), fs, low_hz, high_hz)


def ripple_envelopes(X: np.ndarray, fs: float, low_hz: float, high_hz: float, smooth_ms: float,
                     stats: dict | None = None, line_hz: float = 60.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """(Z, zmed, stats): robust-z ripple envelope per channel, and its median across channels."""
    Z = np.empty(X.shape, dtype=np.float32)
    st = dict(stats or {})
    for j in range(X.shape[0]):
        env = envelope(ripple_band(X[j], fs, low_hz, high_hz, line_hz), fs, smooth_ms / 1000.0)
        z, c, s = _robust(env, st.get(f"env_centre_{j}"), st.get(f"env_scale_{j}"))
        st[f"env_centre_{j}"], st[f"env_scale_{j}"] = c, s
        Z[j] = z
    return Z, np.median(Z, axis=0), st


def ripple_profile(Z: np.ndarray, zmed: np.ndarray) -> np.ndarray:
    """Per channel: mean of the top 0.5 % of the local envelope (z minus the probe-wide median).
    High where ripple-band bursts are large *and* not shared with the rest of the probe."""
    k = max(10, int(Z.shape[1] * 0.005))
    local = Z - zmed
    return np.array([float(np.mean(np.partition(local[j], -k)[-k:])) for j in range(Z.shape[0])])


def supported_profile(profile: np.ndarray, depth: np.ndarray, support_um: float = 50.0) -> np.ndarray:
    """Median of each contact's value with its neighbours within ``support_um``: a layer spans
    several contacts, a single noisy contact does not, so it cannot win on its own."""
    return np.array([float(np.median(profile[np.abs(depth - d) <= support_um + 1e-9])) for d in depth])


def layer_index(supported: np.ndarray, depth: np.ndarray, reach_um: float = 100.0) -> int:
    """Contact nearest the centre of mass of the bump around the supported profile's peak (the
    neighbour median makes a plateau, so the plain argmax would sit on its edge)."""
    k = int(np.argmax(supported))
    w = np.clip(supported - np.median(supported), 0.0, None) * (np.abs(depth - depth[k]) <= reach_um + 1e-9)
    if w.sum() <= 0:
        return k
    return int(np.argmin(np.abs(depth - float((w * depth).sum() / w.sum()))))


def layer_confidence(supported: np.ndarray) -> float:
    """Peak of the supported profile in robust SDs above the rest of the probe (> ~4 = a clear layer)."""
    med = float(np.median(supported))
    mad = float(1.4826 * np.median(np.abs(supported - med))) or 1e-9
    return (float(supported.max()) - med) / mad


def _pool(depth: np.ndarray, i: int, pool_um: float) -> np.ndarray:
    return np.flatnonzero(np.abs(depth - depth[i]) <= pool_um + 1e-9)


def _sw_groups(depth: np.ndarray, i: int, near_um: float, far_um: float) -> tuple[np.ndarray, np.ndarray]:
    d = depth - depth[i]
    return np.flatnonzero((d >= near_um) & (d <= far_um)), np.flatnonzero((d <= -near_um) & (d >= -far_um))


def sharp_wave_signal(X: np.ndarray, fs: float, depth: np.ndarray, i: int, near_um: float, far_um: float,
                      low_hz: float = 2.0, high_hz: float = 40.0) -> np.ndarray | None:
    """Deep-minus-shallow low-frequency LFP around ripple channel ``i`` (negative = sharp wave), or
    None when the ripple channel is too near an end of the probe to have both sides."""
    deep, shallow = _sw_groups(depth, i, near_um, far_um)
    if not deep.size or not shallow.size:
        return None
    return bandpass(X[deep].mean(axis=0).astype(float) - X[shallow].mean(axis=0).astype(float), fs, low_hz, high_hz)


def _band_spectrum(y: np.ndarray, fs: float, nfft: int = 1 << 14) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if y.size < 8:
        return np.fft.rfftfreq(nfft, d=1.0 / fs), np.zeros(nfft // 2 + 1)
    y = (y - y.mean()) * np.hanning(y.size)
    return np.fft.rfftfreq(nfft, d=1.0 / fs), np.abs(np.fft.rfft(y, n=nfft)) ** 2 / y.size


def line_fraction(x: np.ndarray, k: int, fs: float, low_hz: float, high_hz: float, line_hz: float,
                  half_s: float = 0.2, half_hz: float = 6.0) -> float:
    """Share of an event's *excess* ripple-band power that lies within ``half_hz`` of a mains harmonic.

    Excess = spectrum of the ±``half_s`` around sample ``k`` of the un-notched signal ``x`` minus the
    mean spectrum of the two flanking windows, so steady hum cancels. A burst of hum leaves a
    narrow excess at 180 Hz (score near 1); a ripple is tens of Hz wide and scores low even when it
    happens to be centred there. The window is wide (±200 ms) because a steady notch rings on
    either side of a large hum burst, and those side lobes are candidates too."""
    h = int(round(half_s * fs))
    f, ev = _band_spectrum(x[max(k - h, 0):k + h + 1], fs)
    flanks = [x[max(k - 3 * h, 0):max(k - h, 0)], x[k + h + 1:k + 3 * h + 1]]
    base = [_band_spectrum(seg, fs)[1] for seg in flanks if seg.size >= h]
    excess = np.clip(ev - (np.mean(base, axis=0) if base else 0.0), 0.0, None)
    band = (f >= low_hz) & (f <= high_hz)
    n = np.round(f / line_hz)
    near = band & (n >= 1) & (np.abs(f - n * line_hz) <= half_hz)
    total = float(excess[band].sum())
    return float(excess[near].sum() / total) if total > 0 else 0.0


def _signals(ts, layout, channel, p, stats=None, t0=None, t1=None, progress=None):
    """Everything the detector thresholds, on the decimated grid. ``stats`` (from a saved result)
    makes a short window reproduce the whole-session z-scores."""
    names, depth, geometry = depth_order(ts, layout, None if channel == "auto" else channel)
    if len(names) < 3:
        raise ValueError("laminar SWR detection needs at least 3 good headstage channels")
    t, X, fsd = decimated(ts, names, TARGET_HZ, t0, t1, progress, (0.05, 0.6))
    if progress:
        progress(0.62, "ripple envelopes")
    st = dict(stats or {})
    Z, zmed, est = ripple_envelopes(X, fsd, p["low_hz"], p["high_hz"], p["smooth_ms"], stats, p.get("line_hz", 60.0))
    st.update(est)
    if stats and "ripple_index" in stats:
        i = int(stats["ripple_index"])
        profile = None
    else:
        profile = ripple_profile(Z, zmed)
        supported = supported_profile(profile, depth)
        i = names.index(channel) if channel in names else layer_index(supported, depth)
        st["ripple_index"] = float(i)
        st["layer_confidence"] = layer_confidence(supported)
    pool = _pool(depth, i, p["pool_um"])
    local, c, s = _robust(np.mean(Z[pool] - zmed, axis=0), st.get("centre"), st.get("scale"))
    st["centre"], st["scale"], st["robust"] = c, s, 1.0
    sw = sharp_wave_signal(X, fsd, depth, i, p["sw_near_um"], p["sw_far_um"])
    swz = None
    if sw is not None:
        swz, c, s = _robust(-sw, st.get("sw_centre"), st.get("sw_scale"))
        st["sw_centre"], st["sw_scale"] = c, s
    return {"t": t, "fs": fsd, "names": names, "depth": depth, "geometry": geometry, "i": i, "pool": pool, "X": X,
            "Z": Z, "zmed": zmed, "local": local, "sw": sw, "swz": swz, "profile": profile,
            "supported": supported_profile(profile, depth) if profile is not None else None, "stats": st}


def swr_laminar_features(ts, channel, stats, layout=None, **p):
    s = _signals(ts, layout, channel, p, stats)
    i, fsd = s["i"], s["fs"]
    up = lambda v: np.interp(ts.t, s["t"], v)            # back to the window's sample rate
    out = {"raw": up(s["X"][i]), "ripple band": up(ripple_band(s["X"][i], fsd, p["low_hz"], p["high_hz"], p.get("line_hz", 60.0))),
           "local ripple z": up(s["local"]), "global ripple z": up(s["zmed"])}
    if s["swz"] is not None:
        out["sharp wave z"] = up(s["swz"])
    if p.get("gate_movement", True) and "move_centre" in stats:
        mz = motion_z(ts, p.get("move_hp_hz", 1.0), p.get("move_smooth_s", 0.5), stats)
        if mz is not None:
            out["movement z"] = np.interp(ts.t, mz[0], mz[1])
    if "RIP_FEAT" in ts.names:
        out["RIP_FEAT (recorded)"] = ts.channel("RIP_FEAT")
    return out


SCAN_POINTS = 6000


def scan_trace(t: np.ndarray, local: np.ndarray, zmed: np.ndarray, points: int = SCAN_POINTS) -> dict:
    """The whole-session detection signal, small enough to save with the result: the local and
    probe-wide ripple z reduced to ``points`` bins by their maximum, so no event is lost."""
    n = t.size
    step = max(1, int(np.ceil(n / points)))
    nb = n // step
    if nb == 0:
        return {"t": [], "local": [], "global": [], "bin_s": 0.0}
    cut = nb * step
    return {"t": np.round(t[:cut:step] + (t[step // 2] - t[0] if step > 1 else 0.0), 4).tolist(),
            "local": np.round(local[:cut].reshape(nb, step).max(axis=1), 2).tolist(),
            "global": np.round(zmed[:cut].reshape(nb, step).max(axis=1), 2).tolist(),
            "bin_s": float(step * (t[1] - t[0])) if n > 1 else 0.0}


def swr_laminar_thresholds(params: dict) -> dict:
    return {"local ripple z": {"z_enter": params["z_enter"], "z_peak": params["z_peak"]},
            "global ripple z": {"global_z": params["global_z"]}, "sharp wave z": {"sw_z": params["sw_z"]},
            "movement z": {"move_z": params["move_z"]}}


@detector("Sharp-wave ripples (laminar, all channels)", [
    Param("low_hz", "float", 120.0, min=80.0, max=300.0, step=5.0, help="ripple band"),
    Param("high_hz", "float", 250.0, min=100.0, max=400.0, step=5.0),
    Param("smooth_ms", "float", 8.0, min=0.0, max=50.0, step=0.5, help="envelope smoothing"),
    Param("line_hz", "float", 60.0, min=0.0, max=60.0, step=10.0, help="mains frequency: its harmonics inside the ripple band (120, 180, 240 Hz) are notched out; 0 disables"),
    Param("line_max", "float", 0.5, min=0.0, max=1.0, step=0.05, help="reject an event when more than this share of its ripple-band power sits within 6 Hz of a mains harmonic (a burst of hum, not a ripple)"),
    Param("pool_um", "float", 50.0, min=0.0, max=200.0, step=25.0, help="average the local envelope over contacts within this of the ripple channel"),
    Param("z_enter", "float", 2.5, min=0.0, max=10.0, step=0.1, help="event while the local ripple z (median / MAD over the session) exceeds this"),
    Param("z_peak", "float", 5.0, min=0.0, max=20.0, step=0.1, help="and its peak exceeds this"),
    Param("min_ms", "float", 20.0, min=0.0, max=200.0, step=1.0),
    Param("max_ms", "float", 400.0, min=0.0, max=1000.0, step=5.0, help="0 = no maximum"),
    Param("merge_ms", "float", 15.0, min=0.0, max=100.0, step=1.0),
    Param("global_z", "float", 4.0, min=0.0, max=30.0, step=0.5, help="reject when the probe-wide median ripple z exceeds this during the event (noise, EMG)"),
    Param("sw_near_um", "float", 75.0, min=25.0, max=300.0, step=25.0, help="sharp wave: contacts this far..."),
    Param("sw_far_um", "float", 300.0, min=50.0, max=800.0, step=25.0, help="...to this far below (minus above) the ripple channel"),
    Param("sw_win_ms", "float", 40.0, min=5.0, max=200.0, step=5.0, help="look for the sharp wave within this of the ripple peak"),
    Param("require_sw", "bool", False, help="reject events without a sharp wave; off = only store sw_score"),
    Param("sw_z", "float", 2.0, min=0.0, max=10.0, step=0.1, help="sharp wave present when its robust z exceeds this"),
    Param("gate_movement", "bool", True, help="reject events during movement (needs accelerometer / aux channels)"),
    Param("move_z", "float", 2.0, min=0.0, max=20.0, step=0.1),
    Param("move_hp_hz", "float", 1.0, min=0.1, max=20.0, step=0.1),
    Param("move_smooth_s", "float", 0.5, min=0.05, max=5.0, step=0.05),
], features=swr_laminar_features, thresholds=swr_laminar_thresholds, category="swr")
def swr_laminar(ts: TimeSeries, channel: str, layout=None, progress=None, **p):
    """Ripples localised in depth plus a sharp wave below them, from every good electrode, z-scored
    over the whole session. Channel 'auto' (or any channel not on the probe) lets the detector
    find the ripple layer; otherwise the chosen channel is used. See the module docstring."""
    s = _signals(ts, layout, channel, p, progress=progress)
    t, names, i = s["t"], s["names"], s["i"]
    if progress:
        progress(0.75, f"thresholding on {names[i]}")
    events = _epochs(s["local"], t, p["z_enter"], p["z_peak"], p["min_ms"] / 1000.0, p["max_ms"] / 1000.0, p["merge_ms"] / 1000.0)
    stats = s["stats"]
    mz = motion_z(ts, p["move_hp_hz"], p["move_smooth_s"]) if p["gate_movement"] else None
    if mz is not None:
        stats.update(mz[2])
    w = int(round(p["sw_win_ms"] / 1000.0 * s["fs"]))
    kept, rejected = [], []
    for e in events:
        a, b = int(np.searchsorted(t, e["start"], "left")), int(np.searchsorted(t, e["end"], "right"))
        k = int(np.searchsorted(t, e["peak"], "left"))
        e["global_score"] = float(s["zmed"][a:b].max()) if b > a else 0.0
        e["sw_score"] = float(s["swz"][max(k - w, 0):k + w + 1].max()) if s["swz"] is not None else None
        if mz is not None:
            e["move_score"] = float(np.interp(e["peak"], mz[0], mz[1]))
        if p["line_hz"] > 0:
            e["line_frac"] = line_fraction(s["X"][i], k, s["fs"], p["low_hz"], p["high_hz"], p["line_hz"])
        if e["global_score"] > p["global_z"]:
            e["reason"] = "global"
        elif e.get("line_frac", 0.0) > p["line_max"]:
            e["reason"] = "line_noise"
        elif mz is not None and e["move_score"] > p["move_z"]:
            e["reason"] = "movement"
        elif p["require_sw"] and e["sw_score"] is not None and e["sw_score"] < p["sw_z"]:
            e["reason"] = "no_sharp_wave"
        (rejected if "reason" in e else kept).append(e)
    deep, shallow = _sw_groups(s["depth"], i, p["sw_near_um"], p["sw_far_um"])
    extra = {"channel": names[i], "channel_requested": channel, "rejected": rejected, "n_rejected": len(rejected),
             "n_candidates": len(events), "movement_gate": mz is not None,
             "n_rejected_movement": sum(e["reason"] == "movement" for e in rejected),
             "n_rejected_global": sum(e["reason"] == "global" for e in rejected),
             "n_rejected_line_noise": sum(e["reason"] == "line_noise" for e in rejected),
             "n_rejected_no_sharp_wave": sum(e["reason"] == "no_sharp_wave" for e in rejected),
             "n_with_sharp_wave": sum(1 for e in kept if e["sw_score"] is not None and e["sw_score"] >= p["sw_z"]),
             "geometry": s["geometry"], "pool_channels": [names[j] for j in s["pool"]],
             "sw_deep_channels": [names[j] for j in deep], "sw_shallow_channels": [names[j] for j in shallow],
             "layer_confidence": stats.get("layer_confidence"),
             "layer_confident": bool(channel in names or stats.get("layer_confidence", 0.0) >= LAYER_CONFIDENCE_MIN),
             "ripple_profile": {"names": names, "depth_um": s["depth"].tolist(),
                                "value": s["profile"].tolist() if s["profile"] is not None else None,
                                "supported": s["supported"].tolist() if s["supported"] is not None else None}}
    extra["scan"] = scan_trace(t, s["local"], s["zmed"])
    return kept, stats, extra


# ---------------------------------------------------------------------------
# the standard offline ripple detector: normalised squared signal
# ---------------------------------------------------------------------------

NSS_RATE_HZ = 1250.0          # the LFP rate the method is written for
NSS_SMOOTH_MS = 8.8           # 11 samples at 1250 Hz, the window FMAToolbox's FindRipples uses


def nss_signal(ts: TimeSeries, channel: str, low_hz: float, high_hz: float, order: int,
               smooth_ms: float, target_hz: float = NSS_RATE_HZ, stats: dict | None = None,
               t0: float | None = None, t1: float | None = None, progress=None):
    """(t, band-passed signal, NSS, stats) for one channel.

    The signal is decimated to ~1250 Hz, band-passed (Butterworth, ``order``), **squared**,
    smoothed, and normalised to the mean and SD of that squared signal: the *normalised squared
    signal*. Pass ``stats`` to normalise a short window with the whole-recording reference instead
    of its own."""
    t, X, fsd = decimated(ts, [channel], target_hz, t0=t0, t1=t1, progress=progress)
    filt = bandpass(np.asarray(X[0], dtype=float), fsd, low_hz, high_hz, order=int(order))
    sq = filt ** 2
    win = int(round(smooth_ms / 1000.0 * fsd))
    if win > 1:
        sq = moving_average(sq, win)
    if stats is None:
        centre, scale = float(sq.mean()), float(sq.std())
        stats = {"centre": centre, "scale": scale or 1.0, "robust": 0.0}
    return t, filt, (sq - stats["centre"]) / (stats["scale"] or 1.0), stats


def nss_events(nss: np.ndarray, t: np.ndarray, low_sd: float, high_sd: float, min_ms: float,
               max_ms: float, merge_ms: float) -> tuple[list[dict], list[dict]]:
    """(kept, rejected). A candidate runs between the ``low_sd`` crossings either side of a peak
    over ``high_sd``; candidates closer than ``merge_ms`` are one event, and those outside the
    duration limits are rejected with reason ``duration`` rather than dropped, so they stay
    browsable in the Events tab."""
    above = nss > low_sd
    if not above.any():
        return [], []
    d = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, above.size)
    runs: list[list[int]] = []
    for a, b in zip(starts, ends):
        if runs and t[a] - t[runs[-1][1] - 1] < merge_ms / 1000.0:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    kept, rejected = [], []
    for a, b in runs:
        k = a + int(np.argmax(nss[a:b]))
        if nss[k] < high_sd:
            continue                      # never crossed the peak threshold: not a candidate at all
        dur = float(t[b - 1] - t[a])
        e = {"start": float(t[a]), "end": float(t[b - 1]), "peak": float(t[k]),
             "score": float(nss[k]), "duration": dur}
        if dur * 1000.0 < min_ms or (max_ms > 0 and dur * 1000.0 > max_ms):
            rejected.append({**e, "reason": "duration"})
        else:
            kept.append(e)
    return kept, rejected


def nss_features(ts, channel, stats, low_hz, high_hz, order, smooth_ms, **_):
    _, filt, nss, _ = nss_signal(ts, channel, low_hz, high_hz, order, smooth_ms, stats=stats)
    raw = ts.channel(channel)
    at = lambda y: np.interp(ts.t, np.linspace(float(ts.t[0]), float(ts.t[-1]), y.size), y)
    return {"raw": raw, "ripple band": at(filt), "NSS": at(nss)}


def nss_thresholds(params: dict) -> dict:
    return {"NSS": {"low_sd": params["low_sd"], "high_sd": params["high_sd"]}}


@detector("Ripples (normalised squared signal)", [
    Param("low_hz", "float", 120.0, min=80.0, max=300.0, step=5.0, help="ripple band"),
    Param("high_hz", "float", 200.0, min=100.0, max=400.0, step=5.0),
    Param("order", "int", 3, min=1, max=8, help="Butterworth order (scipy only)"),
    Param("smooth_ms", "float", NSS_SMOOTH_MS, min=0.0, max=50.0, step=0.4,
          help="boxcar over the squared signal before normalising; 8.8 ms = the 11 samples at 1250 Hz FindRipples uses"),
    Param("high_sd", "float", 5.0, min=0.0, max=20.0, step=0.1, help="a ripple peak exceeds this many SDs of the NSS"),
    Param("low_sd", "float", 2.0, min=0.0, max=10.0, step=0.1, help="its start and end are where the NSS falls back through this"),
    Param("min_ms", "float", 15.0, min=0.0, max=200.0, step=1.0, help="shorter events are rejected"),
    Param("max_ms", "float", 100.0, min=0.0, max=1000.0, step=5.0, help="longer ones too; 0 = no maximum"),
    Param("merge_ms", "float", 30.0, min=0.0, max=200.0, step=1.0, help="candidates closer than this are one event"),
    Param("rate_hz", "float", NSS_RATE_HZ, min=500.0, max=5000.0, step=50.0, help="the signal is decimated to about this first"),
    Param("gate_movement", "bool", False, help="also reject events during movement (needs aux channels)"),
    Param("move_z", "float", 2.0, min=0.0, max=20.0, step=0.1),
], features=nss_features, thresholds=nss_thresholds, category="swr")
def swr_nss(ts: TimeSeries, channel: str, layout=None, progress=None, **p):
    """The standard offline ripple detector (Buzsáki lab; FMAToolbox ``FindRipples``).

    The LFP is taken at ~1250 Hz, band-passed 120-200 Hz (Butterworth, order 3), squared, smoothed
    and normalised to its own mean and SD — the *normalised squared signal*. A ripple is a peak of
    the NSS above 5 SD; its start and end are where the NSS falls back through 2 SD, and it is kept
    when it lasts 15-100 ms. One channel, no laminar criterion and no sharp wave: with
    ``channel="auto"`` the contact whose ripple band carries the most power is used.

    Thresholds are SDs of a *squared* signal, which is far from normal, so "5 SD" is a convention
    of this method rather than a probability. Rejected events stay in the result with their reason.
    """
    low_hz, high_hz, order = p["low_hz"], p["high_hz"], int(p["order"])
    rate = float(p["rate_hz"])
    names = ts.headstage_channels() or list(ts.names)
    bad = set((layout or {}).get("bad") or [])
    good = [n for n in names if n not in bad] or names
    if channel in ("", "auto", None) or channel not in ts.names:
        if progress:
            progress(0.05, f"choosing a channel from {len(good)} electrodes")
        t, X, fsd = decimated(ts, good, rate, progress=progress, span=(0.05, 0.45))
        power = [float(np.std(bandpass(np.asarray(row, dtype=float), fsd, low_hz, high_hz, order=order)))
                 for row in X]
        channel = good[int(np.argmax(power))]
        chosen = {"channel": channel, "channel_power_uV": dict(zip(good, np.round(power, 2).tolist()))}
    else:
        chosen = {"channel": channel}
    if progress:
        progress(0.5, f"normalised squared signal on {channel}")
    t, _, nss, stats = nss_signal(ts, channel, low_hz, high_hz, order, p["smooth_ms"], rate,
                                  progress=progress)
    if progress:
        progress(0.85, "thresholding")
    kept, rejected = nss_events(nss, t, p["low_sd"], p["high_sd"], p["min_ms"], p["max_ms"], p["merge_ms"])
    if p.get("gate_movement"):
        mz = motion_z(ts, 1.0, 0.5)
        if mz is not None:
            mt, mv, mstats = mz
            stats.update(mstats)
            still = []
            for e in kept:
                m = float(np.max(np.interp([e["start"], e["peak"], e["end"]], mt, mv)))
                e["move_score"] = m
                if m > p["move_z"]:
                    rejected.append({**e, "reason": "movement"})
                else:
                    still.append(e)
            kept = still
            rejected.sort(key=lambda e: e["start"])
    extra = {**chosen, "rejected": rejected, "n_rejected": len(rejected),
             "scan": scan_trace(t, nss, np.zeros_like(nss)), "rate_hz": float(ts.fs and rate)}
    return kept, stats, extra
