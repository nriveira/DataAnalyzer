"""Channel quality: find contacts whose noise or correlation with their depth neighbours makes
them outliers, so the CSD (a second spatial derivative, which amplifies any single-channel
oddity into a fake source/sink pair) can interpolate across them.

Automatic flags come from ``channel_quality``; the user can add or clear flags by hand and
both are kept in ``<recording>/dataanalyzer/channels/channels.json``.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analysis import bandpass
from .io import TimeSeries


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / (mad or (x.std() or 1.0))


def channel_quality(ts: TimeSeries, layout: dict | None, sample_s: float = 30.0, noise_z: float = 2.5,
                    corr_min: float = 0.35, n_samples: int = 3, min_ratio: float = 1.5, corr_drop: float = 0.12) -> dict:
    """Stats per headstage channel from a few short samples spread over the recording.

    ``hf_noise``: MAD of the >300 Hz signal (µV). ``lf_std``: std of the 1-300 Hz signal.
    ``neighbour_corr``: best correlation of the 1-300 Hz signal with the adjacent contacts on
    the probe (recorded neighbours without geometry), so a good contact beside a bad one is not
    dragged down. A channel is flagged when its log noise
    or log std is an outlier (robust z beyond ``noise_z`` either way) or its neighbour
    correlation falls below ``corr_min`` while most channels correlate well."""
    fs = float(ts.fs)
    names = ts.headstage_channels()
    if not names:
        return {"names": [], "stats": {}, "auto": [], "params": {}}
    if layout and layout.get("contacts"):
        order = sorted((c for c in layout["contacts"] if c["name"] in names), key=lambda c: (c["shank"], -c["y_um"], c["x_um"]))
        ordered = [c["name"] for c in order]
        shank_of = {c["name"]: c["shank"] for c in order}
        ordered += [n for n in names if n not in shank_of]
    else:
        ordered, shank_of = list(names), {}
    n = ts.n
    seg = int(min(sample_s * fs, n))
    starts = np.linspace(0, max(0, n - seg), n_samples).astype(int) if n > seg else [0]
    hf = np.zeros(len(ordered)); lf = np.zeros(len(ordered)); corr = np.zeros(len(ordered)); cnt = np.zeros(len(ordered))
    for s0 in starts:
        blk = ts.block(int(s0), int(s0) + seg, ordered)
        low = np.vstack([bandpass(blk[:, j], fs, 1.0, min(300.0, fs / 2.5)) for j in range(len(ordered))])
        high = blk.T - low if fs > 700 else np.zeros_like(low)
        hf += np.median(np.abs(high - np.median(high, axis=1, keepdims=True)), axis=1) * 1.4826
        lf += low.std(axis=1)
        c = np.corrcoef(low)
        for j, nm in enumerate(ordered):
            nb = [k for k in (j - 1, j + 1) if 0 <= k < len(ordered) and shank_of.get(ordered[k]) == shank_of.get(nm)]
            if nb:   # best neighbour: a good contact beside a bad one still matches its other side
                corr[j] += float(max(c[j, k] for k in nb)); cnt[j] += 1
    m = len(starts)
    hf, lf = hf / m, lf / m
    corr = np.where(cnt > 0, corr / np.maximum(cnt, 1), np.nan)

    def local_dev(v):
        """log deviation of each channel from the mean of its probe neighbours; a laminar
        gradient (larger LFP near the tip) is not an outlier, a single odd channel is.
        Returns (deviation in log units, robust z of it)."""
        lv = np.log(v + 1e-9)
        dev = np.zeros_like(lv)
        for j, nm in enumerate(ordered):
            nb = [k for k in (j - 2, j - 1, j + 1, j + 2) if 0 <= k < len(ordered) and shank_of.get(ordered[k]) == shank_of.get(nm)]
            dev[j] = lv[j] - (np.mean(lv[nb]) if nb else lv[j])
        return dev, _robust_z(dev)

    floor = math.log(min_ratio)          # a flag needs a real difference, not just a tiny one that happens to be rare
    d_hf, z_hf = local_dev(hf) if fs > 700 else (np.zeros_like(hf), np.zeros_like(hf))
    d_lf, z_lf = local_dev(lf)
    good_corr = np.nanmedian(corr) if np.isfinite(corr).any() else np.nan
    flags = {}
    for j, nm in enumerate(ordered):
        reasons = []
        if abs(z_hf[j]) > noise_z and abs(d_hf[j]) > floor:
            reasons.append("hf noise vs neighbours" if d_hf[j] > 0 else "hf too quiet vs neighbours")
        if abs(z_lf[j]) > noise_z and abs(d_lf[j]) > floor:
            reasons.append("lfp amplitude vs neighbours" if d_lf[j] > 0 else "lfp flat vs neighbours")
        if np.isfinite(corr[j]) and np.isfinite(good_corr) and good_corr > 0.6 and (corr[j] < corr_min or corr[j] < good_corr - corr_drop):
            reasons.append("uncorrelated with neighbours")
        if reasons:
            flags[nm] = reasons
    stats = {nm: {"hf_noise": float(hf[j]), "lf_std": float(lf[j]), "neighbour_corr": None if not np.isfinite(corr[j]) else float(corr[j]),
                  "z_hf": float(z_hf[j]), "z_lf": float(z_lf[j])} for j, nm in enumerate(ordered)}
    return {"names": ordered, "stats": stats, "auto": flags, "median_corr": None if not np.isfinite(good_corr) else float(good_corr),
            "params": {"sample_s": sample_s, "noise_z": noise_z, "corr_min": corr_min, "n_samples": n_samples,
                       "min_ratio": min_ratio, "corr_drop": corr_drop},
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def interpolate_bad(rows: np.ndarray, names: list[str], bad: list[str] | set[str] | None) -> np.ndarray:
    """Replace rows of ``bad`` channels (rows ordered along the probe) by the mean of the nearest
    good rows above and below; edges copy their nearest good neighbour."""
    if not bad:
        return rows
    bad = set(bad)
    out = np.array(rows, dtype=float, copy=True)
    good = [i for i, n in enumerate(names) if n not in bad]
    if not good:
        return out
    for i, n in enumerate(names):
        if n not in bad:
            continue
        above = max((g for g in good if g < i), default=None)
        below = min((g for g in good if g > i), default=None)
        if above is not None and below is not None:
            w = (i - above) / (below - above)
            out[i] = (1 - w) * rows[above] + w * rows[below]
        else:
            out[i] = rows[above if above is not None else below]
    return out


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def channels_path(ts: TimeSeries) -> Path:
    from .detect import results_dir
    return results_dir(ts, "channels") / "channels.json"


def load(ts: TimeSeries) -> dict | None:
    p = channels_path(ts)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def save(ts: TimeSeries, state: dict) -> Path:
    p = channels_path(ts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=1))
    return p


def effective_bad(state: dict | None) -> list[str]:
    """Automatic flags plus manual additions, minus manual clears."""
    if not state:
        return []
    auto = set((state.get("quality") or {}).get("auto", {}))
    manual = state.get("manual", {})
    return sorted((auto | set(manual.get("bad", []))) - set(manual.get("good", [])))
