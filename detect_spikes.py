#!/usr/bin/env python3
"""Detect spikes in a recording and save them beside it (see dataanalyzer/spikes.py).

    python detect_spikes.py ExampleSession
    python detect_spikes.py ExampleSession --channels CH20,CH13 --threshold 5 --band 300 6000
    python detect_spikes.py ExampleSession --backend spikeinterface     # if installed
"""

from __future__ import annotations

import argparse
import sys

from dataanalyzer import load, spikes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="Open Ephys session/recording folder (or a data file)")
    p.add_argument("--stream")
    p.add_argument("--channels", help="comma list of channel names; default: all µV channels")
    p.add_argument("--threshold", type=float, default=4.5, help="in robust noise SD units (default 4.5)")
    p.add_argument("--band", type=float, nargs=2, default=(300.0, 6000.0), metavar=("LOW", "HIGH"))
    p.add_argument("--refractory-ms", type=float, default=1.0)
    p.add_argument("--polarity", choices=["neg", "pos", "both"], default="neg")
    p.add_argument("--chunk-s", type=float, default=20.0)
    p.add_argument("--backend", choices=["numpy", "spikeinterface"], default="numpy")
    a = p.parse_args(argv)

    ts = load(a.path, stream=a.stream)
    chans = [c.strip() for c in a.channels.split(",")] if a.channels else None

    def progress(frac, msg):
        print(f"\r{frac * 100:5.1f}%  {msg}          ", end="", flush=True)

    res = spikes.detect_spikes(ts, chans, a.threshold, tuple(a.band), a.refractory_ms, a.polarity, a.chunk_s,
                               progress, backend=a.backend)
    out = spikes.save(ts, res)
    print()
    s = spikes.summary(res)
    for name, n in zip(s["channel_names"], s["counts"]):
        print(f"  {name:10s} {n:8d}")
    print(f"{s['n_spikes']} spikes in {res['meta'].get('elapsed_s', '?')} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
