#!/usr/bin/env python3
"""Serve a time series file in the browser viewer.

    python serve.py                               # native folder dialog, then serve what it holds
    python serve.py ExampleSession                # one Open Ephys session (memory-mapped)
    python serve.py /data/sessions                # folder of sessions: switch between them in the viewer
    python serve.py ExampleSession --analyses my_filters.py   # your @analysis functions, reloadable
    python serve.py recording.csv                 # time column auto-detected
    python serve.py signal.npy --fs 30000
    python serve.py --demo                        # synthetic signal
    python serve.py data.csv --port 8010 --no-browser
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataanalyzer import load, serve, synthetic
from dataanalyzer import dialog
from dataanalyzer.dialog import pick_folder
from dataanalyzer.server import DataServer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", help="Open Ephys session directory, or a CSV/TSV, .npy or .npz file")
    p.add_argument("--stream", help="Open Ephys stream name when a recording has several")
    p.add_argument("--root", help="folder to scan for sessions (default: PATH itself when it is a directory)")
    p.add_argument("--dialog", action="store_true", help="with no PATH, open the folder dialog instead of starting empty")
    p.add_argument("--no-dialog", action="store_true", help=argparse.SUPPRESS)   # the old default; now the behaviour anyway
    p.add_argument("--probe", metavar="FILE|NAME",
                   help="probe geometry: probeinterface JSON, CSV (channel,x_um,y_um[,shank,site]), or a built-in "
                        "name (a1x32-poly2, a4x8); default: probe.json/probe.csv near the recording, else a1x32-poly2")
    p.add_argument("--analyses", action="append", default=[], metavar="FILE",
                   help="Python file with @analysis functions; repeatable; reloadable from the viewer")
    p.add_argument("--fs", type=float, help="sample rate in Hz (required for .npy, or CSV without a time column)")
    p.add_argument("--time-column", help="name of the CSV time column if it is not auto-detected")
    p.add_argument("--demo", action="store_true", help="serve a synthetic signal instead of a file")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.root) if args.root else None
    ts = None
    if args.demo:
        ts = synthetic()
    elif args.path:
        given = Path(args.path)
        if given.is_dir() and not args.stream:
            root = root or given            # the server picks the first recording under it
        else:
            ts = load(given, fs=args.fs, time_column=args.time_column, stream=args.stream)
            root = root or (given if given.is_dir() else given.parent)
    elif root is None and args.dialog:
        chosen = pick_folder()
        last = dialog.remembered_root()
        if chosen:
            root = Path(chosen)
        elif dialog.LAST_ERROR and last is not None:
            # the dialog could not be shown (not the same as Cancel): carry on with the last folder rather than nothing
            print(f"folder dialog failed ({dialog.LAST_ERROR}); opening the last folder instead: {last}")
            root = last
        else:
            print("no folder chosen" + (f" (dialog failed: {dialog.LAST_ERROR})" if dialog.LAST_ERROR else " (dialog cancelled)")
                  + "; use the Choose folder button in the viewer, or paste a path next to it")

    if root is None and ts is None:
        last = dialog.remembered_root()
        print("no folder loaded: choose one in the viewer (Choose folder…, or paste a path beside it)"
              + (f"\n  last folder used: {last}" if last is not None else ""))

    try:
        server = DataServer((args.host, args.port), ts, root=None, plugins=args.analyses, probe=args.probe)
    except OSError as e:
        if e.errno not in (48, 98):          # EADDRINUSE on macOS / Linux
            raise
        print(f"port {args.port} is already in use, probably by another DataAnalyzer server: open http://{args.host}:{args.port} "
              f"to use that one, stop it (see `lsof -i :{args.port}`), or start this one with --port {args.port + 1}", file=sys.stderr)
        return 1
    if root is not None:
        info = server.set_root(root, auto_open=ts is None)
        n = len(info["sessions"])
        print(f"root: {server.root}  ({n} session{'s' if n != 1 else ''} found)")
    if server.probe is not None:
        print(f"probe: {server.probe.name} ({server.probe.source})")
    if server.ts is not None:
        s = server.ts.summary()
        print(f"{s['source']}: {s['n_channels']} channels, {s['n']} samples, fs={s['fs']}, "
              f"{s['duration']:.2f}s, events={s['events'] or 'none'}")
    serve(server, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
