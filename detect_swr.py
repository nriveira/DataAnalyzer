#!/usr/bin/env python3
"""Laminar sharp-wave ripple detection over one recording or a folder of them, saved beside each
(see dataanalyzer/swr.py), with the comparison against online detections recorded in the data.

    python detect_swr.py DATA                          # every recording under DATA, broadband stream
    python detect_swr.py DATA --only "Record Node 110" # only paths containing this text
    python detect_swr.py DATA --only "Record Node 110" --min-s 60   # and only recordings longer than a minute
    python detect_swr.py DATA/2026-09-18_09-15-01 --channel A_CH11 --set z_peak=6 --set require_sw=1
    python detect_swr.py DATA --set low_hz=120 --set high_hz=170 --tag 120-170   # kept beside the default result
    python detect_swr.py DATA --dry-run                # do not write results

The viewer loads what this saves (Browse tab spans, Events tab with its feature table, Compare detections).
"""

from __future__ import annotations

import argparse
import sys

from dataanalyzer import detect, features, swr  # noqa: F401  (swr registers the detector)
from dataanalyzer.server import DataServer


def pick_stream(streams: list[str], want: str | None) -> str | None:
    if want:
        return next((s for s in streams if want.lower() in s.lower()), None)
    return next((s for s in streams if "broadband" in s.lower()), streams[0] if streams else None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="a recording, a session, or a folder of sessions")
    p.add_argument("--stream", help="text the stream name must contain (default: the broadband stream, else the first)")
    p.add_argument("--only", help="only recordings whose path contains this text, e.g. 'Record Node 110'")
    p.add_argument("--min-s", type=float, default=0.0, help="skip recordings shorter than this many seconds")
    p.add_argument("--channel", default="auto", help="ripple channel (default: auto = find the ripple layer)")
    p.add_argument("--set", action="append", default=[], metavar="PARAM=VALUE", help="detector parameter, repeatable")
    p.add_argument("--tol-ms", type=float, default=50.0, help="matching tolerance for the online comparison")
    p.add_argument("--probe", help="probe file or built-in name (default: probe.json/csv near the recording, else assumed)")
    p.add_argument("--no-features", action="store_true", help="skip the per-event features (peak Hz, cycles, hum share, sharp wave...) shown in the Events tab")
    p.add_argument("--detector", default="swr_nss", choices=["swr_nss", "swr_laminar", "swr"],
                   help="swr_nss (default): the standard offline method, 120-200 Hz normalised squared signal; "
                        "swr_laminar: this project's depth-aware detector")
    p.add_argument("--tag", help="save as <detector>__<tag> next to the default result instead of replacing it, e.g. --tag 120-170")
    p.add_argument("--dry-run", action="store_true", help="do not save the results")
    a = p.parse_args(argv)
    params = dict(kv.split("=", 1) for kv in a.set)

    srv = DataServer(("127.0.0.1", 0), None, probe=a.probe) if a.probe else DataServer(("127.0.0.1", 0), None)
    import dataanalyzer.server as _s
    _s.remember_root = lambda path: None            # a batch run should not change the viewer's remembered folder
    sessions = [s for s in srv.set_root(a.path, auto_open=False)["sessions"] if s["kind"] == "openephys"]
    if a.only:
        sessions = [s for s in sessions if a.only in s["path"]]
    short = [s for s in sessions if s.get("duration_s") is not None and s["duration_s"] < a.min_s]
    sessions = [s for s in sessions if s not in short]
    if short:
        print(f"skipping {len(short)} recordings shorter than {a.min_s:g} s: " + ", ".join(f"{s['label']} ({s['duration_s']:.0f} s)" for s in short))
    if not sessions:
        print("no Open Ephys recordings found", file=sys.stderr)
        return 1
    print(f"{'recording':44s} {'stream':18s} {'dur s':>6s} {'channel':>10s} {'conf':>5s} {'kept':>5s} {'w/ SW':>5s} {'rej':>4s}  online (n: both / on rejected / online only)")
    for s in sessions:
        stream = pick_stream(s["streams"], a.stream)
        try:
            srv.open_session(s["path"], stream)
            ts = srv.ts
            for w in ts.format_notes.get("warnings", []):
                print(f"  ! {w}")
            res = detect.run(ts, a.detector, a.channel, params, save=False, layout=srv.probe_layout(), tag=a.tag)
            if not a.no_features:       # a fresh detection has none, and the Events tab table and filters need them
                features.event_features(ts, srv.probe_layout(), res, srv.layers, srv.spikes)
            if not a.dry_run:
                res["path"] = str(detect.save_result(ts, res))
        except Exception as e:                      # keep going: one bad recording should not stop the batch
            print(f"{s['label'][:44]:44s} {str(stream)[:18]:18s} ERROR {type(e).__name__}: {e}")
            continue
        online = []
        for name, r in detect.online_detections(ts).items():
            c = detect.compare(res, r, a.tol_ms)
            online.append(f"{r['stream']} TTL{r['line']} {c['n_b']}: {c['n_matched']} / {c['n_b_on_rejected']} / {c['n_only_b']}")
        conf = res.get("layer_confidence") or 0.0
        mark = ' ' if res.get("layer_confident", True) else '?'
        sw = res.get("n_with_sharp_wave")
        print(f"{s['label'][:44]:44s} {str(stream)[:18]:18s} {ts.duration:6.0f} {res['channel']:>10s} {conf:4.1f}{mark} "
              f"{res['n_events']:5d} {('%5d' % sw) if sw is not None else '    -'} {res.get('n_rejected', 0):4d}  " + ("; ".join(online) or "none recorded"))
    srv.server_close()
    print("\nconf = how far the ripple layer stands out of the depth profile (robust SDs); '?' = below "
          f"{swr.LAYER_CONFIDENCE_MIN:g}, the channel is a guess: inspect the Events tab or pass --channel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
