# DataAnalyzer

View and analyse Open Ephys time-series recordings (Intan RHD headstage, NeuroNexus A1x32-Poly2, GLANCE
sessions with an online Ripple Detector) through a small local HTTP server and a browser page, plus
Jupyter notebooks that redo the analyses step by step.

## Running

Only `numpy` is required (`scipy` optional; `spikeinterface` + `mountainsort5` for spike sorting).
No build step: run from the repository root with Python 3.13.

```
python serve.py                      # start empty and choose a folder in the viewer
python serve.py path/to/sessions     # a folder of sessions, or one recording
python serve.py --demo               # synthetic signal, no data needed
python tests/test_dataanalyzer.py    # all tests (plain script)
```

## Notebooks

- `notebooks/event_aligned.ipynb` — pick a recording and a TTL line, align on its rising or falling
  edge, cut the signal out around every event, and measure the latency from a reference line.
- `notebooks/event_aligned_<session>.ipynb` — that notebook run on one saline latency test, outputs saved.
- `notebooks/latency_comparison*.ipynb` — closed-loop latency compared across sessions: digital input
  (the test pulse, TTL6) → OpenEphys detection time (Ripple Detector, TTL8) → trigger (TTL2).
- `notebooks/detection_methods.ipynb` — the ripple, dentate-spike and theta detectors written out from
  scratch, checked against the viewer's results.

The recordings themselves are not in the repository; the notebooks keep their outputs so the results can
be read without them. `CLAUDE.md` documents the architecture, the recording formats and the findings.
