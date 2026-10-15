"""Filtered views under development. Edit, then press Reload in the viewer.

    python serve.py ExampleSession --analyses filters_dev.py

Each function takes the visible window of the recording as ``ts`` (a
TimeSeries; ``ts.channel(name)`` gives float samples in physical units,
``ts.fs`` the sample rate) plus the parameters declared in its ``Param`` list,
and returns a Result of one or more traces. The window already includes a
little padding beyond what is shown, so filter edge effects fall off-screen.
"""

from __future__ import annotations

import numpy as np

from dataanalyzer.analysis import Param, Result, Trace, analysis, bandpass, envelope, moving_average


@analysis("Notch 60 Hz", [
    Param("notch_hz", "float", 60.0, min=1.0, max=1000.0, step=1.0, help="line frequency"),
    Param("q", "float", 30.0, min=1.0, max=200.0, step=1.0, help="quality factor; higher = narrower"),
    Param("harmonics", "int", 1, min=1, max=5, help="also remove 2f, 3f, ..."),
    Param("show_raw", "bool", True),
])
def notch(ts, channel, notch_hz, q, harmonics, show_raw):
    """Remove mains interference with an IIR notch (scipy) or a narrow FFT gap (fallback)."""
    fs = float(ts.fs)
    y = ts.channel(channel)
    out = y.copy()
    try:
        from scipy import signal
        for h in range(1, harmonics + 1):
            f0 = notch_hz * h
            if f0 >= fs / 2:
                break
            b, a = signal.iirnotch(f0, q, fs=fs)
            out = signal.filtfilt(b, a, out)
    except ImportError:
        spec = np.fft.rfft(out)
        f = np.fft.rfftfreq(out.size, 1 / fs)
        for h in range(1, harmonics + 1):
            f0 = notch_hz * h
            spec[np.abs(f - f0) < f0 / (2 * q)] = 0
        out = np.fft.irfft(spec, n=out.size)
    traces = [Trace(f"{channel} raw", ts.t, y)] if show_raw else []
    traces.append(Trace(f"{channel} notched", ts.t, out))
    return Result(traces)


@analysis("Ripple band power", [
    Param("low_hz", "float", 150.0, min=50.0, max=400.0, step=5.0),
    Param("high_hz", "float", 250.0, min=50.0, max=500.0, step=5.0),
    Param("smooth_s", "float", 0.02, min=0.0, max=0.5, step=0.005, help="power smoothing window"),
    Param("z_threshold", "float", 3.0, min=0.0, max=10.0, step=0.5, help="mark samples above this z-score"),
])
def ripple_power(ts, channel, low_hz, high_hz, smooth_s, z_threshold):
    """Band-pass, squared envelope, z-scored over the visible window, with crossings marked."""
    fs = float(ts.fs)
    filt = bandpass(ts.channel(channel), fs, low_hz, high_hz)
    power = moving_average(envelope(filt, fs) ** 2, max(1, int(round(smooth_s * fs))))
    z = (power - power.mean()) / (power.std() or 1.0)
    above = z > z_threshold
    onsets = np.flatnonzero(np.diff(above.astype(int)) == 1) + 1
    return Result([
        Trace(f"{channel} ripple power (z)", ts.t, z),
        Trace(f"threshold {z_threshold:g}", ts.t, np.full(ts.n, z_threshold)),
        Trace(f"crossings (n={onsets.size})", ts.t[onsets], None, kind="events"),
    ], ylabel="z")
