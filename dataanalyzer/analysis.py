"""Parameterised analyses on a single channel.

Each analysis is a function registered with :func:`analysis`, along with a list
of :class:`Param` describing its tunable inputs. The server exposes the schema
so the viewer can build a form for it, and calls the function with the
user-chosen values. Adding a new analysis means writing one function here.

Every analysis returns a :class:`Result`: one or more traces on a shared x
axis (time in seconds, or frequency in Hz). Event-style outputs use
``kind="events"`` and carry x positions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np

from .io import TimeSeries

try:  # scipy gives better filters; everything has a numpy fallback
    from scipy import signal as _sig
except Exception:  # pragma: no cover
    _sig = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class Param:
    name: str
    kind: str                       # "float" | "int" | "bool" | "choice" | "channel"
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    help: str = ""

    def coerce(self, raw: Any) -> Any:
        if raw is None:
            return self.default
        if self.kind == "float":
            v = float(raw)
        elif self.kind == "int":
            v = int(round(float(raw)))
        elif self.kind == "bool":
            v = str(raw).lower() in ("1", "true", "yes", "on")
        elif self.kind == "choice":
            v = str(raw)
            if self.choices and v not in self.choices:
                raise ValueError(f"{self.name}: {v!r} not one of {self.choices}")
        elif self.kind == "channel":            # a channel name, "auto", or "none"; validated by the server
            v = str(raw).strip()
        else:
            raise ValueError(f"unknown param kind {self.kind!r}")
        if self.min is not None and isinstance(v, (int, float)):
            v = max(v, self.min)
        if self.max is not None and isinstance(v, (int, float)):
            v = min(v, self.max)
        return v


@dataclass
class Trace:
    name: str
    x: np.ndarray
    y: np.ndarray | None = None     # None for events and images
    kind: str = "line"              # "line" | "events" | "image"
    f: np.ndarray | None = None     # image: frequency axis (rows)
    z: np.ndarray | None = None     # image: (n_f, n_x) values
    zlabel: str = ""                # image: colour-bar label
    log_f: bool = False             # image: draw the frequency axis logarithmically
    row_labels: list[str] | None = None   # image: label rows (e.g. channel names) instead of numeric ticks
    cmap: str = "viridis"           # image: "viridis" | "diverging" (symmetric about 0, blue = negative)


@dataclass
class Result:
    traces: list[Trace]
    domain: str = "time"            # "time" | "frequency"
    xlabel: str = "time (s)"
    ylabel: str = ""


@dataclass
class Analysis:
    name: str
    label: str
    fn: Callable[..., Result]
    params: list[Param] = field(default_factory=list)
    doc: str = ""
    plugin: str | None = None       # file path when loaded via load_plugin
    multichannel: bool = False      # uses every channel of the window; the channel choice is ignored

    @property
    def wants_layout(self) -> bool:
        return "layout" in self.fn.__code__.co_varnames[: self.fn.__code__.co_argcount + self.fn.__code__.co_kwonlyargcount]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "doc": self.doc,
            "plugin": self.plugin,
            "multichannel": self.multichannel,
            "params": [asdict(p) for p in self.params],
        }

    def coerce(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {p.name: p.coerce(raw.get(p.name)) for p in self.params}

    def run(self, ts: TimeSeries, channel: str, raw_params: dict[str, Any] | None = None) -> Result:
        return self.fn(ts, channel, **self.coerce(raw_params or {}))


REGISTRY: dict[str, Analysis] = {}


def analysis(label: str, params: list[Param] | None = None, multichannel: bool = False):
    """Decorator registering ``fn(ts, channel, **params) -> Result`` under ``fn.__name__``.

    A function with a ``layout`` keyword receives the probe geometry (see probe.Probe.layout) or
    None. ``multichannel=True`` marks analyses that use all channels of the window."""
    def wrap(fn: Callable[..., Result]) -> Callable[..., Result]:
        REGISTRY[fn.__name__] = Analysis(fn.__name__, label, fn, params or [], (fn.__doc__ or "").strip(),
                                         multichannel=multichannel)
        return fn
    return wrap


def schemas() -> list[dict]:
    return [a.schema() for a in REGISTRY.values()]


# ---------------------------------------------------------------------------
# Signal helpers (numpy fallbacks when scipy is absent)
# ---------------------------------------------------------------------------

def _fs(ts: TimeSeries) -> float:
    if ts.fs is None:
        raise ValueError("this analysis needs a regular sample rate")
    return float(ts.fs)


def moving_average(y: np.ndarray, width: int) -> np.ndarray:
    """Boxcar average with edge padding, O(n) via a cumulative sum (a 1 s window at 30 kHz is 30k taps)."""
    width = max(1, int(width))
    if width == 1:
        return y.astype(float, copy=True)
    pad = width // 2
    ypad = np.pad(np.asarray(y, dtype=float), (pad, width - 1 - pad), mode="edge")
    c = np.cumsum(np.insert(ypad, 0, 0.0))
    return (c[width:] - c[:-width]) / width


def bandpass(y: np.ndarray, fs: float, low: float | None, high: float | None, order: int = 4) -> np.ndarray:
    """Zero-phase band/low/high-pass. Butterworth via scipy, FFT brick-wall otherwise."""
    nyq = fs / 2.0
    low = None if not low or low <= 0 else min(low, nyq * 0.999)
    high = None if not high or high >= nyq else high
    if low is None and high is None:
        return y.copy()
    if _sig is not None:
        if low is not None and high is not None:
            sos = _sig.butter(order, [low, high], btype="band", fs=fs, output="sos")
        elif low is not None:
            sos = _sig.butter(order, low, btype="high", fs=fs, output="sos")
        else:
            sos = _sig.butter(order, high, btype="low", fs=fs, output="sos")
        return _sig.sosfiltfilt(sos, y)
    spec = np.fft.rfft(y - y.mean() if low else y)
    f = np.fft.rfftfreq(y.size, d=1.0 / fs)
    mask = np.ones_like(f, dtype=bool)
    if low is not None:
        mask &= f >= low
    if high is not None:
        mask &= f <= high
    return np.fft.irfft(spec * mask, n=y.size)


def remove_line(y: np.ndarray, fs: float, line_hz: float = 60.0, low: float = 0.0, high: float | None = None,
                width_hz: float = 3.0) -> np.ndarray:
    """Notch out the mains harmonics that fall between ``low`` and ``high`` (zero-phase).

    A 60 Hz notch upstream leaves 120/180/240 Hz untouched, and 180 Hz sits in the middle of the
    ripple band. ``width_hz`` is the -3 dB width of each notch; ``line_hz`` <= 0 disables."""
    if not line_hz or line_hz <= 0:
        return y
    high = min(high if high else fs / 2.0, fs / 2.0 - width_hz)
    harmonics = [k * line_hz for k in range(1, int(high // line_hz) + 2) if low - width_hz <= k * line_hz <= high + width_hz]
    if not harmonics:
        return y
    if _sig is not None:
        out = np.asarray(y, dtype=float)
        for f0 in harmonics:
            b, a = _sig.iirnotch(f0, f0 / width_hz, fs=fs)
            out = _sig.filtfilt(b, a, out)
        return out
    spec = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), d=1.0 / fs)
    for f0 in harmonics:
        spec[np.abs(f - f0) <= width_hz / 2.0] = 0.0
    return np.fft.irfft(spec, n=len(y))


def envelope(y: np.ndarray, fs: float, smooth_s: float = 0.0) -> np.ndarray:
    if _sig is not None:
        from scipy.fft import next_fast_len
        n = y.size
        env = np.abs(_sig.hilbert(y, N=next_fast_len(n)))[:n]     # avoid slow FFT lengths on long recordings
    else:
        env = np.abs(y)
        smooth_s = max(smooth_s, 3.0 / fs)
    if smooth_s > 0:
        env = moving_average(env, int(round(smooth_s * fs)))
    return env


def welch_psd(y: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(min(max(8, nperseg), y.size))
    if _sig is not None:
        return _sig.welch(y, fs=fs, nperseg=nperseg)
    step = nperseg // 2
    win = np.hanning(nperseg)
    scale = fs * (win ** 2).sum()
    acc = None
    count = 0
    for start in range(0, y.size - nperseg + 1, max(1, step)):
        seg = (y[start:start + nperseg] - y[start:start + nperseg].mean()) * win
        p = np.abs(np.fft.rfft(seg)) ** 2 / scale
        p[1:-1] *= 2
        acc = p if acc is None else acc + p
        count += 1
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, acc / max(count, 1)


def decimate_for(y: np.ndarray, fs: float, f_max: float) -> tuple[np.ndarray, float]:
    """Low-pass and subsample so the rate is ~4x f_max: a 30 kHz trace is far too fine for a
    500 Hz spectrogram and the cost drops accordingly. Returns (y, new_fs)."""
    step = int(fs // max(4.0 * f_max, 1.0))
    if step < 2:
        return y, fs
    return bandpass(y, fs, None, (fs / step) / 2.5)[::step], fs / step


def stft_power(y: np.ndarray, fs: float, win_s: float, overlap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, f, power) with a Hann window; power in units²/Hz like Welch."""
    nper = int(max(8, min(round(win_s * fs), y.size)))
    step = int(max(1, round(nper * (1.0 - overlap))))
    if _sig is not None:
        f, t, p = _sig.spectrogram(y, fs=fs, nperseg=nper, noverlap=nper - step, window="hann", scaling="density", mode="psd")
        return t, f, p
    win = np.hanning(nper)
    scale = fs * (win ** 2).sum()
    starts = np.arange(0, y.size - nper + 1, step)
    frames = np.stack([y[s:s + nper] * win for s in starts]) if starts.size else np.empty((0, nper))
    p = np.abs(np.fft.rfft(frames, axis=1)) ** 2 / scale
    p[:, 1:-1] *= 2
    return (starts + nper / 2) / fs, np.fft.rfftfreq(nper, 1 / fs), p.T


def morlet_power(y: np.ndarray, fs: float, freqs: np.ndarray, cycles: float) -> np.ndarray:
    """Morlet wavelet power (n_freqs, n) via one FFT of the signal and Gaussian filters in the
    frequency domain. Normalised so a unit-amplitude sine at f gives power ~1 at f."""
    n = y.size
    from scipy.fft import next_fast_len  # scipy present whenever this matters; numpy fallback below
    nfft = next_fast_len(2 * n)
    Y = np.fft.fft(y - y.mean(), nfft)
    w = 2 * np.pi * np.fft.fftfreq(nfft, 1 / fs)
    out = np.empty((freqs.size, n))
    for i, f0 in enumerate(freqs):
        sigma_t = cycles / (2 * np.pi * f0)
        H = np.exp(-0.5 * (sigma_t ** 2) * (w - 2 * np.pi * f0) ** 2)   # analytic Morlet, positive freqs only
        H[w < 0] = 0.0
        H *= 2.0
        out[i] = np.abs(np.fft.ifft(Y * H)[:n]) ** 2
    return out


# ---------------------------------------------------------------------------
# Registered analyses
# ---------------------------------------------------------------------------

@analysis("Raw signal")
def raw(ts: TimeSeries, channel: str) -> Result:
    """The channel as recorded."""
    return Result([Trace(channel, ts.t, ts.channel(channel))])


@analysis("Moving average", [
    Param("window_s", "float", 0.1, min=0.0, max=60.0, step=0.01, help="averaging window, seconds"),
    Param("show_raw", "bool", True, help="overlay the unsmoothed signal"),
])
def smooth(ts: TimeSeries, channel: str, window_s: float, show_raw: bool) -> Result:
    """Boxcar moving average with edge padding."""
    y = ts.channel(channel)
    width = int(round(window_s * _fs(ts)))
    traces = [Trace(f"{channel} raw", ts.t, y)] if show_raw else []
    traces.append(Trace(f"{channel} smoothed {window_s:g}s", ts.t, moving_average(y, width)))
    return Result(traces)


@analysis("Band-pass filter", [
    Param("low_hz", "float", 1.0, min=0.0, step=0.1, help="high-pass corner; 0 disables"),
    Param("high_hz", "float", 100.0, min=0.0, step=0.1, help="low-pass corner; 0 disables"),
    Param("order", "int", 4, min=1, max=8, help="Butterworth order (scipy only)"),
    Param("show_raw", "bool", False),
])
def filtered(ts: TimeSeries, channel: str, low_hz: float, high_hz: float, order: int, show_raw: bool) -> Result:
    """Zero-phase Butterworth band-pass (FFT brick-wall when scipy is missing)."""
    y = ts.channel(channel)
    out = bandpass(y, _fs(ts), low_hz or None, high_hz or None, order)
    traces = [Trace(f"{channel} raw", ts.t, y)] if show_raw else []
    traces.append(Trace(f"{channel} {low_hz:g}-{high_hz:g} Hz", ts.t, out))
    return Result(traces)


@analysis("Band envelope", [
    Param("low_hz", "float", 100.0, min=0.0, step=1.0),
    Param("high_hz", "float", 250.0, min=0.0, step=1.0),
    Param("smooth_s", "float", 0.01, min=0.0, max=5.0, step=0.001, help="post-smoothing of the envelope"),
    Param("show_filtered", "bool", True),
])
def band_envelope(ts: TimeSeries, channel: str, low_hz: float, high_hz: float, smooth_s: float, show_filtered: bool) -> Result:
    """Band-pass then analytic-signal amplitude envelope."""
    fs = _fs(ts)
    filt = bandpass(ts.channel(channel), fs, low_hz or None, high_hz or None)
    env = envelope(filt, fs, smooth_s)
    traces = [Trace(f"{channel} {low_hz:g}-{high_hz:g} Hz", ts.t, filt)] if show_filtered else []
    traces.append(Trace("envelope", ts.t, env))
    return Result(traces)


@analysis("Power spectrum (Welch)", [
    Param("nperseg", "int", 1024, min=16, max=65536, help="segment length in samples"),
    Param("max_hz", "float", 0.0, min=0.0, help="truncate the x axis; 0 shows to Nyquist"),
    Param("log_power", "bool", True, help="plot 10*log10(power)"),
])
def psd(ts: TimeSeries, channel: str, nperseg: int, max_hz: float, log_power: bool) -> Result:
    """Welch power spectral density of the visible window."""
    f, p = welch_psd(ts.channel(channel), _fs(ts), nperseg)
    if max_hz > 0:
        keep = f <= max_hz
        f, p = f[keep], p[keep]
    if log_power:
        p = 10 * np.log10(np.maximum(p, 1e-30))
    return Result([Trace(f"{channel} PSD", f, p)], domain="frequency", xlabel="frequency (Hz)",
                  ylabel="dB" if log_power else "power")


@analysis("Spectrogram (STFT)", [
    Param("win_s", "float", 0.25, min=0.005, max=10.0, step=0.005, help="window length; frequency resolution = 1 / window"),
    Param("overlap", "float", 0.75, min=0.0, max=0.95, step=0.05, help="fraction of window overlap"),
    Param("min_hz", "float", 0.0, min=0.0, step=1.0),
    Param("max_hz", "float", 300.0, min=1.0, step=10.0, help="top of the frequency axis; the signal is decimated to suit"),
    Param("log_power", "bool", True, help="10·log10 power"),
    Param("normalize", "bool", False, help="z-score each frequency row over the window (whitening, shows bursts)"),
    Param("log_freq", "bool", False),
])
def spectrogram(ts: TimeSeries, channel: str, win_s, overlap, min_hz, max_hz, log_power, normalize, log_freq) -> Result:
    """Short-time Fourier transform power as a heat map (time x frequency)."""
    fs = _fs(ts)
    y, fsd = decimate_for(ts.channel(channel), fs, max_hz)
    t, f, p = stft_power(y, fsd, win_s, overlap)
    keep = (f >= min_hz) & (f <= max_hz)
    if log_freq:
        keep &= f > 0
    f, p = f[keep], p[keep]
    z = 10 * np.log10(np.maximum(p, 1e-30)) if log_power else p
    if normalize:
        z = (z - z.mean(axis=1, keepdims=True)) / (z.std(axis=1, keepdims=True) + 1e-12)
    return Result([Trace(f"{channel} STFT", ts.t[0] + t, None, "image", f=f, z=z,
                         zlabel="z" if normalize else ("dB" if log_power else "power"), log_f=log_freq)],
                  ylabel="frequency (Hz)")


@analysis("Wavelet power (Morlet)", [
    Param("min_hz", "float", 2.0, min=0.1, step=0.5),
    Param("max_hz", "float", 300.0, min=1.0, step=10.0),
    Param("n_freqs", "int", 60, min=4, max=200),
    Param("cycles", "float", 7.0, min=2.0, max=30.0, step=0.5, help="wavelet width in cycles; more = sharper frequency, blurrier time"),
    Param("log_freq", "bool", True, help="log-spaced frequencies and axis"),
    Param("log_power", "bool", True),
    Param("normalize", "bool", False, help="z-score each frequency over the window"),
])
def wavelet(ts: TimeSeries, channel: str, min_hz, max_hz, n_freqs, cycles, log_freq, log_power, normalize) -> Result:
    """Morlet wavelet power (time x frequency); better time resolution at high frequencies than the STFT."""
    fs = _fs(ts)
    y, fsd = decimate_for(ts.channel(channel), fs, max_hz)
    max_hz = min(max_hz, fsd / 2.2)
    freqs = np.geomspace(min_hz, max_hz, n_freqs) if log_freq else np.linspace(min_hz, max_hz, n_freqs)
    p = morlet_power(y, fsd, freqs, cycles)
    z = 10 * np.log10(np.maximum(p, 1e-30)) if log_power else p
    if normalize:
        z = (z - z.mean(axis=1, keepdims=True)) / (z.std(axis=1, keepdims=True) + 1e-12)
    t = ts.t[0] + np.arange(y.size) / fsd
    return Result([Trace(f"{channel} wavelet", t, None, "image", f=freqs, z=z,
                         zlabel="z" if normalize else ("dB" if log_power else "power"), log_f=log_freq)],
                  ylabel="frequency (Hz)")


@analysis("Threshold events", [
    Param("band_low_hz", "float", 0.0, min=0.0, help="optional band-pass before detection; 0 = none"),
    Param("band_high_hz", "float", 0.0, min=0.0),
    Param("threshold_sd", "float", 3.0, min=0.0, step=0.1, help="threshold in standard deviations of the envelope"),
    Param("min_gap_s", "float", 0.05, min=0.0, step=0.01, help="merge crossings closer than this"),
    Param("use_envelope", "bool", True, help="detect on the amplitude envelope rather than the signal"),
])
def threshold_events(ts: TimeSeries, channel: str, band_low_hz: float, band_high_hz: float,
                     threshold_sd: float, min_gap_s: float, use_envelope: bool) -> Result:
    """Mark times where the (optionally filtered) signal exceeds a z-score threshold."""
    fs = _fs(ts)
    y = ts.channel(channel)
    if band_low_hz or band_high_hz:
        y = bandpass(y, fs, band_low_hz or None, band_high_hz or None)
    det = envelope(y, fs) if use_envelope else y
    z = (det - det.mean()) / (det.std() or 1.0)
    above = z > threshold_sd
    onsets = np.flatnonzero(np.diff(above.astype(int)) == 1) + 1
    if above.size and above[0]:
        onsets = np.insert(onsets, 0, 0)
    if onsets.size and min_gap_s > 0:
        keep = np.insert(np.diff(ts.t[onsets]) >= min_gap_s, 0, True)
        onsets = onsets[keep]
    return Result([
        Trace(f"{channel} detection signal (z)", ts.t, z),
        Trace(f"threshold {threshold_sd:g} sd", ts.t, np.full(ts.n, threshold_sd)),
        Trace(f"events (n={onsets.size})", ts.t[onsets], None, kind="events"),
    ], ylabel="z")


# ---------------------------------------------------------------------------
# Plugins: analyses under development live in their own file
# ---------------------------------------------------------------------------

def load_plugin(path) -> list[str]:
    """Execute a Python file so its ``@analysis`` decorators register (re-registering on reload).

    The file should import ``analysis``, ``Param``, ``Result`` and ``Trace`` from
    ``dataanalyzer.analysis``. Returns the analysis names defined by that file.
    """
    import importlib.util
    from pathlib import Path

    path = Path(path).resolve()
    before = set(REGISTRY)
    touched: list[str] = []
    spec = importlib.util.spec_from_file_location(f"dataanalyzer_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, a in REGISTRY.items():
        if name not in before or getattr(a.fn, "__module__", "") == module.__name__:
            a.plugin = str(path)
            touched.append(name)
    return touched


@analysis("Movement (accelerometer)", [
    Param("hp_hz", "float", 1.0, min=0.1, max=20.0, step=0.1, help="high-pass to remove gravity / posture"),
    Param("smooth_s", "float", 0.5, min=0.02, max=5.0, step=0.02, help="RMS smoothing window"),
    Param("show_axes", "bool", True, help="overlay the high-passed aux axes"),
])
def movement(ts: TimeSeries, channel: str, hp_hz: float, smooth_s: float, show_axes: bool) -> Result:
    """RMS of the high-passed acceleration across all aux channels; the channel choice is ignored."""
    from .detect import aux_channels, motion_signal
    aux = aux_channels(ts)
    if not aux:
        raise ValueError("no accelerometer (aux) channels in this recording")
    unit = ts.unit(aux[0]) or "a.u."
    traces = []
    if show_axes:
        fs = _fs(ts)
        for c in aux:
            traces.append(Trace(f"{c} hp", ts.t, bandpass(ts.channel(c), fs, hp_hz, None, order=2)))
    t, m = motion_signal(ts, hp_hz, smooth_s, aux)
    traces.append(Trace(f"movement ({unit} RMS)", t, m))
    return Result(traces, ylabel=unit)


def multitaper_spectrogram(y: np.ndarray, fs: float, win_s: float, step_s: float, nw: float = 3.0,
                           n_tapers: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre times, frequencies, power) of a multitaper spectrogram.

    Each window is multiplied by every Slepian (DPSS) taper in turn and the periodograms are
    averaged: several tapers give an estimate with far less variance than one window function, at
    the cost of smearing the spectrum over ``nw / win_s`` Hz. ``n_tapers`` defaults to 2*nw - 1,
    the usual choice. Without scipy there are no DPSS sequences, so a single Hann window is used
    and the result is an ordinary spectrogram (noisier, same units)."""
    n = int(round(win_s * fs))
    step = max(1, int(round(step_s * fs)))
    if n < 4 or y.size < n:
        return np.empty(0), np.empty(0), np.empty((0, 0))
    k = int(n_tapers or max(1, round(2 * nw - 1)))
    if _sig is not None:
        tapers = _sig.windows.dpss(n, nw, k)
    else:
        tapers = np.hanning(n)[None, :] / np.sqrt(np.sum(np.hanning(n) ** 2))
    starts = np.arange(0, y.size - n + 1, step)
    f = np.fft.rfftfreq(n, 1 / fs)
    out = np.empty((starts.size, f.size))
    for i, s in enumerate(starts):
        seg = y[s:s + n] - y[s:s + n].mean()
        spec = np.abs(np.fft.rfft(tapers * seg, axis=-1)) ** 2
        out[i] = spec.mean(axis=0) * (2.0 / fs)
    return (starts + n / 2) / fs, f, out


def band_power(f: np.ndarray, spec: np.ndarray, low: float, high: float) -> np.ndarray:
    """Power summed over ``low``-``high`` Hz for every row of a spectrogram."""
    k = (f >= low) & (f <= high)
    if not k.any():
        return np.zeros(spec.shape[0])
    return np.trapezoid(spec[:, k], f[k], axis=1) if hasattr(np, "trapezoid") else np.trapz(spec[:, k], f[k], axis=1)
