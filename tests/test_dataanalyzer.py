#!/usr/bin/env python3
"""Correctness checks on synthetic signals plus an end-to-end pass over the HTTP API.

    python tests/test_dataanalyzer.py            # all
    python tests/test_dataanalyzer.py server     # only tests whose name contains "server"
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os  # noqa: E402
# the server remembers the last data root in the user's config; tests must not overwrite it
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="dataanalyzer_test_config_")

from dataanalyzer import analysis as an  # noqa: E402
from dataanalyzer import io as dio  # noqa: E402
from dataanalyzer import channels, csd, dentate, detect, features, live, phase, probe, sections, server, sorting, spikes, stim, swr, triggered  # noqa: E402

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def power_at(y, fs, f0, bw=2.0):
    spec = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(y.size, 1 / fs)
    return spec[(f > f0 - bw) & (f < f0 + bw)].sum()


# ---------------------------------------------------------------- signal processing

@test
def bandpass_keeps_inband_and_rejects_outofband():
    fs, t = 1000.0, np.arange(0, 10, 1 / 1000.0)
    y = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 200 * t)
    out = an.bandpass(y, fs, 5, 50)
    ratio_in = power_at(out, fs, 10) / power_at(y, fs, 10)
    ratio_out = power_at(out, fs, 200) / power_at(y, fs, 200)
    assert ratio_in > 0.9, ratio_in
    assert ratio_out < 1e-3, ratio_out
    assert out.shape == y.shape


@test
def moving_average_preserves_length_and_constants():
    y = np.full(500, 3.0)
    assert np.allclose(an.moving_average(y, 25), 3.0)
    ramp = np.arange(100.0)
    sm = an.moving_average(ramp, 5)
    assert sm.shape == ramp.shape and np.allclose(sm[10:-10], ramp[10:-10])


@test
def envelope_tracks_burst_amplitude():
    fs = 1000.0
    t = np.arange(0, 4, 1 / fs)
    amp = np.where((t > 1) & (t < 2), 2.0, 0.2)
    env = an.envelope(amp * np.sin(2 * np.pi * 150 * t), fs, smooth_s=0.02)
    assert 1.6 < env[(t > 1.2) & (t < 1.8)].mean() < 2.4
    assert env[(t > 2.5)].mean() < 0.5


@test
def psd_peak_is_at_the_tone():
    fs = 1000.0
    t = np.arange(0, 8, 1 / fs)
    f, p = an.welch_psd(np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.default_rng(1).standard_normal(t.size), fs, 1024)
    assert abs(f[np.argmax(p)] - 60) < 1.5


@test
def spectrogram_and_wavelet_find_the_tone_and_serialise():
    fs = 1000.0
    t = np.arange(0, 8, 1 / fs)
    y = np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.default_rng(2).standard_normal(t.size)
    y[(t > 4) & (t < 4.1)] += 3 * np.sin(2 * np.pi * 200 * t[(t > 4) & (t < 4.1)])     # a 100 ms burst at 200 Hz
    ts = dio.TimeSeries(t, y, ["x"], fs)
    for name in ("spectrogram", "wavelet"):
        res = an.REGISTRY[name].run(ts, "x", {"max_hz": 300})
        tr = res.traces[0]
        assert tr.kind == "image" and tr.z.shape == (tr.f.size, tr.x.size) and res.ylabel.startswith("frequency")
        assert abs(tr.f[np.argmax(tr.z.mean(axis=1))] - 60) < 3, name        # dominant tone
        col = tr.z[:, (tr.x > 4.02) & (tr.x < 4.08)].mean(axis=1) - tr.z.mean(axis=1)
        assert abs(tr.f[np.argmax(col)] - 200) < 15, (name, tr.f[np.argmax(col)])   # burst shows up at 200 Hz
        j = server.result_to_json(res, 3.5, 4.5, 400)["traces"][0]
        assert j["kind"] == "image" and len(j["x"]) <= 400 and len(j["z"]) == len(j["f"]) and len(j["z"][0]) == len(j["x"])
        assert 3.4 <= j["x"][0] and j["x"][-1] <= 4.6 and j["zmax"] > j["zmin"]
    # decimation keeps the rate ~4x max_hz and the 30 kHz path works
    yd, fsd = an.decimate_for(np.random.default_rng(0).standard_normal(300_000), 30000.0, 300.0)
    assert fsd == 1200.0 and yd.size == 12_000


@test
def threshold_events_finds_synthetic_bursts():
    ts = dio.synthetic(duration=20, fs=1000)
    res = an.REGISTRY["threshold_events"].run(ts, "bursts", {"band_low_hz": 100, "band_high_hz": 250, "threshold_sd": 3, "min_gap_s": 0.5})
    ev = [tr for tr in res.traces if tr.kind == "events"][0]
    assert ev.x.size == 5, ev.x
    assert np.allclose(ev.x, [2, 6, 10, 14, 18], atol=0.3), ev.x


@test
def param_coercion_clamps_and_defaults():
    p = an.Param("w", "float", 1.0, min=0.0, max=2.0)
    assert p.coerce(None) == 1.0 and p.coerce("5") == 2.0 and p.coerce(-1) == 0.0
    assert an.Param("b", "bool", False).coerce("true") is True
    assert an.Param("k", "int", 3).coerce("4.6") == 5


@test
def decimate_minmax_preserves_extremes():
    x = np.linspace(0, 1, 100_000)
    y = np.sin(2 * np.pi * 3 * x)
    y[12345] = 50.0
    y[67890] = -50.0
    dx, dy = server.decimate_minmax(x, y, 2000)
    assert dx.size <= 2000 and dy.max() == 50.0 and dy.min() == -50.0
    assert np.all(np.diff(dx) > 0)
    sx, sy = server.decimate_minmax(x[:100], y[:100], 2000)
    assert sx.size == 100


@test
def overview_and_block_match_direct_reads():
    with tempfile.TemporaryDirectory() as d:
        rec = make_fake_openephys(Path(d), n=50_000)
        ts = dio.load(rec)
        dio.OVERVIEW_BUCKETS, saved = 1000, dio.OVERVIEW_BUCKETS
        try:
            stride, centres, mins, maxs = ts.overview()
        finally:
            dio.OVERVIEW_BUCKETS = saved
        assert stride == 50 and mins.shape == (1000, 3) and centres.shape == (1000,)
        first = ts.names[0]                       # recorded position 1 (named by headstage input via the channel map)
        ch = ts.channel(first)
        assert np.allclose(mins[:, 0], ch.reshape(1000, 50).min(axis=1))
        assert np.allclose(maxs[:, 0], ch.reshape(1000, 50).max(axis=1))
        blk = ts.block(100, 200, ["A1_AUX1", first])
        assert blk.shape == (100, 2) and np.allclose(blk[:, 1], ch[100:200]) and np.allclose(blk[:, 0], ts.channel("A1_AUX1")[100:200])
        assert ts.overview() is ts.overview()   # cached


# ---------------------------------------------------------------- loaders

@test
def timeseries_window_and_scale():
    t = np.arange(1000) / 100.0
    raw = np.column_stack([np.arange(1000), np.arange(1000) * 2]).astype(np.int16)
    ts = dio.TimeSeries(t, raw, ["a", "b"], scale=[0.5, 1.0], units=["uV", "mV"], events={"e": np.array([1.0, 5.0, 9.0])})
    assert ts.fs == 100.0 and ts.channel("a")[10] == 5.0 and ts.channel("b").dtype == float
    w = ts.window(2.0, 6.0)
    assert w.n == 401 and w.t[0] == 2.0 and list(w.events["e"]) == [5.0]
    assert w.summary()["events"] == {"e": 1}


@test
def csv_and_npy_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.csv"
        t = np.arange(50) / 10.0
        np.savetxt(p, np.column_stack([t, np.sin(t), np.cos(t)]), delimiter=",", header="time,s,c", comments="")
        ts = dio.load(p)
        assert ts.names == ["s", "c"] and abs(ts.fs - 10) < 1e-9 and ts.n == 50
        q = Path(d) / "y.npy"
        np.save(q, np.random.default_rng(0).standard_normal((4, 300)))  # (k, n) orientation
        ts2 = dio.load(q, fs=200)
        assert ts2.n == 300 and ts2.n_channels == 4 and ts2.fs == 200


def make_fake_openephys(root: Path, n=3000, fs=1000.0, nch=3, ticks=True) -> Path:
    """Minimal Open Ephys binary layout with a TTL stream and a message, timestamps as clock ticks."""
    rec = root / "Record Node 1" / "experiment1" / "recording1"
    cont = rec / "continuous" / "Src-100.Stream"
    ttl = rec / "events" / "Src-100.Stream" / "TTL"
    msg = rec / "events" / "MessageCenter"
    for p in (cont, ttl, msg):
        p.mkdir(parents=True)
    t = np.arange(n) / fs
    data = np.column_stack([np.round(np.sin(2 * np.pi * 5 * t) * 1000), np.arange(n) % 100, np.ones(n)]).astype(np.int16)
    data.tofile(cont / "continuous.dat")
    start = 5000
    np.save(cont / "sample_numbers.npy", np.arange(start, start + n, dtype=np.int64))
    np.save(cont / "timestamps.npy", (np.arange(start, start + n) if ticks else np.arange(start, start + n) / fs).astype(float))
    ev_samples = np.array([start + 100, start + 600, start + 1100, start + 1600], dtype=np.int64)
    np.save(ttl / "sample_numbers.npy", ev_samples)
    np.save(ttl / "timestamps.npy", ev_samples / fs)
    np.save(ttl / "states.npy", np.array([1, -1, 2, -2], dtype=np.int16))
    np.save(ttl / "full_words.npy", np.array([1, 0, 2, 0], dtype=np.uint64))
    np.save(msg / "sample_numbers.npy", np.array([start + 2000], dtype=np.int64))
    np.save(msg / "timestamps.npy", np.array([(start + 2000) / fs]))
    np.save(msg / "text.npy", np.array(["stim on"]))
    meta = {
        "GUI version": "1.0.2",
        "continuous": [{
            "folder_name": "Src-100.Stream/", "sample_rate": fs, "stream_name": "Stream", "num_channels": nch,
            "channels": [{"channel_name": " ", "bit_volts": 0.195, "units": "uV"},
                         {"channel_name": " ", "bit_volts": 0.195, "units": "uV"},
                         {"channel_name": "A1_AUX1", "bit_volts": 3.74e-05, "units": "mV"}],
        }],
        "events": [
            {"folder_name": "Src-100.Stream/TTL/", "channel_name": "TTL Input", "sample_rate": fs, "type": "int16", "stream_name": "Stream"},
            {"folder_name": "MessageCenter/", "channel_name": "Messages", "sample_rate": fs, "type": "string", "stream_name": "Stream"},
        ],
        "spikes": [],
    }
    (rec / "structure.oebin").write_text(json.dumps(meta))
    (root / "Record Node 1" / "settings.xml").write_text("""<?xml version="1.0"?>
<SETTINGS><SIGNALCHAIN>
 <PROCESSOR name="Sources/Intan RHD USB" nodeId="100"/>
 <PROCESSOR name="Filters/Channel Map" nodeId="107"><CUSTOM_PARAMETERS><STREAM>
   <CH index="1" enabled="1"/><CH index="0" enabled="1"/><CH index="2" enabled="1"/><CH index="5" enabled="0"/>
 </STREAM></CUSTOM_PARAMETERS></PROCESSOR>
 <PROCESSOR name="Record Node" nodeId="1"/>
</SIGNALCHAIN></SETTINGS>""")
    return rec


@test
def openephys_loader_memmaps_scales_and_reads_events():
    with tempfile.TemporaryDirectory() as d:
        rec = make_fake_openephys(Path(d))
        ts = dio.load(Path(d))  # session root -> finds structure.oebin
        assert ts.lazy and ts.n == 3000 and ts.fs == 1000.0
        assert ts.names == ["CH2", "CH1", "A1_AUX1"] and ts.units == ["uV", "uV", "mV"]   # recorded order kept, named by headstage input
        assert ts.channel_info[0] == {"recorded": 1, "input": 2, "kind": "headstage"} and ts.window(5.5, 6).channel_info == ts.channel_info
        assert ts.headstage_channels() == ["CH2", "CH1"] and ts.aux_channels() == ["A1_AUX1"]
        assert ts.format_notes["time"] == "timestamps (ticks)" and ts.format_notes["inputs"].startswith("channel map")
        assert ts.results_subdir == "" and detect.results_dir(ts) == rec / "dataanalyzer" / "events"   # single stream: as before
        assert abs(ts.t[0] - 5.0) < 1e-9, ts.t[:3]            # tick-count timestamps converted to seconds
        assert np.allclose(ts.channel("CH1")[:3], [0, 0.195, 0.39])   # recorded position 2 (ramp) is headstage input 1
        assert set(ts.events) == {"TTL1 on", "TTL1 off", "TTL2 on", "TTL2 off", "messages"}
        assert ts.events["TTL1 on"].tolist() == [5.1] and ts.event_labels["messages"] == ["stim on"]
        assert ts.summary()["events"]["messages"] == 1
        ts2 = dio.load(rec / "structure.oebin", stream="Stream")
        assert ts2.n == ts.n
        w = ts.window(5.5, 6.5)
        assert w.lazy and abs(w.t[0] - 5.5) < 1e-9 and w.n == 1001
        # timestamps already in seconds are used as they are
        rec_s = make_fake_openephys(Path(d) / "seconds", ticks=False)
        ts_s = dio.load(rec_s)
        assert ts_s.format_notes["time"] == "timestamps" and abs(ts_s.t[0] - 5.0) < 1e-9


def make_fake_glance(root: Path, n=30000, fs=1000.0) -> Path:
    """The GLANCE layout: two Record Nodes (110 processed + Ripple Detector, 121 raw), a broadband and a
    decimated LFP stream in each, named channels, 'a.u.' aux, RIP_* outputs written as uV, and a
    timestamps.npy that restarts every buffer (only sample_numbers.npy is a usable clock)."""
    sess = root / "2026-01-01_00-00-00"
    start, dec = 7000, 10
    rng = np.random.default_rng(0)

    def stream(rec, folder, names, units, bits, n_s, fs_s, start_s, data):
        cont = rec / "continuous" / folder
        cont.mkdir(parents=True)
        data.astype(np.int16).tofile(cont / "continuous.dat")
        np.save(cont / "sample_numbers.npy", np.arange(start_s, start_s + n_s, dtype=np.int64))
        np.save(cont / "timestamps.npy", (np.arange(n_s) % 190) / fs_s)          # restarts every 190 samples
        return {"folder_name": folder + "/", "sample_rate": fs_s, "stream_name": folder.split(".")[-1],
                "source_processor_name": "GLANCE Neuro", "source_processor_id": 100, "num_channels": len(names),
                "channels": [{"channel_name": nm, "bit_volts": b, "units": u, "description": "x"}
                             for nm, u, b in zip(names, units, bits)]}

    def ttl(rec, folder, samples, states):
        f = rec / "events" / folder / "TTL"
        f.mkdir(parents=True)
        np.save(f / "sample_numbers.npy", np.array(samples, dtype=np.int64))
        np.save(f / "states.npy", np.array(states, dtype=np.int16))
        np.save(f / "timestamps.npy", np.full(len(samples), 8.2e-312))           # garbage, as in the real files
        np.save(f / "full_words.npy", np.zeros(len(samples), dtype=np.uint64))

    wide = np.column_stack([rng.normal(0, 200, n), rng.normal(0, 200, n), rng.normal(-9000, 50, n)])
    for node, sfx, mapped in ((110, "A", True), (121, "B", False)):
        rec = sess / f"Record Node {node}" / "experiment1" / "recording1"
        bb_names = (["A_CH2", "A_CH1"] if mapped else ["A_CH1", "A_CH2"]) + ["A_AUX0_1"]
        bb = wide[:, [1, 0, 2]] if mapped else wide
        lfp = wide[::dec, :2]
        extra = ["RIP_FEAT", "RIP_ON", "RIP_OFF"] if mapped else []
        conts = [
            stream(rec, f"GLANCE_Neuro-100.BroadbandStream-{sfx}", bb_names + extra, ["uV", "uV", "a.u."] + ["uV"] * len(extra),
                   [0.195, 0.195, 1.0] + [0.195] * len(extra), n, fs, start,
                   np.column_stack([bb] + [np.full(n, 500.0)] * len(extra))),
            stream(rec, f"GLANCE_Neuro-100.LFPStream-{sfx}", ["LFP_A_CH1", "LFP_A_CH2"] + extra, ["uV"] * (2 + len(extra)),
                   [0.195] * (2 + len(extra)), n // dec, fs / dec, start // dec,
                   np.column_stack([lfp] + [np.full(n // dec, 500.0)] * len(extra))),
        ]
        ttl(rec, f"GLANCE_Neuro-100.BroadbandStream-{sfx}", [start + 3000, start + 3060], [4, -4])   # same input, both nodes
        events = [{"folder_name": f"GLANCE_Neuro-100.BroadbandStream-{sfx}/TTL/", "channel_name": "TTL Input",
                   "sample_rate": fs, "type": "int16", "stream_name": f"BroadbandStream-{sfx}", "source_processor": "GLANCE Neuro"}]
        if mapped:
            ttl(rec, f"Ripple_Detector-125.BroadbandStream-{sfx}", [start + 2000, start + 2060, start + 9000, start + 9050], [2, -2, 2, -2])
            ttl(rec, f"Ripple_Detector-125.LFPStream-{sfx}", [(start + 2010) // dec, (start + 2050) // dec], [1, -1])
            events += [{"folder_name": f"Ripple_Detector-125.{st}Stream-{sfx}/TTL/", "channel_name": "Ripple Detector output",
                       "sample_rate": r, "type": "int16", "stream_name": f"{st}Stream-{sfx}", "source_processor": "Ripple Detector"}
                      for st, r in (("Broadband", fs), ("LFP", fs / dec))]
        (rec / "structure.oebin").write_text(json.dumps({"GUI version": "1.0.2", "continuous": conts, "events": events, "spikes": []}))
        # a Channel Map whose (unnamed) streams match by count: it must NOT be applied to named channels
        # (the same settings.xml is written beside both nodes, as Open Ephys does: the whole chain, a
        # Splitter whose second branch starts at <SWITCH number="1"/>)
        two = lambda pa="", pb="", na=3, nb=2, sfx="-A": (f'<STREAM name="BroadbandStream{sfx}" channel_count="{na}"><PARAMETERS {pa}/></STREAM>'
                                                           f'<STREAM name="LFPStream{sfx}" channel_count="{nb}"><PARAMETERS {pb}/></STREAM>')
        (sess / f"Record Node {node}" / "settings.xml").write_text(f"""<?xml version="1.0"?>
<SETTINGS><SIGNALCHAIN>
 <PROCESSOR name="GLANCE Neuro" nodeId="100">{two(sfx="")}</PROCESSOR>
 <PROCESSOR name="Splitter" nodeId="120">{two(sfx="")}</PROCESSOR>
 <PROCESSOR name="Notch Filter" nodeId="115">{two('enable_stream="1" low_cut="59.0" high_cut="61.0" channels="3"', 'enable_stream="1" low_cut="59.0" high_cut="61.0" channels=""')}</PROCESSOR>
 <PROCESSOR name="Common Avg Ref" nodeId="119">{two('enable_stream="1" affected="3" reference="3" gain="100.0"', 'enable_stream="1" affected="" reference="" gain="100.0"')}</PROCESSOR>
 <PROCESSOR name="Channel Map" nodeId="111">{two('enable_stream="1"', 'enable_stream="1"')}<CUSTOM_PARAMETERS>
  <STREAM><CH index="1" enabled="1"/><CH index="0" enabled="1"/><CH index="2" enabled="1"/></STREAM>
  <STREAM><CH index="0" enabled="1"/><CH index="1" enabled="1"/></STREAM>
 </CUSTOM_PARAMETERS></PROCESSOR>
 <PROCESSOR name="Ripple Detector" nodeId="125">{two('enable_stream="1" Ripple_Input="2" Ripple_Out="1" method="0" ripple_std="4.0" ripple_std_off="2.0"', 'enable_stream="1" Ripple_Input="1" Ripple_Out="0" method="0" ripple_std="5.0" ripple_std_off="2.0"', 6, 5)}</PROCESSOR>
 <PROCESSOR name="Record Node" nodeId="110">{two(na=6, nb=5)}</PROCESSOR>
 <SWITCH number="1"/>
 <PROCESSOR name="Record Node" nodeId="121">{two(sfx="-B")}</PROCESSOR>
</SIGNALCHAIN></SETTINGS>""")
    return sess


@test
def openephys_event_times_survive_dropped_samples():
    # a stream that dropped samples has a jump in its sample numbers; an event's sample number must be
    # looked up there, not used as an offset from the first sample (that put every later event late)
    with tempfile.TemporaryDirectory() as d:
        sess = make_fake_glance(Path(d))
        recA = sess / "Record Node 110" / "experiment1" / "recording1"
        sn_path = recA / "continuous" / "GLANCE_Neuro-100.BroadbandStream-A" / "sample_numbers.npy"
        sn = np.load(sn_path)
        sn[5000:] += 500                                   # 0.5 s dropped between 12 s and 12.5 s
        np.save(sn_path, sn)
        ts = dio.load(recA, stream="BroadbandStream-A")
        assert np.allclose(ts.events["Ripple Detector TTL2 on"], [9.0, 16.0])
        i = int(np.searchsorted(ts.t, 16.0))
        assert abs(ts.t[i] - 16.0) < 1e-9 and i == 16000 - 7000 - 500


@test
def glance_lfp_clock_from_broadband_ticks():
    # the GLANCE writer stores the LFP stream's timestamps as broadband-rate acquisition ticks, and its LFP
    # sample numbers may start from an unrelated count (SalineLatencyTest1014/Minimum: 276 s off); the
    # LFP time axis, and LFP events seen from the broadband stream, must follow the ticks
    with tempfile.TemporaryDirectory() as d:
        sess = make_fake_glance(Path(d))
        recA = sess / "Record Node 110" / "experiment1" / "recording1"
        lfp_dir = recA / "continuous" / "GLANCE_Neuro-100.LFPStream-A"
        sn = np.load(lfp_dir / "sample_numbers.npy")
        i = np.arange(sn.size)
        np.save(lfp_dir / "timestamps.npy", 7000 + 10.0 * (i // 4 * 4) + 0.0003 * (i % 4))   # 1 kHz ticks, one per 4-sample buffer
        np.save(lfp_dir / "sample_numbers.npy", sn + 80000)
        ev = recA / "events" / "Ripple_Detector-125.LFPStream-A" / "TTL" / "sample_numbers.npy"
        np.save(ev, np.load(ev) + 80000)
        lfp = dio.load(recA, stream="LFPStream-A")
        assert lfp.format_notes["time"].startswith("sample_numbers + offset") and np.allclose(lfp.t, 7.0 + np.arange(lfp.n) / 100.0)
        assert np.allclose(lfp.events["Ripple Detector TTL1 on"], [9.01], atol=0.01)
        bb = dio.load(recA, stream="BroadbandStream-A")
        assert np.allclose(bb.events["Ripple Detector LFPStream-A TTL1 on"], [9.01], atol=0.01)


@test
def glance_loader_uses_sample_numbers_names_and_channel_kinds():
    with tempfile.TemporaryDirectory() as d:
        sess = make_fake_glance(Path(d))
        recA = sess / "Record Node 110" / "experiment1" / "recording1"
        ts = dio.load(recA, stream="BroadbandStream-A")
        assert ts.format_notes == {"time": "sample_numbers", "inputs": "channel names",
                                   "streams": ["BroadbandStream-A", "LFPStream-A"]}
        assert abs(ts.t[0] - 7.0) < 1e-9 and np.all(np.diff(ts.t) > 0) and abs(ts.duration - 29.999) < 1e-6
        assert ts.window(8, 9).n == 1001
        assert ts.names[:3] == ["A_CH2", "A_CH1", "A_AUX0_1"]
        assert ts.headstage_channels() == ["A_CH2", "A_CH1"] and ts.aux_channels() == ["A_AUX0_1"]
        assert ts.channels_of("other") == ["RIP_FEAT", "RIP_ON", "RIP_OFF"]
        assert [c["input"] for c in ts.channel_info] == [2, 1, None, None, None, None]
        # online detections of both streams land on this stream's clock, named by where they came from
        assert np.allclose(ts.events["Ripple Detector TTL2 on"], [9.0, 16.0])
        assert np.allclose(ts.events["Ripple Detector LFPStream-A TTL1 on"], [9.01], atol=0.01)
        assert {(x["stream"], x["line"]) for x in ts.ttl} == {("BroadbandStream-A", 2), ("LFPStream-A", 1), ("BroadbandStream-A", 4)}
        # each stream keeps its own saved results
        lfp = dio.load(recA, stream="LFPStream-A")
        assert detect.results_dir(ts) == recA / "dataanalyzer" / "BroadbandStream-A" / "events"
        assert detect.results_dir(lfp, "spikes") == recA / "dataanalyzer" / "LFPStream-A" / "spikes"
        assert lfp.fs == 100.0 and abs(lfp.t[0] - 7.0) < 1e-9 and [c["input"] for c in lfp.channel_info][:2] == [1, 2]
        assert np.allclose(lfp.events["Ripple Detector BroadbandStream-A TTL2 on"], [9.0, 16.0])
        # the raw node is NOT in mapped order although a Channel Map with the same channel count exists
        raw = dio.load(sess / "Record Node 121", stream="BroadbandStream-B")
        assert [c["input"] for c in raw.channel_info] == [1, 2, None]
        # a session's Record Nodes see the same acquisition, so each one also gets the TTL lines only the other recorded,
        # named after the node they come from — but a line both nodes recorded is not added twice
        assert "TTL4 on" in raw.events and not [k for k in raw.events if "TTL4" in k and "Record Node" in k]
        assert "TTL4 on" in ts.events and not [k for k in ts.events if "TTL4" in k and "Record Node" in k]
        sib = "Ripple Detector BroadbandStream-A TTL2 on (Record Node 110)"
        assert np.allclose(raw.events[sib], ts.events["Ripple Detector TTL2 on"]), sorted(raw.events)
        assert np.allclose(raw.events["Ripple Detector LFPStream-A TTL1 on (Record Node 110)"], ts.events["Ripple Detector LFPStream-A TTL1 on"])
        assert {(x["line"], x.get("node")) for x in raw.ttl} == {(4, ""), (2, "Record Node 110"), (1, "Record Node 110")}
        assert "pulses_BroadbandStream_A_TTL2" in detect.online_detections(raw)
        # the signal chain: what this stream went through, ending at the Record Node that wrote it
        chain = ts.recording_meta["signal_chain"]
        path = [c["name"] for c in chain if c["on_path"]]
        assert path == ["GLANCE Neuro", "Splitter", "Notch Filter", "Common Avg Ref", "Channel Map", "Ripple Detector", "Record Node"], path
        by = {c["name"] + str(c["node_id"]): c for c in chain}
        assert by["Record Node110"]["recorder"] and not by["Record Node121"]["on_path"]
        assert by["Common Avg Ref119"]["summary"].startswith("ch 1-2 minus the mean of ch 1-2") and by["Common Avg Ref119"]["alters"]   # the XML lists the excluded channel
        assert by["Channel Map111"]["alters"] and "input 2" in by["Channel Map111"]["summary"]
        rd = by["Ripple Detector125"]
        assert (rd["n_in"], rd["n_out"]) == (3, 6) and "adds 3 channels" in rd["summary"] and "TTL line 2" in rd["summary"] and "band-pass" in rd["warning"]
        lfp_chain = {c["name"]: c for c in dio.load(recA, stream="LFPStream-A").recording_meta["signal_chain"] if c["on_path"]}
        assert not lfp_chain["Channel Map"]["alters"] and "identity" in lfp_chain["Channel Map"]["summary"] and "ch all" in lfp_chain["Common Avg Ref"]["summary"]
        raw_chain = dio.load(sess / "Record Node 121", stream="BroadbandStream-B").recording_meta["signal_chain"]
        assert [c["name"] for c in raw_chain if c["on_path"]] == ["GLANCE Neuro", "Splitter", "Record Node"] and ts.summary()["signal_chain"] == chain
        # geometry and the movement signal follow the channel kinds, not the units
        lay = probe.builtin("a1x32-poly2").layout(ts.names, ts.channel_info, ts.units)
        assert [c["input"] for c in lay["contacts"]] == [2, 1] and "RIP_FEAT" in lay["unmapped"]
        assert detect.motion_signal(ts) is not None
        assert dio.input_from_name("LFP_A_CH32") == 32 and dio.input_from_name("RIP_FEAT") is None \
            and dio.input_from_name("A-019") == 20 and dio.input_from_name("CH7") == 7


# ---------------------------------------------------------------- probe geometry

@test
def probe_builtin_matches_example_channel_map_and_files_load():
    p = probe.builtin("a1x32-poly2")
    assert len(p.contacts) == 32 and len({c.input for c in p.contacts}) == 32
    # the example session's Channel Map (settings.xml) is the Poly2 depth order, tip first
    example_map = [19, 12, 23, 8, 28, 3, 24, 7, 25, 6, 26, 5, 16, 15, 17, 14, 18, 13, 20, 11, 21, 10, 22, 9, 31, 0, 30, 1, 29, 2, 27, 4]
    assert [c.input - 1 for c in p.contacts] == example_map
    names = [f"CH{i + 1}" for i in example_map] + ["A1_AUX1"]
    info = [{"recorded": i + 1, "input": i0 + 1} for i, i0 in enumerate(example_map)] + [{"recorded": 33, "input": None}]
    lay = p.layout(names, info, ["uV"] * 32 + ["mV"])
    assert lay["unmapped"] == ["A1_AUX1"] and len(lay["contacts"]) == 32
    assert [c["depth_index"] for c in lay["contacts"]] == list(range(32))   # recorded order == tip-first depth order
    assert lay["contacts"][0]["name"] == "CH20" and lay["contacts"][0]["y_um"] == 0.0 and lay["contacts"][1]["x_um"] == 43.0
    # names alone (no channel_info) still resolve
    assert len(p.layout(["CH20", "A-018", "foo"], None)["contacts"]) == 2
    assert len(probe.builtin("a4x8").contacts) == 32
    with tempfile.TemporaryDirectory() as d:
        csvp = Path(d) / "probe.csv"
        csvp.write_text("channel,x_um,y_um,shank,site\nCH20,0,0,0,16\nCH13,43,25,0,17\n5,0,50,1,\n")
        q = probe.load_probe(csvp)
        assert [c.input for c in q.contacts] == [20, 13, 5] and q.contacts[2].shank == 1 and q.contacts[1].site == 17
        pi = {"specification": "probeinterface", "probes": [{"contact_positions": [[0, 0], [0, 20]], "device_channel_indices": [3, 1],
                                                              "shank_ids": ["0", "0"], "annotations": {"name": "tiny"}}]}
        jp = Path(d) / "probe.json"; jp.write_text(json.dumps(pi))
        r = probe.load_probe(jp)
        assert r.name == "tiny" and [c.input for c in r.contacts] == [4, 2]
        assert probe.auto_probe(Path(d), 32).name == "tiny"          # file next to the recording wins
        with tempfile.TemporaryDirectory() as empty:
            assert "ASSUMED" in probe.auto_probe(Path(empty), 32).notes
        assert probe.auto_probe(None, 16) is None


# ---------------------------------------------------------------- detection and spikes

def make_movement_recording(fs=1000.0, duration=60.0, seed=13):
    """LFP channel with ripples at 10/20/30/40/50 s; accelerometer (3 mV axes) moving from 35 to 55 s."""
    t = np.arange(0, duration, 1 / fs)
    rng = np.random.default_rng(seed)
    lfp = 0.2 * rng.standard_normal(t.size)
    for r0 in (10.0, 20.0, 30.0, 40.0, 50.0):
        w = (t > r0) & (t < r0 + 0.06)
        lfp += np.where(w, 2.0, 0.0) * np.sin(2 * np.pi * 200 * t)
    moving = (t > 35) & (t < 55)
    aux = np.column_stack([340.0 * g + 0.5 * rng.standard_normal(t.size) + np.where(moving, 60.0, 0.0) * np.sin(2 * np.pi * (3 + k) * t)
                           for k, g in enumerate((0.1, -0.2, 0.97))])      # gravity offsets + shaking
    data = np.column_stack([lfp, aux])
    return dio.TimeSeries(t, data, ["lfp", "AUX1", "AUX2", "AUX3"], fs, units=["uV", "mV", "mV", "mV"], source="synthetic"), t


@test
def movement_gate_and_immobility_detector():
    ts, t = make_movement_recording()
    tm, m = detect.motion_signal(ts, 1.0, 0.5)
    assert m[(tm > 40) & (tm < 50)].mean() > 5 * m[(tm > 5) & (tm < 30)].mean()     # moving >> still
    base = {"z_enter": 2, "z_peak": 3, "min_ms": 20, "max_ms": 0, "ref_channel": "none"}
    r = detect.run(ts, "swr", "lfp", {**base, "gate_movement": False}, save=False)
    assert r["n_events"] == 5 and not r["movement_gate"]
    r = detect.run(ts, "swr", "lfp", {**base, "gate_movement": True, "move_z": 2.0}, save=False)
    assert r["movement_gate"] and r["n_events"] == 3 and r["n_rejected_movement"] == 2 and r["n_rejected_reference"] == 0
    assert np.allclose(sorted(e["peak"] for e in r["rejected"]), [40.03, 50.03], atol=0.03)
    assert all(e["reason"] == "movement" and e["move_score"] > 2 for e in r["rejected"])
    assert "move_centre" in r["stats"]
    feats = detect.DETECTORS["swr"].features(ts.window(39.8, 40.3), "lfp", r["stats"], **r["params"])
    assert "movement z" in feats and feats["movement z"].mean() > 2
    im = detect.run(ts, "immobility", "lfp", {"min_s": 1.0}, save=False)
    still = sorted((e["start"], e["end"]) for e in im["events"])
    assert len(still) == 2 and still[0][0] < 1 and abs(still[0][1] - 35) < 1.5 and abs(still[1][0] - 55) < 1.5, still
    res = an.REGISTRY["movement"].run(ts.window(30, 45), "lfp", {"hp_hz": 1.0, "smooth_s": 0.5})
    names = [tr.name for tr in res.traces]
    assert names[-1] == "movement (mV RMS)" and len(names) == 4
    ts_no_aux = dio.synthetic()
    assert detect.motion_signal(ts_no_aux) is None
    r = detect.run(ts_no_aux, "swr", "bursts", {**base, "gate_movement": True}, save=False)
    assert not r["movement_gate"]          # silently skipped without accelerometer channels

def make_laminar(fs=2000.0, dur=120.0, n=16, pyr=6, seed=5):
    """16 contacts at 25 um. Ripples (150 Hz, 60 ms) centred on contact ``pyr`` and fading over ~2
    contacts, each with a sharp wave below it; lone ripples without one; probe-wide ripple-band
    artefacts; and one contact with its own noise bursts that must not be taken for the layer."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    depth = np.arange(n) * 25.0
    X = 20.0 * rng.standard_normal((n, t.size))
    burst = lambda t0, f, w: np.sin(2 * np.pi * f * (t - t0)) * np.exp(-0.5 * ((t - t0) / w) ** 2)
    swr_t, lone_t, art_t = [10.0, 25.0, 40.0, 55.0, 70.0, 85.0], [100.0, 110.0], [15.0, 47.0, 93.0]
    spread = np.exp(-0.5 * ((depth - depth[pyr]) / 40.0) ** 2)
    for t0 in swr_t + lone_t:
        X += 120.0 * spread[:, None] * burst(t0, 150.0, 0.02)
    for t0 in swr_t:                                         # sharp wave: negative below pyramidale, positive above
        sw = np.exp(-0.5 * ((t - t0) / 0.025) ** 2)
        X += (-400.0 * np.clip((depth - depth[pyr]) / 150.0, -0.5, 1.0))[:, None] * sw
    for t0 in art_t:
        X += 300.0 * burst(t0, 200.0, 0.03)                  # same on every contact: never even a candidate
    X += 300.0 * spread[:, None] * burst(art_t[-1], 200.0, 0.03)   # the last one is larger near the layer: a "global" rejection
    for t0 in np.arange(5.0, dur, 7.0):
        X[13] += 400.0 * burst(t0, 210.0, 0.015)             # one noisy contact
    # mains third harmonic, different on every contact (so no reference removes it) and slowly waxing: far larger than the ripples
    X += (np.linspace(250.0, 20.0, n)[:, None] * (1 + 0.5 * np.sin(2 * np.pi * 0.3 * t))) * np.sin(2 * np.pi * 180.0 * t)
    for t0 in (33.0, 63.0):                                  # bursts of the hum itself near the layer: not ripples
        X += 200.0 * spread[:, None] * burst(t0, 180.0, 0.04)
    names = [f"CH{i + 1}" for i in range(n)]
    ts = dio.TimeSeries(t, X.T, names, fs, units=["uV"] * n)
    layout = {"name": "linear16", "contacts": [{"name": nm, "x_um": 0.0, "y_um": float((n - 1 - i) * 25), "shank": 0} for i, nm in enumerate(names)], "bad": []}
    return ts, layout, names[pyr], swr_t, lone_t, art_t


@test
def laminar_swr_finds_the_layer_rejects_global_noise_and_scores_sharp_waves():
    ts, layout, pyr, swr_t, lone_t, art_t = make_laminar()
    # z_peak 7: at the default 5 a robust z of an envelope crosses by chance ~0.3 times a minute in plain noise
    P = {"gate_movement": False, "z_peak": 7.0}
    res = detect.run(ts, "swr_laminar", "auto", P, save=False, layout=layout)
    assert res["channel"] == pyr and res["layer_confident"], (res["channel"], res["layer_confidence"])
    peaks = np.array([e["peak"] for e in res["events"]])
    hit = lambda t0, arr: arr.size and np.min(np.abs(arr - t0)) < 0.05
    assert all(hit(t0, peaks) for t0 in swr_t + lone_t) and len(peaks) == len(swr_t) + len(lone_t), peaks
    rej = np.array([e["peak"] for e in res["rejected"] if e["reason"] == "global"])
    assert hit(art_t[-1], rej) and not any(hit(t0, peaks) for t0 in art_t), rej
    hum_rej = np.array([e["peak"] for e in res["rejected"] if e["reason"] == "line_noise"])
    assert hit(33.0, hum_rej) and hit(63.0, hum_rej) and not hit(33.0, peaks) and not hit(63.0, peaks), res["rejected"]
    assert max(e["line_frac"] for e in res["events"]) < 0.3, [e["line_frac"] for e in res["events"]]
    sw = {round(e["peak"]): e["sw_score"] for e in res["events"]}
    assert all(sw[round(t0)] > 4 for t0 in swr_t) and all(sw[round(t0)] < 2 for t0 in lone_t), sw
    assert res["n_with_sharp_wave"] == len(swr_t)
    # without the mains notch the 180 Hz hum owns the ripple band and the layer is lost
    hum = detect.run(ts, "swr_laminar", "auto", {**P, "line_hz": 0.0}, save=False, layout=layout)
    assert hum["channel"] != pyr or hum["n_events"] < len(swr_t), (hum["channel"], hum["n_events"])
    # features measure the event, not the hum: 150 Hz ripples on a 180 Hz line that is larger than they are
    features.event_features(ts, layout, res)
    rip = [e["features"] for e in res["events"] if any(abs(e["peak"] - t0) < 0.05 for t0 in swr_t)]
    assert all(abs(f["peak_hz"] - 150) < 20 for f in rip), [round(f["peak_hz"]) for f in rip]
    assert all(abs(f["burst_hz"] - 150) < 12 and f["burst_frac"] > 0.5 for f in rip), [(round(f["burst_hz"]), round(f["burst_frac"], 2)) for f in rip]
    y = ts.channel(pyr); k = int(round(25.0 * ts.fs))
    assert features.excess_spectrum(y, k, ts.fs, 120, 250, 60.0)[0] is not None and features.excess_spectrum(y, 10, ts.fs, 120, 250)[0] is None
    # the sharp-wave criterion turns the lone ripples into rejections
    strict = detect.run(ts, "swr_laminar", "auto", {**P, "require_sw": True}, save=False, layout=layout)
    assert strict["n_events"] == len(swr_t) and strict["n_rejected_no_sharp_wave"] == len(lone_t)
    # a chosen channel is respected, and the saved stats reproduce the session z on a short window
    forced = detect.run(ts, "swr_laminar", "CH9", P, save=False, layout=layout)
    assert forced["channel"] == "CH9" and forced["layer_confident"]
    w = ts.window(24.5, 25.5)
    f = swr.swr_laminar_features(w, res["channel"], res["stats"], layout=layout, **res["params"])
    e25 = next(e for e in res["events"] if abs(e["peak"] - 25.0) < 0.05)
    assert set(f) >= {"raw", "ripple band", "local ripple z", "global ripple z", "sharp wave z"} and f["raw"].size == w.n
    assert abs(f["local ripple z"].max() - e25["score"]) < 0.15 * e25["score"], (f["local ripple z"].max(), e25["score"])
    # the event view carries per-trace thresholds for the viewer
    view = csd.event_view(ts, layout, res, detect.DETECTORS["swr_laminar"], 0)
    assert view["features"]["trace_thresholds"]["sharp wave z"] == {"sw_z": 2.0} and "sw_score" in view["event"]
    # the unfiltered window rides along (min/max pairs on one shared time axis) for the "raw by depth" panel
    raw = view["raw"]
    assert len(raw["data"]) == len(view["lfp"]["data"]) and len(raw["data"][0]) == len(raw["t_ms"]) and np.all(np.diff(raw["t_ms"]) > 0)
    tt, mm = csd._decimate_rows_minmax(np.arange(6001) / 30.0, np.r_[np.zeros(3000), 50.0, np.zeros(3000)][None, :], 2400)
    assert mm.max() == 50.0 and tt.size == mm.shape[1] <= 2400
    # decimation keeps the time grid and the signal
    td, Xd, fsd = swr.decimated(ts, ["CH7"], 500.0)
    assert fsd == 500.0 and np.allclose(np.diff(td), 1 / 500.0) and Xd.shape == (1, td.size)


@test
def time_ranges_restrict_event_analyses_to_good_recording():
    fs, dur, n = 1000.0, 120.0, 4
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(4)
    X = 50.0 * rng.standard_normal((t.size, n))
    X[(t >= 100.0)] = 0.0                                               # unplugged for the last 20 s: flat
    loud = (t >= 40.0) & (t < 43.0); X[loud] *= 30.0                    # a loose connector for 3 s
    for s0 in (10.0, 60.0):                                             # 20 ms stimulation artefacts: events, not bad stretches
        X[(t >= s0) & (t < s0 + 0.02)] += 5000.0
    with tempfile.TemporaryDirectory() as d:
        ts = dio.TimeSeries(t, X, [f"CH{k + 1}" for k in range(n)], fs, source=d, units=["uV"] * n)
        sg = sections.suggest(ts)
        assert [[round(a), round(b)] for a, b in sg["include"]] == [[0, 40], [43, 100]], sg["include"]
        assert [r["kind"] for r in sg["reasons"]] == ["very large", "flat"] and abs(sg["excluded_s"] - 23.0) < 0.2
        assert sections.contains([[0, 40]], 5, 6) and not sections.contains([[0, 40]], 39.9, 40.1) and sections.contains([], 1e9)
        assert sections.normalise([[5, 3], [2, 4], [10, 11], [11, 12], [7, 7]]) == [[2, 5], [10, 12]]
        # an event set: 5 inside the first range, 1 inside the loud stretch, 3 in the unplugged tail
        times = [5, 15, 25, 30, 35, 41, 105, 110, 115]
        res = {"name": "evs", "label": "evs", "online": True, "channel": "CH1", "params": {}, "t0": 0.0, "t1": dur,
               "events": [{"start": s0, "end": s0 + 0.05, "peak": s0 + 0.02, "score": 1.0} for s0 in times], "rejected": []}
        srv = server.DataServer(("127.0.0.1", 0), ts)
        srv.detections["evs"] = res
        tr_all = srv.triggered({"det": "evs", "pre_ms": "100", "post_ms": "100", "channel": "CH1"})
        assert tr_all["n_averaged"] == 9
        saved = srv.update_sections({"include": sg["include"]})
        assert sections.load(ts) == saved["include"] and abs(saved["included_s"] - 97.0) < 0.3 and sections.sections_path(ts).exists()
        assert srv.triggered({"det": "evs", "pre_ms": "100", "post_ms": "100", "channel": "CH1"})["n_averaged"] == 5   # cache keyed on the ranges
        assert srv.event_view({"det": "evs", "index": "avg"})["event"]["n_averaged"] == 5
        assert srv.event_view({"det": "evs", "index": "6"})["event"]["centre"] > 100                                   # one event can still be shown
        ev = srv.events_json(None, None, 1000)["detections"]["evs"]
        assert ev["excluded"] == [5, 6, 7, 8] and len(ev["peaks"]) == 9
        other = {"name": "pulses", "label": "p", "channel": "CH1", "events": [{"start": s0 + 0.01, "end": s0 + 0.02, "peak": s0 + 0.01} for s0 in (25, 41, 110)]}
        srv.detections["pulses"] = other
        c = srv.compare({"a": "evs", "b": "pulses"})
        assert c["restricted"] and (c["n_a"], c["n_b"], c["n_matched"]) == (5, 1, 1) and c["rows_b"][0]["index"] == 0 and c["rows_b"][0]["a_index"] == 2
        assert "in_range" in features.to_csv(res, srv.sections).splitlines()[0]
        cleared = srv.update_sections({"clear": True})
        assert cleared["include"] == [] and not sections.sections_path(ts).exists() and srv.event_view({"det": "evs", "index": "avg"})["event"]["n_averaged"] == 9
        srv.server_close()


@test
def server_reports_when_its_code_is_older_than_the_files():
    srv = server.DataServer(("127.0.0.1", 0), dio.synthetic(duration=2.0))
    assert srv.meta()["code_stale"] is False
    saved = server.CODE_MTIME_AT_START
    try:
        server.CODE_MTIME_AT_START = saved - 60.0          # as if a .py file changed a minute after start
        assert srv.meta()["code_stale"] is True
    finally:
        server.CODE_MTIME_AT_START = saved
        srv.server_close()


@test
def latency_between_a_trigger_and_the_pulse_it_caused():
    mk = lambda name, times: {"name": name, "label": name, "channel": "CH1", "t0": 0.0, "t1": 10.0,
                              "events": [{"start": t, "end": t + 0.002, "peak": t, "score": 1.0} for t in times]}
    trig = mk("trigger", [1.0, 2.0, 3.0, 4.0, 5.0])
    stim = mk("stimulus", [1.012, 2.014, 3.011, 3.970, 5.013])          # the fourth pulse comes *before* its trigger
    loose = detect.compare(trig, stim, 50.0)
    assert loose["n_matched"] == 5 and abs(loose["latency_ms"]["sd"]) > 15                    # the early one ruins the spread
    c = detect.compare(trig, stim, 50.0, causal=True)
    assert c["causal"] and c["n_matched"] == 4 and c["n_only_b"] == 1
    lat = c["latency_ms"]
    assert abs(lat["mean"] - 12.5) < 0.1 and abs(lat["sd"] - 1.29) < 0.05 and lat["n"] == 4 and len(lat["values"]) == 4
    assert abs(lat["median"] - 12.5) < 0.6 and abs(lat["min"] - 11.0) < 0.1 and abs(lat["max"] - 14.0) < 0.1
    assert detect.compare(trig, stim, 5.0, causal=True)["n_matched"] == 0                     # tolerance is the longest latency
    # a time range restricts which pairs count, through the server
    ts = dio.synthetic(duration=10.0, fs=100.0)
    with tempfile.TemporaryDirectory() as d:
        ts.source = d
        srv = server.DataServer(("127.0.0.1", 0), ts)
        srv.detections["trigger"], srv.detections["stimulus"] = trig, stim
        srv.update_sections({"include": [[0.0, 3.5]]})
        r = srv.compare({"a": "trigger", "b": "stimulus", "causal": "1"})
        assert r["restricted"] and r["latency_ms"]["n"] == 3
        assert sections.subtract([[0.0, 10.0]], [[2.0, 3.0]], 0.0, 10.0) == [[0.0, 2.0], [3.0, 10.0]]
        srv.update_sections({"exclude": [[2.0, 3.0]]})                  # excluding works from "whole recording" too
        assert srv.sections == [[0.0, 2.0], [3.0, 3.5]] and sections.load(ts) == srv.sections
        srv.update_sections({"clear": True})
        cut = srv.update_sections({"exclude": [[4.0, 6.0]]})["include"]      # from the whole recording, which ends at t[-1]
        assert len(cut) == 2 and cut[0] == [0.0, 4.0] and cut[1][0] == 6.0 and abs(cut[1][1] - float(ts.t[-1])) < 1e-9
        srv.server_close()


@test
def tagged_runs_are_kept_beside_the_default_result():
    ts, layout, pyr, swr_t, *_ = make_laminar(dur=60.0)
    with tempfile.TemporaryDirectory() as d:
        ts.source = d                                                     # results go to <d>/dataanalyzer/events
        P = {"gate_movement": False, "z_peak": 7.0}
        a = detect.run(ts, "swr_laminar", "auto", P, layout=layout)
        b = detect.run(ts, "swr_laminar", "auto", {**P, "low_hz": 120, "high_hz": 170}, layout=layout, tag="120 - 170 Hz!")
        assert a["name"] == "swr_laminar" and b["name"] == "swr_laminar__120-170-Hz" and b["detector"] == "swr_laminar"
        assert b["label"].endswith("· 120 - 170 Hz!") and b["tag"] == "120 - 170 Hz!"
        saved = detect.load_results(ts)
        assert set(saved) == {"swr_laminar", "swr_laminar__120-170-Hz"} and saved["swr_laminar__120-170-Hz"]["params"]["high_hz"] == 170
        assert detect.detector_of(saved["swr_laminar__120-170-Hz"]) is detect.DETECTORS["swr_laminar"]
        assert detect.detector_of({"name": "theta"}) is detect.DETECTORS["theta"]            # older files: name = detector
        srv = server.DataServer(("127.0.0.1", 0), ts)
        srv.probe_spec = None; srv._apply_probe()
        view = srv.event_view({"det": "swr_laminar__120-170-Hz", "index": "0"})
        assert view["det"] == "swr_laminar__120-170-Hz" and "local ripple z" in view["features"]["traces"]
        assert srv.compare({"a": "swr_laminar", "b": "swr_laminar__120-170-Hz"})["n_b"] == b["n_events"]
        srv.server_close()
        assert detect.delete_result(ts, "swr_laminar__120-170-Hz") and set(detect.load_results(ts)) == {"swr_laminar"}


@test
def pulses_on_both_edges_merge_into_one_event():
    """A stimulator that fires on both edges of a trigger records each stimulus as two pulses a
    fraction of a millisecond apart; the Events tab's "merge pulses ms" makes them one event."""
    ev = [{"start": 1.0000, "end": 1.0002, "peak": 1.0000, "duration": 0.0002, "score": 1.0},
          {"start": 1.0005, "end": 1.0007, "peak": 1.0005, "duration": 0.0002, "score": 3.0},
          {"start": 5.0000, "end": 5.0002, "peak": 5.0000, "duration": 0.0002, "score": 2.0}]
    assert detect.merge_pulses(ev, 0.0) is ev                       # off by default: every pulse kept
    m = detect.merge_pulses(ev, 1.0)
    assert len(m) == 2 and m[0]["n_merged"] == 2 and m[1]["n_merged"] == 1
    assert m[0]["start"] == 1.0 and abs(m[0]["end"] - 1.0007) < 1e-9     # first rising to last falling edge
    assert abs(m[0]["duration"] - 0.0007) < 1e-9 and abs(m[0]["merged_ms"] - 0.5) < 1e-9
    assert m[0]["score"] == 3.0 and m[1]["start"] == 5.0
    assert len(detect.merge_pulses(ev, 0.2)) == 3                   # gap is 0.3 ms: too far to merge
    assert len(detect.merge_pulses(ev, 5000.0)) == 1
    with tempfile.TemporaryDirectory() as d:
        sess = make_fake_glance(Path(d))
        ts = dio.load(sess / "Record Node 110", stream="BroadbandStream-A")
        srv = server.DataServer(("127.0.0.1", 0), ts)
        name = next(n for n in srv.detections if n.startswith(detect.PULSE_PREFIX))
        n_raw = srv.detections[name]["n_events"]
        assert srv.detections_json()["pulses"] == {"merge_ms": 0.0}
        out = srv.set_pulse_merge(5000.0)                           # everything in this fake is within 5 s
        assert out["merge_ms"] == 5000.0 and srv.detections[name]["n_events"] <= n_raw
        assert srv.detections[name]["n_pulses"] == n_raw and detect.load_pulse_settings(ts)["merge_ms"] == 5000.0
        srv._load_saved()                                            # survives a reload of the recording
        assert srv.detections[name]["n_events"] == out["sets"][name]["n_events"]
        srv.set_pulse_merge(0)
        assert srv.detections[name]["n_events"] == n_raw and not detect.pulse_settings_path(ts).exists()
        srv.server_close()


@test
def online_detections_are_imported_and_compared():
    with tempfile.TemporaryDirectory() as d:
        sess = make_fake_glance(Path(d))
        ts = dio.load(sess / "Record Node 110", stream="BroadbandStream-A")
        on = detect.online_detections(ts)
        assert set(on) == {"pulses_BroadbandStream_A_TTL2", "pulses_LFPStream_A_TTL1", "pulses_BroadbandStream_A_TTL4"}
        bb = on["pulses_BroadbandStream_A_TTL2"]
        assert bb["online"] and bb["n_events"] == 2 and abs(bb["events"][0]["start"] - 9.0) < 1e-9 and abs(bb["events"][0]["duration"] - 0.06) < 1e-9
        assert bb["events"][0]["online_feature"] == 500 * 0.195 and abs(bb["events"][0]["score"] - 1.0) < 1e-9     # RIP_FEAT / RIP_ON
        offline = {"name": "swr", "label": "offline", "channel": "A_CH1", "t0": 7.0, "t1": 37.0,
                   "events": [{"start": 8.95, "end": 9.04, "peak": 8.99, "score": 6.0}, {"start": 20.0, "end": 20.05, "peak": 20.02, "score": 5.0}],
                   "rejected": [{"start": 15.98, "end": 16.03, "peak": 16.0, "score": 7.0, "reason": "movement"}]}
        c = detect.compare(offline, bb, tol_ms=50.0)
        assert (c["n_matched"], c["n_only_a"], c["n_only_b"], c["n_b_on_rejected"]) == (1, 1, 0, 1)
        assert [r["status"] for r in c["rows_b"]] == ["matched", "rejected:movement"] and abs(c["latency_ms"]["median"] - 50.0) < 1e-6
        assert [r["status"] for r in c["rows_a"]] == ["matched", "only_a"] and c["a_caught_frac"] == 0.5
        assert detect.compare(offline, bb, tol_ms=50.0, include_rejected=False)["n_only_b"] == 1
        # the server lists them with saved results, serves the comparison, and will not delete them
        srv = server.DataServer(("127.0.0.1", 0), ts)
        assert "pulses_BroadbandStream_A_TTL2" in srv.detections_json()["results"]
        srv.detections["swr"] = offline
        assert srv.compare({"b": "pulses_BroadbandStream_A_TTL2"})["n_matched"] == 1
        view = srv.event_view({"det": "pulses_BroadbandStream_A_TTL2", "index": "0"})
        assert "RIP_FEAT (recorded)" in view["features"]["traces"] and view["event"]["online_threshold"] == 500 * 0.195
        # the detector's own recorded channels come with the event view, from whichever stream has them
        assert dio.find_extra_channels(ts)[1:] == ("BroadbandStream-A", ["RIP_FEAT", "RIP_ON", "RIP_OFF"])
        lfp_only = dio.load(sess / "Record Node 110", stream="LFPStream-A")
        assert dio.find_extra_channels(lfp_only)[1] == "LFPStream-A"
        ex = view["extras"]
        assert ex["names"] == ["RIP_FEAT", "RIP_ON", "RIP_OFF"] and len(ex["data"]) == 3 and len(ex["t_ms"]) == len(ex["data"][0])
        assert abs(ex["data"][0][0] - 500 * 0.195) < 1e-6 and ex["source"] == ""     # same stream here
        raw_node = dio.load(sess / "Record Node 121", stream="BroadbandStream-B")    # no such channels of its own
        assert not raw_node.channels_of("other") and dio.find_extra_channels(raw_node)[1] == "BroadbandStream-A"
        assert bb["label"] == "TTL2 pulses · Ripple Detector" and bb["kind"] == "pulses" and abs(bb["pulse_width_ms"] - 55.0) < 1e-6      # pulses of 60 and 50 ms
        # windows can be centred on either edge of a pulse
        assert view["align"] == "peak" and abs(view["event"]["centre"] - 9.0) < 1e-9
        assert abs(srv.event_view({"det": "pulses_BroadbandStream_A_TTL2", "index": "0", "align": "end"})["event"]["centre"] - 9.06) < 1e-9
        srv.server_close()
        # copies of a channel are reported (the GLANCE LFP stream)
        assert ts.format_notes.get("warnings") is None
        t = np.arange(2000) / 1000.0
        x = np.random.default_rng(0).standard_normal((2000, 3)); x[:, 1] = x[:, 0]
        assert dio.duplicate_channels(dio.TimeSeries(t, x, ["a", "b", "c"], 1000.0)) == [["a", "b"]]


@test
def stimulation_artefacts_are_found_without_a_ttl_pulse():
    fs, dur, n = 20000.0, 40.0, 8
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(9)
    X = 15.0 * rng.standard_normal((t.size, n))
    # unreferenced recording: mains as a sharp transient shared by every channel, 120 times a second
    X += (600.0 * (np.abs(((t * 120.0) % 1.0) - 0.5) < 0.004))[:, None]
    gains = np.linspace(0.4, 1.6, n)                                    # the artefact differs from contact to contact
    trains = [5.0, 15.0, 25.0]
    stim_t = [t0 + k * 0.1 for t0 in trains for k in range(5)]           # 3 trains of 5 pulses, 100 ms apart
    for s0 in stim_t:                                                   # biphasic 0.4 ms + 0.4 ms pulse
        i = int(round(s0 * fs))
        X[i:i + 8] += 800.0 * gains
        X[i + 8:i + 16] -= 500.0 * gains
    i = int(round(32.0 * fs)); X[i:i + 6, 3] += 900.0                    # a "spike" on one contact only: not a stimulus
    seg = slice(int(35.0 * fs), int(35.4 * fs))                         # 400 ms of connector noise on every contact
    X[seg] += 400.0 * rng.standard_normal((seg.stop - seg.start, 1)) * gains
    names = [f"CH{k + 1}" for k in range(n)]
    ts = dio.TimeSeries(t, X, names, fs, units=["uV"] * n)
    res = detect.run(ts, "stim_artifacts", "auto", {"train_gap_ms": 500.0}, save=False)
    starts = np.array([e["start"] for e in res["events"]])
    assert len(starts) == len(stim_t) and np.allclose(starts, stim_t, atol=0.0005), starts
    # the median reference costs the contacts nearest the median artefact size; the amplitude is measured as recorded
    assert all(e["channels_frac"] >= 0.7 and e["amp_uV"] > 800 for e in res["events"]) and res["channel"] == "CH8"    # largest on the last contact
    assert res["n_trains"] == 3 and [e["pulse"] for e in res["events"][:6]] == [0, 1, 2, 3, 4, 0] and abs(res["median_interval_ms"] - 100.0) < 0.1
    assert res["n_rejected_duration"] == 1 and abs(res["rejected"][0]["start"] - 35.0) < 0.01                          # the noise episode
    assert not any(abs(e["start"] - 32.0) < 0.01 for e in res["events"] + res["rejected"])                             # the one-contact spike
    # without the median reference the shared mains transient is "on every channel" too and swamps the result
    unref = detect.run(ts, "stim_artifacts", "auto", {"reference": "none"}, save=False)
    assert unref["n_candidates"] > 10 * len(stim_t) or unref["n_events"] != len(stim_t), (unref["n_candidates"], unref["n_events"])
    assert detect.run(ts, "stim_artifacts", "CH2", {"refractory_ms": 150.0}, save=False)["n_events"] == 9             # every other pulse of each train
    # the saved slope scales reproduce the score on a short window, with its thresholds for the viewer
    w = ts.window(4.9, 5.2)
    f = stim.stim_features(w, res["channel"], res["stats"], **res["params"])
    assert f["artefact score z"].size == w.n and abs(f["artefact score z"].max() - res["events"][0]["score"]) < 0.2 * res["events"][0]["score"]
    assert f["channels over threshold %"].max() >= 70.0 and stim.stim_thresholds(res["params"])["artefact score z"] == {"score_min": 10.0}
    # ...and the detected times drive the triggered average like any other event set
    avg = triggered.triggered_average(ts, res, "CH8", 20, 20, align="start", max_trials=5)
    assert np.array(avg["mean"]).max() > 800 and avg["n_averaged"] == len(stim_t)


@test
def triggered_average_recovers_a_response_locked_to_either_pulse_edge():
    fs = 2000.0
    t = np.arange(0, 60, 1 / fs)
    rng = np.random.default_rng(3)
    y = 30.0 * rng.standard_normal(t.size) + 500.0                      # noise on a DC offset
    starts = np.arange(2.0, 58.0, 0.7)
    width = 0.05
    resp = lambda t0: -200.0 * np.exp(-0.5 * ((t - t0 - 0.010) / 0.004) ** 2)     # a dip 10 ms after the edge
    for s0 in starts:
        y += resp(s0 + width)                                           # the "stimulus" sits on the falling edge
    ts = dio.TimeSeries(t, np.column_stack([y, rng.standard_normal(t.size)]), ["CH1", "CH2"], fs, units=["uV", "uV"])
    pulses = {"name": "pulses_x_TTL1", "label": "TTL1 pulses", "channel": "CH1",
              "events": [{"start": float(s0), "end": float(s0 + width), "peak": float(s0), "score": 0.0} for s0 in starts]}
    end = triggered.triggered_average(ts, pulses, "CH1", 250, 250, align="end", max_trials=20)
    tm, mean = np.array(end["t_ms"]), np.array(end["mean"])
    assert end["n_averaged"] == starts.size and end["n_trials"] == 20 and len(end["trials"][0]) == tm.size == len(end["sd"])
    assert abs(tm[np.argmin(mean)] - 10.0) < 1.5 and mean.min() < -150 and abs(mean[tm < -20].mean()) < 5      # baseline removed the offset
    assert abs(end["median_interval_ms"] - 700.0) < 1e-6
    start = triggered.triggered_average(ts, pulses, "CH1", 250, 250, align="start")
    assert abs(np.array(start["t_ms"])[np.argmin(start["mean"])] - 60.0) < 1.5                                   # same dip, seen from the rising edge
    raw = triggered.triggered_average(ts, pulses, "CH1", 250, 250, align="end", baseline=False)
    assert abs(np.array(raw["mean"])[tm < -20].mean() - 500.0) < 5
    band = triggered.triggered_average(ts, pulses, "CH1", 100, 100, align="end", band=(1.0, 300.0))
    assert band["band"] == [1.0, 300.0] and np.array(band["mean"]).min() < -100
    # normalisations, applied per trial before averaging: baseline z (own SD over the 100 ms before) and session z
    bz = triggered.triggered_average(ts, pulses, "CH1", 250, 250, align="end", norm="baseline_z", base_ms=100.0)
    assert bz["unit"] == "z (baseline)" and abs(bz["base_ms"] - 100.0) < 1e-6 and -8.5 < np.array(bz["mean"]).min() < -5      # -200 uV on 30 uV noise
    sz = triggered.triggered_average(ts, pulses, "CH1", 250, 250, align="end", norm="session_z")
    mu, sd = triggered.session_stats(ts, "CH1")
    assert abs(mu - 500.0) < 10 and 28 < sd < 45 and sz["unit"] == "z (session)" and abs(sz["session_sd"] - sd) < 1e-9
    assert abs(np.array(sz["mean"]).min() - (-200.0 / sd)) < 0.8 and abs(np.array(sz["mean"])[tm < -20].mean()) < 0.3
    try:
        triggered.triggered_average(ts, pulses, "CH1", norm="percent"); raise AssertionError("bad norm accepted")
    except ValueError:
        pass
    try:
        triggered.triggered_average(ts, pulses, "CH1", align="middle"); raise AssertionError("bad align accepted")
    except ValueError:
        pass
    srv = server.DataServer(("127.0.0.1", 0), ts)
    srv.detections["pulses_x_TTL1"] = pulses
    out = srv.triggered({"det": "pulses_x_TTL1", "align": "end", "max_trials": "5"})
    assert out["channel"] == "CH1" and out["n_trials"] == 5 and "compute_ms" in out
    assert out["norm"] == "baseline" and srv.triggered({"det": "pulses_x_TTL1", "norm": "session_z"})["unit"] == "z (session)"
    srv.server_close()


@test
def swr_reference_channel_rejects_shared_artefacts():
    fs = 1000.0
    t = np.arange(0, 60, 1 / fs)
    rng = np.random.default_rng(11)
    near = 0.2 * rng.standard_normal(t.size)
    far = 0.2 * rng.standard_normal(t.size)
    for r0 in (10.0, 20.0, 30.0):                          # genuine ripples: near channel only
        w = (t > r0) & (t < r0 + 0.06)
        near += np.where(w, 2.0, 0.0) * np.sin(2 * np.pi * 200 * t)
    for r0 in (40.0, 50.0):                                # artefacts: on both channels
        w = (t > r0) & (t < r0 + 0.06)
        burst = np.where(w, 2.0, 0.0) * np.sin(2 * np.pi * 200 * t)
        near += burst
        far += burst
    ts = dio.TimeSeries(t, np.column_stack([near, far]), ["near", "far"], fs, units=["uV", "uV"], source="synthetic")
    base = {"z_enter": 2, "z_peak": 3, "min_ms": 20, "max_ms": 0}
    r = detect.run(ts, "swr", "near", {**base, "ref_channel": "none"}, save=False)
    assert r["n_events"] == 5 and r["ref_channel"] is None
    r = detect.run(ts, "swr", "near", {**base, "ref_channel": "far", "ref_z": 2.0}, save=False)
    assert r["n_events"] == 3 and r["n_rejected"] == 2 and r["n_candidates"] == 5 and r["ref_channel"] == "far"
    assert np.allclose(sorted(e["peak"] for e in r["events"]), [10.03, 20.03, 30.03], atol=0.03)
    assert np.allclose(sorted(e["peak"] for e in r["rejected"]), [40.03, 50.03], atol=0.03)
    assert all(e["ref_score"] > 2 for e in r["rejected"]) and all(e["ref_score"] <= 2 for e in r["events"])
    assert "ref_centre" in r["stats"] and "ref_scale" in r["stats"]
    feats = detect.DETECTORS["swr"].features(ts.window(39.9, 40.2), "near", r["stats"], **r["params"])
    assert "ref far envelope z" in feats and feats["ref far envelope z"].max() > 2
    v = csd.event_view(ts, None, {**r, "events": r["rejected"], "set": "rejected"}, detect.DETECTORS["swr"], 0, lfp_band=(0, 0))
    assert v["set"] == "rejected" and v["event"]["ref_score"] > 2 and "ref far envelope z" in v["features"]["traces"]

@test
def swr_and_theta_detectors_find_synthetic_epochs_and_save():
    fs = 1000.0
    t = np.arange(0, 60, 1 / fs)
    rng = np.random.default_rng(3)
    y = 0.2 * rng.standard_normal(t.size)
    theta_on = (t > 10) & (t < 25)                       # 15 s of strong theta
    y += np.where(theta_on, 3.0, 0.3) * np.sin(2 * np.pi * 8 * t)
    ripple_times = [30.0, 40.0, 50.0]
    for r0 in ripple_times:                              # 60 ms ripples
        w = (t > r0) & (t < r0 + 0.06)
        y += np.where(w, 2.0, 0.0) * np.sin(2 * np.pi * 200 * t)
    ts = dio.TimeSeries(t, y, ["lfp"], fs, source="synthetic")
    with tempfile.TemporaryDirectory() as d:
        ts.source = str(Path(d))                          # results land in <d>/dataanalyzer/events
        r = detect.run(ts, "theta", "lfp", {"z_enter": 1.0, "z_peak": 2.0, "min_s": 2.0})
        assert r["n_events"] == 1 and abs(r["events"][0]["start"] - 10) < 1.0 and abs(r["events"][0]["end"] - 25) < 1.0
        assert r["stats"]["robust"] == 1.0 and r["stats"]["scale"] > 0
        r = detect.run(ts, "swr", "lfp", {"z_enter": 2, "z_peak": 4, "min_ms": 20, "max_ms": 200})
        assert r["n_events"] == 3, r["n_events"]
        feats = detect.DETECTORS["swr"].features(ts.window(29.9, 30.2), "lfp", r["stats"], **r["params"])
        assert set(feats) == {"raw", "ripple band", "envelope z"} and feats["envelope z"].max() > 4
        assert np.allclose(sorted(e["peak"] for e in r["events"]), [30.03, 40.03, 50.03], atol=0.03)
        saved = detect.load_results(ts)
        assert set(saved) == {"theta", "swr"} and saved["swr"]["n_events"] == 3 and Path(r["path"]).exists()
        assert detect.delete_result(ts, "theta") and set(detect.load_results(ts)) == {"swr"}


@test
def spike_detector_finds_injected_spikes_and_roundtrips():
    fs = 30000.0
    t = np.arange(0, 4, 1 / fs)
    rng = np.random.default_rng(5)
    data = rng.standard_normal((t.size, 2)) * 10.0        # ~10 µV noise
    spike_t = np.arange(0.5, 3.5, 0.25)                   # 12 spikes on channel 0
    spk = -80.0 * np.exp(-((np.arange(-30, 31) / fs) ** 2) / (2 * (0.0002) ** 2))   # 0.2 ms wide negative spike
    for st in spike_t:
        i = int(st * fs)
        data[i - 30:i + 31, 0] += spk
    ts = dio.TimeSeries(t, data, ["a", "b"], fs, units=["uV", "uV"], source="synthetic")
    res = spikes.detect_spikes(ts, ["a", "b"], threshold_sd=5.5, chunk_s=1.0)
    ta = res["times"][res["channels"] == 0]
    assert ta.size == 12 and np.allclose(ta, spike_t, atol=0.0005), ta
    assert (res["channels"] == 1).sum() <= 2               # noise-only channel: at most a stray crossing
    with tempfile.TemporaryDirectory() as d:
        ts.source = str(Path(d))
        out = spikes.save(ts, res)
        assert out.exists()
        back = spikes.load(ts)
        assert back["channel_names"] == ["a", "b"] and back["times"].size == res["times"].size
        assert spikes.summary(back)["counts"][0] == 12 and back["meta"]["threshold_sd"] == 5.5
        # what the Spikes tab shows: per-channel counts and a raster, then one channel's waveforms
        ov = spikes.overview(back, ts.duration, bins=40)
        assert ov["counts"][0] == 12 and ov["n_spikes"] == back["times"].size and len(ov["raster"]["counts"]) == 2
        assert len(ov["raster"]["t"]) == 40 and sum(ov["raster"]["counts"][0]) == 12
        assert abs(ov["rate_hz"][0] - 12 / ts.duration) < 0.01 and ov["amp_percentiles"]["a"][1] > 50
        w = spikes.waveforms(ts, back, "a", pre_ms=1.0, post_ms=2.0)
        mean, tms = np.array(w["mean"]), np.array(w["t_ms"])
        assert w["n_spikes"] == 12 and w["n_waveforms"] == 12 and mean.size == tms.size
        assert mean.min() < -50 and abs(tms[np.argmin(mean)]) < 0.1        # the trough sits at the detected time
        assert w["threshold_uV"] and abs(w["threshold_uV"] - 5.5 * w["noise_sd"]) < 1e-6
        assert sum(w["isi_hist"]["counts"]) == 11 and w["isi_hist"]["violations"] == 0
        assert len(w["amp_hist"]["counts"]) == len(w["amp_hist"]["edges"]) - 1
        try:
            spikes.waveforms(ts, back, "nope"); raise AssertionError("unknown channel accepted")
        except KeyError:
            pass
        srv = server.DataServer(("127.0.0.1", 0), ts)
        assert srv.spikes and spikes.overview(srv.spikes, ts.duration)["n_spikes"] == back["times"].size
        srv.server_close()


@test
def phase_locks_spikes_to_the_cycle_inside_epochs_only():
    """The Theta tab: spikes placed on the peak of an 8 Hz cycle come out at 0 deg with a high
    mean vector length, and spikes outside the epochs are left out of the count."""
    fs, dur, f0 = 2000.0, 60.0, 8.0
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(3)
    lfp = 200.0 * np.sin(2 * np.pi * f0 * t) + rng.standard_normal(t.size) * 5.0
    ts = dio.TimeSeries(t, np.column_stack([lfp, lfp]), ["ref", "b"], fs, units=["uV", "uV"], source="synthetic")
    ph = phase.band_phase(ts, "ref", (5.0, 10.0))
    assert abs(ph["fs"] - 1000.0) < 1 and ph["phase"].min() >= 0 and ph["phase"].max() <= 360

    peaks = np.arange(0.25 / f0, dur, 1 / f0)                      # every peak of the cycle
    inside = peaks[(peaks > 10) & (peaks < 20)]
    outside = peaks[(peaks > 30) & (peaks < 40)] + 0.5 / f0        # troughs, and outside the epoch
    ranges = phase.event_ranges([{"start": 10.0, "end": 20.0}])
    assert ranges.shape == (1, 2)
    src = {"times": np.sort(np.concatenate([inside, outside])), "names": ["u"], "labels": ["u"], "kind": "spikes",
           "group": np.zeros(inside.size + outside.size, dtype=int)}
    tab = phase.phase_table(ph, ranges, src)
    g = tab["groups"][0]
    assert g["n"] == inside.size and tab["n_spikes"] == src["times"].size      # the troughs outside are not counted
    assert min(g["mean_deg"], 360 - g["mean_deg"]) < 15, g["mean_deg"]         # on the peak = 0 deg
    assert g["mvl"] > 0.9 and g["p"] < 1e-9, g
    assert sum(g["counts"]) == g["n"] and len(tab["edges"]) == phase.BINS + 1

    flat = phase.circular_stats(rng.uniform(0, 360, 500))
    assert flat["mvl"] < 0.12 and flat["p"] > 0.05, flat                       # an even spread is not called locked
    assert phase.circular_stats(np.empty(0))["n"] == 0
    # the phase of a trough is 180 deg away from the phase of a peak
    troughs, _ = phase.phase_at(ph, inside + 0.5 / f0)
    assert abs(float(np.median(troughs)) - 180.0) < 15, np.median(troughs)
    # aligning on the cycle: the average over whole epochs would cancel, this one does not
    pk = phase.cycle_times(ph, ranges, "peak")
    assert abs(pk.size - 10 * f0) <= 2 and np.all(np.diff(pk) > 0)
    assert np.median(np.interp(pk, t, lfp)) > 150                              # a peak really is a peak
    assert np.median(np.interp(phase.cycle_times(ph, ranges, "trough"), t, lfp)) < -150
    av = phase.cycle_average(ph, ranges, src, "peak")
    tm, mean = np.array(av["t_ms"]), np.array(av["mean"])
    assert av["n_averaged"] > 70 and abs(tm[np.argmax(mean)]) < 10             # the average peaks at 0 ms
    assert np.ptp(mean) > 300 and len(av["sd"]) == tm.size                     # 400 uV peak to peak survives averaging
    e = np.array(av["edges"]); centres = (e[:-1] + e[1:]) / 2
    assert abs(centres[np.argmax(av["groups"][0]["rate"])]) < 15               # the spikes sit on the peak
    bare = dio.TimeSeries(t, lfp[:, None], ["ref"], fs, units=["uV"], source=tempfile.mkdtemp())
    for kind in ("spikes", "units"):                                           # nothing detected yet: say so
        try:
            phase.spike_source(bare, kind); raise AssertionError("missing spikes accepted")
        except FileNotFoundError as e:
            assert "Spikes tab" in str(e), e
    rot = phase.rotate(ph, "trough")                                           # 0 deg at the trough instead
    assert rot["phase"][0] == (ph["phase"][0] + 180) % 360
    assert np.allclose(np.sort(phase.cycle_times(rot, ranges, "trough")),
                       np.sort(phase.cycle_times(ph, ranges, "trough")))       # turning the frame does not move the troughs
    rav = phase.cycle_average(rot, ranges, src, "trough")
    assert abs(np.array(rav["t_ms"])[np.argmin(rav["band_mean"])]) < 10        # the trough sits at 0 ms
    ev = phase.epoch_view(ph, ranges, 0, src)
    assert ev["t0"] == 10.0 and ev["t1"] == 20.0 and len(ev["t"]) == len(ev["filtered"]) == len(ev["phase"])
    assert len(ev["t"]) <= phase.TRACE_POINTS and ev["spikes"][0]["group"] == 0
    assert len(ev["spikes"][0]["t"]) == inside.size


@test
def nss_detector_finds_injected_ripples_with_the_paper_thresholds():
    """The standard offline method: 120-200 Hz, squared, normalised, peak over 5 SD, edges at
    2 SD, 15-100 ms."""
    fs = 2500.0
    t = np.arange(0, 120, 1 / fs)
    rng = np.random.default_rng(31)
    y = rng.standard_normal(t.size) * 30.0 + 150.0 * np.sin(2 * np.pi * 7 * t)   # noise plus theta
    want = [10.0, 25.0, 40.0, 55.0, 70.0, 85.0]
    for st in want:                                      # 60 ms, 150 Hz ripples
        k = (t >= st) & (t < st + 0.06)
        y[k] += 220.0 * np.sin(2 * np.pi * 150 * t[k]) * np.hanning(int(k.sum()))
    long_one = 100.0                                     # 300 ms of in-band noise: too long to be a ripple
    k = (t >= long_one) & (t < long_one + 0.3)
    y[k] += 200.0 * np.sin(2 * np.pi * 150 * t[k])
    ts = dio.TimeSeries(t, y[:, None], ["A"], fs, units=["uV"], source="synthetic")
    r = detect.run(ts, "swr_nss", "A", {}, save=False)
    got = np.array([e["peak"] for e in r["events"]])
    assert r["params"]["low_hz"] == 120 and r["params"]["high_hz"] == 200 and r["params"]["order"] == 3
    assert r["params"]["high_sd"] == 5 and r["params"]["low_sd"] == 2 and r["params"]["min_ms"] == 15
    assert len(got) == len(want), [round(x, 2) for x in got]
    assert all(any(w <= g <= w + 0.06 for g in got) for w in want), got   # each peak inside its ripple
    for e in r["events"]:
        assert 0.015 <= e["duration"] <= 0.100 and e["score"] >= 5, e
    assert any(long_one <= e["peak"] <= long_one + 0.3 and e["reason"] == "duration" for e in r["rejected"]), r["rejected"]
    assert r["stats"]["scale"] > 0 and r["stats"]["robust"] == 0.0     # mean / SD, not robust
    assert len(r["scan"]["t"]) == len(r["scan"]["local"]) and max(r["scan"]["local"]) > 5
    # the same signal, filtered and normalised again on a window, is what the Events tab draws
    f = detect.DETECTORS["swr_nss"].features(ts.window(9.5, 10.5), "A", r["stats"], **r["params"])
    assert set(f) == {"raw", "ripple band", "NSS"} and max(f["NSS"]) > 5
    assert detect.DETECTORS["swr_nss"].thresholds(r["params"]) == {"NSS": {"low_sd": 2.0, "high_sd": 5.0}}


@test
def dentate_spikes_are_found_by_amplitude_and_kept_apart():
    """Peaks of the 1-200 Hz LFP over six times its median absolute amplitude, 50 ms apart,
    largest on the contact that carries them."""
    fs, nch = 1250.0, 8
    t = np.arange(0, 60, 1 / fs)
    rng = np.random.default_rng(13)
    X = rng.standard_normal((t.size, nch)) * 25.0 + 120.0 * np.sin(2 * np.pi * 7 * t)[:, None]
    want = [5.0, 12.0, 19.0, 26.0, 33.0, 40.0, 47.0]
    shape = lambda k: np.exp(-((np.arange(k) - k / 2) / (k / 6)) ** 2)
    for st in want:                                      # 25 ms positive spike, largest on channel 5
        i = int(st * fs); k = int(0.025 * fs)
        for c in range(nch):
            X[i:i + k, c] += 900.0 * shape(k) * np.exp(-((c - 5) ** 2) / 4.0)
    i = int(5.03 * fs); k = int(0.025 * fs)              # a second spike 30 ms after the first
    X[i:i + k, 5] += 700.0 * shape(k)
    names = [f"CH{c + 1}" for c in range(nch)]
    ts = dio.TimeSeries(t, X, names, fs, units=["uV"] * nch, source="synthetic")
    r = detect.run(ts, "dentate_spikes", "auto", {}, save=False)
    assert r["channel"] == "CH6", r["channel"]           # the contact with the largest spikes
    assert r["params"]["factor"] == 6 and r["params"]["refractory_ms"] == 50
    got = np.array([e["peak"] for e in r["events"]])
    assert len(got) == len(want), [round(x, 2) for x in got]   # the 30 ms neighbour is not a second event
    assert all(min(abs(got - w)) < 0.03 for w in want), got
    assert all(e["score"] >= 1 and e["amp_uV"] > r["threshold_uV"] for e in r["events"])
    prof = r["depth_profile"]
    assert prof["max_channel"] == "CH6" and len(prof["mean_uV"]) == nch
    assert prof["mean_uV"][5] > prof["mean_uV"][0] > 0                   # positive, peaking at CH6
    assert all(e.get("max_channel") == "CH6" for e in r["events"])
    quiet = detect.run(ts, "dentate_spikes", "auto", {"factor": 20.0}, save=False)
    assert quiet["n_events"] == 0                                        # nothing reaches 20x


@test
def theta_phase_coherence_finds_the_reversal_depth():
    """Theta reverses across the pyramidal layer: the depth profile of the phase against a
    reference contact should put a ~180 deg step exactly where the reversal was built in."""
    fs, dur, f0, nch = 1000.0, 40.0, 8.0, 16
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(21)
    theta = np.sin(2 * np.pi * f0 * t)
    X = np.empty((t.size, nch))
    for i in range(nch):
        amp = 200.0 if i < 8 else 120.0                      # above the layer the rhythm is larger
        flip = 1.0 if i < 8 else -1.0                        # ...and below it, half a cycle away
        X[:, i] = flip * amp * theta + rng.standard_normal(t.size) * 15.0
    X[:, 3] = rng.standard_normal(t.size) * 15.0             # one dead contact: no theta at all
    names = [f"CH{i + 1}" for i in range(nch)]
    ts = dio.TimeSeries(t, X, names, fs, units=["uV"] * nch, source="synthetic")
    layout = {"name": "fake", "contacts": [{"name": n, "x_um": 0.0, "y_um": -25.0 * i, "shank": 0}
                                           for i, n in enumerate(names)], "bad": []}
    ranges = phase.event_ranges([{"start": 5.0, "end": 35.0}])
    out = phase.coherence_profile(ts, ranges, (5.0, 10.0), layout=layout)
    rows = out["rows"]
    assert [r["name"] for r in rows] == names and out["seconds"] > 29     # shallow -> deep, inside the epoch
    assert out["channel"] in names[:8]                                    # reference = the largest theta
    locked = [r for r in rows if r["name"] != "CH4"]
    assert all(r["plv"] > 0.95 for r in locked), [(r["name"], r["plv"]) for r in locked]
    assert rows[3]["plv"] < 0.5, rows[3]                                  # the dead contact follows nothing
    for r in locked[:7]:
        assert abs(r["phase_deg"]) < 15, r                                # in phase with the reference
    for r in rows[8:]:
        assert abs(abs(r["phase_deg"]) - 180) < 15, r                     # reversed below the layer
    rev = out["reversal"]
    assert rev and rev["above"] == "CH8" and rev["below"] == "CH9", rev
    assert abs(abs(rev["step_deg"]) - 180) < 15 and rev["depth_um"] == -187.5, rev
    assert "CH8" in out["reversal_note"] and out["geometry"]
    # no reversal when every contact rides the same cycle
    flat = dio.TimeSeries(t, np.column_stack([200.0 * theta + rng.standard_normal(t.size) * 15.0] * nch),
                          names, fs, units=["uV"] * nch, source="synthetic")
    out2 = phase.coherence_profile(flat, ranges, (5.0, 10.0), layout=layout)
    assert out2["reversal"] is None and "no phase reversal" in out2["reversal_note"]
    try:
        phase.coherence_profile(ts, phase.event_ranges([{"start": 5.0, "end": 5.2}]), (5.0, 10.0), layout=layout)
        raise AssertionError("accepted a fifth of a second of signal")
    except ValueError as e:
        assert "too little signal" in str(e)


@test
def multitaper_theta_labels_the_top_percentile_of_the_session():
    """5-12 Hz power from a 4 s / 2 s multitaper spectrogram, high theta = over the 90th
    percentile of the session's own distribution."""
    fs, nch = 1250.0, 4
    t = np.arange(0, 400, 1 / fs)
    rng = np.random.default_rng(17)
    env = np.zeros(t.size)
    for a in (50, 150, 250):                              # three 20 s bouts of strong theta
        env[(t > a) & (t < a + 20)] = 1.0
    X = rng.standard_normal((t.size, nch)) * 20.0
    for c in range(nch):
        X[:, c] += (60.0 + 90.0 * env) * np.sin(2 * np.pi * 8 * t) * (1.0 if c == 2 else 0.35)
    names = [f"CH{c + 1}" for c in range(nch)]
    ts = dio.TimeSeries(t, X, names, fs, units=["uV"] * nch, source="synthetic")
    r = detect.run(ts, "theta_mt", "auto", {}, save=False)
    assert r["channel"] == "CH3"                          # the contact with the most theta power
    assert r["params"]["win_s"] == 4 and r["params"]["step_s"] == 2 and r["params"]["percentile"] == 90
    assert r["n_events"] >= 3, r["n_events"]
    covered = sum(e["duration"] for e in r["events"])
    assert 20 < covered < 80, covered                     # about a tenth of 400 s, by construction
    for a in (50, 150, 250):                              # every bout holds at least one epoch...
        assert any(e["end"] > a and e["start"] < a + 20 for e in r["events"]), (a, r["events"])
    assert all(any(a < e["peak"] < a + 20 for a in (50, 150, 250)) for e in r["events"]), r["events"]
    # ...and nothing outside them: a percentile threshold cuts inside the bouts, so they are patchy
    assert all(e["score"] >= 1 for e in r["events"]) and r["stats"]["scale"] > 0
    assert r["threshold_power"] > 0 and len(r["scan"]["t"]) == len(r["scan"]["z"])   # the Theta tab's scan
    assert max(r["scan"]["z"]) > 1 and r["thresholds"] == {"high": 1.0}
    f = detect.DETECTORS["theta_mt"].features(ts.window(150.0, 155.0), "CH3", r["stats"], **r["params"])
    assert set(f) == {"raw", "theta band", "theta power (x threshold)"}
    assert max(f["theta power (x threshold)"]) > 1       # inside a bout, above the session threshold
    quiet = detect.DETECTORS["theta_mt"].features(ts.window(100.0, 105.0), "CH3", r["stats"], **r["params"])
    assert max(quiet["theta power (x threshold)"]) < 1   # between bouts, below it


@test
def multitaper_spectrogram_resolves_a_tone():
    fs = 1000.0
    t = np.arange(0, 30, 1 / fs)
    y = np.sin(2 * np.pi * 8 * t) + 0.05 * np.random.default_rng(1).standard_normal(t.size)
    tc, f, spec = an.multitaper_spectrogram(y, fs, 4.0, 2.0)
    assert tc.size == 14 and abs(tc[1] - tc[0] - 2.0) < 1e-9 and spec.shape == (tc.size, f.size)
    assert abs(f[int(np.argmax(spec[0]))] - 8.0) < 0.3
    p = an.band_power(f, spec, 5, 12)
    assert p.min() > 0 and np.ptp(p) / p.mean() < 0.2     # a steady tone gives a steady power
    assert an.band_power(f, spec, 100, 200).mean() < p.mean() / 100
    assert an.multitaper_spectrogram(y[:10], fs, 4.0, 2.0)[0].size == 0


@test
def theta_delta_ratio_tracks_the_state_of_the_signal():
    """The Theta tab's activity trace: high while theta dominates, low while delta does."""
    fs = 1000.0
    t = np.arange(0, 120, 1 / fs)
    rng = np.random.default_rng(12)
    y = rng.standard_normal(t.size) * 10.0
    run = t < 60                                        # theta first, then a slow-wave stretch
    y[run] += 200.0 * np.sin(2 * np.pi * 8 * t[run])
    y[~run] += 400.0 * np.sin(2 * np.pi * 2 * t[~run])
    aux = rng.standard_normal(t.size) * 0.5
    aux[run] += 4.0 * np.sin(2 * np.pi * 6 * t[run])    # the animal moves while theta runs
    ts = dio.TimeSeries(t, np.column_stack([y, aux]), ["lfp", "AUX1"], fs, units=["uV", "mV"], source="synthetic")
    a = phase.activity(ts, "lfp", smooth_s=2.0, points=200)
    tt, ratio = np.array(a["t"]), np.array(a["ratio"])
    early, late = ratio[(tt > 5) & (tt < 55)], ratio[(tt > 65) & (tt < 115)]
    assert np.median(early) > 10 * np.median(late), (np.median(early), np.median(late))
    assert len(tt) == len(ratio) == len(a["theta"]) == len(a["delta"]) and len(tt) <= 200
    assert a["bin_s"] > 0 and a["median"] > 0 and a["theta_band"] == [5.0, 10.0] and a["delta_band"] == [1.0, 4.0]
    mot = np.array(a["motion"])                          # the accelerometer, on the same bins
    assert mot.size == tt.size and np.median(mot[tt < 55]) > np.median(mot[tt > 65])
    assert phase.activity(dio.TimeSeries(t, y[:, None], ["lfp"], fs, units=["uV"], source="x"),
                          "lfp", points=50)["motion"] is None      # no aux channels: no accelerometer


@test
def theta_detector_saves_a_whole_session_scan():
    fs = 1000.0
    t = np.arange(0, 120, 1 / fs)
    rng = np.random.default_rng(7)
    y = rng.standard_normal(t.size) * 20.0
    burst = (t > 40) & (t < 60)
    y[burst] += 300.0 * np.sin(2 * np.pi * 8 * t[burst])
    ts = dio.TimeSeries(t, y[:, None], ["a"], fs, units=["uV"], source="synthetic")
    r = detect.run(ts, "theta", "a", {"z_enter": 1.0, "z_peak": 2.0, "min_s": 1.0}, save=False)
    assert any(e["start"] < 45 and e["end"] > 55 for e in r["events"]), r["events"]   # the burst is one epoch
    sc = r["scan"]
    assert len(sc["t"]) == len(sc["z"]) and sc["bin_s"] > 0
    assert max(sc["z"]) > 2.0 and r["thresholds"] == {"z_enter": 1.0, "z_peak": 2.0}


@test
def mountainsort_separates_two_cells_and_roundtrips():
    """The Spikes tab's sorting path, on two cells with different waveforms on a 4-contact probe."""
    ok, msg = sorting.available()
    if not ok:
        print("      skipped:", msg)
        return
    fs, dur = 30000.0, 6.0
    rng = np.random.default_rng(11)
    n, nc = int(fs * dur), 4
    data = rng.standard_normal((n, nc)) * 8.0
    def shape(width_ms, amp):
        k = np.arange(-45, 46) / fs * 1000.0
        return amp * (-np.exp(-(k / width_ms) ** 2) + 0.35 * np.exp(-((k - 1.5 * width_ms) / (1.7 * width_ms)) ** 2))
    cells = [(np.arange(0.2, dur - 0.2, 0.031), shape(0.25, 230.0), [1.0, 0.55, 0.1, 0.0]),   # near contact 0
             (np.arange(0.31, dur - 0.2, 0.047), shape(0.45, 180.0), [0.0, 0.1, 0.6, 1.0])]   # near contact 3
    for times, w, weights in cells:
        for st in times:
            i = int(st * fs)
            for c, g in enumerate(weights):
                data[i - 45:i + 46, c] += g * w
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        raw = (data / 0.195).astype(np.int16)                      # int16 on disk, like continuous.dat
        raw.tofile(d / "continuous.dat")
        mm = np.memmap(d / "continuous.dat", dtype=np.int16, mode="r", shape=(n, nc))
        names = [f"CH{i + 1}" for i in range(nc)]
        ts = dio.TimeSeries(np.arange(n) / fs, mm, names, fs, units=["uV"] * nc,
                            scale=[0.195] * nc, source=str(d))
        layout = {"contacts": [{"name": nm, "x_um": 0.0, "y_um": 25.0 * i} for i, nm in enumerate(names)], "bad": []}
        seen = []
        res = sorting.sort_mountainsort(ts, layout=layout, threshold_sd=5.0, scheme="1",
                                        progress=lambda f, m: seen.append(m))
        assert seen and res["meta"]["scheme"] == "1" and res["channel_names"] == names
        injected = len(cells[0][0]) + len(cells[1][0])
        # scheme 2 (what a recording over five minutes gets) names its parameters differently
        two = sorting.sort_mountainsort(ts, layout=layout, threshold_sd=5.0, scheme="2")
        assert two["meta"]["scheme"] == "2"
        assert {r["peak_channel"] for r in two["meta"]["units"] if r["n_spikes"] > 0.1 * injected} == {"CH1", "CH4"}, two["meta"]["units"]
        rows = sorted(res["meta"]["units"], key=lambda r: -r["n_spikes"])
        # clustering is not deterministic: a cell may be split and a few spikes may land in a stray
        # unit, but each cell must come out as a unit at its own end of the probe
        big = [r for r in rows if r["n_spikes"] > 0.1 * injected]
        assert {r["peak_channel"] for r in big} == {"CH1", "CH4"}, rows
        found = sum(r["n_spikes"] for r in rows)
        assert 0.8 * injected < found <= injected, (found, injected)
        for r in big:
            assert r["snr"] > 3 and r["isi_violation_frac"] < 0.05, r
        sorting.save(ts, res)
        back = sorting.load(ts)
        assert back["unit_ids"] == res["unit_ids"] and back["times"].size == res["times"].size
        # what the viewer asks for
        sm = sorting.summary(back, ts.duration, bins=40)
        assert len(sm["raster"]["counts"]) == len(back["unit_ids"]) and len(sm["raster"]["t"]) == 40
        assert sm["n_spikes"] == back["times"].size and sm["units"] and sm["meta"]["sorter"] == "mountainsort5"
        u = rows[0]["unit"]
        det = sorting.unit_detail(back, u, max_channels=3)
        assert det["unit"] == u and len(det["channels"]) == 3 and len(det["templates"]) == 3
        assert len(det["templates"][0]) == len(det["t_ms"]) and sum(det["isi_hist"]["counts"]) > 0
        assert rows[0]["peak_channel"] in det["channels"]
        try:
            sorting.unit_detail(back, 9999); raise AssertionError("unknown unit accepted")
        except KeyError:
            pass


@test
def csd_of_a_gaussian_sink_and_event_view():
    # a point-like sink at 200 µm on a 25 µm linear array: CSD minimum at that depth, sources beside it
    depth = np.arange(0, 800, 25.0)[::-1]                    # shallow -> deep as event_view orders it
    t = np.linspace(-0.05, 0.05, 301)
    sink = -np.exp(-((depth - 200) / 40.0) ** 2)             # potential minimum at 200 µm
    phi = sink[:, None] * np.exp(-(t / 0.01) ** 2)[None, :] * 100.0
    c = csd.csd(phi, 25.0, smooth=False)
    i_min = int(np.argmin(c[:, 150]))
    assert abs(depth[i_min] - 200) <= 25, depth[i_min]
    assert c[:, 150].max() > 0 and c[i_min, 150] < 0    # sources flanking the sink
    assert csd.csd(phi, 25.0, smooth=True).shape == phi.shape
    # no geometry: recorded order with unit spacing; with geometry: shallow -> deep on the detection channel's shank
    lay = probe.builtin("a1x32-poly2").layout([f"CH{i}" for i in range(1, 33)] + ["A1_AUX1"], None, ["uV"] * 32 + ["mV"])
    ax = csd.depth_axis(lay, [f"CH{i}" for i in range(1, 33)], "CH20")
    assert ax["geometry"] and ax["spacing_um"] == 25.0 and ax["names"][-1] == "CH20" and len(ax["names"]) == 32
    ax0 = csd.depth_axis(None, ["a", "b"])
    assert ax0["names"] == ["a", "b"] and not ax0["geometry"]

    # event_view on a synthetic 4-channel recording with a ripple on the detection channel
    fs = 1000.0
    tt = np.arange(0, 20, 1 / fs)
    rng = np.random.default_rng(7)
    data = 0.2 * rng.standard_normal((tt.size, 4))
    for r0 in (5.0, 10.0, 15.0):
        w = (tt > r0) & (tt < r0 + 0.06)
        data[:, 1] += np.where(w, 2.0, 0.0) * np.sin(2 * np.pi * 200 * tt)
    ts = dio.TimeSeries(tt, data, ["c0", "c1", "c2", "c3"], fs, units=["uV"] * 4, source="synthetic")
    res = detect.run(ts, "swr", "c1", {"z_enter": 2, "z_peak": 4, "min_ms": 20, "max_ms": 0}, save=False)
    assert res["n_events"] == 3
    v = csd.event_view(ts, None, res, detect.DETECTORS["swr"], 1, pre_ms=100, post_ms=100, lfp_band=(0, 0))
    assert v["index"] == 1 and abs(v["event"]["peak"] - 10.03) < 0.05 and v["lfp"]["names"] == ["c0", "c1", "c2", "c3"]
    assert len(v["t_ms"]) == len(v["lfp"]["data"][0]) == len(v["csd"]["data"][0]) and -101 < v["t_ms"][0] <= -99
    assert set(v["features"]["traces"]) == {"raw", "ripple band", "envelope z"} and max(v["features"]["traces"]["envelope z"]) > 4
    a = csd.event_view(ts, None, res, detect.DETECTORS["swr"], None, pre_ms=50, post_ms=50, lfp_band=(0, 0))
    assert a["average"] and a["event"]["n_averaged"] == 3 and len(a["lfp"]["data"]) == 4

    # continuous CSD map for the Browse tab: multichannel, rows ordered deep -> shallow, diverging colours
    A = an.REGISTRY["csd_map"]
    assert A.multichannel and A.wants_layout and A.schema()["multichannel"]
    r = A.fn(ts.window(9.5, 10.5), "c1", layout=None, **A.coerce({"lfp_low": 0, "lfp_high": 300}))
    img = r.traces[0]
    assert img.kind == "image" and img.cmap == "diverging" and img.z.shape[0] == 4 and img.row_labels == ["c3", "c2", "c1", "c0"]
    j = server.result_to_json(r, 9.9, 10.2, 300)["traces"][0]
    assert j["cmap"] == "diverging" and j["zmin"] == -j["zmax"] and j["row_labels"] == ["c3", "c2", "c1", "c0"] and len(j["x"]) <= 300
    with_lfp = A.fn(ts.window(9.5, 10.5), "c1", layout=None, **A.coerce({"lfp_low": 0, "lfp_high": 300, "show_lfp": True}))
    assert len(with_lfp.traces) == 5 and with_lfp.traces[1].kind == "line"


@test
def layer_estimate_recovers_synthetic_laminar_profile():
    # 12 channels at 25 µm, shallow -> deep in recorded order. Events carry a ripple on the "pyramidal"
    # channel (index 5) and a sharp-wave negative deflection centred deeper (index 8, radiatum) with a
    # weaker one shallower (index 2, oriens).
    fs, n_ch = 2000.0, 12
    t = np.arange(0, 40, 1 / fs)
    rng = np.random.default_rng(21)
    data = 0.5 * rng.standard_normal((t.size, n_ch))
    peaks = np.arange(2.0, 40, 2.0)
    idx = np.arange(n_ch)
    for pk in peaks:
        w = np.exp(-((t - pk) / 0.02) ** 2)                                   # 20 ms sharp wave
        data += -60.0 * np.outer(w, np.exp(-((idx - 8) / 1.2) ** 2))         # radiatum sink
        data += -25.0 * np.outer(w, np.exp(-((idx - 2) / 1.2) ** 2))         # oriens sink (weaker)
        rip = np.exp(-((t - pk) / 0.015) ** 2) * np.sin(2 * np.pi * 160 * t + rng.uniform(0, 2 * np.pi))
        data[:, 5] += 40.0 * rip                                            # ripple only at pyramidale
    names = [f"c{i}" for i in range(n_ch)]
    ts = dio.TimeSeries(t, data, names, fs, units=["uV"] * n_ch, source="synthetic")
    depth_um = [(n_ch - 1 - i) * 25.0 for i in range(n_ch)]                 # index 0 shallowest
    layout = {"name": "fake", "contacts": [{"name": nm, "shank": 0, "x_um": 0.0, "y_um": d, "recorded": i + 1, "input": i + 1}
                                            for i, (nm, d) in enumerate(zip(names, depth_um))]}
    result = {"name": "swr", "label": "SWR", "channel": "c5", "params": {}, "stats": {},
              "events": [{"start": p - 0.02, "end": p + 0.02, "peak": p, "score": 5.0} for p in peaks]}
    est = csd.layer_estimate(ts, layout, result, pre_ms=250, post_ms=100, ripple_band=(120, 200))
    L = est["layers"]
    assert L["pyramidale"]["channel"] == "c5", L["pyramidale"]
    assert L["radiatum"]["channel"] == "c8" and L["radiatum"]["depth_um"] < L["pyramidale"]["depth_um"], L["radiatum"]
    assert L["oriens"]["channel"] == "c2" and L["oriens"]["depth_um"] > L["pyramidale"]["depth_um"], L["oriens"]
    assert est["diagnostics"]["pyr_margin"] > 1.5 and est["diagnostics"]["agree_pyr_ripple_power"]
    assert L["radiatum"]["extent_um"][0] <= L["radiatum"]["depth_um"] <= L["radiatum"]["extent_um"][1]
    assert len(est["profiles"]["ripple_source"]) == n_ch and est["n_events"] == len(peaks)
    with tempfile.TemporaryDirectory() as d:
        ts.source = str(Path(d))
        csd.save_layers(ts, est)
        back = csd.load_layers(ts)
        assert back["layers"]["pyramidale"]["channel"] == "c5" and Path(back["path"]).exists()


@test
def channel_quality_flags_one_noisy_contact_and_interpolates():
    fs, n_ch = 2000.0, 10
    t = np.arange(0, 120, 1 / fs)
    rng = np.random.default_rng(31)
    common = an.bandpass(rng.standard_normal(t.size), fs, 1, 100) * 50
    data = np.column_stack([common * (1 + 0.15 * i) + 3 * rng.standard_normal(t.size) for i in range(n_ch)])   # laminar gradient
    data[:, 4] += 60 * rng.standard_normal(t.size)                       # one noisy contact
    names = [f"c{i}" for i in range(n_ch)]
    ts = dio.TimeSeries(t, data, names, fs, units=["uV"] * n_ch, source="synthetic")
    layout = {"name": "fake", "contacts": [{"name": nm, "shank": 0, "x_um": 0.0, "y_um": (n_ch - i) * 25.0, "recorded": i + 1, "input": i + 1}
                                            for i, nm in enumerate(names)]}
    q = channels.channel_quality(ts, layout, sample_s=10)
    assert list(q["auto"]) == ["c4"], q["auto"]                          # the gradient is not flagged, the noisy one is
    assert q["median_corr"] > 0.8
    rows = np.arange(5)[:, None] * np.ones((5, 4))
    out = channels.interpolate_bad(rows, ["a", "b", "c", "d", "e"], ["c", "e"])
    assert np.allclose(out[2], 2) and np.allclose(out[4], 3) and np.allclose(out[1], 1)
    state = {"quality": q, "manual": {"bad": ["c7"], "good": ["c4"]}}
    assert channels.effective_bad(state) == ["c7"]
    with tempfile.TemporaryDirectory() as d:
        ts.source = str(Path(d)); channels.save(ts, state); assert channels.load(ts)["manual"]["bad"] == ["c7"]


@test
def event_features_table_csv_and_filters():
    ts, t = make_movement_recording(fs=2000.0)
    base = {"z_enter": 2, "z_peak": 3, "min_ms": 20, "max_ms": 0, "ref_channel": "none", "gate_movement": True}
    r = detect.run(ts, "swr", "lfp", base, save=False)
    assert r["n_events"] == 3 and r["n_rejected"] == 2
    spk = {"times": np.sort(np.concatenate([np.random.default_rng(1).uniform(0, 60, 600), np.repeat([10.02, 20.02, 30.02], 15)]))}
    features.event_features(ts, None, r, None, spk)
    rows = features.table(r)["rows"]
    assert len(rows) == 5 and all("features" in e for e in r["events"] + r["rejected"])
    acc = [x for x in rows if x["set"] == "accepted"]
    assert all(150 < x["peak_hz"] < 250 for x in acc), [x["peak_hz"] for x in acc]
    assert all(x["n_cycles"] > 3 for x in acc) and all(x["mua_z"] > 3 for x in acc)
    assert all(x["control_ratio_db"] > 10 for x in acc)
    assert all(x["move_z"] > 2 for x in rows if x["set"] == "rejected")
    txt = features.to_csv(r)
    assert txt.splitlines()[0].startswith("set,index,reason,start,end,peak,in_range,duration_ms") and len(txt.splitlines()) == 6
    out = features.apply_filter(r, {"n_cycles": {"min": 1000}})           # impossible rule: everything moves
    assert out["moved"] == 3 and r["n_events"] == 0 and all(e["reason"] == "filter:n_cycles" for e in r["rejected"] if "filter" in str(e.get("reason")))
    back = features.restore_filtered(r)
    assert back["restored"] == 3 and r["n_events"] == 3 and r["n_rejected"] == 2


# ---------------------------------------------------------------- server

def http_get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def http_post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


PLUGIN_SRC = '''
from dataanalyzer.analysis import analysis, Param, Result, Trace
@analysis("Scaled copy", [Param("gain", "float", 2.0, min=0, max=10)])
def scaled(ts, channel, gain):
    """multiply by gain"""
    return Result([Trace("scaled", ts.t, ts.channel(channel) * gain)])
'''


@test
def server_endpoints_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        make_fake_openephys(root / "sessA")
        plugin = root / "my_filters.py"
        plugin.write_text(PLUGIN_SRC)
        ts = dio.synthetic(duration=20, fs=1000)
        srv = server.serve_in_background(ts, port=0, root=root, plugins=[plugin])
        try:
            base = srv.url
            meta = http_get(base + "/api/meta")
            assert meta["data"]["channels"] == ["mixed", "bursts", "random_walk"]
            names = [a["name"] for a in meta["analyses"]]
            assert "scaled" in names and "filtered" in names
            assert [a for a in meta["analyses"] if a["name"] == "scaled"][0]["plugin"] == str(plugin.resolve())

            html = urllib.request.urlopen(base + "/").read().decode()
            assert "<canvas" in html

            r = http_get(base + "/api/analyze?analysis=filtered&channel=mixed&low_hz=100&high_hz=250&t0=5&t1=7&points=500")
            assert r["domain"] == "time" and r["params"]["low_hz"] == 100.0 and r["unit"] == "uV"
            tr = r["traces"][-1]
            assert len(tr["x"]) <= 500 and 5 <= tr["x"][0] and tr["x"][-1] <= 7
            assert max(tr["y"]) > 0.5  # burst at t=6 survives the window + filter

            r = http_get(base + "/api/analyze?analysis=scaled&channel=random_walk&gain=3&t0=1&t1=2")
            raw = http_get(base + "/api/analyze?analysis=raw&channel=random_walk&t0=1&t1=2")
            assert np.allclose(r["traces"][0]["y"][:20], np.array(raw["traces"][0]["y"][:20]) * 3)

            r = http_get(base + "/api/analyze?analysis=psd&channel=mixed&nperseg=1024&max_hz=100")
            assert r["domain"] == "frequency" and r["traces"][0]["x"][-1] <= 100

            # all channels: raw from a block read (fine) and from the overview (coarse), other analyses per channel
            r = http_get(base + "/api/multi?analysis=raw&channels=all&t0=1&t1=1.5&points=400")
            assert [c["name"] for c in r["channels"]] == ["mixed", "bursts", "random_walk"] and r["source"] == "block"
            assert all(len(c["traces"][0]["x"]) <= 400 for c in r["channels"])
            dio.OVERVIEW_BUCKETS, saved_buckets = 500, dio.OVERVIEW_BUCKETS   # synthetic data is small; force the coarse path
            try:
                assert ts.overview_stride() == 40 and ts._overview is None   # fine windows must not build it
                r = http_get(base + "/api/multi?analysis=raw&channels=unit:uV&points=200")
            finally:
                dio.OVERVIEW_BUCKETS = saved_buckets
            assert [c["name"] for c in r["channels"]] == ["mixed", "bursts"] and r["source"] == "overview"
            assert len(r["channels"][0]["traces"][0]["x"]) <= 200 and max(r["channels"][1]["traces"][0]["y"]) > 0.8
            r = http_get(base + "/api/multi?analysis=filtered&channels=mixed,bursts&low_hz=100&high_hz=250&t0=5&t1=7")
            assert len(r["channels"]) == 2 and r["channels"][1]["traces"][-1]["name"] == "bursts 100-250 Hz"
            try:
                http_get(base + "/api/multi?analysis=filtered&channels=all&t0=0&t1=20&low_hz=1&high_hz=10")
                server.MULTI_MAX_SAMPLES, saved = 100, server.MULTI_MAX_SAMPLES
                try:
                    http_get(base + "/api/multi?analysis=filtered&channels=all&t0=0&t1=20&low_hz=1&high_hz=10")
                    assert False, "cap should trigger"
                except urllib.error.HTTPError as e:
                    assert e.code == 400 and b"zoom in" in e.read()
                finally:
                    server.MULTI_MAX_SAMPLES = saved
            except urllib.error.HTTPError:
                raise

            # probe geometry: synthetic data has no probe; the fake session has 2 uV channels -> none either, unless given
            pr = http_get(base + "/api/probe")
            assert pr["name"] is None and "no probe" in pr["notes"]

            # detections run as jobs and are saved beside the data (synthetic -> ./synthetic/... under a temp cwd)
            import os, time as _time
            cwd = os.getcwd(); os.chdir(root)
            try:
                dl = http_get(base + "/api/detections")
                assert {"theta", "swr", "immobility"} <= {d["name"] for d in dl["detectors"]} and dl["results"] == {} and dl["spikes"] is None
                job = http_post(base + "/api/detect", {"name": "swr", "channel": "bursts",
                                                        "params": {"z_enter": 2, "z_peak": 3, "min_ms": 20, "max_ms": 0, "ref_channel": "auto"}})["job"]
                for _ in range(200):
                    js = [j for j in http_get(base + "/api/jobs")["jobs"] if j["id"] == job][0]
                    if js["status"] != "running":
                        break
                    _time.sleep(0.05)
                assert js["status"] == "done", js
                dl = http_get(base + "/api/detections")
                sw = dl["results"]["swr"]
                assert sw["ref_channel"] == "mixed" and sw["n_events"] + sw["n_rejected"] == 5 and "rejected" not in sw
                assert (root / "synthetic_dataanalyzer" / "events" / "swr.json").exists()
                ev = http_get(base + "/api/events?t0=0&t1=9")
                assert len(ev["detections"]["swr"]["peaks"]) == sw["n_events"] and len(ev["detections"]["swr"]["rejected_peaks"]) == sw["n_rejected"]
                assert ev["detections"]["swr"]["ref_channel"] == "mixed"
                if sw["n_events"] == 0:      # keep the rest of the test meaningful whatever the reference decided
                    http_post(base + "/api/detect", {"name": "swr", "channel": "bursts", "params": {"z_enter": 2, "z_peak": 3, "min_ms": 20, "max_ms": 0, "ref_channel": "none"}})
                    _time.sleep(1.0)
                job = http_post(base + "/api/spikes/run", {"channels": "unit:uV", "threshold_sd": 6, "chunk_s": 5})["job"]
                for _ in range(400):
                    js = [j for j in http_get(base + "/api/jobs")["jobs"] if j["id"] == job][0]
                    if js["status"] != "running":
                        break
                    _time.sleep(0.05)
                assert js["status"] == "done", js
                sp = http_get(base + "/api/detections")["spikes"]
                assert sp["channel_names"] == ["mixed", "bursts"] and (root / "synthetic_dataanalyzer" / "spikes" / "spikes.npz").exists()
                win = http_get(base + "/api/spikes?t0=0&t1=20")
                assert win["n"] == sp["n_spikes"]
                v = http_get(base + "/api/event?det=swr&index=0&pre_ms=50&post_ms=50&lfp_low=0&lfp_high=0")
                assert v["index"] == 0 and v["n"] >= 1 and v["lfp"]["names"] == ["mixed", "bursts"] and "compute_ms" in v
                assert http_get(base + "/api/event?det=swr&index=avg")["average"] is True
                job = http_post(base + "/api/features", {"det": "swr"})["job"]
                for _ in range(200):
                    js = [j for j in http_get(base + "/api/jobs")["jobs"] if j["id"] == job][0]
                    if js["status"] != "running":
                        break
                    _time.sleep(0.05)
                assert js["status"] == "done", js
                tb = http_get(base + "/api/features?det=swr")
                assert tb["computed"] and len(tb["rows"]) == tb["meta"]["n"] and [c["key"] for c in tb["columns"]][0] == "duration_ms"
                assert http_get(base + "/api/detections")["results"]["swr"]["has_features"]
                csv_txt = urllib.request.urlopen(base + "/api/features.csv?det=swr").read().decode()
                assert csv_txt.startswith("set,index,reason")
                n_before = http_get(base + "/api/detections")["results"]["swr"]["n_events"]
                out = http_post(base + "/api/filter", {"det": "swr", "rules": {"duration_ms": {"max": 0.001}}})
                assert out["n_events"] == 0 and out["moved"] == n_before
                assert http_post(base + "/api/filter", {"det": "swr", "restore": True})["n_events"] == n_before
                ch = http_get(base + "/api/channels")
                assert ch["bad"] == [] and ch["quality"] is None
                ch = http_post(base + "/api/channels", {"bad": ["mixed"]})
                assert ch["bad"] == ["mixed"] and (root / "synthetic_dataanalyzer" / "channels" / "channels.json").exists()
                assert http_post(base + "/api/channels", {"good": ["mixed"]})["bad"] == []
                assert http_get(base + "/api/layers")["layers"] is None
                try:
                    http_post(base + "/api/layers", {"det": "swr"})       # synthetic has 2 headstage channels
                    assert False, "expected refusal"
                except urllib.error.HTTPError as e:
                    assert e.code == 400 and b"five channels" in e.read()
                assert http_post(base + "/api/layers", {"remove": True})["removed"] is False
                assert http_post(base + "/api/detect", {"name": "swr", "remove": True})["removed"] is True
                assert http_get(base + "/api/detections")["results"] == {}
            finally:
                os.chdir(cwd)

            ev = http_get(base + "/api/events?t0=0&t1=9")
            assert ev["events"]["burst centre"]["x"] == [2.0, 6.0]

            # live traces from another process
            v = live.publish("theta power", ts.t[::100], np.arange(ts.n // 100, dtype=float), url=base, note="demo")
            snap = http_get(base + f"/api/live?since=-1&t0=0&t1=20")
            assert snap["version"] == v and snap["traces"][0]["name"] == "theta power" and snap["traces"][0]["meta"] == {"note": "demo"}
            assert http_get(base + f"/api/live?since={v}")["changed"] is False
            live.publish("onsets", [1.0, 2.0], kind="events", url=base)
            assert http_get(base + f"/api/live?since={v}")["changed"] is True
            live.clear(url=base)
            assert http_get(base + "/api/live?since=-1")["traces"] == []

            # browse and open another session
            sessions = http_get(base + "/api/sessions")["sessions"]
            oe = [s for s in sessions if s["kind"] == "openephys"]
            assert len(oe) == 1 and oe[0]["streams"] == ["Stream"] and oe[0]["path"].startswith("sessA")
            opened = http_post(base + "/api/open", {"path": oe[0]["path"], "stream": "Stream"})
            assert opened["data"]["channels"] == ["CH2", "CH1", "A1_AUX1"] and opened["session"]["path"] == oe[0]["path"]
            r = http_get(base + "/api/analyze?analysis=raw&channel=CH1&t0=5.0&t1=5.01")
            assert np.allclose(r["traces"][0]["y"][:3], [0, 0.195, 0.39])
            # explicit probe applies to the opened session; depth ordering follows y_um (shallow first)
            srv.probe_spec = probe.builtin("a1x32-poly2"); srv._apply_probe()
            pr = http_get(base + "/api/probe")
            assert [c["name"] for c in pr["contacts"]] == ["CH2", "CH1"] and pr["unmapped"] == ["A1_AUX1"]
            assert srv.farthest_channel("CH2") == "CH1"       # 2 mapped contacts: the other one
            srv.probe_spec = None; srv._apply_probe()
            assert srv.farthest_channel("CH2") == "CH1"       # no probe: far end of the recorded order
            srv.probe_spec = probe.builtin("a1x32-poly2"); srv._apply_probe()
            res = srv.resolve_channel_params(detect.DETECTORS["swr"], "CH2", {"ref_channel": "auto"})
            assert res["ref_channel"] == "CH1"
            try:   # the fake session has only two headstage channels: the CSD map must refuse, not crash
                http_get(base + "/api/analyze?analysis=csd_map&channel=CH2&t0=5&t1=5.5&lfp_low=0&lfp_high=300")
                assert False, "expected a 400"
            except urllib.error.HTTPError as e:
                assert e.code == 400 and b"three channels" in e.read()
            r = http_get(base + "/api/multi?analysis=raw&channels=all&t0=5&t1=5.1&order=depth")
            y = {c.input: c.y_um for c in srv.probe.contacts}
            expect = sorted(["CH2", "CH1"], key=lambda nm: -y[int(nm[2:])]) + ["A1_AUX1"]
            assert [c["name"] for c in r["channels"]] == expect and r["order"] == "depth"
            srv.probe_spec = None; srv._apply_probe()
            try:
                http_post(base + "/api/open", {"path": "../../etc"})
                assert False, "path escape should fail"
            except urllib.error.HTTPError as e:
                assert e.code == 400

            # set a new root by path: a folder holding one recording auto-opens it
            make_fake_openephys(root / "other" / "sessB", n=1500)
            rr = http_post(base + "/api/root", {"path": str(root / "other")})
            assert rr["root"] == str((root / "other").resolve()) and len(rr["sessions"]) == 1
            assert rr["opened"]["path"] == rr["sessions"][0]["path"] and rr["data"]["n"] == 1500
            # the native dialog, with the backend replaced
            server.pick_folder = lambda initial=None, title="": str(root)
            try:
                pr = http_post(base + "/api/pick", {})
                assert pr["cancelled"] is False and len(pr["sessions"]) >= 2 and pr["data"] is not None
                server.pick_folder = lambda initial=None, title="": None
                assert http_post(base + "/api/pick", {})["cancelled"] is True
            finally:
                from dataanalyzer import dialog
                server.pick_folder = dialog.pick_folder
            http_post(base + "/api/open", {"path": oe[0]["path"], "stream": "Stream"})

            # edit the plugin and reload
            plugin.write_text(PLUGIN_SRC.replace("* gain", "* gain + 100"))
            rl = http_post(base + "/api/reload", {})
            assert "scaled" in rl["loaded"][str(plugin.resolve())]
            r = http_get(base + "/api/analyze?analysis=scaled&channel=A1_AUX1&gain=1&t0=5&t1=5.01")
            assert abs(r["traces"][0]["y"][0] - (3.74e-05 + 100)) < 1e-9

            try:
                http_get(base + "/api/analyze?analysis=nope")
                assert False
            except urllib.error.HTTPError as e:
                assert e.code == 400
        finally:
            srv.shutdown()
            srv.server_close()


# ---------------------------------------------------------------- runner

def main(argv):
    pattern = argv[1] if len(argv) > 1 else ""
    failed = 0
    for fn in TESTS:
        if pattern and pattern not in fn.__name__:
            continue
        try:
            fn()
            print(f"ok    {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
