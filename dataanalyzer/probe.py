"""Probe geometry: where each recorded channel sits on the electrode.

A :class:`Probe` is a list of contacts, each tied to a headstage input number
(1-based, the ``CH<n>`` naming Open Ephys uses) with a position in µm. Sources:

* built-ins for the NeuroNexus probes used in this lab (tables copied from
  ``../Neuronexus/probe_map.py``, which documents their provenance and the
  validation against recorded data);
* a probeinterface JSON file (``device_channel_indices`` are 0-based amplifier
  channels);
* a CSV with columns ``channel, x_um, y_um`` and optional ``shank, site``.

:meth:`Probe.layout` joins the geometry with a loaded recording's
``channel_info`` so the viewer can draw the probe with the recorded channel
names, and so depth-ordered analyses (CSD) can sort channels by ``y_um``.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .io import HEADSTAGE_UNITS, input_from_name

# NeuroNexus H32 site number (1-based, list index = site-1) -> Intan RHD2132 amplifier
# channel (0-based). Standard H32 > RHD2132 pathway, identical to probeinterface's.
H32_TO_RHD2132 = [16, 17, 18, 20, 21, 22, 31, 30, 29, 27, 26, 25, 24, 28, 23, 19,
                  12, 8, 3, 7, 6, 5, 4, 2, 1, 0, 9, 10, 11, 13, 14, 15]

# A1x32-Poly2-5mm-50s-177: site number at each depth position, tip first
# (H32Maps.pdf p.9). Two staggered columns, 25 µm effective spacing up the shank.
POLY2_DEPTH_SITES = [16, 17, 15, 18, 14, 19, 13, 20, 12, 21, 11, 22,
                     1, 32, 2, 31, 3, 30, 4, 29, 5, 28, 6, 27,
                     7, 26, 8, 25, 9, 24, 10, 23]
POLY2_SPACING_UM = 25.0
POLY2_COLUMN_DX_UM = 43.0   # not on the map sheet; a working value

# A4x8-5mm-50-200-177: 4 shanks × 8 sites, single column, 50 µm pitch, 200 µm shank pitch
A4X8_CONTACT_SITE = [1, 8, 2, 7, 3, 6, 4, 5, 9, 16, 10, 15, 11, 14, 12, 13,
                     17, 24, 18, 23, 19, 22, 20, 21, 25, 32, 26, 31, 27, 30, 28, 29]

PROBE_FILE_NAMES = ("probe.json", "probe.csv")


@dataclass
class Contact:
    input: int                  # headstage input, 1-based (Open Ephys "CH<n>")
    x_um: float
    y_um: float                 # height above the tip; 0 = deepest
    shank: int = 0
    site: int | None = None     # manufacturer site number, if known
    column: int = 0


@dataclass
class Probe:
    name: str
    contacts: list[Contact]
    source: str = ""            # file path or "built-in"
    notes: str = ""

    def by_input(self) -> dict[int, Contact]:
        return {c.input: c for c in self.contacts}

    def to_json(self) -> dict:
        return {"name": self.name, "source": self.source, "notes": self.notes,
                "contacts": [asdict(c) for c in self.contacts]}

    def layout(self, names: list[str], channel_info: list[dict] | None, units: list[str] | None = None) -> dict:
        """Geometry for each recorded channel, in recorded order. Unmapped channels (aux, or
        inputs the probe does not list) are reported separately."""
        by_input = self.by_input()
        out, unmapped = [], []
        for i, name in enumerate(names):
            kind = channel_info[i].get("kind") if channel_info and i < len(channel_info) else None
            if (kind and kind != "headstage") or (units and units[i] not in HEADSTAGE_UNITS):
                unmapped.append(name)               # aux / ADC channels and processor outputs are not electrodes
                continue
            inp = None
            if channel_info and i < len(channel_info):
                inp = channel_info[i].get("input")
            if inp is None:
                inp = input_from_name(name)
            c = by_input.get(inp) if inp is not None else None
            if c is None:
                unmapped.append(name)
                continue
            out.append({"name": name, "recorded": i + 1, "input": inp, "unit": units[i] if units else "",
                        **asdict(c)})
        # depth index within each shank: 0 = deepest (smallest y)
        for shank in {c["shank"] for c in out}:
            ranked = sorted((c for c in out if c["shank"] == shank), key=lambda c: (c["y_um"], c["x_um"]))
            for d, c in enumerate(ranked):
                c["depth_index"] = d
        return {"name": self.name, "source": self.source, "notes": self.notes,
                "contacts": out, "unmapped": unmapped, "n_contacts": len(self.contacts)}


# ---------------------------------------------------------------------------
# built-ins
# ---------------------------------------------------------------------------

def builtin(name: str) -> Probe:
    key = name.lower().replace("_", "-")
    if key in ("a1x32-poly2", "poly2", "a1x32"):
        contacts = []
        for d, site in enumerate(POLY2_DEPTH_SITES):
            amp = H32_TO_RHD2132[site - 1]
            contacts.append(Contact(input=amp + 1, x_um=(d % 2) * POLY2_COLUMN_DX_UM, y_um=d * POLY2_SPACING_UM,
                                    shank=0, site=site, column=d % 2))
        return Probe("A1x32-Poly2-5mm-50s-177-H32", contacts, "built-in",
                     "site order from the NeuroNexus H32 map sheet; column separation 43 µm is a working value")
    if key in ("a4x8", "a4x8-5mm-50-200-177"):
        contacts = []
        for i in range(32):
            shank, d = i // 8, i % 8
            amp = H32_TO_RHD2132[i]
            contacts.append(Contact(input=amp + 1, x_um=shank * 200.0, y_um=d * 50.0, shank=shank,
                                    site=A4X8_CONTACT_SITE[i]))
        return Probe("A4x8-5mm-50-200-177-H32", contacts, "built-in")
    raise ValueError(f"unknown built-in probe {name!r}; known: a1x32-poly2, a4x8")


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def load_probe(path: str | Path) -> Probe:
    path = Path(path)
    if not path.exists():
        return builtin(str(path))          # allow --probe a1x32-poly2
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        return _load_csv(path)
    raise ValueError(f"unsupported probe file {path}")


def _load_json(path: Path) -> Probe:
    d = json.loads(path.read_text())
    if "probes" in d:                       # probeinterface
        p = d["probes"][0]
        pos = p["contact_positions"]
        dev = p.get("device_channel_indices") or list(range(len(pos)))
        shanks = p.get("shank_ids") or ["0"] * len(pos)
        ids = p.get("contact_ids") or [None] * len(pos)
        contacts = []
        for (x, y), amp, sh, cid in zip(pos, dev, shanks, ids):
            if amp is None or int(amp) < 0:
                continue
            site = int(cid) if cid not in (None, "") and str(cid).isdigit() else None
            contacts.append(Contact(input=int(amp) + 1, x_um=float(x), y_um=float(y), shank=int(sh or 0), site=site))
        name = (p.get("annotations") or {}).get("name") or path.stem
        return Probe(name, contacts, str(path), "probeinterface JSON; device_channel_indices are 0-based")
    contacts = [Contact(int(c["input"]), float(c["x_um"]), float(c["y_um"]), int(c.get("shank", 0)),
                        c.get("site"), int(c.get("column", 0))) for c in d["contacts"]]
    return Probe(d.get("name", path.stem), contacts, str(path), d.get("notes", ""))


def _load_csv(path: Path) -> Probe:
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    contacts = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter=delim):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            raw = row.get("channel") or row.get("input") or row.get("ch") or row.get("name") or ""
            inp = input_from_name(raw) if not raw.isdigit() else int(raw)
            if inp is None:
                raise ValueError(f"{path}: cannot read channel {raw!r}")
            x = float(row.get("x_um") or row.get("x") or 0)
            y = float(row.get("y_um") or row.get("y") or 0)
            site = row.get("site")
            contacts.append(Contact(inp, x, y, int(row.get("shank") or 0), int(site) if site else None))
    return Probe(path.stem, contacts, str(path))


def auto_probe(rec_dir: str | Path | None, n_headstage: int) -> Probe | None:
    """A probe.json/probe.csv near the recording, else the lab's current probe when the
    channel count fits (flagged as assumed)."""
    if rec_dir:
        rec_dir = Path(rec_dir)
        for parent in [rec_dir, *rec_dir.parents[:3]]:
            for fn in PROBE_FILE_NAMES:
                if (parent / fn).exists():
                    return load_probe(parent / fn)
    if n_headstage == 32:
        p = builtin("a1x32-poly2")
        p.notes = "ASSUMED (no probe file found): " + p.notes + ". Pass --probe FILE|a4x8 to override."
        return p
    return None
