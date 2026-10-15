"""Publish traces to a running viewer from a separate process.

    from dataanalyzer.live import publish
    publish("ripple power", t, power)            # appears in the viewer's Live panel
    publish("ripple onsets", onsets, kind="events")

Within the same process as the server, call ``server.store.publish`` directly
instead; this module only exists for code that runs elsewhere.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np

DEFAULT_URL = "http://127.0.0.1:8000"


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(f"{url.rstrip('/')}/api/live", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def publish(name: str, x, y=None, kind: str = "line", url: str = DEFAULT_URL, **meta) -> int:
    body = {"name": name, "kind": kind, "meta": meta,
            "x": np.asarray(x, dtype=float).reshape(-1).tolist(),
            "y": None if y is None else np.asarray(y, dtype=float).reshape(-1).tolist()}
    return int(_post(url, body)["version"])


def remove(name: str, url: str = DEFAULT_URL) -> int:
    return int(_post(url, {"remove": name})["version"])


def clear(url: str = DEFAULT_URL) -> int:
    return int(_post(url, {"clear": True})["version"])
