"""DataAnalyzer: view and analyse time series through a lightweight local server.

The ``@analysis`` decorator for registering new analyses lives in
``dataanalyzer.analysis`` (imported here as a submodule, not re-exported, so
the module name stays usable).
"""

from . import analysis  # noqa: F401  (submodule; keep the name bound to the module)
from . import channels, csd, detect, features, probe, spikes  # noqa: F401
from .analysis import REGISTRY, Param, Result, Trace
from .io import TimeSeries, load, load_csv, load_npy, load_npz, load_openephys, synthetic
from .server import DataServer, LiveStore, serve, serve_in_background

__all__ = [
    "TimeSeries", "load", "load_csv", "load_npy", "load_npz", "load_openephys", "synthetic",
    "REGISTRY", "Param", "Result", "Trace", "analysis",
    "DataServer", "LiveStore", "serve", "serve_in_background",
]
