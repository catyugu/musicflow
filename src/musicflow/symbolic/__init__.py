"""Symbolic score input: the written score and the route it is performed in.

Two public entry points, both re-exported here and nowhere else:

* :func:`load_score` reads MusicXML, MXL or MIDI into the shared :class:`musicflow.model.Score`.
* :func:`build_route` lays the written measures out in performance order.
"""

from .loading import load_score
from .route import build_route

__all__ = ["build_route", "load_score"]
