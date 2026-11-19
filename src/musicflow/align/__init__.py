"""Score-to-audio alignment: the performance position to recording time map.

One public entry point, :func:`align`, re-exported here and nowhere else. Everything else in the
package (features, the staged DTW, onset snapping, reviewed anchors, the report) is internal.
"""

from .alignment import align

__all__ = ["align"]
