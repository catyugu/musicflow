"""Frame compositing and MP4 encoding for the two display modes.

``render_switch`` shows the original PDF one staff system at a time; ``render_flow`` moves one
engraved strip past a fixed reading line. Both compute every frame from the absolute recording
time and share the same segmented encoder, so an interrupted run resumes segment by segment.
"""

from .flow import render_flow
from .switch import render_switch

__all__ = ["render_flow", "render_switch"]
