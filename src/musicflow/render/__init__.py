"""Encode the original PDF system or the continuous engraved view."""

from .flow import render_flow
from .switch import render_switch

__all__ = ["render_flow", "render_switch"]
