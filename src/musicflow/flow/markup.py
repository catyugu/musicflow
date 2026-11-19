"""Geometry of the single-system SVG that Verovio renders for the flow mode.

Verovio nests the engraver's coordinate space in an inner
``<svg class="definition-scale">`` with its own ``viewBox`` while the outer ``<svg>``
carries the layout-unit size. ``FlowLayout`` reports layout units and one raster pixel
is one layout unit, so every coordinate this module returns is converted to the outer
space.

Note ids (``mf-NNNNNN``) land on the ``<g class="note">`` element, whose children are
positioned in absolute coordinates except for the glyph placements, which are
``<use transform="translate(x, y) scale(...)">``. The horizontal position of a note is
therefore the accumulated transform of the outermost glyph placement inside its node,
with every ancestor transform composed in.
"""

from __future__ import annotations

import re
from collections.abc import Set

from lxml import etree

from ..errors import AnalysisError

SVG_NAMESPACE = "http://www.w3.org/2000/svg"

_NOTE_ID = re.compile(r"mf-\d{6}")
_TRANSFORM_FUNCTION = re.compile(r"(\w+)\s*\(([^)]*)\)")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ROOT_TAG = re.compile(r"<svg\b[^>]*>")
_INNER_TAG = re.compile(r'<svg\b[^>]*\bclass="definition-scale"[^>]*>')
_ROOT_ID = re.compile(r'<svg\b[^>]*\bid="([^"]+)"')

Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _attribute(tag: str, name: str) -> str | None:
    found = re.search(rf'\b{name}="([^"]*)"', tag)
    return found.group(1) if found else None


def _length(tag: str, name: str) -> float:
    value = _attribute(tag, name)
    if value is None:
        raise AnalysisError(f"the Verovio SVG root has no {name} attribute")
    try:
        return float(value.removesuffix("px"))
    except ValueError as exc:
        raise AnalysisError(f"the Verovio SVG has an unreadable {name}: {value}") from exc


def _set_length(tag: str, name: str, value: str) -> str:
    return re.sub(rf'\b{name}="[^"]*"', f'{name}="{value}"', tag, count=1)


def _stabilize_ids(markup: str) -> str:
    """Replace Verovio's random per-rendering id token with a fixed one.

    Verovio derives a random token from the toolkit instance and suffixes every internal
    id with it, so the same input renders to different bytes each time. The token is not
    part of any contract here (notes are addressed by their injected ``mf-`` ids), but a
    stable artifact keeps the diagnostic SVG and cached runs comparable.
    """
    root = _ROOT_TAG.search(markup)
    token = _ROOT_ID.search(root.group(0)) if root is not None else None
    if token is None:
        return markup
    suffix = token.group(1)
    return markup.replace(f"-{suffix}", "-mf").replace(f'id="{suffix}"', 'id="mf-strip"')


def _transform_matrix(value: str) -> Matrix:
    """Compose an SVG transform list into a matrix, ignoring unsupported functions."""
    matrix = IDENTITY
    for name, arguments in _TRANSFORM_FUNCTION.findall(value):
        numbers = [float(v) for v in _NUMBER.findall(arguments)]
        if name == "translate" and numbers:
            step: Matrix = (1.0, 0.0, 0.0, 1.0, numbers[0],
                            numbers[1] if len(numbers) > 1 else 0.0)
        elif name == "scale" and numbers:
            step = (numbers[0], 0.0, 0.0, numbers[1] if len(numbers) > 1 else numbers[0],
                    0.0, 0.0)
        elif name == "matrix" and len(numbers) == 6:
            step = (numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5])
        else:
            continue
        matrix = _compose(matrix, step)
    return matrix


def _compose(outer: Matrix, inner: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _glyph_placement(note: etree._Element) -> etree._Element | None:
    """The element whose transform places a note's visible glyph.

    The notehead group wins because accidentals and articulation marks are translated
    too and would shift the reported position by up to a staff space. Rests carry their
    glyph directly, so they fall back to the first translated element of the note.
    """
    head = next((e for e in note.iter() if e.get("class") == "notehead"), None)
    for candidate in (head, note):
        if candidate is None:
            continue
        for element in candidate.iter():
            if element.get("transform"):
                return element
    return None


def _walk(
    element: etree._Element,
    matrix: Matrix,
    wanted: Set[str],
    positions: dict[str, float],
) -> None:
    transform = element.get("transform")
    if transform:
        matrix = _compose(matrix, _transform_matrix(transform))
    identifier = element.get("id")
    if identifier is not None and identifier in wanted and identifier not in positions:
        placement = _glyph_placement(element)
        if placement is not None:
            glyph = _compose(matrix, _transform_matrix(placement.get("transform") or ""))
            positions[identifier] = glyph[4]
    for child in element:
        if isinstance(child.tag, str):
            _walk(child, matrix, wanted, positions)


class SvgDocument:
    """One rendered single-system page, addressed in layout units."""

    def __init__(self, markup: str) -> None:
        self.markup = _stabilize_ids(markup)
        root = _ROOT_TAG.search(self.markup)
        inner = _INNER_TAG.search(self.markup)
        if root is None or inner is None:
            raise AnalysisError("the Verovio SVG has no single-system page geometry")
        self._root_tag = root.group(0)
        self._inner_tag = inner.group(0)
        self.width = _length(self._root_tag, "width")
        self.height = _length(self._root_tag, "height")
        view_box = _attribute(self._inner_tag, "viewBox")
        if view_box is None:
            raise AnalysisError("the Verovio SVG has no definition-scale viewBox")
        try:
            values = [float(v) for v in view_box.replace(",", " ").split()]
        except ValueError as exc:
            raise AnalysisError(f"unreadable definition-scale viewBox: {view_box}") from exc
        if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
            raise AnalysisError(f"unreadable definition-scale viewBox: {view_box}")
        inner_width, inner_height = values[2], values[3]
        self.scale = min(self.width / inner_width, self.height / inner_height)
        self.offset_x = (self.width - inner_width * self.scale) / 2
        self.offset_y = (self.height - inner_height * self.scale) / 2
        self._elements: etree._Element | None = None

    def _element(self) -> etree._Element:
        if self._elements is None:
            parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)
            self._elements = etree.fromstring(self.markup.encode("utf-8"), parser=parser)
        return self._elements

    def note_x(self, wanted: Set[str]) -> dict[str, float]:
        """Horizontal layout-unit position of every wanted note id found in the page."""
        positions: dict[str, float] = {}
        if not wanted:
            return positions
        _walk(self._element(), IDENTITY, wanted, positions)
        for identifier, inner_x in tuple(positions.items()):
            positions[identifier] = self.offset_x + inner_x * self.scale
        return positions

    def staff_space(self) -> float | None:
        """Vertical distance between two staff lines, in layout units."""
        for group in self._element().iter(f"{{{SVG_NAMESPACE}}}g"):
            if group.get("class") != "staff":
                continue
            lines = set()
            for path in group.iter(f"{{{SVG_NAMESPACE}}}path"):
                numbers = [float(v) for v in _NUMBER.findall(path.get("d") or "")]
                if len(numbers) == 4 and numbers[1] == numbers[3]:
                    lines.add(numbers[1])
            ordered = sorted(lines)
            gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False) if b > a]
            if gaps:
                return min(gaps) * self.scale
        return None

    def window(self, x0: float, y0: float, width: int, height: int) -> str:
        """Markup for the crop ``[x0, x0 + width) x [y0, y0 + height)`` at one pixel per unit.

        The inner page is pinned to its own layout-unit size, so the injected root
        ``viewBox`` moves the camera without rescaling the engraving.
        """
        root = _set_length(self._root_tag, "width", f"{width}px")
        root = _set_length(root, "height", f"{height}px")
        root = root.replace("<svg", f'<svg viewBox="{x0} {y0} {width} {height}"', 1)
        inner = self._inner_tag.replace(
            'class="definition-scale"',
            f'class="definition-scale" x="0" y="0" '
            f'width="{self.width}" height="{self.height}"',
            1,
        )
        body = self.markup[len(self._root_tag):].replace(self._inner_tag, inner, 1)
        return root + body
