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
from itertools import pairwise

from lxml import etree

from ..errors import AnalysisError

SVG_NAMESPACE = "http://www.w3.org/2000/svg"

_TRANSFORM_FUNCTION = re.compile(r"(\w+)\s*\(([^)]*)\)")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ROOT_TAG = re.compile(r"<svg\b[^>]*>")
_INNER_TAG = re.compile(r'<svg\b[^>]*\bclass="definition-scale"[^>]*>')

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


def _transform_matrix(value: str) -> Matrix:
    """Compose an SVG transform list into a matrix, ignoring unsupported functions."""
    matrix = IDENTITY
    for name, arguments in _TRANSFORM_FUNCTION.findall(value):
        numbers = [float(v) for v in _NUMBER.findall(arguments)]
        if name == "translate" and numbers:
            step: Matrix = (1.0, 0.0, 0.0, 1.0, numbers[0], numbers[1] if len(numbers) > 1 else 0.0)
        elif name == "scale" and numbers:
            step = (numbers[0], 0.0, 0.0, numbers[1] if len(numbers) > 1 else numbers[0], 0.0, 0.0)
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


def _walk_staff_lines(
    element: etree._Element,
    matrix: Matrix,
    lines: list[tuple[float, float, float]],
) -> None:
    """Collect horizontal staff-line segments, with every ancestor transform applied."""
    transform = element.get("transform")
    if transform:
        matrix = _compose(matrix, _transform_matrix(transform))
    if etree.QName(element).localname == "path":
        parent = element.getparent()
        if parent is not None and parent.get("class") == "staff":
            numbers = [float(v) for v in _NUMBER.findall(element.get("d") or "")]
            if len(numbers) == 4 and numbers[1] == numbers[3]:
                start = (
                    matrix[0] * numbers[0] + matrix[2] * numbers[1] + matrix[4],
                    matrix[1] * numbers[0] + matrix[3] * numbers[1] + matrix[5],
                )
                end = (
                    matrix[0] * numbers[2] + matrix[2] * numbers[3] + matrix[4],
                    matrix[1] * numbers[2] + matrix[3] * numbers[3] + matrix[5],
                )
                if abs(start[1] - end[1]) < 1e-6:
                    lines.append((min(start[0], end[0]), start[1], max(start[0], end[0])))
    for child in element:
        if isinstance(child.tag, str):
            _walk_staff_lines(child, matrix, lines)


class SvgDocument:
    """One rendered single-system page, addressed in layout units.

    The markup is kept exactly as Verovio produced it. It carries a stylesheet scoped by
    the page's own id (``#<token> path {stroke: currentColor}`` and the rule that hides
    the invisible break anchors), so rewriting any id would silently drop every
    stroke-only element: staff lines, stems, barlines, slurs and hairpins. Verovio's id
    token is fixed by the ``xmlIdSeed`` option instead of by post-processing.
    """

    def __init__(self, markup: str) -> None:
        self.markup = markup
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
        rows = sorted({round(line[1], 6) for line in self.staff_lines()})
        gaps = [b - a for a, b in pairwise(rows) if b > a + 1e-9]
        return min(gaps) if gaps else None

    def staff_lines(self) -> tuple[tuple[float, float, float], ...]:
        """Every horizontal staff line as ``(x0, y, x1)`` in layout units."""
        lines: list[tuple[float, float, float]] = []
        to_layout: Matrix = (self.scale, 0.0, 0.0, self.scale, self.offset_x, self.offset_y)
        _walk_staff_lines(self._element(), to_layout, lines)
        return tuple(lines)

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
            f'class="definition-scale" x="0" y="0" width="{self.width}" height="{self.height}"',
            1,
        )
        body = self.markup[len(self._root_tag) :].replace(self._inner_tag, inner, 1)
        return root + body
