"""MusicXML container handling and stable note identifiers.

Every note in a score gets an ``mf-NNNNNN`` identifier, assigned in document order across the whole
document, rests included. The identifier is written into the ``id`` attribute of the ``<note>``
element so that Verovio propagates it into its SVG (``id="mf-000123"``) and its timemap, which is
what lets the flow renderer attach a horizontal position to each score note.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from lxml import etree

from .errors import InputError

NOTE_ID_PREFIX = "mf-"
NOTE_ID_DIGITS = 6

MUSICXML_SUFFIXES = (".musicxml", ".xml", ".mxl")
MIDI_SUFFIXES = (".mid", ".midi")


def note_id(number: int) -> str:
    return f"{NOTE_ID_PREFIX}{number:0{NOTE_ID_DIGITS}d}"


def _parser() -> etree.XMLParser:
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        huge_tree=True,
        remove_comments=False,
        remove_pis=False,
    )


def read_musicxml_bytes(path: Path) -> bytes:
    """Return the plain MusicXML payload of ``path``, unwrapping an MXL container if needed."""
    if not path.is_file():
        raise InputError(f"symbolic score not found: {path}")
    if path.suffix.lower() != ".mxl":
        return path.read_bytes()
    try:
        with zipfile.ZipFile(path) as archive:
            container_name = next(
                (n for n in archive.namelist() if n.lower() == "meta-inf/container.xml"), None
            )
            if container_name is None:
                raise InputError(f"{path} is not an MXL container: no META-INF/container.xml")
            container = etree.fromstring(archive.read(container_name), parser=_parser())
            rootfiles = [
                element.get("full-path")
                for element in container.iter()
                if etree.QName(element).localname == "rootfile" and element.get("full-path")
            ]
            if not rootfiles:
                raise InputError(f"{path} has no rootfile in META-INF/container.xml")
            target = rootfiles[0]
            if target not in archive.namelist():
                raise InputError(f"{path} refers to missing {target}")
            return archive.read(target)
    except zipfile.BadZipFile as exc:
        raise InputError(f"{path} is not a readable zip container: {exc}") from exc


def parse_musicxml(path: Path) -> etree._ElementTree:
    """Parse a MusicXML or MXL file into an element tree."""
    payload = read_musicxml_bytes(path)
    try:
        return etree.ElementTree(etree.fromstring(payload, parser=_parser()))
    except etree.XMLSyntaxError as exc:
        raise InputError(f"{path} is not valid XML: {exc}") from exc


def assign_note_ids(tree: etree._ElementTree) -> tuple[str, ...]:
    """Give every ``<note>`` element in document order a stable ``mf-`` id.

    Returns the ids in document order. Existing ids are replaced: the numbering is the contract
    between the score model and the Verovio layout, so it must be reproducible from the source
    bytes alone.
    """
    ids: list[str] = []
    for number, element in enumerate(tree.iter("{*}note"), start=1):
        identifier = note_id(number)
        element.set("id", identifier)
        ids.append(identifier)
    if not ids:
        raise InputError("the symbolic score contains no note elements")
    return tuple(ids)


def write_normalized_musicxml(source: Path, destination: Path) -> tuple[Path, int]:
    """Write an id-annotated copy of ``source`` to ``destination`` and return it with the count."""
    tree = parse_musicxml(source)
    ids = assign_note_ids(tree)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(destination), encoding="utf-8", xml_declaration=True)
    return destination, len(ids)
