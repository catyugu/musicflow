"""Run directories, fingerprints and cached pipeline steps.

Every analysis stage owns one directory inside a run workspace:

    <workspace>/<step>/
        meta.json      schema version, cache key, and the tool versions that produced it
        payload.json   the step's serialised result
        ...            any binary side files the step wrote (page rasters, strip tiles, ...)

A step is skipped when its recorded key equals the key computed from the current inputs. The key
covers input file contents (sha256), the parameters that influence the step, and the versions of
the tools involved, so a re-run after changing any of them recomputes and everything else stays
cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Content hash of a file, streamed so that large audio files stay cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(files: Mapping[str, Path], params: Mapping[str, Any]) -> str:
    """Cache key from the content of ``files`` plus a JSON-serialisable parameter mapping."""
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(sha256_file(files[name]).encode())
        digest.update(b"\0")
    digest.update(json.dumps(params, sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


def hash_payload(payload: Any) -> str:
    """Stable digest of a JSON-serialisable artifact, used to key downstream steps."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically, so an interrupted run never leaves a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Step:
    """One cacheable stage of the pipeline."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def meta_path(self) -> Path:
        return self.path / "meta.json"

    @property
    def payload_path(self) -> Path:
        return self.path / "payload.json"

    def cached(self, key: str) -> dict[str, Any] | None:
        """Return the stored payload when it was produced from the same key."""
        if not (self.meta_path.is_file() and self.payload_path.is_file()):
            return None
        try:
            meta = read_json(self.meta_path)
        except (json.JSONDecodeError, OSError):
            logger.debug("discarding unreadable step metadata: %s", self.meta_path)
            return None
        if not isinstance(meta, dict) or meta.get("schema") != SCHEMA_VERSION:
            return None
        if meta.get("key") != key:
            return None
        return read_json(self.payload_path)

    def store(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.path.mkdir(parents=True, exist_ok=True)
        write_json(self.payload_path, payload)
        write_json(self.meta_path, {"schema": SCHEMA_VERSION, "key": key})
        return payload

    def run(self, key: str, compute: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
        """Return the cached payload or compute, store and return a fresh one."""
        payload = self.cached(key)
        if payload is not None:
            logger.info("step %s: cached", self.path.name)
            return payload
        logger.info("step %s: running", self.path.name)
        self.path.mkdir(parents=True, exist_ok=True)
        return self.store(key, compute(self.path))


class Workspace:
    """A run directory holding one step per pipeline stage."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def step(self, name: str) -> Step:
        return Step(self.root / name)

    def file(self, relative: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
