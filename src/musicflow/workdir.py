"""Run directories, content digests and cached pipeline steps.

Each expensive stage owns one directory: ``meta.json`` holds the schema version and cache key,
``payload.json`` the serialised result, next to any binary side file the step wrote. A step is
skipped when its recorded key equals the key of the current inputs — the content digests of the
files it read, the parameters that influence it, and the versions of the tools involved — and its
payload is rejected too when the binary files it names are gone, which is what ``validate`` answers.
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


def digest_file(path: Path) -> str:
    """Content hash of a file, streamed so that large audio files stay cheap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(digests: Mapping[str, str], params: Mapping[str, Any]) -> str:
    """Cache key from the digests of the files a step read plus its parameters."""
    digest = hashlib.sha256()
    for name in sorted(digests):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(digests[name].encode())
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
        self.meta_path = path / "meta.json"
        self.payload_path = path / "payload.json"

    def run(
        self,
        key: str,
        compute: Callable[[Path], dict[str, Any]],
        *,
        validate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        """Return the cached payload or compute, store and return a fresh one.

        ``validate`` sees the stored payload and answers whether the side files it names still
        exist; a payload whose files are gone is recomputed instead of being reused.
        """
        payload: dict[str, Any] | None = None
        if self.meta_path.is_file() and self.payload_path.is_file():
            try:
                meta = read_json(self.meta_path)
            except (json.JSONDecodeError, OSError):
                logger.debug("discarding unreadable step metadata: %s", self.meta_path)
                meta = None
            if (
                isinstance(meta, dict)
                and meta.get("schema") == SCHEMA_VERSION
                and meta.get("key") == key
            ):
                payload = read_json(self.payload_path)
        if payload is not None and (validate is None or validate(payload)):
            logger.info("step %s: cached", self.path.name)
            return payload
        if payload is None:
            logger.info("step %s: running", self.path.name)
        else:
            logger.info("step %s: side files are gone, running", self.path.name)
        self.path.mkdir(parents=True, exist_ok=True)
        payload = compute(self.path)
        write_json(self.payload_path, payload)
        write_json(self.meta_path, {"schema": SCHEMA_VERSION, "key": key})
        return payload


class Workspace:
    """A run directory holding one step per pipeline stage."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def step(self, name: str) -> Step:
        return Step(self.root / name)
