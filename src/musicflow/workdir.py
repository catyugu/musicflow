"""Content-addressed stage artifacts and atomic JSON reports."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import msgspec

logger = logging.getLogger(__name__)


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def key(*inputs: Any) -> str:
    return hashlib.sha256(msgspec.json.encode(inputs, order="deterministic")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        pending = Path(temporary) / path.name
        pending.write_bytes(msgspec.json.format(msgspec.json.encode(payload), indent=1))
        pending.replace(path)


def read_json(path: Path) -> Any:
    return msgspec.json.decode(path.read_bytes())


class Step:
    def __init__(self, path: Path, digest: str) -> None:
        self.path = path / digest
        self.digest = digest

    def run[T](
        self,
        payload_type: type[T],
        compute: Callable[[Path], T],
        *,
        present: Callable[[T], bool] = lambda _: True,
    ) -> T:
        cache = self.path / "cache.json"
        try:
            digest, payload = msgspec.json.decode(cache.read_bytes(), type=tuple[str, payload_type])
        except (OSError, msgspec.DecodeError, AssertionError):
            pass
        else:
            if digest == self.digest and present(payload):
                logger.info("step %s: cached", self.path.parent.name)
                return payload

        logger.info("step %s: running", self.path.parent.name)
        self.path.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.path) as temporary:
            root = Path(temporary)
            payload = compute(root)
            pending = root / cache.name
            pending.write_bytes(msgspec.json.encode((self.digest, payload)))
            for side in root.rglob("*"):
                if side.is_file() and side != pending:
                    target = self.path / side.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    side.replace(target)
            pending.replace(cache)
        return payload


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root

    def step(self, name: str, digest: str) -> Step:
        return Step(self.root / name, digest)
