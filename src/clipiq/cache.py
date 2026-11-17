"""Versioned, content-addressed JSON cache. No pickle or notebook state."""
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
SCHEMA = 1


def encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__clipiq_type__": "array", "dtype": str(value.dtype), "data": value.tolist()}
    if isinstance(value, pd.DataFrame):
        return {"__clipiq_type__": "frame", "columns": list(value.columns),
                "data": encode(value.to_dict(orient="records"))}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if value.get("__clipiq_type__") == "array":
            return np.asarray(value["data"], dtype=value["dtype"])
        if value.get("__clipiq_type__") == "frame":
            return pd.DataFrame(decode(value["data"]), columns=value["columns"])
        return {k: decode(v) for k, v in value.items()}
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def fingerprint(path: Path) -> dict:
    """Stream the full file; same-name replacements cannot reuse old analysis."""
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "size": path.stat().st_size}


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".clipiq-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(encode(value), handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class StageCache:
    def __init__(self, directory: Path, force: bool = False):
        self.directory = Path(directory)
        self.force = force
        self.events: list[dict] = []

    def get(self, stage: str, inputs: dict, compute: Callable[[], Any],
            validate: Callable[[Any], None] | None = None) -> Any:
        key = digest({"schema": SCHEMA, "stage": stage, "inputs": inputs})
        path = self.directory / stage / f"{key}.json"
        status = "forced" if self.force else "miss"
        if path.exists() and not self.force:
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                if envelope["key"] != key or digest(envelope["payload"]) != envelope["checksum"]:
                    raise ValueError("cache checksum/key mismatch")
                value = decode(envelope["payload"])
                if validate:
                    validate(value)
                self.events.append({"stage": stage, "key": key, "status": "hit"})
                LOG.info("%s: cache hit", stage)
                return value
            except (ValueError, KeyError, TypeError, OSError):
                status = "corrupt_recomputed"
        LOG.info("%s: %s", stage, status)
        value = compute()
        if validate:
            validate(value)
        payload = encode(value)
        atomic_json(path, {"key": key, "checksum": digest(payload), "payload": payload})
        self.events.append({"stage": stage, "key": key, "status": status})
        return value
