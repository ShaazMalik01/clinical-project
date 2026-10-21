"""Config-change tracking: every edit to protocol / rules / settings is audit-logged with a
key-level diff ("risk.prior.outlier_prob: 0.15 -> 0.30"), not just 'file changed'."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import yaml

from .hashchain import AuditLog


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(d: Any, prefix: str = "") -> dict:
    """{'a': {'b': [1, 2]}} -> {'a.b[0]': 1, 'a.b[1]': 2}"""
    out: dict = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def diff_configs(old: Any, new: Any, limit: int = 60) -> list:
    a, b = flatten(old), flatten(new)
    rows = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            rows.append({"key": k, "change": "added", "old": None, "new": b[k]})
        elif k not in b:
            rows.append({"key": k, "change": "removed", "old": a[k], "new": None})
        elif a[k] != b[k]:
            rows.append({"key": k, "change": "modified", "old": a[k], "new": b[k]})
    return rows[:limit]


class ConfigWatcher:
    """Compare config files with the last state recorded in the audit log; log the difference."""

    def __init__(self, log: AuditLog, files: dict, snapshot_dir):
        self.log = log
        self.files = {k: Path(v) for k, v in files.items()}
        self.snap = Path(snapshot_dir)
        self.snap.mkdir(parents=True, exist_ok=True)

    def _snapshot(self, name: str, path: Path, sha: str) -> Path:
        target = self.snap / f"{name}.{sha[:16]}{path.suffix}"
        if not target.exists():
            shutil.copy2(path, target)
        return target

    def _last_state(self) -> dict | None:
        last = None
        for e in self.log.entries():
            if e.get("event") in ("CONFIG_BASELINE", "CONFIG_CHANGE"):
                last = e["payload"].get("files")
        return last

    def current(self) -> dict:
        return {n: file_sha256(p) for n, p in self.files.items() if p.exists()}

    def check(self, actor: str = "system") -> dict | None:
        """Log CONFIG_BASELINE (first time) or CONFIG_CHANGE (if something changed).
        Returns the logged payload, or None if nothing changed."""
        now = self.current()
        for n, sha in now.items():
            self._snapshot(n, self.files[n], sha)
        prev = self._last_state()
        if prev is None:
            payload = {"files": now}
            self.log.append("CONFIG_BASELINE", "config", "all", payload, actor)
            return payload
        changed = [n for n in now if prev.get(n) != now[n]]
        removed = [n for n in prev if n not in now]
        if not changed and not removed:
            return None
        changes = []
        for n in changed:
            old_sha = prev.get(n)
            old_path = next(iter(self.snap.glob(f"{n}.{old_sha[:16]}.*")), None) if old_sha else None
            try:
                if old_path is None:
                    raise FileNotFoundError
                diff = diff_configs(yaml.safe_load(old_path.read_text(encoding="utf-8")),
                                    yaml.safe_load(self.files[n].read_text(encoding="utf-8")))
            except Exception:                                # noqa: BLE001
                diff = [{"key": "(file)", "change": "modified", "old": old_sha, "new": now[n]}]
            changes.append({"file": n, "old_sha256": old_sha, "new_sha256": now[n], "diff": diff})
        payload = {"files": now, "changed": changes, "removed": removed}
        self.log.append("CONFIG_CHANGE", "config", ",".join(changed + removed), payload, actor)
        return payload
