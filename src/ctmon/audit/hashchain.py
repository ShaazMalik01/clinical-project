"""Append-only, hash-chained audit log (tamper-EVIDENT, in the spirit of 21 CFR Part 11 audit trails).

Every entry stores the hash of the previous entry, so changing, deleting, inserting or
re-ordering any entry breaks the chain from that point on:

    entry_n.hash = H( seq, ts, actor, event, entity, entity_id, payload, entry_{n-1}.hash )

Three layers of protection (each closes a gap the previous one leaves open):
  1. the chain itself        -> detects edit / delete / insert / reorder
  2. an anchor (head hash)   -> detects truncation of the tail and a full chain rewrite
  3. optional HMAC key       -> an attacker without the key cannot produce valid hashes at all

Honest limit: this is tamper-EVIDENT, not tamper-PROOF. Someone with write access to the log AND
the anchor AND the key can rewrite history; in production, publish the anchor somewhere the
attacker cannot reach (WORM storage, e-mail, a git commit, a printed report).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

GENESIS = "0" * 64
BODY_FIELDS = ("seq", "ts", "actor", "event", "entity", "entity_id", "payload", "prev_hash")
ENV_KEY = "CTMON_AUDIT_KEY"

# Event vocabulary (documentation; the log accepts any string)
EVENTS = {
    "RUN_START": "a detection run started (data fingerprint recorded)",
    "FLAG_RAISED": "a detector raised an alert (rule, ML, forensic)",
    "SITE_RECOMMENDATION": "a site-level monitoring recommendation was issued",
    "DECISION": "a monitor marked an alert as a true issue or a false alarm",
    "MODEL_UPDATE": "the feedback learner was re-fitted from decisions",
    "CONFIG_BASELINE": "first recorded state of the configuration files",
    "CONFIG_CHANGE": "a configuration file changed (key-level diff recorded)",
    "EXPORT": "a report was downloaded",
    "VERIFY": "the chain was verified",
}


# ================================================================== helpers
def clean(o: Any) -> Any:
    """Make any payload strictly JSON-native so that hash(write) == hash(read-back)."""
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return [clean(v) for v in sorted(o, key=str)]
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "item") and callable(o.item):          # numpy scalars
        try:
            return clean(o.item())
        except Exception:
            pass
    return str(o)


def canonical(body: dict) -> bytes:
    """Deterministic serialisation of the hashed fields."""
    return json.dumps({k: body[k] for k in BODY_FIELDS}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def entry_digest(body: dict, key: bytes | None = None) -> str:
    data = canonical(body)
    if key:
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_dataframe(df) -> str:
    import pandas as pd
    h = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.sha256(h + ",".join(map(str, df.columns)).encode()).hexdigest()


def hash_tables(tables: dict) -> str:
    """One fingerprint for a whole dataset (order-independent across tables)."""
    parts = [f"{k}:{hash_dataframe(v)}" for k, v in sorted(tables.items())]
    return sha256_text("|".join(parts))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def short_payload(payload: dict, maxlen: int = 140) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


class AuditError(RuntimeError):
    """Raised when the log cannot safely be appended to (e.g. corrupted tail)."""


# ============================================================= verification
@dataclass
class Problem:
    seq: int | None
    kind: str
    detail: str


@dataclass
class VerifyReport:
    ok: bool
    n_entries: int
    head_seq: int
    head_hash: str
    problems: list = field(default_factory=list)
    hmac_used: bool = False
    anchor_checked: bool = False
    verified_at: str = ""

    @property
    def first_bad_seq(self) -> int | None:
        seqs = [p.seq for p in self.problems if p.seq is not None]
        return min(seqs) if seqs else None

    @property
    def kinds(self) -> list:
        return sorted({p.kind for p in self.problems})

    def summary(self) -> str:
        if self.ok:
            prot = "HMAC + " if self.hmac_used else ""
            prot += "anchor" if self.anchor_checked else "chain only"
            return f"OK: {self.n_entries} entries verified ({prot}); head {self.head_hash[:12]}…"
        return (f"TAMPERING DETECTED: {len(self.problems)} problem(s) "
                f"({', '.join(self.kinds)}); first at seq {self.first_bad_seq}")

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame([{"seq": p.seq, "problem": p.kind, "detail": p.detail}
                             for p in self.problems], columns=["seq", "problem", "detail"])


# ================================================================== the log
class AuditLog:
    """JSON-lines file; one entry per line; never edited in place, only appended."""

    def __init__(self, path, key: str | bytes | None | bool = None, anchor_path=None,
                 default_actor: str = "system", auto_anchor: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.anchor_path = Path(anchor_path) if anchor_path else self.path.with_suffix(".anchor.json")
        if key is False:
            self.key = None                                  # explicitly unkeyed (ignore env)
        elif key is None:
            env = os.environ.get(ENV_KEY)
            self.key = env.encode() if env else None
        else:
            self.key = key.encode() if isinstance(key, str) else bytes(key)
        self.default_actor = default_actor
        self.auto_anchor = auto_anchor
        self._lock = threading.RLock()

    # ------------------------------------------------------------ reading
    def _lines(self) -> list:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:
            return [ln for ln in f.read().split(b"\n") if ln.strip()]

    def _last_line(self) -> bytes | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        with open(self.path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos, data = f.tell(), b""
            while pos > 0:
                step = min(4096, pos)
                pos -= step
                f.seek(pos)
                data = f.read(step) + data
                stripped = data.rstrip(b"\n")
                if b"\n" in stripped:
                    return stripped.rsplit(b"\n", 1)[1]
            return data.rstrip(b"\n") or None

    def _tail(self) -> tuple:
        last = self._last_line()
        if last is None:
            return 0, GENESIS
        try:
            e = json.loads(last.decode("utf-8"))
            return int(e["seq"]), str(e["hash"])
        except Exception as ex:                              # noqa: BLE001
            raise AuditError("The last audit entry is unreadable; run verify() before appending.") from ex

    def __len__(self) -> int:
        return len(self._lines())

    def entries(self) -> Iterator[dict]:
        for raw in self._lines():
            try:
                yield json.loads(raw.decode("utf-8"))
            except Exception:                                # noqa: BLE001
                yield {"seq": None, "event": "UNREADABLE", "entity": "", "entity_id": "",
                       "actor": "", "ts": "", "payload": {"raw": raw[:80].decode("utf-8", "replace")},
                       "hash": "", "prev_hash": ""}

    def to_frame(self, limit: int | None = None):
        import pandas as pd
        rows = []
        for e in self.entries():
            rows.append({"seq": e.get("seq"), "ts": e.get("ts"), "actor": e.get("actor"),
                         "event": e.get("event"), "entity": e.get("entity"),
                         "entity_id": e.get("entity_id"),
                         "payload": short_payload(e.get("payload", {})),
                         "hash": str(e.get("hash", ""))[:16], "prev_hash": str(e.get("prev_hash", ""))[:16]})
        df = pd.DataFrame(rows, columns=["seq", "ts", "actor", "event", "entity", "entity_id",
                                         "payload", "hash", "prev_hash"])
        return df.tail(limit) if limit else df

    # ------------------------------------------------------------ writing
    def append_many(self, records: Iterable[dict], checkpoint: bool | None = None) -> list:
        """Append several entries in one atomic-ish write. Each record: event (required),
        entity, entity_id, payload, actor, ts (optional)."""
        with self._lock:
            seq, prev = self._tail()
            out, lines = [], []
            for r in records:
                seq += 1
                body = {"seq": seq, "ts": r.get("ts") or now_iso(),
                        "actor": str(r.get("actor") or self.default_actor),
                        "event": str(r["event"]), "entity": str(r.get("entity", "")),
                        "entity_id": str(r.get("entity_id", "")),
                        "payload": clean(r.get("payload") or {}), "prev_hash": prev}
                body["hash"] = entry_digest(body, self.key)
                prev = body["hash"]
                out.append(body)
                lines.append(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False))
            if not out:
                return []
            with open(self.path, "ab") as f:
                f.write(("\n".join(lines) + "\n").encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            if self.auto_anchor if checkpoint is None else checkpoint:
                self.checkpoint()
            return out

    def append(self, event: str, entity: str = "", entity_id: str = "", payload: dict | None = None,
               actor: str | None = None, ts: str | None = None, checkpoint: bool | None = None) -> dict:
        return self.append_many([{"event": event, "entity": entity, "entity_id": entity_id,
                                  "payload": payload, "actor": actor, "ts": ts}], checkpoint)[0]

    # ------------------------------------------------------------ anchor
    def head(self) -> dict:
        seq, h = self._tail()
        return {"seq": seq, "hash": h}

    def checkpoint(self) -> dict:
        """Write the current head hash to the anchor file (publish it somewhere safe in production)."""
        a = {**self.head(), "ts": now_iso()}
        tmp = self.anchor_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(a, indent=2), encoding="utf-8")
        os.replace(tmp, self.anchor_path)
        return a

    def read_anchor(self) -> dict | None:
        try:
            return json.loads(self.anchor_path.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            return None

    # ------------------------------------------------------------ verify
    def verify(self, check_anchor: bool = True, expected_head: dict | None = None) -> VerifyReport:
        problems: list = []
        prev_hash: str | None = GENESIS
        expected_seq, last_ts = 1, None
        stored: dict = {}
        n, head_seq, head_hash = 0, 0, GENESIS

        for raw in self._lines():
            n += 1
            try:
                e = json.loads(raw.decode("utf-8"))
                for k in (*BODY_FIELDS, "hash"):
                    if k not in e:
                        raise KeyError(k)
                seq = int(e["seq"])
            except Exception as ex:                          # noqa: BLE001
                problems.append(Problem(expected_seq, "bad_json",
                                        f"line {n} is not a valid audit entry ({type(ex).__name__})"))
                expected_seq += 1
                prev_hash = None                             # chain cannot be followed through garbage
                continue

            if seq != expected_seq:
                problems.append(Problem(seq, "seq_gap", f"expected seq {expected_seq}, found {seq} "
                                                         f"(entry deleted, inserted or reordered)"))
            expected_seq = seq + 1

            if prev_hash is not None and e["prev_hash"] != prev_hash:
                problems.append(Problem(seq, "chain_break",
                                        "prev_hash does not match the previous entry's hash"))
            try:
                recomputed = entry_digest(e, self.key)
            except Exception:                                # noqa: BLE001
                recomputed = ""
            if not hmac.compare_digest(str(e["hash"]), recomputed):
                problems.append(Problem(seq, "hash_mismatch",
                                        "stored hash differs from the recomputed hash (content altered)"))
            ts = str(e["ts"])
            if last_ts is not None and ts < last_ts:
                problems.append(Problem(seq, "timestamp_regress",
                                        f"timestamp {ts} is earlier than the previous entry ({last_ts})"))
            last_ts = ts
            stored[seq] = str(e["hash"])
            prev_hash = str(e["hash"])
            head_seq, head_hash = seq, str(e["hash"])

        anchors = []
        if expected_head:
            anchors.append(("expected_head", expected_head))
        if check_anchor:
            a = self.read_anchor()
            if a:
                anchors.append(("anchor", a))
        for label, a in anchors:
            aseq, ahash = int(a.get("seq", 0)), str(a.get("hash", ""))
            if aseq == 0:
                continue
            if aseq not in stored:
                problems.append(Problem(aseq, "truncated",
                                        f"{label} points at seq {aseq} but the log ends at seq {head_seq} "
                                        f"(entries removed)"))
            elif stored[aseq] != ahash:
                problems.append(Problem(aseq, "anchor_mismatch",
                                        f"entry {aseq} no longer matches the {label} (history rewritten)"))

        return VerifyReport(ok=not problems, n_entries=n, head_seq=head_seq, head_hash=head_hash,
                            problems=problems, hmac_used=bool(self.key),
                            anchor_checked=bool(anchors), verified_at=now_iso())

    # ------------------------------------------------------------ archive
    def archive(self, dest_dir) -> Path | None:
        """Move the log (and anchor) away: used only by an explicit 'reset' so history is never deleted."""
        if not self.path.exists():
            return None
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = dest / f"{self.path.stem}.{stamp}{self.path.suffix}"
        shutil.move(str(self.path), target)
        if self.anchor_path.exists():
            shutil.move(str(self.anchor_path), dest / f"{self.path.stem}.{stamp}.anchor.json")
        return target

    # ------------------------------------------------- typed convenience API
    def log_run_start(self, run_id: str, data_sha256: str, sizes: dict, actor: str = "pipeline") -> dict:
        return self.append("RUN_START", "run", run_id,
                           {"run_id": run_id, "data_sha256": data_sha256, "table_rows": sizes}, actor)

    def log_flags(self, alerts, run_id: str, actor: str = "pipeline") -> int:
        recs = []
        for r in alerts.itertuples(index=False):
            recs.append({"event": "FLAG_RAISED", "entity": "alert", "entity_id": r.alert_id,
                         "actor": actor,
                         "payload": {"run_id": run_id, "source": r.source, "level": r.level,
                                     "severity": r.severity, "priority": round(float(r.priority), 4),
                                     "site_id": r.site_id, "subject_id": r.subject_id,
                                     "visit_num": r.visit_num, "headline": r.headline,
                                     "card_sha256": sha256_text(str(r.card))}})
        return len(self.append_many(recs))

    def log_site_recommendations(self, site_risk, run_id: str, actor: str = "pipeline") -> int:
        recs = [{"event": "SITE_RECOMMENDATION", "entity": "site", "entity_id": r.site_id, "actor": actor,
                 "payload": {"run_id": run_id, "tier": r.tier, "risk_index": round(float(r.risk_index), 4),
                             "ci": [round(float(r.ci_lo), 4), round(float(r.ci_hi), 4)],
                             "breached_kris": r.breached_kris}}
                for r in site_risk.itertuples(index=False)]
        return len(self.append_many(recs))

    def log_decision(self, alert_id: str, is_true_issue: bool, reviewer: str, comment: str = "",
                     source: str = "") -> dict:
        return self.append("DECISION", "alert", alert_id,
                           {"is_true_issue": bool(is_true_issue), "comment": comment, "source": source},
                           actor=reviewer)

    def log_model_update(self, payload: dict, actor: str = "feedback-learner") -> dict:
        return self.append("MODEL_UPDATE", "model", "feedback-learner", payload, actor)

    def log_export(self, name: str, n_rows: int, filters: dict, actor: str) -> dict:
        return self.append("EXPORT", "report", name, {"rows": int(n_rows), "filters": filters}, actor)

    def log_verification(self, report: VerifyReport, actor: str) -> dict:
        return self.append("VERIFY", "audit-log", "chain",
                           {"ok": report.ok, "entries": report.n_entries,
                            "problems": report.kinds, "head": report.head_hash[:16]}, actor)
