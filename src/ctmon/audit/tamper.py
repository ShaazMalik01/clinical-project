"""Tamper simulation: attack COPIES of the audit log and check that verify() notices.

The real log is never modified. Each scenario states what the attacker did, which protection
layer was active, and whether the attack was detected. Two scenarios are deliberately shown
WITHOUT the layer that stops them, to make the limits of a bare hash chain visible.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pandas as pd

from .hashchain import GENESIS, AuditLog, entry_digest, now_iso

MIN_ENTRIES = 12


def _dump(e: dict) -> str:
    return json.dumps(e, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load(path: Path) -> list:
    return [json.loads(ln) for ln in path.read_bytes().decode("utf-8").split("\n") if ln.strip()]


def _save(path: Path, entries: list, raw_lines: dict | None = None) -> None:
    lines = [(raw_lines or {}).get(i, _dump(e)) for i, e in enumerate(entries)]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _rechain(entries: list, start: int) -> None:
    """What an attacker who knows the algorithm but NOT the HMAC key can do: recompute hashes."""
    prev = entries[start - 1]["hash"] if start > 0 else GENESIS
    for i in range(start, len(entries)):
        e = entries[i]
        e["prev_hash"] = prev
        e["hash"] = entry_digest(e, None)
        prev = e["hash"]


def _pick(entries: list, event: str, lo: int, hi: int) -> int | None:
    for i in range(lo, hi):
        if entries[i]["event"] == event:
            return i
    return None


def _target(entries: list) -> tuple:
    """A meaningful entry to attack, preferably a DECISION (flip a monitor's verdict) or a
    FLAG_RAISED (downgrade a critical flag), somewhere in the middle of the log."""
    n = len(entries)
    lo, hi = n // 4, (3 * n) // 4
    i = _pick(entries, "DECISION", lo, hi)
    if i is not None:
        return i, "flip a monitor's decision (true issue -> false alarm)"
    i = _pick(entries, "FLAG_RAISED", lo, hi)
    if i is not None:
        return i, "downgrade a raised flag's severity to 'minor'"
    return n // 2, "alter an entry's content"


def _apply_edit(e: dict) -> None:
    p = e["payload"]
    if e["event"] == "DECISION" and "is_true_issue" in p:
        p["is_true_issue"] = not p["is_true_issue"]
    elif e["event"] == "FLAG_RAISED":
        p["severity"] = "minor"
    else:
        p["edited"] = True


def _keyed_clone(entries: list, path: Path, key: bytes) -> AuditLog:
    """Rebuild the same history under an HMAC key (to demonstrate the key protection)."""
    keyed = AuditLog(path, key=key, auto_anchor=False)
    keyed.append_many([{"event": e["event"], "entity": e["entity"], "entity_id": e["entity_id"],
                        "payload": e["payload"], "actor": e["actor"], "ts": e["ts"]} for e in entries])
    return keyed


def run_tamper_suite(log: AuditLog, workdir) -> pd.DataFrame:
    base = _load(log.path)
    if len(base) < MIN_ENTRIES:
        raise ValueError(f"Need at least {MIN_ENTRIES} audit entries for the tamper test, have {len(base)}.")
    work = Path(workdir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    head = log.head()
    anchor = {"seq": head["seq"], "hash": head["hash"], "ts": now_iso()}
    n = len(base)
    k, what = _target(base)
    rows: list = []

    def scenario(name: str, attack: str, mutate, protection: str, expected: bool,
                 *, key=False, use_anchor: bool = True, entries_src=None, keyed_clone=None):
        entries = copy.deepcopy(entries_src or base)
        raw: dict = {}
        mutate(entries, raw)
        p = work / f"{name}-{len(rows)}.jsonl"
        _save(p, entries, raw)
        ap = p.with_suffix(".anchor.json")
        if use_anchor:
            ap.write_text(json.dumps(anchor), encoding="utf-8")
        v = AuditLog(p, key=key, anchor_path=ap, auto_anchor=False).verify(check_anchor=use_anchor)
        rows.append({"scenario": name, "attack": attack, "protection": protection,
                     "detected": not v.ok, "expected": expected,
                     "as_designed": (not v.ok) == expected,
                     "problems": ", ".join(v.kinds) or "none",
                     "first_bad_seq": v.first_bad_seq})

    # ------------------------------------------------- caught by the chain itself
    def edit(es, raw):
        _apply_edit(es[k])
    scenario("edit_entry", f"Edit entry #{k + 1}: {what}", edit, "hash chain", True, use_anchor=False)

    def delete(es, raw):
        del es[k]
    scenario("delete_entry", f"Delete entry #{k + 1}", delete, "hash chain", True, use_anchor=False)

    def insert(es, raw):
        forged = copy.deepcopy(es[k])
        forged["payload"] = {"forged": True}
        es.insert(k + 1, forged)
    scenario("insert_forged", f"Insert a forged entry after #{k + 1}", insert, "hash chain", True,
             use_anchor=False)

    def reorder(es, raw):
        es[k], es[k + 1] = es[k + 1], es[k]
    scenario("reorder_entries", f"Swap entries #{k + 1} and #{k + 2}", reorder, "hash chain", True,
             use_anchor=False)

    def corrupt(es, raw):
        raw[k] = "{this is not json"
    scenario("corrupt_line", f"Overwrite entry #{k + 1} with garbage", corrupt, "hash chain", True,
             use_anchor=False)

    def edit_rehash(es, raw):
        _apply_edit(es[k])
        es[k]["hash"] = entry_digest(es[k], None)           # fix only the edited entry's own hash
    scenario("edit_and_rehash_one_entry",
             f"Edit entry #{k + 1} AND recompute its hash (so it looks valid)", edit_rehash,
             "hash chain (next entry's link breaks)", True, use_anchor=False)

    # ------------------------------------------- attacks that need more than the chain
    def rewrite(es, raw):
        _apply_edit(es[k])
        _rechain(es, k)                                     # recompute every later hash
    scenario("rewrite_whole_chain", f"Edit entry #{k + 1} and recompute ALL later hashes", rewrite,
             "hash chain only (no anchor, no key)", False, use_anchor=False)
    scenario("rewrite_whole_chain", f"Edit entry #{k + 1} and recompute ALL later hashes", rewrite,
             "chain + anchor", True, use_anchor=True)

    def truncate(es, raw):
        del es[-5:]
    scenario("truncate_tail", "Delete the last 5 entries (hide recent activity)", truncate,
             "hash chain only (no anchor, no key)", False, use_anchor=False)
    scenario("truncate_tail", "Delete the last 5 entries (hide recent activity)", truncate,
             "chain + anchor", True, use_anchor=True)

    # ------------------------------------------------ HMAC key (independent demo)
    key = b"demo-secret-key-not-known-to-attacker"
    keyed = _keyed_clone(base, work / "keyed-source.jsonl", key)
    keyed_entries = _load(keyed.path)
    scenario("rewrite_whole_chain",
             f"Edit entry #{k + 1}, recompute ALL later hashes (attacker lacks the secret key)",
             rewrite, "chain + HMAC key", True, key=key, use_anchor=False, entries_src=keyed_entries)

    df = pd.DataFrame(rows)
    df.attrs["n_entries"] = n
    return df
