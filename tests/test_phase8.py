import json

import numpy as np
import pandas as pd
import pytest

from ctmon.audit import (AuditError, AuditLog, ConfigWatcher, GENESIS, diff_configs, hash_tables,
                         run_tamper_suite)


@pytest.fixture
def log(tmp_path):
    lg = AuditLog(tmp_path / "audit.jsonl")
    for i in range(30):
        if i % 5 == 0:
            lg.log_decision(f"A{i}", bool(i % 2), "rev1", "c", "ML")
        else:
            lg.append("FLAG_RAISED", "alert", f"A{i}", {"severity": "critical", "score": np.float64(0.5)})
    return lg


def _rewrite(path, fn):
    es = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    fn(es)
    path.write_text("\n".join(json.dumps(e, sort_keys=True, separators=(",", ":")) for e in es) + "\n")


def test_chain_links_and_genesis(log):
    es = list(log.entries())
    assert es[0]["prev_hash"] == GENESIS
    assert all(es[i]["prev_hash"] == es[i - 1]["hash"] for i in range(1, len(es)))
    assert [e["seq"] for e in es] == list(range(1, 31))
    assert log.verify().ok


def test_numpy_payloads_survive_roundtrip(log):
    assert log.verify().ok                      # numpy float64 was written, hash recomputed after read-back


def test_edit_is_detected_at_the_edited_entry(log):
    _rewrite(log.path, lambda es: es[9]["payload"].update(severity="minor"))
    r = log.verify(check_anchor=False)
    assert not r.ok and r.first_bad_seq == 10 and "hash_mismatch" in r.kinds


def test_delete_and_reorder_detected(log, tmp_path):
    _rewrite(log.path, lambda es: es.pop(5))
    assert {"seq_gap", "chain_break"} <= set(log.verify(check_anchor=False).kinds)


def test_truncation_needs_the_anchor(log):
    _rewrite(log.path, lambda es: es.__delitem__(slice(-4, None)))
    assert log.verify(check_anchor=False).ok            # the chain alone cannot see this
    r = log.verify(check_anchor=True)
    assert not r.ok and "truncated" in r.kinds


def test_hmac_blocks_a_full_rewrite(tmp_path):
    lg = AuditLog(tmp_path / "k.jsonl", key="secret", auto_anchor=False)
    for i in range(10):
        lg.append("FLAG_RAISED", "alert", f"A{i}", {"v": i})
    assert lg.verify().ok and lg.verify().hmac_used
    from ctmon.audit.tamper import _rechain

    def attack(es):
        es[3]["payload"]["v"] = 999
        _rechain(es, 3)                                 # attacker recomputes with plain SHA-256
    _rewrite(lg.path, attack)
    assert "hash_mismatch" in lg.verify().kinds


def test_append_continues_chain_across_instances(tmp_path):
    p = tmp_path / "a.jsonl"
    AuditLog(p).append("RUN_START", payload={"x": 1})
    lg2 = AuditLog(p)
    lg2.append("FLAG_RAISED", "alert", "A1", {})
    assert lg2.verify().ok and len(lg2) == 2


def test_refuses_to_append_after_tail_corruption(log):
    with open(log.path, "ab") as f:
        f.write(b"garbage-line\n")
    with pytest.raises(AuditError):
        log.append("EXPORT")


def test_full_tamper_suite_behaves_as_designed(log, tmp_path):
    before = log.path.read_bytes()
    df = run_tamper_suite(log, tmp_path / "t")
    assert df["as_designed"].all()
    assert df[df.expected]["detected"].all() and not df[~df.expected]["detected"].any()
    assert log.path.read_bytes() == before and log.verify().ok      # original never touched


def test_config_watcher_logs_key_level_diff(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("risk:\n  prior:\n    outlier_prob: 0.15\nseed: 42\n")
    lg = AuditLog(tmp_path / "audit.jsonl")
    w = ConfigWatcher(lg, {"settings": cfg}, tmp_path / "snap")
    assert w.check("alice")["files"]["settings"]
    assert w.check("alice") is None                      # nothing changed -> nothing logged
    cfg.write_text("risk:\n  prior:\n    outlier_prob: 0.30\nseed: 42\n")
    payload = w.check("alice")
    d = payload["changed"][0]["diff"][0]
    assert d["key"] == "risk.prior.outlier_prob" and d["old"] == 0.15 and d["new"] == 0.3
    assert [e["event"] for e in lg.entries()] == ["CONFIG_BASELINE", "CONFIG_CHANGE"] and lg.verify().ok


def test_diff_configs_add_remove_modify():
    rows = diff_configs({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4})
    assert {(r["key"], r["change"]) for r in rows} == {("b", "modified"), ("c", "added")}


def test_hash_tables_is_sensitive_to_data():
    a = {"t": pd.DataFrame({"x": [1, 2, 3]})}
    b = {"t": pd.DataFrame({"x": [1, 2, 4]})}
    assert hash_tables(a) == hash_tables({"t": pd.DataFrame({"x": [1, 2, 3]})}) != hash_tables(b)
