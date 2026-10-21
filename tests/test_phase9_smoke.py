import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

PAGES = ["Overview", "Site risk", "Deviations", "Anomalies", "Forensics", "Feedback and triage", "Audit trail"]


def _fresh_modules():
    for m in [m for m in sys.modules if m == "common" or m.startswith("views")]:
        del sys.modules[m]


def test_every_dashboard_page_renders(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from fake_bundle import make_fake_bundle

    make_fake_bundle(tmp_path)
    monkeypatch.setenv("CTMON_ROOT", str(tmp_path))
    _fresh_modules()
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])

    # a monitor decision is written to the audit log and the chain stays valid
    at.sidebar.radio[0].set_value("Feedback and triage").run()
    btn = [b for b in at.button if b.label == "True issue"][0]
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    from ctmon.audit import AuditLog
    log = AuditLog(tmp_path / "data" / "audit" / "audit_log.jsonl")
    events = [e["event"] for e in log.entries()]
    assert events.count("DECISION") == 7 and log.verify().ok

    # the tamper test runs from the UI
    at.sidebar.radio[0].set_value("Audit trail").run()
    [b for b in at.button if b.label == "Run tamper simulation"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "NOT detected" in str(at.dataframe[-1].value.to_dict())
