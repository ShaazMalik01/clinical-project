import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from test_phase10 import synthetic_outputs   # noqa: E402

from ctmon.evaluation.final_eval import (evaluate_all, sens_forensics_tiers, sens_ml_contamination,  # noqa: E402
                                         sens_rule_severity)
from ctmon.evaluation.report import build_markdown, save_report, why_hybrid_text   # noqa: E402


def _full(n_boot=60):
    out = synthetic_outputs(seed=3)
    res = evaluate_all(out, n_boot=n_boot, seed=2)
    t = res["truth"]
    sens = {"ml": sens_ml_contamination(out, t, (0.01, 0.03), (0.05, 0.10, 0.20)),
            "forensics": sens_forensics_tiers(out, t), "rules": sens_rule_severity(out, t),
            "fusion": res["fusion"]}
    return res, sens


def test_report_and_figures_are_written(tmp_path):
    res, sens = _full()
    ms = pd.DataFrame({"f1_mean": [0.7], "f1_sd": [0.02]}, index=pd.Index(["hybrid"], name="system"))
    save_report(res, sens, None, ms, tmp_path, n_sites=8, notes="synthetic test")
    md = (tmp_path / "docs" / "evaluation_report.md").read_text()
    for h in ("## 1. Overall", "## 4. Ablation", "## 5. Does the hybrid win", "## 7. Sensitivity", "# Limitations"):
        assert h in md
    assert (tmp_path / "docs" / "phase10_main.png").stat().st_size > 20_000
    assert (tmp_path / "docs" / "phase10_sensitivity.png").stat().st_size > 20_000
    assert (tmp_path / "docs" / "LIMITATIONS.md").exists()


def test_narrative_does_not_overclaim():
    res, _ = _full(n_boot=120)
    best = res["overall"].drop(index="hybrid (all three)")["f1"].idxmax()
    d = res["boot_diff"].loc[best]
    text = " ".join(why_hybrid_text(res))
    if d["dF1_lo"] <= 0:
        assert "not distinguishable from zero" in text and "clear improvement" not in text
    else:
        assert "clear improvement" in text
    assert "The price" in text and "Recall" in text
