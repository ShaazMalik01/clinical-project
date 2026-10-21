"""Figures and the markdown evaluation report. The narrative is generated FROM the numbers, so it only
claims what the numbers support (for example, it says so when the hybrid's F1 gain is not significant)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .final_eval import (CATEGORIES, CATEGORY_NAME, MAIN_SYSTEMS, limitations_markdown, md_table, roc_points)

HYB = "hybrid (all three)"
COLORS = {"rules-only": "#0F6E7A", "ML-only": "#5B6CC4", "forensics-only": "#8E4B9B", HYB: "#B42318"}


# ============================================================ narrative
def why_hybrid_text(res: dict) -> list:
    o = res["overall"]
    h = o.loc[HYB]
    singles = o.drop(index=HYB)
    best_f1, best_rec = singles["f1"].idxmax(), singles["recall"].idxmax()
    out = [f"**Recall.** The hybrid finds {h['recall']:.0%} of the anomalous subjects; the best single detector "
           f"({best_rec}) finds {singles.loc[best_rec, 'recall']:.0%}. A union can only add recall, so this is expected, "
           f"not a surprise."]
    if "boot_diff" in res:
        d = res["boot_diff"].loc[best_f1]
        verdict = ("a statistically clear improvement (95% interval above zero)" if d["dF1_lo"] > 0 else
                   "not distinguishable from zero at 95% (the interval includes zero), so the data do not prove "
                   "the hybrid has a better F1 than this detector")
        out.append(f"**F1 against the strongest single detector ({best_f1}).** Difference {d['dF1']:+.3f} "
                   f"(95% interval {d['dF1_lo']:+.3f} to {d['dF1_hi']:+.3f}); this is {verdict}.")
        others = res["boot_diff"].drop(index=best_f1)
        for name, r in others.iterrows():
            out.append(f"Versus {name}: F1 difference {r['dF1']:+.3f} (95% interval {r['dF1_lo']:+.3f} to "
                       f"{r['dF1_hi']:+.3f}).")
    drop = singles.loc[best_f1, "precision"] - h["precision"]
    out.append(f"**The price.** Precision goes from {singles.loc[best_f1, 'precision']:.2f} ({best_f1}) to "
               f"{h['precision']:.2f} for the hybrid ({-drop:+.2f}), because every detector's false alarms are "
               f"added together. Whether that trade is worth it depends on how costly a missed anomaly is "
               f"compared with a wasted review.")
    by_pat, _, _ = res["coverage"]
    single_only = by_pat[~by_pat.index.str.contains(r"\+|missed", regex=True)]["subjects"].sum()
    missed = int(by_pat["subjects"].get("missed by all", 0))
    tot = int(by_pat["subjects"].sum())
    out.append(f"**Why combining helps.** {single_only} of {tot} anomalous subjects ({single_only / tot:.0%}) are "
               f"caught by exactly one detector, so removing any detector loses real findings; {missed} "
               f"({missed / tot:.0%}) are missed by all three.")
    cat = res["category"]
    for c in CATEGORIES:
        d = cat[cat["category"] == c]
        if d.empty:
            continue
        r = d.set_index("system")["recall"]
        singles_r = r.drop(index=HYB)
        out.append(f"- {CATEGORY_NAME[c]} (n={int(d['n_pos'].iloc[0])}): best single is {singles_r.idxmax()} at "
                   f"{singles_r.max():.0%}; hybrid {r[HYB]:.0%}.")
    ab = res["ablation"]
    best_combo = ab["f1"].idxmax()
    full = "+".join(["rules", "ML", "forensics"])
    if best_combo != full:
        missing = [n for n in ("rules", "ML", "forensics") if n not in best_combo.split("+")]
        out.append(f"**Honest caveat.** The F1-best combination is {best_combo} (F1 {ab.loc[best_combo, 'f1']:.3f}), "
                   f"higher than the full hybrid ({h['f1']:.3f}). Adding {', '.join(missing)} changes recall by "
                   f"{h['recall'] - ab.loc[best_combo, 'recall']:+.3f} and precision by "
                   f"{h['precision'] - ab.loc[best_combo, 'precision']:+.3f} at the configured thresholds. So the "
                   f"full hybrid is the recall-best choice, not the F1-best one; the right choice depends on the "
                   f"relative cost of a missed anomaly and a wasted review, and on the review budget given to "
                   f"{', '.join(missing)} (see the sensitivity section).")
    loo = res["loo"]
    top = loo["recall_gain"].idxmax()
    out.append(f"**Most valuable component by recall:** {top} (removing it loses {loo.loc[top, 'recall_gain']:.1%} "
               f"recall).")
    return out


# ============================================================ figures
def make_main_figure(res: dict, path) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    o = res["overall"]
    names = list(o.index)
    if "boot_ci" in res:
        ci = res["boot_ci"].loc[names]
        ax[0, 0].bar(names, ci["f1"], color=[COLORS[n] for n in names],
                     yerr=[ci["f1"] - ci["f1_lo"], ci["f1_hi"] - ci["f1"]], capsize=5)
    else:
        ax[0, 0].bar(names, o["f1"], color=[COLORS[n] for n in names])
    ax[0, 0].set(title="F1 for finding anomalous subjects (95% bootstrap interval)", ylim=(0, 1.05))
    ax[0, 0].tick_params(axis="x", rotation=20)

    for n, (fpr, tpr) in roc_points(res["truth"], res["comp"]).items():
        ax[0, 1].plot(fpr, tpr, color=COLORS[n], label=f"{n} (AUC {o.loc[n, 'auc']:.2f})")
    ax[0, 1].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0, 1].set(title="ROC curves, any anomaly", xlabel="false positive rate", ylabel="true positive rate")
    ax[0, 1].legend(loc="lower right")

    cat = res["category"].pivot(index="system", columns="category", values="recall").reindex(index=names)
    cat = cat[[c for c in CATEGORIES if c in cat.columns]]
    im = ax[0, 2].imshow(cat.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax[0, 2].set_xticks(range(cat.shape[1]), [f"{c}\n{CATEGORY_NAME[c]}" for c in cat.columns], fontsize=8)
    ax[0, 2].set_yticks(range(len(cat)), cat.index)
    for i in range(cat.shape[0]):
        for j in range(cat.shape[1]):
            ax[0, 2].text(j, i, f"{cat.iat[i, j]:.2f}", ha="center", va="center",
                          color="white" if cat.iat[i, j] > 0.6 else "black")
    ax[0, 2].set_title("Recall by anomaly category")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.04)

    ab = res["ablation"].sort_values("f1")
    ax[1, 0].barh(ab.index, ab["f1"], color=["#B42318" if k == 3 else "#9AA8B1" for k in ab["k"]])
    for y, (f, r, p) in enumerate(zip(ab["f1"], ab["recall"], ab["precision"])):
        ax[1, 0].text(f + 0.01, y, f"R {r:.2f}  P {p:.2f}", va="center", fontsize=8)
    ax[1, 0].set(title="Ablation: F1 for every combination of detectors", xlim=(0, 1.1))

    by_pat, _, _ = res["coverage"]
    bp = by_pat.sort_values("subjects")
    ax[1, 1].barh(bp.index, bp["subjects"], color="#0F6E7A")
    ax[1, 1].set(title="Which detectors caught each anomalous subject", xlabel="subjects")

    for n in names:
        d = res["budget"][res["budget"]["system"] == n]
        ax[1, 2].plot(d["budget"] * 100, d["recall_at_k"], marker="o", color=COLORS[n], label=n)
    ax[1, 2].set(title="Equal workload: recall when reviewing the top b% of subjects",
                 xlabel="share of subjects reviewed (%)", ylabel="recall")
    ax[1, 2].legend()
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_sensitivity_figure(sens: dict, path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    m = sens["ml"]
    m = m[m["system"] == HYB].pivot(index="visit_q", columns="subject_q", values="f1")
    im = ax[0, 0].imshow(m.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax[0, 0].set_xticks(range(m.shape[1]), [f"{c:.0%}" for c in m.columns])
    ax[0, 0].set_yticks(range(m.shape[0]), [f"{r:.1%}" for r in m.index])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax[0, 0].text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax[0, 0].set(title="Hybrid F1 vs ML review budget", xlabel="top share of subjects flagged",
                 ylabel="top share of visits flagged")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.04)

    f = sens["forensics"]
    f = f[f["system"] == HYB].assign(label=lambda d: d["high_q"].map("{:.2f}".format) + "/" +
                                     d["medium_q"].map("{:.2f}".format))
    for t, color in (("HIGH", "#9AA8B1"), ("HIGH+MEDIUM", "#8E4B9B")):
        d = f[f["flag_tiers"] == t]
        ax[0, 1].plot(d["label"], d["f1"], marker="o", color=color, label=f"flag {t}")
    ax[0, 1].set(title="Hybrid F1 vs forensic tier cut-offs (high_q/medium_q)", ylabel="F1")
    ax[0, 1].tick_params(axis="x", rotation=60)
    ax[0, 1].legend()

    r = sens["rules"]
    for k, (name, color) in enumerate((("rules-only", "#0F6E7A"), (HYB, "#B42318"))):
        d = r[r["system"] == name]
        ax[1, 0].bar(np.arange(len(d)) + k * 0.38, d["f1"], 0.36, color=color, label=name)
    ax[1, 0].set_xticks(np.arange(3) + 0.19, ["all severities", "major and above", "critical only"])
    ax[1, 0].set(title="F1 vs the severity at which a rule violation counts as a flag", ylabel="F1")
    ax[1, 0].legend()

    fu = sens["fusion"]
    w = 0.27
    for k, (col, color) in enumerate((("precision", "#E07B39"), ("recall", "#0F6E7A"), ("f1", "#B42318"))):
        ax[1, 1].bar(np.arange(len(fu)) + k * w, fu[col], w, color=color, label=col)
    ax[1, 1].set_xticks(np.arange(len(fu)) + w, fu.index, fontsize=8)
    ax[1, 1].set(title="Fusion rule: any detector vs agreement", ylim=(0, 1.05))
    ax[1, 1].legend()
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ============================================================ markdown
def build_markdown(res: dict, sens: dict | None = None, site_tbl: pd.DataFrame | None = None,
                   multiseed: pd.DataFrame | None = None, n_sites: int = 12, notes: str = "") -> str:
    p = res["prevalence"]
    L = ["# Evaluation report", "",
         f"Evaluated subjects: {p['subjects']} ({p['positive']} with at least one injected anomaly: "
         f"{', '.join(f'{c} {p[c]}' for c in CATEGORIES)}). {notes}", "",
         "## 1. Overall performance (any anomaly)", "", md_table(res["overall"])]
    if "boot_ci" in res:
        L += ["", "Bootstrap 95% intervals (subjects resampled):", "", md_table(res["boot_ci"])]
    L += ["", "## 2. Per anomaly category", "",
          "Precision and F1 here are measured against clean subjects only; AUC separates the category's subjects "
          "from clean ones.", "", md_table(res["category"].drop(columns=[]).round(3)),
          "", "## 3. Recall for each injected anomaly type", "", md_table(res["subtype"]),
          "", "## 4. Ablation: every combination of detectors", "", md_table(res["ablation"]),
          "", "Marginal value of each component (hybrid minus the hybrid without it):", "", md_table(res["loo"]),
          "", "## 5. Does the hybrid win, and why?", ""]
    L += [f"{t}" if t.startswith("-") else f"{t}\n" for t in why_hybrid_text(res)]
    by_pat, by_cat, uniq = res["coverage"]
    L += ["", "Which detectors caught each anomalous subject:", "", md_table(by_pat),
          "", "By category:", "", md_table(by_cat),
          "", "## 6. Same workload for everyone", "",
          "Each system reviews its own top b% of subjects.", "", md_table(res["budget"]),
          "", "Fusion rule:", "", md_table(res["fusion"])]
    if sens:
        L += ["", "## 7. Sensitivity", "",
              "ML review budget (hybrid):", "",
              md_table(sens["ml"][sens["ml"]["system"] == HYB][["visit_q", "subject_q", "precision", "recall", "f1"]]),
              "", "Forensic tier cut-offs (hybrid):", "",
              md_table(sens["forensics"][sens["forensics"]["system"] == HYB][
                  ["high_q", "medium_q", "flag_tiers", "precision", "recall", "f1"]]),
              "", "Rule severity threshold:", "", md_table(sens["rules"][["rules_flag_from", "system", "precision", "recall", "f1"]])]
        if "ml" in sens:
            h = sens["ml"][sens["ml"]["system"] == HYB]["f1"]
            L += ["", f"Across the ML budget grid the hybrid F1 ranges from {h.min():.2f} to {h.max():.2f} "
                      f"(spread {h.max() - h.min():.2f}); across forensic cut-offs from "
                      f"{sens['forensics'][sens['forensics']['system'] == HYB]['f1'].min():.2f} to "
                      f"{sens['forensics'][sens['forensics']['system'] == HYB]['f1'].max():.2f}."]
    if site_tbl is not None and len(site_tbl):
        L += ["", "## 8. Site level", "", "Rank of each truly bad site (1 = riskiest) by each system.", "",
              md_table(site_tbl)]
    if multiseed is not None and len(multiseed):
        L += ["", "## 9. Replicate datasets (not used for any tuning)", "", md_table(multiseed)]
    L += ["", limitations_markdown(p, n_sites)]
    return "\n".join(L)


def save_report(res, sens, site_tbl, multiseed, root, n_sites: int = 12, notes: str = "") -> dict:
    root = Path(root)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    make_main_figure(res, root / "docs" / "phase10_main.png")
    if sens:
        make_sensitivity_figure(sens, root / "docs" / "phase10_sensitivity.png")
    md = build_markdown(res, sens, site_tbl, multiseed, n_sites, notes)
    (root / "docs" / "evaluation_report.md").write_text(md, encoding="utf-8")
    (root / "docs" / "LIMITATIONS.md").write_text(limitations_markdown(res["prevalence"], n_sites), encoding="utf-8")
    return {"report": root / "docs" / "evaluation_report.md"}
