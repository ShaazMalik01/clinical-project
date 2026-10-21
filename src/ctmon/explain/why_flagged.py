"""'Why flagged' cards: one plain-English explanation per alert, for every detector."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..ml.features import SUBJECT_MODEL_COLS
from .zscores import build_visit_frames, describe_subject_feature, top_visit_features, zscores

SEV_RANK = {"critical": 3, "major": 2, "minor": 1}


@dataclass
class Card:
    alert_id: str
    level: str
    source: str
    site_id: str
    subject_id: str | None
    visit_num: int | None
    severity: str
    priority: float
    headline: str
    summary: str
    evidence: list = field(default_factory=list)
    reference: str = ""
    next_step: str = ""

    def to_text(self) -> str:
        lines = [f"[{self.severity.upper()}] {self.headline}", f"Why flagged: {self.summary}", "Evidence:"]
        lines += [f"  {i}. {e}" for i, e in enumerate(self.evidence, 1)]
        if self.reference:
            lines.append(f"Reference: {self.reference}")
        if self.next_step:
            lines.append(f"Suggested next step: {self.next_step}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["card"] = self.to_text()
        return d


_RULE_PRIORITY = {"critical": 0.95, "major": 0.75, "minor": 0.45}
_RULE_NEXT = {
    "critical": "Escalate to the study lead, verify against source documents and file an "
                "important protocol-deviation report.",
    "major": "Verify against source documents; document the deviation and a corrective action.",
    "minor": "Document the deviation and track this site's trend.",
}
_RULE_NEXT_OUT = ("Verify the value against the source document; if it is a transcription or unit "
                  "error, raise a data query for correction.")


def rule_cards(dev: pd.DataFrame) -> list:
    cards = []
    for r in dev.itertuples(index=False):
        vn = int(r.visit_num) if pd.notna(r.visit_num) else None
        where = f"subject {r.subject_id}" + (f", visit {vn}" if vn is not None else "")
        cards.append(Card(
            alert_id=f"RUL-{r.deviation_id}", level="record", source="RULE",
            site_id=str(r.site_id), subject_id=str(r.subject_id), visit_num=vn,
            severity=r.severity, priority=_RULE_PRIORITY[r.severity],
            headline=f"{r.rule_name} — {where}",
            summary=f"Rule {r.rule_id} ({r.category}) is violated; severity {r.severity}.",
            evidence=[str(r.evidence), f"Observed: {r.observed} | Expected: {r.expected}"],
            reference=str(r.reference),
            next_step=_RULE_NEXT_OUT if r.category == "OUT" else _RULE_NEXT[r.severity]))
    return cards


def ml_visit_cards(vs: pd.DataFrame, frames: dict, dev: pd.DataFrame | None = None) -> list:
    hits: dict = {}
    if dev is not None and len(dev):
        for sid, vn, rid in zip(dev["subject_id"], dev["visit_num"], dev["rule_id"]):
            if pd.notna(vn):
                hits.setdefault((sid, int(vn)), set()).add(rid)
    n_total = len(vs)
    rank = vs["ensemble"].rank(ascending=False, method="min")
    cards = []
    for r in vs[vs["flag"].astype(bool)].itertuples():
        fr = frames[r.view]
        pos = fr.index[(r.subject_id, int(r.visit_num))]
        top = top_visit_features(fr, pos, 3)
        pct = 100.0 * float(rank.loc[r.Index]) / n_total
        votes = int(r.votes)
        ev = [t["text"] for t in top]
        ev.append(f"Detector percentile ranks: Isolation Forest {r.iforest:.3f}, "
                  f"Mahalanobis {r.mahal:.3f}, LOF {r.lof:.3f}")
        rules = hits.get((r.subject_id, int(r.visit_num)))
        if rules:
            ev.append("Also caught by the rule engine: " + ", ".join(sorted(rules)))
        kind = "laboratory" if r.view == "labs" else "vital-sign"
        cards.append(Card(
            alert_id=f"MLV-{r.subject_id}-V{int(r.visit_num)}", level="visit", source="ML",
            site_id=str(r.site_id), subject_id=str(r.subject_id), visit_num=int(r.visit_num),
            severity="major" if votes >= 3 else "minor", priority=float(r.ensemble),
            headline=f"Unusual {kind} values — subject {r.subject_id}, visit {int(r.visit_num)}",
            summary=(f"Among the top {pct:.2f}% most unusual of {n_total:,} visits; "
                     f"{votes} of 3 detectors independently rate it as extreme."),
            evidence=ev, reference="ICH E6 4.9.1 (accurate, verifiable data)",
            next_step=("Check the source document; if it is a transcription or unit error, raise a "
                       "data query. If the value is genuine, assess clinical significance.")))
    return cards


def ml_subject_cards(ss: pd.DataFrame, sf: pd.DataFrame, fz_subjects: pd.DataFrame | None = None,
                     dev: pd.DataFrame | None = None) -> list:
    X = sf[SUBJECT_MODEL_COLS]
    z, st = zscores(X)
    rank = ss["ensemble"].rank(ascending=False, method="min")
    n_dev = dev.groupby("subject_id").size().to_dict() if dev is not None and len(dev) else {}
    ftier = dict(zip(fz_subjects["subject_id"], fz_subjects["tier"])) if fz_subjects is not None else {}
    cards = []
    for r in ss[ss["flag"].astype(bool)].itertuples():
        zr = z.loc[r.subject_id]
        order = zr.abs().sort_values(ascending=False).index[:3]
        ev = [describe_subject_feature(f, float(X.at[r.subject_id, f]), float(zr[f]),
                                       float(st.at[f, "median"])) for f in order]
        if n_dev.get(r.subject_id):
            ev.append(f"Rule engine: {n_dev[r.subject_id]} deviation(s) on this subject")
        if ftier.get(r.subject_id) in ("HIGH", "MEDIUM"):
            ev.append(f"Forensic screen: {ftier[r.subject_id]} fabrication-suspicion tier")
        pct = 100.0 * float(rank.loc[r.Index]) / len(ss)
        votes = int(r.votes)
        cards.append(Card(
            alert_id=f"MLS-{r.subject_id}", level="subject", source="ML",
            site_id=str(r.site_id), subject_id=str(r.subject_id), visit_num=None,
            severity="major" if votes >= 3 else "minor", priority=float(r.ensemble),
            headline=f"Unusual subject profile — {r.subject_id}",
            summary=(f"Among the top {pct:.1f}% most atypical of {len(ss)} subjects across 16 "
                     f"profile features; {votes} of 3 detectors agree."),
            evidence=ev, reference="Risk-based monitoring: central statistical review (ICH E6 5.18)",
            next_step=("Review this subject's source documents (eligibility, consent, vitals "
                       "charts); the listed features show where to look first.")))
    return cards


_PATTERN = {"digit_preference": "terminal-digit preference", "low_variance": "abnormally low variability",
            "copy_forward": "copy-forward vitals", "duplicate": "possible duplicate subject",
            "none": "combined weak signals"}
_NEXT_FOR = {
    "copy_forward": "Request the original vitals charts for the listed visits and compare.",
    "duplicate": ("Compare identity/source documents of both subjects and check for duplicate "
                  "enrolment; contact the site PI."),
    "digit_preference": "Check the measurement procedure and device; compare with source records.",
    "low_variance": "Verify that each visit's vitals were measured; compare with source records.",
    "none": "Review source data for this subject.",
}


def forensic_cards(fz: pd.DataFrame) -> list:
    cards = []
    for r in fz[fz["tier"].isin(["HIGH", "MEDIUM"])].itertuples():
        pat = str(r.primary_pattern)
        ev = [s for s in str(r.reasons).split("; ") if s] or ["Statistical integrity screen flagged this subject."]
        sev = "critical" if pat == "duplicate" else ("major" if r.tier == "HIGH" else "minor")
        cards.append(Card(
            alert_id=f"FOR-{r.subject_id}", level="subject", source="FORENSIC",
            site_id=str(r.site_id), subject_id=str(r.subject_id), visit_num=None,
            severity=sev, priority=float(r.suspicion),
            headline=f"Possible data-integrity issue — subject {r.subject_id}",
            summary=(f"Fabrication-suspicion score {r.suspicion:.2f} ({r.tier} tier); main pattern: "
                     f"{_PATTERN.get(pat, pat)}. This is statistical suspicion, not proof."),
            evidence=ev, reference="ICH E6 4.9.1; data-integrity (ALCOA+) principles",
            next_step=_NEXT_FOR.get(pat, _NEXT_FOR["none"])))
    return cards


_SITE_SEV = {"FULL_AUDIT": "critical", "TARGETED_ONSITE": "major", "REMOTE_REVIEW": "minor"}


def site_cards(site_risk: pd.DataFrame) -> list:
    cards = []
    for r in site_risk[site_risk["tier"] != "ROUTINE"].itertuples():
        cards.append(Card(
            alert_id=f"SIT-{r.site_id}", level="site", source="RISK", site_id=str(r.site_id),
            subject_id=None, visit_num=None, severity=_SITE_SEV[r.tier],
            priority=float(min(1.0, r.risk_index)),
            headline=f"Site {r.site_id}: {r.tier.replace('_', ' ').lower()} recommended",
            summary=(f"Bayesian risk index {r.risk_index:.2f} (95% CI {r.ci_lo:.2f}-{r.ci_hi:.2f})."),
            evidence=[str(r.reason), "Main drivers: " + str(r.drivers)],
            reference="ICH E6(R2) 5.0.1 / 5.18: risk-based monitoring",
            next_step=str(r.recommendation)))
    return cards


def build_alert_table(tables: dict, deviations: pd.DataFrame, ml, fz,
                      site_risk: pd.DataFrame | None = None, frames: dict | None = None):
    frames = frames or build_visit_frames(tables)
    cards = rule_cards(deviations)
    cards += ml_visit_cards(ml.visit_scores, frames, deviations)
    cards += ml_subject_cards(ml.subject_scores, ml.subject_features, fz.subject_scores, deviations)
    cards += forensic_cards(fz.subject_scores)
    if site_risk is not None:
        cards += site_cards(site_risk)
    cards.sort(key=lambda c: (-SEV_RANK[c.severity], -c.priority, c.alert_id))
    df = pd.DataFrame([c.to_dict() for c in cards])
    if len(df):
        df["evidence"] = df["evidence"].map(lambda e: " || ".join(e))
    return df, cards


def save_cards_json(cards: list, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([c.to_dict() for c in cards], indent=2, default=str), encoding="utf-8")


def save_cards_markdown(cards: list, path, per_source: int = 4) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 'Why flagged' cards: examples", ""]
    for src in ("RULE", "ML", "FORENSIC", "RISK"):
        sel = [c for c in cards if c.source == src][:per_source]
        if not sel:
            continue
        lines += [f"## {src}", ""]
        for c in sel:
            lines += ["```text", c.to_text(), "```", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
