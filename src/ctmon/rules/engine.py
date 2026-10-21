"""Inference engine: loads the knowledge base (YAML) and runs it on the trial data."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .rules_library import COLS, META_COLS, META_TYPES, RULE_TYPES

SEVERITY_RANK = {"minor": 1, "major": 2, "critical": 3}
DEV_COLS = ["deviation_id", "rule_id", "rule_name", "category", "severity",
            "subject_id", "site_id", "visit_num", "record_id", "table", "field",
            "observed", "expected", "evidence", "reference", "detects"]
ESC_COLS = ["rule_id", "rule_name", "level", "entity_id", "site_id", "score", "detail", "action"]


@dataclass
class Rule:
    id: str
    name: str
    type: str
    severity: str = "major"
    category: str = "PD"
    description: str = ""
    reference: str = ""
    detects: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MetaRule:
    id: str
    name: str
    type: str
    action: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleResult:
    deviations: pd.DataFrame
    escalations: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        if self.deviations.empty:
            return pd.DataFrame()
        t = pd.crosstab(self.deviations["rule_id"], self.deviations["severity"])
        for s in SEVERITY_RANK:
            if s not in t:
                t[s] = 0
        t["total"] = t.sum(axis=1)
        return t[["critical", "major", "minor", "total"]].sort_values("total", ascending=False)


class RuleEngine:
    def __init__(self, rules_cfg: dict, protocol: dict):
        self.protocol = protocol
        self.rules = [Rule(**r) for r in rules_cfg.get("rules", [])]
        self.meta_rules = [MetaRule(**m) for m in rules_cfg.get("meta_rules", [])]
        self._validate()

    @classmethod
    def from_files(cls, rules_path: str | Path, protocol_path: str | Path) -> "RuleEngine":
        with open(rules_path, encoding="utf-8") as f:
            rules_cfg = yaml.safe_load(f)
        with open(protocol_path, encoding="utf-8") as f:
            protocol = yaml.safe_load(f)
        return cls(rules_cfg, protocol)

    def _validate(self) -> None:
        ids = [r.id for r in self.rules] + [m.id for m in self.meta_rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate rule ids in knowledge base")
        for r in self.rules:
            if r.type not in RULE_TYPES:
                raise ValueError(f"Rule {r.id}: unknown type '{r.type}'. Known: {sorted(RULE_TYPES)}")
            if r.severity not in SEVERITY_RANK:
                raise ValueError(f"Rule {r.id}: bad severity '{r.severity}'")
        for m in self.meta_rules:
            if m.type not in META_TYPES:
                raise ValueError(f"Meta-rule {m.id}: unknown type '{m.type}'. Known: {sorted(META_TYPES)}")

    def covered_subtypes(self) -> set[str]:
        return {s for r in self.rules if r.enabled for s in r.detects}

    def run(self, tables: dict[str, pd.DataFrame]) -> RuleResult:
        frames = []
        for r in self.rules:
            if not r.enabled:
                continue
            out = RULE_TYPES[r.type](tables, self.protocol, r.params)
            if out.empty:
                continue
            out = out.copy()
            out["severity"] = out["severity"].fillna(r.severity)
            out["rule_id"], out["rule_name"] = r.id, r.name
            out["category"], out["reference"] = r.category, r.reference
            out["detects"] = "|".join(r.detects)
            frames.append(out)

        if frames:
            devs = pd.concat(frames, ignore_index=True)
            smap = tables["subjects"].set_index("subject_id")["site_id"]
            devs["site_id"] = devs["site_id"].fillna(devs["subject_id"].map(smap))
            devs["_rank"] = devs["severity"].map(SEVERITY_RANK)
            devs = devs.sort_values(["_rank", "subject_id", "visit_num", "rule_id"],
                                    ascending=[False, True, True, True]).reset_index(drop=True)
            devs["deviation_id"] = [f"D{i + 1:05d}" for i in range(len(devs))]
            devs = devs[DEV_COLS]
        else:
            devs = pd.DataFrame(columns=DEV_COLS)

        return RuleResult(devs, self._run_meta(devs, tables))

    def _run_meta(self, devs: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for m in self.meta_rules:
            out = META_TYPES[m.type](devs, tables, self.protocol, m.params)
            if out.empty:
                continue
            out = out.copy()
            out["rule_id"], out["rule_name"], out["action"] = m.id, m.name, m.action
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=ESC_COLS)
        return pd.concat(frames, ignore_index=True)[ESC_COLS]
