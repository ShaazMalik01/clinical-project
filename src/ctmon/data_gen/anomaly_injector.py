"""Anomaly injector: corrupts a CLEAN trial with labeled anomalies.

Categories : PD (protocol deviation), OUT (outlier), FAB (fabrication), SAE (late safety reporting).
SITE = site-level misconduct is represented by two 'bad sites' that receive a much higher share
of injected anomalies. Every injection is logged in the ground-truth table.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .generator import DATE_COLS

VITAL_FIELDS = ["sbp", "dbp", "hr", "weight_kg", "temp_c"]


class AnomalyInjector:
    def __init__(self, tables: dict[str, pd.DataFrame], protocol: dict,
                 settings: dict, seed: int):
        self.t = {k: v.copy() for k, v in tables.items()}   # never mutate the clean data
        self.protocol = protocol
        self.cfg = settings["injection"]
        self.rng = np.random.default_rng(seed)
        self.visit_cfg = {v["num"]: v for v in protocol["visits"]}
        self.lab_visits = set(protocol["lab_visits"])
        self.deadline = protocol["safety"]["sae_report_deadline_days"]

        subs = self.t["subjects"]
        self.site_of: dict[str, str] = dict(zip(subs.subject_id, subs.site_id))
        self.next_idx: dict[str, int] = subs.groupby("site_id").size().to_dict()
        self.locked: set[str] = set()
        self.touched: set[tuple[str, int]] = set()
        self.gt: list[dict[str, Any]] = []
        self._n = 0

        ids = self.t["sites"].site_id.tolist()
        a, b = self.rng.choice(ids, size=2, replace=False)
        self.bad = {"sloppy": str(a), "fabricating": str(b)}

    @staticmethod
    def _s(x: Any) -> str | None:
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            return x.date().isoformat()
        return str(x)

    def _log(self, category: str, subtype: str, severity: str, table: str,
             subject_id: str, visit_num: int | None = None, field: str | None = None,
             original: Any = None, injected: Any = None, granularity: str = "record",
             description: str = "", record_id: str | None = None,
             related: str | None = None) -> None:
        self._n += 1
        site = self.site_of.get(subject_id)
        self.gt.append({
            "anomaly_id": f"A{self._n:04d}", "category": category, "subtype": subtype,
            "severity": severity, "table": table, "subject_id": subject_id,
            "site_id": site, "visit_num": visit_num, "field": field,
            "original_value": self._s(original), "injected_value": self._s(injected),
            "granularity": granularity, "record_id": record_id,
            "related_subject_id": related,
            "in_bad_site": site in self.bad.values(), "description": description})

    def _pick(self, items: list, n: int, profile: str | None = None,
              getter: Callable[[Any], str] | None = None) -> list:
        items = list(items)
        n = min(n, len(items))
        if n == 0:
            return []
        get = getter or (lambda x: self.site_of[x])
        w = np.array([self.cfg["site_boost"] if profile and get(x) == self.bad[profile] else 1.0
                      for x in items])
        idx = self.rng.choice(len(items), size=n, replace=False, p=w / w.sum())
        return [items[i] for i in idx]

    def _free_subjects(self, min_last_visit: int = 1) -> list[str]:
        s = self.t["subjects"]
        ok = s[(s.last_visit_num >= min_last_visit) & (~s.subject_id.isin(self.locked))]
        return ok.subject_id.tolist()

    @staticmethod
    def _idx(df: pd.DataFrame, sid: str, vn: int) -> int | None:
        hits = df.index[(df["subject_id"] == sid) & (df["visit_num"] == vn)]
        return int(hits[0]) if len(hits) else None

    def _shift_visit(self, sid: str, vn: int, days: int) -> None:
        delta = pd.Timedelta(days=days)
        for name, col in (("visits", "actual_date"), ("vitals", "date"),
                          ("labs", "date"), ("dosing", "date")):
            df = self.t[name]
            m = (df["subject_id"] == sid) & (df["visit_num"] == vn)
            df.loc[m, col] = df.loc[m, col] + delta

    def inject_ineligible(self, n: int) -> None:
        for sid in self._pick(self._free_subjects(), n, "sloppy"):
            self.locked.add(sid)
            crit = str(self.rng.choice(["age", "sbp", "creatinine"]))
            if crit == "age":
                sub = self.t["subjects"]
                i = int(sub.index[sub.subject_id == sid][0])
                old, new = int(sub.at[i, "age"]), int(self.rng.integers(77, 86))
                sub.at[i, "age"] = new
                table, field = "subjects", "age"
            elif crit == "sbp":
                vit = self.t["vitals"]
                i = self._idx(vit, sid, 1)
                old, new = int(vit.at[i, "sbp"]), int(self.rng.integers(162, 186))
                vit.at[i, "sbp"] = new
                table, field = "vitals", "sbp"
            else:
                lab = self.t["labs"]
                i = self._idx(lab, sid, 1)
                old, new = float(lab.at[i, "creatinine"]), round(float(self.rng.uniform(1.6, 2.3)), 2)
                lab.at[i, "creatinine"] = new
                table, field = "labs", "creatinine"
            self._log("PD", "ineligible_enrolled", "critical", table, sid, 1, field, old, new,
                      description=f"Violates eligibility criterion ({crit}) at screening")

    def inject_consent_violation(self, n: int) -> None:
        for sid in self._pick(self._free_subjects(), n, "sloppy"):
            self.locked.add(sid)
            vis, con = self.t["visits"], self.t["consent"]
            scr = vis.loc[(vis.subject_id == sid) & (vis.visit_num == 1), "actual_date"].iloc[0]
            m = con.subject_id == sid
            old = con.loc[m, "consent_date"].iloc[0]
            new = scr + pd.Timedelta(days=int(self.rng.integers(1, 6)))
            con.loc[m, "consent_date"] = new
            self._log("PD", "procedure_before_consent", "critical", "consent", sid, 1,
                      "consent_date", old, new,
                      description="Screening procedures done before informed consent")

    def inject_copy_forward(self, n: int) -> None:
        for sid in self._pick(self._free_subjects(8), n, "fabricating"):
            self.locked.add(sid)
            vit = self.t["vitals"]
            last = int(self.t["subjects"].set_index("subject_id").at[sid, "last_visit_num"])
            L = int(self.rng.integers(3, 6))
            start = int(self.rng.integers(3, last - L + 1))
            src = vit.loc[self._idx(vit, sid, start), VITAL_FIELDS]
            for k in range(1, L + 1):
                i = self._idx(vit, sid, start + k)
                for f in VITAL_FIELDS:
                    vit.at[i, f] = src[f]
                self._log("FAB", "copy_forward", "major", "vitals", sid, start + k,
                          "|".join(VITAL_FIELDS), None, f"copy of visit {start}",
                          description=f"Vitals identical to visit {start} (copy-paste)")

    def inject_low_variance(self, n: int) -> None:
        for sid in self._pick(self._free_subjects(8), n, "fabricating"):
            self.locked.add(sid)
            vit = self.t["vitals"]
            rows = vit.index[vit.subject_id == sid]
            base = {f: int(vit.at[rows[0], f]) for f in ("sbp", "dbp", "hr")}
            for i in rows:
                for f, c in base.items():
                    vit.at[i, f] = c + int(self.rng.integers(-1, 2))
            self._log("FAB", "low_variance", "major", "vitals", sid, None, "sbp|dbp|hr",
                      None, "values within +/-1 of constant", granularity="subject",
                      description="Implausibly constant vitals across all visits")

    def inject_digit_preference(self, n: int) -> None:
        for sid in self._pick(self._free_subjects(), n, "fabricating"):
            self.locked.add(sid)
            vit = self.t["vitals"]
            rows = vit.index[vit.subject_id == sid]
            vit.loc[rows, "sbp"] = (np.round(vit.loc[rows, "sbp"] / 10) * 10).astype(int)
            vit.loc[rows, "dbp"] = (np.round(vit.loc[rows, "dbp"] / 5) * 5).astype(int)
            vit.loc[rows, "hr"] = (np.round(vit.loc[rows, "hr"] / 5) * 5).astype(int)
            self._log("FAB", "digit_preference", "major", "vitals", sid, None, "sbp|dbp|hr",
                      None, "rounded to multiples of 10/5", granularity="subject",
                      description="Terminal-digit preference (values rounded)")

    def inject_duplicates(self, n: int) -> None:
        sites = self.t["sites"].site_id.tolist()
        for src in self._pick(self._free_subjects(8), n, "fabricating"):
            src_site = self.site_of[src]
            others = [s for s in sites if s != src_site]
            new_site = src_site if self.rng.random() < 0.6 else str(self.rng.choice(others))
            self.next_idx[new_site] += 1
            new_id = f"{new_site}-{self.next_idx[new_site]:03d}"
            self.site_of[new_id] = new_site
            shift = pd.Timedelta(days=int(self.rng.integers(20, 120)))

            parts = {}
            for name in ("subjects", "consent", "visits", "vitals", "labs", "dosing"):
                df = self.t[name]
                p = df[df.subject_id == src].copy()
                p["subject_id"] = new_id
                if "site_id" in p.columns:
                    p["site_id"] = new_site
                for c in DATE_COLS.get(name, []):
                    p[c] = p[c] + shift
                parts[name] = p

            parts["subjects"]["bmi"] = (parts["subjects"]["bmi"] + self.rng.normal(0, 0.1)).round(1)
            v = parts["vitals"]
            for c in ("sbp", "dbp", "hr"):
                v[c] = v[c] + self.rng.integers(-1, 2, size=len(v))
            v["weight_kg"] = (v["weight_kg"] + self.rng.normal(0, 0.1, len(v))).round(1)
            v["temp_c"] = (v["temp_c"] + self.rng.choice([-0.1, 0.0, 0.1], len(v))).round(1)
            lb = parts["labs"]
            for c, d in (("alt", 1), ("creatinine", 2), ("glucose", 1), ("hemoglobin", 1)):
                lb[c] = (lb[c] * (1 + self.rng.normal(0, 0.01, len(lb)))).round(d)

            for name, p in parts.items():
                self.t[name] = pd.concat([self.t[name], p], ignore_index=True)
            self.locked.update([src, new_id])
            self._log("FAB", "duplicate_subject", "critical", "subjects", new_id, None,
                      "subject_id", src, new_id, granularity="subject", related=src,
                      description=f"Near-duplicate of {src} (same demographics, ~same vitals)")

    def _record_pool(self, df: pd.DataFrame, extra: pd.Series | None = None) -> list[tuple[str, int]]:
        m = ~df.subject_id.isin(self.locked)
        if extra is not None:
            m &= extra
        return [(r.subject_id, int(r.visit_num)) for r in df[m].itertuples()
                if (r.subject_id, int(r.visit_num)) not in self.touched]

    def inject_out_of_window(self, n: int) -> None:
        vis = self.t["visits"]
        pool = self._record_pool(vis, (vis.visit_num >= 3) & (vis.status == "Completed"))
        for sid, vn in self._pick(pool, n, "sloppy", lambda x: self.site_of[x[0]]):
            self.touched.add((sid, vn))
            w = self.visit_cfg[vn]["window"]
            mag = int(self.rng.integers(2 * w + 1, 2 * w + 5))
            shift = mag * int(self.rng.choice([-1, 1]))
            i = self._idx(vis, sid, vn)
            old = vis.at[i, "actual_date"]
            self._shift_visit(sid, vn, shift)
            sev = "major" if mag - w >= 5 else "minor"
            self._log("PD", "visit_out_of_window", sev, "visits", sid, vn, "actual_date",
                      old, old + pd.Timedelta(days=shift),
                      description=f"Visit moved {shift:+d} days (window +/-{w})")

    def inject_missed_visits(self, n: int) -> None:
        vis = self.t["visits"]
        last = self.t["subjects"].set_index("subject_id")["last_visit_num"].to_dict()
        pool = [x for x in self._record_pool(vis, (vis.visit_num >= 3) & (vis.status == "Completed"))
                if x[1] < last[x[0]]]
        for sid, vn in self._pick(pool, n, "sloppy", lambda x: self.site_of[x[0]]):
            self.touched.add((sid, vn))
            vis = self.t["visits"]
            i = self._idx(vis, sid, vn)
            old = vis.at[i, "actual_date"]
            vis.at[i, "status"] = "Missed"
            vis.at[i, "actual_date"] = pd.NaT
            for name in ("vitals", "labs", "dosing"):
                df = self.t[name]
                self.t[name] = df[~((df.subject_id == sid) & (df.visit_num == vn))]
            self._log("PD", "missed_visit", "major", "visits", sid, vn, "status",
                      old, "Missed", description="Visit not performed; subject continued in study")

    def inject_wrong_dose(self, n: int) -> None:
        dos = self.t["dosing"]
        pool = self._record_pool(dos)
        for sid, vn in self._pick(pool, n, "sloppy", lambda x: self.site_of[x[0]]):
            self.touched.add((sid, vn))
            i = self._idx(dos, sid, vn)
            correct = int(dos.at[i, "dose_mg"])
            new = int(self.rng.choice([d for d in (0, 5, 10, 15, 20, 30, 40) if d != correct]))
            dos.at[i, "dose_mg"] = new
            self._log("PD", "wrong_dose", "critical" if new >= 30 else "major", "dosing",
                      sid, vn, "dose_mg", correct, new,
                      description=f"Dispensed {new} mg; protocol dose for arm is {correct} mg")

    def inject_value_outliers(self, n: int) -> None:
        vit = self.t["vitals"]
        pool = self._record_pool(vit)
        for sid, vn in self._pick(pool, n, None, lambda x: self.site_of[x[0]]):
            self.touched.add((sid, vn))
            use_lab = vn in self.lab_visits and self.rng.random() < 0.4
            if use_lab:
                df, table = self.t["labs"], "labs"
                kind = str(self.rng.choice(["alt_extreme", "creatinine_extreme", "glucose_unit_mix"]))
            else:
                df, table = vit, "vitals"
                kind = str(self.rng.choice(["sbp_extra_digit", "hr_extreme", "weight_decimal_shift",
                                            "temp_unit_mix", "sbp_crisis"]))
            i = self._idx(df, sid, vn)
            if kind == "sbp_extra_digit":
                field, old = "sbp", df.at[i, "sbp"]; new = int(old) * 10
            elif kind == "hr_extreme":
                field, old = "hr", df.at[i, "hr"]; new = int(self.rng.choice([0, 250, 300]))
            elif kind == "weight_decimal_shift":
                field, old = "weight_kg", df.at[i, "weight_kg"]; new = round(float(old) * 10, 1)
            elif kind == "temp_unit_mix":
                field, old = "temp_c", df.at[i, "temp_c"]; new = round(float(old) * 9 / 5 + 32, 1)
            elif kind == "sbp_crisis":
                field, old = "sbp", df.at[i, "sbp"]; new = int(self.rng.integers(205, 236))
            elif kind == "alt_extreme":
                field, old = "alt", df.at[i, "alt"]; new = round(float(old) * float(self.rng.uniform(8, 15)), 1)
            elif kind == "creatinine_extreme":
                field, old = "creatinine", df.at[i, "creatinine"]; new = round(float(old) * float(self.rng.uniform(4, 8)), 2)
            else:
                field, old = "glucose", df.at[i, "glucose"]; new = round(float(old) / 18.0, 1)
            df.at[i, field] = new
            plausible = kind in ("sbp_crisis", "alt_extreme", "creatinine_extreme")
            self._log("OUT", "value_outlier", "minor" if plausible else "major", table, sid, vn,
                      field, old, new,
                      description=f"{kind} ({'extreme but plausible' if plausible else 'physically impossible'})")

    def inject_late_sae(self, n: int) -> None:
        subs, vis = self.t["subjects"].set_index("subject_id"), self.t["visits"]
        new_rows = []
        for k, sid in enumerate(self._pick(subs.index.tolist(), n, "sloppy"), start=1):
            base = subs.at[sid, "baseline_date"]
            last = vis.loc[vis.subject_id == sid, "actual_date"].max()
            span = max(1, (last - base).days)
            onset = base + pd.Timedelta(days=int(self.rng.integers(1, span + 1)))
            delay = int(self.rng.integers(3, 16))
            ae_id = f"AE-L{k:04d}"
            new_rows.append({"ae_id": ae_id, "subject_id": sid, "site_id": self.site_of[sid],
                             "term": str(self.rng.choice(["Myocardial infarction", "Stroke",
                                                          "Hypertensive crisis", "Acute kidney injury"])),
                             "severity": "Severe", "is_serious": True,
                             "onset_date": onset, "report_date": onset + pd.Timedelta(days=delay)})
            self._log("SAE", "late_sae_report", "critical", "adverse_events", sid, None,
                      "report_date", f"deadline {onset + pd.Timedelta(days=self.deadline)}",
                      onset + pd.Timedelta(days=delay), record_id=ae_id,
                      description=f"SAE reported {delay} days after onset (deadline {self.deadline} day)")
        self.t["adverse_events"] = pd.concat(
            [self.t["adverse_events"], pd.DataFrame(new_rows)], ignore_index=True)

    def run(self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        c = self.cfg["counts"]
        self.inject_ineligible(c["ineligible_enrolled"])
        self.inject_consent_violation(c["procedure_before_consent"])
        self.inject_copy_forward(c["copy_forward"])
        self.inject_low_variance(c["low_variance"])
        self.inject_digit_preference(c["digit_preference"])
        self.inject_duplicates(c["duplicate_subjects"])
        self.inject_out_of_window(c["visit_out_of_window"])
        self.inject_missed_visits(c["missed_visit"])
        self.inject_wrong_dose(c["wrong_dose"])
        self.inject_value_outliers(c["value_outliers"])
        self.inject_late_sae(c["late_sae"])
        return self._finalize()

    def _finalize(self):
        order = {"sites": ["site_id"], "subjects": ["subject_id"], "consent": ["subject_id"],
                 "visits": ["subject_id", "visit_num"], "vitals": ["subject_id", "visit_num"],
                 "labs": ["subject_id", "visit_num"], "dosing": ["subject_id", "visit_num"],
                 "adverse_events": ["subject_id", "onset_date"]}
        for name, cols in order.items():
            self.t[name] = self.t[name].sort_values(cols).reset_index(drop=True)

        gt = pd.DataFrame(self.gt)
        gt["visit_num"] = pd.to_numeric(gt["visit_num"]).astype("Int64")

        site_gt = self.t["sites"][["site_id"]].copy()
        names = {self.bad["sloppy"]: "sloppy_process", self.bad["fabricating"]: "fabrication"}
        site_gt["profile"] = site_gt.site_id.map(names).fillna("clean")
        site_gt["is_bad_site"] = site_gt.profile != "clean"
        site_gt["category"] = np.where(site_gt.is_bad_site, "SITE", "")
        counts = gt.groupby("site_id").size()
        site_gt["n_injected_anomalies"] = site_gt.site_id.map(counts).fillna(0).astype(int)
        return self.t, gt, site_gt
