"""Rule library: every function is a reusable *rule type* (an inference primitive).

Level-1 rule type : fn(tables, protocol, params) -> DataFrame[COLS]
Level-2 (meta)    : fn(deviations, tables, protocol, params) -> DataFrame[META_COLS]
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

COLS = ["subject_id", "site_id", "visit_num", "record_id", "table", "field",
        "observed", "expected", "evidence", "severity"]
META_COLS = ["level", "entity_id", "site_id", "score", "detail"]


def _out(n: int, **cols) -> pd.DataFrame:
    """Build a standard output frame. Scalars are broadcast; missing cols -> NA."""
    data = {}
    for c in COLS:
        v = cols.get(c)
        if v is None:
            data[c] = [pd.NA] * n
        elif isinstance(v, (str, int, float)):
            data[c] = [v] * n
        else:
            data[c] = list(v)
    df = pd.DataFrame(data, columns=COLS)
    df["visit_num"] = pd.array(df["visit_num"].tolist(), dtype="Int64")
    return df


def _ref(protocol: dict, dotted: str):
    node = protocol
    for key in dotted.split("."):
        node = node[key]
    return node


def _bound(protocol: dict, p: dict, which: str):
    if which in p:
        return p[which]
    if f"{which}_ref" in p:
        return _ref(protocol, p[f"{which}_ref"])
    return None


def bounded_field(tables, protocol, p):
    df = tables[p["table"]]
    visit = p.get("visit")
    if visit is not None:
        df = df[df["visit_num"] == visit]
    f = p["field"]
    lo, hi = _bound(protocol, p, "min"), _bound(protocol, p, "max")
    mask = pd.Series(False, index=df.index)
    parts = []
    if lo is not None:
        mask |= df[f] < lo
        parts.append(f">= {lo}")
    if hi is not None:
        mask |= df[f] > hi
        parts.append(f"<= {hi}")
    bad = df[mask]
    exp = " and ".join(parts)
    return _out(len(bad), subject_id=bad["subject_id"], site_id=bad["site_id"],
                visit_num=visit, table=p["table"], field=f,
                observed=bad[f].astype(str), expected=exp,
                evidence=[f"{f} = {v}, required {exp}" for v in bad[f]])


def plausible_range(tables, protocol, p):
    df = tables[p["table"]]
    frames = []
    for f in p["fields"]:
        lo, hi = protocol["plausible_ranges"][f]
        bad = df[(df[f] < lo) | (df[f] > hi)]
        if len(bad) == 0:
            continue
        frames.append(_out(
            len(bad), subject_id=bad["subject_id"], site_id=bad["site_id"],
            visit_num=bad["visit_num"], table=p["table"], field=f,
            observed=bad[f].astype(str), expected=f"{lo}-{hi}",
            evidence=[f"{f} = {v} is physically implausible (possible range {lo}-{hi}); "
                      f"likely entry or unit error" for v in bad[f]]))
    return pd.concat(frames, ignore_index=True) if frames else _out(0)


def consent_before_procedure(tables, protocol, p):
    scr = tables["visits"].query("visit_num == 1 and status == 'Completed'")[["subject_id", "actual_date"]]
    m = tables["consent"].merge(scr, on="subject_id")
    bad = m[m["consent_date"] > m["actual_date"]]
    gap = (bad["consent_date"] - bad["actual_date"]).dt.days
    return _out(len(bad), subject_id=bad["subject_id"], site_id=bad["site_id"], visit_num=1,
                table="consent", field="consent_date",
                observed=bad["consent_date"].dt.date.astype(str),
                expected="<= " + bad["actual_date"].dt.date.astype(str),
                evidence=[f"Consent signed {c.date()}, {g} day(s) AFTER screening procedures on {a.date()}"
                          for c, a, g in zip(bad["consent_date"], bad["actual_date"], gap)])


def visit_window(tables, protocol, p):
    v = tables["visits"]
    v = v[v["status"] == "Completed"].copy()
    win = {x["num"]: x["window"] for x in protocol["visits"]}
    v["window"] = v["visit_num"].map(win)
    v["dev"] = (v["actual_date"] - v["planned_date"]).dt.days
    v["excess"] = v["dev"].abs() - v["window"]
    bad = v[v["excess"] > 0]
    sev = np.where(bad["excess"] >= p.get("major_excess_days", 5), "major", "minor")
    return _out(len(bad), subject_id=bad["subject_id"], site_id=bad["site_id"],
                visit_num=bad["visit_num"], table="visits", field="actual_date",
                observed=bad["actual_date"].dt.date.astype(str),
                expected=[f"planned {pd_.date()} +/-{w}d" for pd_, w in zip(bad["planned_date"], bad["window"])],
                evidence=[f"{r.visit_name}: {r.dev:+d} days from plan (window +/-{int(r.window)}), "
                          f"{int(r.excess)} day(s) outside" for r in bad.itertuples()],
                severity=sev)


def missed_visit(tables, protocol, p):
    v = tables["visits"]
    bad = v[v["status"] == "Missed"]
    return _out(len(bad), subject_id=bad["subject_id"], site_id=bad["site_id"],
                visit_num=bad["visit_num"], table="visits", field="status",
                observed="Missed", expected="Completed",
                evidence=[f"Visit {r.visit_num} ({r.visit_name}) not performed; planned "
                          f"{r.planned_date.date()}" for r in bad.itertuples()])


def dose_vs_arm(tables, protocol, p):
    arm_dose = {a: d["dose_mg"] for a, d in protocol["arms"].items()}
    d = tables["dosing"].copy()
    d["expected"] = d["arm"].map(arm_dose)
    bad = d[d["dose_mg"] != d["expected"]]
    sev = np.where(bad["dose_mg"] > p.get("critical_above_mg", 25), "critical", "major")
    return _out(len(bad), subject_id=bad["subject_id"], site_id=bad["site_id"],
                visit_num=bad["visit_num"], table="dosing", field="dose_mg",
                observed=bad["dose_mg"].astype(str), expected=bad["expected"].astype(str),
                evidence=[f"Arm {r.arm}: protocol dose {r.expected} mg, dispensed {r.dose_mg} mg"
                          for r in bad.itertuples()],
                severity=sev)


def sae_deadline(tables, protocol, p):
    deadline = protocol["safety"]["sae_report_deadline_days"]
    ae = tables["adverse_events"]
    ae = ae[ae["is_serious"].astype(bool)].copy()
    ae["delay"] = (ae["report_date"] - ae["onset_date"]).dt.days
    bad = ae[ae["delay"] > deadline]
    return _out(len(bad), subject_id=bad["subject_id"], site_id=bad["site_id"],
                record_id=bad["ae_id"], table="adverse_events", field="report_date",
                observed=bad["delay"].astype(str) + " days",
                expected=f"<= {deadline} day(s)",
                evidence=[f"SAE '{r.term}' onset {r.onset_date.date()} reported after {r.delay} day(s) "
                          f"(deadline {deadline})" for r in bad.itertuples()])


def cross_field_bp(tables, protocol, p):
    v = tables["vitals"]
    bad = v[v["dbp"] >= v["sbp"]]
    return _out(len(bad), subject_id=bad["subject_id"], site_id=bad["site_id"],
                visit_num=bad["visit_num"], table="vitals", field="sbp|dbp",
                observed=bad["sbp"].astype(str) + "/" + bad["dbp"].astype(str),
                expected="DBP < SBP",
                evidence=[f"BP {r.sbp}/{r.dbp}: diastolic >= systolic" for r in bad.itertuples()])


RULE_TYPES: dict[str, Callable] = {
    "bounded_field": bounded_field,
    "plausible_range": plausible_range,
    "consent_before_procedure": consent_before_procedure,
    "visit_window": visit_window,
    "missed_visit": missed_visit,
    "dose_vs_arm": dose_vs_arm,
    "sae_deadline": sae_deadline,
    "cross_field_bp": cross_field_bp,
}


def _meta(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=META_COLS)


def repeat_subject(devs, tables, protocol, p):
    if devs.empty:
        return _meta([])
    g = devs.groupby("subject_id").agg(
        site_id=("site_id", "first"),
        n=("deviation_id", "count"),
        n_crit=("severity", lambda s: int((s == "critical").sum())),
        rules=("rule_id", lambda s: ", ".join(sorted(set(s)))))
    g = g[g["n"] >= p.get("min_deviations", 3)].reset_index()
    return _meta([dict(level="subject", entity_id=r.subject_id, site_id=r.site_id,
                       score=float(r.n),
                       detail=f"{r.n} deviations ({r.n_crit} critical): {r.rules}")
                  for r in g.itertuples()])


def site_hotspot(devs, tables, protocol, p):
    n_subj = tables["subjects"].groupby("site_id").size()
    crit = (devs[devs["severity"] == "critical"].groupby("site_id").size()
            .reindex(n_subj.index, fill_value=0))
    rate = crit / n_subj
    med = float(rate.median())
    hit = (crit >= p.get("min_critical", 5)) & (rate >= p.get("ratio_to_median", 3.0) * med)
    return _meta([dict(level="site", entity_id=s, site_id=s, score=round(float(rate[s]), 3),
                       detail=f"{int(crit[s])} critical deviations / {int(n_subj[s])} subjects "
                              f"(rate {rate[s]:.2f} vs median {med:.2f})")
                  for s in rate.index[hit]])


META_TYPES: dict[str, Callable] = {
    "repeat_subject": repeat_subject,
    "site_hotspot": site_hotspot,
}
