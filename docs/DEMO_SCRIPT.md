# 2-minute demo script

**Before you start (5 minutes earlier):** `python -m streamlit run app/streamlit_app.py`, browser at 100% zoom (full screen),
sidebar visible, reviewer name set to your name, "Show synthetic ground truth" OFF. Do one dry run. Close other tabs.

| Time | Do | Say |
|---|---|---|
| 0:00 | Overview page, hero visible | "Trials must follow a protocol and keep trustworthy data. Visiting every site is expensive, so this system tells a monitor where to look first, and why. The data is synthetic, so I can check it against known truth." |
| 0:15 | Point at the 'Where to act first' tiles (S08, S06) | "It combines rules, machine learning, forensics and a Bayesian risk model. Two sites need a for-cause audit, and the other ten stay routine." |
| 0:30 | Click the **3D risk city** tab, rotate once | "Each tower is one site and one indicator. Towers through the amber plane are above the tolerated multiple of their peers." |
| 0:45 | Click **Open site detail** on S08, show recommendation and evidence | "Many protocol problems stand out here: consent, eligibility, dosing, late safety reports. Intervals are credible intervals, so small sites are not punished unfairly." |
| 1:00 | Go to **Anomalies**, open a card for an ML visit alert | "Every alert explains itself: what is unusual, how far from normal, and what to check. We tested these explanations: the top feature matched the corrupted field 45 of 45 times." |
| 1:15 | Go to **Forensics**, select S06, show the last-digit chart | "This site looks different: invented data piles up on last digits 0 and 5, and its CRP values break Benford's law. Rules cannot see this at all." |
| 1:30 | Go to **Feedback and triage**, click **True issue** once | "A monitor's decision trains the ranking and goes straight into the audit log." |
| 1:40 | Go to **Audit trail**, press **Run tamper simulation** | "Everything is hash-chained. I attack copies of the log eleven ways, and every attack is caught. Two are shown without the extra protection on purpose, to show what a bare chain cannot catch." |
| 1:55 | Look at the camera or panel | "On results: the hybrid has F1 0.882 against 0.817 for the best single detector. It is synthetic data, so it shows the design works, not that it is validated on real trials." |

## If something breaks
* Dashboard will not start: run `python run_all.py`, then retry. Still no: show `docs/phase10_main.png`, `docs/why_flagged_examples.md` and `docs/evaluation_report.md`.
* 3D view is blank: use the **2D heatmap** radio on *Site risk*, and mention WebGL is disabled.
* Audit page says BROKEN: someone edited the log; run `python build_dashboard_data.py --reset-audit`.

## Three questions to be ready for
1. "Is hybrid really better?" Say: better than every single detector, but rules plus forensics has the best F1 (0.973); ML buys recall at a precision cost.
2. "Real data?" Say: no, synthetic, with the circularity caveat.
3. "Why can I trust the audit log?" Say: it is tamper-evident, with an anchor and optional HMAC; not a validated GxP system.
