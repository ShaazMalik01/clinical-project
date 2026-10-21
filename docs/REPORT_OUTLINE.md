# Project report outline

Target length: 18 to 25 pages. Numbers below are from the shipped run (seed 42); regenerate with `python run_all.py --eval`.
Verify every reference against the original source before submitting.

## Title page
Clinical Trial Compliance Monitoring and Anomaly Detection System: a hybrid, explainable, auditable approach.

## Abstract (draft, about 170 words)
Clinical trials must follow a protocol and Good Clinical Practice, and monitoring this compliance by visiting every site is
expensive. We build a hybrid monitoring system on a synthetic 12-site trial with 359 injected anomaly records. It combines a
YAML rule-based expert system, three unsupervised detectors (Isolation Forest, LOF, robust Mahalanobis), data-integrity
forensics (terminal-digit preference, low variability, copy-forward, near-duplicate subjects, Benford's law with an
applicability gate) and a Bayesian Beta-Binomial site-risk model that protects small sites from unfair alarms. Every alert
carries a plain-English explanation whose top feature matched the corrupted field in 45 of 45 injected outliers, monitor
feedback adapts the review queue, and all flags, decisions and configuration changes are stored in a hash-chained audit log
that detected all attempted tampering that its protection layers cover. The hybrid reaches F1 0.882 (95% interval 0.848 to
0.909) against 0.817 for the best single detector, with recall 1.000, and it holds on replicate datasets. The best F1 overall
comes from rules plus forensics (0.973), showing that ML trades precision for recall. Results are synthetic and not validated
on real trials.

## 1. Introduction
* Context: trial compliance, protocol deviations, GCP, ICH E6, why monitoring is expensive (100% source-data verification).
* Motivation: risk-based monitoring, central statistical monitoring, data-fabrication cases.
* Objectives (3 to 4 bullets) and contributions: hybrid design, forensics with applicability gate, Bayesian small-site
  protection, validated explanations, tamper-evident audit trail, honest ablation.

## 2. Problem statement
Formal statement: given trial tables (subjects, visits, vitals, labs, dosing, adverse events, consent) and a protocol, rank
records, subjects and sites by the evidence that they violate the protocol or contain unreliable data, explain each
decision, and recommend a monitoring action. Three anomaly families: honest error, process failure, fabrication.

## 3. Literature context (about 1.5 pages)
* Regulation: ICH E6(R2), ICH E6(R3), FDA 2013 guidance on risk-based monitoring, 21 CFR Part 11.
* Fraud and statistical monitoring: Buyse et al. (1999), Venet et al. (2012); Benford's law: Nigrini (2012).
* Anomaly detection: Liu, Ting and Zhou (2008) Isolation Forest; Breunig et al. (2000) LOF; Rousseeuw and Van Driessen (1999) MCD.
* Bayesian shrinkage: Efron and Morris (1977); Gelman et al., *Bayesian Data Analysis*.
* Gap: most published tools are one technique; few combine rules, ML, forensics and risk scoring with explanations and an audit trail.

## 4. System design and methodology
4.1 Architecture (use the Mermaid diagram from the README; one paragraph per layer).
4.2 Synthetic data: tables, realistic structure, 11 injected anomaly types plus the CRP lab, two bad sites (why synthetic).
4.3 Rule engine: knowledge base, rule types, meta-rules, ICH references, extensibility (new rule = YAML entry).
4.4 ML ensemble: features at three levels, why three detectors, robust covariance (masking), rank normalisation, weights 0.40/0.35/0.25, review-budget thresholds.
4.5 Forensics: each test with its null hypothesis; Benford gate; evidence scoring `s = -log10 p`, noisy-OR; FDR control.
4.6 Bayesian site risk: KRIs, Beta-Binomial with outlier component, credible intervals, risk index, tier rules and critical triggers.
4.7 Explainability: robust z-scores, card structure, fidelity test against the corrupted field.
4.8 Human in the loop: logistic learner, exploration, label noise, decision-theoretic threshold `p* = review cost / miss cost`.
4.9 Audit trail: hash chain, anchor, HMAC, config diffs, tamper suite.
4.10 Dashboard: pages and design choices (include screenshots you take yourself: Overview, 3D risk city, a why-flagged card, Audit tamper test).

## 5. Experimental setup
Dataset sizes, prevalence (210 anomalous of 575 evaluated subjects), unit of evaluation (subject), definitions of flags,
metrics (precision, recall, F1, ROC-AUC, AP), bootstrap, ablation lattice, replicate seeds 101 and 202 (no tuning on them).

## 6. Results
| Table or figure | Source |
|---|---|
| Per-phase sanity checks (clean data silent, recall of rules, forensics, bad sites ranked) | README section 4 |
| Overall metrics with intervals | `docs/evaluation_report.md` section 1 |
| Per-category precision, recall, F1, AUC | section 2 |
| Ablation of all 7 combinations and marginal value | section 4 |
| Which detector caught which anomaly | section 5 (coverage table) |
| Equal-workload comparison | section 6 |
| Sensitivity (ML budget, forensic cut-offs, rule severity, fusion rule) | section 7, `docs/phase10_sensitivity.png` |
| Site-level ranks | section 8 |
| Replicates | section 9 |
| Main figure | `docs/phase10_main.png` |
| Audit tamper results | `docs/phase8_tamper_results.csv` |

Discussion points to write: rules are precise but blind to fabrication; forensics are precise but narrow; ML adds
recall on plausible-but-extreme values and profile anomalies at a precision cost; the hybrid wins against every single
detector but not against rules plus forensics on F1; the ML review budget moves hybrid F1 between 0.76 and 0.97.

## 7. Limitations (copy the honest version from `docs/LIMITATIONS.md`)
Synthetic data; circularity (we designed both anomalies and detectors; hybrid recall 1.000 is a symptom of this); high
prevalence; thresholds tuned on the main dataset; fusion weights set by judgement; subject-level scoring; simulated monitor;
not a validated GxP system.

## 8. Future work
Real or semi-real data (public trial data sets with expert-seeded errors); anomalies written by someone who has not seen the
detectors; adversarial fabrication that adapts; EDC query integration; multivariate and temporal models; calibrated
probabilities with conformal prediction; model-change control; user study with real monitors.

## 9. Ethics and regulatory notes
Privacy and consent (synthetic data); risk of false accusations; human review; transparency of rules and explanations;
audit trail principles; what validation (GAMP 5) a real deployment would need.

## 10. Conclusion
Three sentences: what was built, what the evidence shows, what it does not show.

## References (verify before use)
1. ICH E6(R2), Guideline for Good Clinical Practice, 2016; ICH E6(R3), finalised 2025.
2. U.S. FDA, *Guidance for Industry: Oversight of Clinical Investigations, A Risk-Based Approach to Monitoring*, 2013.
3. 21 CFR Part 11, Electronic Records; Electronic Signatures.
4. Buyse M. et al., "The role of biostatistics in the prevention, detection and treatment of fraud in clinical trials", *Statistics in Medicine* 18 (1999).
5. Venet D. et al., "Statistical monitoring to detect centers with atypical data patterns in multicenter clinical trials", *Clinical Trials* 9 (2012).
6. Nigrini M., *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*, Wiley, 2012.
7. Liu F. T., Ting K. M., Zhou Z.-H., "Isolation Forest", IEEE ICDM, 2008.
8. Breunig M. et al., "LOF: Identifying density-based local outliers", ACM SIGMOD, 2000.
9. Rousseeuw P., Van Driessen K., "A fast algorithm for the minimum covariance determinant estimator", *Technometrics* 41 (1999).
10. Efron B., Morris C., "Stein's paradox in statistics", *Scientific American* 236 (1977).
11. Gelman A. et al., *Bayesian Data Analysis*, 3rd ed., CRC Press, 2013.
