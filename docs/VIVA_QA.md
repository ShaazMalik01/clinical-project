# 25 viva questions with strong answers

Numbers are from the shipped run (seed 42). Say them confidently, and always add the limitation when it is relevant.

**1. What problem does your project solve?**
Clinical trials must follow a protocol and Good Clinical Practice, and the data must be reliable. Checking every record at every site is slow and expensive, so regulators encourage risk-based monitoring. My system ranks records, subjects and sites by evidence of non-compliance or unreliable data, explains each alert in plain English, and recommends an action: routine monitoring, remote review, targeted on-site visit or for-cause audit.

**2. Why a hybrid? Why not just machine learning?**
Each method sees something the others cannot. Rules are exact but only catch what someone wrote (0 of 12 plausible-but-extreme values, 0 fabrication). Forensics catch fabrication fingerprints but nothing else. ML catches unusual values and profiles but has false alarms. On 575 evaluated subjects the hybrid reached F1 0.882 versus 0.817 for the best single detector (difference +0.065, 95% interval +0.012 to +0.118) with recall 1.000, and 56% of anomalous subjects are caught by exactly one detector. The honest caveat: the best F1 among all seven combinations is rules plus forensics (0.973), because ML adds 0.048 recall but costs 0.206 precision at my review budget. So the hybrid is the recall-best choice, and which is better depends on the cost of a miss versus a wasted review.

**3. Why Bayesian for site risk?**
A site KRI is "x events out of n opportunities", which is Binomial, and the Beta prior is conjugate, so the posterior is a Beta and I get credible intervals and probabilities like P(index above 0.6) directly. The key benefit is shrinkage: a small site with 2 events out of 6 should not be judged as 33%. In a 150-replicate simulation, rate error on small clean sites fell from 0.093 to 0.056 and false alarms from 16.7% to 8.7%, with 94% interval coverage.

**4. How do you avoid false positives?**
Several layers. Rules are deterministic and cite a reference; on clean data they produce 0 deviations. Forensic p-values use Benjamini-Hochberg false-discovery control and empirical nulls estimated from the data; on clean data there are 0 HIGH-tier subjects and 0 duplicate pairs. ML flags only a fixed review budget (top 2% of visits, 10% of subjects). Site risk uses shrinkage and credible intervals, and all-clean data gives all sites ROUTINE. Finally a human reviews every alert, and feedback adapts the threshold.

**5. Why synthetic data?**
Real trial data is confidential (GDPR, HIPAA) and has no ground truth, because we never know which records are truly wrong. With a simulator I know exactly which anomalies I injected, so I can measure precision and recall honestly. The cost is realism, which I state as the main limitation.

**6. Isn't your evaluation circular, since you injected anomalies your detectors look for?**
Yes, partly, and I say so. Copy-forward, digit preference and low variability were injected because forensic tests target them, so forensic recall is optimistic. Hybrid recall of 1.000 is a symptom of that. I reduced the damage by keeping threshold choices fixed, evaluating two untouched replicate datasets (hybrid F1 0.862 +/- 0.008), and reporting uncertainty. A convincing test would use anomalies written by someone who has not seen the detectors.

**7. Explain your three ML detectors. Why three?**
Isolation Forest isolates points with random splits; anomalies need fewer splits. LOF compares a point's local density with its neighbours. Mahalanobis distance measures distance from the centre using the correlation structure. They have different blind spots, so agreement raises confidence; the `votes` field says how many detectors independently rate an item extreme. At visit level the ensemble has AUC 0.999 on value outliers.

**8. Why robust covariance (MCD)?**
Ordinary covariance is inflated by the very outliers we want to find, which hides them (masking). MCD fits on the most compact 75% of the data. A unit test plants a 10% outlier cluster and shows MCD still sees it far from the core.

**9. How did you choose ensemble weights and thresholds?**
Weights 0.40/0.35/0.25 come from priors: Isolation Forest is the most stable, Mahalanobis the most interpretable, LOF the noisiest. They are label-free, because real systems have no labels. I also offer a consensus option and a weight-grid sensitivity check. The 2% and 10% thresholds are review budgets, an operational decision, and the sensitivity analysis shows they matter: hybrid F1 ranges from 0.76 to 0.97 across the ML budget grid.

**10. Why is Benford's law only applied to CRP?**
Benford needs data spread over at least two orders of magnitude. Blood pressure and routine labs are not, so a Benford test on them would flag honest sites. My code checks the range (P99 divided by P1 at least 100), the sample size, and that most sites conform before using the test. Only CRP passes, and it uses one value per subject because repeated measures are not independent.

**11. What are terminal-digit preference and the variance test?**
Real readings have roughly uniform last digits; people inventing or rounding numbers favour 0 and 5. I test each subject's share of 0/5 endings against the typical share with a binomial test. The variance test asks whether a subject's visit-to-visit variability is implausibly low after removing a trend. Both are calibrated against the other subjects, not against theory.

**12. How do copy-forward and duplicate detection stay precise?**
Five continuous measurements matching exactly at two visits has a chance probability near 1 in 100,000, so an exact match is strong evidence. For duplicates I first block on demographics (sex, age, height, BMI), then compare vitals and labs relative to how different unrelated subjects are. Result: all 8 duplicate pairs found with precision 1.000 and recall 1.000, and copy-forward recall 15 of 15 subjects.

**13. Explain the Beta-Binomial model and the outlier component.**
Each site's true rate theta has a Beta prior fitted from all sites by maximum marginal likelihood. If two bad sites distort that prior, or strong shrinkage hides them, the model fails. So the prior is a mixture: most sites follow the shared Beta, with probability 0.15 a site follows a flat Beta(1,1). The data decide per site, which keeps small sites protected and real problem sites unshrunk.

**14. How are tiers and recommendations decided?**
A weighted excess-risk index (rate above 1.5 times the peer rate, capped) is computed from posterior draws. Costlier actions need more evidence: remote review needs P(index at least 0.15) of 0.5, on-site 0.30 at 0.7, audit 0.60 at 0.8. Critical-class indicators (critical deviations, late SAE, forensic suspicion) can trigger an audit alone, so a single fabrication signal is not diluted by an average. Result: both bad sites are FULL_AUDIT and the other 10 are ROUTINE.

**15. How do you know your explanations are right?**
I tested faithfulness. For every injected value outlier I know which field I corrupted, and I checked whether the card's top feature names it: 45 of 45 at rank 1. Most projects never validate explanations.

**16. Why robust z-scores instead of SHAP or deep learning?**
My detectors are distance based, so "how many robust standard deviations from typical" is exact, model-agnostic and understandable by a monitor. SHAP is a possible extension. Deep learning would need much more data and would be harder to justify and audit in a regulated setting.

**17. How does the feedback loop work?**
The monitor marks alerts true or false. A regularised logistic regression on alert features learns which signals matter and in which direction, and its ranking is blended with the unsupervised score as labels accumulate. I use some exploration so labels are not only from the top of the queue, simulate 5% monitor error, and set the review threshold by decision theory: review when P(true) times miss cost exceeds review cost (default 0.25). It improves the ranking of the unreviewed queue in simulation, most when the base ranking is weak.

**18. How does the audit trail detect tampering, and what can it not detect?**
Each entry stores the hash of the previous one, so editing, deleting, inserting or reordering breaks the chain at that point. A bare chain cannot detect truncation of the tail or a full rewrite by someone who knows the algorithm. Those are caught by an anchor (a saved head hash, ideally stored elsewhere) and by an optional HMAC key. My tamper test runs 11 attacks and shows both detected and, deliberately, undetected cases. It is tamper-evident, not tamper-proof.

**19. How does this relate to 21 CFR Part 11 and GxP?**
It borrows the principles (secure, time-stamped, attributable, append-only audit trails) but it is not a validated system. A real deployment needs computerised-system validation (GAMP 5), access control, electronic signatures and change control for any model update.

**20. Which metrics matter, and why precision, recall and F1?**
Recall is the share of real problems found; a miss in a trial can harm patients or invalidate data. Precision is the share of alerts that are real; low precision wastes monitor time and breeds alert fatigue. F1 balances the two and AUC measures ranking quality regardless of threshold. I also compare systems at equal workload (review the top b%), and report bootstrap intervals so differences are not over-read.

**21. What would change on real data?**
Prevalence of problems would be far lower, so precision would drop at the same recall. Data would be messier (missing values, units, amendments). Fabricators adapt to known checks. And I would need to recalibrate thresholds with real monitors.

**22. What is the weakest part of your project?**
The evaluation is on synthetic data with anomalies I designed, so it is optimistic. Within the system, the ML component has the lowest precision (0.647) and lowers the hybrid's F1 compared with rules plus forensics. I would address it with independently written anomalies, public datasets with seeded errors, and a smaller ML review budget.

**23. What is novel compared with a standard anomaly detector?**
It is not one model but a monitoring system: forensics with an applicability gate for Benford, small-site protection through an outlier-robust Bayesian model, explanations that are validated against ground truth, a feedback loop with a decision-theoretic threshold, and a tamper-evident audit trail with a live tamper test. It also reports its own failures: the ablation and the caveat that the full hybrid is not the F1-best combination.

**24. How would you deploy or scale it?**
Run it as a nightly batch job on the EDC export: generate alerts, write them to a queue with the why-flagged cards, and send decisions back as queries. Heavy steps are batch and vectorised; duplicate blocking would use sorting or locality-sensitive hashing for large trials. Model updates would go through change control with versioned configs, which my config-diff logging already supports.

**25. What are the ethical risks?**
A false accusation of misconduct harms real people, so the system says "suspicion, not proof", requires human review, shows its evidence, and logs every decision. Real data would need privacy protection and access control. Bias is also a concern: site comparisons should account for population differences, which my peer-based baselines only partly do.
