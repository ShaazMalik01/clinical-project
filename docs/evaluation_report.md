# Evaluation report

Evaluated subjects: 575 (210 with at least one injected anomaly: PD 103, OUT 43, FAB 58, SAE 20). Main dataset seed 42; thresholds fixed before evaluation.

## 1. Overall performance (any anomaly)

| system | n_flagged | precision | recall | f1 | auc | ap |
|---|---|---|---|---|---|---|
| rules-only | 145 | 1.000 | 0.690 | 0.817 | 0.845 | 0.804 |
| ML-only | 156 | 0.647 | 0.481 | 0.552 | 0.653 | 0.614 |
| forensics-only | 58 | 0.983 | 0.271 | 0.425 | 0.642 | 0.558 |
| hybrid (all three) | 266 | 0.789 | 1.000 | 0.882 | 0.990 | 0.984 |

Bootstrap 95% intervals (subjects resampled):

| system | f1 | f1_lo | f1_hi | auc | auc_lo | auc_hi |
|---|---|---|---|---|---|---|
| rules-only | 0.817 | 0.772 | 0.857 | 0.845 | 0.814 | 0.875 |
| ML-only | 0.552 | 0.487 | 0.604 | 0.653 | 0.598 | 0.702 |
| forensics-only | 0.425 | 0.339 | 0.499 | 0.642 | 0.609 | 0.673 |
| hybrid (all three) | 0.882 | 0.848 | 0.909 | 0.990 | 0.984 | 0.995 |

## 2. Per anomaly category

Precision and F1 here are measured against clean subjects only; AUC separates the category's subjects from clean ones.

| category | system | n_pos | precision | recall | f1 | auc |
|---|---|---|---|---|---|---|
| PD | rules-only | 103 | 1.000 | 1.000 | 1.000 | 1.000 |
| PD | ML-only | 103 | 0.368 | 0.311 | 0.337 | 0.574 |
| PD | forensics-only | 103 | 0.000 | 0.000 | 0.000 | 0.502 |
| PD | hybrid (all three) | 103 | 0.648 | 1.000 | 0.786 | 0.983 |
| OUT | rules-only | 43 | 1.000 | 0.791 | 0.883 | 0.895 |
| OUT | ML-only | 43 | 0.439 | 1.000 | 0.610 | 0.996 |
| OUT | forensics-only | 43 | 0.000 | 0.000 | 0.000 | 0.528 |
| OUT | hybrid (all three) | 43 | 0.434 | 1.000 | 0.606 | 0.998 |
| FAB | rules-only | 58 | 1.000 | 0.034 | 0.067 | 0.517 |
| FAB | ML-only | 58 | 0.345 | 0.500 | 0.408 | 0.593 |
| FAB | forensics-only | 58 | 0.983 | 0.983 | 0.983 | 0.991 |
| FAB | hybrid (all three) | 58 | 0.509 | 1.000 | 0.674 | 0.995 |
| SAE | rules-only | 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAE | ML-only | 20 | 0.068 | 0.200 | 0.101 | 0.457 |
| SAE | forensics-only | 20 | 0.667 | 0.100 | 0.174 | 0.559 |
| SAE | hybrid (all three) | 20 | 0.263 | 1.000 | 0.417 | 1.000 |

## 3. Recall for each injected anomaly type

| subtype | n | rules-only | ML-only | forensics-only | hybrid (all three) |
|---|---|---|---|---|---|
| copy_forward | 15 | 0.000 | 0.667 | 1.000 | 1.000 |
| digit_preference | 15 | 0.000 | 0.800 | 0.933 | 1.000 |
| duplicate_subject | 16 | 0.125 | 0.062 | 1.000 | 1.000 |
| ineligible_enrolled | 15 | 1.000 | 0.800 | 0.000 | 1.000 |
| late_sae_report | 20 | 1.000 | 0.200 | 0.100 | 1.000 |
| low_variance | 12 | 0.000 | 0.500 | 1.000 | 1.000 |
| missed_visit | 26 | 1.000 | 0.192 | 0.000 | 1.000 |
| procedure_before_consent | 15 | 1.000 | 0.333 | 0.000 | 1.000 |
| value_outlier | 43 | 0.791 | 1.000 | 0.000 | 1.000 |
| visit_out_of_window | 35 | 1.000 | 0.229 | 0.000 | 1.000 |
| wrong_dose | 25 | 1.000 | 0.120 | 0.000 | 1.000 |

## 4. Ablation: every combination of detectors

| components | k | n_flagged | precision | recall | f1 | auc | recall_PD | recall_OUT | recall_FAB | recall_SAE |
|---|---|---|---|---|---|---|---|---|---|---|
| rules | 1 | 145 | 1.000 | 0.690 | 0.817 | 0.845 | 1.000 | 0.791 | 0.034 | 1.000 |
| ML | 1 | 156 | 0.647 | 0.481 | 0.552 | 0.653 | 0.311 | 1.000 | 0.500 | 0.200 |
| forensics | 1 | 58 | 0.983 | 0.271 | 0.425 | 0.642 | 0.000 | 0.000 | 0.983 | 0.100 |
| rules+ML | 2 | 238 | 0.769 | 0.871 | 0.817 | 0.885 | 1.000 | 1.000 | 0.534 | 1.000 |
| rules+forensics | 2 | 201 | 0.995 | 0.952 | 0.973 | 0.975 | 1.000 | 0.791 | 0.983 | 1.000 |
| ML+forensics | 2 | 186 | 0.699 | 0.619 | 0.657 | 0.763 | 0.311 | 1.000 | 1.000 | 0.300 |
| rules+ML+forensics | 3 | 266 | 0.789 | 1.000 | 0.882 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |

Marginal value of each component (hybrid minus the hybrid without it):

| component | recall_gain | precision_change | f1_gain | auc_gain |
|---|---|---|---|---|
| rules | 0.381 | 0.091 | 0.226 | 0.227 |
| ML | 0.048 | -0.206 | -0.091 | 0.015 |
| forensics | 0.129 | 0.021 | 0.065 | 0.105 |

## 5. Does the hybrid win, and why?

**Recall.** The hybrid finds 100% of the anomalous subjects; the best single detector (rules-only) finds 69%. A union can only add recall, so this is expected, not a surprise.

**F1 against the strongest single detector (rules-only).** Difference +0.065 (95% interval +0.012 to +0.118); this is a statistically clear improvement (95% interval above zero).

Versus ML-only: F1 difference +0.330 (95% interval +0.280 to +0.390).

Versus forensics-only: F1 difference +0.457 (95% interval +0.373 to +0.545).

**The price.** Precision goes from 1.00 (rules-only) to 0.79 for the hybrid (-0.21), because every detector's false alarms are added together. Whether that trade is worth it depends on how costly a missed anomaly is compared with a wasted review.

**Why combining helps.** 117 of 210 anomalous subjects (56%) are caught by exactly one detector, so removing any detector loses real findings; 0 (0%) are missed by all three.

- protocol deviation (n=103): best single is rules-only at 100%; hybrid 100%.
- value outlier (n=43): best single is ML-only at 100%; hybrid 100%.
- fabrication (n=58): best single is forensics-only at 98%; hybrid 100%.
- late SAE report (n=20): best single is rules-only at 100%; hybrid 100%.
**Honest caveat.** The F1-best combination is rules+forensics (F1 0.973), higher than the full hybrid (0.882). Adding ML changes recall by +0.048 and precision by -0.206 at the configured thresholds. So the full hybrid is the recall-best choice, not the F1-best one; the right choice depends on the relative cost of a missed anomaly and a wasted review, and on the review budget given to ML (see the sensitivity section).

**Most valuable component by recall:** rules (removing it loses 38.1% recall).


Which detectors caught each anomalous subject:

| index | subjects | share |
|---|---|---|
| rules | 80 | 0.381 |
| rules+ML | 63 | 0.300 |
| ML+forensics | 28 | 0.133 |
| forensics | 27 | 0.129 |
| ML | 10 | 0.048 |
| rules+forensics | 2 | 0.010 |

By category:

| index | PD | OUT | FAB | SAE |
|---|---|---|---|---|
| ML | 0 | 9 | 1 | 0 |
| ML+forensics | 0 | 0 | 28 | 0 |
| forensics | 0 | 0 | 27 | 0 |
| rules | 71 | 0 | 0 | 14 |
| rules+ML | 32 | 34 | 0 | 4 |
| rules+forensics | 0 | 0 | 2 | 2 |

## 6. Same workload for everyone

Each system reviews its own top b% of subjects.

| system | budget | k | precision_at_k | recall_at_k |
|---|---|---|---|---|
| rules-only | 0.050 | 29 | 1.000 | 0.138 |
| rules-only | 0.100 | 58 | 1.000 | 0.276 |
| rules-only | 0.150 | 87 | 1.000 | 0.414 |
| rules-only | 0.200 | 115 | 1.000 | 0.548 |
| ML-only | 0.050 | 29 | 1.000 | 0.138 |
| ML-only | 0.100 | 58 | 0.879 | 0.243 |
| ML-only | 0.150 | 87 | 0.736 | 0.305 |
| ML-only | 0.200 | 115 | 0.617 | 0.338 |
| forensics-only | 0.050 | 29 | 1.000 | 0.138 |
| forensics-only | 0.100 | 58 | 0.983 | 0.271 |
| forensics-only | 0.150 | 87 | 0.759 | 0.314 |
| forensics-only | 0.200 | 115 | 0.643 | 0.352 |
| hybrid (all three) | 0.050 | 29 | 1.000 | 0.138 |
| hybrid (all three) | 0.100 | 58 | 1.000 | 0.276 |
| hybrid (all three) | 0.150 | 87 | 1.000 | 0.414 |
| hybrid (all three) | 0.200 | 115 | 1.000 | 0.548 |

Fusion rule:

| fusion | tp | fp | fn | precision | recall | f1 | n_flagged |
|---|---|---|---|---|---|---|---|
| any detector (OR) | 210 | 56 | 0 | 0.789 | 1.000 | 0.882 | 266 |
| at least 2 agree | 93 | 0 | 117 | 1.000 | 0.443 | 0.614 | 93 |
| all 3 agree | 0 | 0 | 210 |  | 0.000 | 0.000 | 0 |

## 7. Sensitivity

ML review budget (hybrid):

| visit_q | subject_q | precision | recall | f1 |
|---|---|---|---|---|
| 0.005 | 0.050 | 0.972 | 0.976 | 0.974 |
| 0.005 | 0.100 | 0.940 | 0.976 | 0.958 |
| 0.005 | 0.150 | 0.899 | 0.976 | 0.936 |
| 0.005 | 0.200 | 0.844 | 0.981 | 0.907 |
| 0.010 | 0.050 | 0.925 | 1.000 | 0.961 |
| 0.010 | 0.100 | 0.897 | 1.000 | 0.946 |
| 0.010 | 0.150 | 0.861 | 1.000 | 0.925 |
| 0.010 | 0.200 | 0.814 | 1.000 | 0.897 |
| 0.020 | 0.050 | 0.811 | 1.000 | 0.896 |
| 0.020 | 0.100 | 0.789 | 1.000 | 0.882 |
| 0.020 | 0.150 | 0.761 | 1.000 | 0.864 |
| 0.020 | 0.200 | 0.727 | 1.000 | 0.842 |
| 0.030 | 0.050 | 0.742 | 1.000 | 0.852 |
| 0.030 | 0.100 | 0.724 | 1.000 | 0.840 |
| 0.030 | 0.150 | 0.700 | 1.000 | 0.824 |
| 0.030 | 0.200 | 0.677 | 1.000 | 0.808 |
| 0.050 | 0.050 | 0.654 | 1.000 | 0.791 |
| 0.050 | 0.100 | 0.644 | 1.000 | 0.784 |
| 0.050 | 0.150 | 0.625 | 1.000 | 0.769 |
| 0.050 | 0.200 | 0.610 | 1.000 | 0.758 |

Forensic tier cut-offs (hybrid):

| high_q | medium_q | flag_tiers | precision | recall | f1 |
|---|---|---|---|---|---|
| 0.010 | 0.100 | HIGH | 0.788 | 0.971 | 0.870 |
| 0.010 | 0.100 | HIGH+MEDIUM | 0.792 | 1.000 | 0.884 |
| 0.010 | 0.200 | HIGH | 0.788 | 0.971 | 0.870 |
| 0.010 | 0.200 | HIGH+MEDIUM | 0.789 | 1.000 | 0.882 |
| 0.010 | 0.300 | HIGH | 0.788 | 0.971 | 0.870 |
| 0.010 | 0.300 | HIGH+MEDIUM | 0.789 | 1.000 | 0.882 |
| 0.050 | 0.100 | HIGH | 0.792 | 1.000 | 0.884 |
| 0.050 | 0.100 | HIGH+MEDIUM | 0.792 | 1.000 | 0.884 |
| 0.050 | 0.200 | HIGH | 0.792 | 1.000 | 0.884 |
| 0.050 | 0.200 | HIGH+MEDIUM | 0.789 | 1.000 | 0.882 |
| 0.050 | 0.300 | HIGH | 0.792 | 1.000 | 0.884 |
| 0.050 | 0.300 | HIGH+MEDIUM | 0.789 | 1.000 | 0.882 |
| 0.100 | 0.100 | HIGH | 0.792 | 1.000 | 0.884 |
| 0.100 | 0.100 | HIGH+MEDIUM | 0.792 | 1.000 | 0.884 |
| 0.100 | 0.200 | HIGH | 0.792 | 1.000 | 0.884 |
| 0.100 | 0.200 | HIGH+MEDIUM | 0.789 | 1.000 | 0.882 |
| 0.100 | 0.300 | HIGH | 0.792 | 1.000 | 0.884 |
| 0.100 | 0.300 | HIGH+MEDIUM | 0.789 | 1.000 | 0.882 |

Rule severity threshold:

| rules_flag_from | system | precision | recall | f1 |
|---|---|---|---|---|
| minor | rules-only | 1.000 | 0.690 | 0.817 |
| minor | hybrid (all three) | 0.789 | 1.000 | 0.882 |
| major | rules-only | 1.000 | 0.671 | 0.803 |
| major | hybrid (all three) | 0.787 | 0.986 | 0.875 |
| critical | rules-only | 1.000 | 0.271 | 0.427 |
| critical | hybrid (all three) | 0.743 | 0.771 | 0.757 |

Across the ML budget grid the hybrid F1 ranges from 0.76 to 0.97 (spread 0.22); across forensic cut-offs from 0.87 to 0.88.

## 8. Site level

Rank of each truly bad site (1 = riskiest) by each system.

| bad_site_type | system | rank | of | auc | tier |
|---|---|---|---|---|---|
| sloppy_process | rules-only | 1 | 12 | 1.000 |  |
| sloppy_process | ML-only | 2 | 12 | 0.909 |  |
| sloppy_process | forensics-only | 4 | 12 | 0.364 |  |
| sloppy_process | hybrid (Bayesian risk model) | 1 | 12 | 1.000 | FULL_AUDIT |
| fabrication | rules-only | 3 | 12 | 0.818 |  |
| fabrication | ML-only | 1 | 12 | 1.000 |  |
| fabrication | forensics-only | 1 | 12 | 1.000 |  |
| fabrication | hybrid (Bayesian risk model) | 2 | 12 | 0.909 | FULL_AUDIT |

## 9. Replicate datasets (not used for any tuning)

| system | precision main | precision replicates | recall main | recall replicates | f1 main | f1 replicates | auc main | auc replicates |
|---|---|---|---|---|---|---|---|---|
| rules-only | 1.000 | 1.000 +/- 0.000 | 0.690 | 0.693 +/- 0.017 | 0.817 | 0.818 +/- 0.012 | 0.845 | 0.846 +/- 0.009 |
| ML-only | 0.647 | 0.578 +/- 0.037 | 0.481 | 0.441 +/- 0.037 | 0.552 | 0.501 +/- 0.038 | 0.653 | 0.623 +/- 0.012 |
| forensics-only | 0.983 | 1.000 +/- 0.000 | 0.271 | 0.270 +/- 0.004 | 0.425 | 0.425 +/- 0.005 | 0.642 | 0.625 +/- 0.001 |
| hybrid (all three) | 0.789 | 0.757 +/- 0.013 | 1.000 | 1.000 +/- 0.000 | 0.882 | 0.862 +/- 0.008 | 0.990 | 0.983 +/- 0.006 |

# Limitations (read this before quoting any number)

**1. The data is synthetic.** Every result comes from a simulator we wrote and anomalies we injected.
No real clinical-trial data was used, and nothing here has been validated on a real trial. Real data has
transcription noise, missing values, mixed units, protocol amendments and audit trails that this simulator
does not model.

**2. Circularity.** The same person designed the anomalies and the detectors. Copy-forward, digit
preference and low variability were injected *because* the forensic tests look for them, so forensic recall
is optimistic. Real fabrication is subtler and adapts to known checks. A fair test needs anomalies
written by someone who has not seen the detectors.

**3. Small sample, high prevalence.** 575 evaluated subjects at 12 sites, 210 of them anomalous
(37%). Real trials have far fewer true problems, so real precision would be much lower
at the same recall. Confidence intervals come from bootstrapping subjects; they do not capture
simulator misspecification. Only two sites are 'bad', so site-level metrics rest on two examples.

**4. Thresholds were set on the main dataset.** Contamination rates, tier cut-offs and fusion rules were
chosen on seed 42. The replicate seeds in the multi-seed table were not used for any tuning, which is why
they are the more honest numbers.

**5. Hybrid design choices are ours.** The evidence weights used to fuse detectors (see the module
docstring) are fixed by judgement, not learned. Union fusion raises recall by construction; the
interesting question is what it costs in precision, which the ablation table shows.

**6. Subject-level evaluation hides record-level errors.** A subject counts as detected if the detector
flags the subject, even if it points at a different visit than the injected anomaly.

**7. The monitor in the feedback loop is simulated.** Real reviewers disagree, tire, and are biased by the
queue order.

**8. Not a validated GxP system.** A real deployment needs computerised-system validation (GAMP 5),
change control for model updates, access control, and electronic-signature controls beyond this hash-chained
log.

**9. A flag is a reason to look, not a finding of misconduct.** Statistical suspicion must go through
human review and documented follow-up before any conclusion about a site or person.
