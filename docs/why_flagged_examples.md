# 'Why flagged' cards: examples

## RULE

```text
[CRITICAL] Age outside eligibility range — subject S01-005
Why flagged: Rule ELIG-AGE (PD) is violated; severity critical.
Evidence:
  1. age = 83, required >= 18 and <= 75
  2. Observed: 83 | Expected: >= 18 and <= 75
Reference: Protocol: inclusion criteria; ICH E6 4.5.1
Suggested next step: Escalate to the study lead, verify against source documents and file an important protocol-deviation report.
```

```text
[CRITICAL] Age outside eligibility range — subject S01-021
Why flagged: Rule ELIG-AGE (PD) is violated; severity critical.
Evidence:
  1. age = 80, required >= 18 and <= 75
  2. Observed: 80 | Expected: >= 18 and <= 75
Reference: Protocol: inclusion criteria; ICH E6 4.5.1
Suggested next step: Escalate to the study lead, verify against source documents and file an important protocol-deviation report.
```

```text
[CRITICAL] Late SAE report — subject S01-022
Why flagged: Rule SAFE-SAE-01 (SAE) is violated; severity critical.
Evidence:
  1. SAE 'Acute kidney injury' onset 2025-08-07 reported after 11 day(s) (deadline 1)
  2. Observed: 11 days | Expected: <= 1 day(s)
Reference: ICH E6 4.11.1; 21 CFR 312.64(b)
Suggested next step: Escalate to the study lead, verify against source documents and file an important protocol-deviation report.
```

## ML

```text
[MAJOR] Unusual laboratory values — subject S08-054, visit 12
Why flagged: Among the top 0.01% most unusual of 7,024 visits; 3 of 3 detectors independently rate it as extreme.
Evidence:
  1. Creatinine = 4.04 mg/dL (typical across the trial: 0.88; +20.3 robust SD)
  2. Hemoglobin is +0.9 g/dL away from this subject's own typical value (+3.0 robust SD)
  3. Glucose is -3.6 mg/dL away from this subject's own typical value (-0.9 robust SD)
  4. Detector percentile ranks: Isolation Forest 1.000, Mahalanobis 1.000, LOF 1.000
Reference: ICH E6 4.9.1 (accurate, verifiable data)
Suggested next step: Check the source document; if it is a transcription or unit error, raise a data query. If the value is genuine, assess clinical significance.
```

```text
[MAJOR] Unusual laboratory values — subject S03-045, visit 12
Why flagged: Among the top 0.03% most unusual of 7,024 visits; 3 of 3 detectors independently rate it as extreme.
Evidence:
  1. Creatinine = 6.08 mg/dL (typical across the trial: 0.88; +33.4 robust SD)
  2. Hemoglobin is +0.3 g/dL away from this subject's own typical value (+1.0 robust SD)
  3. Glucose = 83.3 mg/dL (typical across the trial: 93.0; -0.9 robust SD)
  4. Detector percentile ranks: Isolation Forest 0.999, Mahalanobis 1.000, LOF 1.000
Reference: ICH E6 4.9.1 (accurate, verifiable data)
Suggested next step: Check the source document; if it is a transcription or unit error, raise a data query. If the value is genuine, assess clinical significance.
```

```text
[MAJOR] Unusual laboratory values — subject S04-051, visit 12
Why flagged: Among the top 0.04% most unusual of 7,024 visits; 3 of 3 detectors independently rate it as extreme.
Evidence:
  1. Creatinine = 4.28 mg/dL (typical across the trial: 0.88; +21.8 robust SD)
  2. Hemoglobin is -0.4 g/dL away from this subject's own typical value (-1.3 robust SD)
  3. Glucose = 104 mg/dL (typical across the trial: 93.0; +1.1 robust SD)
  4. Detector percentile ranks: Isolation Forest 0.999, Mahalanobis 1.000, LOF 1.000
Reference: ICH E6 4.9.1 (accurate, verifiable data)
Suggested next step: Check the source document; if it is a transcription or unit error, raise a data query. If the value is genuine, assess clinical significance.
```

## FORENSIC

```text
[CRITICAL] Possible data-integrity issue — subject S04-015
Why flagged: Fabrication-suspicion score 0.92 (HIGH tier); main pattern: possible duplicate subject. This is statistical suspicion, not proof.
Evidence:
  1. near-duplicate of S04-056 (similarity 0.91)
Reference: ICH E6 4.9.1; data-integrity (ALCOA+) principles
Suggested next step: Compare identity/source documents of both subjects and check for duplicate enrolment; contact the site PI.
```

```text
[CRITICAL] Possible data-integrity issue — subject S01-028
Why flagged: Fabrication-suspicion score 0.90 (HIGH tier); main pattern: possible duplicate subject. This is statistical suspicion, not proof.
Evidence:
  1. near-duplicate of S01-059 (similarity 0.94)
Reference: ICH E6 4.9.1; data-integrity (ALCOA+) principles
Suggested next step: Compare identity/source documents of both subjects and check for duplicate enrolment; contact the site PI.
```

```text
[CRITICAL] Possible data-integrity issue — subject S01-059
Why flagged: Fabrication-suspicion score 0.90 (HIGH tier); main pattern: possible duplicate subject. This is statistical suspicion, not proof.
Evidence:
  1. near-duplicate of S01-028 (similarity 0.94)
Reference: ICH E6 4.9.1; data-integrity (ALCOA+) principles
Suggested next step: Compare identity/source documents of both subjects and check for duplicate enrolment; contact the site PI.
```

## RISK

```text
[CRITICAL] Site S08: full audit recommended
Why flagged: Bayesian risk index 0.94 (95% CI 0.55-1.29).
Evidence:
  1. risk index 0.94 [95% CI 0.55-1.29]; P(index >= 0.60) = 0.96
  2. Main drivers: Critical deviation rate: 23.8% vs peers 4.6%; Protocol deviation rate: 51.6% vs peers 9.3%; Missing visits: 2.1% vs peers 1.0%
Reference: ICH E6(R2) 5.0.1 / 5.18: risk-based monitoring
Suggested next step: For-cause audit: full on-site audit (source-data verification, consent forms, SAE chain, data-integrity review); consider pausing enrolment until the review is complete.
```

```text
[CRITICAL] Site S06: full audit recommended
Why flagged: Bayesian risk index 0.74 (95% CI 0.51-0.89).
Evidence:
  1. critical-class KRI far above peers (forensic_rate); risk index 0.74 [95% CI 0.51-0.89]; P(index >= 0.60) = 0.91
  2. Main drivers: Forensic suspicion rate: 40.3% vs peers 5.6%; Critical deviation rate: 6.1% vs peers 4.6%
Reference: ICH E6(R2) 5.0.1 / 5.18: risk-based monitoring
Suggested next step: For-cause audit: full on-site audit (source-data verification, consent forms, SAE chain, data-integrity review); consider pausing enrolment until the review is complete.
```
