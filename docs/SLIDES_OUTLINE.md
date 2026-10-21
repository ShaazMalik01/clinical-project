# 12-slide presentation (about 10 minutes, plus questions)

Keep each slide to a headline and at most four short bullets. Speaker notes are what you say.

| # | Slide | On the slide | Speaker notes (about 45 seconds each) |
|---|---|---|---|
| 1 | Title | Project name, your name, course, one-line promise: "Find the sites and records that need a monitor's attention, and say why." | Introduce the problem in one sentence: trials must follow a protocol and the data must be trustworthy. |
| 2 | The problem | Protocol deviations, GCP and ICH E6; why 100% source verification is costly; three anomaly types: honest error, process failure, fabrication | Explain compliance simply. Fabrication is the hardest because fake data tries to look normal. |
| 3 | Idea in one picture | Architecture diagram (README Mermaid); four layers: rules, ML, forensics, Bayesian risk | One system, four complementary views. Each answers a different question. |
| 4 | Data and ground truth | Synthetic 12-site trial, 608 subjects, 359 injected anomalies, two bad sites (S08 sloppy, S06 fabricating) | Real data is confidential and has no labels. Injecting anomalies gives exact answers, so we can measure. Say the limitation now. |
| 5 | Rule engine (expert system) | YAML knowledge base, 11 rules, ICH references, clean data gives 0 alarms, precision 1.000 | Transparent and auditable. Blind to anything nobody wrote a rule for: 0 of 12 plausible outliers, 0 fabrication. |
| 6 | ML ensemble | Isolation Forest + LOF + robust Mahalanobis, rank-normalised; finds 12 of 12 plausible outliers (visit AUC 0.999) | Three detectors with different blind spots. Robust covariance avoids masking. |
| 7 | Forensics (our unique part) | Terminal digit, low variance, copy-forward, duplicates, Benford with a gate; AUC 0.991, HIGH-tier precision 1.000 | Fake data leaves statistical fingerprints. Benford only on fields spread over two orders of magnitude, otherwise false alarms. |
| 8 | Bayesian site risk | Beta-Binomial with outlier component; intervals; 4 actions; both bad sites FULL_AUDIT, 10 clean sites ROUTINE | Shrinkage protects small sites: in simulation, false alarms fell from 16.7% to 8.7%. Real bad sites are not masked. |
| 9 | Explain and learn | Why-flagged card screenshot; fidelity 45 of 45; feedback loop and adapted threshold | Every alert says what is unusual and what to check. We tested the explanations against the field we corrupted. |
| 10 | Audit trail | Hash-chain diagram; tamper table (11 scenarios); anchor and HMAC | Tamper-evident, not tamper-proof. Two rows show what a bare chain cannot catch, which is why anchor and HMAC exist. |
| 11 | Results | Table: rules 0.817, ML 0.552, forensics 0.425, hybrid 0.882 F1; recall 1.000; ablation: rules+forensics 0.973 | Be honest: hybrid beats every single detector, but ML costs precision. The best F1 is rules plus forensics. Replicates agree. |
| 12 | Limits and next steps | Synthetic data, circularity, thresholds tuned on seed 42, not GxP-validated; future work | End with what the results do not show, then thank the audience. This honesty is a strength. |

Visual tips: use the dashboard's 3D risk city on slide 8 and the glass hero on slide 3; take your own screenshots after
running the app. Use `docs/phase10_main.png` on slide 11.
