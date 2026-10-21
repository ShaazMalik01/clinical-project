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
