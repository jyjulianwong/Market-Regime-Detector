# v4 improvement report

Written 2026-10-29, following an external accuracy check of the v3 model's
28 Oct 2026 output. This document is intended to be the **complete starting
context for the agent that trains v4**. Read it end to end before touching
code. It separates what was verified in this repository from what is an
unverified claim or a hypothesis, and says which is which.

Companion documents: `docs/model_cards/v3/MODEL_CARD.md` (full v3 development
history and all ablation numbers) and `ARCHITECTURE.md` (module structure and
data flow). `HMM_ACCURACY_PLAN.md`, referenced throughout the v3 card, is **not
in the repo**; its acceptance bar is restated in section 6 as the v3 card
describes it.

---

## 1. Summary

An evaluator compared the production v3 model's 28 Oct 2026 output with
public macro data and judged it "broadly right on inflation, but the growth
call looks overconfident." My assessment after reading the v3 card and code:

1. **The evaluator is right that the growth probability should not be trusted
   at face value, but its argument is partly wrong.** The strongest evidence is
   already in our own model card (negative skill vs. climatology on growth),
   not in the GDP and payroll comparison it used.
2. **The inflation call (99.9% UP) is plausible but extreme.** It deserves the
   same calibration treatment.
3. **There is no calibration step anywhere in the pipeline.** Posteriors are raw
   HMM outputs. This is the cheapest and most defensible v4 improvement.
4. **Growth has a thin, fragile training sample** (151 months, effectively one
   recession) and a documented, unexplained lean toward the DOWN state.
5. **The only real-time equity or market-price features tested were rejected
   on v1.** None have been retested on v3, so that question is open.

Recommended v4 order of work: diagnostics (no model change), then posterior
calibration, then growth feature and sample work, then presentation changes.
Section 7 gives each experiment with an acceptance bar.

---

## 2. System facts the training agent needs

All verified from the repo unless marked otherwise.

**Model structure.** Two independent 2-state Gaussian HMMs, one per axis
(growth, inflation), with diagonal covariance. States are labelled UP or DOWN
after fitting by a sign-weighted composite of state means. The API exposes
`p_growth_up` and `p_inflation_up` (the UP entry of each posterior). The
quadrant name, for example "Stagflation", is derived from the two axes
(growth down, inflation up).

**Production model.** `mrd-20261028-b4887a5e` (v3), promoted to production on
2026-10-28. Previous production `mrd-20261018-c94a9da2` is archived.
`publish_model.py` only publishes to `staging`; only `promote_model.py` can
promote. Keep that discipline for v4.

**v3 features** (`src/market_regime_detector/common/feature_config.py`):

| Axis | Feature | Source | Native freq / lag |
|---|---|---|---|
| growth | `indpro_3m_change` | ALFRED INDPRO | monthly, ~45d |
| growth | `payems_3m_change` | ALFRED PAYEMS | monthly, ~35d |
| growth | `icsa_4wk_avg_change` (sign -1) | ALFRED ICSA | weekly, ~4d |
| growth | `t10y2y_level` (identity, deliberate exception) | FRED | daily, 0d |
| growth | `copper_gold_3m_momentum` (HG=F / GC=F) | yfinance | daily, 0d |
| growth | `empire_state_level` | ALFRED GACDISA066MSFRBNY | monthly, ~16d |
| inflation | `cpilfesl_3m_annualized_change` | ALFRED | monthly, ~15d |
| inflation | `t10yie_1m_change`, `t10yie_3m_change` | FRED | daily, 0d |
| inflation | `dfii10_1m_change` (sign -1) | FRED | daily, 0d |
| inflation | `dcoilwtico_3m_change` | FRED | daily, 0d |
| inflation | `gasregw_12wk_change` | FRED | weekly, 0d |

**Labels (evaluation only, never features)**
(`src/market_regime_detector/training/labels.py`). These definitions matter for
interpreting any external comparison:

- **Growth UP** = `CFNAIMA3 > 0`, the Chicago Fed National Activity Index
  3-month average being above zero ("above-trend growth"). Growth DOWN therefore
  means *below-trend*, not *contracting*. A 2.2% GDP print with a soft labor
  market can be perfectly consistent with DOWN. Calling the DOWN/UP quadrant
  "stagflation" oversells what the growth label measures.
- **Inflation UP** = 3-month-annualized CPI exceeds the trailing year-over-year
  rate as of 3 months earlier ("inflation accelerating relative to recent
  trend"). It is about *acceleration*, not the level. September CPI going from
  3.4% to 3.7% y/y is direct support for UP.
- Secondary growth label: `1 - USREC` (NBER recessions). A sanity check only.
- Labels are knowable with a 1-month lag (`label_lag_months=1`). The scoring
  harness applies the lag.

**Training and evaluation.**
- Walk-forward expanding-window evaluation. Robust scaling (`median_mad`, clip
  3.0) is applied identically at fit, evaluation, benchmark and live-inference
  time (`common/scaling.py`).
- `HMMHyperparameters` (`training/build_model.py`): `n_restarts=10`,
  `min_state_duration_months=2.0`, `scaler_clip=3.0`, among others.
- Scripts: `scripts/evaluate_model.py --feature-config <version>` (scores a
  config against four causal baselines: climatology, calibrated persistence,
  fixed rule, logistic; writes `evaluation.json` and `evaluation.md`, publishes
  nothing), `scripts/train_model.py --feature-config <version>`,
  `scripts/publish_model.py`, `scripts/promote_model.py`.
- Metrics available in `training/scoring.py`: `log_loss`, `brier`,
  `skill_score` (vs climatology), `auc`, `accuracy`, `calibration_table` (10
  equal-width bins, returns table and ECE), `block_bootstrap_ci`,
  `flip_rate_per_year`, `mean_regime_duration_months`, `detection_lag`.
- Prior eval outputs: `eval/baseline-v1/` and `eval/phase1-robust-scaling/`.
  There is no checked-in v3 evaluation output; regenerate it (section 7, step
  D3).

**Staleness.** A series is stale if its latest `realtime_start` is older than
its frequency tolerance (3d daily, 10d weekly, 40d monthly). The API ORs the
batch job's `is_stale` with a request-time check (`generated_at` older than
36h). A stale flag does not alter the probabilities. It is a warning only.

---

## 3. The trigger: evaluator feedback on the 28 Oct 2026 output

**Model output being checked:** Stagflation, 83.3% confidence, P(growth up) =
16.6%, P(inflation up) = 99.9%, flagged stale.

**Evaluator's claims** (all **unverified by me**: I had no access to the data,
and its sources were secondary summaries, not BLS/BEA primary releases):

- September CPI +0.6% m/m, 3.7% y/y (August 3.4%); energy +20.4% y/y, gasoline
  +34.2% y/y; core CPI only 2.6% y/y; record September gasoline price per AAA.
- September payrolls +29k, unemployment 4.2% (range-bound 4.1–4.3% since
  March), 12-month average payroll gain about 45k/month.
- Q2 GDP third estimate 2.2% annualized (Q1 2.5%).
- No current GDPNow reading found; the figures it found were from early
  August (about 5–6% for Q3).
- Stale inputs: oil (DCOILWTICO), TIPS (DFII10), breakevens (T10YIE), 5–8 days
  old.

**Evaluator's conclusions:** inflation call well supported; growth call
"directionally plausible" but P(growth up) should be nearer "uncertain" than
16.6%; the stale flag is fair; HMM posteriors are overconfident and not
calibrated odds.

**Action item before relying on any of the numbers above:** check them against
primary sources (BLS, BEA, Chicago Fed) as part of diagnostic D1/D2.

---

## 4. Findings

### 4.1 Verified from the repo and model card

- **F1. Growth skill vs climatology is negative.** v3 on its 151-row sample:
  log loss 1.449, AUC 0.815, skill -1.139. v2 restricted to the same 151
  dates: 1.941 / 0.722 / -1.866. Negative skill means the probabilities are
  worse than always predicting the base rate, even though the ranking (AUC) is
  decent. This is the main reason the 83% should be distrusted, and it is
  independent of the evaluator's argument.
- **F2. Probabilities are never calibrated.** No post-processing step exists
  between the HMM posterior and the API (`prediction/predict.py::
  _axis_posteriors`). `scoring.calibration_table` measures it but nothing
  corrects it.
- **F3. Growth DOWN-state occupancy rose from 38% to 70% of evaluated months**
  when robust scaling was introduced in Phase 1 (v1 config). The v3 card flags
  this as unexplained and never revisits it. The v3 figure is unknown.
- **F4. The growth sample is thin.** `empire_state_level`'s ALFRED vintage
  history only begins 2014-03-17, so adding it cut growth training from 209
  rows (2009-05 to 2026-09) to 151 rows (2014-03 to 2026-09). The window holds
  essentially one recession (COVID). The 2008-09 episode shows 0 months in
  validation. Inflation has 282 rows.
- **F5. The v3 vs v2 improvement is smaller than the headline.** On the same
  151 dates the Empire State feature's own contribution is log loss 1.941 to
  1.449, about 25% relative. The raw 2.554 to 1.449 gap mixes a window effect
  with the feature effect.
- **F6. Growth leans on slow inputs.** Fast inputs are claims, the yield curve,
  copper/gold, and one monthly survey. INDPRO (45d lag) and PAYEMS (35d lag)
  carry the output and labor signal.
- **F7. No equity-price feature is in production.** SPY, IWM, XLY, XLP, XLI,
  XLU, VIXCLS and DGS2 are all in the lake. The only traded-market growth
  feature is copper/gold. The Phase 5 Group 1 ablation (on **v1**, not v3)
  rejected all of them:

  | Candidate | log loss (v1 to cand.) | AUC | Note |
  |---|---|---|---|
  | `iwm_spy_3m_momentum` | 2.554 to 2.818 | 0.662 to 0.563 | worse |
  | `xly_xlp_3m_momentum` | 2.554 to 2.942 | 0.662 to 0.568 | worse |
  | `xli_xlu_3m_momentum` | 2.554 to 2.875 | 0.662 to 0.608 | worse |
  | `vix_level_last` / `_mean` | 5.968 / 6.123 | 0.344 / 0.411 | far worse |
  | `vix_1m_change` | 2.554 to 2.239 | 0.662 to 0.664 | better loss, **+1 skipped fold** |
  | `vix_1m_pct_change`, `vix_3m_change` | no improvement | | |
  | `dgs2_1m_change` | 2.554 to 3.041 | 0.662 to 0.528 | worse, +3 skipped folds |

### 4.2 Assessment of the evaluator's reasoning

- **Right:** growth 83% is overstated; posteriors are not calibrated odds.
- **Overstated:** "the HMM may underweight output data." `indpro_3m_change` is
  a growth feature. The better hypothesis is that output is one of six
  features, the strongest feature is a regional manufacturing survey, and the
  rest are slow or market-based. Unverified: needs D1.
- **Wrong target:** it compared GDP and payrolls to the DOWN call. The model is
  scored against CFNAI-MA3 > 0 (below-trend, not contraction). Its own summary
  ("below-trend but not contracting") is consistent with DOWN. The check that
  matters is September's CFNAI-MA3 and recent months' values.
- **Weak evidence cited:** August GDPNow readings are 2.5 months old and say
  nothing about the current state.
- **Inflation side:** it is supported by the label definition (acceleration)
  and the feature set (breakevens, oil, gasoline). Core CPI at 2.6% matters
  less here than in a level-based definition. 99.9% is still extreme for a
  ~282-row, 2-state model.
- **Staleness:** the flag is correct and informational. Gasoline is weekly and
  current, which supports the inflation call surviving the stale oil/TIPS
  inputs.
- **Possible naming problem (hypothesis):** "Stagflation" for (below-trend
  growth, accelerating inflation) invites exactly the misreading the evaluator
  made.

### 4.3 Hypotheses (unverified, to be tested)

- H1. The 16.6% growth reading is driven mainly by `empire_state_level` and/or
  `copper_gold_3m_momentum`, not by the output and labor series. Test: D1.
- H2. The DOWN lean (F3) partly reflects the label's own base rate. CFNAI-MA3
  has spent long stretches at or below zero since 2009, so a high DOWN
  occupancy might be correct rather than a bug. Test: D2.
- H3. Calibrating the posteriors will cut ECE and log loss materially without
  changing AUC. Test: section 7, E1.
- H4. Some equity or VIX features that failed on v1 will behave differently on
  the v3 base and window. Low prior, but untested. Test: E3.

---

## 5. A short HMM and calibration primer (for context, not for implementation)

- Each axis has two hidden states. Each month the model computes a posterior,
  the probability of each state given the features. `p_growth_up` is the UP
  entry.
- The posterior assumes the fitted model (two Gaussian clusters on about 150
  months) is correct. If it isn't, outputs can still be extreme.
- **Calibrated** means stated probabilities match observed frequencies: of
  all months where the model said 80%, about 80% should turn out that way.
- **Calibration** is a post-processing map from raw posterior to a corrected
  probability, fitted on out-of-sample predictions. Platt scaling (a 1-D
  logistic regression on the raw logit) suits small samples. Isotonic
  regression is more flexible but needs more data than we have for growth.
- Calibration fixes numbers, not rankings. A monotone map leaves AUC unchanged.
  F1 (decent AUC, negative skill) is exactly the profile calibration helps.
- It must be fit on walk-forward out-of-sample posteriors, ideally using only
  data available at each point in time. Fitting on in-sample posteriors would
  just calibrate the overconfidence away from nothing.

---

## 6. Constraints that must hold for v4

These come from the v3 card and project conventions. Do not relax them without
saying so explicitly.

- **Labels are pre-registered** (CFNAIMA3 > 0, the CPI acceleration rule,
  USREC). Never change them to improve a score. `CFNAI`/`CFNAIMA3` and
  `CPIAUCSL` stay out of the feature set (label circularity).
- **No look-ahead.** Every transform must pass the no-look-ahead property test
  (appending future rows never changes an earlier row's output). Use ALFRED
  vintages where real ones exist.
- **Acceptance bar for a new feature** (as the v3 card describes
  `HMM_ACCURACY_PLAN.md`): improve log loss, AUC and skill versus the base
  config **with no increase in skipped folds**. Aggregate metrics don't
  penalize skipped folds, so always report the count.
- **Compare on the same sample.** F4/F5 show a different training window
  changes the numbers on its own. Restrict the baseline to the identical
  decision dates whenever a candidate changes the sample.
- **Overfit guard.** About 150 growth rows with diagonal 2-state HMMs. Prefer
  one strong feature to several transforms of the same series.
- **Do not tune on holdout.** Use the dev/holdout split from
  `training/harness.run_evaluation`.
- **Never promote automatically.** Publish v4 to `staging` only, and document
  the decision in the v4 model card.
- **Backward compatibility.** New artifact keys must default safely when
  missing, as `growth_clip`/`inflation_clip` do in
  `prediction/model_loader.load_model_artifact`. A calibrator should default to
  the identity map so models published before it still load and predict
  unchanged.

---

## 7. Proposed work, in order

### Step D: diagnostics (no model change)

- **D1. Explain the current growth call.** Build the 28 Oct 2026 growth
  feature vector (via `prediction/predict.py` internals or
  `scripts/run_prediction.py`), then report each feature's robust z-score and
  its contribution to the log-likelihood ratio between the DOWN and UP states.
  Report which features drive P(growth up) = 16.6%. Tests H1.
- **D2. Check growth against its real label.** Pull recent CFNAIMA3 values
  (including September 2026 if published) and compare with the model's
  walk-forward growth labels over the last 24 months. Report the confusion
  matrix. Compute the label's UP base rate on v3's 151-row window and compare
  with v3's DOWN occupancy (`training/occupancy.build_occupancy_report`).
  Tests H2 and F3.
- **D3. Regenerate the v3 evaluation.** Run `scripts/evaluate_model.py
  --feature-config v3` and keep the output. Extract `calibration_table` and
  ECE for both axes, the reliability curve for the high-confidence bins
  (especially p < 0.2 and p > 0.9), and flip rate and regime duration.
- **D4. Verify the evaluator's data from primary sources** (BLS CPI and
  Employment Situation, BEA GDP, Chicago Fed CFNAI). Record which of its
  numbers hold.
- **D5. Staleness audit.** List each input's age at the 28 Oct run and confirm
  gasoline is current. Confirm the stale flag doesn't change posteriors.

### Step E: experiments

- **E1. Posterior calibration (highest value, lowest risk).**
  - Fit a Platt calibrator per axis on out-of-sample walk-forward posteriors,
    using an expanding window so each calibrated value only uses past data.
  - Compare against (a) no calibration, (b) simple shrink toward the base rate,
    and (c) a hard cap such as 10–90%.
  - Metrics: log loss, Brier, ECE, skill vs climatology, block-bootstrap CIs.
    AUC should be unchanged (monotone map). Report the CI, since growth has
    about 150 points.
  - Implementation sketch: store calibrator parameters in the artifact
    (identity default when absent), apply in `_axis_posteriors`, add the
    corresponding loader and publisher support and tests.
  - Do not use isotonic on growth unless D3 shows enough data per bin.
- **E2. Recover the growth sample.** Test restoring pre-2014 rows for
  `empire_state_level` using the plan's non-vintage fallback (observation date
  plus `publication_lag_days`, about 16d) for dates ALFRED can't vintage.
  Risk: seasonal revisions leak mild look-ahead, so document it. Evaluate v3
  vs a restored-window v3 on the identical dates, and check whether the 2008–09
  episode now validates. Expected gain is stability, not necessarily a better
  headline score.
- **E3. Retest market features on the v3 base.** Candidates: the three equity
  ratios, `vix_1m_change`, `nfci_4wk_change`, `empire_state_3m_change`. Each
  under the section 6 bar and the identical-sample rule. Prior is low (F7),
  but v3's window and base differ from v1.
- **E4. Additional fast growth indicators.** Consider `WEI` (starts 2008; needs
  a dedicated shorter-window config, per the v3 card) and `GDPNOW` (needs a
  deliberate as-of design across a quarter). Both were deferred, not rejected.
  Treat as a separate experimental config, not a change to the default window.
- **E5. Investigate the DOWN lean.** If D2 shows H2 is false, i.e. occupancy
  exceeds the label base rate, look at the state-labelling composite, the
  prior and transition initialisation, `min_state_duration_months`, and
  scaler clip. Change one thing at a time and re-score.
- **E6. Structural options, last.** Covariance type, number of states, sticky
  transitions. Low priority given the sample; only try if E1–E5 leave a clear
  gap.

### Step P: presentation and process

- **P1. Cap or label confidence in the API/UI.** Even after E1, avoid showing
  99.9%. Consider confidence bands or a "low confidence" tag when the
  calibrated probability is near 0.5 or when reliability in that bin is poor.
- **P2. Reconsider the "Stagflation" label.** Because growth DOWN means
  below-trend, not contraction, consider wording such as "below-trend growth,
  accelerating inflation," and document the label definitions next to the
  quadrant name.
- **P3. Add a primary-source validation checklist** to the release process so
  future external checks compare against the right target (CFNAI-MA3 and the
  acceleration rule), not GDP and level CPI.
- **P4. Ongoing scoring.** Once each month's CFNAI-MA3 and CPI publish, log the
  prior predictions against the realised labels so calibration can be
  monitored out of sample.

---

## 8. Deliverables expected from the v4 agent

1. A v4 model card at `docs/model_cards/v4/MODEL_CARD.md`, in the same style as
   v3: before/after numbers, decisions, bugs found, negative results recorded.
2. Diagnostic outputs D1–D5 summarised, with the raw evaluation JSON/MD checked
   into `eval/` as earlier phases did.
3. For each experiment: the metric table, the skipped-fold count, the sample
   used, and an explicit adopt/reject decision against the section 6 bar.
4. If E1 is adopted: calibrator code, artifact support with identity default,
   tests, and a note on how the API presents calibrated values.
5. A v4 artifact published to **staging** only, with a recommendation on
   promotion for a human to decide.

## 9. Open questions for the owner

- Is renaming the quadrant labels (P2) in scope for v4, or only a docs change?
- Is a 10–90% display cap acceptable as a stopgap if E1 is not ready?
- Is loosening the PIT rule for pre-2014 `empire_state_level` (E2) acceptable if
  documented, or must v4 stay strictly vintage-based?
- Does the lake/AWS access needed for D1–D3 and E-steps exist in the training
  agent's environment? I did not check whether this session has it.
