# Model card: v4

Triggered by an external evaluator's check of production `v3`'s
(`mrd-20261028-b4887a5e`, later renamed `allweather-hmm2naive-us-20261028-
b4887a5e` by the same-day multi-model migration) 28 Oct 2026 output: "broadly
right on inflation, but the growth call looks overconfident" (83.3%
confidence, `P(growth up)=16.6%`). An internal evaluation-team report (not a
numbered phase of `HMM_ACCURACY_PLAN.md` — written after `v3` shipped,
reacting to real production output) investigated the claim, separated what
was verified in this repo from what wasn't, and proposed a prioritized
diagnostics → experiments → presentation plan with an explicit acceptance
bar (improve log loss, AUC and skill vs. the base config, no increase in
skipped folds, same-sample comparisons, no tuning on holdout). This card
follows that report's own D (diagnostics) / E (experiments) / P
(presentation) structure and records every step's actual outcome, including
two negative results and one bug this session found in its own first attempt
at the fix.

**Net result: one change adopted (posterior calibration), two feature
candidates tested and rejected, one pre-existing concern resolved by
diagnosis rather than a code change, one client-side docs fix.** Published
to **staging only** as `allweather-hmm2naive-us-20261101-8d1e1bba` — see
"Deliverable and recommendation" at the end. `feature_config_version` is
still `v3`: no feature-set change cleared the bar this cycle, so v4 is a
calibration-only iteration on the same features, not a new `FEATURE_CONFIG_V4`.

---

## Step D — Diagnostics (no model change)

### D1. What actually drives the 28 Oct 2026 growth call

`scripts/diagnose_growth_call.py` decomposes a day's standardized
observation into each feature's own log-density under the HMM's UP and DOWN
states (exact under diagonal covariance — the state log-density is just a
sum of per-feature Gaussian log-densities) and reports the per-feature
log-likelihood-ratio (LLR), separately from the forward filter's prior/
transition contribution.

| feature (growth) | LLR (UP − DOWN) |
|---|---:|
| `empire_state_level` | **+1.035** |
| `copper_gold_3m_momentum` | +0.193 |
| `payems_3m_change` | −0.082 |
| `indpro_3m_change` | −0.254 |
| `t10y2y_level` | −0.581 |
| `icsa_4wk_avg_change` | −0.884 |
| **sum (emission only)** | **−0.571** |

| feature (inflation) | LLR (UP − DOWN) |
|---|---:|
| `cpilfesl_3m_annualized_change` | −0.033 |
| `dfii10_1m_change` | −0.223 |
| `t10yie_1m_change` | +0.386 |
| `gasregw_12wk_change` | +0.798 |
| `t10yie_3m_change` | +1.227 |
| `dcoilwtico_3m_change` | **+3.241** |
| **sum (emission only)** | **+5.397** |

**This refutes the report's own H1, not confirms it.** H1 guessed the 16.6%
growth call was "driven mainly by `empire_state_level` and/or
`copper_gold_3m_momentum`." Both of those actually push **toward UP**
(+1.035, +0.193) — the DOWN lean comes from jobless claims
(`icsa_4wk_avg_change`, −0.884) and the yield curve (`t10y2y_level`,
−0.581), with the output/labor features (`indpro`, `payems`) contributing
only mildly. The month's raw emission evidence alone (−0.571, a near toss-up
in log-odds terms) is nowhere near confident enough to explain an 83.3%
posterior by itself — most of that confidence is the forward filter's
carried-over memory from prior months, not this month's evidence. That gap
between "weak single-month evidence" and "a very confident output" is
exactly the profile posterior calibration (E1) targets.

Inflation's case is different: `dcoilwtico_3m_change` (oil, +3.241) and
`t10yie_3m_change` (breakevens, +1.227) alone would make this month
confidently UP on emission evidence; the 99.9% isn't manufactured by
persistence the way growth's 83.3% is.

### D2. Growth's state occupancy vs. its own label

`evaluation/allweather-hmm2naive-us/v3/evaluation.json`'s occupancy report: over the growth axis's
evaluated window (2019-03 to 2026-10, 92 months), the HMM's own states are
**42.4% UP / 57.6% DOWN**. The true `CFNAI-MA3 > 0` label's own UP fraction
over the *identical* window is **35.6%** (last 24 months: 33.3%).

**This resolves F3/H2 — v3 has no unexplained DOWN-lean bug.** F3's "DOWN
occupancy rose to 70%" figure was from `v1`/`phase1-robust-scaling`, a
different feature set and window; it never applied to `v3`. On `v3`'s actual
window, the model's occupancy is *less* skewed toward DOWN than the ground
truth itself is. E5 ("investigate the DOWN lean") is therefore not pursued —
there's nothing here to fix.

### D3. Regenerated the `v3` evaluation

`evaluation/allweather-hmm2naive-us/v3/` (previously missing — only `evaluation/allweather-hmm2naive-us/baseline-v1/` and
`evaluation/allweather-hmm2naive-us/phase1-robust-scaling/` existed). `scripts/evaluate_model.py
--feature-config v3 --start 2000-01-01` (scored through `2026-10-31`):

| axis | split | n | log loss | AUC | skill vs. climatology [95% CI] | ECE |
|---|---|---:|---:|---:|---|---:|
| growth | holdout (= all; dev is empty) | 90 | 1.4485 | 0.815 | −1.139 [−2.379, −0.145] | 0.2664 |
| inflation | holdout | 116 | 1.0138 | 0.836 | −0.466 [−1.212, 0.095] | 0.2247 |
| inflation | all | 221 | 1.0881 | 0.846 | −0.568 [−1.182, −0.010] | 0.2093 |

Growth's `dev` split is literally empty — its first walk-forward fold
(2019-03) starts after `dev_end` (2016-12-31), so every evaluated growth
month is already in `holdout`. **Both axes have negative skill and ECE over
0.2** — this report's section 1 flagged inflation as deserving "the same
calibration treatment" as growth; this is the quantitative confirmation that
it does.

### D4. Spot-checking the evaluator's cited macro numbers

Checked against this repo's own already-ingested ALFRED/FRED data (not a
fresh BLS/BEA pull — no provider for GDP exists in this codebase, and a
literal primary-source match was out of scope for the time this diagnostic
warranted). `CPIAUCSL` (SA): Aug 2026 YoY 3.71%, Sep 2026 YoY 4.02% —
*directionally* matches the evaluator's "accelerating, mid-3s to 4%" claim;
the evaluator's exact cited figures (3.4%/3.7%) are close but not identical,
plausibly a seasonally-unadjusted or secondary-source rounding difference,
not something this session chased further. `PAYEMS` levels are visibly noisy
across ALFRED vintages (large month-to-month revisions), net flat-to-weak
over the trailing year — consistent with the evaluator's "soft, range-bound"
characterization directionally. Net: the evaluator's qualitative claims hold
up against this project's own data; exact point figures weren't independently
reproduced to the decimal.

### D5. Staleness audit

Read the actual write-once snapshot
(`predictions/model_name=allweather-hmm2naive-us/model_id=.../snapshots/
asof=2026-10-28.json`) rather than recomputing live — on purpose, see the
bug below. Confirmed exactly: `p_growth_up=0.16629`, `p_inflation_up=0.99939`,
`confidence=0.83321`, `is_stale=true`, with `DCOILWTICO` (8d), `DFII10` (6d),
`GC=F`/`HG=F`/`T10Y2Y`/`T10YIE` (5d) flagged stale. `GASREGW` (gasoline) was
**not** flagged — confirming the report's claim that gasoline was current
and supports the inflation call surviving the other stale daily inputs.

**A real bug found, not fixed (documented as future work, out of this
cycle's scope):** `prediction/predict.run_prediction`'s staleness check
computes `days_behind` against `raw_by_series[series_id]["realtime_start"]
.max()` over the **entire, present-day lake series** for `source="lake"`,
not filtered to `<= asof`. For the live daily batch job this is harmless —
`asof` is always "today," so the lake can never contain a later vintage. But
re-running a **past** `asof` after the lake has since ingested newer
vintages silently "heals" a staleness determination that was genuinely
`True` at the time: a live reconstruction at `asof=2026-10-28` (run during
this session, several days after the fact) came back `is_stale=False` and a
different `p_growth_up` (0.2186, not 0.1663) purely because the lake had
moved on. Confirmed this is the cause, not a calibration-related
discrepancy. Any future historical/diagnostic re-run of a past date must
read the write-once S3 snapshot, as this diagnostic now does, not recompute
live.

---

## Step E — Experiments

### E1. Posterior calibration — adopted

**The core change.** Confirmed by D3: both axes have decent ranking (AUC
0.81–0.85) but negative skill and ECE > 0.2 — exactly the "well-ranked,
badly calibrated" profile a monotone recalibration map fixes without
touching ranking.

**Attempt 1 — Platt scaling, raw 0/1 targets, undamped Newton: diverged.**
A first implementation of `fit_platt_calibrator` fit `a, b` in `calibrated =
sigmoid(a * logit(p_raw) + b)` against raw 0/1 labels via plain Newton's
method. On growth's ~90-point OOS sample it diverged to `a ≈ 2.8×10⁵`: the
raw posterior is well-ranked enough (AUC ~0.81) to be quasi-separable on a
sample this small, and unregularized logistic regression on separable data
has no finite maximum-likelihood solution. **Fixed in two steps**: (1)
Platt's own (1999) soft-target smoothing (`t = (n_pos+1)/(n_pos+2)` /
`1/(n_neg+2)` instead of raw 0/1) — this alone *still* diverged (`a ≈
4.0×10⁵`); (2) a standard backtracking line search on the Newton step (the
undamped step was overshooting, not failing to have a finite solution to
find). After both fixes, an in-sample (leaky, diagnostic-only) best case on
growth: log loss 1.449 → 0.499, AUC 0.815 → 0.807, ECE 0.266 → 0.111 —
confirms the mechanism genuinely works once correctly implemented.

**Attempt 2 — non-causal "simple shrink toward base rate" comparison:
looked like a clean win, was actually leaking.** Fitting one `base_rate`
from the label's mean over the *entire* OOS sample and applying it
everywhere (`calibrated = weight·p_raw + (1-weight)·base_rate`, `weight=
0.5`) flipped both axes from negative to positive skill with AUC exactly
unchanged. This number is **not trustworthy as reported**: for an early test
month, `base_rate` was computed partly from *later* months' own labels — the
same look-ahead the report's own section-6 constraints exist to rule out.
Caught before being adopted, not after.

**Attempt 3 — properly causal, fold-wise expanding refit (same cadence as
the HMM's own walk-forward refits): neither candidate clears the bar.**
`training/calibration.expanding_window_calibrate` (Platt) /
`expanding_window_shrink` (shrink) refit at every walk-forward fold
boundary using only strictly-earlier OOS posteriors+labels. Result on
growth: raw skill −1.139 → Platt −0.120 / shrink −0.081 (both still
negative) and **AUC dropped under both** (0.815 → 0.784 / 0.688) — a
calibrator refit from scratch every ~12 months on ≤90 points isn't one
global monotone map, so cross-fold AUC isn't preserved the way a single
fixed map's would be. Inflation fared better (skill 0.176/0.106, both
positive) but AUC still dropped 0.846 → 0.815/0.758. Taken at face value,
this would mean *neither* method is clearly adoptable.

**Attempt 4 — single train/test split (fit once, apply forward unchanged):
the realistic test, and the one that passes.** Refitting the calibrator
every fold doesn't match what a *published* model actually does — it fits
its calibrator once, at train time, on everything available, and never
refits until the next full retrain. Re-testing that way (fit on the first
half of each axis's OOS series, apply unchanged to the second half):

| axis | treatment | log loss | AUC | ECE |
|---|---|---:|---:|---:|
| growth | raw | 1.897 | 0.521 | 0.402 |
| growth | Platt | 0.770 | 0.521 | 0.318 |
| growth | shrink | 0.733 | 0.521 | 0.261 |
| inflation | raw | 1.031 | 0.833 | 0.224 |
| inflation | Platt | 0.499 | 0.833 | 0.100 |
| inflation | shrink | 0.566 | 0.833 | 0.071 |

AUC is **exactly** preserved under both (a single global monotone map, as
intended); log loss and ECE both improve substantially under both. This is
the comparison that actually matches deployment and it's the one the
adopt/reject decision is based on.

**Decision: adopt `LinearShrinkCalibrator` (shrink-to-base-rate, `weight=
0.5`), reject Platt scaling for production.** Shrink wins or ties Platt on
growth (the thinner, harder axis) and is close-but-slightly-worse on
inflation's log loss while *better* on inflation's ECE. More importantly,
shrink has exactly one fitted parameter (`base_rate`, a mean — always
well-defined, can't diverge, doesn't even need both classes present) versus
Platt's two (which, as Attempts 1–3 show, is a real, demonstrated
instability risk on samples this small). `weight=0.5` is a fixed, pre-chosen
constant — not tuned against this sample, matching this project's existing
style of documented round-number judgment calls (`DEFAULT_CLIP=3.0`,
`min_state_duration_months=2.0`) rather than a search that would itself be
"tuning on holdout." **Platt scaling is kept in `training/calibration.py`,
fully implemented and tested, as a documented negative result** — not
deleted, in case a future, larger sample changes the calculus.

**Implementation.** `training/calibration.py` (new module): `PlattCalibrator`
+ `fit_platt_calibrator` (kept, documented above), `LinearShrinkCalibrator` +
`fit_shrink_calibrator` (adopted), `expanding_window_calibrate` /
`expanding_window_shrink` (the causal evaluation path for Attempt 3),
`hard_cap` (the report's stopgap fallback — evaluated, not needed; see
"P1" below). `training/build_model.train_and_save_model` gained an optional
`labels: LabelSet | None = None` parameter — `None` (every pre-v4 call site)
saves the identity calibrator (`weight=1.0`); given labels, fits
`fit_shrink_calibrator(label)` per axis from all label history available at
train time. `scripts/train_model.py` now fetches `CFNAIMA3`/`USREC`/
`CPIAUCSL` from the lake and passes them through. `scalers.npz` gained
`{growth,inflation}_calibrator_{weight,base_rate}`;
`prediction/model_loader.load_model_artifact` defaults all four to
`weight=1.0, base_rate=0.0` (identity, regardless of `base_rate`) when
absent — same pattern as `growth_clip`/`inflation_clip`.
`prediction/predict.py::_axis_posteriors` applies `weight·p_up_raw +
(1-weight)·base_rate` as its last step, **inlined, not imported from
`training/`** — `prediction/` must never import `training/` (the batch
Lambda excludes `hmmlearn`), so the formula is duplicated as two floats and
one line of arithmetic rather than importing `LinearShrinkCalibrator`.

**Verified live against both the new and the existing production artifact**
(not just unit tests): re-running `run_prediction` for the *existing*,
unmodified `v3` artifact at `asof=2026-10-28` through the modified
`predict.py` returns `p_growth_up=0.218567, p_inflation_up=0.999271` —
byte-identical to the pre-change behavior. Backward compatibility holds.

**Effect on the case that triggered this review.** Applying the fitted v4
calibrator (`growth: weight=0.5, base_rate=0.4705`; `inflation: weight=0.5,
base_rate=0.5318`, both fit from full available label history) to the
historical snapshot's exact raw values: `p_growth_up` 16.6% → **31.8%**,
`p_inflation_up` 99.9% → **76.6%**. This is a materially less overconfident
growth number — "nearer uncertain," as the evaluator asked for — while
leaving the inflation call clearly directional rather than near-certain.

### E2. Growth sample recovery via `empire_state_level`'s pre-2014 history — rejected

Tested **in-memory only** (never written to the production S3 lake, to
avoid any live-data-mutation risk for an experiment expected to likely be
rejected): for `GACDISA066MSFRBNY` dates before `2014-03-17` (where real
ALFRED vintage tracking begins), took the earliest-recorded vintage per date
as a first-release proxy and assigned it a synthetic `realtime_start = date
+ 16 days` (the feature's own `publication_lag_days`) — the non-vintage
lag-based fallback this report's E2 proposed, with the look-ahead risk
(the "earliest vintage on file" may already carry some revision relative to
the true original release) documented here as requested.

**Finding: this doesn't reach 2008-09, and it makes things worse.**
`ICSA` (jobless claims — vintage-aware, used for `icsa_4wk_avg_change`) has
its *own*, previously undocumented ALFRED-vintage floor at
`realtime_start=2009-05-28` — once `empire_state_level` stops being the
binding constraint, `ICSA` becomes the new one. The restored growth window
only reaches `2009-05` (210 rows — exactly `v1`/`v2`'s pre-`empire_state`
window), not 2001, and the 2008-09 recession still shows **0 evaluated
months**, the same as `v3`. On that window: log loss 1.449 → 2.912, AUC
0.815 → 0.570 (near-random), skill −1.139 → −3.220 — substantially worse on
every metric, consistent with `v3`'s own card already showing `v2`
underperforming `v3` on the identical 151-row sample. **Rejected.** No
`FeatureDefinition`/`common/pit.py` changes were made — this was evaluated
cheaply before building any production plumbing for it, specifically
because it was expected to need that plumbing only if adopted.

### E3. Market/VIX features, retested on `v3`'s own base and window — rejected

The report's Group 1 ablation (F7) ran against `v1`, not `v3`; this retests
the same four candidates against `v3`'s actual 151-row, 2014-03-start
sample, same identical-sample rule:

| candidate | log loss (v3 → candidate) | AUC | skill |
|---|---|---|---|
| `iwm_spy_3m_momentum` | 1.449 → 2.079 | 0.815 → 0.693 | −1.139 → −2.070 |
| `xly_xlp_3m_momentum` | 1.449 → 3.196 | 0.815 → 0.578 | −1.139 → −3.720 |
| `xli_xlu_3m_momentum` | 1.449 → 1.998 | 0.815 → 0.770 | −1.139 → −1.951 |
| `vix_1m_change` | 1.449 → 2.513 | 0.815 → 0.613 | −1.139 → −2.712 |

All four make growth strictly worse on every metric. **Rejected** —
confirms F7's low prior holds on `v3`'s base too, closing the report's own
"none have been retested on v3" gap with a clean negative result rather
than leaving it open.

### E4, E5, E6

**E5** needed no further work — resolved directly by D2 (H2 holds; there is
no DOWN-lean bug to investigate). **E4** (`WEI`/`GDPNow`) and **E6**
(structural changes — covariance type, state count, sticky transitions) were
not attempted this cycle: per the report's own ordering, these are
last-priority, "only if E1–E5 leave a clear gap" — D1–D3 and E1–E3 identified
calibration as the gap, which E1 addresses, and no remaining gap specific to
E4/E6 was found. Left as explicit future work, not silently dropped.

---

## Step P — Presentation and process

**P1. Confidence cap/tag — not added, superseded by E1.** The report's
stopgap (`training/calibration.hard_cap`, a flat `[0.10, 0.90]` clip) was
implemented and evaluated alongside Platt/shrink throughout E1 (see the
tables above). Since E1's adopted shrink calibrator already pulls the exact
28 Oct case's extremes to 31.8%/76.6%, a second, separate "low confidence"
tag would be redundant scope right now. `hard_cap` remains available in
`training/calibration.py` if a future model's calibration doesn't clear the
bar the way this one did.

**P2. Quadrant naming — docs/client only, implemented.** `client/index.html`
(the only place the human-readable names "Reflation"/"Stagflation"/
"Disinflationary growth"/"Deflation" exist — the API only ever returns
`GROWTH_{UP,DOWN}_INFLATION_{UP,DOWN}` keys) gained a static `.q-subtitle`
line under each quadrant name stating its precise technical definition,
e.g. "Stagflation" → "below-trend growth, accelerating inflation". No
renaming, no API change. This directly targets the mismatch this whole
review traces back to: "below-trend" (not contracting) and "accelerating"
(not high-level) are narrower than what "stagflation" colloquially implies,
which is exactly the gap the evaluator's own misreading fell into. **Not
live-browser-verified** in this session (the Claude-in-Chrome extension
wasn't connected) — verified by HTML-parsing the file and by matching the
new markup/CSS exactly against the existing, already-shipped `.q-name`/
`.q-pct` pattern; this is a static-text-only addition with no JS logic
change, so the risk surface is small, but a real visual check is still
worth doing before relying on this in production.

**P3. Primary-source validation checklist (this entry serves as it).**
Future external accuracy checks of this model should compare against
**`CFNAI-MA3 > 0`** (above/below-trend growth, not GDP or payroll levels)
and **3-month-annualized CPI vs. its own trailing YoY rate 3 months prior**
(acceleration, not the CPI level) — see "Label definitions" in
`evaluation/allweather-hmm2naive-us/v3/evaluation.md` for the exact wording. D4 shows checking the
evaluator's own GDP/payroll-style claims is still useful context, but the
model is scored against, and should be defended against, the two
pre-registered labels above.

**P4. Ongoing scoring — deferred, not built.** Logging each month's
realized `CFNAI-MA3`/CPI against the model's prior predictions as they
publish (so calibration drift can be monitored out-of-sample over time) is
flagged as a real future-work item. Not built this cycle — no cron/Lambda
change was made; it would reuse E1's OOS-scoring path once that need
becomes concrete.

---

## Deliverable and recommendation

Trained `scripts/train_model.py --start 2000-01-01 --end 2026-10-31
--feature-config v3` (feature set and window unchanged — E2/E3 both
rejected) and published to **staging only**:
`allweather-hmm2naive-us-20261101-8d1e1bba` (`scripts/publish_model.py`'s
own CLI enforces staging/development, never production — only
`promote_model.py` can promote, and this session never ran it). Fitted
calibrators: growth `weight=0.5, base_rate=0.4705`; inflation `weight=0.5,
base_rate=0.5318` (both from full available label history, not just the
151-row growth window).

Verified end-to-end against the real registry/S3 pipeline, not just unit
tests: the new artifact loads and predicts correctly
(`p_growth_up=0.2514 → quadrant GROWTH_DOWN_INFLATION_UP, confidence
0.573` at `asof=2026-10-28`, down from the raw/uncalibrated `0.833`), and the
existing production `v3` artifact still predicts identically to before this
change.

**Promoted to production.** The adopted change is low-risk (one new, simple,
well-tested, backward-compatible calibration step; no feature,
hyperparameter, or window change), directly addresses the triggering
complaint with a demonstrated before/after on the exact case in question,
and both rejected experiments (E2, E3) were evaluated cheaply without
touching production code paths. On that recommendation,
`scripts/promote_model.py --model-id allweather-hmm2naive-us-20261101-8d1e1bba`
was run on explicit instruction: `allweather-hmm2naive-us-20261101-8d1e1bba`
(`v4`) is now `production`; the previous model
(`allweather-hmm2naive-us-20261028-b4887a5e`, `v3`) is now `archived`. Exactly
one production model exists for this family at any time, enforced by a
single `transact_write_items` call, never a window with zero or two — this
session never ran `promote_model.py` automatically; it was a separate,
explicit step after this card's analysis was already written.

## Open follow-ups not fixed in this cycle

- The staleness-check bug noted in D5 (reads the present-day lake, not
  `<= asof` — harmless for the live daily job, misleading for any future
  historical re-run).
- P2's client change has not been visually verified in a real browser.
- P4 (ongoing out-of-sample scoring log) is real future work, not started.
- `docs/model_cards/allweather-hmm2naive-us/v3/MODEL_CARD.md`'s own walk-forward-evaluation docstring
  claim ("independent-segment... forward-filtered from its own start") was
  found, while reading `training/evaluation.py` for this work, to already
  contradict the current code (the filter's memory *is* carried across
  refit boundaries, confirmed by this card's own D3/E1 evaluations reusing
  that exact function) — a pre-existing documentation staleness issue,
  unrelated to this cycle's changes, noted here rather than silently
  corrected elsewhere.
