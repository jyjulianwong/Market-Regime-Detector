# Model card: v3

The full development history behind `FEATURE_CONFIG_V3`, the production
feature config as of this writing — every phase of `HMM_ACCURACY_PLAN.md`'s
8-phase accuracy-improvement plan (Phase 6 was skipped: conditional on
Phase 5 needing multi-series combination support beyond the existing ratio
feature, which it didn't), in order, with the actual before/after numbers,
decisions, and bugs found and fixed along the way.

For what the code currently looks like (module structure, import rules,
data flow), see `ARCHITECTURE.md` — this document is the history of how it
got there, not a description of the present structure.

---

## Phase 0 — Measurement harness

A repeatable yardstick so every later phase is judged on the same terms:
`scripts/evaluate_model.py --feature-config <version>` scores a named
`FeatureConfig` via `training/harness.run_evaluation` against four causal
baselines (climatology, calibrated persistence, fixed rule, logistic) on
dev/holdout splits, writing `evaluation.json`/`evaluation.md` without
publishing anything.

**State-occupancy report** (`training/occupancy.build_occupancy_report`)
answers "is a state just one episode?" directly — motivated by the question:
was the growth DOWN state really just COVID, is inflation's DOWN/UP really
just 2021-23? On the `v1` reference run, growth's DOWN state spans `2016-05`
to `2026-09` (38% of months) — not literally just COVID — but fold 6
(`2020-05` to `2021-04`, the COVID-era refit) is visibly the most extreme
fold by emission z-means (`indpro_3m_change` ≈ -3.0 vs. every other fold's
DOWN state sitting around -1 to -1.3), which is exactly the outlier
amplification Phase 1's robust scaling targets below.

## Phase 1 — Robust scaling and clipping

**Robust scaling + clipping** (`common/scaling.py`, `HMM_ACCURACY_PLAN.md` Phase 1):
replaces a plain `sklearn.preprocessing.StandardScaler` everywhere a feature is
standardized before reaching the HMM, the logistic benchmark, or a live
prediction — a training window with even one outlier month (e.g. ICSA's March
2020 spike) would otherwise inflate that feature's `scale` for every other row
the scaler is ever applied to, which is exactly what let a single outlier carve
out its own HMM state. `fit_robust_scaler(X, clip, method)` computes a center/
scale that isn't dominated by the outlier (`method="winsorized"`: mean/std of
the column after clipping it to its own 1st/99th percentile; `method=
"median_mad"`: median and MAD × 1.4826); `apply_scaler(X, center, scale, clip)`
standardizes and then clips the result to `[-clip, clip]` — applied identically
at `trainer.fit_hmm` (fit time), `evaluation.walk_forward_evaluate` (scoring
time — clipping only at fit time would leave an early fold's tiny training-
window std producing triple-digit z-scores on a later outlier month),
`benchmarks.logistic_baseline` (so "agreement with the HMM" isn't partly an
artifact of which model's scaling handled outliers better), and
`prediction/predict.py`'s `_axis_posteriors` (live inference).

Defaults (`scaler_method="median_mad"`, `common.scaling.DEFAULT_CLIP = 3.0`)
were picked by a real-data ablation over clip in `{1.5, 2, 2.5, 3, 4, 5}` ×
method in `{winsorized, median_mad}`, scored with the full evaluation harness
(`scripts/evaluate_model.py`'s `run_evaluation`) against `FEATURE_CONFIG_V1` on
the full `2000-01-01`–present window. `clip=3.0` beat every looser clip on
every metric (log loss, AUC, accuracy, skill vs. climatology, skipped-fold
count) on both axes, and beat every tighter clip tried too — not monotonic
below 3, so this wasn't "tighter is always better." `median_mad` narrowly beat
`winsorized` at `clip=3.0` on both axes. Before/after, `v1` feature config,
full history, no holdout split held out of this comparison (see
`eval/baseline-v1/` → `eval/phase1-robust-scaling/`):

| axis      | metric              | before (StandardScaler) | after (median_mad, clip=3) |
|-----------|---------------------|-------------------------:|----------------------------:|
| growth    | log loss            | 5.290                    | 2.554                        |
| growth    | AUC                 | 0.421                    | 0.662                        |
| growth    | skill vs climatology| -6.665                   | -2.701                       |
| growth    | skipped folds        | 2                         | 0                             |
| inflation | log loss            | 2.314                    | 1.260                         |
| inflation | AUC                 | 0.421                    | 0.755                         |
| inflation | skill vs climatology| -2.335                    | -0.816                        |
| inflation | skipped folds         | 2                           | 0                              |

Every headline metric improved substantially and both axes now evaluate every
fold (previously 2 skipped folds each, from folds whose expanding window
hadn't yet "seen" the outlier-deflated-then-reinflated scale needed to pass the
sanity checks). **Caveat worth carrying into later phases**: growth's DOWN-state
occupancy rose from 38% to 70% of evaluated months (`training/occupancy`'s
`overall` report) — still spanning 2015–2026, so not a single episode, but a
large shift toward DOWN that Phase 1's scope (scaling only) doesn't explain or
fix. `skill_vs_climatology` is still negative on both axes — Phase 1 alone was
never expected to beat climatology outright; `HMM_ACCURACY_PLAN.md`'s later
phases (baseline fairness, feature pruning/surprise features) are where that's
expected to keep closing.

`training/build_model.py`'s `scalers.npz` gained `growth_clip`/`inflation_clip`
keys (`HMMHyperparameters.scaler_clip`, plumbed through every `fit_hmm`/
`walk_forward_evaluate` call); `prediction/model_loader.load_model_artifact`
defaults a missing key to `np.inf` (no clipping) so a model published before
this change keeps its original, unclipped behavior on load rather than being
silently clipped under an artifact that predates `common/scaling.py`.

## Phase 2 — Fix the baseline

**Baseline fix** (`training/baseline.py::baseline_signal`, `HMM_ACCURACY_PLAN.md`
Phase 2): the model-card "agreement with baseline" comparison
(`training/build_model.py` → `report.compare_to_baseline`) averaged **raw**
feature values into its sign-weighted composite — whichever feature happened to
have the largest native units (ICSA's raw claims count, in the thousands)
swamped every rate-of-change feature, making the comparison close to
meaningless. Fixed the same way `benchmarks.fixed_rule_baseline` (the
evaluation harness's own causal "rule" baseline, a separate consumer of the
same idea) already did it: standardize every feature with its own *expanding*
(causal, no look-ahead) mean/std before combining, dropping rows before
`min_observations` (24) expanding-window observations exist rather than
computing a degenerate early z-score. Before/after, measured as label
agreement with the real HMM's walk-forward out-of-sample labels on
`FEATURE_CONFIG_V1` + Phase 1's current scaler defaults, full history:

| axis      | agreement with baseline (before, raw) | agreement with baseline (after, expanding z-score) |
|-----------|---------------------------------------:|-----------------------------------------------------:|
| growth    | 38.3%                                    | 71.1%                                                  |
| inflation | 75.7%                                    | 74.8%                                                  |

Growth improved sharply — ICSA's raw-unit dominance was specifically a
growth-axis problem (growth's feature set is the one with a weekly claims-count
series; inflation's features are already closer in native scale). Inflation is
within noise of unchanged, which is expected: the fix removes a source of
meaninglessness in the comparison, it doesn't by itself make the HMM or the
baseline "more right" — `report.compare_to_baseline`'s own docstring already
flags that agreement is not evidence of correctness.

## Phase 3 — Transforms, config typing, feature ablation

**New transforms + config typing** (`common/transforms.py`, `common/feature_config.py`,
`HMM_ACCURACY_PLAN.md` Phase 3): `FeatureDefinition.transform_params` relaxed
from `dict[str, int]` to `dict[str, int | float | str]` (`apply_transform`'s
signature matches) — the existing int-only assumption was never actually
enforced at runtime (`_feature_definition_from_dict` only asserts the value is
a `dict`), just an overly narrow type hint; `FEATURE_CONFIG_V1.content_hash()`
is confirmed unchanged by this (same hash before/after:
`5ccf8d28...0232d0e`), since no value actually changed, only its static type.
Three new transforms were added to `TRANSFORMS`, each with a no-look-ahead
property test (appending future rows never changes an earlier row's output):
`detrend` (deviation from a trailing rolling mean — "how far is this value from
its own recent trend"), `second_diff` (`diff` applied twice at the same lag —
"acceleration"), `rolling_zscore` (trailing-window, not full-sample, z-score).

**Feature-set ablation — no net-positive addition found this round.** Five
candidates (surprise/acceleration variants of the plan's suggested
strongest features, added one at a time to `FEATURE_CONFIG_V1` and scored with
the real harness against the Phase 1 baseline, full history, `all` split):

| candidate              | axis      | log loss (v1 baseline → candidate) | AUC (v1 → candidate) | skill (v1 → candidate) | skipped folds |
|------------------------|-----------|---------------------------------------:|------------------------:|---------------------------:|--------------:|
| `payems_12m_detrend`   | growth    | 2.554 → 5.176                          | 0.662 → 0.413           | -2.701 → -6.500            | 0 → 0         |
| `indpro_12m_detrend`   | growth    | 2.554 → 3.265                          | 0.662 → 0.480           | -2.701 → -3.731            | 0 → 0         |
| `icsa_accel`           | growth    | 2.554 → 2.388                          | 0.662 → 0.659           | -2.701 → -2.389            | 0 → 3         |
| `cpilfesl_12m_detrend` | inflation | 1.260 → 3.276                          | 0.755 → 0.496           | -0.816 → -3.722            | 0 → 0         |
| `dcoilwtico_accel`     | inflation | 1.260 → 1.476                          | 0.755 → 0.779           | -0.816 → -1.128            | 0 → 0         |

Every `*_12m_detrend` candidate made things clearly worse — a monthly level's
deviation from its own trailing mean is evidently a noisier, more redundant
signal for this HMM than the rate-of-change feature already covering the same
series, not an independent "surprise" on top of it. `icsa_accel` is the one
candidate with a marginally better log loss/skill, but at the cost of 3
newly-skipped folds (vs. 0 for the Phase 1 baseline) — a real stability
regression the aggregate numbers alone don't show, since skipped folds are
simply absent rather than counted as a loss. `dcoilwtico_accel`'s AUC improved
slightly but log loss and skill both got worse. None clears the bar given
`HMM_ACCURACY_PLAN.md` Appendix B's overfitting risk (a ~300-point, 2-state,
diag-covariance HMM is already sample-constrained; adding a feature that isn't
a clear net win adds dimensionality for no benefit) — **`FEATURE_CONFIG_V2` was
not created this round; `FEATURE_CONFIG_V1` remains the only registered,
production config.** The new transforms stay registered and tested for a
future candidate (e.g. once Phase 4 brings in genuinely new series) to use —
this is a recorded negative result, not an unfinished implementation.

## Phase 4 — Data availability spike

**Phase 4 data-availability spike** (`HMM_ACCURACY_PLAN.md`, research + ingestion,
no `FeatureConfig`/model change): checked real vendor availability for every
candidate series the plan lists, against real FRED/ALFRED/yfinance, before
deciding what to pull into the lake. Per-date ALFRED vintage counts (not just
"does ALFRED return rows") settled which candidates are genuinely point-in-time
vs. not — e.g. `GDPNOW` has a median of 19 distinct `realtime_start`s per
observation date (confirms real, frequent re-estimation, contrary to the
plan's worry that nowcast backcasts might not be true PIT), `BAMLH0A0HYM2` has
exactly 1 vintage per date (never revised — consistent with a market-derived
spread) but its *history* only starts `2023-10-27` (ICE licensing, as the plan
anticipated), and `NFCI` isn't in ALFRED's vintage database at all (0 rows),
settling its ingestion `source` as `"fred"` rather than guessing.

| series                                      | verdict    | reason |
|-----------------------------------------------|------------|--------|
| `CFNAI`                                       | skip       | label circularity — `CFNAIMA3` (heavy revision history confirmed: median 36 vintages/date) *is* the growth label |
| `CPIAUCSL`                                    | skip       | label circularity — *is* the inflation label's underlying series (already excluded, `EVALUATION_SERIES`) |
| `BAMLH0A0HYM2` (HY OAS spread)                | skip       | FRED history starts `2023-10-27` only (ICE licensing) — unusable at any training window back to 2000; would need an ETF proxy (HYG/IEF/LQD) to revisit |
| `UNRATE`, `PERMIT`, `RSAFS`, `RRSFS`          | **ingested** | long history (1990+), real revisions confirmed, no circularity |
| `NFCI`                                        | **ingested** | long history, not in ALFRED (`source="fred"`) |
| `STLFSI4`                                     | **ingested, later reclassified** (Phase 5) | long history, heavily revised (confirmed in ALFRED) — but see the Phase 5 note below: the revision density turned out too extreme to paginate practically, so it was moved to `source="fred"` |
| `GACDISA066MSFRBNY`, `GACDFSA066MSFRBPHI`     | **ingested** | long history, real revisions confirmed; Phase 5 still needs the ~15-20 day publication-lag fix the plan calls out before either becomes a feature |
| `SPY`, `IWM`, `XLY`, `XLP`, `XLI`, `XLU`      | **ingested** | long history (1993-2000+ depending on ticker), daily, no vintage concept (equities) |
| `PCEPI`, `PCEPILFE`, `CES0500000003`, `MEDCPIM158SFRBCLE`, `TRMMEANCPIM158SFRBCLE` | **ingested** | long history, real revisions confirmed, not the label series (different index from `CPIAUCSL` — correlated, not circular) |
| `T5YIFR`, `DTWEXBGS`, `MICH`, `GASREGW`       | **ingested** | long-to-moderate history, not meaningfully revised (`source="fred"`) |
| `WEI`                                          | deferred   | starts 2008 — would truncate the shared training window (any row with a null feature is dropped axis-wide); worth a dedicated shorter-window experimental config, not a blanket window change for every feature |
| `GDPNOW`                                       | deferred   | genuine PIT vintages confirmed, but quarterly native frequency needs a deliberate as-of design across a quarter — not low-hanging fruit |
| `IHLIDXUS` (Indeed)                           | deferred   | starts 2020 only — same truncation tradeoff as `WEI` |
| `DCOILBRENTEU`                                 | skip       | near-duplicate of the already-used `DCOILWTICO` (WTI); Brent/WTI are highly correlated, no clear case for both |
| Cleveland Fed inflation nowcast, NY Fed Staff Nowcast, Manheim, NY Fed supply-chain index, card spending | skip | no public FRED/yfinance series id found this pass (nowcasts) or proprietary/short (private data), matching the plan's own expectation |
| `VIXCLS`, `DGS2`                              | **ingested** (Phase 5) | checked for availability in this spike's script (both 1990+, no vintage concept) but omitted from this table and `FIRST_PASS_SERIES` by oversight; closed in Phase 5 before building the "daily market" feature group that the plan names them for |

**A real bug found and fixed in Phase 5, not assumed correct**: `STLFSI4`'s
Phase 5 Group 2 ablation produced `NaN` log loss/AUC instead of a number.
Traced to `AlfredProvider.fetch()`'s `asof=None` path — `fredapi`'s
`get_series_all_releases` issues one unbounded request against ALFRED's
`series/observations` endpoint, which silently caps at 100,000 rows with no
truncation signal (`fredapi` never paginates). `STLFSI4`'s real revision
density is extreme enough to hit that cap even within a single 6-year
`realtime_start` window (confirmed directly against the live API, not
assumed) — so Phase 4's original backfill silently landed only `1993-10-10`
through `2003-10-10` of `STLFSI4`'s real `1993-2026` history. Every walk-forward
fold after 2003 then saw a frozen, zero-variance feature value (the PIT lookup
had nothing newer to find), which is what actually produced the `NaN`
metrics — not a flaw in the harness.

Fixed in `AlfredProvider.fetch()`: a new `_fetch_all_releases_paginated`
helper recursively bisects the `realtime_start`/`realtime_end` window until
each sub-query is under the cap, raising loudly if even a single-day window
still can't fit rather than guessing. But `STLFSI4` specifically turned out
too revision-dense to paginate *practically* even with this fix — a 6-year
window still hits 100,000 rows, meaning full-history pagination would need an
impractical number of narrow sub-requests. Resolution: the 11 corrupted
`STLFSI4` lake partitions (`year=1993`-`2003`, populated before the fix) were
deleted from the real S3 lake and `STLFSI4` was reclassified from
`source="alfred"`/vintage-aware to `source="fred"`/non-vintage in
`series_catalog.py` — the same trade-off already made for `NFCI`. Re-ingested
clean: `1993-12-31` through `2026-10-16`, one row per date, no true
point-in-time revision tracking for this series going forward.

**Training-window decision (open decision #4): keep the current ~2000+ window
as the default.** Only `WEI`/`GDPNOW`/`IHLIDXUS` would benefit from a shorter
window, and `BAMLH0A0HYM2` (the other motivating case) is unusable at *any*
window back to 2000 anyway (its real history starts 2023). A shorter window
for every feature to accommodate one or two short-history series would throw
away 8-18 years of training data project-wide — if `WEI` or similar earns its
place via ablation later, that's a dedicated shorter-window experimental
config, not the default.

**Label-circularity decision (open decision #2): keep `CFNAIMA3`/`CPIAUCSL` as
evaluation labels; `CFNAI` and `CPIAUCSL` stay excluded as features.** Not a
close call — `CFNAIMA3` is a 3-month average of `CFNAI` itself (near-tautological
if fed back in as a feature), and `CPIAUCSL` *is* the inflation label's input
series. Every other inflation candidate ingested this phase (PCE, median/
trimmed-mean CPI, wages, gasoline) is a *different* series correlated with but
not equal to the label — ordinary feature engineering (e.g. gasoline as a
faster-releasing leading indicator of headline CPI), not circularity, so none
of those needed excluding.

**Scope note**: `FIRST_PASS_SERIES` is what the deployed weekly ingest Lambda
(`prediction/handler.py`'s `"ingest"` action) iterates in production — adding
23 entries here means the *next* scheduled weekly ingest run (already deployed,
see "Infrastructure and deploy" in `CLAUDE.md`) will start pulling all of them
into the real S3 lake too, not just this session's one-off backfill. None of
them are referenced by `FEATURE_CONFIG_V1` yet, so this has no effect on the
model, cost is a few more provider API calls per week (all well within FRED's
free-tier limits) and S3 storage for 23 more small partitions.

## Phase 5 — Lower-lag features

**Phase 5 Group 1 (daily market) — no net-positive addition found this round
either.** `HMM_ACCURACY_PLAN.md` Phase 5 names VIX, the 2y yield, the 5y5y
inflation breakeven, the dollar index, and three cyclical/defensive or
small/large-cap ETF ratios as lower-lag candidates. `VIXCLS`/`DGS2` were
ingested this round to close a Phase 4 documentation gap (see "Phase 4
data-availability spike" above); the rest were already in the lake. Nine
candidates added one at a time to `FEATURE_CONFIG_V1`, scored the same way as
Phase 3, full history, `all` split:

| candidate               | axis      | log loss (v1 → candidate) | AUC (v1 → candidate) | skill (v1 → candidate) | skipped folds |
|-------------------------|-----------|------------------------------:|-------------------------:|----------------------------:|--------------:|
| `vix_level_last`        | growth    | 2.554 → 5.968                 | 0.662 → 0.344            | -2.701 → -7.647             | 0 → 0         |
| `vix_level_mean`        | growth    | 2.554 → 6.123                 | 0.662 → 0.411            | -2.701 → -7.872             | 0 → 0         |
| `vix_1m_change` (diff)  | growth    | 2.554 → 2.239                 | 0.662 → 0.664            | -2.701 → -2.325             | 0 → 1         |
| `vix_1m_pct_change`     | growth    | 2.554 → 2.501                 | 0.662 → 0.578            | -2.701 → -2.715             | 0 → 1         |
| `vix_3m_change` (diff)  | growth    | 2.554 → 2.838                 | 0.662 → 0.590            | -2.701 → -3.112             | 0 → 0         |
| `dgs2_1m_change`        | growth    | 2.554 → 3.041                 | 0.662 → 0.528            | -2.701 → -3.109             | 0 → 3         |
| `t5yifr_1m_change`      | inflation | 1.260 → 1.289                 | 0.755 → 0.767            | -0.816 → -0.861             | 0 → 1         |
| `dtwexbgs_1m_change`    | inflation | 1.260 → 1.509                 | 0.755 → 0.706            | -0.816 → -1.167             | 0 → 0         |
| `iwm_spy_3m_momentum`   | growth    | 2.554 → 2.818                 | 0.662 → 0.563            | -2.701 → -3.084             | 0 → 0         |
| `xly_xlp_3m_momentum`   | growth    | 2.554 → 2.942                 | 0.662 → 0.568            | -2.701 → -3.263             | 0 → 0         |
| `xli_xlu_3m_momentum`   | growth    | 2.554 → 2.875                 | 0.662 → 0.608            | -2.701 → -3.166             | 0 → 0         |

VIX as a raw level (either aggregation) is dramatically worse — unsurprising,
since a level mixes the slow secular decline of implied vol over 2000-2026
with the state signal, exactly the "level vs. change" problem `T10Y2Y` is the
one deliberate V1 exception to. The one candidate with a real log loss/skill
improvement, `vix_1m_change`, hits the identical trade-off Phase 3's
`icsa_accel` did: better aggregate numbers, but a newly-skipped fold (2014-15)
the aggregate metrics don't penalize. Two follow-up transforms tried to keep
the improvement without the skip (`vix_1m_pct_change`, `vix_3m_change`) —
neither did; both lose on log loss/AUC/skill *and* one still carries a skip.
Every other candidate is clearly worse on every metric. **`FEATURE_CONFIG_V2`
was not created from this group; `FEATURE_CONFIG_V1` remains the only
registered, production config** — the same `icsa_accel` precedent applies: a
newly-skipped fold is a real stability regression the plan's own Phase 5
acceptance bar ("no increase in skipped folds") explicitly rules out, and the
aggregate-metric improvement alone isn't enough to override that given
Appendix B's overfitting risk on a ~300-point sample.

**Phase 5 Group 2 (weekly official) — the first net-positive addition:
`FEATURE_CONFIG_V2` created.** Candidates: `NFCI`/`STLFSI4` (both financial-
stress/conditions composite indices, growth axis, level and 4-week-change
variants) and `GASREGW` (weekly gasoline price, inflation axis, 4-week and
12-week change variants). Scored the same way as Group 1, full history,
`all` split:

| candidate              | axis      | log loss (v1 → candidate) | AUC (v1 → candidate) | skill (v1 → candidate) | skipped folds |
|------------------------|-----------|------------------------------:|-------------------------:|----------------------------:|--------------:|
| `nfci_level`           | growth    | 2.554 → 4.234                 | 0.662 → 0.566            | -2.701 → -5.135             | 0 → 0         |
| `nfci_4wk_change`      | growth    | 2.554 → 2.875                 | 0.662 → 0.565            | -2.701 → -3.167             | 0 → 0         |
| `stlfsi4_level`        | growth    | 2.554 → 4.194                 | 0.662 → 0.466            | -2.701 → -5.078             | 0 → 0         |
| `stlfsi4_4wk_change`   | growth    | 2.554 → 2.937                 | 0.662 → 0.527            | -2.701 → -3.256             | 0 → 0         |
| `gasregw_4wk_change`   | inflation | 1.260 → 1.296                 | 0.755 → 0.806            | -0.816 → -0.869             | 0 → 0         |
| `gasregw_12wk_change`  | inflation | **1.260 → 1.088**             | **0.755 → 0.846**        | **-0.816 → -0.568**         | **0 → 0**     |

Both financial-stress indices are clearly worse on every metric, as a growth
feature — the same pattern as Group 1's VIX-as-a-level failure: a composite
index that's already a weighted blend of ~100+ underlying series apparently
doesn't give this 2-state HMM an orthogonal signal beyond what V1's
rate-of-change features already capture, and may be *diluting* the signal by
adding a noisy, redundant dimension (Appendix B's overfitting risk, directly
observed). `gasregw_4wk_change` is a wash (AUC up, log loss and skill
slightly down). `gasregw_12wk_change` is a clean win on every metric with no
skipped-fold regression — the only candidate across both groups so far to
clear the plan's acceptance bar outright, not a marginal trade-off like
Group 1's `vix_1m_change`. A faster-releasing, weekly gasoline price feeding
directly into the inflation basket beats every V1 inflation feature's
monthly-or-slower cadence, which is exactly the "lower-lag features are
likely the highest-value change" hypothesis section 0.1 opened with.
**`FEATURE_CONFIG_V2` = `FEATURE_CONFIG_V1` + `gasregw_12wk_change`**, now
registered in `FEATURE_CONFIGS` alongside `v1`.

**An ALFRED pagination bug found and fixed mid-ablation** (see
`ARCHITECTURE.md`'s `ingestion/` section for the full writeup): `STLFSI4`'s
first run through this ablation produced `NaN` metrics, traced to
`AlfredProvider` silently truncating its real vintage history past ALFRED's
100,000-row request cap. Fixed (`_fetch_all_releases_paginated` bisects the
request window instead of issuing one unbounded call), but `STLFSI4`'s
revision density turned out too extreme to paginate practically either way —
it was reclassified to non-vintage `source="fred"` and the corrected numbers
above are from that clean re-ingest, not the original broken run.

**Phase 5 Group 3 (in-month surveys) — the strongest single-feature win in
all of Phase 5.** Candidates: Empire State (`GACDISA066MSFRBNY`) and Philly
Fed (`GACDFSA066MSFRBPHI`) regional manufacturing surveys, growth axis, level
and 3-month-change variants. Both were checked for real ALFRED vintages this
round (median 5 and 3 revisions/date respectively, safely under the
pagination cap fixed above) and confirmed genuine — `vintage_aware=True` is
used directly rather than the plan's non-vintage fallback lag rule, which was
written for the case where no real vintages exist. Observed real publication
lag (first vintage vs. the observation date stamped to the 1st of the
reference month): ~14-17 days for Empire State, ~15-21 days for Philly Fed —
both recorded as `publication_lag_days` metadata per the existing convention
(documentation only; `realtime_start` supersedes it operationally). Scored
against `FEATURE_CONFIG_V2`, full history, `all` split:

| candidate                 | axis   | log loss (v2 → candidate) | AUC (v2 → candidate) | skill (v2 → candidate) | skipped folds |
|----------------------------|--------|------------------------------:|-------------------------:|----------------------------:|--------------:|
| `empire_state_level`       | growth | **2.554 → 1.449**             | **0.662 → 0.815**        | **-2.701 → -1.139**         | **0 → 0**     |
| `empire_state_3m_change`   | growth | 2.554 → 1.488                 | 0.662 → 0.834            | -2.701 → -1.198             | 0 → 0         |
| `philly_fed_level`         | growth | 2.554 → 2.885                 | 0.662 → 0.571            | -2.701 → -3.190             | 0 → 0         |
| `philly_fed_3m_change`     | growth | 2.554 → 3.191                 | 0.662 → 0.556            | -2.701 → -3.635             | 0 → 0         |

Empire State's survey level nearly halves growth log loss with no
skipped-fold regression — by far the largest single-feature improvement
found across all of Phase 5 (Groups 1-3), confirming section 0.1's "lower-lag
features are likely the highest-value change" hypothesis more strongly than
Group 2's `gasregw_12wk_change` did. The 3-month-change variant is a
near-tie against the level (better AUC, slightly worse log loss/skill) — not
an independent second win, just the same signal transformed, so only the
level was adopted (Appendix B's overfitting-risk principle: one strong
feature over two redundant transforms of the same series). Philly Fed's
equivalent survey fails on both variants, clearly worse on every metric —
a genuine axis-specific finding, not a methodology gap, since the identical
treatment worked decisively for Empire State. **`FEATURE_CONFIG_V3` =
`FEATURE_CONFIG_V2` + `empire_state_level`**, now registered in
`FEATURE_CONFIGS`.

**Caveat found and checked during Phase 7's retrain, not assumed away**:
`empire_state_level`'s real ALFRED vintage history only starts `2014-03-17`
(confirmed by checking `min(realtime_start)` per observation date — every
date before that, back to the series' real 2001 start, has its *earliest*
vintage stamped `2014-03-17`, meaning ALFRED's vintage tracking for this
series simply doesn't reach further back, even though raw observation dates
do). `_build_axis_df` drops any row with a null feature, so adding this
feature truncates V3's growth training sample from V2's 209 rows (`2009-05`
to `2026-09`) down to 151 rows (`2014-03` to `2026-09`) — the table above
compared `v2_baseline` on its full 209-row sample against candidates on
that smaller 151-row sample, not strictly apples-to-apples, since a
different (more recent) sub-period could simply be easier to fit regardless
of which feature is added. Rerun with `v2` restricted to the exact same 151
decision dates `v3` used: `v2` (restricted) scores log loss `1.941`, AUC
`0.722`, skill `-1.866` — clearly better than `v2`'s full-sample numbers
(confirming the 2014-2026 window genuinely is somewhat easier on its own),
but still clearly worse than `v3`'s `1.449`/`0.815`/`-1.139` on the identical
window. Both effects are real; `empire_state_level` contributes a genuine,
independent improvement on top of the window effect, so the adoption
decision stands — but the feature-attributable improvement is the
apples-to-apples gap (`1.941` → `1.449`, ~25% relative), not the larger
raw gap the table above shows at face value.

**Phase 5 Group 4 (published nowcasts) — deferred, no change.** The plan's
only Group 4 candidate is `GDPNOW`, already resolved in the Phase 4
data-availability spike above: genuine PIT vintages confirmed, but its
quarterly native frequency needs a deliberate as-of design across a quarter
that this phase didn't attempt — not low-hanging fruit, per that existing
decision. No code change; this closes out Phase 5's four groups.
`FEATURE_CONFIG_V3` is the final result of Phase 5: one growth win (Empire
State) and one inflation win (gasoline), on top of V1, out of 15 candidates
tested across Groups 1-3.

## Phase 7 — Integration, release, and documentation

**Retrained and published `v3`.**
`scripts/train_model.py` gained a `--feature-config` flag (mirroring
`scripts/evaluate_model.py`'s existing one) since `train_and_save_model`
already accepted the parameter but the CLI had no way to select anything but
`v1`. Retrained full-history (`2000-01-01` to today) with `v3`: growth hash
`c5e2fdfd...`, 151 growth rows / 282 inflation rows (growth's row count is
the `empire_state_level` truncation documented above, carried through to the
real training run, not just the ablation), 0 skipped folds on either axis.
Known-episode validation: 2020 COVID (12 months, mean confidence 0.954) and
2021-23 inflation shock (36 months, 0.903) both validate with high
confidence; the 2008-09 financial crisis episode shows 0 months (confirmed
pre-existing, not a new regression — `v2`'s growth sample already starts
`2009-05`, after the crisis, for reasons predating this plan). Published to
the model registry as `mrd-20261028-b4887a5e`, stage `staging` (never
`production` — `publish_model.py`'s own CLI enforces that; only
`promote_model.py` can). Verified both ways round-trip through the current
codebase: the new `v3` artifact predicts correctly, and the existing
production artifact (`mrd-20261018-c94a9da2`, published before this
session's Phase 1 scaler/clip work) still loads and predicts — backward
compatibility holds.

**Promoted to production.** `scripts/promote_model.py --model-id
mrd-20261028-b4887a5e` atomically swapped the DynamoDB production pointer —
`mrd-20261028-b4887a5e` (`v3`) is now `production`; the previous model
(`mrd-20261018-c94a9da2`, published before this plan's Phase 1 work) is now
`archived`. Exactly one production model exists at any time, enforced by a
single `transact_write_items` call, never a window with zero or two. The
accompanying batch Lambda invoke is best-effort (see
`training.promote.invoke_batch_lambda_best_effort`) — the DynamoDB promotion
is the actual invariant and already succeeded by the time that invoke runs;
a fresh prediction for the new model was already generated and written
during verification (`predictions/model_id=mrd-20261028-b4887a5e/`), so the
API has something current to serve immediately regardless of whether the
async invoke itself lands.
