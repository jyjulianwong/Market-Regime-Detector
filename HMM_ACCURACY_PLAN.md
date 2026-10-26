# HMM Accuracy Improvement Plan

Implementation plan for improving the accuracy of the growth/inflation 2-state Gaussian HMMs.
Written for a Claude Code session that has the S3 data lake and external vendor connectivity
(FRED, ALFRED, yfinance). Work phase by phase; **each phase ends with a walk-forward
evaluation against the previous phase** so we can see what actually moved accuracy.

## 0. Background and findings

### 0.1 Problems being addressed

1. **Outliers hijack a state.** ICSA's 4-week-average change spiked by ~1M in March 2020 vs. a few
   thousand normally. After standardization that single episode inflates ICSA's std, leaving ICSA
   near zero everywhere else. Gaussian HMMs tend to give such an outlier its own state ("DOWN =
   COVID", "UP = not a catastrophe"). Inflation in 2021-23 probably has the same issue.
2. **The rule-based baseline is unscaled** (`training/baseline.py`): it averages raw feature values,
   so ICSA (claims count) swamps everything and "agreement with baseline" is close to meaningless.
3. **Features are weak/redundant and slow.** No "surprise" features, near-duplicate features, a
   level (`T10Y2Y`) mixed with changes, and mostly high-lag monthly data. Lower-lag features are
   likely the highest-value change for a nowcast.
4. **Transform/combination support is limited** (single-series transforms only; ratio is the only
   multi-series case).

### 0.2 Code facts verified in the repo (do not re-derive)

| Fact | Where | Consequence |
|---|---|---|
| `StandardScaler` is fit on the *entire* training window | `training/trainer.py:84-85` | One outlier inflates std for all months. |
| Walk-forward scales each test fold with that fold's train-window mean/std | `training/evaluation.py:90-92` | Early folds (train < 2020) have tiny ICSA std, so 2020 test z-scores are in the hundreds and the forward filter treats them as near-certain. **Clipping must also be applied at scoring time**, not just fit time. |
| Scaler stats are saved as `growth_mean/scale`, `inflation_mean/scale` in `scalers.npz` | `training/build_model.py:121-125`, `prediction/model_loader.py:77-90` | Artifact format must change; old artifacts must still load. |
| Prediction applies the saved scaler | `prediction/predict.py` (~lines 155-170) | Must mirror any scaling/clipping change. `prediction/` cannot import `training/`. |
| Logistic benchmark uses its own per-fold `StandardScaler` | `training/benchmarks.py:173-182` | Must get the same robust scaling + clipping or the comparison is unfair. |
| Baseline averages raw features × sign | `training/baseline.py:19-22` | Needs scaled input (causal!). |
| `transform_params` is `dict[str, int]` | `common/feature_config.py:37`, `_feature_definition_from_dict` | Must relax for float/str params. |
| `TRANSFORMS` registry of single-series pd.Series→pd.Series functions | `common/transforms.py` | Adding a single-series transform needs no other code change. |
| Only multi-series feature is a ratio, special-cased in 4 places | `common/features.py:88`, `training/dataset.py:_build_feature_column` + `_asset_manifest_entries`, `prediction/predict.py` (`_build_daily_axis_df`, staleness/tolerance block ~270-320), `used_series_ids` | Generalizing means touching all of them; predict.py duplicates training logic deliberately. |
| `FeatureConfig.content_hash()` covers transform *name + params*, not code | `common/feature_config.py:84` | Changing an existing transform's behavior without renaming/bumping version leaves `dataset_id` unchanged. Always add a new name or new config version. |
| Daily series are resampled to monthly (`aggregation` = last/mean) then transformed; non-daily series (weekly/monthly) are transformed on the raw series then as-of looked up | `common/features.py:77-85` | For weekly series, `n_periods`/`window` params are in **weeks**, not months. Be explicit in feature names. |
| Rows with *any* null feature are dropped | `training/dataset.py:_build_axis_df` | A short-history series truncates the whole axis sample (e.g. WEI starts 2008). |
| Non-vintage point-in-time rule is `date + publication_lag_days <= asof` | `common/pit.py:39-41` | Survey data stamped to the 1st of a month needs a lag matching its real release (see Phase 5). |
| Evaluation-only label series: `CFNAIMA3`, `CPIAUCSL`, `USREC` | `ingestion/series_catalog.py:33-37` | **Label circularity risk** if CFNAI or headline CPI become features (see Phase 4). |
| Weekly ingest Lambda iterates `FIRST_PASS_SERIES` only | `prediction/handler.py`, `series_catalog.py` | New *feature* series go in `FIRST_PASS_SERIES`; evaluation-only series stay in `EVALUATION_SERIES`. |

### 0.3 Review verdicts on the original proposals

- **Clip with StandardScaler**: agree, but clipping z-scores after a plain `StandardScaler` leaves
  the inflated std in place (the root cause). Use a **robust fit (median/MAD or winsorized
  mean/std) and then clip**, and apply it identically at fit, walk-forward scoring, benchmark and
  live prediction.
- **Baseline unscaled**: agree. Fix with a **causal** (expanding-window / fold-train) scaler, not a
  full-sample scaler (that would leak the future into the benchmark).
- **Surprise/acceleration features**: agree; most are single-series transforms (easy).
- **New growth/inflation series**: agree on direction; verify history length/licensing first, and
  resolve label circularity.
- **Lower-lag features**: agree, highest value. Published nowcasts (GDPNow, NY Fed Staff Nowcast)
  are the riskiest because backcasts get re-estimated (no true point-in-time history); do those
  last or skip.
- **Multi-series combinations**: defer until a concrete feature needs one; FRED-provided spread
  series cover many cases.

## 1. Open decisions (ask the user before/at the start of the relevant phase)

1. **Which two features are the "weakly-justified" ones to remove?** Default guess: `t10y2y_level`
   and `dfii10_1m_change` (its sign is already flagged as debatable in `feature_config.py`
   comments). Confirm before Phase 3.
2. **Label circularity**: keep `CFNAIMA3`/`CPIAUCSL` as evaluation labels (then CFNAI and headline
   CPI **cannot** be features), or switch labels? Decide before Phase 4/5.
3. **`T10Y2Y`**: drop, keep as level, or convert to change/surprise? Decide via ablation in Phase 3.
4. **Training window**: is a shorter window (e.g. 2008+) acceptable if it unlocks WEI/HY-spread
   features? Decide in Phase 4.

## 2. Ground rules (apply to every phase)

- Run the repo's own checks before every commit: lint, format, `mypy`, `pytest` (see
  `pyproject.toml`/`ARCHITECTURE.md` for the exact commands).
- **Never** use look-ahead: all new logic must go through `point_in_time_series`; scaler/clip stats
  for an evaluation fold come from that fold's training rows only.
- `common/` must not import `hmmlearn`/`sklearn`-training-only packages (shared with the batch
  Lambda). `prediction/` must not import `training/`. Where logic is needed by both, put it in
  `common/`.
- Never change an existing transform's behavior in place; add a new name and/or bump the config
  version. Bump `FeatureConfig.version` at the end of every feature-affecting phase (v2, v3...).
- Keep `FEATURE_CONFIG_V1` intact for reproducing the current baseline numbers.
- Don't alter published model artifacts in place. New artifact format must be backward compatible
  on load.
- One logical change per commit/PR; include before/after evaluation tables in the PR description.

## Phase 0 - Measurement harness (do first)

**Goal:** a repeatable yardstick so every later phase is judged on the same terms.

Tasks:
1. Inspect `scripts/evaluate_model.py`, `training/harness.py`, `training/report.py`,
   `training/scoring.py`, `training/labels.py`, `training/benchmarks.py` and document in
   `ARCHITECTURE.md` (or a short note) which metrics exist (log loss, agreement with baseline,
   logistic benchmark, skipped folds).
2. Make it possible to run the **whole evaluation for an arbitrary `FeatureConfig`** (e.g. a
   `--feature-config v1|v2|...` flag resolving to a registry of configs in
   `common/feature_config.py`). Keep default = v1.
3. Add a **state-occupancy / state-means report** per axis per fold: fraction of months in each
   state, the per-feature emission means in z-units, and the dates of the first/last months in each
   state. Specifically make "is DOWN just COVID?" (growth) and "is DOWN/UP just 2021-23?"
   (inflation) a visible output.
4. Record v1 results as the reference (commit as a JSON/markdown artifact under a `docs/` or
   `reports/` path the repo already uses, or the PR description).

Acceptance: running the evaluation twice on v1 gives identical numbers (fixed `random_state`), and
the occupancy report shows the current COVID-state behavior (confirming the hypothesis before we
fix it).

## Phase 1 - Robust scaling + clipping (end to end)

**Goal:** outliers no longer create their own state; scoring-time z-scores are bounded.

Design:
- Add a small shared module, e.g. `common/scaling.py` (pure numpy; no sklearn), defining:
  - `fit_robust_scaler(X, clip=4.0, method="winsorized")` returning `center`, `scale`, `clip`.
    Prefer winsorized mean/std (1st/99th percentile) or median/MAD scaled by 1.4826; pick via
    experiment in this phase and record why.
  - `apply_scaler(X, center, scale, clip)` = `clip((X - center)/scale, -clip, +clip)`.
  - Guard `scale == 0` (constant feature).
- `FitResult` gains `scaler_clip` (and optionally `scaler_method`); `fit_hmm` uses the shared
  module instead of `StandardScaler`. Keep the same field names `scaler_mean`/`scaler_scale`
  (center/scale) to limit churn.
- Apply the identical `apply_scaler` in: `training/evaluation.py:90-92`,
  `training/benchmarks.py:173-182` (logistic benchmark: fit with the same fit function on fold
  train rows), and `prediction/predict.py` (~155-170).
- Artifact: add `growth_clip` / `inflation_clip` to `scalers.npz` in `training/build_model.py`;
  `prediction/model_loader.py` must default to "no clip" (`np.inf`) when keys are missing so old
  published models still load. Update `ARCHITECTURE.md` artifact docs.

Tests (extend `tests/training/test_trainer.py`, `test_evaluation.py`, `test_benchmarks.py`,
`tests/prediction/test_model_loader.py`, `test_predict.py`, add `tests/common/test_scaling.py`):
- Synthetic series with a few huge outliers: fitted state is not "outliers only"; scale is not
  inflated beyond tolerance.
- Training-time scaled observations == prediction-time scaled observations for the same raw row.
- Old-format `scalers.npz` (no clip keys) loads and behaves as before.
- Clipping is symmetric and bounded; zero-variance feature does not NaN.

Experiments (real data, local session):
- Compare: plain StandardScaler (v1), clip-only, robust+clip with clip in {3, 4, 5}.
- Report occupancy table, log loss vs. labels, baseline agreement, number of skipped folds.
- Specifically confirm the 2020 fold no longer produces near-certain DOWN posteriors purely from
  one ICSA print, and that the growth DOWN state is not only Mar-Jun 2020.

Acceptance: outlier unit test passes; occupancy shows DOWN state spans more than COVID; no
regression in log loss beyond noise; old artifacts load.

## Phase 2 - Fix the baseline

**Goal:** a meaningful benchmark for "agreement with baseline".

Tasks:
1. Change `training/baseline.py::baseline_signal` to accept pre-scaled (causal) values, or
   accept scaler params and apply them. Preferred: compute the scaler inside the walk-forward loop
   (fold-train rows only), so the baseline is evaluated on the same folds as the HMM, using the
   shared `apply_scaler` from Phase 1.
2. Update callers (`training/harness.py`, `scripts/evaluate_model.py`, report code).
3. Keep the sign-weighted average composite and the `>= 0 -> UP` tie rule.

Tests (`tests/training/test_baseline.py`): with one feature in huge units and others small, the
composite is not dominated by the large-unit feature; baseline is unchanged when the scaler is
fit only on rows before the test fold (no look-ahead: mutating future rows doesn't change earlier
signals).

Acceptance: ICSA no longer dominates; agreement-with-baseline metric is recomputed and recorded.

## Phase 3 - Transforms, config typing, pruned v2 feature set

**Goal:** richer single-series transforms, pruned/ablated feature set -> `FEATURE_CONFIG_V2`.

Tasks:
1. Relax `transform_params` typing to `dict[str, int | float | str]` in `FeatureDefinition`,
   `_feature_definition_from_dict` (remove the unchecked-int assumption), and
   `transforms.apply_transform`. Verify JSON round trip (`to_dict`/`from_dict`) and
   `content_hash` stability for v1 (hash of v1 must not change, or the change must be understood
   and noted).
2. Add transforms to `TRANSFORMS` (new names only):
   - `detrend`: `x - rolling_mean(x, window)` (e.g. window=12), "deviation from trailing 12m trend".
   - `second_diff`: change of the change (acceleration), with `n_periods`.
   - `rolling_zscore`: `(x - rolling_mean)/rolling_std` over `window`.
   - Optionally `pct_change_detrend` (change minus trailing mean change) if needed.
   Each must be strictly backward-looking (only trailing windows) and have unit tests including a
   "no look-ahead" property test (appending future rows doesn't change earlier outputs).
3. Build `FEATURE_CONFIG_V2` from V1 with these edits, each as an **ablation**:
   - Remove `t10yie_1m_change` or `t10yie_3m_change` (near-duplicates) - keep the better by
     ablation.
   - Remove the two weakly-justified features (**confirm with user**, see section 1).
   - `T10Y2Y`: test (a) dropped, (b) as level, (c) as 3m change, (d) as `detrend`.
   - Add surprise/acceleration variants of the strongest features (e.g. `payems` / `indpro` /
     `cpilfesl` deviation from trailing 12m trend; `icsa` acceleration).
4. Register configs (e.g. `FEATURE_CONFIGS = {"v1": ..., "v2": ...}`) and wire the Phase 0
   selector.
5. Update `ARCHITECTURE.md` transform list.

Process: one change at a time vs. V1+Phase-1 scaling; keep a results table (feature set -> log
loss, occupancy, skipped folds, baseline agreement). With ~300 monthly observations and diag
covariance, adding many features overfits - prefer few, strong, non-redundant ones per axis.

Acceptance: V2 selected by ablation with documented evidence; hash/versions distinct from V1;
tests pass.

## Phase 4 - Data availability spike (research only, no model code changes)

**Goal:** decide which candidate series are actually usable, using real vendor access.

For each candidate, record in a table (put it in `docs/` or the PR): series id, source
(fred/alfred/yfinance), native frequency, first date, last date, revision-prone
(vintage_aware?), real publication lag, licensing/history caveats, axis, proposed sign.

Candidates to check:

- **Growth:** CFNAI (`CFNAI`), high-yield OAS (`BAMLH0A0HYM2`), unemployment rate (`UNRATE`),
  building permits (`PERMIT`), retail sales (`RSAFS`/`RRSFS`).
- **Inflation:** headline CPI (`CPIAUCSL`), PCE (`PCEPI`/`PCEPILFE`), 5y5y forward breakeven
  (`T5YIFR`), median CPI (`MEDCPIM158SFRBCLE`) / trimmed-mean (`TRMMEANCPIM158SFRBCLE`), wages
  (`CES0500000003`), UMich inflation expectations (`MICH`).
- **Daily market:** SPY, VIX (`VIXCLS`), 2y/10y yields (`DGS2`, `DGS10`), Brent
  (`DCOILBRENTEU`), dollar index (`DTWEXBGS`), small-vs-large cap (e.g. IWM/SPY), cyclical vs
  defensive sectors (e.g. XLY/XLP, XLI/XLU) via yfinance.
- **Weekly official:** Weekly Economic Index (`WEI`), Chicago Fed financial conditions (`NFCI`),
  St. Louis stress (`STLFSI4`), EIA gasoline (`GASREGW`).
- **Published nowcasts:** Atlanta Fed GDPNow, NY Fed Staff Nowcast, Cleveland Fed inflation
  nowcast - check *how much genuine point-in-time vintage history* exists (published backcasts
  were re-estimated). ALFRED may hold vintages for some (e.g. `GDPNOW`); if not truly PIT, exclude.
- **In-month surveys:** Empire State (`GACDISA066MSFRBNY`), Philly Fed (`GACDFSA066MSFRBPHI`),
  UMich preliminary, Indeed job postings (`IHLIDXUS`).
- **Fast private data:** Manheim, NY Fed supply-chain index, card spending - likely proprietary/
  short; record and probably skip.

Specific checks:
- `BAMLH0A0HYM2`: ICE licensing restricts FRED history to roughly the last ~3 years; if too short,
  consider an alternative (e.g. ETF-based spread proxies via yfinance: HYG vs IEF/LQD) or skip.
- Series start dates vs. the training window: `dataset._build_axis_df` drops any row with a null
  feature, so a short-history series truncates the entire axis sample (e.g. `WEI` begins 2008).
  Quantify the effective training months for each candidate set and decide the window
  trade-off (open decision #4).
- **Label circularity (open decision #2):** `EVALUATION_SERIES` currently labels with `CFNAIMA3`,
  `CPIAUCSL`, `USREC`. If `CFNAI` or headline CPI are used as features, either exclude them as
  features or change the labels. Do **not** do both. Prefer features that are not near-duplicates
  of the label series.
- ALFRED vintage availability for each revision-prone series (needed for `vintage_aware=True`).

Deliverable: a decision table (keep/skip + reason) and any new provider code needed in
`ingestion/providers/` (e.g. a provider for non-FRED sources). Add retained series to
`FIRST_PASS_SERIES` in `ingestion/series_catalog.py` and ingest them into the lake with
`scripts/ingest_data.py` (confirm the ingest Lambda config/terraform schedules include them).

## Phase 5 - Lower-lag features (in order of confidence; ablate after each group)

**Goal:** reduce nowcast lag. Each group: add `FeatureDefinition`s to `FEATURE_CONFIG_V3` (or
V2.x), run ablation vs. the previous best.

Per-feature requirements:
- Correct `native_frequency`, `vintage_aware`, `publication_lag_days`, `aggregation`, `sign`.
- For weekly series, remember transforms run on the raw weekly series (no monthly resample), so
  `n_periods`/`window` are in weeks; for daily series they run after the monthly resample.
- Also check `prediction/predict.py`'s per-frequency staleness tolerance dict (~line 294) covers
  the frequency used.
- Inference is mid-month: verify partial-month behavior for each new series (daily mean/last over
  a partial month is automatic; weekly/monthly use as-of lookup).

Groups:
1. **Daily market:** VIX (level or change, `aggregation="mean"` or "last" - ablate), HY spread (if
   history allows), `T5YIFR`, 2y yield change, dollar index, Brent, cyclical/defensive and
   small/large ratios. Note: ratios currently use the one hardcoded ratio path
   (`numerator_series_id`/`denominator_series_id`) which already works for yfinance pairs.
2. **Weekly official:** `NFCI`, `STLFSI4`, `GASREGW` (feeds headline CPI directly), `WEI` (if the
   shorter window is accepted).
3. **In-month surveys:** Empire/Philly. Their observation `date` is stamped to the first of the
   reference month but released mid-month; set `publication_lag_days` to the real release lag
   (~15-20 days after the stamped date, non-vintage-aware rule `date + lag <= asof`) so a
   month-end decision date sees the current month's survey but a mid-month inference day does
   not look ahead. Add a unit test for this boundary in `tests/common/test_features.py`/PIT
   tests.
4. **Published nowcasts:** only if Phase 4 found genuine point-in-time vintages. Otherwise skip
   and document why.

Acceptance per group: ablation table, no increase in skipped folds, no log-loss regression,
prediction-path parity test (`tests/prediction/test_predict.py`) passing with the new series.

## Phase 6 - Multi-series feature support (only if a concrete feature needs it)

Currently only `numerator/denominator` ratios are supported. Only do this if Phase 4/5 finds a
valuable feature that FRED doesn't already publish as a single series (spreads, differences,
weighted blends, regression residuals).

Design (when needed):
- Replace `numerator_series_id`/`denominator_series_id` with `input_series_ids: tuple[str, ...]`
  and a `combine: str` registry name (e.g. `ratio`, `diff`, `weighted`), with a `combine_params`
  dict. Keep the ratio fields working (or migrate v1 configs on load) so old `config.json` files
  still deserialize.
- Generalize `build_ratio_feature_value` into `build_combined_feature_value`.
- Update **all four** special-case sites together: `training/dataset.py`
  (`_build_feature_column`, `_asset_manifest_entries`), `common/features.py`,
  `prediction/predict.py` (`_build_daily_axis_df`, `used_series_ids` use, live-source provider
  lookup, per-series staleness tolerance - all branch on `is_ratio`), and
  `common/feature_config.py::used_series_ids`. `predict.py` deliberately duplicates the training
  logic, so every change must be made twice; add a parity test that builds one decision date's
  feature vector via both paths and asserts equality.
- `content_hash` must include the new fields.
- Optional (separate, later): a post-feature composite step ("average these features into one").

## Phase 7 - Integration, release, and documentation

1. Choose final config (V2/V3) with the evidence table; keep V1 reproducible.
2. Retrain, evaluate, and (via existing `scripts/build_model`/`publish_model.py`/`promote_model.py`
   flow) publish a new model version; verify the Lambda predicts with the new artifact
   (including the new `*_clip` keys) and the old artifact still loads.
3. Update `ARCHITECTURE.md` (scaler/clip, new transforms, new series, artifact format, label
   policy) and `README.md` if user-facing behavior changed.
4. Version bump per repo convention (`scripts/bump_calver.py`).

## Appendix A - Test checklist

- `tests/common/test_scaling.py` (new): robustness, clip bounds, zero variance.
- `tests/common/test_transforms.py`: each new transform, backward-looking property.
- `tests/common/test_feature_config.py`: float/str params round trip; hash stability for v1.
- `tests/training/test_trainer.py`, `test_evaluation.py`, `test_benchmarks.py`,
  `test_baseline.py`, `test_build_model.py`: scaler/clip plumbing and artifact keys.
- `tests/prediction/test_model_loader.py`, `test_predict.py`: backward-compat load and
  train/predict scaling parity.
- PIT/feature tests: negative/short publication lags and mid-month inference.

## Appendix B - Risks

- **Overfitting** from adding many features to a ~300-point, 2-state diag-covariance HMM:
  ablate one at a time; prefer fewer, stronger features.
- **Sample truncation** from short-history series (affects all months of the axis).
- **Look-ahead** from nowcasts re-estimated in hindsight and from non-causal scaler fitting.
- **Label circularity** (CFNAI/CPI as features while also being evaluation labels).
- **Artifact compatibility**: new `*_clip` fields; loader must default for old models.
- **Hash/versioning**: modifying transform code without renaming leaves `content_hash` unchanged.
- **Vendor quirks**: ICE licensing on spreads, ALFRED vintage gaps, yfinance symbol changes.
