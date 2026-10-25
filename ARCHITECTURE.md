# Architecture

A map of where the code lives and how data moves through it. All diagrams are
[Mermaid](https://mermaid.js.org/) — GitHub renders them inline, and they're plain text
you can edit in place or paste into <https://mermaid.live> for a visual editor.

- [1. Deployables and import rules](#1-deployables-and-import-rules)
- [2. Code diagrams (C4 level 4)](#2-code-diagrams-c4-level-4)
- [3. Data flow](#3-data-flow)
- [4. S3 / DynamoDB layout](#4-s3--dynamodb-layout)
- [5. Where to find things](#5-where-to-find-things)

---

## 1. Deployables and import rules

Four things run, and the package boundaries exist to keep each one's dependency set small.

```mermaid
flowchart LR
    subgraph local["Local only<br/>never deployed"]
        S_ING["scripts/ingest_data.py"]
        S_TRAIN["scripts/train_model.py"]
        S_PUB["scripts/publish_model.py"]
        S_PROM["scripts/promote_model.py"]
        S_PRED["scripts/run_prediction.py"]
    end

    subgraph batch["Batch Lambda<br/>no hmmlearn"]
        H["prediction/handler.py<br/>handler()"]
    end

    subgraph server["Server Lambda<br/>no pandas / numpy"]
        MAIN["server/main.py<br/>Mangum handler"]
    end

    TRAINING["training/<br/>(hmmlearn, sklearn)"]
    PREDICTION["prediction/"]
    INGESTION["ingestion/"]
    SERVER["server/"]
    COMMON["common/<br/>settings · features · registry · AWS clients"]

    S_ING --> INGESTION
    S_TRAIN --> TRAINING
    S_PUB --> TRAINING
    S_PROM --> TRAINING
    S_PRED --> PREDICTION
    H --> INGESTION
    H --> PREDICTION
    MAIN --> SERVER

    TRAINING --> INGESTION
    TRAINING --> COMMON
    PREDICTION --> INGESTION
    PREDICTION --> COMMON
    INGESTION --> COMMON
    SERVER --> COMMON

    TRAINING -. "never imported by" .-x PREDICTION
    PREDICTION -. "never imported by" .-x SERVER
```

**Hard rules**

| Rule | Why |
|---|---|
| `prediction/` never imports `training/` | `training/trainer.py` imports `hmmlearn`, which the batch Lambda image excludes. Inference uses the pure-NumPy `common/forward_filter.py` instead. `prediction/predict.py` deliberately duplicates ~15 lines of axis-matrix assembly from `training/dataset.py` to respect this. |
| `server/` never imports `pandas` / `numpy` / `hmmlearn` | It is a zip Lambda that only reads JSON from S3 and items from DynamoDB. |
| Training reads the **lake only** | `training/dataset.py` never calls a provider; ingestion is the sole writer of `data/`. |
| Providers never write to S3 | `DataProvider.fetch` returns a DataFrame; `ingestion/ingest.py` + `lake.py` own validation and writes. |
| Models are plain `.npz` / `.json` | No pickle anywhere — `prediction/model_loader.py` rebuilds everything from NumPy arrays and JSON. |

---

## 2. Code diagrams (C4 level 4)

Each diagram covers one package. Module-level functions are shown as a `<<module>>` box
(Mermaid has no native "module" shape); dataclasses are shown as `<<dataclass>>`. Only
public classes and functions are listed — leading-underscore helpers are omitted.
Arrows read "uses".

### 2.1 `common/` — shared foundation

```mermaid
classDiagram
    direction LR

    class Settings {
        <<pydantic BaseSettings>>
        MRD_ env prefix
        aws_region
        s3_bucket_name
        dynamodb_table_name
        fred_api_key
        alpha_vantage_api_key
        batch_lambda_function_name
        cors_allowed_origins
        staleness_max_age_hours
    }
    class settings_py {
        <<module settings.py>>
        get_settings() Settings
    }

    class aws_clients {
        <<module aws_s3 / aws_dynamodb / aws_lambda>>
        get_s3_client()
        get_dynamodb_client()
        get_dynamodb_resource()
        get_lambda_client()
    }

    class model_registry {
        <<module model_registry.py>>
        ModelStage
        put_model_item(client, table, item)
        get_model(client, table, model_id)
        get_production_model(client, table)
        list_models_by_stage(client, table, stage)
        list_all_models(client, table)
        promote_model(client, table, model_id)
    }

    class FeatureDefinition {
        <<dataclass frozen>>
        name
        axis : Axis
        series_id
        source
        native_frequency : Frequency
        vintage_aware
        publication_lag_days
        transform
        transform_params
        sign
        aggregation
        numerator_series_id
        denominator_series_id
        is_ratio() bool
    }
    class FeatureConfig {
        <<dataclass frozen>>
        version
        features : tuple~FeatureDefinition~
        to_dict()
        from_dict(data)$
        content_hash() str
    }
    class feature_config_py {
        <<module feature_config.py>>
        Axis : GROWTH / INFLATION
        Frequency : DAILY / WEEKLY / MONTHLY
        FEATURE_CONFIG_V1
        used_series_ids(config) list
    }

    class features {
        <<module features.py>>
        build_feature_value(raw_df, feature, eval_date)
        build_ratio_feature_value(num_df, den_df, feature, eval_date)
        as_of_lookup(series, date)
        latest_value(series)
    }
    class pit {
        <<module pit.py>>
        point_in_time_series(raw_df, eval_date, ...)
    }
    class transforms {
        <<module transforms.py>>
        apply_transform(series, name, params)
        identity / pct_change / diff
        pct_change_annualized
        rolling_mean_diff
    }
    class forward_filter {
        <<module forward_filter.py>>
        filtered_posterior(X, start_prob, transmat, means, vars)
    }
    class quadrants {
        <<module quadrants.py>>
        QUADRANT_NAMES
        quadrant_probabilities(p_growth_up, p_inflation_up)
        best_regime(quadrants)
    }
    class misc {
        <<module>>
        configure_logging(level)
        JSONFormatter
        get_git_sha()
        DISCLAIMER
    }

    settings_py --> Settings
    aws_clients ..> Settings : region
    FeatureConfig "1" *-- "many" FeatureDefinition
    feature_config_py --> FeatureConfig
    features --> pit
    features --> transforms
    features --> FeatureDefinition
    model_registry --> aws_clients : client passed in
```

### 2.2 `ingestion/` — vendors → lake

```mermaid
classDiagram
    direction TB

    class DataProvider {
        <<abstract>>
        source : str
        fetch(series_id, start, end, asof) DataFrame
    }
    class FredProvider {
        fetch() latest values via fredapi
    }
    class AlfredProvider {
        fetch() every historical vintage
    }
    class YFinanceProvider {
        fetch() daily close, no vintages
    }
    class AlphaVantageProvider {
        fetch() wired but unused by first-pass series
    }
    class factory {
        <<module providers/factory.py>>
        PROVIDER_NAMES
        build_provider(name) DataProvider
    }

    class SeriesSpec {
        <<dataclass frozen>>
        series_id
        source
        axis
    }
    class series_catalog {
        <<module series_catalog.py>>
        FIRST_PASS_SERIES : list~SeriesSpec~
    }

    class ingest {
        <<module ingest.py>>
        ingest_series(series_id, source, start, end, bucket, asof)
        ingest_incremental(series_id, source, bucket, end, overlap_days=30)
    }
    class lake {
        <<module lake.py>>
        write_to_lake(s3, bucket, df)
        read_series(s3, bucket, series_id) DataFrame
        get_last_ingested_date(s3, bucket, series_id)
    }
    class schema {
        <<module schema.py>>
        CANONICAL_COLUMNS
        NATURAL_KEY
        validate_schema(df) DataFrame
        to_naive_utc(series)
    }
    class retry {
        <<module retry.py>>
        with_retries(fn, ...)
    }

    DataProvider <|-- FredProvider
    DataProvider <|-- AlfredProvider
    DataProvider <|-- YFinanceProvider
    DataProvider <|-- AlphaVantageProvider
    factory ..> DataProvider : creates
    factory ..> FredProvider
    factory ..> AlfredProvider
    factory ..> YFinanceProvider
    factory ..> AlphaVantageProvider
    series_catalog "1" *-- "many" SeriesSpec

    ingest --> factory : build_provider
    ingest --> lake : write_to_lake, get_last_ingested_date
    lake --> schema : validate_schema
    FredProvider ..> retry
    AlfredProvider ..> retry
    YFinanceProvider ..> retry
    AlphaVantageProvider ..> retry
    YFinanceProvider ..> schema : to_naive_utc
```

### 2.3 `training/` — lake → published model (runs locally)

```mermaid
classDiagram
    direction TB

    class DatasetResult {
        <<dataclass>>
        growth_df
        inflation_df
        manifest : dict
    }
    class dataset {
        <<module dataset.py>>
        build_dataset(s3, bucket, start, end, feature_config) DatasetResult
    }

    class HMMHyperparameters {
        <<dataclass frozen>>
        n_restarts
        covariance_type
        min_state_duration_months
        composite_separation_margin
        initial_window_months
        refit_every_months
        random_state
        to_dict()
    }
    class build_model {
        <<module build_model.py>>
        DEFAULT_HYPERPARAMETERS
        train_and_save_model(output_dir, growth_df, inflation_df, feature_config, hyperparameters)
    }

    class FitResult {
        <<dataclass>>
        feature_names
        scaler_mean / scaler_scale
        start_prob / transition_matrix
        means / variances
        state_labels : UP or DOWN
        composites
        log_likelihood
    }
    class trainer {
        <<module trainer.py>>
        fit_hmm(df, feature_defs, n_restarts, ...) FitResult
    }
    class evaluation {
        <<module evaluation.py>>
        walk_forward_evaluate(df, feature_defs, ...) DataFrame
    }
    class baseline {
        <<module baseline.py>>
        baseline_signal(axis_df, feature_defs)
    }
    class report {
        <<module report.py>>
        compare_to_baseline(walk_forward, baseline)
        combine_quadrant_probabilities(growth, inflation)
        episode_report(quadrants, value_column)
        render_model_card(...)
    }
    class publisher {
        <<module publisher.py>>
        PublishStage
        publish_model(local_dir, manifest, s3, bucket, ddb, table, stage) model_id
    }
    class promote {
        <<module promote.py>>
        invoke_batch_lambda_best_effort(lambda_client, fn_name, model_id)
    }

    dataset --> DatasetResult
    build_model --> HMMHyperparameters
    build_model --> trainer : fit_hmm (full history, saved)
    build_model --> evaluation : walk_forward_evaluate (diagnostic only)
    build_model --> baseline
    build_model --> report
    trainer --> FitResult
    evaluation --> trainer : refit per fold
    evaluation ..> forward_filter : causal posteriors
    dataset ..> lake : read_series
    dataset ..> features : build_feature_value
    dataset ..> feature_config_py : used_series_ids
    publisher ..> model_registry : put_model_item
    promote ..> aws_lambda : async invoke

    class lake {
        <<ingestion.lake>>
    }
    class features {
        <<common.features>>
    }
    class feature_config_py {
        <<common.feature_config>>
    }
    class forward_filter {
        <<common.forward_filter>>
    }
    class model_registry {
        <<common.model_registry>>
    }
    class aws_lambda {
        <<AWS Lambda>>
    }
```

### 2.4 `prediction/` — batch Lambda (no `hmmlearn`)

```mermaid
classDiagram
    direction TB

    class handler {
        <<module handler.py>>
        handler(event, context) dict
        action ingest → _handle_ingest
        action predict → _handle_predict
    }

    class HMMArtifact {
        <<dataclass frozen>>
        start_prob / transition_matrix
        means / variances
        feature_names
        state_labels
        up_state : int
    }
    class ModelArtifact {
        <<dataclass frozen>>
        growth : HMMArtifact
        inflation : HMMArtifact
        growth_scaler_mean / scale
        inflation_scaler_mean / scale
        feature_config : FeatureConfig
    }
    class model_loader {
        <<module model_loader.py>>
        load_model_artifact(s3, bucket, model_id) ModelArtifact
    }

    class PredictionResult {
        <<dataclass>>
        model_id
        dataset_id
        asof
        generated_at
        observation_date
        is_stale
        warnings
        history : DataFrame
    }
    class predict {
        <<module predict.py>>
        DEFAULT_LOOKBACK_DAYS = 730
        run_prediction(s3, ddb, bucket, table, model_id, source, start, end, asof) PredictionResult
    }
    class outputs {
        <<module outputs.py>>
        to_latest_json(result)
        to_history_json(result)
        write_prediction_outputs(s3, bucket, result)
    }

    handler --> predict : predict action
    handler --> outputs : predict action
    handler --> ingest_incremental : ingest action
    predict --> model_loader
    predict --> PredictionResult
    model_loader --> ModelArtifact
    ModelArtifact "1" *-- "2" HMMArtifact
    predict ..> common_registry : get_model / get_production_model
    predict ..> common_features : build_feature_value
    predict ..> forward_filter : filtered_posterior
    predict ..> quadrants : quadrant_probabilities
    predict ..> lake_read : read_series (source=lake)
    predict ..> provider_factory : build_provider (source=live)
    outputs --> PredictionResult

    class ingest_incremental {
        <<ingestion.ingest>>
    }
    class common_registry {
        <<common.model_registry>>
    }
    class common_features {
        <<common.features>>
    }
    class forward_filter {
        <<common.forward_filter>>
    }
    class quadrants {
        <<common.quadrants>>
    }
    class lake_read {
        <<ingestion.lake>>
    }
    class provider_factory {
        <<ingestion.providers.factory>>
    }
```

### 2.5 `server/` — read-only API (server Lambda)

```mermaid
classDiagram
    direction LR

    class main {
        <<module main.py>>
        handler = Mangum(app)
    }
    class app {
        <<module app.py>>
        app : FastAPI
        CORS from Settings
    }
    class routes {
        <<module routes.py · prefix /v1>>
        GET /health
        GET /models
        GET /models/model_id
        GET /predictions/latest
        GET /predictions/history
    }
    class schemas {
        <<module schemas.py · pydantic>>
        HealthResponse
        ModelSummaryResponse
        QuadrantProbabilities
        LatestSnapshot
        LatestPredictionResponse
        HistoryRecord
        HistoryResponse
    }
    class reads {
        <<module reads.py>>
        cached_get_model(table, id)
        cached_get_production_model(table)
        cached_list_models(table, stage)
        resolve_model(table, model_id)
        cached_read_json(bucket, key)
    }
    class TTLCache {
        get_or_set(key, factory)
        clear()
    }

    main --> app
    app --> routes : include_router
    routes --> schemas : response_model
    routes --> reads
    reads --> TTLCache : CACHE singleton
    reads ..> model_registry : DynamoDB (common)
    reads ..> aws_s3 : S3 GetObject (common)

    class model_registry {
        <<common.model_registry>>
    }
    class aws_s3 {
        <<common.aws_s3>>
    }
```

The route layer ORs the batch job's generation-time `is_stale` with a request-time
check (`generated_at` older than `Settings.staleness_max_age_hours`).

---

## 3. Data flow

### 3.1 End to end: vendors → lake → model → predictions → API

```mermaid
flowchart TB
    subgraph vendors["External vendors"]
        FRED["FRED<br/>latest values"]
        ALFRED["ALFRED<br/>historical vintages"]
        YF["Yahoo Finance<br/>HG=F, GC=F"]
        AV["Alpha Vantage<br/>(wired, unused)"]
    end

    subgraph ingest["Ingestion — ingestion/"]
        PROV["DataProvider.fetch()<br/>Fred / Alfred / YFinance / AlphaVantage"]
        VAL["validate_schema()<br/>canonical columns, dedupe"]
        WRITE["write_to_lake()<br/>merge + compact per year"]
    end

    LAKE[("S3 lake<br/>data/asset=ID/year=YYYY/data.parquet")]

    subgraph train["Training — training/ (local)"]
        DS["build_dataset()<br/>point-in-time features, monthly"]
        FIT["train_and_save_model()<br/>fit_hmm ×2, walk-forward, baseline"]
        PUB["publish_model()"]
    end

    LOCAL["Local model dir<br/>growth_hmm.npz · inflation_hmm.npz · scalers.npz<br/>labels.json · config.json · metrics.json · model_card.md"]
    MODELS[("S3 models/model_id/")]
    MANI[("S3 datasets/dataset_id/manifest.json")]
    REG[("DynamoDB registry<br/>model_id → stage")]
    PROMO["promote_model()<br/>stage → production"]

    subgraph pred["Prediction — prediction/ (batch Lambda)"]
        LOAD["load_model_artifact()"]
        FEAT["daily features<br/>(same common/ code as training)"]
        FILT["filtered_posterior()<br/>causal NumPy forward filter"]
        QUAD["quadrant_probabilities()"]
        OUT["write_prediction_outputs()"]
    end

    PRED[("S3 predictions/model_id=ID/<br/>latest.json · history.json · snapshots/asof=DATE.json")]

    subgraph api["API — server/ (server Lambda)"]
        ROUTES["FastAPI /v1 routes<br/>+ TTL cache"]
    end
    CLIENT["client/index.html<br/>(GitHub Pages)"]

    FRED --> PROV
    ALFRED --> PROV
    YF --> PROV
    AV --> PROV
    PROV --> VAL --> WRITE --> LAKE

    LAKE --> DS --> FIT --> LOCAL --> PUB
    PUB --> MODELS
    PUB --> MANI
    PUB --> REG
    REG --> PROMO
    PROMO -. "async invoke with model_id" .-> LOAD

    REG --> LOAD
    MODELS --> LOAD
    LAKE --> FEAT
    LOAD --> FILT
    FEAT --> FILT --> QUAD --> OUT --> PRED

    REG --> ROUTES
    PRED --> ROUTES
    ROUTES --> CLIENT
```

Schedules (EventBridge → batch Lambda, see `terraform/main.tf`):

| Rule | Cadence | Event | Does |
|---|---|---|---|
| ingest | Sun 05:00 UTC | `{"action": "ingest"}` | `ingest_incremental` for every series in `FIRST_PASS_SERIES` (per-series failures logged, not fatal) |
| predict | Mon–Fri 06:30 UTC | `{"action": "predict"}` | `run_prediction` off the lake, then `write_prediction_outputs`; **never ingests** |

### 3.2 Ingestion sequence

```mermaid
sequenceDiagram
    autonumber
    participant EB as EventBridge (weekly)
    participant H as handler._handle_ingest
    participant I as ingest_incremental
    participant L as lake.py
    participant F as build_provider
    participant P as DataProvider
    participant V as Vendor API
    participant S3 as S3 lake

    EB->>H: {"action": "ingest"}
    loop each SeriesSpec in FIRST_PASS_SERIES
        H->>I: ingest_incremental(series_id, source, bucket)
        I->>L: get_last_ingested_date()
        L->>S3: read this year's + last year's partition
        S3-->>L: max(date)
        Note over I: start = last_date − 30d overlap<br/>(or today − 10y on first run)
        I->>F: build_provider(source)
        F-->>I: provider
        I->>P: fetch(series_id, start, end)
        P->>V: HTTP (with_retries)
        V-->>P: raw observations
        P-->>I: DataFrame
        I->>L: write_to_lake(df)
        L->>L: validate_schema()
        L->>S3: read existing year partition(s)
        L->>L: concat + dedupe on (series_id, date, realtime_start)
        L->>S3: overwrite data.parquet
    end
```

The 30-day overlap re-fetches a trailing window so recent revisions are picked up; the
dedupe key keeps the newest `ingested_at`, so re-running any range is idempotent.

### 3.3 Training sequence

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Developer
    participant TM as scripts/train_model.py
    participant DS as build_dataset
    participant S3 as S3 lake
    participant BM as train_and_save_model
    participant FH as fit_hmm
    participant WF as walk_forward_evaluate
    participant PM as scripts/publish_model.py
    participant PB as publish_model
    participant REG as DynamoDB registry
    participant PR as scripts/promote_model.py

    Dev->>TM: --start --end --output-dir
    TM->>DS: build_dataset(s3, bucket, start, end)
    DS->>S3: read_series() for each used_series_ids()
    DS->>DS: month-end decision dates →<br/>build_feature_value() per feature (point-in-time)
    DS-->>TM: DatasetResult(growth_df, inflation_df, manifest)
    TM->>BM: train_and_save_model(growth_df, inflation_df)
    BM->>FH: fit full history, growth axis
    BM->>FH: fit full history, inflation axis
    Note over FH: n_restarts random inits, StandardScaler,<br/>sanity checks (min state duration, composite separation)<br/>label states UP / DOWN by signed composite
    BM->>WF: expanding-window refits (diagnostic only)
    WF->>FH: refit per fold
    WF-->>BM: causal out-of-sample p_up per month
    BM->>BM: baseline_signal, compare_to_baseline,<br/>combine_quadrant_probabilities, episode_report
    BM-->>TM: writes 7 files to output-dir
    TM-->>Dev: model dir + dataset manifest

    Dev->>PM: publish (stage = staging by default)
    PM->>PB: publish_model(local_dir, manifest, ...)
    PB->>PB: model_id = mrd-YYYYMMDD-gitsha
    PB->>S3: upload models/model_id/*
    PB->>S3: put datasets/dataset_id/manifest.json
    PB->>REG: put_model_item(stage=staging)

    Dev->>PR: promote model_id
    PR->>REG: promote_model() → production (previous one demoted)
    PR-->>PR: invoke_batch_lambda_best_effort(model_id)
```

Only the full-history fit is saved; the walk-forward fits exist to produce honest
out-of-sample metrics in `metrics.json` and `model_card.md`.

### 3.4 Prediction and serving sequence

```mermaid
sequenceDiagram
    autonumber
    participant EB as EventBridge (daily) / promote script
    participant H as handler._handle_predict
    participant RP as run_prediction
    participant REG as DynamoDB registry
    participant ML as load_model_artifact
    participant S3 as S3
    participant OUT as write_prediction_outputs
    participant Browser as client (browser)
    participant API as server routes
    participant C as TTLCache

    EB->>H: {"action": "predict"} or {"model_id": ...}
    H->>RP: run_prediction(model_id?, source="lake")
    RP->>REG: get_production_model() or get_model(model_id)
    RP->>ML: load_model_artifact(model_id)
    ML->>S3: models/model_id/{labels,config}.json, *.npz
    ML-->>RP: ModelArtifact (no pickle)
    RP->>S3: read_series() for each series in feature_config
    Note over RP: Per series: stale if latest realtime_start is older<br/>than its frequency tolerance (3d / 10d / 40d)
    RP->>RP: daily build_feature_value() for growth + inflation<br/>(last 730 days)
    RP->>RP: standardize → filtered_posterior() per axis →<br/>p_growth_up, p_inflation_up
    RP->>RP: quadrant_probabilities(), argmax → regime + confidence
    RP-->>H: PredictionResult
    H->>OUT: write_prediction_outputs()
    OUT->>S3: predictions/model_id=ID/latest.json (overwrite)
    OUT->>S3: predictions/model_id=ID/history.json (overwrite)
    OUT->>S3: snapshots/asof=DATE.json (write-once)

    Browser->>API: GET /v1/predictions/latest
    API->>C: resolve_model() — cached production model
    C->>REG: get_production_model() on miss
    API->>C: cached_read_json(latest.json)
    C->>S3: GetObject on miss
    API->>API: is_stale ||= generated_at older than 36h
    API-->>Browser: LatestPredictionResponse
```

---

## 4. S3 / DynamoDB layout

One versioned S3 bucket (`Settings.s3_bucket_name`) plus one DynamoDB table.

| Location | Written by | Read by | Contents |
|---|---|---|---|
| `s3://…/data/asset=<series_id>/year=<yyyy>/data.parquet` | `ingestion/lake.py::write_to_lake` | `read_series` (training, prediction) | Raw vintaged observations: `date, value, realtime_start, source, series_id, ingested_at` |
| `s3://…/datasets/<dataset_id>/manifest.json` | `training/publisher.py` | humans, audits | Dataset id, feature-config hash, date range, per-asset row counts and date ranges, warnings |
| `s3://…/models/<model_id>/` | `training/publisher.py` | `prediction/model_loader.py` | `growth_hmm.npz`, `inflation_hmm.npz`, `scalers.npz`, `labels.json`, `config.json`, `metrics.json`, `model_card.md` |
| `s3://…/predictions/model_id=<id>/latest.json` | `prediction/outputs.py` | `server/routes.py` | Latest quadrant probabilities, regime, confidence |
| `s3://…/predictions/model_id=<id>/history.json` | `prediction/outputs.py` | `server/routes.py` | Daily history over the lookback window |
| `s3://…/predictions/model_id=<id>/snapshots/asof=<date>.json` | `prediction/outputs.py` | humans | Write-once record of what was published for that as-of date |
| DynamoDB `models` table | `common/model_registry.py` (publisher, promote script) | `prediction/predict.py`, `server/reads.py` | `model_id` → `stage` (`development`/`staging`/`production`/`archived`), `dataset_id`, S3 URIs, summary metrics. Exactly one `production` item. |

---

## 5. Where to find things

| I want to… | Look at |
|---|---|
| Add or change a data series | `ingestion/series_catalog.py` (what to ingest) **and** `common/feature_config.py` (`FEATURE_CONFIG_V1` — how it becomes a feature) |
| Add a new data vendor | Subclass `DataProvider` in `ingestion/providers/`, register in `providers/factory.py` |
| Change a feature transform | `common/transforms.py` (`apply_transform`), referenced by name from `FeatureDefinition.transform` |
| Change point-in-time / publication-lag logic | `common/pit.py`, `common/features.py` |
| Change HMM fitting or its sanity checks | `training/trainer.py` (`fit_hmm`), hyperparameters in `training/build_model.py` |
| Change evaluation / model card | `training/evaluation.py`, `training/baseline.py`, `training/report.py` |
| Change how inference works | `common/forward_filter.py`, `prediction/predict.py` |
| Change the published JSON shape | `prediction/outputs.py` **and** `server/schemas.py` (keep in sync) |
| Add an API endpoint | `server/routes.py`, `server/schemas.py`, `server/reads.py` |
| Change schedules, IAM, Lambdas | `terraform/main.tf`, `docker/batch-lambda.Dockerfile`, `scripts/package_lambda.sh` |
| Change CI/CD | `.github/workflows/deploy-terraform.yml`, `deploy-client.yml` |
| Run something by hand | `scripts/` — `ingest_data.py`, `train_model.py`, `publish_model.py`, `promote_model.py`, `run_prediction.py` (thin wrappers, no logic of their own) |

Tests mirror the source tree: `tests/common`, `tests/ingestion` (+ `providers/`),
`tests/training`, `tests/prediction`, `tests/server`.
