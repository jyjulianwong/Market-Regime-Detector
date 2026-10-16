import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum


class Axis(StrEnum):
    GROWTH = "growth"
    INFLATION = "inflation"


class Frequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass(frozen=True)
class FeatureDefinition:
    """One entry in a `FeatureConfig` — brief section 2.4: "every feature definition
    (source series, transform, lag, sign) lives in a versioned FeatureConfig."

    `sign` is the per-feature sign table from section 2.3's state labelling: +1 if a
    higher transformed value means this feature's axis is "up", -1 otherwise. It
    lives here, not in training code, so flipping an interpretation never requires a
    code change — just a new `FeatureConfig` version.
    """

    name: str
    axis: Axis
    series_id: str
    source: str
    native_frequency: Frequency
    vintage_aware: bool
    publication_lag_days: int
    transform: str
    transform_params: dict[str, int]
    sign: int
    aggregation: str = "last"
    # Set only for the one multi-series feature (copper/gold momentum) — a human
    # readable `series_id` like "HG=F/GC=F" plus these two so callers know to fetch
    # both legs and use `build_ratio_feature_value` instead of `build_feature_value`.
    numerator_series_id: str | None = None
    denominator_series_id: str | None = None

    def __post_init__(self) -> None:
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be 1 or -1, got {self.sign}")
        has_ratio_legs = (
            self.numerator_series_id is not None or self.denominator_series_id is not None
        )
        if has_ratio_legs and (
            self.numerator_series_id is None or self.denominator_series_id is None
        ):
            raise ValueError("numerator_series_id and denominator_series_id must be set together")

    @property
    def is_ratio(self) -> bool:
        return self.numerator_series_id is not None


@dataclass(frozen=True)
class FeatureConfig:
    version: str
    features: tuple[FeatureDefinition, ...]

    def to_dict(self) -> dict[str, object]:
        """A JSON-serializable representation — e.g. Phase 4's `config.json`."""
        return _as_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FeatureConfig":
        """The inverse of `to_dict()` — Phase 6's `prediction/model_loader.py` uses
        this to reconstruct the `FeatureConfig` a published model was trained with
        from its `config.json`.
        """
        raw_features = data["features"]
        assert isinstance(raw_features, list)
        features = tuple(_feature_definition_from_dict(f) for f in raw_features)
        version = data["version"]
        assert isinstance(version, str)
        return cls(version=version, features=features)

    def content_hash(self) -> str:
        """A stable hash of every field that defines this config's behavior — Phase
        4's dataset manifest embeds this (brief section 4.2's `feature_config_hash`)
        so a training run's inputs are reproducible.
        """
        payload = json.dumps(_as_jsonable(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_jsonable(config: FeatureConfig) -> dict[str, object]:
    return {
        "version": config.version,
        "features": [
            {
                "name": f.name,
                "axis": f.axis.value,
                "series_id": f.series_id,
                "source": f.source,
                "native_frequency": f.native_frequency.value,
                "vintage_aware": f.vintage_aware,
                "publication_lag_days": f.publication_lag_days,
                "transform": f.transform,
                "transform_params": f.transform_params,
                "sign": f.sign,
                "aggregation": f.aggregation,
                "numerator_series_id": f.numerator_series_id,
                "denominator_series_id": f.denominator_series_id,
            }
            for f in config.features
        ],
    }


def _feature_definition_from_dict(data: dict[str, object]) -> FeatureDefinition:
    name = data["name"]
    series_id = data["series_id"]
    source = data["source"]
    transform = data["transform"]
    aggregation = data["aggregation"]
    axis = data["axis"]
    native_frequency = data["native_frequency"]
    assert isinstance(name, str)
    assert isinstance(series_id, str)
    assert isinstance(source, str)
    assert isinstance(transform, str)
    assert isinstance(aggregation, str)
    assert isinstance(axis, str)
    assert isinstance(native_frequency, str)
    transform_params = data["transform_params"]
    assert isinstance(transform_params, dict)
    sign = data["sign"]
    assert isinstance(sign, int)
    publication_lag_days = data["publication_lag_days"]
    assert isinstance(publication_lag_days, int)

    return FeatureDefinition(
        name=name,
        axis=Axis(axis),
        series_id=series_id,
        source=source,
        native_frequency=Frequency(native_frequency),
        vintage_aware=bool(data["vintage_aware"]),
        publication_lag_days=publication_lag_days,
        transform=transform,
        transform_params=transform_params,
        sign=sign,
        aggregation=aggregation,
        numerator_series_id=data.get("numerator_series_id"),  # type: ignore[arg-type]
        denominator_series_id=data.get("denominator_series_id"),  # type: ignore[arg-type]
    )


def used_series_ids(feature_config: FeatureConfig) -> list[str]:
    """Every distinct series ID `feature_config` needs — a single-series feature
    contributes its own `series_id`; a ratio feature contributes both legs. Shared
    by `training/dataset.py` (builds the training matrix) and
    `prediction/predict.py` (builds one day's feature vector), since both need to
    know which lake/provider series to read before any feature can be computed.
    """
    series_ids: set[str] = set()
    for feature in feature_config.features:
        if feature.is_ratio:
            assert feature.numerator_series_id is not None
            assert feature.denominator_series_id is not None
            series_ids.add(feature.numerator_series_id)
            series_ids.add(feature.denominator_series_id)
        else:
            series_ids.add(feature.series_id)
    return sorted(series_ids)


# Brief section 2.4's first-pass feature set. Exact transform shapes (periods,
# pct-change vs. diff, annualization) and signs are my judgment calls where the
# brief names a series/feature but not every numeric parameter — see CLAUDE.md for
# the full table and reasoning, including the two deliberate exceptions: T10Y2Y is
# fed as a raw level (transform "identity") rather than a rate of change, and
# DFII10's sign is a debatable real-yield/inflation call.
FEATURE_CONFIG_V1 = FeatureConfig(
    version="v1",
    features=(
        FeatureDefinition(
            name="indpro_3m_change",
            axis=Axis.GROWTH,
            series_id="INDPRO",
            source="alfred",
            native_frequency=Frequency.MONTHLY,
            vintage_aware=True,
            publication_lag_days=45,
            transform="pct_change",
            transform_params={"n_periods": 3},
            sign=1,
        ),
        FeatureDefinition(
            name="payems_3m_change",
            axis=Axis.GROWTH,
            series_id="PAYEMS",
            source="alfred",
            native_frequency=Frequency.MONTHLY,
            vintage_aware=True,
            publication_lag_days=35,
            transform="pct_change",
            transform_params={"n_periods": 3},
            sign=1,
        ),
        FeatureDefinition(
            name="icsa_4wk_avg_change",
            axis=Axis.GROWTH,
            series_id="ICSA",
            source="alfred",
            native_frequency=Frequency.WEEKLY,
            vintage_aware=True,
            publication_lag_days=4,
            transform="rolling_mean_diff",
            transform_params={"window": 4, "diff_periods": 1},
            sign=-1,
        ),
        FeatureDefinition(
            name="t10y2y_level",
            axis=Axis.GROWTH,
            series_id="T10Y2Y",
            source="fred",
            native_frequency=Frequency.DAILY,
            vintage_aware=False,
            publication_lag_days=0,
            transform="identity",
            transform_params={},
            sign=1,
        ),
        FeatureDefinition(
            name="copper_gold_3m_momentum",
            axis=Axis.GROWTH,
            series_id="HG=F/GC=F",
            source="yfinance",
            native_frequency=Frequency.DAILY,
            vintage_aware=False,
            publication_lag_days=0,
            transform="pct_change",
            transform_params={"n_periods": 3},
            sign=1,
            numerator_series_id="HG=F",
            denominator_series_id="GC=F",
        ),
        FeatureDefinition(
            name="cpilfesl_3m_annualized_change",
            axis=Axis.INFLATION,
            series_id="CPILFESL",
            source="alfred",
            native_frequency=Frequency.MONTHLY,
            vintage_aware=True,
            publication_lag_days=15,
            transform="pct_change_annualized",
            transform_params={"n_periods": 3, "periods_per_year": 12},
            sign=1,
        ),
        FeatureDefinition(
            name="t10yie_1m_change",
            axis=Axis.INFLATION,
            series_id="T10YIE",
            source="fred",
            native_frequency=Frequency.DAILY,
            vintage_aware=False,
            publication_lag_days=0,
            transform="diff",
            transform_params={"n_periods": 1},
            sign=1,
        ),
        FeatureDefinition(
            name="t10yie_3m_change",
            axis=Axis.INFLATION,
            series_id="T10YIE",
            source="fred",
            native_frequency=Frequency.DAILY,
            vintage_aware=False,
            publication_lag_days=0,
            transform="diff",
            transform_params={"n_periods": 3},
            sign=1,
        ),
        FeatureDefinition(
            name="dfii10_1m_change",
            axis=Axis.INFLATION,
            series_id="DFII10",
            source="fred",
            native_frequency=Frequency.DAILY,
            vintage_aware=False,
            publication_lag_days=0,
            transform="diff",
            transform_params={"n_periods": 1},
            sign=-1,
        ),
        FeatureDefinition(
            name="dcoilwtico_3m_change",
            axis=Axis.INFLATION,
            series_id="DCOILWTICO",
            source="fred",
            native_frequency=Frequency.DAILY,
            vintage_aware=False,
            publication_lag_days=0,
            transform="pct_change",
            transform_params={"n_periods": 3},
            sign=1,
        ),
    ),
)
