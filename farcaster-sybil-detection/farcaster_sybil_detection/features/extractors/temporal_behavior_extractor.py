from typing import Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


def validate_fid_schema(df: pl.LazyFrame, extractor_name: str):
    if "fid" not in df.columns:
        raise ValueError(f"'fid' column missing in DataFrame from {extractor_name}")
    if df.schema["fid"] != pl.Int64:
        raise TypeError(
            f"'fid' column in {extractor_name} must be of type Int64, got {df.schema['fid']}"
        )


class TemporalBehaviorExtractor(FeatureExtractor):
    """Temporal patterns and behavioral analysis"""

    def __init__(self, data_loader: DatasetLoader):
        super().__init__(data_loader)
        self.feature_names = self.get_feature_names()

    @classmethod
    def get_feature_names(cls):
        return [
            # Activity timing
            "hour_diversity",
            "weekday_diversity",
            # "peak_activity_hours",
            "inactive_periods",
            # "activity_regularity",
            "daily_active_hours",
            # Temporal patterns
            "posting_frequency",
            "response_latency",
            "interaction_timing",
            "engagement_windows",
            # "activity_cycles",
            # "seasonal_patterns",
            # Burst analysis
            "burst_frequency",
            "burst_intensity",
            "burst_duration",
            "inter_burst_interval",
            "burst_engagement_ratio",
            "burst_impact",
            # Consistency metrics
            "temporal_consistency",
            # "engagement_stability",  # Removed due to missing action_type
            # "pattern_predictability",
            "rhythm_score",
            # "routine_strength",
            "variability_index",
            # Activity distribution
            "prime_time_ratio",
            "off_hours_activity",
            "weekend_activity_ratio",
            # "timezone_alignment",
            # "local_time_preference",
            "global_reach",
            # Advanced temporal
            "activity_entropy",
            "temporal_clustering",
            # "periodicity_strength",
            "trend_stability",
            # "temporal_novelty",
            "adaptation_rate",
            # missing
            "avg_follow_latency_seconds",
            "cross_channel_activity",
            "multi_channel_ratio",
        ]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "casts": {
                "columns": [
                    "fid",
                    "timestamp",
                    "deleted_at",
                    "text",
                    "parent_url",
                    "parent_hash",
                    "parent_fid",
                    "embeds",
                    "mentions",
                    "root_parent_hash",
                ],
                "source": "farcaster",
            },
            "reactions": {
                "columns": ["fid", "timestamp", "deleted_at"],
                "source": "farcaster",
            },
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.debug("Extracting temporal behavior features...")

            # Extract each feature category
            timing_metrics = self._extract_timing_metrics(loaded_datasets)
            pattern_metrics = self._extract_pattern_metrics(loaded_datasets)
            burst_metrics = self._extract_burst_metrics(loaded_datasets)
            consistency_metrics = self._extract_consistency_metrics(loaded_datasets)
            distribution_metrics = self._extract_distribution_metrics(loaded_datasets)
            advanced_metrics = self._extract_advanced_temporal_metrics(loaded_datasets)
            channel_metrics = self._extract_channel_metrics(loaded_datasets)

            # Debugging: Check schemas and row counts
            self.logger.debug(f"Schema of timing_metrics: {timing_metrics.schema}")

            self.logger.debug(f"Schema of pattern_metrics: {pattern_metrics.schema}")

            self.logger.debug(
                f"Schema of consistency_metrics: {consistency_metrics.schema}"
            )

            self.logger.debug(
                f"Schema of distribution_metrics: {distribution_metrics.schema}"
            )

            self.logger.debug(f"Schema of advanced_metrics: {advanced_metrics.schema}")

            # Combine all features
            result = df.clone()
            for metrics, name in zip(
                [
                    timing_metrics,
                    pattern_metrics,
                    burst_metrics,
                    consistency_metrics,
                    distribution_metrics,
                    advanced_metrics,
                    channel_metrics,
                ],
                [
                    "timing_metrics",
                    "pattern_metrics",
                    "burst_metrics",
                    "consistency_metrics",
                    "distribution_metrics",
                    "advanced_metrics",
                    "channel_metrics",
                ],
            ):
                if metrics is not None:
                    # Validate schema
                    validate_fid_schema(metrics, name)
                    # Proceed with join
                    result = result.join(metrics, on="fid", how="left")
                    self.logger.debug(
                        f"Joined {name} with result. Current schema: {result.schema}"
                    )

            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting temporal behavior features: {e}")
            raise

    def _extract_channel_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")

        if casts is None:
            return None

        return (
            casts.filter(
                pl.col("deleted_at").is_null(), pl.col("parent_url").is_not_null()
            )
            .group_by("fid")
            .agg(
                [
                    pl.n_unique("parent_url").alias("unique_channels"),
                    (pl.n_unique("parent_url") / pl.count()).alias(
                        "cross_channel_activity"
                    ),
                    (pl.col("parent_url").is_not_null().sum() / pl.count()).alias(
                        "multi_channel_ratio"
                    ),
                ]
            )
        )

    def _extract_timing_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            self.logger.warning("Casts dataset not found for timing metrics.")
            return pl.DataFrame({"fid": []}).lazy()

        return (
            casts.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("timestamp").dt.hour().alias("hour"),
                    pl.col("timestamp").dt.weekday().alias("weekday"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    pl.col("hour").n_unique().alias("hour_diversity"),
                    pl.col("weekday").n_unique().alias("weekday_diversity"),
                    # pl.col("hour").mode().alias("peak_activity_hours"),
                    (pl.col("timestamp").diff().dt.total_hours() > 24)
                    .sum()
                    .alias("inactive_periods"),
                    # pl.col("hour").value_counts().std().alias("activity_regularity"),
                    pl.col("hour").n_unique().alias("daily_active_hours"),
                ]
            )
            .with_columns([pl.col("fid").cast(pl.Int64)])
        )

    def _extract_pattern_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            self.logger.warning("Casts dataset not found for pattern metrics.")
            return pl.DataFrame({"fid": []}).lazy()

        return (
            casts.filter(pl.col("deleted_at").is_null())
            .with_columns(
                [
                    pl.col("timestamp")
                    .diff()
                    .dt.total_seconds()
                    .mean()
                    .alias("avg_follow_latency_seconds"),
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("timestamp").dt.hour().alias("hour"),
                    pl.col("timestamp").dt.weekday().alias("weekday"),
                    pl.col("timestamp").diff().dt.total_hours().alias("hours_between"),
                    pl.col("timestamp")
                    .diff()
                    .dt.total_hours()
                    .alias("time_diff"),  # Added
                ]
            )
            .group_by("fid")
            .agg(
                [
                    pl.col("avg_follow_latency_seconds")
                    .mean()
                    .alias("avg_follow_latency_seconds"),
                    pl.count().alias("posting_frequency"),
                    pl.col("time_diff").mean().alias("response_latency"),
                    pl.col("time_diff").std().alias("interaction_timing"),
                    pl.col("hours_between")
                    .filter(pl.col("hours_between") < 1)
                    .count()
                    .alias("engagement_windows"),
                    # pl.col("hour")
                    # .value_counts()
                    # .filter(pl.col("count") > pl.col("count").mean())
                    # .count()
                    # .alias("activity_cycles"),
                    # pl.col("timestamp")
                    # .dt.weekday()
                    # .value_counts()
                    # .std()
                    # .alias("seasonal_patterns"),
                ]
            )
            .with_columns([pl.col("fid").cast(pl.Int64)])
        )

    def _extract_burst_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            self.logger.warning("Casts dataset not found for burst metrics.")
            return pl.DataFrame({"fid": []}).lazy()

        return (
            casts.filter(pl.col("deleted_at").is_null())
            .with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    # Define bursts as activities within 5 minutes
                    pl.col("timestamp")
                    .diff()
                    .dt.total_minutes()
                    .lt(5)
                    .alias("is_burst"),
                    pl.col("timestamp")
                    .diff()
                    .dt.total_minutes()
                    .alias("minutes_between"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Burst frequency
                    pl.col("is_burst").sum().alias("burst_frequency"),
                    # Burst intensity (actions per burst)
                    (
                        pl.col("is_burst").sum()
                        / (pl.col("is_burst").sum() + 1)  # Prevent division by zero
                    ).alias("burst_intensity"),
                    # Burst duration (mean minutes between actions in bursts)
                    pl.col("minutes_between")
                    .filter(pl.col("is_burst"))
                    .mean()
                    .alias("burst_duration"),
                    # Inter-burst interval (mean minutes between bursts)
                    pl.col("minutes_between")
                    .filter(~pl.col("is_burst"))
                    .mean()
                    .alias("inter_burst_interval"),
                    # Burst engagement ratio (proportion of actions in bursts)
                    (pl.col("is_burst").sum() / pl.count()).alias(
                        "burst_engagement_ratio"
                    ),
                    # Burst impact (total bursts)
                    pl.col("is_burst").sum().alias("burst_impact"),
                ]
            )
            .with_columns([pl.col("fid").cast(pl.Int64)])  # Added for consistency
        )

    def _extract_consistency_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            self.logger.warning("Casts dataset not found for consistency metrics.")
            return pl.DataFrame({"fid": []}).lazy()

        return (
            casts.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("timestamp").dt.hour().alias("hour"),
                    pl.col("timestamp").dt.weekday().alias("weekday"),
                    pl.col("timestamp").diff().dt.total_hours().alias("hours_between"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Temporal consistency (std of hours_between)
                    pl.col("hours_between").std().alias("temporal_consistency"),
                    # Engagement stability (removed due to missing action_type)
                    # pl.col("action_type")
                    #     .value_counts()
                    #     .std()
                    #     .alias("engagement_stability"),
                    # Pattern predictability (std of hour counts)
                    # pl.col("hour").value_counts().std().alias("pattern_predictability"),
                    # Rhythm score (number of unique hours)
                    pl.col("hour").n_unique().cast(pl.Float64).alias("rhythm_score"),
                    # Routine strength (count of weekdays with above-average activity)
                    # pl.col("weekday")
                    # .value_counts()
                    # .filter(pl.col("count") > pl.col("count").mean())
                    # .count()
                    # .alias("routine_strength"),
                    # Variability index (difference between 90th and 10th percentiles)
                    (
                        pl.col("hours_between").quantile(0.9)
                        - pl.col("hours_between").quantile(0.1)
                    ).alias("variability_index"),
                ]
            )
            .with_columns([pl.col("fid").cast(pl.Int64)])
        )

    def _extract_distribution_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            self.logger.warning("Casts dataset not found for distribution metrics.")
            return pl.DataFrame({"fid": []}).lazy()

        return (
            casts.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("timestamp").dt.hour().alias("hour"),
                    pl.col("timestamp").dt.weekday().alias("weekday"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Prime time ratio (9 AM to 10 PM)
                    (
                        pl.col("hour").is_between(9, 22).cast(pl.Float64).sum()
                        / pl.count()
                    ).alias("prime_time_ratio"),
                    # Off-hours activity (<9 AM or >10 PM)
                    (
                        (pl.col("hour").lt(9) | pl.col("hour").gt(22))
                        .cast(pl.Float64)
                        .sum()
                        / pl.count()
                    ).alias("off_hours_activity"),
                    # Weekend activity ratio (Saturday and Sunday)
                    (
                        pl.col("weekday").is_in([5, 6]).cast(pl.Float64).sum()
                        / pl.count()
                    ).alias("weekend_activity_ratio"),
                    # Timezone alignment (number of peak activity hours)
                    # pl.col("hour")
                    # .value_counts()
                    # .filter(pl.col("count") > pl.col("count").mean())
                    # .count()
                    # .alias("timezone_alignment"),
                    # Local time preference (mode hour)
                    # pl.col("hour").mode().alias("local_time_preference"),
                    # Global reach (unique hours / 24)
                    (pl.col("hour").n_unique() / 24.0).alias("global_reach"),
                ]
            )
            .with_columns([pl.col("fid").cast(pl.Int64)])
        )

    def _extract_advanced_temporal_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            self.logger.warning(
                "Casts dataset not found for advanced temporal metrics."
            )
            return pl.DataFrame({"fid": []}).lazy()

        # Calculate hour distributions and entropy
        hour_metrics = (
            casts.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("timestamp").dt.hour().alias("hour"),
                ]
            )
            .group_by(["fid", "hour"])
            .agg([pl.count().alias("hour_count")])
            .group_by("fid")
            .agg(
                [
                    (
                        -(
                            (pl.col("hour_count") / pl.col("hour_count").sum())
                            * (pl.col("hour_count") / pl.col("hour_count").sum()).log(2)
                        ).sum()
                    ).alias("activity_entropy")
                ]
            )
        )

        # Calculate time-based metrics
        time_metrics = (
            casts.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("timestamp").diff().dt.total_hours().alias("hours_between"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Temporal clustering
                    (
                        pl.col("hours_between")
                        .filter(pl.col("hours_between") < 1)
                        .count()
                        / pl.count()
                    ).alias("temporal_clustering"),
                    # Trend stability
                    pl.col("hours_between")
                    .rolling_std(window_size=24, min_periods=1)
                    .mean()
                    .alias("trend_stability"),
                    # Adaptation rate
                    pl.col("hours_between")
                    .rolling_mean(window_size=24, min_periods=1)
                    .diff()
                    .std()
                    .alias("adaptation_rate"),
                ]
            )
        )

        # Combine all metrics
        return (
            hour_metrics.join(time_metrics, on="fid", how="left")
            .with_columns([pl.col("fid").cast(pl.Int64)])
            .select(
                [
                    "fid",
                    "activity_entropy",
                    "temporal_clustering",
                    "trend_stability",
                    "adaptation_rate",
                ]
            )
        )
