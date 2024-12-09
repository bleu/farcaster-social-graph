from typing import List, Dict
import polars as pl
import re
import numpy as np
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class TemporalBehaviorExtractor(FeatureExtractor):
    """Temporal patterns and behavioral analysis"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            # Activity timing
            "hour_diversity",
            "weekday_diversity",
            "peak_activity_hours",
            "inactive_periods",
            "activity_regularity",
            "daily_active_hours",
            # Temporal patterns
            "posting_frequency",
            "response_latency",
            "interaction_timing",
            "engagement_windows",
            "activity_cycles",
            "seasonal_patterns",
            # Burst analysis
            "burst_frequency",
            "burst_intensity",
            "burst_duration",
            "inter_burst_interval",
            "burst_engagement_ratio",
            "burst_impact",
            # Consistency metrics
            "temporal_consistency",
            "engagement_stability",
            "pattern_predictability",
            "rhythm_score",
            "routine_strength",
            "variability_index",
            # Activity distribution
            "prime_time_ratio",
            "off_hours_activity",
            "weekend_activity_ratio",
            "timezone_alignment",
            "local_time_preference",
            "global_reach",
            # Advanced temporal
            "activity_entropy",
            "temporal_clustering",
            "periodicity_strength",
            "trend_stability",
            "temporal_novelty",
            "adaptation_rate",
        ]

    def get_dependencies(self) -> List[str]:
        return []

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            # "activity_logs": {
            #     "columns": ["fid", "timestamp", "action_type"],
            #     "source": "farcaster",
            # },
            "casts": {
                "columns": ["fid", "timestamp", "deleted_at"],
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
            self.logger.info("Extracting temporal behavior features...")

            # Extract each feature category
            timing_metrics = self._extract_timing_metrics(loaded_datasets)
            pattern_metrics = self._extract_pattern_metrics(loaded_datasets)
            burst_metrics = self._extract_burst_metrics(loaded_datasets)
            consistency_metrics = self._extract_consistency_metrics(loaded_datasets)
            distribution_metrics = self._extract_distribution_metrics(loaded_datasets)
            advanced_metrics = self._extract_advanced_temporal_metrics(loaded_datasets)

            # Combine all features
            result = df.clone()
            for metrics in [
                timing_metrics,
                pattern_metrics,
                burst_metrics,
                consistency_metrics,
                distribution_metrics,
                advanced_metrics,
            ]:
                if metrics is not None:
                    self.logger.info(
                        f"Joining features: {metrics.columns} on {result.columns}"
                    )

                    result = result.join(metrics, on="fid", how="left")

            import ipdb

            ipdb.set_trace()
            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting temporal behavior features: {e}")
            raise

    def _extract_timing_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        activity_logs = loaded_datasets.get("activity_logs")
        if activity_logs is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            activity_logs.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("timestamp").dt.hour().alias("hour"),
                    pl.col("timestamp").dt.weekday().alias("weekday"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Hour diversity
                    pl.col("hour").n_unique().alias("hour_diversity"),
                    pl.col("weekday").n_unique().alias("weekday_diversity"),
                    # Active periods
                    pl.col("hour").mode().alias("peak_activity_hours"),
                    (pl.col("timestamp").diff().dt.total_hours() > 24)
                    .sum()
                    .alias("inactive_periods"),
                    # Activity regularity
                    pl.col("hour").value_counts().std().alias("activity_regularity"),
                    pl.col("hour").n_unique().alias("daily_active_hours"),
                ]
            )
        )

    def _extract_pattern_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        activity_logs = loaded_datasets.get("activity_logs")
        if activity_logs is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            activity_logs.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    # Calculate time differences between consecutive actions
                    pl.col("timestamp").diff().dt.total_seconds().alias("time_diff"),
                    # Hour of day for frequency analysis
                    pl.col("timestamp").dt.hour().alias("hour"),
                    # Calculate rolling windows
                    pl.col("timestamp").diff().dt.total_hours().alias("hours_between"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Posting frequency (posts per hour)
                    (
                        pl.count()
                        / (
                            pl.col("timestamp").max().dt.unix_timestamp()
                            - pl.col("timestamp").min().dt.unix_timestamp()
                        )
                        * 3600
                    ).alias("posting_frequency"),
                    # Response latency
                    pl.col("time_diff").mean().alias("response_latency"),
                    # Interaction timing variance
                    pl.col("time_diff").std().alias("interaction_timing"),
                    # Activity windows
                    pl.col("hours_between")
                    .filter(pl.col("hours_between") < 1)
                    .count()
                    .alias("engagement_windows"),
                    # Activity cycle detection
                    pl.col("hour")
                    .value_counts()
                    .filter(pl.col("count") > pl.col("count").mean())
                    .count()
                    .alias("activity_cycles"),
                    # Weekly patterns
                    pl.col("timestamp")
                    .dt.weekday()
                    .value_counts()
                    .std()
                    .alias("seasonal_patterns"),
                ]
            )
        )

    def _extract_burst_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        activity_logs = loaded_datasets.get("activity_logs")
        if activity_logs is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            activity_logs.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    # Define bursts as activities within 5 minutes
                    pl.col("timestamp").diff().dt.minutes().lt(5).alias("is_burst"),
                    pl.col("timestamp").diff().dt.minutes().alias("minutes_between"),
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
                        / pl.col("is_burst").sum().over(["fid", "is_burst"])
                    ).alias("burst_intensity"),
                    # Burst duration
                    pl.when(pl.col("is_burst"))
                    .then(pl.col("minutes_between"))
                    .alias("burst_duration")
                    .mean(),
                    # Time between bursts
                    pl.when(~pl.col("is_burst"))
                    .then(pl.col("minutes_between"))
                    .alias("inter_burst_interval")
                    .mean(),
                    # Engagement during bursts
                    (pl.col("is_burst").sum() / pl.count()).alias(
                        "burst_engagement_ratio"
                    ),
                    # Impact of burst activities
                    pl.col("is_burst")
                    .sum()
                    .over(["fid", "action_type"])
                    .mean()
                    .alias("burst_impact"),
                ]
            )
        )

    def _extract_consistency_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        activity_logs = loaded_datasets.get("activity_logs")
        if activity_logs is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            activity_logs.with_columns(
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
                    # Temporal consistency
                    pl.col("hours_between").std().alias("temporal_consistency"),
                    # Engagement stability
                    pl.col("action_type")
                    .value_counts()
                    .std()
                    .alias("engagement_stability"),
                    # Pattern predictability
                    pl.col("hour").value_counts().std().alias("pattern_predictability"),
                    # Daily rhythm score
                    pl.col("hour").n_unique().cast(pl.Float64).alias("rhythm_score"),
                    # Routine strength
                    pl.col("weekday")
                    .value_counts()
                    .filter(pl.col("count") > pl.col("count").mean())
                    .count()
                    .alias("routine_strength"),
                    # Variability index
                    pl.col("hours_between")
                    .quantile(0.9)
                    .sub(pl.col("hours_between").quantile(0.1))
                    .alias("variability_index"),
                ]
            )
        )

    def _extract_distribution_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        activity_logs = loaded_datasets.get("activity_logs")
        if activity_logs is None:
            return pl.DataFrame({"fid": []}).lazy()

        def is_prime_time(hour: int) -> bool:
            return 9 <= hour <= 22

        return (
            activity_logs.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("timestamp").dt.hour().alias("hour"),
                    pl.col("timestamp").dt.weekday().alias("weekday"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Prime time ratio
                    (pl.col("hour").apply(is_prime_time).sum() / pl.count()).alias(
                        "prime_time_ratio"
                    ),
                    # Off-hours activity
                    (
                        pl.col("hour").lt(9).or_(pl.col("hour").gt(22)).sum()
                        / pl.count()
                    ).alias("off_hours_activity"),
                    # Weekend activity
                    (pl.col("weekday").ge(5).sum() / pl.count()).alias(
                        "weekend_activity_ratio"
                    ),
                    # Timezone alignment
                    pl.col("hour")
                    .value_counts()
                    .filter(pl.col("count") > pl.col("count").mean())
                    .count()
                    .alias("timezone_alignment"),
                    # Local time preference
                    pl.col("hour").mode().alias("local_time_preference"),
                    # Global reach (activity spread across hours)
                    (pl.col("hour").n_unique() / 24.0).alias("global_reach"),
                ]
            )
        )

    def _extract_advanced_temporal_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        activity_logs = loaded_datasets.get("activity_logs")
        if activity_logs is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            activity_logs.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("timestamp").diff().dt.total_hours().alias("hours_between"),
                    pl.col("timestamp").dt.hour().alias("hour"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Activity entropy
                    -(
                        (pl.col("hour").value_counts() / pl.col("hour").count())
                        * (pl.col("hour").value_counts() / pl.col("hour").count()).log()
                    )
                    .sum()
                    .alias("activity_entropy"),
                    # Temporal clustering
                    pl.col("hours_between")
                    .filter(pl.col("hours_between") < 1)
                    .count()
                    .truediv(pl.count())
                    .alias("temporal_clustering"),
                    # Periodicity strength
                    pl.col("hour")
                    .value_counts()
                    .std()
                    .truediv(pl.col("hour").value_counts().mean())
                    .alias("periodicity_strength"),
                    # Trend stability
                    pl.col("hours_between")
                    .rolling_std(24)
                    .mean()
                    .alias("trend_stability"),
                    # Temporal novelty
                    pl.col("hour")
                    .n_unique()
                    .truediv(pl.col("hour").count())
                    .alias("temporal_novelty"),
                    # Adaptation rate
                    pl.col("hours_between")
                    .rolling_mean(24)
                    .diff()
                    .std()
                    .alias("adaptation_rate"),
                ]
            )
        )
