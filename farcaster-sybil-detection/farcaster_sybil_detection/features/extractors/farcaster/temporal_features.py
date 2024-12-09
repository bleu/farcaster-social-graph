from typing import Dict, List, Optional

import polars as pl

from farcaster_sybil_detection.data.dataset_loader import DatasetLoader
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor

from ..base import FeatureConfig, FeatureExtractor


class TemporalFeatureExtractor(FeatureExtractor):
    """Extract temporal-based features with burst detection"""

    def get_dependencies(self) -> List[str]:
        """List of required input columns"""
        return ["fid", "timestamp", "deleted_at"]

    def get_required_datasets(self) -> Dict[str, Dict]:
        """Declare the datasets required for this feature extractor."""
        return {
            "links": {
                "columns": ["fid", "timestamp", "deleted_at"],
                "source": "farcaster",
            }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """Extract temporal features from LazyFrame"""
        try:
            self.logger.info("Extracting temporal features...")
            links = loaded_datasets.get("links")

            if links is None:
                self.logger.warning("No links data available for extraction.")
                return df

            # Convert timestamp to datetime if not already
            links = links.with_columns(
                [pl.col("timestamp").cast(pl.Datetime).alias("timestamp")]
            )

            temporal_features = (
                links.group_by("fid")
                .agg(
                    [
                        pl.col("timestamp").count().alias("total_activity"),
                        pl.col("timestamp")
                        .diff()
                        .dt.total_hours()
                        .mean()
                        .alias("avg_hours_between_actions"),
                        pl.col("timestamp")
                        .diff()
                        .dt.total_hours()
                        .std()
                        .alias("std_hours_between_actions"),
                        pl.col("timestamp")
                        .dt.weekday()
                        .std()
                        .alias("weekday_variance"),
                        (pl.col("timestamp").diff().dt.total_hours() < 1)
                        .sum()
                        .alias("rapid_actions"),
                        (pl.col("timestamp").diff().dt.total_hours() > 24)
                        .sum()
                        .alias("long_gaps"),
                        pl.col("timestamp")
                        .diff()
                        .dt.total_hours()
                        .quantile(0.9)
                        .alias("p90_time_between_actions"),
                        pl.col("timestamp")
                        .diff()
                        .dt.total_hours()
                        .quantile(0.1)
                        .alias("p10_time_between_actions"),
                        (pl.col("timestamp").diff().dt.total_hours() < 1)
                        .sum()
                        .alias("actions_in_bursts"),
                        (pl.col("timestamp").max() - pl.col("timestamp").min())
                        .dt.total_hours()
                        .alias("time_span"),
                    ]
                )
                .with_columns(
                    [
                        (
                            pl.col("actions_in_bursts") / (pl.col("total_activity") + 1)
                        ).alias("burst_activity_ratio"),
                        (
                            pl.col("time_span")
                            / (
                                (pl.col("total_activity") + 1)
                                * pl.col("avg_hours_between_actions")
                            )
                        ).alias("activity_spread"),
                        (
                            pl.col("std_hours_between_actions")
                            / (pl.col("avg_hours_between_actions") + 1)
                        ).alias("temporal_irregularity"),
                        (pl.col("total_activity") / (pl.col("time_span") + 1)).alias(
                            "follow_velocity"
                        ),
                    ]
                )
            )

            extracted_features = temporal_features.select(
                [
                    "fid",
                    "total_activity",
                    "avg_hours_between_actions",
                    "std_hours_between_actions",
                    "weekday_variance",
                    "rapid_actions",
                    "long_gaps",
                    "p90_time_between_actions",
                    "p10_time_between_actions",
                    "actions_in_bursts",
                    "burst_activity_ratio",
                    "activity_spread",
                    "temporal_irregularity",
                    "follow_velocity",
                ]
            )

            self.logger.info("Temporal features extracted successfully.")
            return extracted_features

        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {e}")
            raise
