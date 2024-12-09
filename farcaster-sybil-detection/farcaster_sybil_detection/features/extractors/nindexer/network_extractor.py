from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class EnhancedNetworkExtractor(FeatureExtractor):
    """Enhanced network analysis using nindexer data"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "network_age_hours",
            "total_follows",
            "follow_rate_per_hour",
            "avg_follow_latency_seconds",
            "latest_follower_count",
            "latest_following_count",
            "latest_follow_ratio",
            "follow_velocity",
            "network_growth_stability",
        ]

    def get_dependencies(self) -> List[str]:
        return ["fid"]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "follows": {
                "columns": [
                    "fid",
                    "target_fid",
                    "timestamp",
                    "created_at",
                    "deleted_at",
                ],
                "source": "nindexer",
            },
            "follow_counts": {
                "columns": ["fid", "follower_count", "following_count", "created_at"],
                "source": "nindexer",
            },
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting enhanced network features...")
            follows = loaded_datasets.get("follows")
            follow_counts = loaded_datasets.get("follow_counts")

            if follows is None or follow_counts is None:
                return self._get_default_features(df)

            # Process follows data
            follow_metrics = (
                follows.filter(pl.col("deleted_at").is_null())
                .with_columns(
                    [
                        pl.col("timestamp").cast(pl.Datetime),
                        pl.col("created_at").cast(pl.Datetime),
                    ]
                )
                .group_by("fid")
                .agg(
                    [
                        (pl.col("timestamp").max() - pl.col("timestamp").min())
                        .dt.total_hours()
                        .alias("network_age_hours"),
                        pl.len().alias("total_follows"),
                        (pl.col("created_at") - pl.col("timestamp"))
                        .dt.total_seconds()
                        .mean()
                        .alias("avg_follow_latency_seconds"),
                    ]
                )
                .with_columns(
                    [
                        (
                            pl.col("total_follows") / (pl.col("network_age_hours") + 1)
                        ).alias("follow_rate_per_hour")
                    ]
                )
            )

            # Process follow counts data
            count_metrics = (
                follow_counts.sort("created_at")
                .group_by("fid")
                .agg(
                    [
                        pl.col("follower_count").last().alias("latest_follower_count"),
                        pl.col("following_count")
                        .last()
                        .alias("latest_following_count"),
                        pl.col("follower_count")
                        .diff()
                        .std()
                        .alias("follower_growth_std"),
                        pl.col("following_count")
                        .diff()
                        .std()
                        .alias("following_growth_std"),
                    ]
                )
                .with_columns(
                    [
                        (
                            pl.col("latest_follower_count")
                            / (pl.col("latest_following_count") + 1)
                        ).alias("latest_follow_ratio"),
                        # Calculate velocity and stability
                        (
                            pl.col("follower_growth_std")
                            + pl.col("following_growth_std")
                        ).alias("network_growth_stability"),
                    ]
                )
            )

            # Combine metrics
            result = df.join(follow_metrics, on="fid", how="left")
            result = result.join(count_metrics, on="fid", how="left")

            # Calculate follow velocity based on time-weighted growth
            result = result.with_columns(
                [
                    (
                        pl.col("latest_follower_count")
                        / (pl.col("network_age_hours") + 1)
                    ).alias("follow_velocity")
                ]
            )

            return result.select(["fid"] + self.feature_names).fill_null(0)

        except Exception as e:
            self.logger.error(f"Error extracting enhanced network features: {e}")
            raise

    def _get_default_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Return DataFrame with default zero values for all features"""
        return df.with_columns(
            [
                pl.lit(0.0).alias("network_age_hours"),
                pl.lit(0).alias("total_follows"),
                pl.lit(0.0).alias("follow_rate_per_hour"),
                pl.lit(0.0).alias("avg_follow_latency_seconds"),
                pl.lit(0).alias("latest_follower_count"),
                pl.lit(0).alias("latest_following_count"),
                pl.lit(0.0).alias("latest_follow_ratio"),
                pl.lit(0.0).alias("follow_velocity"),
                pl.lit(0.0).alias("network_growth_stability"),
            ]
        )
