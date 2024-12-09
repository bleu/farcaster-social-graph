from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class AuthenticityExtractor(FeatureExtractor):
    """Extract comprehensive authenticity indicators"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "authenticity_score",
            "profile_completeness",
            "network_balance",
            "update_naturalness",
            "behavior_consistency",
            "temporal_authenticity",
        ]

    def get_dependencies(self) -> List[str]:
        return [
            "has_bio",
            "has_avatar",
            "verification_count",
            "has_ens",
            "following_count",
            "follower_count",
            "total_updates",
            "avg_update_interval",
            "profile_update_consistency",
        ]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "user_data": {
                "columns": ["fid", "timestamp", "type", "deleted_at"],
                "source": "farcaster",
            }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting authenticity features...")
            user_data = loaded_datasets.get("user_data")

            # Start with basic authenticity metrics
            result = df.with_columns(
                [
                    # Profile completeness (0-1)
                    (
                        (
                            pl.col("has_bio").fill_null(0)
                            + pl.col("has_avatar").fill_null(0)
                            + pl.col("has_ens").fill_null(0)
                            + (pl.col("verification_count").fill_null(0) > 0).cast(
                                pl.Int64
                            )
                        )
                        / 4.0
                    ).alias("profile_completeness"),
                    # Network balance (0-1)
                    (
                        pl.when(
                            pl.col("following_count").fill_null(0)
                            + pl.col("follower_count").fill_null(0)
                            > 0
                        )
                        .then(
                            1.0
                            - (
                                pl.col("following_count").fill_null(0)
                                - pl.col("follower_count").fill_null(0)
                            ).abs()
                            / (
                                pl.col("following_count").fill_null(0)
                                + pl.col("follower_count").fill_null(0)
                            )
                        )
                        .otherwise(0.0)
                    ).alias("network_balance"),
                ]
            )

            # Add update naturalness if user_data available
            if user_data is not None:
                update_patterns = (
                    user_data.filter(pl.col("deleted_at").is_null())
                    .with_columns([pl.col("timestamp").cast(pl.Datetime)])
                    .group_by("fid")
                    .agg(
                        [
                            pl.col("timestamp")
                            .diff()
                            .dt.total_hours()
                            .std()
                            .alias("update_time_std"),
                            pl.col("timestamp")
                            .diff()
                            .dt.total_hours()
                            .mean()
                            .alias("avg_update_interval"),
                        ]
                    )
                    .with_columns(
                        [
                            (
                                pl.col("update_time_std")
                                / pl.col("avg_update_interval")
                            ).alias("update_consistency")
                        ]
                    )
                )

                result = result.join(update_patterns, on="fid", how="left")
                result = result.with_columns(
                    [
                        (1.0 - pl.col("update_consistency").clip(0.0, 1.0)).alias(
                            "update_naturalness"
                        )
                    ]
                )
            else:
                result = result.with_columns([pl.lit(0.0).alias("update_naturalness")])

            # Calculate behavior consistency
            result = result.with_columns(
                [
                    (
                        pl.when(pl.col("total_updates") > 0)
                        .then(1.0 - pl.col("profile_update_consistency").clip(0.0, 1.0))
                        .otherwise(0.0)
                    ).alias("behavior_consistency")
                ]
            )

            # Calculate temporal authenticity
            result = result.with_columns(
                [
                    (
                        pl.col("avg_update_interval").fill_null(0.0)
                        / (pl.col("total_updates").fill_null(0.0) + 1.0)
                    ).alias("temporal_authenticity")
                ]
            )

            # Calculate final authenticity score
            result = result.with_columns(
                [
                    (
                        pl.col("profile_completeness").fill_null(0.0) * 0.3
                        + pl.col("network_balance").fill_null(0.0) * 0.2
                        + pl.col("update_naturalness").fill_null(0.0) * 0.2
                        + pl.col("behavior_consistency").fill_null(0.0) * 0.2
                        + pl.col("temporal_authenticity").fill_null(0.0) * 0.1
                    ).alias("authenticity_score")
                ]
            )

            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting authenticity features: {e}")
            raise

    def _get_default_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Return DataFrame with default zero values for all features"""
        return df.with_columns(
            [
                pl.lit(0.0).alias("authenticity_score"),
                pl.lit(0.0).alias("profile_completeness"),
                pl.lit(0.0).alias("network_balance"),
                pl.lit(0.0).alias("update_naturalness"),
                pl.lit(0.0).alias("behavior_consistency"),
                pl.lit(0.0).alias("temporal_authenticity"),
            ]
        )
