from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class DerivedFeatureExtractor(FeatureExtractor):
    """Calculate derived and composite features"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "follower_ratio_log",
            "unique_follower_ratio_log",
            "follow_velocity_log",
            "has_more_followers",
            "follow_balance_ratio",
            "follower_ratio_capped",
            "unique_follower_ratio_capped",
            "follow_velocity_capped",
        ]

    def get_dependencies(self) -> List[str]:
        return [
            "following_count",
            "follower_count",
            # 'follower_ratio',
            # 'unique_follower_ratio',
            "follow_velocity",
        ]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {}  # Uses only existing features

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting derived features...")

            derived_features = df.with_columns(
                [
                    # Log transformations
                    pl.col("follower_ratio")
                    .fill_null(0.0)
                    .log1p()
                    .alias("follower_ratio_log"),
                    pl.col("unique_follower_ratio")
                    .fill_null(0.0)
                    .log1p()
                    .alias("unique_follower_ratio_log"),
                    pl.col("follow_velocity")
                    .fill_null(0.0)
                    .log1p()
                    .alias("follow_velocity_log"),
                    # Binary flags
                    (
                        pl.when(
                            pl.col("follower_count").fill_null(0)
                            > pl.col("following_count").fill_null(0)
                        )
                        .then(1)
                        .otherwise(0)
                    ).alias("has_more_followers"),
                    # Balance ratios
                    (
                        (
                            pl.col("following_count").fill_null(0)
                            - pl.col("follower_count").fill_null(0)
                        ).abs()
                        / (
                            pl.col("following_count").fill_null(0)
                            + pl.col("follower_count").fill_null(0)
                            + 1
                        )
                    ).alias("follow_balance_ratio"),
                ]
            )

            # Cap extreme values
            for col in ["follower_ratio", "unique_follower_ratio", "follow_velocity"]:
                if col in derived_features.columns:
                    p99 = (
                        derived_features.collect()
                        .select(pl.col(col).fill_null(0.0).quantile(0.99))
                        .item()
                    )
                    derived_features = derived_features.with_columns(
                        [
                            pl.col(col)
                            .fill_null(0.0)
                            .clip(0.0, p99)
                            .alias(f"{col}_capped")
                        ]
                    )

            return derived_features.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting derived features: {e}")
            raise
