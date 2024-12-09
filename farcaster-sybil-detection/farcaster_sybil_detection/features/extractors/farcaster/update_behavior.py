from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class UpdateBehaviorExtractor(FeatureExtractor):
    """Extract update behavior patterns"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "profile_update_consistency",
            "total_updates",
            "avg_update_interval",
            "update_time_std",
            "rapid_updates",
            "update_regularity",
            "content_update_frequency",
        ]

    def get_dependencies(self) -> List[str]:
        return ["fid"]

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
            self.logger.info("Extracting update behavior features...")
            user_data = loaded_datasets.get("user_data")

            if user_data is None:
                return self._get_default_features(df)

            update_features = (
                user_data.filter(pl.col("deleted_at").is_null())
                .with_columns([pl.col("timestamp").cast(pl.Datetime)])
                .sort(["fid", "timestamp"])
                .group_by("fid")
                .agg(
                    [
                        pl.len().alias("total_updates"),
                        pl.col("timestamp")
                        .diff()
                        .dt.total_hours()
                        .mean()
                        .alias("avg_update_interval"),
                        pl.col("timestamp")
                        .diff()
                        .dt.total_hours()
                        .std()
                        .alias("update_time_std"),
                        (pl.col("timestamp").diff().dt.total_hours() < 1)
                        .sum()
                        .alias("rapid_updates"),
                        pl.col("timestamp")
                        .diff()
                        .dt.total_hours()
                        .std()
                        .alias("update_regularity"),
                        (pl.col("type") == "content")
                        .sum()
                        .alias("content_update_frequency"),
                    ]
                )
            )

            update_features = update_features.with_columns(
                [
                    (
                        pl.col("update_time_std") / (pl.col("avg_update_interval") + 1)
                    ).alias("profile_update_consistency")
                ]
            )

            result = df.join(update_features, on="fid", how="left")
            return result.select(["fid"] + self.feature_names).fill_null(0)

        except Exception as e:
            self.logger.error(f"Error extracting update behavior features: {e}")
            raise

    def _get_default_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Return DataFrame with default zero values for all features"""
        return df.with_columns(
            [
                pl.lit(0.0).alias("profile_update_consistency"),
                pl.lit(0).alias("total_updates"),
                pl.lit(0.0).alias("avg_update_interval"),
                pl.lit(0.0).alias("update_time_std"),
                pl.lit(0).alias("rapid_updates"),
                pl.lit(0.0).alias("update_regularity"),
                pl.lit(0).alias("content_update_frequency"),
            ]
        )
