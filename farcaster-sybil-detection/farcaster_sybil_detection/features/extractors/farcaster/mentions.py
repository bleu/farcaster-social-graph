import re
from typing import Dict, List

import polars as pl

from farcaster_sybil_detection.data.dataset_loader import DatasetLoader
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor

from ..base import FeatureConfig, FeatureExtractor


class MentionsExtractor(FeatureExtractor):
    """Extract user mention patterns"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "casts_with_mentions",
            "total_mentions",
            "avg_mentions_per_cast",
            "mention_frequency",
            "mention_ratio",
        ]

    def get_dependencies(self) -> List[str]:
        return ["fid", "cast_count"]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "casts": {
                "columns": ["fid", "mentions", "deleted_at"],
                "source": "farcaster",
            }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting mention features...")
            casts = loaded_datasets.get("casts")

            if casts is None:
                self.logger.warning("No casts data available")
                return df

            mention_features = (
                casts.filter(pl.col("deleted_at").is_null())
                .with_columns(
                    [
                        pl.when(
                            pl.col("mentions").is_not_null()
                            & (pl.col("mentions") != "")
                            & (pl.col("mentions") != "[]")
                        )
                        .then(pl.col("mentions").str.json_decode().list.len())
                        .otherwise(0)
                        .alias("mention_count"),
                        (
                            pl.col("mentions").is_not_null()
                            & (pl.col("mentions") != "")
                            & (pl.col("mentions") != "[]")
                        )
                        .cast(pl.Int64)
                        .alias("has_mentions"),
                    ]
                )
                .group_by("fid")
                .agg(
                    [
                        pl.col("has_mentions").sum().alias("casts_with_mentions"),
                        pl.col("mention_count").sum().alias("total_mentions"),
                        pl.col("mention_count").mean().alias("avg_mentions_per_cast"),
                    ]
                )
            )

            # Join and add derived metrics
            result = df.join(mention_features, on="fid", how="left")
            result = result.with_columns(
                [
                    (pl.col("casts_with_mentions") / pl.col("cast_count")).alias(
                        "mention_frequency"
                    ),
                    (pl.col("avg_mentions_per_cast") / pl.col("cast_count")).alias(
                        "mention_ratio"
                    ),
                ]
            )

            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting mention features: {e}")
            raise
