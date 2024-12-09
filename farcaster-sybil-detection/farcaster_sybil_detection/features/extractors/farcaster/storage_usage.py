import re
from typing import Dict, List

import polars as pl

from farcaster_sybil_detection.data.dataset_loader import DatasetLoader
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor

from ..base import FeatureConfig, FeatureExtractor


class StorageExtractor(FeatureExtractor):
    """Extract storage usage patterns"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "avg_storage_units",
            "max_storage_units",
            "storage_update_count",
        ]

    def get_dependencies(self) -> List[str]:
        return ["fid"]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "storage": {
                "columns": ["fid", "units", "deleted_at"],
                "source": "farcaster",
            }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting storage features...")
            storage = loaded_datasets.get("storage")

            if storage is None:
                self.logger.warning("No storage data available")
                return df

            storage_features = (
                storage.filter(pl.col("deleted_at").is_null())
                .group_by("fid")
                .agg(
                    [
                        pl.col("units").mean().alias("avg_storage_units"),
                        pl.col("units").max().alias("max_storage_units"),
                        pl.len().alias("storage_update_count"),
                    ]
                )
            )

            return storage_features.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting storage features: {e}")
            raise
