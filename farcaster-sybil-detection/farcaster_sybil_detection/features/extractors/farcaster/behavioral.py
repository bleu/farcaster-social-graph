from typing import List, Dict, Optional
import polars as pl
import numpy as np
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class BehavioralFeatureExtractor(FeatureExtractor):
    """Extract behavioral features from user activity"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "cast_count",
            "reply_count",
            "total_reactions",
            "avg_cast_length",
            "rapid_actions",
            "avg_hours_between_actions",
            "std_hours_between_actions",
        ]

    def get_dependencies(self) -> List[str]:
        return ["fid", "casts", "parent_hash", "timestamp"]

    def get_required_datasets(self) -> Dict[str, Dict]:
        """
        Declare the datasets required for this feature extractor.
        """
        return {
            "casts": {
                "columns": ["fid", "casts", "parent_hash", "timestamp"],
                "source": "farcaster",
            }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting behavioral features...")
            casts = loaded_datasets.get("casts")

            if casts is None:
                self.logger.warning("No casts data available for extraction.")
                return df

            # Group and aggregate behavioral metrics
            behavioral_features = casts.group_by("fid").agg(
                [
                    pl.col("casts").count().alias("cast_count"),
                    pl.col("parent_hash").is_not_null().sum().alias("reply_count"),
                    pl.col("total_reactions").sum().alias("total_reactions"),
                    pl.col("casts")
                    .map_elements(function=lambda x: np.mean([len(cast) for cast in x]))
                    .alias("avg_cast_length"),
                    (pl.col("timestamp").diff().dt.total_hours() < 1)
                    .sum()
                    .alias("rapid_actions"),
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
                ]
            )

            # Select required features
            extracted_features = behavioral_features.select(
                ["fid"] + self.feature_names
            )

            self.logger.info("Behavioral features extracted successfully.")
            return extracted_features
        except Exception as e:
            self.logger.error(f"Error extracting behavioral features: {e}")
            raise
