from typing import List, Dict, Optional
import polars as pl
from farcaster_sybil_detection.features.extractors.base import (
    FeatureExtractor,
    FeatureConfig,
)
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class InfluenceFeatureExtractor(FeatureExtractor):
    """Extract influence metrics"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "influence_score",
            "reach_metrics",
            "engagement_impact",
            "network_centrality",
        ]

    def get_dependencies(self) -> List[str]:
        """List of required input columns"""
        return ["fid", "activity_count", "reaction_count", "follower_count"]

    def get_required_datasets(self) -> Dict[str, Dict]:
        """
        Declare the datasets required for this feature extractor.
        """
        return {
            "influence_logs": {
                "columns": [
                    "fid",
                    "activity_count",
                    "reaction_count",
                    "follower_count",
                ],
                "source": "farcaster",
            }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """Extract influence metrics from LazyFrame"""
        try:
            self.logger.info("Extracting influence features...")
            influence_logs = loaded_datasets.get("influence_logs")

            if influence_logs is None:
                self.logger.warning("No influence_logs data available for extraction.")
                return df

            # Example feature extraction logic
            influence_features = influence_logs.group_by("fid").agg(
                [
                    pl.col("activity_count").sum().alias("influence_score"),
                    pl.col("reaction_count").mean().alias("reach_metrics"),
                    pl.col("follower_count").mean().alias("engagement_impact"),
                    pl.col("follower_count").n_unique().alias("network_centrality"),
                ]
            )

            # Select required features
            extracted_features = influence_features.select(["fid"] + self.feature_names)

            self.logger.info("Influence features extracted successfully.")
            return extracted_features

        except Exception as e:
            self.logger.error(f"Error extracting influence features: {e}")
            raise
