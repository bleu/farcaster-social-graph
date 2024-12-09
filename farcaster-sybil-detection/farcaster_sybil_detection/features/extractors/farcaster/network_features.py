from typing import List, Dict, Optional
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class NetworkFeatureExtractor(FeatureExtractor):
    """Extract network-based features"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "follower_count",
            "following_count",
            "follow_ratio",
            "unique_followers",
            "unique_following",
            "network_growth_rate",
            "follow_velocity",
        ]

    def get_dependencies(self) -> List[str]:
        """List of required input columns"""
        return ["fid", "target_fid"]

    def get_required_datasets(self) -> Dict[str, Dict]:
        """
        Declare the datasets required for this feature extractor.
        """
        return {
            "network": {
                "columns": ["fid", "target_fid", "timestamp", "action_type"],
                "source": "nindexer",
            }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """Extract network features from LazyFrame"""
        try:
            self.logger.info("Extracting network features...")
            network = loaded_datasets.get("network")

            if network is None:
                self.logger.warning("No network data available for extraction.")
                return df

            # Group and aggregate follower and following counts
            followers = (
                network.filter(pl.col("action_type") == "follow")
                .group_by("fid")
                .agg(
                    [
                        pl.col("target_fid").n_unique().alias("follower_count"),
                        pl.col("target_fid").n_unique().alias("unique_followers"),
                    ]
                )
            )

            following = (
                network.filter(pl.col("action_type") == "follow")
                .group_by("target_fid")
                .agg(
                    [
                        pl.col("fid").n_unique().alias("following_count"),
                        pl.col("fid").n_unique().alias("unique_following"),
                    ]
                )
                .rename({"target_fid": "fid"})
            )

            # Join with main DataFrame
            network_features = df.join(followers, on="fid", how="left").join(
                following, on="fid", how="left"
            )

            # Calculate ratios and growth rates
            network_features = network_features.with_columns(
                [
                    (pl.col("follower_count") / (pl.col("following_count") + 1)).alias(
                        "follow_ratio"
                    ),
                    # Placeholder for 'network_growth_rate' and 'follow_velocity'
                    # These would require historical data and timestamps to calculate
                    pl.lit(0.0).alias(
                        "network_growth_rate"
                    ),  # Replace with actual computation
                    pl.lit(0.0).alias(
                        "follow_velocity"
                    ),  # Replace with actual computation
                ]
            )

            # Select required features
            extracted_features = network_features.select(["fid"] + self.feature_names)

            self.logger.info("Network features extracted successfully.")
            return extracted_features

        except Exception as e:
            self.logger.error(f"Error extracting network features: {e}")
            raise
