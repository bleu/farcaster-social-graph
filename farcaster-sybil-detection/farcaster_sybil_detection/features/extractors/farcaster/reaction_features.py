from typing import List, Dict, Optional
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig


class ReactionFeatureExtractor(FeatureExtractor):
    """Extract reaction/engagement patterns"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "total_reactions",
            "like_count",
            "recast_count",
            "unique_users_reacted_to",
            "reaction_diversity",
            "like_ratio",
            "recast_ratio",
        ]

    def get_dependencies(self) -> List[str]:
        """List of required input columns from the main DataFrame"""
        # return ['reaction_type', 'target_fid', 'timestamp', 'deleted_at']
        return []

    def get_required_datasets(self) -> Dict[str, Dict]:
        """
        Declare the datasets required for this feature extractor.
        """
        return {
            "reactions": {
                "columns": [
                    "fid",
                    "reaction_type",
                    "target_fid",
                    "timestamp",
                    "deleted_at",
                ],
                "source": "farcaster",
            }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """Extract reaction features from LazyFrame"""
        try:
            self.logger.info("Extracting reaction features...")
            reactions = loaded_datasets.get("reactions")

            if reactions is None:
                self.logger.warning("No reactions data available for extraction.")
                return df  # Returning the input LazyFrame
            # Filter valid reactions
            valid_reactions = reactions.filter(pl.col("deleted_at").is_null())

            # Group and aggregate
            reaction_features = valid_reactions.group_by("fid").agg(
                [
                    pl.count().alias("total_reactions"),
                    (pl.col("reaction_type") == 1)
                    .sum()
                    .alias("like_count"),  # Assuming 1 = like
                    (pl.col("reaction_type") == 2)
                    .sum()
                    .alias("recast_count"),  # Assuming 2 = recast
                    pl.col("target_fid").n_unique().alias("unique_users_reacted_to"),
                    pl.col("reaction_type").count().alias("reaction_diversity"),
                ]
            )

            # Calculate ratios
            reaction_features = reaction_features.with_columns(
                [
                    (pl.col("like_count") / (pl.col("total_reactions") + 1)).alias(
                        "like_ratio"
                    ),
                    (pl.col("recast_count") / (pl.col("total_reactions") + 1)).alias(
                        "recast_ratio"
                    ),
                ]
            )

            # Reaction type distribution
            unique_reaction_types = (
                valid_reactions.select(["reaction_type"])
                .unique()
                .cast(pl.Int64)
                .collect()["reaction_type"]
                .to_list()
            )
            reaction_type_distribution = (
                valid_reactions.group_by(["fid", "reaction_type"])
                .agg(pl.count().alias("count"))
                .collect()
                .pivot(index="fid", on="reaction_type", values="count")
                .fill_null(0)
                .rename({str(k): f"reaction_type_{k}" for k in unique_reaction_types})
            )

            # Join with main reaction_features
            reaction_features = reaction_features.join(
                reaction_type_distribution.lazy(), on="fid", how="left"
            )

            # Select required features
            extracted_features = reaction_features.select(["fid"] + self.feature_names)

            self.logger.info("Reaction features extracted successfully.")
            self.logger.info(
                f"Extracted features: {extracted_features.collect().head(5)}"
            )
            return extracted_features

        except Exception as e:
            self.logger.error(f"Error extracting reaction features: {e}")
            raise
