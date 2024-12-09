from typing import List, Dict, Optional
import polars as pl
import numpy as np
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader
import logging


class NeynarScoreFeatureExtractor(FeatureExtractor):
    """Extract Neynar score features from user scores dataset"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "neynar_score",
            "avg_neynar_score",
            "neynar_score_std",
            "score_trend",
            "score_divergence",
            "relative_score_diff",
        ]
        self.logger.info(
            f"{self.__class__.__name__} initialized with features: {self.feature_names}"
        )

    def get_dependencies(self) -> List[str]:
        """List of dependencies on other feature extractors or intrinsic keys."""
        self.logger.debug("get_dependencies called.")
        # Assuming 'authenticity_score' is an existing feature from another extractor
        # return ['fid', 'authenticity_score']
        return []

    def get_required_datasets(self) -> Dict[str, Dict]:
        """
        Declare the datasets required for this feature extractor.
        """
        self.logger.debug("get_required_datasets called.")
        return {
            "neynar_user_scores": {
                "columns": ["fid", "score", "created_at"],
                "source": "nindexer",
            },
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """Extract Neynar score features using Polars' LazyFrame with native functions."""
        try:
            self.logger.info("Starting extraction of Neynar score features.")

            neynar_scores = loaded_datasets.get("neynar_user_scores")

            if neynar_scores is None:
                self.logger.warning(
                    "No 'neynar_user_scores' dataset available. Skipping Neynar score feature extraction."
                )
                return df.with_columns(
                    [
                        pl.lit(0.0).alias("neynar_score"),
                        pl.lit(0.0).alias("avg_neynar_score"),
                        pl.lit(0.0).alias("neynar_score_std"),
                        pl.lit(0.0).alias("score_trend"),
                        pl.lit(0.0).alias("score_divergence"),
                        pl.lit(0.0).alias("relative_score_diff"),
                    ]
                )

            # Process Neynar scores
            self.logger.debug("Processing Neynar user scores.")
            score_features = (
                neynar_scores.with_columns(
                    [
                        pl.col("created_at").cast(pl.Datetime),
                        pl.col("score").fill_null(0.0).cast(pl.Float64),
                    ]
                )
                .group_by("fid")
                .agg(
                    [
                        pl.col("score").last().alias("neynar_score"),
                        pl.col("score").mean().alias("avg_neynar_score"),
                        pl.col("score").std().alias("neynar_score_std"),
                        (pl.col("score").last() - pl.col("score").first()).alias(
                            "score_trend"
                        ),
                    ]
                )
            )
            self.logger.info("Neynar score features aggregated.")

            # Join with the main DataFrame
            self.logger.debug("Joining Neynar score features with the main DataFrame.")
            result = df.join(score_features, on="fid", how="left")
            self.logger.info("Neynar score features joined successfully.")

            self.logger.warning(
                "Authenticity scores not available. Skipping divergence calculations."
            )
            result = result.with_columns(
                [
                    pl.lit(0.0).alias("score_divergence"),
                    pl.lit(0.0).alias("relative_score_diff"),
                ]
            )

            # Fill remaining nulls with 0.0
            self.logger.debug("Filling remaining nulls with 0.0.")
            result = result.with_columns(
                [
                    pl.col("neynar_score").fill_null(0.0),
                    pl.col("avg_neynar_score").fill_null(0.0),
                    pl.col("neynar_score_std").fill_null(0.0),
                    pl.col("score_trend").fill_null(0.0),
                    pl.col("score_divergence").fill_null(0.0),
                    pl.col("relative_score_diff").fill_null(0.0),
                ]
            )
            self.logger.info("All null values filled with 0.0.")

            # Select only required features
            self.logger.debug("Selecting only the required Neynar score features.")
            extracted_features = result.select(["fid"] + self.feature_names)
            self.logger.info("Neynar score features extracted successfully.")

            # Log feature summary
            self.logger.debug(
                "Extracted Neynar score feature data preview (first 5 rows):"
            )
            sample = extracted_features.limit(5)
            self.logger.debug(f"Sample Neynar score features:\n{sample}")

            return extracted_features.lazy()

        except Exception as e:
            self.logger.error(f"Error extracting Neynar score features: {str(e)}")
            # Return original dataframe with default columns if error occurs
            self.logger.info(
                "Returning default Neynar score feature columns with zero values."
            )
            return df.with_columns(
                [
                    pl.lit(0.0).alias("neynar_score"),
                    pl.lit(0.0).alias("avg_neynar_score"),
                    pl.lit(0.0).alias("neynar_score_std"),
                    pl.lit(0.0).alias("score_trend"),
                    pl.lit(0.0).alias("score_divergence"),
                    pl.lit(0.0).alias("relative_score_diff"),
                ]
            )
