import re
from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
import logging


class ProfileFeatureExtractor(FeatureExtractor):
    """Extract profile-based features using Polars LazyFrame with native functions."""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "has_ens",
            "has_bio",
            "has_avatar",
            "verification_count",
            "has_display_name",
            "profile_completeness",
            "fname_random_numbers",
            "fname_wallet_pattern",
            "fname_excessive_symbols",
            "fname_airdrop_terms",
            "fname_has_year",
            "bio_random_numbers",
            "bio_wallet_pattern",
            "bio_excessive_symbols",
            "bio_airdrop_terms",
            "bio_has_year",
        ]

    def get_dependencies(self) -> List[str]:
        """List of dependencies on other feature extractors."""
        return []

    def get_required_datasets(self) -> Dict[str, Dict]:
        """Declare the datasets required for this feature extractor."""
        return {
            "profile_with_addresses": {
                "columns": [
                    "fid",
                    "fname",
                    "bio",
                    "avatar_url",
                    "verified_addresses",
                    "display_name",
                ],
                "source": "farcaster",
            }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """Extract profile features from LazyFrame using native Polars functions."""
        try:
            self.logger.info("Extracting profile features...")
            profiles = loaded_datasets.get("profile_with_addresses")
            if profiles is None:
                self.logger.warning(
                    "No 'profile_with_addresses' data available for extraction."
                )
                return df

            # Ensure 'fid' is of type Int64
            profiles = profiles.with_columns(pl.col("fid").cast(pl.Int64))

            # Extract basic features
            profile_features = profiles.with_columns(
                [
                    pl.col("fname")
                    .str.contains(r"\.eth$", literal=False)
                    .cast(pl.Int64)
                    .alias("has_ens"),
                    pl.col("bio").is_not_null().cast(pl.Int64).alias("has_bio"),
                    pl.col("avatar_url")
                    .is_not_null()
                    .cast(pl.Int64)
                    .alias("has_avatar"),
                    # Compute verification_count
                    pl.when(pl.col("verified_addresses").is_in(["[]", "", None]))
                    .then(0)
                    .otherwise(pl.col("verified_addresses").str.split(",").len())
                    .alias("verification_count"),
                    pl.col("display_name")
                    .is_not_null()
                    .cast(pl.Int64)
                    .alias("has_display_name"),
                ]
            )

            # Calculate profile completeness
            profile_features = profile_features.with_columns(
                [
                    (
                        (
                            pl.col("has_bio")
                            + pl.col("has_avatar")
                            + pl.col("has_ens")
                            + (pl.col("verification_count") > 0).cast(pl.Int64)
                        ).cast(pl.Float64)
                        / 4.0
                    ).alias("profile_completeness")
                ]
            )

            # Define regex patterns
            regex_patterns = {
                "fname_random_numbers": r"\d{4,}",
                "fname_wallet_pattern": r"0x[a-fA-F0-9]{40}",
                "fname_excessive_symbols": r"[_.\-]{2,}",
                "fname_airdrop_terms": r"(?i)\b(?:airdrop|farm|degen|wojak)\b",
                "fname_has_year": r"20[12]\d",
                "bio_random_numbers": r"\d{4,}",
                "bio_wallet_pattern": r"0x[a-fA-F0-9]{40}",
                "bio_excessive_symbols": r"[_.\-]{2,}",
                "bio_airdrop_terms": r"(?i)\b(?:airdrop|farm|degen|wojak)\b",
                "bio_has_year": r"20[12]\d",
            }

            # Add pattern features using native Polars functions
            profile_features = profile_features.with_columns(
                [
                    # Fname Features
                    pl.col("fname")
                    .str.contains(regex_patterns["fname_random_numbers"])
                    .cast(pl.Float64)
                    .alias("fname_random_numbers"),
                    pl.col("fname")
                    .str.contains(regex_patterns["fname_wallet_pattern"])
                    .cast(pl.Float64)
                    .alias("fname_wallet_pattern"),
                    pl.col("fname")
                    .str.contains(regex_patterns["fname_excessive_symbols"])
                    .cast(pl.Float64)
                    .alias("fname_excessive_symbols"),
                    pl.col("fname")
                    .str.contains(regex_patterns["fname_airdrop_terms"])
                    .cast(pl.Float64)
                    .alias("fname_airdrop_terms"),
                    pl.col("fname")
                    .str.contains(regex_patterns["fname_has_year"])
                    .cast(pl.Float64)
                    .alias("fname_has_year"),
                    # Bio Features
                    pl.col("bio")
                    .str.contains(regex_patterns["bio_random_numbers"])
                    .cast(pl.Float64)
                    .alias("bio_random_numbers"),
                    pl.col("bio")
                    .str.contains(regex_patterns["bio_wallet_pattern"])
                    .cast(pl.Float64)
                    .alias("bio_wallet_pattern"),
                    pl.col("bio")
                    .str.contains(regex_patterns["bio_excessive_symbols"])
                    .cast(pl.Float64)
                    .alias("bio_excessive_symbols"),
                    pl.col("bio")
                    .str.contains(regex_patterns["bio_airdrop_terms"])
                    .cast(pl.Float64)
                    .alias("bio_airdrop_terms"),
                    pl.col("bio")
                    .str.contains(regex_patterns["bio_has_year"])
                    .cast(pl.Float64)
                    .alias("bio_has_year"),
                ]
            )

            # Verify that all expected features are present
            extracted_features = profile_features.columns
            missing_features = set(self.feature_names) - set(extracted_features)
            if missing_features:
                error_msg = f"Missing expected feature fields: {missing_features}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            self.logger.info("Profile features extracted successfully.")
            self.logger.debug(f"Extracted Features: {extracted_features}")
            return profile_features

        except Exception as e:
            self.logger.error(f"Error extracting profile features: {e}")
            raise

    def _default_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Return the DataFrame with default feature columns set to zero."""
        self.logger.info(
            "Adding default zeroed feature columns due to extraction failure."
        )
        return df.with_columns(
            [
                pl.lit(0.0).alias("has_ens"),
                pl.lit(0).alias("has_bio"),
                pl.lit(0).alias("has_avatar"),
                pl.lit(0).alias("verification_count"),
                pl.lit(0).alias("has_display_name"),
                pl.lit(0.0).alias("profile_completeness"),
                pl.lit(0.0).alias("fname_random_numbers"),
                pl.lit(0.0).alias("fname_wallet_pattern"),
                pl.lit(0.0).alias("fname_excessive_symbols"),
                pl.lit(0.0).alias("fname_airdrop_terms"),
                pl.lit(0.0).alias("fname_has_year"),
                pl.lit(0.0).alias("bio_random_numbers"),
                pl.lit(0.0).alias("bio_wallet_pattern"),
                pl.lit(0.0).alias("bio_excessive_symbols"),
                pl.lit(0.0).alias("bio_airdrop_terms"),
                pl.lit(0.0).alias("bio_has_year"),
            ]
        )
