from typing import List, Dict
import polars as pl
import re
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class UserIdentityExtractor(FeatureExtractor):
    """Core user identity features including profile, verification, and storage"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            # Profile basics
            "has_ens",
            "has_bio",
            "has_avatar",
            "has_display_name",
            "profile_completeness",
            "display_name_length",
            "bio_length",
            # Profile content analysis
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
            "fname_entropy",
            "bio_entropy",
            # Verification metrics
            "total_verifications",
            "eth_verifications",
            "verification_consistency",
            "platforms_verified",
            "verification_span_days",
            "avg_hours_between_verifications",
            "platform_diversity",
            "verification_velocity",
            "sequential_verifications",
            "verification_gaps",
            # Storage metrics
            "storage_units",
            "storage_utilization",
            "storage_update_frequency",
            "storage_growth_rate",
            "storage_stability",
            "storage_efficiency",
            # Derived identity metrics
            "identity_strength",
            "verification_quality",
            "profile_authenticity",
            "resource_utilization",
        ]

    def get_dependencies(self) -> List[str]:
        return []

    def get_required_datasets(self) -> Dict[str, Dict]:
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
            },
            "verifications": {
                "columns": ["fid", "claim", "timestamp", "deleted_at"],
                "source": "farcaster",
            },
            "account_verifications": {
                "columns": ["fid", "platform", "verified_at"],
                "source": "farcaster",
            },
            "storage": {
                "columns": ["fid", "units", "timestamp", "deleted_at"],
                "source": "farcaster",
            },
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting user identity features...")

            # Extract each feature category
            profile_features = self._extract_profile_features(loaded_datasets)
            verification_features = self._extract_verification_features(loaded_datasets)
            storage_features = self._extract_storage_features(loaded_datasets)

            # Combine all features
            result = df.clone()
            for features in [profile_features, verification_features, storage_features]:
                if features is not None:
                    self.logger.info(f"Joining features")
                    result = result.join(features, on="fid", how="left")
                    self.logger.info(f"Joined features")

            # Calculate derived metrics
            result = self._calculate_derived_metrics(result)

            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting user identity features: {e}")
            raise

    def _extract_profile_features(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """Vectorized profile content analysis"""
        profiles = loaded_datasets.get("profile_with_addresses")
        if profiles is None:
            return pl.DataFrame({"fid": []}).lazy()

        return profiles.with_columns(
            [
                # Basic features
                pl.col("fname").str.contains(r"\.eth$").cast(pl.Int64).alias("has_ens"),
                pl.col("bio").is_not_null().cast(pl.Int64).alias("has_bio"),
                pl.col("avatar_url").is_not_null().cast(pl.Int64).alias("has_avatar"),
                pl.col("display_name")
                .is_not_null()
                .cast(pl.Int64)
                .alias("has_display_name"),
                # Content length metrics
                pl.col("display_name")
                .str.len_chars()
                .fill_null(0)
                .alias("display_name_length"),
                pl.col("bio").str.len_chars().fill_null(0).alias("bio_length"),
                # Suspicious patterns in fname
                pl.col("fname")
                .str.contains(r"\d{4,}")
                .cast(pl.Float64)
                .alias("fname_random_numbers"),
                pl.col("fname")
                .str.contains(r"0x[a-fA-F0-9]{40}")
                .cast(pl.Float64)
                .alias("fname_wallet_pattern"),
                pl.col("fname")
                .str.contains(r"[_.\-]{2,}")
                .cast(pl.Float64)
                .alias("fname_excessive_symbols"),
                pl.col("fname")
                .str.contains(r"(?i)\b(airdrop|farm|degen|wojak)\b")
                .cast(pl.Float64)
                .alias("fname_airdrop_terms"),
                pl.col("fname")
                .str.contains(r"\b(19|20)\d{2}\b")
                .cast(pl.Float64)
                .alias("fname_has_year"),
                # Suspicious patterns in bio
                pl.col("bio")
                .str.contains(r"\d{4,}")
                .cast(pl.Float64)
                .alias("bio_random_numbers"),
                pl.col("bio")
                .str.contains(r"0x[a-fA-F0-9]{40}")
                .cast(pl.Float64)
                .alias("bio_wallet_pattern"),
                pl.col("bio")
                .str.contains(r"[_.\-]{2,}")
                .cast(pl.Float64)
                .alias("bio_excessive_symbols"),
                pl.col("bio")
                .str.contains(r"(?i)\b(airdrop|farm|degen|wojak)\b")
                .cast(pl.Float64)
                .alias("bio_airdrop_terms"),
                pl.col("bio")
                .str.contains(r"\b(19|20)\d{2}\b")
                .cast(pl.Float64)
                .alias("bio_has_year"),
                # Character diversity (approximation of entropy using unique characters)
                (
                    pl.col("fname")
                    .str.replace_all(r"[^a-zA-Z0-9]", "")
                    .str.to_lowercase()
                    .str.explode()
                    .n_unique()
                    .truediv(pl.col("fname").str.len_chars() + 1)
                ).alias("fname_entropy"),
                (
                    pl.col("bio")
                    .str.replace_all(r"[^a-zA-Z0-9]", "")
                    .str.to_lowercase()
                    .str.explode()
                    .n_unique()
                    .truediv(pl.col("bio").str.len_chars() + 1)
                ).alias("bio_entropy"),
            ]
        ).with_columns(
            [
                # Profile completeness score
                (
                    (
                        pl.col("has_bio")
                        + pl.col("has_avatar")
                        + pl.col("has_ens")
                        + pl.col("has_display_name")
                    )
                    / 4.0
                ).alias("profile_completeness"),
            ]
        )

    def _extract_verification_features(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        verifications = loaded_datasets.get("verifications")
        account_verifs = loaded_datasets.get("account_verifications")

        if verifications is None and account_verifs is None:
            return pl.DataFrame({"fid": []}).lazy()

        features = pl.DataFrame(schema={"fid": pl.Int64}).lazy()

        if verifications is not None:
            onchain_features = (
                verifications.filter(pl.col("deleted_at").is_null())
                .with_columns([pl.col("timestamp").cast(pl.Datetime)])
                .group_by("fid")
                .agg(
                    [
                        pl.len().alias("total_verifications"),
                        pl.col("claim")
                        .str.contains("ethSignature")
                        .sum()
                        .alias("eth_verifications"),
                        pl.col("timestamp")
                        .diff()
                        .dt.total_hours()
                        .std()
                        .alias("verification_timing_std"),
                        pl.col("timestamp")
                        .diff()
                        .dt.total_hours()
                        .mean()
                        .alias("avg_hours_between_verifications"),
                        (pl.col("timestamp").diff().dt.total_hours() < 1)
                        .sum()
                        .alias("sequential_verifications"),
                        (pl.col("timestamp").diff().dt.total_hours() > 24)
                        .sum()
                        .alias("verification_gaps"),
                    ]
                )
                .with_columns(
                    [
                        (
                            pl.col("verification_timing_std")
                            / (pl.col("avg_hours_between_verifications") + 1)
                        ).alias("verification_consistency")
                    ]
                )
            )
            features = features.join(onchain_features, on="fid", how="left")

        if account_verifs is not None:
            platform_features = (
                account_verifs.with_columns([pl.col("verified_at").cast(pl.Datetime)])
                .group_by("fid")
                .agg(
                    [
                        pl.n_unique("platform").alias("platforms_verified"),
                        pl.col("platform").n_unique().alias("platform_diversity"),
                        (pl.col("verified_at").max() - pl.col("verified_at").min())
                        .dt.total_days()
                        .alias("verification_span_days"),
                        pl.len().cast(pl.Float64).alias("verification_velocity"),
                    ]
                )
            )
            features = features.join(platform_features, on="fid", how="left")

        return features

    def _extract_storage_features(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        storage = loaded_datasets.get("storage")
        if storage is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            storage.filter(pl.col("deleted_at").is_null())
            .with_columns([pl.col("timestamp").cast(pl.Datetime)])
            .group_by("fid")
            .agg(
                [
                    pl.col("units").mean().alias("storage_units"),
                    pl.col("units").max().alias("storage_utilization"),
                    pl.len().alias("storage_update_frequency"),
                    (pl.col("units").last() - pl.col("units").first()).alias(
                        "storage_growth"
                    ),
                    pl.col("units").std().alias("storage_stability"),
                    (pl.col("units").mean() / pl.col("units").max()).alias(
                        "storage_efficiency"
                    ),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("storage_growth") / pl.col("storage_update_frequency")
                    ).alias("storage_growth_rate")
                ]
            )
        )

    def _calculate_derived_metrics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(
            [
                # Identity strength combines profile and verification metrics
                (
                    (
                        pl.col("profile_completeness") * 0.4
                        + pl.col("verification_consistency").fill_null(0) * 0.3
                        + (pl.col("platforms_verified").fill_null(0) / 5.0) * 0.3
                    )
                ).alias("identity_strength"),
                # Verification quality considers diversity and consistency
                (
                    (
                        pl.col("platform_diversity").fill_null(0) * 0.4
                        + pl.col("verification_consistency").fill_null(0) * 0.3
                        + (
                            pl.col("eth_verifications").fill_null(0)
                            / pl.col("total_verifications").fill_null(1)
                        )
                        * 0.3
                    )
                ).alias("verification_quality"),
                # Profile authenticity based on content analysis
                (
                    1.0
                    - (
                        pl.col("fname_random_numbers").fill_null(0) * 0.2
                        + pl.col("fname_wallet_pattern").fill_null(0) * 0.2
                        + pl.col("fname_airdrop_terms").fill_null(0) * 0.2
                        + pl.col("bio_airdrop_terms").fill_null(0) * 0.2
                        + pl.col("bio_wallet_pattern").fill_null(0) * 0.2
                    )
                ).alias("profile_authenticity"),
                # Resource utilization efficiency
                (
                    (
                        pl.col("storage_efficiency").fill_null(0) * 0.5
                        + pl.col("storage_stability").fill_null(0) * 0.3
                        + (1.0 - pl.col("storage_growth_rate").fill_null(0).clip(0, 1))
                        * 0.2
                    )
                ).alias("resource_utilization"),
            ]
        )
