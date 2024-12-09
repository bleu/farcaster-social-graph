from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class VerificationFeatureExtractor(FeatureExtractor):
    """Extract verification-related features"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "total_verifications",
            "eth_verifications",
            "verification_timing_std",
            "platforms_verified",
            "verification_span_days",
            "avg_hours_between_verifications",
            "std_hours_between_verifications",
            "verification_consistency",
        ]

    def get_dependencies(self) -> List[str]:
        return ["fid"]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "verifications": {
                "columns": ["fid", "claim", "timestamp", "deleted_at"],
                "source": "farcaster",
            },
            "account_verifications": {
                "columns": ["fid", "platform", "verified_at"],
                "source": "farcaster",
            },
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting verification features...")
            verifications = loaded_datasets.get("verifications")
            account_verifs = loaded_datasets.get("account_verifications")

            result = df

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
                            pl.col("timestamp")
                            .diff()
                            .dt.total_hours()
                            .std()
                            .alias("std_hours_between_verifications"),
                        ]
                    )
                )
                result = result.join(onchain_features, on="fid", how="left")

            if account_verifs is not None:
                platform_features = (
                    account_verifs.with_columns(
                        [pl.col("verified_at").cast(pl.Datetime)]
                    )
                    .group_by("fid")
                    .agg(
                        [
                            pl.n_unique("platform").alias("platforms_verified"),
                            (pl.col("verified_at").max() - pl.col("verified_at").min())
                            .dt.total_days()
                            .alias("verification_span_days"),
                        ]
                    )
                )
                result = result.join(platform_features, on="fid", how="left")

            # Calculate verification consistency
            result = result.with_columns(
                [
                    (
                        pl.col("std_hours_between_verifications")
                        / (pl.col("avg_hours_between_verifications") + 1)
                    ).alias("verification_consistency")
                ]
            )

            return result.select(["fid"] + self.feature_names).fill_null(0)

        except Exception as e:
            self.logger.error(f"Error extracting verification features: {e}")
            raise
