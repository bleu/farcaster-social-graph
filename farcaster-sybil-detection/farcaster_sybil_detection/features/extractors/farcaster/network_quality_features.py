from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class NetworkQualityExtractor(FeatureExtractor):
    """Extract network quality metrics with enhanced power user analysis"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "power_reply_count",
            "power_mentions_count",
            "power_user_reactions",
            "power_user_interaction_ratio",
            "follower_authenticity_score",
            "network_diversity",
            "engagement_quality",
        ]

    def get_dependencies(self) -> List[str]:
        return [
            "fid",
            # 'engagement_score',
            "following_count",
            "follower_count",
        ]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "power_users": {"columns": ["fid"], "source": "farcaster"},
            "casts": {
                "columns": ["fid", "parent_fid", "mentions", "timestamp", "deleted_at"],
                "source": "farcaster",
            },
            "reactions": {
                "columns": ["fid", "target_fid", "timestamp", "deleted_at"],
                "source": "farcaster",
            },
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting network quality features...")
            power_users = loaded_datasets.get("power_users")
            casts = loaded_datasets.get("casts")
            reactions = loaded_datasets.get("reactions")

            if power_users is None or casts is None or reactions is None:
                self.logger.warning("Required datasets not available")
                return self._get_default_features(df)

            # Get power user FIDs
            power_fids = (
                power_users.select("fid").collect()["fid"].cast(pl.Int64).unique()
            )
            power_fid_str = str(power_fids[0])  # For string matching in mentions

            # Calculate power user interactions in casts
            cast_metrics = (
                casts.filter(pl.col("deleted_at").is_null())
                .with_columns(
                    [
                        pl.col("parent_fid")
                        .cast(pl.Int64)
                        .is_in(power_fids)
                        .alias("is_power_reply"),
                        pl.when(
                            pl.col("mentions").is_not_null()
                            & pl.col("mentions").str.contains(power_fid_str)
                        )
                        .then(1)
                        .otherwise(0)
                        .alias("has_power_mention"),
                    ]
                )
                .group_by("fid")
                .agg(
                    [
                        pl.sum("is_power_reply").alias("power_reply_count"),
                        pl.sum("has_power_mention").alias("power_mentions_count"),
                    ]
                )
            )

            # Calculate power user interactions in reactions
            reaction_metrics = (
                reactions.filter(pl.col("deleted_at").is_null())
                .with_columns(
                    [
                        pl.col("target_fid")
                        .cast(pl.Int64)
                        .is_in(power_fids)
                        .alias("is_power_reaction")
                    ]
                )
                .group_by("fid")
                .agg(
                    [
                        pl.sum("is_power_reaction").alias("power_user_reactions"),
                        pl.len().alias("total_reactions"),
                    ]
                )
            )

            # Combine metrics and calculate derived features
            result = df.join(cast_metrics, on="fid", how="left")
            result = result.join(reaction_metrics, on="fid", how="left")

            result = result.with_columns(
                [
                    pl.col("power_reply_count").fill_null(0),
                    pl.col("power_mentions_count").fill_null(0),
                    pl.col("power_user_reactions").fill_null(0),
                    pl.col("total_reactions").fill_null(0),
                ]
            )

            # Calculate quality metrics
            result = result.with_columns(
                [
                    # Power user interaction ratio
                    (
                        (
                            pl.col("power_reply_count")
                            + pl.col("power_mentions_count")
                            + pl.col("power_user_reactions")
                        )
                        / (pl.col("total_reactions") + 1)
                    ).alias("power_user_interaction_ratio"),
                    # Network diversity (engagement spread)
                    (
                        pl.col("engagement_score") / (pl.col("following_count") + 1)
                    ).alias("network_diversity"),
                    # Follower authenticity score
                    (
                        1.0
                        - (pl.col("following_count") - pl.col("follower_count")).abs()
                        / (pl.col("following_count") + pl.col("follower_count") + 1)
                    ).alias("follower_authenticity_score"),
                    # Placeholder for engagement quality (can be enhanced)
                    (
                        (
                            pl.col("power_user_reactions")
                            / (pl.col("total_reactions") + 1)
                        )
                        * (pl.col("follower_count") / (pl.col("following_count") + 1))
                    ).alias("engagement_quality"),
                ]
            )

            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting network quality features: {e}")
            raise

    def _get_default_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Return DataFrame with default zero values for all features"""
        return df.with_columns(
            [
                pl.lit(0).alias("power_reply_count"),
                pl.lit(0).alias("power_mentions_count"),
                pl.lit(0).alias("power_user_reactions"),
                pl.lit(0.0).alias("power_user_interaction_ratio"),
                pl.lit(0.0).alias("follower_authenticity_score"),
                pl.lit(0.0).alias("network_diversity"),
                pl.lit(0.0).alias("engagement_quality"),
            ]
        )
