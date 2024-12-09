from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class PowerUserInteractionExtractor(FeatureExtractor):
    """Analyze interactions with power users"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "power_user_replies",
            "power_user_mentions",
            "power_user_reactions",
            "power_user_interaction_ratio",
            "avg_power_user_response_time",
            "power_user_engagement_quality",
        ]

    def get_dependencies(self) -> List[str]:
        return [
            "fid",
            #  'total_casts',
            "total_reactions",
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
            self.logger.info("Extracting power user interaction features...")
            power_users = loaded_datasets.get("power_users")
            casts = loaded_datasets.get("casts")
            reactions = loaded_datasets.get("reactions")

            if not all([power_users, casts, reactions]):
                return self._get_default_features(df)

            # Get power user FIDs
            power_fids = (
                power_users.select("fid").collect()["fid"].cast(pl.Int64).unique()
            )
            power_fid_str = str(power_fids[0])

            # Process cast interactions
            power_cast_features = (
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
                        pl.col("timestamp").cast(pl.Datetime),
                    ]
                )
                .group_by("fid")
                .agg(
                    [
                        pl.sum("is_power_reply").alias("power_user_replies"),
                        pl.sum("has_power_mention").alias("power_user_mentions"),
                        pl.len().alias("total_casts"),
                    ]
                )
            )

            # Process reaction interactions
            power_reaction_features = (
                reactions.filter(pl.col("deleted_at").is_null())
                .with_columns(
                    [
                        pl.col("target_fid")
                        .cast(pl.Int64)
                        .is_in(power_fids)
                        .alias("is_power_reaction"),
                        pl.col("timestamp").cast(pl.Datetime),
                    ]
                )
                .group_by("fid")
                .agg(
                    [
                        pl.sum("is_power_reaction").alias("power_user_reactions"),
                        pl.when(pl.col("is_power_reaction"))
                        .then(pl.col("timestamp").diff().dt.total_hours().mean())
                        .alias("avg_power_user_response_time"),
                        pl.len().alias("total_reactions"),
                    ]
                )
            )

            # Combine features and calculate metrics
            result = df.join(power_cast_features, on="fid", how="left")
            result = result.join(power_reaction_features, on="fid", how="left")

            result = result.with_columns(
                [
                    # Fill nulls
                    pl.col("power_user_replies").fill_null(0),
                    pl.col("power_user_mentions").fill_null(0),
                    pl.col("power_user_reactions").fill_null(0),
                    pl.col("avg_power_user_response_time").fill_null(0),
                    # Calculate interaction ratio
                    (
                        (
                            pl.col("power_user_replies").fill_null(0)
                            + pl.col("power_user_mentions").fill_null(0)
                            + pl.col("power_user_reactions").fill_null(0)
                        )
                        / (
                            pl.col("total_casts").fill_null(0)
                            + pl.col("total_reactions").fill_null(0)
                            + 1
                        )
                    ).alias("power_user_interaction_ratio"),
                    # Calculate engagement quality
                    (
                        pl.col("power_user_reactions").fill_null(0)
                        / (
                            pl.col("power_user_replies").fill_null(0)
                            + pl.col("power_user_mentions").fill_null(0)
                            + 1
                        )
                    ).alias("power_user_engagement_quality"),
                ]
            )

            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting power user interaction features: {e}")
            raise

    def _get_default_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(
            [
                pl.lit(0).alias("power_user_replies"),
                pl.lit(0).alias("power_user_mentions"),
                pl.lit(0).alias("power_user_reactions"),
                pl.lit(0.0).alias("power_user_interaction_ratio"),
                pl.lit(0.0).alias("avg_power_user_response_time"),
                pl.lit(0.0).alias("power_user_engagement_quality"),
            ]
        )
