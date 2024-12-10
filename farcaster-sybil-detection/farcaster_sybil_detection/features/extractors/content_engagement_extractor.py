from typing import List, Dict
import polars as pl
import re
import numpy as np
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class ContentEngagementExtractor(FeatureExtractor):
    """Content creation and engagement pattern analysis"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            # Basic content metrics
            "cast_count",
            "reply_count",
            "mention_count",
            "avg_cast_length",
            # "media_usage_rate",
            "link_usage_rate",
            # "unique_mentions",
            # "mention_diversity",
            # Content characteristics
            # "media_ratio",
            # "link_ratio",
            # "text_ratio",
            # "avg_media_per_cast",
            # "avg_links_per_cast",
            "content_type_diversity",
            "content_complexity",
            # Engagement metrics
            "total_reactions",
            "like_count",
            "recast_count",
            # "reaction_ratio",
            "engagement_rate",
            "viral_coefficient",
            "audience_reach",
            "engagement_consistency",
            # Conversation metrics
            "conversation_depth",
            # "reply_quality",
            "conversation_initiation_rate",
            # 'response_rate',
            "thread_participation",
            # "discussion_impact",
            # Content quality
            "content_originality",
            # "topic_consistency",
            "vocabulary_richness",
            # "sentence_complexity",
            "hashtag_usage",
            # Spam detection
            # "spam_pattern_score",
            # "bot_likelihood",
            # "template_usage",
            # "repetition_rate",
            # "promotional_content_ratio",
            # "automation_signals",
            # Advanced metrics
            # "content_virality",
            # "influence_per_cast",
            # "engagement_quality",
            # "content_sustainability",
            # "audience_retention",
            # "content_effectiveness",
        ]

    def get_dependencies(self) -> List[str]:
        return []

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "casts": {
                "columns": [
                    "fid",
                    "text",
                    "parent_hash",
                    "mentions",
                    "deleted_at",
                    "timestamp",
                    "embeds",
                ],
                "source": "farcaster",
            },
            "reactions": {
                "columns": [
                    "fid",
                    "reaction_type",
                    "target_fid",
                    "timestamp",
                    "deleted_at",
                ],
                "source": "farcaster",
            },
            "follow_counts": {
                "columns": ["fid", "follower_count", "following_count"],
                "source": "nindexer",
            },
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting content engagement features...")

            # Extract each feature category
            content_metrics = self._extract_content_metrics(loaded_datasets)
            engagement_metrics = self._extract_engagement_metrics(loaded_datasets)
            conversation_metrics = self._extract_conversation_metrics(loaded_datasets)
            quality_metrics = self._extract_quality_metrics(loaded_datasets)
            spam_metrics = self._extract_spam_metrics(loaded_datasets)
            # advanced_metrics = self._extract_advanced_metrics(loaded_datasets)

            # Combine all features
            result = df.clone()
            for metrics in [
                content_metrics,
                engagement_metrics,
                conversation_metrics,
                quality_metrics,
                spam_metrics,
                # advanced_metrics,
            ]:
                if metrics is not None:
                    result = result.join(metrics, on="fid", how="left")

            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting content engagement features: {e}")
            raise

    def _extract_content_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            casts.filter(pl.col("deleted_at").is_null())
            .with_columns(
                [
                    # Basic metrics - cast to Float64 explicitly
                    pl.col("text")
                    .str.len_chars()
                    .cast(pl.Float64)
                    .alias("cast_length"),
                    pl.col("parent_hash")
                    .is_not_null()
                    .cast(pl.Float64)
                    .alias("is_reply"),
                    pl.col("text")
                    .str.contains(r"@\w+")
                    .cast(pl.Float64)
                    .alias("has_mentions"),
                    # Media and link analysis
                    pl.col("text")
                    .str.count_matches(r"https?://")
                    .cast(pl.Float64)
                    .alias("link_count"),
                    # Content analysis
                    pl.col("text")
                    .str.split(" ")
                    .list.len()
                    .cast(pl.Float64)
                    .alias("word_count"),
                    pl.col("text")
                    .str.split(".")
                    .list.len()
                    .cast(pl.Float64)
                    .alias("sentence_count"),
                    pl.col("text")
                    .str.split(" ")
                    .list.unique()
                    .list.len()
                    .cast(pl.Float64)
                    .alias("unique_words"),
                    # Average word length
                    pl.col("text")
                    .str.split(" ")
                    .list.eval(pl.element().str.len_chars())
                    .mean()
                    .cast(pl.Float64)
                    .alias("avg_word_length"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Volume metrics
                    pl.len().cast(pl.Float64).alias("cast_count"),
                    pl.col("cast_length").mean().alias("avg_cast_length"),
                    pl.col("is_reply").sum().alias("reply_count"),
                    pl.col("has_mentions").sum().alias("mention_count"),
                    # Media and link metrics
                    (pl.col("link_count").sum() / pl.len()).alias("link_usage_rate"),
                    # Content diversity
                    (pl.col("unique_words") / (pl.col("word_count") + 1))
                    .mean()
                    .alias("content_type_diversity"),
                    # Complexity metrics
                    pl.col("avg_word_length")
                    .mul(pl.col("sentence_count"))
                    .mean()
                    .fill_null(0)  # Handle NaN values
                    .alias("content_complexity"),
                ]
            )
        )

    def _extract_engagement_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        reactions = loaded_datasets.get("reactions")
        follow_counts = loaded_datasets.get("follow_counts")

        if reactions is None:
            return pl.DataFrame({"fid": []}).lazy()

        # Calculate basic reaction metrics
        reaction_metrics = (
            reactions.filter(pl.col("deleted_at").is_null())
            .group_by("fid")
            .agg(
                [
                    pl.len().cast(pl.Float64).alias("total_reactions"),
                    (pl.col("reaction_type") == 1)
                    .sum()
                    .cast(pl.Float64)
                    .alias("like_count"),
                    (pl.col("reaction_type") == 2)
                    .sum()
                    .cast(pl.Float64)
                    .alias("recast_count"),
                    pl.n_unique("target_fid").cast(pl.Float64).alias("unique_reactors"),
                ]
            )
        )

        # Add audience-aware metrics if user metrics available
        if follow_counts is not None:
            reaction_metrics = reaction_metrics.join(
                follow_counts.select(["fid", "follower_count"]), on="fid", how="left"
            ).with_columns(
                [
                    (pl.col("total_reactions") / (pl.col("follower_count") + 1))
                    .fill_null(0)  # Handle NaN values
                    .alias("engagement_rate"),
                    (pl.col("unique_reactors") / (pl.col("follower_count") + 1))
                    .fill_null(0)  # Handle NaN values
                    .alias("audience_reach"),
                ]
            )

        # Calculate derived engagement metrics
        return reaction_metrics.with_columns(
            [
                (pl.col("recast_count") / (pl.col("total_reactions") + 1))
                .fill_null(0)  # Handle NaN values
                .alias("viral_coefficient"),
                (
                    pl.col("total_reactions").std()
                    / (pl.col("total_reactions").mean() + 1)
                )
                .fill_null(0)  # Handle NaN values
                .alias("engagement_consistency"),
            ]
        )

    def _extract_conversation_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            casts.filter(pl.col("deleted_at").is_null())
            .with_columns(
                [
                    pl.col("text").is_not_null().cast(pl.Int64).alias("is_reply"),
                    # pl.col("replies").list.len().alias("reply_count"),
                    # Track conversation depth
                    pl.col("parent_hash").cum_count().over("fid").alias("thread_depth"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Conversation initiation vs participation
                    (pl.col("parent_hash").is_null().sum() / pl.len()).alias(
                        "conversation_initiation_rate"
                    ),
                    # # Response patterns
                    # (pl.col("replies").list.len().sum() / pl.len()).alias(
                    #     "response_rate"
                    # ),
                    # Thread participation
                    pl.col("thread_depth").mean().alias("conversation_depth"),
                    # Discussion impact
                    # (pl.col("reply_count").sum() / pl.len()).alias("discussion_impact"),
                    # Thread participation patterns
                    (pl.col("is_reply").sum() / pl.len()).alias("thread_participation"),
                    # Reply quality (based on engagement received on replies)
                    # (
                    #     pl.when(pl.col("is_reply"))
                    #     .then(pl.col("reply_count"))
                    #     .otherwise(0)
                    #     .mean()
                    # ).alias("reply_quality"),
                ]
            )
        )

    def _extract_quality_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            return pl.DataFrame({"fid": []}).lazy()

        # First let's check the schema of our input
        self.logger.info(f"Input casts schema: {casts.schema}")

        # Try each metric separately to identify which one causes the error
        try:
            base = casts.filter(pl.col("deleted_at").is_null())

            # Vocabulary richness
            vocab_df = base.with_columns(
                [
                    pl.col("text").str.split(" ").alias("words"),
                    pl.col("fid").alias("fid_vocab"),
                ]
            ).select(
                [
                    "fid_vocab",
                    pl.col("words").list.n_unique().alias("word_unique_count"),
                ]
            )

            # Basic hashtag usage
            hashtag_df = base.with_columns(
                [
                    pl.col("text").str.count_matches(r"https?://").alias("link_count"),
                    pl.col("fid").alias("fid_hash"),
                ]
            )

            # Content originality
            originality_df = base.group_by("fid").agg(
                [
                    pl.n_unique("text").alias("unique_text_count"),
                    pl.count("text").alias("total_text_count"),
                ]
            )

            # Combine metrics
            result = (
                vocab_df.rename({"fid_vocab": "fid"})
                .join(
                    hashtag_df.group_by("fid_hash")
                    .agg([pl.col("link_count").mean().alias("hashtag_usage")])
                    .rename({"fid_hash": "fid"}),
                    on="fid",
                    how="left",
                )
                .join(originality_df, on="fid", how="left")
                .with_columns(
                    [
                        pl.col("word_unique_count").mean().alias("vocabulary_richness"),
                        pl.col("unique_text_count")
                        .truediv(pl.col("total_text_count"))
                        .alias("content_originality"),
                    ]
                )
                .select(
                    [
                        "fid",
                        "vocabulary_richness",
                        "hashtag_usage",
                        "content_originality",
                    ]
                )
            )

            self.logger.info(f"Final quality metrics schema: {result.schema}")
            return result

        except Exception as e:
            self.logger.error(f"Error in quality metrics: {str(e)}")
            raise

    def _extract_spam_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """Vectorized spam detection using Polars native operations"""
        casts = loaded_datasets.get("casts")
        if casts is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            casts.filter(pl.col("deleted_at").is_null())
            .with_columns(
                [
                    # Spam keywords - combine all patterns into single regex
                    pl.col("text")
                    .str.count_matches(
                        r"\b(airdrop|giveaway|limited|offer|earn|claim|buy|sell|profit|token|moon|urgent|hurry)\b"
                    )
                    .alias("spam_keyword_count"),
                    # Promotional patterns
                    pl.col("text")
                    .str.count_matches(r"\b(check out|click here|sign up|join now)\b")
                    .alias("promotional_count"),
                    # Links and mentions
                    pl.col("text")
                    .str.count_matches(r"https?://[^\s]+")
                    .alias("link_count"),
                    # Suspicious patterns
                    pl.col("text")
                    .str.count_matches(r"[!$@#%]{2,}|\d{10,}|[A-Z]{5,}")
                    .alias("suspicious_pattern_count"),
                    # Time differences for burst detection
                    pl.col("timestamp").cast(pl.Datetime),
                ]
            )
            .sort("timestamp")
            .group_by("fid")
            .agg(
                [
                    # Normalized spam scores
                    (pl.col("spam_keyword_count").sum() / pl.len()).alias(
                        "spam_keyword_ratio"
                    ),
                    (pl.col("promotional_count").sum() / pl.len()).alias(
                        "promotional_ratio"
                    ),
                    (pl.col("suspicious_pattern_count").sum() / pl.len()).alias(
                        "suspicious_pattern_ratio"
                    ),
                    # Burst detection using time differences
                    (
                        pl.col("timestamp")
                        .diff()
                        .dt.total_seconds()
                        .filter(pl.col("timestamp").diff().dt.total_seconds() < 60)
                        .count()
                        / pl.len()
                    ).alias("rapid_post_ratio"),
                    # Content uniqueness
                    (pl.col("text").n_unique() / pl.len()).alias("content_uniqueness"),
                    # Combined spam score
                    (
                        (
                            pl.col("spam_keyword_count").sum()
                            + pl.col("promotional_count").sum() * 2
                            + pl.col("suspicious_pattern_count").sum() * 1.5
                        )
                        / pl.len()
                    ).alias("spam_score"),
                ]
            )
        )

    def _extract_advanced_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        reactions = loaded_datasets.get("reactions")
        follow_counts = loaded_datasets.get("follow_counts")

        if casts is None or reactions is None:
            return pl.DataFrame({"fid": []}).lazy()

        # Calculate virality and influence metrics
        cast_metrics = (
            casts.filter(pl.col("deleted_at").is_null())
            .join(
                reactions.filter(pl.col("deleted_at").is_null())
                .group_by("target_fid")
                .agg(
                    [
                        pl.len().alias("reactions_received"),
                        pl.n_unique("fid").alias("unique_reactors"),
                    ]
                )
                .rename({"target_fid": "fid"}),
                on="fid",
                how="left",
            )
            .with_columns(
                [
                    pl.col("reactions_received")
                    .fill_null(0)
                    .alias("reactions_received"),
                    pl.col("unique_reactors").fill_null(0).alias("unique_reactors"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Virality metrics
                    (pl.col("reactions_received") / pl.len()).alias("content_virality"),
                    (pl.col("unique_reactors") / pl.len()).alias("influence_per_cast"),
                    # Engagement quality (weighted by unique reactors)
                    (
                        pl.col("reactions_received")
                        * pl.col("unique_reactors")
                        / (pl.len() * pl.col("unique_reactors").max())
                    ).alias("engagement_quality"),
                    # Content sustainability (consistency of engagement)
                    (
                        1
                        - pl.col("reactions_received").std()
                        / (pl.col("reactions_received").mean() + 1)
                    ).alias("content_sustainability"),
                ]
            )
        )

        # Add audience retention if user metrics available
        if follow_counts is not None:
            cast_metrics = cast_metrics.join(
                follow_counts.select(["fid", "follower_count"]), on="fid", how="left"
            ).with_columns(
                [
                    # Audience retention (engagement relative to follower count)
                    (pl.col("unique_reactors") / (pl.col("follower_count") + 1)).alias(
                        "audience_retention"
                    )
                ]
            )

        # Calculate overall content effectiveness
        return cast_metrics.with_columns(
            [
                (
                    (
                        pl.col("content_virality") * 0.3
                        + pl.col("engagement_quality") * 0.3
                        + pl.col("content_sustainability") * 0.2
                        + pl.col("audience_retention").fill_null(0) * 0.2
                    )
                ).alias("content_effectiveness")
            ]
        )
