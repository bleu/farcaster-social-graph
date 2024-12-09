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
            # "link_usage_rate",
            # "unique_mentions",
            # "mention_diversity",
            # Content characteristics
            # "media_ratio",
            # "link_ratio",
            # "text_ratio",
            # "avg_media_per_cast",
            # "avg_links_per_cast",
            # "content_type_diversity",
            # "content_complexity",
            # Engagement metrics
            # "total_reactions",
            # "like_count",
            # "recast_count",
            # "reaction_ratio",
            # "engagement_rate",
            # "viral_coefficient",
            # "audience_reach",
            # "engagement_consistency",
            # # Conversation metrics
            # "conversation_depth",
            # "reply_quality",
            # "conversation_initiation_rate",
            # # 'response_rate',
            # "thread_participation",
            # "discussion_impact",
            # # Content quality
            # "content_originality",
            # "topic_consistency",
            # "vocabulary_richness",
            # "sentence_complexity",
            # "hashtag_usage",
            # "url_credibility",
            # # Spam detection
            # "spam_pattern_score",
            # "bot_likelihood",
            # "template_usage",
            # "repetition_rate",
            # "promotional_content_ratio",
            # "automation_signals",
            # # Advanced metrics
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
            # engagement_metrics = self._extract_engagement_metrics(loaded_datasets)
            # conversation_metrics = self._extract_conversation_metrics(loaded_datasets)
            # quality_metrics = self._extract_quality_metrics(loaded_datasets)
            # spam_metrics = self._extract_spam_metrics(loaded_datasets)
            # advanced_metrics = self._extract_advanced_metrics(loaded_datasets)

            # Combine all features
            result = df.clone()
            for metrics in [
                content_metrics,
                # engagement_metrics,
                # conversation_metrics,
                # quality_metrics,
                # spam_metrics,
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
                    # Basic metrics
                    pl.col("text").str.len_chars().alias("cast_length"),
                    pl.col("parent_hash")
                    .is_not_null()
                    .cast(pl.Int64)
                    .alias("is_reply"),
                    pl.col("text")
                    .str.contains(r"@\w+")
                    .cast(pl.Int64)
                    .alias("has_mentions"),
                    # Media and link analysis
                    # pl.col("embeds").str.count_matches("image").alias("media_count"),
                    # pl.col("text").str.count_matches(r"https?://").alias("link_count"),
                    # Content analysis
                    # pl.col("text").str.split(" ").count().alias("word_count"),
                    # pl.col("text").str.split(".").count().alias("sentence_count"),
                    # pl.col("text")
                    # .str.split(" ")
                    # .list.unique()
                    # .count()
                    # .alias("unique_words"),
                    # pl.col("text")
                    # .str.split(" ")
                    # .str.len_chars()
                    # .mean()
                    # .alias("avg_word_length"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Volume metrics
                    pl.len().alias("cast_count"),
                    pl.col("cast_length").mean().alias("avg_cast_length"),
                    pl.col("is_reply").sum().alias("reply_count"),
                    pl.col("has_mentions").sum().alias("mention_count"),
                    # Media and link metrics
                    # (pl.col("media_count").sum() / pl.len()).alias("media_usage_rate"),
                    # (pl.col("link_count").sum() / pl.len()).alias("link_usage_rate"),
                    # Content diversity
                    # (pl.col("unique_words") / (pl.col("word_count") + 1))
                    # .mean()
                    # .alias("content_type_diversity"),
                    # Complexity metrics
                    # pl.col("avg_word_length")
                    # .mul(pl.col("sentence_count"))
                    # .mean()
                    # .alias("content_complexity"),
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
                    pl.len().alias("total_reactions"),
                    (pl.col("reaction_type") == 1).sum().alias("like_count"),
                    (pl.col("reaction_type") == 2).sum().alias("recast_count"),
                    pl.n_unique("target_fid").alias("unique_reactors"),
                ]
            )
        )

        # Add audience-aware metrics if user metrics available
        if follow_counts is not None:
            reaction_metrics = reaction_metrics.join(
                follow_counts.select(["fid", "follower_count"]), on="fid", how="left"
            ).with_columns(
                [
                    (pl.col("total_reactions") / (pl.col("follower_count") + 1)).alias(
                        "engagement_rate"
                    ),
                    (pl.col("unique_reactors") / (pl.col("follower_count") + 1)).alias(
                        "audience_reach"
                    ),
                ]
            )

        # Calculate derived engagement metrics
        return reaction_metrics.with_columns(
            [
                (pl.col("recast_count") / (pl.col("total_reactions") + 1)).alias(
                    "viral_coefficient"
                ),
                (
                    pl.col("total_reactions").std()
                    / (pl.col("total_reactions").mean() + 1)
                ).alias("engagement_consistency"),
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
                    # pl.col("text")
                    # .is_not_null()
                    # .cast(pl.Int64)
                    # .alias("is_reply"),
                    # pl.col('replies').list.len().alias('reply_count'),
                    # Track conversation depth
                    pl.col("parent_hash")
                    .cum_count()
                    .over("fid")
                    .alias("thread_depth"),
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
                    # (pl.col('replies').list.len().sum() / pl.len())
                    # .alias('response_rate'),
                    # Thread participation
                    pl.col("thread_depth").mean().alias("conversation_depth"),
                    # Discussion impact
                    (pl.col("reply_count").sum() / pl.len()).alias("discussion_impact"),
                    # Thread participation patterns
                    (pl.col("is_reply").sum() / pl.len()).alias("thread_participation"),
                    # Reply quality (based on engagement received on replies)
                    (
                        pl.when(pl.col("is_reply"))
                        .then(pl.col("reply_count"))
                        .otherwise(0)
                        .mean()
                    ).alias("reply_quality"),
                ]
            )
        )

    def _extract_quality_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            casts.filter(pl.col("deleted_at").is_null())
            .with_columns(
                [
                    pl.col("text")
                    .str.split(" ")
                    .list.n_unique()
                    .alias("vocabulary_richness"),
                    pl.col("text")
                    .str.split(".")
                    .str.split(" ")
                    .list.len()
                    .mean()
                    .alias("sentence_complexity"),
                    pl.col("text")
                    .str.count_matches(r"https?://")
                    .alias("hashtag_usage"),
                    pl.col("text").str.count_matches(r"#\w+").alias("url_credibility"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Content originality (based on text uniqueness)
                    pl.col("text").n_unique() / pl.len().alias("content_originality"),
                    # Topic consistency (based on word overlap between posts)
                    pl.col("text")
                    .str.split(" ")
                    .list.n_unique()
                    .mean()
                    .alias("topic_consistency"),
                ]
            )
        )

    def _extract_spam_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        casts = loaded_datasets.get("casts")
        if casts is None:
            return pl.DataFrame({"fid": []}).lazy()
        return (
            casts.filter(pl.col("deleted_at").is_null())
            .with_columns(
                [
                    # Vectorized spam word detection
                    pl.col("text")
                    .str.count_matches(
                        r"\b(buy|sell|profit|earn|airdrop|giveaway|limited|offer)\b"
                    )
                    .alias("spam_words"),
                    # Promotional terms
                    pl.col("text")
                    .str.count_matches(r"\b(check out|click here|sign up|join now)\b")
                    .alias("promotional_terms"),
                    # Excessive symbols
                    pl.col("text")
                    .str.count_matches(r"[!$@#%]{2,}")
                    .alias("excessive_symbols"),
                    pl.col("timestamp")
                    .diff()
                    .dt.total_seconds()
                    .alias("time_between_posts"),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    pl.col("spam_words").mean().alias("spam_pattern_score"),
                    (pl.col("time_between_posts") < 60).mean().alias("bot_likelihood"),
                    pl.col("promotional_terms")
                    .mean()
                    .alias("promotional_content_ratio"),
                    (
                        (pl.col("time_between_posts") < 60).mean()
                        + (pl.col("text").n_unique() / pl.len())
                    ).alias("automation_signals"),
                    (pl.col("text").n_unique() / pl.len()).alias("repetition_rate"),
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
