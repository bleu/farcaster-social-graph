from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class ReputationMetaExtractor(FeatureExtractor):
    """Reputation scores and derived meta features"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            # Core reputation scores
            # "authenticity_score",
            "influence_score",
            # "trust_score",
            # "quality_score",
            "activity_score",
            # Network reputation
            "network_quality",
            "peer_reputation",
            "authority_score",
            "centrality_score",
            "relationship_health",
            "community_standing",
            # # Engagement reputation
            # "engagement_quality",
            # "content_impact",
            # "interaction_value",
            # "contribution_score",
            # "participation_quality",
            # "value_generation",
            # Behavioral reputation
            "behavior_consistency",
            "pattern_reliability",
            "activity_authenticity",
            "response_quality",
            "communication_style",
            "collaboration_score",
            # Trust metrics
            "verification_strength",
            "identity_confidence",
            "spam_resistance",
            "abuse_likelihood",
            "credibility_score",
            "reliability_index",
            # Meta scores
            "growth_trajectory",
            "sustainability_index",
            "adaptability_score",
            "resilience_metric",
            "innovation_index",
            "potential_score",
            # were missing
            "verification_timing_std",
            "sequential_verifications",
            "verification_gaps",
            "neynar_score",
            "avg_neynar_score",
            "neynar_score_std",
            "score_trend",
            "score_divergence",
            "relative_score_diff",
            "reach_metrics",
            "engagement_impact",
        ]

    def get_dependencies(self) -> List[str]:
        return ["fid"]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "neynar_user_scores": {
                "columns": ["fid", "score", "created_at"],
                "source": "nindexer",
            },
            "follow_counts": {
                "columns": ["fid", "follower_count", "following_count", "created_at"],
                "source": "nindexer",
            },
            # "engagement_metrics": {
            #     "columns": ["fid", "engagement_rate", "quality_score"],
            #     "source": "farcaster",
            # },
            "verifications": {
                "columns": ["fid", "timestamp", "deleted_at", "claim"],
                "source": "farcaster",
            },
            # "activity_data": {
            #     "columns": ["fid", "activity_type", "timestamp", "impact_score"],
            #     "source": "farcaster",
            # },
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting reputation meta features...")

            # Extract each category of features
            core_scores = self._extract_core_scores(loaded_datasets)
            network_reputation = self._extract_network_reputation(loaded_datasets)
            # engagement_reputation = self._extract_engagement_reputation(loaded_datasets)
            behavioral_reputation = self._extract_behavioral_reputation(loaded_datasets)
            trust_metrics = self._extract_trust_metrics(loaded_datasets)
            meta_scores = self._extract_meta_scores(loaded_datasets)
            verification_metrics = self._extract_verification_metrics(loaded_datasets)

            # Combine all features
            result = df.clone()
            for features in [
                core_scores,
                network_reputation,
                # engagement_reputation,
                behavioral_reputation,
                trust_metrics,
                meta_scores,
                verification_metrics,
            ]:
                if features is not None:
                    result = result.join(features, on="fid", how="left")

            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting reputation meta features: {e}")
            raise

    def _extract_verification_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        verifications = loaded_datasets.get("verifications")
        neynar_scores = loaded_datasets.get("neynar_user_scores")

        if verifications is None:
            return pl.DataFrame({"fid": []}).lazy()

        verification_metrics = (
            verifications.filter(pl.col("deleted_at").is_null())
            .with_columns([pl.col("timestamp").cast(pl.Datetime)])
            .group_by("fid")
            .agg(
                [
                    pl.col("timestamp")
                    .diff()
                    .dt.total_hours()
                    .std()
                    .alias("verification_timing_std"),
                    (pl.col("timestamp").diff().dt.total_hours() < 1)
                    .sum()
                    .alias("sequential_verifications"),
                    (pl.col("timestamp").diff().dt.total_hours() > 24)
                    .sum()
                    .alias("verification_gaps"),
                ]
            )
        )

        if neynar_scores is not None:
            neynar_metrics = (
                neynar_scores.sort("created_at")
                .group_by("fid")
                .agg(
                    [
                        pl.col("score").mean().alias("avg_neynar_score"),
                        pl.col("score").std().alias("neynar_score_std"),
                        pl.col("score").last().alias("neynar_score"),
                        (pl.col("score").last() - pl.col("score").first()).alias(
                            "score_trend"
                        ),
                        (pl.col("score").max() - pl.col("score").min()).alias(
                            "score_divergence"
                        ),
                        (
                            (pl.col("score").last() - pl.col("score").first())
                            / pl.col("score").first()
                        ).alias("relative_score_diff"),
                    ]
                )
            )
            verification_metrics = verification_metrics.join(
                neynar_metrics, on="fid", how="left"
            )

        return verification_metrics.with_columns(
            [
                # pl.col("unique_reactors").alias("reach_metrics"),
                # (pl.col("total_reactions") / pl.col("cast_count")).alias(
                #     "engagement_impact"
                # ),
            ]
        )

    def _extract_core_scores(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        neynar_scores = loaded_datasets.get("neynar_user_scores")
        follow_counts = loaded_datasets.get("follow_counts")

        result = pl.DataFrame(schema={"fid": pl.Int64}).lazy()

        # Process Neynar scores
        if neynar_scores is not None:
            neynar_features = (
                neynar_scores.sort("created_at")
                .group_by("fid")
                .agg(
                    [
                        pl.col("score").last().alias("neynar_score"),
                        (pl.col("score").last() - pl.col("score").first())
                        .truediv(pl.col("score").first())
                        .alias("score_growth"),
                    ]
                )
            )
            result = result.join(neynar_features, on="fid", how="left")

        # Process user metrics
        if follow_counts is not None:
            activity_features = (
                follow_counts.sort("created_at")
                .group_by("fid")
                .agg(
                    [
                        (
                            pl.col("follower_count").last()
                            / (pl.col("following_count").last() + 1)
                        ).alias("influence_score"),
                        (
                            pl.col("follower_count").last()
                            / pl.col("follower_count").first()
                        ).alias("activity_score"),
                    ]
                )
            )
            result = result.join(activity_features, on="fid", how="left")

        return result

    def _extract_network_reputation(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        follow_counts = loaded_datasets.get("follow_counts")
        activity_data = loaded_datasets.get("activity_data")

        if follow_counts is None and activity_data is None:
            return None

        result = pl.DataFrame(schema={"fid": pl.Int64}).lazy()

        if follow_counts is not None:
            network_features = (
                follow_counts.with_columns(
                    [
                        pl.col("follower_count").cast(pl.Float64),
                        pl.col("following_count").cast(pl.Float64),
                    ]
                )
                .group_by("fid")
                .agg(
                    [
                        # Network quality based on follower/following ratio
                        (
                            pl.col("follower_count") / (pl.col("following_count") + 1)
                        ).alias("network_quality"),
                        # Centrality score based on total connections
                        (
                            (pl.col("follower_count") + pl.col("following_count"))
                            / pl.col("follower_count").max()
                        ).alias("centrality_score"),
                        # Relationship health based on balance
                        (
                            1
                            - (
                                pl.col("follower_count") - pl.col("following_count")
                            ).abs()
                            / (pl.col("follower_count") + pl.col("following_count") + 1)
                        ).alias("relationship_health"),
                    ]
                )
            )
            result = result.join(network_features, on="fid", how="left")

        if activity_data is not None:
            interaction_features = activity_data.group_by("fid").agg(
                [
                    # Peer reputation based on interaction quality
                    pl.col("impact_score").mean().alias("peer_reputation"),
                    # Authority score based on high-impact activities
                    pl.col("impact_score")
                    .filter(pl.col("impact_score") > pl.col("impact_score").mean())
                    .count()
                    .truediv(pl.count())
                    .alias("authority_score"),
                    # Community standing based on consistent engagement
                    pl.col("impact_score")
                    .std()
                    .truediv(pl.col("impact_score").mean())
                    .alias("community_standing"),
                ]
            )
            result = result.join(interaction_features, on="fid", how="left")

        return result

    def _extract_engagement_reputation(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        engagement_metrics = loaded_datasets.get("engagement_metrics")
        activity_data = loaded_datasets.get("activity_data")

        if engagement_metrics is None and activity_data is None:
            return None

        result = pl.DataFrame(schema={"fid": pl.Int64}).lazy()

        if engagement_metrics is not None:
            engagement_features = engagement_metrics.group_by("fid").agg(
                [
                    # Quality of engagement
                    pl.col("quality_score").mean().alias("engagement_quality"),
                    # Impact of content
                    (pl.col("engagement_rate") * pl.col("quality_score"))
                    .mean()
                    .alias("content_impact"),
                    # Value generated through interactions
                    (pl.col("quality_score") / pl.col("engagement_rate"))
                    .mean()
                    .alias("interaction_value"),
                ]
            )
            result = result.join(engagement_features, on="fid", how="left")

        if activity_data is not None:
            contribution_features = activity_data.group_by("fid").agg(
                [
                    # Quality of participation
                    pl.col("impact_score").mean().alias("participation_quality"),
                    # Overall contribution score
                    (
                        pl.col("impact_score")
                        * pl.col("impact_score").count().over("activity_type")
                    )
                    .mean()
                    .alias("contribution_score"),
                    # Value generation capability
                    pl.col("impact_score")
                    .filter(pl.col("impact_score") > pl.col("impact_score").mean())
                    .mean()
                    .alias("value_generation"),
                ]
            )
            result = result.join(contribution_features, on="fid", how="left")

        return result

    def _extract_behavioral_reputation(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        activity_data = loaded_datasets.get("activity_data")
        if activity_data is None:
            return None

        return (
            activity_data.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("impact_score").cast(pl.Float64),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Consistency in behavior
                    pl.col("timestamp")
                    .diff()
                    .dt.total_hours()
                    .std()
                    .truediv(pl.col("timestamp").diff().dt.total_hours().mean())
                    .alias("behavior_consistency"),
                    # Reliability of activity patterns
                    pl.col("timestamp")
                    .diff()
                    .dt.total_hours()
                    .filter(pl.col("impact_score") > pl.col("impact_score").mean())
                    .std()
                    .alias("pattern_reliability"),
                    # Authenticity of activities
                    (
                        pl.col("impact_score").std() / pl.col("impact_score").mean()
                    ).alias("activity_authenticity"),
                    # Quality of responses
                    pl.col("impact_score")
                    .filter(pl.col("activity_type") == "response")
                    .mean()
                    .alias("response_quality"),
                    # Communication style consistency
                    pl.col("activity_type")
                    .value_counts()
                    .std()
                    .truediv(pl.col("activity_type").value_counts().mean())
                    .alias("communication_style"),
                    # Collaboration tendency
                    pl.col("impact_score")
                    .filter(
                        pl.col("activity_type").is_in(["reply", "mention", "reaction"])
                    )
                    .mean()
                    .alias("collaboration_score"),
                ]
            )
        )

    def _extract_trust_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        verifications = loaded_datasets.get("verifications")

        if verifications is None:
            return None

        result = pl.DataFrame(schema={"fid": pl.Int64}).lazy()

        if verifications is not None:
            # Calculate verification metrics
            trust_features = (
                verifications.filter(pl.col("deleted_at").is_null())
                .group_by("fid")
                .agg(
                    [
                        # Count total verifications
                        pl.count().alias("verification_count"),
                        # Count unique verification types from claim
                        pl.col("claim").n_unique().alias("platform_count"),
                    ]
                )
                .with_columns(
                    [
                        # Strength of verification
                        (pl.col("verification_count") * pl.col("platform_count")).alias(
                            "verification_strength"
                        ),
                        # Confidence in identity (normalized by max platforms)
                        (pl.col("platform_count") / 10.0).alias("identity_confidence"),
                        # Spam resistance (inverse of verification frequency)
                        (1 - 1 / (pl.col("verification_count") + 1)).alias(
                            "spam_resistance"
                        ),
                        # Abuse likelihood (inverse of platform diversity)
                        (1 - pl.col("platform_count") / pl.col("verification_count"))
                        .clip(0, 1)
                        .alias("abuse_likelihood"),
                        # Credibility score
                        (pl.col("platform_count") / pl.col("verification_count")).alias(
                            "credibility_score"
                        ),
                        # Reliability index
                        (
                            (pl.col("verification_count") * pl.col("platform_count"))
                            / 100.0
                        )
                        .clip(0, 1)
                        .alias("reliability_index"),
                    ]
                )
            )
            result = result.join(trust_features, on="fid", how="left")

        return result

    def _extract_meta_scores(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        follow_counts = loaded_datasets.get("follow_counts")
        activity_data = loaded_datasets.get("activity_data")

        if follow_counts is None and activity_data is None:
            return None

        result = pl.DataFrame(schema={"fid": pl.Int64}).lazy()

        if follow_counts is not None:
            growth_features = (
                follow_counts.sort("created_at")
                .group_by("fid")
                .agg(
                    [
                        # Growth trajectory
                        (
                            (
                                pl.col("follower_count").last()
                                - pl.col("follower_count").first()
                            )
                            / pl.col("follower_count").first()
                        ).alias("growth_trajectory"),
                        # Sustainability of growth
                        (
                            1
                            - pl.col("follower_count").diff().std()
                            / pl.col("follower_count").diff().mean()
                        ).alias("sustainability_index"),
                    ]
                )
            )
            result = result.join(growth_features, on="fid", how="left")

        if activity_data is not None:
            meta_features = activity_data.group_by("fid").agg(
                [
                    # Adaptability to different contexts
                    pl.col("activity_type")
                    .n_unique()
                    .truediv(pl.col("activity_type").count())
                    .alias("adaptability_score"),
                    # Resilience to negative impact
                    (
                        1
                        - pl.col("impact_score")
                        .filter(pl.col("impact_score") < 0)
                        .count()
                        / pl.count()
                    ).alias("resilience_metric"),
                    # Innovation in interactions
                    pl.col("activity_type")
                    .value_counts()
                    .filter(pl.col("count") < pl.col("count").mean())
                    .count()
                    .truediv(pl.col("activity_type").n_unique())
                    .alias("innovation_index"),
                    # Future potential score
                    (
                        pl.col("impact_score").mean()
                        * pl.col("impact_score").count().over("activity_type").mean()
                    ).alias("potential_score"),
                ]
            )
            result = result.join(meta_features, on="fid", how="left")

        return result
