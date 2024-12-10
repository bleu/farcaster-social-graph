from typing import Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class NetworkAnalysisExtractor(FeatureExtractor):
    """Comprehensive network analysis including basic and advanced metrics"""

    def __init__(self, data_loader: DatasetLoader):
        super().__init__(data_loader)
        self.feature_names = self.get_feature_names()

    @classmethod
    def get_feature_names(cls):
        return [
            # Basic network metrics
            "follow_ratio",
            # Growth and velocity
            "network_growth_rate",
            "follow_velocity",
            "network_age_hours",
            "growth_stability",
            "follower_growth_rate",
            "following_growth_rate",
            # Network quality metrics
            "follow_reciprocity",
            "network_density",
            # "cluster_coefficient",
            "network_reach",
            # "influential_followers",
            # "influencer_ratio",
            # Network stability
            "network_churn_rate",
            "relationship_longevity",
            "network_volatility",
            "stable_connections",
            # Advanced metrics
            "network_centrality",
            "bridge_score",
            "community_embedding",
            "network_diversity",
            # were missing
            # "follower_authenticity_score",
            # "follower_ratio_log",
            # "unique_follower_ratio_log",
            # "follow_velocity_log",
            # "follower_ratio_capped",
            # "unique_follower_ratio_capped",
            # "follow_velocity_capped",
            # "target_url_diversity",
            # "follower_retention",
        ]

    def get_required_datasets(self) -> Dict[str, Dict]:
        return {
            "follows": {
                "columns": [
                    "fid",
                    "target_fid",
                    "timestamp",
                    "created_at",
                    "deleted_at",
                ],
                "source": "nindexer",
            },
            "follow_counts": {
                "columns": ["fid", "follower_count", "following_count", "created_at"],
                "source": "nindexer",
            },
            "reactions": {
                "columns": ["fid", "target_fid", "timestamp", "deleted_at"],
                "source": "farcaster",
            },
            # 'user_influence': {
            #     'columns': ['fid', 'influence_score', 'timestamp'],
            #     'source': 'nindexer'
            # }
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.debug("Extracting network analysis features...")

            # Extract each feature category
            basic_metrics = self._extract_basic_metrics(loaded_datasets)
            growth_metrics = self._extract_growth_metrics(loaded_datasets)
            quality_metrics = self._extract_quality_metrics(loaded_datasets)
            stability_metrics = self._extract_stability_metrics(loaded_datasets)
            centrality_metrics = self._extract_centrality_metrics(loaded_datasets)
            advanced_metrics = self._extract_advanced_metrics(loaded_datasets)
            # derived_metrics = self._extract_derived_metrics(loaded_datasets)
            engagement_diversity_metrics = self._extract_engagement_diversity(
                loaded_datasets
            )

            # Combine all features
            result = df.clone()
            for metrics in [
                basic_metrics,
                growth_metrics,
                quality_metrics,
                stability_metrics,
                centrality_metrics,
                advanced_metrics,
                # derived_metrics,
                engagement_diversity_metrics,
            ]:
                if metrics is not None:
                    result = result.join(metrics, on="fid", how="left")

            return result.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting network analysis features: {e}")
            raise

    def _extract_engagement_diversity(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        links = loaded_datasets.get("links")
        reactions = loaded_datasets.get("reactions")

        if links is None or reactions is None:
            return None

        return (
            reactions.filter(pl.col("deleted_at").is_null())
            .group_by("fid")
            .agg(
                [
                    pl.n_unique("target_url").alias("target_url_diversity"),
                    # Calculate retention
                    (1 - pl.col("deleted_at").is_not_null().sum() / pl.len()).alias(
                        "follower_retention"
                    ),
                ]
            )
        )

    def _extract_derived_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        follows = loaded_datasets.get("follows")
        if follows is None:
            return pl.DataFrame({"fid": []}).lazy()

        derived = (
            follows.filter(pl.col("deleted_at").is_null())
            .group_by("fid")
            .agg(
                [
                    pl.col("follower_count").log1p().alias("follower_ratio_log"),
                    pl.col("unique_followers")
                    .log1p()
                    .alias("unique_follower_ratio_log"),
                    pl.col("follow_velocity").log1p().alias("follow_velocity_log"),
                ]
            )
            .with_columns(
                [
                    pl.col("follower_ratio")
                    .clip(0, pl.col("follower_ratio").quantile(0.99))
                    .alias("follower_ratio_capped"),
                    pl.col("unique_follower_ratio")
                    .clip(0, pl.col("unique_follower_ratio").quantile(0.99))
                    .alias("unique_follower_ratio_capped"),
                    pl.col("follow_velocity")
                    .clip(0, pl.col("follow_velocity").quantile(0.99))
                    .alias("follow_velocity_capped"),
                ]
            )
        )

        # Calculate follower authenticity score
        return derived.with_columns(
            [
                (
                    1.0
                    - (pl.col("following_count") - pl.col("follower_count")).abs()
                    / (pl.col("following_count") + pl.col("follower_count") + 1)
                ).alias("follower_authenticity_score")
            ]
        )

    def _extract_centrality_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """Vectorized network centrality calculation"""
        follows = loaded_datasets.get("follows")
        if follows is None:
            return pl.DataFrame({"fid": []}).lazy()

        # Clean follows data
        follows_clean = follows.filter(pl.col("deleted_at").is_null())

        # Calculate outgoing connections (following)
        out_degree = follows_clean.group_by("fid").agg(
            [
                pl.col("target_fid").count().alias("out_degree"),
                pl.col("target_fid").n_unique().alias("unique_out_degree"),
            ]
        )

        # Calculate incoming connections (followers)
        in_degree = (
            follows_clean.group_by("target_fid")
            .agg(
                [
                    pl.col("fid").count().alias("in_degree"),
                    pl.col("fid").n_unique().alias("unique_in_degree"),
                ]
            )
            .rename({"target_fid": "fid"})
        )

        # Join and calculate metrics
        return out_degree.join(in_degree, on="fid", how="outer").with_columns(
            [
                # Normalize degrees by total network size
                pl.col("out_degree").fill_null(0),
                pl.col("in_degree").fill_null(0),
                pl.col("unique_out_degree").fill_null(0),
                pl.col("unique_in_degree").fill_null(0),
                # Calculate centrality scores
                (
                    (pl.col("out_degree") + pl.col("in_degree"))
                    / (pl.col("out_degree").max() + pl.col("in_degree").max())
                )
                .fill_null(0)
                .alias("centrality_score"),
                # Calculate bridging score
                (
                    pl.col("unique_out_degree")
                    * pl.col("unique_in_degree")
                    / (pl.col("out_degree") + pl.col("in_degree") + 1)
                )
                .fill_null(0)
                .alias("bridge_score"),
                # Calculate clustering coefficient
                (
                    pl.col("unique_out_degree")
                    * pl.col("unique_in_degree")
                    / ((pl.col("out_degree") + 1) * (pl.col("in_degree") + 1))
                )
                .fill_null(0)
                .alias("clustering_coefficient"),
            ]
        )

    def _extract_basic_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        follows = loaded_datasets.get("follows")
        follow_counts = loaded_datasets.get("follow_counts")
        if follows is None or follow_counts is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            follows.filter(pl.col("deleted_at").is_null())
            .with_columns([pl.col("timestamp").cast(pl.Datetime)])
            .group_by("fid")
            .agg(
                [
                    # Basic counts
                    pl.col("target_fid").count().alias("following_count"),
                    pl.col("target_fid").n_unique().alias("unique_following"),
                    # Calculate unique followers
                    pl.col("target_fid")
                    .filter(pl.col("target_fid").is_not_null())
                    .n_unique()
                    .alias("unique_followers"),
                    # Calculate ratios
                    (
                        pl.col("target_fid").count()
                        / (pl.col("target_fid").n_unique() + 1)
                    ).alias("follow_ratio"),
                ]
            )
            .join(
                follow_counts.group_by("fid").agg(
                    [pl.col("follower_count").last().alias("follower_count")]
                ),
                on="fid",
                how="left",
            )
        )

    def _extract_growth_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        follows = loaded_datasets.get("follows")
        follow_counts = loaded_datasets.get("follow_counts")

        if follows is None or follow_counts is None:
            return pl.DataFrame({"fid": []}).lazy()

        # Process follows for growth metrics
        growth_features = (
            follows.filter(pl.col("deleted_at").is_null())
            .with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("created_at").cast(pl.Datetime),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Time-based metrics
                    (pl.col("timestamp").max() - pl.col("timestamp").min())
                    .dt.total_hours()
                    .alias("network_age_hours"),
                    # Growth rates
                    (
                        pl.col("timestamp").count()
                        / pl.col("timestamp").diff().dt.total_hours().mean()
                    ).alias("follow_velocity"),
                    # Stability metrics
                    pl.col("timestamp")
                    .diff()
                    .dt.total_hours()
                    .std()
                    .alias("growth_stability"),
                ]
            )
        )

        # Process follow counts for trending metrics
        count_features = (
            follow_counts.sort("created_at")
            .group_by("fid")
            .agg(
                [
                    (
                        (
                            pl.col("follower_count").last()
                            - pl.col("follower_count").first()
                        )
                        / pl.col("follower_count").first()
                    ).alias("follower_growth_rate"),
                    (
                        (
                            pl.col("following_count").last()
                            - pl.col("following_count").first()
                        )
                        / pl.col("following_count").first()
                    ).alias("following_growth_rate"),
                    (
                        pl.col("follower_count").diff().std()
                        + pl.col("following_count").diff().std()
                    ).alias("network_growth_rate"),
                ]
            )
        )

        return growth_features.join(count_features, on="fid", how="left")

    def _extract_quality_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        follows = loaded_datasets.get("follows")
        user_influence = loaded_datasets.get("user_influence")

        if follows is None:
            return pl.DataFrame({"fid": []}).lazy()

        # Calculate reciprocal follows
        follows_clean = follows.filter(pl.col("deleted_at").is_null())

        reciprocal_follows = (
            follows_clean.join(
                follows_clean.select(["target_fid", "fid"]).rename(
                    {"target_fid": "reciprocal_target"}
                ),
                left_on=["fid", "target_fid"],
                right_on=["reciprocal_target", "fid"],
            )
            .group_by("fid")
            .agg(
                [
                    pl.count().alias("reciprocal_count"),
                    (pl.count() / pl.col("target_fid").count()).alias(
                        "follow_reciprocity"
                    ),
                ]
            )
        )

        # Calculate network density and reach
        network_metrics = follows_clean.group_by("fid").agg(
            [
                (pl.n_unique("target_fid") / pl.count()).alias("network_density"),
                pl.col("target_fid").n_unique().alias("network_reach"),
            ]
        )

        # Calculate influence-based metrics if available
        if user_influence is not None:
            influence_metrics = (
                follows_clean.join(
                    user_influence.select(["fid", "influence_score"]),
                    left_on="target_fid",
                    right_on="fid",
                )
                .group_by("fid")
                .agg(
                    [
                        pl.col("influence_score").mean().alias("influential_followers"),
                        (pl.col("influence_score") > pl.col("influence_score").mean())
                        .sum()
                        .alias("influencer_ratio"),
                    ]
                )
            )

            network_metrics = network_metrics.join(
                influence_metrics, on="fid", how="left"
            )

        return reciprocal_follows.join(network_metrics, on="fid", how="left")

    def _extract_stability_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        follows = loaded_datasets.get("follows")
        if follows is None:
            return pl.DataFrame({"fid": []}).lazy()

        return (
            follows.filter(pl.col("deleted_at").is_null())
            .with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    pl.col("created_at").cast(pl.Datetime),
                ]
            )
            .group_by("fid")
            .agg(
                [
                    # Calculate churn rate
                    (pl.col("deleted_at").is_not_null().sum() / pl.len()).alias(
                        "network_churn_rate"
                    ),
                    # Calculate average relationship duration
                    (
                        pl.col("deleted_at").fill_null(pl.col("timestamp"))
                        - pl.col("created_at")
                    )
                    .dt.total_days()
                    .mean()
                    .alias("relationship_longevity"),
                    # Calculate network volatility
                    pl.col("timestamp")
                    .diff()
                    .dt.total_hours()
                    .std()
                    .alias("network_volatility"),
                    # Calculate stable connections (lasting > 30 days)
                    (
                        (
                            pl.col("deleted_at").fill_null(pl.col("timestamp"))
                            - pl.col("created_at")
                        ).dt.total_days()
                        > 30
                    )
                    .sum()
                    .alias("stable_connections"),
                ]
            )
        )

    def _extract_advanced_metrics(
        self, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        follows = loaded_datasets.get("follows")
        if follows is None:
            return pl.DataFrame({"fid": []}).lazy()

        follows_clean = follows.filter(pl.col("deleted_at").is_null())

        # Calculate network centrality
        centrality = follows_clean.group_by("fid").agg(
            [
                # Degree centrality
                (pl.len() + pl.col("target_fid").n_unique()).alias(
                    "network_centrality"
                ),
                # Bridge score (connections between different communities)
                pl.col("target_fid").n_unique().alias("bridge_score"),
            ]
        )

        # Calculate community metrics
        community_metrics = follows_clean.group_by("fid").agg(
            [
                # Community embedding strength
                (pl.col("target_fid").n_unique() / pl.len()).alias(
                    "community_embedding"
                ),
                # Network diversity
                pl.col("target_fid").n_unique().alias("network_diversity"),
            ]
        )

        return centrality.join(community_metrics, on="fid", how="left")

    def _get_default_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Return DataFrame with default zero values for all features"""
        return df.with_columns(
            [pl.lit(0).alias(feature) for feature in self.feature_names]
        )
