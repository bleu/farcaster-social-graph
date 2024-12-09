from typing import List, Dict
import polars as pl
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.config import FeatureConfig
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader
import re


class CastBehaviorExtractor(FeatureExtractor):
    """Extract cast behavior features with content analysis"""

    def __init__(self, config: FeatureConfig, data_loader: DatasetLoader):
        super().__init__(config, data_loader)
        self.feature_names = [
            "cast_count",
            "reply_count",
            "mentions_count",
            "avg_cast_length",
            "link_count",
            "media_count",
            "has_link_and_media",
            "casts_with_links",
            "casts_with_media",
            "link_ratio",
            "media_ratio",
            "airdrop_mention_ratio",
            "spam_keyword_ratio",
            "avg_at_symbol_ratio",
            "avg_dollar_symbol_ratio",
            "avg_link_ratio",
            "casts_with_both",
            "multimedia_ratio",
            "template_usage_ratio",
            "cta_heavy_ratio",
            "urgency_ratio",
            "emoji_spam_ratio",
            "price_mention_ratio",
            "symbol_spam_ratio",
            "airdrop_term_ratio",
        ]

    def get_dependencies(self) -> List[str]:
        return ["fid"]

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
            }
        }

    def _analyze_spam_patterns(self, text: str) -> Dict[str, int]:
        if not text:
            return {"airdrop": 0, "money": 0, "rewards": 0, "claim": 0, "moxie": 0}

        text = text.lower()
        spam_keywords = ["airdrop", "money", "rewards", "claim", "moxie", "nft", "drop"]
        return {word: text.count(word) for word in spam_keywords}

    def _get_symbol_ratios(self, text: str) -> Dict[str, float]:
        if not text:
            return {"at_symbol_ratio": 0, "dollar_symbol_ratio": 0, "link_ratio": 0}

        total_length = len(text)
        return {
            "at_symbol_ratio": (
                text.count("@") / total_length if total_length > 0 else 0
            ),
            "dollar_symbol_ratio": (
                text.count("$") / total_length if total_length > 0 else 0
            ),
            "link_ratio": (
                len(re.findall(r"http[s]?://", text)) / total_length
                if total_length > 0
                else 0
            ),
        }

    def _analyze_content_patterns(self, text: str) -> Dict[str, int]:
        if not text:
            return {
                "template_structure": 0,
                "multiple_cta": 0,
                "urgency_terms": 0,
                "excessive_emojis": 0,
                "price_mentions": 0,
                "excessive_symbols": 0,
                "airdrop_terms": 0,
            }

        text = text.lower()
        return {
            "template_structure": int(
                bool(re.findall(r"\[.*?\]|\{.*?\}|\<.*?\>", text))
            ),
            "multiple_cta": int(
                len(re.findall(r"click|join|follow|claim|grab", text)) > 2
            ),
            "urgency_terms": int(
                bool(re.findall(r"hurry|limited|fast|quick|soon|ending", text))
            ),
            "excessive_emojis": int(
                len(re.findall(r"[\U0001F300-\U0001F9FF]", text)) > 5
            ),
            "price_mentions": int(bool(re.findall(r"\$\d+|\d+\$", text))),
            "excessive_symbols": int(bool(re.findall(r"[_.\-]{2,}", text))),
            "airdrop_terms": int(
                any(
                    term in text.lower()
                    for term in ["airdrop", "farm", "degen", "wojak"]
                )
            ),
        }

    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        try:
            self.logger.info("Extracting cast behavior features...")
            casts = loaded_datasets.get("casts")

            if casts is None:
                self.logger.warning("No casts data available")
                return df

            # Helper functions for counting links and media
            def count_links(text):
                if not text:
                    return 0
                url_patterns = ["http://", "https://", "www."]
                return sum(1 for pattern in url_patterns if pattern in text.lower())

            def count_media(embeds):
                if not embeds or embeds == "[]":
                    return 0
                try:
                    return embeds.lower().count("image")
                except:
                    return 0

            # Calculate features using LazyFrame operations
            cast_features = (
                casts.filter(pl.col("deleted_at").is_null())
                .with_columns(
                    [
                        pl.when(pl.col("text").is_not_null())
                        .then(pl.col("text").map_elements(len, return_dtype=pl.Int64))
                        .otherwise(0)
                        .alias("cast_length"),
                        pl.col("parent_hash")
                        .is_not_null()
                        .cast(pl.Int64)
                        .alias("is_reply"),
                        (
                            pl.col("mentions").is_not_null()
                            & (pl.col("mentions") != "")
                            & (pl.col("mentions") != "[]")
                        )
                        .cast(pl.Int64)
                        .alias("has_mentions"),
                        pl.when(pl.col("text").is_not_null())
                        .then(
                            pl.col("text").map_elements(
                                count_links, return_dtype=pl.Int64
                            )
                        )
                        .otherwise(0)
                        .alias("link_count"),
                        pl.when(pl.col("embeds").is_not_null())
                        .then(
                            pl.col("embeds").map_elements(
                                count_media, return_dtype=pl.Int64
                            )
                        )
                        .otherwise(0)
                        .alias("media_count"),
                        (
                            pl.when(pl.col("text").is_not_null())
                            .then(
                                pl.col("text").map_elements(
                                    count_links, return_dtype=pl.Int64
                                )
                            )
                            .otherwise(0)
                            > 0
                            & pl.when(pl.col("embeds").is_not_null())
                            .then(
                                pl.col("embeds").map_elements(
                                    count_media, return_dtype=pl.Int64
                                )
                            )
                            .otherwise(0)
                            > 0
                        )
                        .cast(pl.Int64)
                        .alias("has_link_and_media"),
                        pl.col("text")
                        .map_elements(self._analyze_spam_patterns, return_dtype=pl.Utf8)
                        .alias("spam_counts"),
                        pl.col("text")
                        .map_elements(self._get_symbol_ratios, return_dtype=pl.Utf8)
                        .alias("symbol_ratios"),
                        pl.col("text")
                        .map_elements(
                            self._analyze_content_patterns, return_dtype=pl.Utf8
                        )
                        .alias("content_patterns"),
                    ]
                )
                .group_by("fid")
                .agg(
                    [
                        pl.len().alias("cast_count"),
                        pl.col("cast_length").mean().alias("avg_cast_length"),
                        pl.col("is_reply").sum().alias("reply_count"),
                        pl.col("has_mentions").sum().alias("mentions_count"),
                        pl.col("link_count").sum().alias("total_links"),
                        (pl.col("link_count") > 0).sum().alias("casts_with_links"),
                        (pl.col("link_count") / pl.len()).alias("link_ratio"),
                        pl.col("media_count").sum().alias("total_media"),
                        (pl.col("media_count") > 0).sum().alias("casts_with_media"),
                        (pl.col("media_count") / pl.len()).alias("media_ratio"),
                        (
                            pl.col("spam_counts")
                            .map_elements(lambda x: x["airdrop"])
                            .sum()
                            / pl.len()
                        ).alias("airdrop_mention_ratio"),
                        (
                            pl.col("spam_counts")
                            .map_elements(lambda x: sum(x.values()))
                            .sum()
                            / pl.len()
                        ).alias("spam_keyword_ratio"),
                        pl.col("symbol_ratios")
                        .map_elements(lambda x: x["at_symbol_ratio"])
                        .mean()
                        .alias("avg_at_symbol_ratio"),
                        pl.col("symbol_ratios")
                        .map_elements(lambda x: x["dollar_symbol_ratio"])
                        .mean()
                        .alias("avg_dollar_symbol_ratio"),
                        pl.col("symbol_ratios")
                        .map_elements(lambda x: x["link_ratio"])
                        .mean()
                        .alias("avg_link_ratio"),
                        pl.col("has_link_and_media").sum().alias("casts_with_both"),
                        (pl.col("has_link_and_media").sum() / pl.len()).alias(
                            "multimedia_ratio"
                        ),
                        (
                            pl.col("content_patterns")
                            .map_elements(lambda x: x["template_structure"])
                            .sum()
                            / pl.len()
                        ).alias("template_usage_ratio"),
                        (
                            pl.col("content_patterns")
                            .map_elements(lambda x: x["multiple_cta"])
                            .sum()
                            / pl.len()
                        ).alias("cta_heavy_ratio"),
                        (
                            pl.col("content_patterns")
                            .map_elements(lambda x: x["urgency_terms"])
                            .sum()
                            / pl.len()
                        ).alias("urgency_ratio"),
                        (
                            pl.col("content_patterns")
                            .map_elements(lambda x: x["excessive_emojis"])
                            .sum()
                            / pl.len()
                        ).alias("emoji_spam_ratio"),
                        (
                            pl.col("content_patterns")
                            .map_elements(lambda x: x["price_mentions"])
                            .sum()
                            / pl.len()
                        ).alias("price_mention_ratio"),
                        (
                            pl.col("content_patterns")
                            .map_elements(lambda x: x["excessive_symbols"])
                            .sum()
                            / pl.len()
                        ).alias("symbol_spam_ratio"),
                        (
                            pl.col("content_patterns")
                            .map_elements(lambda x: x["airdrop_terms"])
                            .sum()
                            / pl.len()
                        ).alias("airdrop_term_ratio"),
                    ]
                )
            )

            return cast_features.select(["fid"] + self.feature_names)

        except Exception as e:
            self.logger.error(f"Error extracting cast behavior features: {e}")
            raise
