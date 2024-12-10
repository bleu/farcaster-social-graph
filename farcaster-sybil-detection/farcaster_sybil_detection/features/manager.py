from pathlib import Path
from typing import Dict, List, Optional

from farcaster_sybil_detection.features.registry import FeatureRegistry
from farcaster_sybil_detection.utils.with_logging import add_logging
import polars as pl
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor

from farcaster_sybil_detection.features.interface import IFeatureProvider

from ..config.defaults import Config


class FeatureSet:
    """Track feature dependencies and versioning"""

    def __init__(self, name: str, version: str, dependencies: List[str] = None):
        self.name = name
        self.version = version  # Version of feature calculation logic
        self.dependencies = dependencies or []
        self.checkpoint_path: Optional[Path] = None
        self.last_modified: Optional[float] = None


@add_logging
class FeatureManager(IFeatureProvider):
    INTRINSIC_KEYS = {"fid"}

    """Manages feature extraction and dependencies"""

    def __init__(self, config: Config, registry: FeatureRegistry):
        self.config = config
        self.registry = registry
        self.data_loader = DatasetLoader(config)

    def get_enabled_extractors(self) -> Dict[str, FeatureExtractor]:
        """Initialize all feature extractors"""

        extractors = self.registry.get_enabled_names()
        return {name: self.registry.get_extractor(name) for name in extractors}

    def _is_numeric_dtype(self, dtype) -> bool:
        """Helper to check if a dtype is numeric"""
        dtype_str = str(dtype).lower()
        return any(t in dtype_str for t in ["int", "float", "decimal"])

    def _safe_join_features(
        self, df: pl.LazyFrame, new_features: pl.LazyFrame, feature_name: str
    ) -> pl.LazyFrame:
        """Optimized join operation"""
        try:
            # Ensure minimal columns are being joined
            join_cols = ["fid"] + [
                col
                for col in new_features.columns
                if col not in df.columns and col != "fid"
            ]

            # Use join_asof for time series data if needed
            result = df.join(
                new_features.select(join_cols),
                on="fid",
                how="left",
            )
            return result

        except Exception as e:
            self.logger.error(f"Error joining {feature_name}: {e}")
            raise

    def _log_memory(self, message: str):
        """Log memory usage with message"""
        import psutil

        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024
        self.logger.debug(f"{message} - Memory usage: {memory:.2f} MB")

    def _process_features_incrementally(
        self,
        feature_matrix: pl.LazyFrame,
        new_features_lf: pl.LazyFrame,
        feature_name: str,
    ) -> pl.DataFrame:
        """Process features in chunks with comprehensive type handling"""
        self.logger.debug(f"Starting incremental processing for {feature_name}")

        result = None
        row_count = 0

        try:
            for chunk in new_features_lf.collect(streaming=True):
                chunk_size = len(chunk)
                self.logger.debug(f"Processing chunk of size {chunk_size}")

                # Cast each column individually to handle errors gracefully
                safe_chunk = pl.DataFrame(chunk)
                safe_chunk = safe_chunk.with_columns(
                    pl.col("fid").cast(pl.Int64).fill_null(0).alias("fid")
                )

                result = safe_chunk

                # Concatenate
                result = pl.concat([result, safe_chunk], how="diagonal_relaxed").unique(
                    subset=["fid"]
                )

                row_count += chunk_size
                self.logger.debug(f"Processed {row_count} total rows")

                # Clean up
                del chunk
                del safe_chunk
                import gc

                gc.collect()

            if result is not None:
                # Final cleanup to ensure all types are correct
                final_result = result

                self.logger.debug(f"Final result schema: {final_result.schema}")
                return final_result

            return None

        except Exception as e:
            self.logger.error(f"Error in incremental processing: {str(e)}")
            if result is not None:
                self.logger.error(f"Result schema: {result.schema}")
            raise

    def build_feature_matrix(
        self, target_fids: Optional[List[int]] = None
    ) -> pl.DataFrame:
        try:
            self._log_memory("Starting feature matrix build")

            # Initialize with base FIDs
            if target_fids is not None:
                feature_matrix = (
                    pl.DataFrame({"fid": target_fids})
                    .with_columns({"fid": pl.col("fid").cast(pl.Int64)})
                    .lazy()
                )
                base_fids = target_fids
            else:
                feature_matrix = (
                    self.data_loader.load_lazy_dataset(
                        "profile_with_addresses",
                        columns=["fid"],
                        source="farcaster",
                    )
                    .select(["fid"])
                    .with_columns({"fid": pl.col("fid").cast(pl.Int64)})
                    .limit(self.config.sample_size)
                    .unique()
                )
                base_fids = feature_matrix.select("fid").collect()["fid"].to_list()

            self.logger.debug(f"Base FIDs: {len(base_fids)}")
            self.logger.debug(
                f"Feature matrix schema: {feature_matrix.schema} ({len(feature_matrix.columns)} columns)"
            )
            self.logger.debug(f"Feature matrix size: {(feature_matrix.count())}")

            for feature_name, extractor_class in self.get_enabled_extractors().items():
                self._log_memory(f"Starting {feature_name}")

                # Try to get cached result first
                cache_key = self._get_filtered_dataset_cache_key(
                    feature_name, base_fids
                )
                cached_features = self._get_cached_filtered_dataset(cache_key)

                if cached_features is not None:
                    self.logger.debug(f"Using cached features for {feature_name}")
                    feature_matrix = self._safe_join_features(
                        feature_matrix, cached_features, feature_name
                    )
                    continue

                # Extract features
                new_features_lf = extractor_class(data_loader=self.data_loader).run(
                    feature_matrix, target_fids=base_fids
                )

                if new_features_lf is not None:
                    new_features_df = new_features_lf.collect()

                    # Cache the result
                    self._cache_filtered_dataset(new_features_df, cache_key)

                    # Join with feature matrix
                    feature_matrix = self._safe_join_features(
                        feature_matrix,
                        pl.DataFrame(new_features_df).lazy(),
                        feature_name,
                    )

                self._log_memory(f"Completed {feature_name}")

            # Final collection
            self.logger.debug("Collecting final feature matrix")
            result = feature_matrix.collect()
            self._log_memory("Feature matrix build completed")

            return result

        except Exception as e:
            self.logger.error(f"Error building feature matrix: {e}")
            raise

    def get_features_for_fid(self, fid: int) -> Optional[pl.DataFrame]:
        """Get features for a specific FID"""
        matrix = self.build_feature_matrix(target_fids=[fid])
        features = matrix.filter(pl.col("fid") == fid)
        if len(features) == 0:
            return None
        return features

    def get_features_for_fids(self, fids: List[int]) -> pl.DataFrame:
        """Get features for multiple FIDs"""
        matrix = self.build_feature_matrix(target_fids=fids)
        return matrix.filter(pl.col("fid").is_in(fids))

    def get_available_features(self) -> List[str]:
        """Get list of available features"""
        features = set()
        for extractor in self.get_enabled_extractors().values():
            features.update(extractor.feature_names)
        return sorted(list(features))

    def _get_filtered_dataset_cache_key(
        self, dataset_name: str, fids: List[int]
    ) -> str:
        """Create cache key for filtered dataset"""
        fids_hash = hash(tuple(sorted(fids)))
        return f"{dataset_name}_filtered_{fids_hash}"

    def _cache_filtered_dataset(self, df: pl.DataFrame, cache_key: str):
        """Cache filtered dataset to parquet"""
        cache_path = self.config.checkpoint_dir / f"{cache_key}.parquet"
        df.write_parquet(cache_path)
        return cache_path

    def _get_cached_filtered_dataset(self, cache_key: str) -> Optional[pl.LazyFrame]:
        """Try to get cached filtered dataset"""
        cache_path = self.config.checkpoint_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            return pl.scan_parquet(cache_path)
        return None
