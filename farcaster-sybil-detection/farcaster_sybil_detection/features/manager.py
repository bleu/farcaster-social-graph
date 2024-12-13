import os
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
                missing_fids = base_fids

                # Try to get cached result first
                cached_features = self._get_cached_filtered_dataset(feature_name,base_fids)

                if cached_features is not None:
                    self.logger.debug(f"Using cached features for {feature_name}")
                    # Find missing FIDs
                    cached_fids = cached_features.select('fid').collect()['fid'].to_list()
                    missing_fids = list(set(base_fids) - set(cached_fids))
                    
                if len(missing_fids) == 0:  # If no missing FIDs
                    feature_matrix = self._safe_join_features(
                        feature_matrix, cached_features, feature_name
                    )
                    continue
                # Compute features only for missing FIDs
                missing_features_lf = extractor_class(data_loader=self.data_loader).run(
                    feature_matrix.filter(pl.col('fid').is_in(missing_fids)), 
                    target_fids=missing_fids
                )
                
                if missing_features_lf is not None:
                    missing_features_df = missing_features_lf.collect()

                    # Cache the new data
                    cache_key = self._get_filtered_dataset_cache_key(
                        feature_name, missing_fids
                    )
                    self._cache_filtered_dataset(missing_features_df, cache_key)

                    # Combine cached and new features
                    combined_features = pl.concat([
                        cached_features.collect(),
                        missing_features_df
                    ]) if cached_features is not None else missing_features_df
                    
                    
                    # Join with feature matrix
                    feature_matrix = self._safe_join_features(
                        feature_matrix,
                        combined_features.lazy(),
                        feature_name
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
    
    def _cache_filtered_dataset(self,df,cache_key:str):
        # Cache file with hash
        cache_path = self.config.checkpoint_dir / f"{cache_key}.parquet"
        df.write_parquet(cache_path)
        
        hash = self._read_hash(cache_key)
        hashes_path = self.config.checkpoint_dir / "hashes.parquet"

        # If hashes file wasn't created yet
        if not os.path.exists(hashes_path):
            hashes = pl.DataFrame({"fid":df["fid"],"hash":[hash]*len(df)})
            hashes.write_parquet(hashes_path)
            return
        
        hashes_df = pl.read_parquet(hashes_path)
        if hash in hashes_df["hash"].to_list():
            return
        hashes = pl.DataFrame({"fid":df["fid"],"hash":[hash]*len(df)})
        new_hashes = pl.concat([hashes_df,hashes])
        new_hashes.write_parquet(hashes_path)

    def _get_cached_filtered_dataset(self, feature_name: str, fids: List[int]) -> Optional[pl.LazyFrame]:
        """Try to get cached filtered dataset"""

        # 1. List current hashes for that feature name
        files = [file for file in os.listdir(self.config.checkpoint_dir) if feature_name in file]
        hashes = [self._read_hash(file) for file in files]
        if len(hashes) == 0:
            return

        # 2. Check what fids are on the hashes
        cached_fids_map = {}
        cached_fids = []
        for file in files:
            fids_to_cache = (
                pl.scan_parquet(self.config.checkpoint_dir / file)
                    .select(["fid"])
                    .collect()
                    # Fids we want to get, excluding the ones we already got
                    .filter(pl.col("fid").is_in(list(set(fids) - set(cached_fids))))
            )["fid"].to_list()
            cached_fids += fids_to_cache
            cached_fids_map[file] = fids_to_cache

        # 3. Load the cached dfs and concat them
        cached_dfs = []
        for file, cached_fids in cached_fids_map.items():
            cached_dfs.append(
                pl.scan_parquet(self.config.checkpoint_dir / file)
                .filter(pl.col("fid").is_in(cached_fids))
            )
        final_lf = pl.concat(cached_dfs)
        return final_lf
    
    def _read_hash(self,cache_key:str):
        return cache_key.split("_")[-1].split(".")[0]


