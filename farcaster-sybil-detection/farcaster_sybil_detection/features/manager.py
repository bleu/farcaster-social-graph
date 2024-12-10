import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import polars as pl
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader
from farcaster_sybil_detection.features.extractors.base import FeatureExtractor
from farcaster_sybil_detection.features.extractors.content_engagement_extractor import (
    ContentEngagementExtractor,
)

from farcaster_sybil_detection.features.extractors.network_analysis_extractor import (
    NetworkAnalysisExtractor,
)
from farcaster_sybil_detection.features.extractors.temporal_behavior_extractor import (
    TemporalBehaviorExtractor,
)
from farcaster_sybil_detection.features.extractors.user_identity_extractor import (
    UserIdentityExtractor,
)
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


class FeatureManager(IFeatureProvider):
    INTRINSIC_KEYS = {"fid"}

    """Manages feature extraction and dependencies"""

    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
        self.data_loader = DatasetLoader(config)
        self.extractors = self._initialize_extractors()
        self.feature_sets = self._initialize_feature_sets()

    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _initialize_extractors(self) -> Dict[str, FeatureExtractor]:
        """Initialize all feature extractors"""
        return {
            # new features
            "user_identity": UserIdentityExtractor(self.data_loader),
            "temporal_behavior": TemporalBehaviorExtractor(self.data_loader),
            "network_analysis": NetworkAnalysisExtractor(self.data_loader),
            "content_engagement": ContentEngagementExtractor(self.data_loader),
            # "reputation_meta": ReputationMetaExtractor(
            #     self.data_loader
            # ),
        }

    def _initialize_feature_sets(self) -> Dict[str, FeatureSet]:
        """Initialize feature set configurations"""
        feature_sets = {}
        for name, extractor in self.extractors.items():
            # Get dependencies safely
            deps = extractor.get_dependencies()
            deps = [] if deps is None else deps

            feature_sets[name] = FeatureSet(name=name, version="1.0", dependencies=deps)
            feature_sets[name].checkpoint_path = (
                self.config.checkpoint_dir / f"{name}_features.parquet"
            )
            if feature_sets[name].checkpoint_path.exists():
                feature_sets[name].last_modified = os.path.getmtime(
                    feature_sets[name].checkpoint_path
                )

        return feature_sets

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
            # Process each feature extractor
            for feature_name in self._get_build_order():
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
                extractor = self.extractors[feature_name]
                new_features_lf = extractor.run(feature_matrix, target_fids=base_fids)

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

    def _get_build_order(self) -> List[str]:
        """
        Get correct build order based on feature names and dependencies,
        using get_required_datasets to identify raw data columns.
        """
        self.logger.debug("Determining feature build order...")

        visited = set()
        visiting = set()
        order = []
        missing_feature_deps: Dict[str, List[str]] = {}
        cycles = []

        def get_all_feature_names() -> Dict[str, List[str]]:
            """Get all feature names from each extractor"""
            feature_names = {}
            for extractor_name, extractor in self.extractors.items():
                if hasattr(extractor, "feature_names"):
                    feature_names[extractor_name] = extractor.feature_names
            return feature_names

        def get_raw_columns(extractor_name: str) -> set:
            """Get all raw data columns needed by an extractor"""
            extractor = self.extractors.get(extractor_name)
            if not extractor:
                return set()

            raw_columns = set()
            required_datasets = extractor.get_required_datasets()

            for dataset_info in required_datasets.values():
                if "columns" in dataset_info:
                    raw_columns.update(dataset_info["columns"])

            return raw_columns

        def find_providers_for_feature(feature: str) -> List[str]:
            """Find which extractors provide a given feature name"""
            providers = []
            all_feature_names = get_all_feature_names()
            for extractor_name, features in all_feature_names.items():
                if feature in features:
                    providers.append(extractor_name)
            return providers

        def visit(name: str):
            """Visit a node in dependency graph"""
            self.logger.debug(f"Visiting feature extractor: {name}")

            if name in visiting:
                cycles.append(name)
                return
            if name in visited:
                return

            visiting.add(name)
            extractor = self.extractors.get(name)

            if not extractor:
                self.logger.error(f"Extractor '{name}' is not defined")
                missing_feature_deps[name] = [name]
                visiting.remove(name)
                return

            # Get raw data columns needed by this extractor
            raw_columns = get_raw_columns(name)
            self.logger.debug(f"Raw columns needed by {name}: {raw_columns}")

            # Get feature dependencies
            feature_dependencies = extractor.get_dependencies()
            feature_dependencies = [
                dep
                for dep in feature_dependencies
                if dep not in self.INTRINSIC_KEYS and dep not in raw_columns
            ]

            # Track unmet feature dependencies
            unmet_deps = []

            # Check each feature dependency
            for dep in feature_dependencies:
                # Find which extractors can provide this feature
                providers = find_providers_for_feature(dep)

                if not providers:
                    unmet_deps.append(dep)
                else:
                    # Visit all providers to ensure proper ordering
                    for provider in providers:
                        visit(provider)

            # Record missing feature dependencies
            if unmet_deps:
                missing_feature_deps[name] = unmet_deps

            visiting.remove(name)
            visited.add(name)
            order.append(name)

        # Process all extractors
        for name in self.extractors:
            try:
                visit(name)
            except Exception as e:
                self.logger.error(f"Error processing dependencies for {name}: {e}")
                raise

        # Handle validation results
        if cycles:
            cycle_str = " -> ".join(cycles + [cycles[0]])
            self.logger.error(f"Circular dependency detected: {cycle_str}")
            raise ValueError(f"Circular dependency detected: {cycle_str}")

        if missing_feature_deps:
            error_msgs = []
            for feat, deps in missing_feature_deps.items():
                providers = []
                for dep in deps:
                    dep_providers = find_providers_for_feature(dep)
                    if dep_providers:
                        providers.extend(dep_providers)
                error_msg = f"Feature '{feat}' is missing feature dependencies: {deps}"
                if providers:
                    error_msg += f" (can be provided by: {', '.join(providers)})"
                error_msgs.append(error_msg)
            full_error_msg = "\n".join(error_msgs)
            self.logger.error(
                "Missing Feature Dependencies Detected:\n" + full_error_msg
            )
            raise ValueError("Missing Feature Dependencies:\n" + full_error_msg)

        self.logger.debug(f"Build order determined successfully: {order}")
        return order

    def _needs_rebuild(self, feature_set: FeatureSet) -> bool:
        """Check if feature set needs rebuilding"""
        if feature_set.checkpoint_path is None:
            return True
        return not feature_set.checkpoint_path.exists()

    def _save_checkpoint(
        self, lf: Union[pl.LazyFrame, pl.DataFrame], feature_set: FeatureSet
    ):
        """Save feature checkpoint"""
        try:
            if feature_set.checkpoint_path is None:
                raise ValueError("Checkpoint path is not set.")

            # Collect the LazyFrame into a DataFrame
            if isinstance(lf, pl.LazyFrame):
                df = lf.collect()
            else:
                df = lf
            df.write_parquet(feature_set.checkpoint_path)
            feature_set.last_modified = os.path.getmtime(feature_set.checkpoint_path)
            self.logger.debug(
                f"Checkpoint saved for {feature_set.name} at {feature_set.checkpoint_path}"
            )
        except Exception as e:
            self.logger.error(f"Error saving checkpoint for {feature_set.name}: {e}")
            raise

    def _load_checkpoint(self, feature_set: FeatureSet) -> pl.LazyFrame:
        """Load feature checkpoint if it exists"""
        try:
            if feature_set.checkpoint_path and feature_set.checkpoint_path.exists():
                self.logger.debug(
                    f"Loading checkpoint for {feature_set.name} from {feature_set.checkpoint_path}"
                )
                return pl.scan_parquet(
                    feature_set.checkpoint_path,
                    parallel="prefiltered",
                    low_memory=True,
                )
            else:
                self.logger.warning(f"No checkpoint found for {feature_set.name}")
                return pl.LazyFrame()
        except Exception as e:
            self.logger.error(f"Error loading checkpoint for {feature_set.name}: {e}")
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
        for extractor in self.extractors.values():
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
