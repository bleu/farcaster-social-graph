# farcaster_sybil_detection/features/extractors/base.py

from abc import abstractmethod
from typing import List, Dict, Optional
from farcaster_sybil_detection.utils.with_logging import LoggedABC, add_logging
import polars as pl
from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class BaseFeatureExtractor(LoggedABC):
    """Base feature extractor interface"""

    @abstractmethod
    def get_required_datasets(self) -> Dict[str, Dict]:
        pass


@add_logging
class FeatureExtractor(BaseFeatureExtractor):
    """Abstract base class for all feature extractors. Implements Template Method pattern."""

    def __init__(self, data_loader: DatasetLoader):
        self.data_loader = data_loader
        self.feature_names: List[str] = []

    def run(
        self, input_lf: Optional[pl.LazyFrame], target_fids: Optional[List[int]] = None
    ) -> Optional[pl.LazyFrame]:
        """
        Template method defining the workflow for feature extraction.
        Maintains lazy evaluation throughout.
        """
        try:
            self.logger.debug(f"Starting feature extraction: {self.__class__.__name__}")

            # Create initial lazy frame if needed
            if input_lf is None:
                if target_fids is None:
                    self.logger.warning(
                        "Both input LazyFrame and target_fids are None. Skipping feature extraction."
                    )
                    return None
                self.logger.debug(
                    f"Creating initial LazyFrame with {len(target_fids)} target FIDs"
                )
                input_lf = pl.DataFrame({"fid": target_fids}).lazy()

            # Load required datasets lazily
            required_datasets = self.get_required_datasets()
            loaded_datasets = self.load_required_datasets(
                required_datasets, target_fids
            )

            # Check for empty datasets without collecting
            empty_check = pl.concat(
                [
                    ds.select(pl.count()).filter(pl.col("count") > 0)
                    for ds in loaded_datasets.values()
                ]
            ).collect(engine="gpu")

            if len(empty_check) == 0:
                self.logger.warning("No data available in required datasets")
                return self._get_default_features(input_lf)

            # Extract features while keeping operations lazy
            features_lf = self.extract_features(input_lf, loaded_datasets)
            if features_lf is None:
                self.logger.warning("Feature extraction returned None")
                return self._get_default_features(input_lf)

            # Post-process while maintaining lazy evaluation
            features_lf = self.post_process(features_lf)
            self.logger.debug(
                f"Completed feature extraction: {self.__class__.__name__}"
            )
            return features_lf

        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}: {e}")
            raise

    def _get_default_features(self, input_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Return LazyFrame with default zero values for all features"""
        # Start with input FIDs
        result = input_lf.select(["fid"])

        # Add default zero values for all features
        for feature in self.feature_names:
            result = result.with_columns([pl.lit(0.0).cast(pl.Float64).alias(feature)])

        return result

    def validate_input(self, lf: pl.LazyFrame):
        """
        Validates the input LazyFrame. Ensures required columns are present.
        """
        required_columns = self.get_dependencies()
        available_columns = lf.columns
        missing = [col for col in required_columns if col not in available_columns]
        if missing:
            error_msg = f"Missing required columns: {missing}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.logger.debug("Input validation passed.")

    @abstractmethod
    def get_required_datasets(self) -> Dict[str, Dict]:
        """
        Abstract method for extractors to specify required datasets.
        Returns a dictionary where keys are dataset names and values are dicts with parameters.
        Example:
            {
                'reactions': {'columns': ['fid', 'reaction_type', ...], 'source': 'farcaster'},
                'another_dataset': {'columns': [...], 'source': 'nindexer'},
            }
        """
        pass

    def load_required_datasets(
        self, required_datasets: Dict[str, Dict], target_fids: Optional[List[int]]
    ) -> Dict[str, pl.LazyFrame]:
        """Load required datasets lazily"""
        loaded = {}
        for name, params in required_datasets.items():
            columns = params.get("columns")
            source = params.get("source", "farcaster")

            self.logger.debug(f"Loading dataset '{name}' from source '{source}'")
            if columns:
                self.logger.debug(f"Required columns: {columns}")
            if target_fids:
                self.logger.debug(f"Filtering for {len(target_fids)} FIDs")

            # Load dataset lazily
            dataset_lf = self.data_loader.load_lazy_dataset(
                name, source=source, columns=columns, fids=target_fids
            )

            loaded[name] = dataset_lf

        return loaded

    @abstractmethod
    def extract_features(
        self, df: pl.LazyFrame, loaded_datasets: Dict[str, pl.LazyFrame]
    ) -> pl.LazyFrame:
        """
        Abstract method for extracting specific features.
        Must be implemented by subclasses.
        Receives the main LazyFrame and a dictionary of loaded datasets.
        """
        pass

    def validate_extracted_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Validate features while maintaining lazy evaluation"""
        numeric_cols = [
            col
            for col, dtype in lf.schema.items()
            if dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]
        ]

        # Apply validations lazily
        result = lf
        for col in numeric_cols:
            result = result.filter(pl.col(col) >= 0)

        return result

    def post_process(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Common post-processing steps:
        - Ensure 'fid' is of type Int64
        - Remove duplicate 'fid' entries
        - Fill missing values with zeros
        - Convert non-numeric columns to numeric where applicable
        """
        self.logger.debug("Starting post-processing...")
        lf = self._ensure_fid_type(lf)
        lf = self._ensure_unique_fid(lf)
        lf = self._ensure_no_nulls(lf)
        self.logger.debug(
            "Post-processing completed: cast 'fid', ensure uniqueness, fill nulls, and convert to numeric."
        )
        return lf

    def _ensure_fid_type(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Ensure that 'fid' is of type Int64"""
        return lf.with_columns(pl.col("fid").cast(pl.Int64))

    def _ensure_unique_fid(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Ensure that 'fid' column is unique"""
        return lf.unique(subset=["fid"])

    def _ensure_no_nulls(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Fill missing values with zeros"""
        return lf.fill_null(0)
