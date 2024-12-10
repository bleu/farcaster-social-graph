import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import polars as pl

from farcaster_sybil_detection.config.defaults import Config


class BaseDataLoader(ABC):
    """Abstract base class for data loading"""

    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
        self._cached_data = {}

    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def load_dataset(
        self, name: str, columns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Load a dataset with optional column selection"""
        pass

    @abstractmethod
    def get_columns(self, name: str) -> List[str]:
        """Get available columns for a dataset"""
        pass

    def clear_cache(self):
        """Clear the data cache"""
        self._cached_data = {}
        self.logger.debug("Cache cleared")


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction"""

    @abstractmethod
    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract features from input DataFrame"""
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get list of dependent features needed for extraction"""
        pass
