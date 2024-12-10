from abc import abstractmethod
from typing import List, Optional

from farcaster_sybil_detection.utils.with_logging import LoggedABC, add_logging
import polars as pl

from farcaster_sybil_detection.config.defaults import Config


@add_logging
class BaseDataLoader(LoggedABC):
    """Abstract base class for data loading"""

    def __init__(self, config: Config):
        self.config = config
        self._cached_data = {}

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
