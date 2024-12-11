from typing import Optional, List
from farcaster_sybil_detection.utils.with_logging import LoggedABC
import polars as pl
from abc import abstractmethod

from farcaster_sybil_detection.data.dataset_loader import DatasetLoader


class IFeatureProvider(LoggedABC):
    """Interface for feature access"""

    def __init__(self):
        self.data_loader: DatasetLoader

    @abstractmethod
    def get_features_for_fid(self, fid: int) -> Optional[pl.DataFrame]:
        """Get features for a specific FID"""
        pass

    @abstractmethod
    def get_features_for_fids(self, fids: List[int]) -> pl.DataFrame:
        """Get features for multiple FIDs"""
        pass

    @abstractmethod
    def get_available_features(self) -> List[str]:
        """Get list of available features"""
        pass

    @abstractmethod
    def build_feature_matrix(
        self, target_fids: Optional[List[int]] = None
    ) -> pl.DataFrame:
        """Build the complete feature matrix"""
        pass
