from abc import abstractmethod
from typing import List
from farcaster_sybil_detection.utils.with_logging import LoggedABC, add_logging
import numpy as np
import joblib


class IModel(LoggedABC):
    """Base model interface"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save model to disk"""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model from disk"""
        pass


@add_logging
class BaseModel(IModel):
    """Base class for all models"""

    def __init__(self, checkpoint_path: str):
        self.model = None
        self.feature_names: List[str] = []
        self.checkpoint_path = checkpoint_path

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the positive class."""
        pass

    @abstractmethod
    def get_prediction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get confidence score for prediction"""
        pass

    def save(self) -> None:
        """Save model checkpoint"""
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,  # Save feature names
        }
        if self.checkpoint_path:
            joblib.dump(model_data, self.checkpoint_path)

    def load(self) -> None:
        """Load model checkpoint"""
        if self.checkpoint_path and self.checkpoint_path.exists():
            model_data = joblib.load(self.checkpoint_path)
            self.model = model_data["model"]
            self.feature_names = model_data.get("feature_names", [])
