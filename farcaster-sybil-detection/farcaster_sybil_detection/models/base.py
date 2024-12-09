from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from farcaster_sybil_detection.models.config import ModelConfig
import numpy as np
from dataclasses import dataclass
import joblib
from pathlib import Path


class BaseModel(ABC):
    """Base class for all models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names: List[str] = []

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

    @abstractmethod
    def explain_prediction(self, instance: np.ndarray) -> Dict[str, float]:
        """Get SHAP explanations for predictions"""
        pass

    def save(self) -> None:
        """Save model checkpoint"""
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,  # Save feature names
            "config": self.config,
        }
        if self.config.checkpoint_path:
            joblib.dump(model_data, self.config.checkpoint_path)

    def load(self) -> None:
        """Load model checkpoint"""
        if self.config.checkpoint_path and self.config.checkpoint_path.exists():
            model_data = joblib.load(self.config.checkpoint_path)
            self.model = model_data["model"]
            self.feature_names = model_data.get("feature_names", [])
            self.config = model_data["config"]
