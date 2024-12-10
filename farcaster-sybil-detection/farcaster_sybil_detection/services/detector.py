from farcaster_sybil_detection.utils.with_logging import LoggedABC, add_logging
import polars as pl
from farcaster_sybil_detection.config.defaults import Config
from farcaster_sybil_detection.models.ensemble import OptimizedEnsemble
from farcaster_sybil_detection.services.predictor import Predictor
from farcaster_sybil_detection.services.trainer import Trainer
from typing import Any, Dict, Union
from farcaster_sybil_detection.features.registry import FeatureRegistry
from farcaster_sybil_detection.features.manager import FeatureManager


@add_logging
class DetectorService(LoggedABC):
    """Service layer for Sybil detection"""

    def __init__(self, config: Config, registry: FeatureRegistry):
        self.config = config

        self._setup_components(registry)

    def _setup_components(self, registry: FeatureRegistry):
        """Initialize system components"""
        self.feature_manager = FeatureManager(self.config, registry)

        self.model = OptimizedEnsemble(
            self.config.model_dir / "model.pkl",
        )

        # Check if model checkpoint exists
        if (self.config.model_dir / "model.pkl") and (
            self.config.model_dir / "model.pkl"
        ).exists():
            self.logger.debug("Loading existing model from checkpoint.")
            self.model.load()
        else:
            self.logger.debug(
                "No existing model found. Model will be trained when `train` is called."
            )

        self.predictor = Predictor(self.config, self.model, self.feature_manager)
        self.trainer = Trainer(self.config, self.model, self.feature_manager)

    def train(self, labels_df: pl.DataFrame) -> Dict[str, float]:
        """Train the Sybil detection model"""
        self.logger.debug("Starting training process...")
        metrics = self.trainer.train(labels_df)
        self.logger.debug("Training completed.")
        return metrics

    def predict(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Make a prediction for a given FID or fname"""
        self.logger.debug(f"Making prediction for identifier: {identifier}")
        result = self.predictor.predict(identifier)
        self.logger.debug("Prediction completed.")
        return result
