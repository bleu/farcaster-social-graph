import polars as pl
from farcaster_sybil_detection.config.defaults import Config
from farcaster_sybil_detection.models.config import ModelConfig
from farcaster_sybil_detection.models.ensemble import OptimizedEnsemble
from farcaster_sybil_detection.services.predictor import Predictor
from farcaster_sybil_detection.services.trainer import Trainer
from farcaster_sybil_detection.features.interface import IFeatureProvider
from typing import Any, Dict, Optional, Union
import logging


class DetectorService:
    """Service layer for Sybil detection"""

    def __init__(
        self, config: Config, feature_manager: Optional[IFeatureProvider] = None
    ):
        self.config = config
        self._setup_logging()
        self._setup_components(feature_manager)

    def _setup_logging(self):
        """Setup logging for DetectorService"""
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

    def _setup_components(self, feature_manager: Optional[IFeatureProvider]):
        """Initialize system components"""
        if feature_manager is None:
            self.logger.error("FeatureProvider cannot be None")
            raise ValueError("FeatureProvider must be provided")
        self.feature_manager = feature_manager
        model_config = ModelConfig(
            name="sybil_detector",
            params={},
            checkpoint_path=self.config.model_dir / "model.pkl",
            confidence_thresholds=self.config.confidence_thresholds,
        )
        self.model = OptimizedEnsemble(model_config)

        # Check if model checkpoint exists
        if (
            self.model.config.checkpoint_path
            and self.model.config.checkpoint_path.exists()
        ):
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
