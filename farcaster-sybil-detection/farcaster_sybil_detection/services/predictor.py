import numpy as np
import polars as pl
from farcaster_sybil_detection.features.manager import FeatureManager
from farcaster_sybil_detection.models.base import BaseModel
from farcaster_sybil_detection.config.defaults import Config
from typing import Any, Dict, Optional, Union, Tuple
import logging


class Predictor:
    """Handles predictions with separate feature matrix and ID resolution"""

    def __init__(
        self, config: Config, model: BaseModel, feature_manager: FeatureManager
    ):
        self.config = config
        self.model = model
        self.feature_manager = feature_manager
        self._setup_logging()
        # Cache for FID-fname mapping
        self._id_mapping: Optional[pl.DataFrame] = None

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

    def _load_id_mapping(self) -> pl.DataFrame:
        """Load or retrieve cached ID mapping"""
        if self._id_mapping is None:
            self.logger.debug("Loading ID mapping from profile data...")
            self._id_mapping = self.feature_manager.data_loader.load_dataset(
                "profile_with_addresses", columns=["fid", "fname"]
            )
        return self._id_mapping

    def _resolve_identifier(
        self, identifier: Union[int, str]
    ) -> Tuple[Optional[int], Optional[str]]:
        """Resolve identifier to both FID and fname"""
        id_mapping = self._load_id_mapping()

        if isinstance(identifier, int):
            matching_row = id_mapping.filter(pl.col("fid") == identifier)
            fid = identifier
            fname = matching_row["fname"][0] if not matching_row.is_empty() else None
        elif isinstance(identifier, str):
            matching_row = id_mapping.filter(pl.col("fname") == identifier)
            if matching_row.is_empty():
                return None, identifier
            fid = matching_row["fid"][0]
            fname = identifier
        else:
            raise ValueError("Identifier must be FID (int) or fname (str)")

        return fid, fname

    def predict(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Make prediction for a single identifier"""
        try:
            self.logger.debug(f"Predicting for identifier: {identifier}")

            # First, resolve the identifier
            fid, fname = self._resolve_identifier(identifier)

            if fid is None:
                return {
                    "error": f"No user found with identifier: {identifier}",
                    "status": "not_found",
                }

            # Get features from feature matrix
            features = self.feature_manager.get_features_for_fid(fid)
            if features is None:
                self.logger.warning(
                    f"No features found for FID: {fid}. Building features now."
                )
                # Build features for this FID
                self.feature_manager.build_feature_matrix(target_fids=[fid])
                features = self.feature_manager.get_features_for_fid(fid)
                if features is None:
                    return {
                        "error": f"Failed to build features for FID: {fid}",
                        "status": "feature_build_failed",
                    }
                self.logger.debug(f"Features built successfully for FID: {fid}")

            # Generate prediction
            self.logger.debug(f"Generating prediction using {features}")
            prediction_result = self._generate_prediction(features)

            # Add identifier information
            prediction_result.update(
                {"fid": fid, "fname": fname or "unknown", "status": "success"}
            )

            return prediction_result

        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

    def _generate_prediction(self, features: pl.DataFrame) -> Dict[str, Any]:
        try:
            # Get current feature columns (excluding 'fid')
            feature_cols = [col for col in features.columns if col != "fid"]

            self.logger.debug(f"Generating prediction using shape: {features.shape}")
            print(features)  # Log full feature matrix for debugging

            # Get expected features from model
            model_features = self.model.feature_names
            self.logger.debug(f"Model features: {model_features}")

            if not model_features:
                raise ValueError(
                    "Model has no defined feature set. Please ensure model was properly trained and saved with feature names."
                )

            # Check feature alignment
            current_features = set(feature_cols)
            expected_features = set(model_features)

            missing_features = expected_features - current_features
            extra_features = current_features - expected_features

            if missing_features or extra_features:
                self.logger.warning(
                    f"Feature mismatch - Missing: {missing_features}, Extra: {extra_features}"
                )

            # Use only the features the model expects, in the correct order
            self.logger.debug(f"Predicting for features: {model_features}")
            X = features.select(model_features).to_numpy()

            # Get predictions
            y_prob = self.model.predict_proba(X)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            # Get confidence and ensure it's a float
            confidence = self.model.get_prediction_confidence(X)
            if isinstance(confidence, np.ndarray):
                confidence = float(confidence[0])
            else:
                confidence = float(confidence)

            # Ensure confidence is not nan
            if np.isnan(confidence):
                confidence = 1.0 if y_prob[0] > 0.9 or y_prob[0] < 0.1 else 0.5

            return {
                "prediction": int(y_pred[0]),
                "probability": float(y_prob[0]),
                "confidence": confidence,
                "prediction_label": "bot" if y_pred[0] == 1 else "human",
                "features_used": feature_cols,
            }
        except Exception as e:
            self.logger.error(f"Error generating prediction: {str(e)}")
            raise

    def clear_cache(self):
        """Clear the ID mapping cache"""
        self._id_mapping = None
        self.logger.debug("ID mapping cache cleared")
