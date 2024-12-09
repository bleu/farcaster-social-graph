from typing import Dict, List, Tuple
import logging
from farcaster_sybil_detection.features.interface import IFeatureProvider
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split

from ..config.defaults import Config
from ..models.base import BaseModel
from ..features.manager import FeatureManager
from ..evaluation.metrics import EvaluationMetrics


class Trainer:
    """Handles model training and evaluation"""

    def __init__(
        self, config: Config, model: BaseModel, feature_manager: IFeatureProvider
    ):
        self.config = config
        self.model = model
        self.feature_manager = feature_manager
        self._setup_logging()

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

    def train(
        self, labels_df: pl.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Dict[str, float]:
        """Train and evaluate model"""
        try:
            X, y, feature_names = self._prepare_training_data(labels_df)

            # Validate data
            self._validate_data(X, y, feature_names)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Train model
            self.logger.info("Training model...")
            self.model.fit(X_train, y_train, feature_names)

            # Save the trained model
            self.model.save()

            # Evaluate
            metrics = self._evaluate(X_test, y_test)
            self.logger.info("\nEvaluation metrics:")
            for name, value in metrics.items():
                self.logger.info(f"{name}: {value:.3f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise

    def _prepare_training_data(
        self, labels_df: pl.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data and handle non-numeric columns"""
        self.logger.info("Building feature matrix...")

        # Ensure labels DataFrame has correct fid type
        labels_df = labels_df.with_columns([pl.col("fid").cast(pl.Int64)])

        # Only get features for fids that have labels
        label_fids = labels_df["fid"].unique().to_list()
        self.logger.info(f"Preparing features for {len(label_fids)} labeled fids")

        # Now build feature matrix only for these fids
        matrix = self.feature_manager.build_feature_matrix(target_fids=label_fids)

        # Ensure consistent FID types
        matrix = matrix.with_columns([pl.col("fid").cast(pl.Int64)])

        # Join with labels
        data = matrix.join(labels_df, on="fid", how="inner")
        self.logger.info(f"Initial data shape after joining with labels: {data.shape}")

        # Rest of the method remains the same...
        # Identify and drop non-numeric columns
        column_types_to_drop = {
            col: str(data[col].dtype)
            for col in data.columns
            if str(data[col].dtype) not in ["Int64", "Float64", "Int32", "Float32"]
        }
        columns_to_drop = list(column_types_to_drop.keys())
        if columns_to_drop:
            self.logger.info(
                f"Dropping non-numeric columns: {', '.join([f'{col} ({column_types_to_drop[col]})' for col in columns_to_drop])}"
            )

        # Add identity columns to drop list
        columns_to_drop += ["fid", "bot"]

        # Handle missing values before converting to numpy
        numeric_data = data.drop(columns_to_drop)

        # Fill remaining nulls with 0
        numeric_data = numeric_data.fill_null(0)

        # Convert to numpy arrays
        X = numeric_data.to_numpy()
        feature_names = numeric_data.columns
        y = data["bot"].to_numpy()

        self.logger.info(f"Final feature matrix shape: {X.shape}")
        self.logger.info(f"Number of features: {len(feature_names)}")

        return X, y, feature_names

    def _validate_data(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Validate data before training"""
        # Check for NaN/infinite values
        if np.any(np.isnan(X)):
            nan_features = [
                feature_names[i] for i in np.where(np.isnan(X).any(axis=0))[0]
            ]
            raise ValueError(f"NaN values found in features: {nan_features}")

        if np.any(np.isinf(X)):
            inf_features = [
                feature_names[i] for i in np.where(np.isinf(X).any(axis=0))[0]
            ]
            raise ValueError(f"Infinite values found in features: {inf_features}")

        # Check label distribution
        unique_labels, label_counts = np.unique(y, return_counts=True)
        self.logger.info("Label distribution:")
        for label, count in zip(unique_labels, label_counts):
            self.logger.info(f"  Class {label}: {count} ({count/len(y)*100:.2f}%)")

        # Log feature statistics
        feature_stats = {
            "min": np.min(X, axis=0),
            "max": np.max(X, axis=0),
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
        }

        # Check for potential numerical issues
        for stat_name, values in feature_stats.items():
            large_values = np.where(np.abs(values) > 1e6)[0]
            if len(large_values) > 0:
                affected_features = [feature_names[i] for i in large_values]
                self.logger.warning(
                    f"Large {stat_name} values (>1e6) in features: {affected_features}"
                )

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        # Ensure numpy arrays
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        return EvaluationMetrics.compute_all_metrics(
            y_true=y_test, y_pred=y_pred, y_prob=y_prob
        )
