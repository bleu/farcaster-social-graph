from typing import Dict, List, Tuple
from farcaster_sybil_detection.features.interface import IFeatureProvider
from farcaster_sybil_detection.utils.with_logging import LoggedABC, add_logging
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split

from ..config.defaults import Config
from ..models.base import BaseModel
from ..evaluation.metrics import EvaluationMetrics


@add_logging
class Trainer(LoggedABC):
    """Handles model training and evaluation"""

    def __init__(
        self, config: Config, model: BaseModel, feature_manager: IFeatureProvider
    ):
        self.config = config
        self.model = model
        self.feature_manager = feature_manager

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
            self.logger.debug("Training model...")
            self.model.fit(X_train, y_train, feature_names)

            # Save the trained model
            self.model.save()

            # Evaluate
            metrics = self._evaluate(X_test, y_test)
            self.logger.debug("\nEvaluation metrics:")
            for name, value in metrics.items():
                self.logger.debug(f"{name}: {value:.3f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise

    def _prepare_training_data(
        self, labels_df: pl.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data and handle non-numeric columns"""
        self.logger.debug("Building feature matrix...")

        # Ensure labels DataFrame has correct fid type
        labels_df = labels_df.with_columns([pl.col("fid").cast(pl.Int64)])

        # Only get features for fids that have labels
        label_fids = labels_df["fid"].unique().to_list()
        self.logger.debug(f"Preparing features for {len(label_fids)} labeled fids")

        # Build feature matrix only for these fids - ONLY DO THIS ONCE
        matrix = self.feature_manager.build_feature_matrix(target_fids=label_fids)

        # Ensure consistent FID types
        matrix = matrix.with_columns([pl.col("fid").cast(pl.Int64)])

        # Join with labels
        data = matrix.join(labels_df, on="fid", how="inner")
        self.logger.debug(f"Initial data shape after joining with labels: {data.shape}")

        # Identify non-numeric columns to drop
        columns_to_drop = ["fid"]  # Always drop fid
        for col in data.columns:
            if col != "bot":  # Keep the target variable
                dtype = str(data[col].dtype)
                if "List" in dtype or dtype not in [
                    "Int64",
                    "Float64",
                    "Int32",
                    "Float32",
                    "UInt32",
                ]:
                    columns_to_drop.append(col)

        if columns_to_drop:
            self.logger.debug(f"Dropping non-numeric columns: {columns_to_drop}")

        # Handle missing values before converting to numpy
        numeric_data = data.drop(columns_to_drop)

        # Convert remaining numeric columns to Float64 and handle inf/nan values
        convert_cols = [col for col in numeric_data.columns if col != "bot"]

        expressions = []
        for col in convert_cols:
            expr = (
                pl.when(pl.col(col).is_infinite())
                .then(None)  # Convert inf to null first
                .otherwise(pl.col(col))
                .cast(pl.Float64)
                .alias(col)
            )
            expressions.append(expr)

        # Add bot column to expressions
        expressions.append(pl.col("bot"))

        # Apply transformations
        numeric_data = numeric_data.with_columns(expressions)

        # Replace nulls with median/mean for each column
        expressions = []
        for col in convert_cols:
            median = numeric_data.select(pl.col(col).median()).item()
            expr = (
                pl.col(col)
                .fill_null(median)  # Use median instead of mean for robustness
                .clip(-1e9, 1e9)  # Clip very large values
                .alias(col)
            )
            expressions.append(expr)

        numeric_data = numeric_data.with_columns(expressions)

        # Convert to numpy arrays
        feature_names = [col for col in numeric_data.columns if col != "bot"]
        X = numeric_data.select(feature_names).to_numpy()
        y = numeric_data.select("bot").to_numpy().ravel()

        # Final verification that no inf/nan values remain
        if np.any(~np.isfinite(X)):
            self.logger.warning("Replacing remaining non-finite values")
            X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)

        self.logger.debug(f"Final feature matrix shape: {X.shape}")
        self.logger.debug(f"Number of features: {len(feature_names)}")
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
        self.logger.debug("Label distribution:")
        for label, count in zip(unique_labels, label_counts):
            self.logger.debug(f"  Class {label}: {count} ({count/len(y)*100:.2f}%)")

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
