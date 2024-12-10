from numpy.typing import NDArray, ArrayLike
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import shap

from .base import BaseModel, ModelConfig

N_TRIALS = 50


class OptimizedEnsemble(BaseModel):
    """Ensemble model with hyperparameter optimization, stability tracking, and enhanced prediction capabilities.

    Features:
    - Optimized base models (XGBoost, LightGBM, RandomForest)
    - Stacked ensemble with meta-learner
    - Isotonic calibration for better probability estimates
    - Cross-validation stability tracking
    - SHAP-based feature importance and explanations
    - Prediction diagnostics and confidence estimates
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_models: Dict[str, Any] = {}
        self.calibrated_models: Dict[str, Any] = {}
        self.weights: Optional[List[float]] = None
        self.shap_explainers: Dict[str, Any] = {}
        self.meta_learner = None
        self.cross_val_metrics = {}
        self.scaler = StandardScaler()
        self.confidence_thresholds = config.confidence_thresholds or {
            "high": 0.95,
            "medium": 0.85,
            "low": 0.70,
        }
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

    def get_hyperparameters(self, model_name: str, trial: optuna.Trial) -> Dict:
        """Model-specific hyperparameter spaces"""
        if model_name == "xgb":
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            }
        elif model_name == "lgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            }
        else:  # RandomForest
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
        """Train optimized ensemble with comprehensive evaluation"""
        self.feature_names = feature_names or []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Scale features if needed
        X_scaled = self.scaler.fit_transform(X)

        base_model_configs = {
            "xgb": xgb.XGBClassifier(eval_metric="auc", random_state=42),
            "rf": RandomForestClassifier(
                n_jobs=-1, random_state=42, class_weight="balanced"
            ),
            "lgbm": LGBMClassifier(
                n_jobs=-1, random_state=42, class_weight="balanced", verbose=-1
            ),
        }

        for name, model in base_model_configs.items():
            self.logger.info(f"Optimizing {name}...")
            study = optuna.create_study(
                direction="maximize", study_name=f"optuna_{name}"
            )

            def objective(trial):
                params = self.get_hyperparameters(name, trial)
                model.set_params(**params)
                cv_scores = cross_val_score(
                    model, X_scaled, y, cv=cv, scoring="average_precision", n_jobs=-1
                )
                return cv_scores.mean()

            study.optimize(objective, n_trials=N_TRIALS, timeout=600)

            # Train base model (uncalibrated)
            best_model = type(model)(**study.best_params)
            best_model.fit(X_scaled, y)
            self.base_models[name] = best_model

            # Create SHAP explainer on uncalibrated model
            try:
                self.shap_explainers[name] = shap.TreeExplainer(best_model)
            except Exception as e:
                self.logger.warning(f"SHAP explainer failed for {name}: {e}")

            # Create calibrated version for predictions
            calibrated = CalibratedClassifierCV(best_model, cv=5, method="isotonic")
            calibrated.fit(X_scaled, y)
            self.calibrated_models[name] = calibrated

            self.logger.info(f"{name} best score: {study.best_value:.4f}")

        # Build meta-learner using calibrated predictions
        self._build_stacked_model(X_scaled, y)

        # Get stability metrics
        stability_metrics = self.add_cross_validation_stability(X_scaled, y)
        self.logger.info("\nCross-validation Stability Metrics:")
        for metric, value in stability_metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")

        # Split validation data for weight optimization
        X_val, X_test, y_val, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Optimize ensemble weights
        self.weights = self._optimize_weights(X_val, y_val)

        # Create final weighted ensemble
        self.model = VotingClassifier(
            estimators=[(name, model) for name, model in self.calibrated_models.items()]
            + [("meta_learner", self.meta_learner)],
            voting="soft",
            weights=self.weights,
        )

        # Final fit
        self.model.fit(X_scaled, y)

        # Evaluate stability of final model
        final_predictions, unstable_indices = self.predict_with_stability(X_test)
        self.logger.info(f"Number of unstable predictions: {len(unstable_indices)}")

    def predict_with_stability(self, X: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Make predictions with stability assessment

        Args:
            X: Input features array

        Returns:
            Tuple containing:
            - adjusted_predictions: Array of shape (n_samples, 2) with calibrated probabilities
            - unstable_indices: List of indices where predictions are unstable (high variance between models)
        """  # Scale features
        X_scaled = self.scaler.transform(X)

        # Get predictions from base models
        base_predictions = np.zeros((len(self.calibrated_models), len(X_scaled)))
        for i, model in enumerate(self.calibrated_models.values()):
            base_predictions[i] = model.predict_proba(X_scaled)[:, 1]

        # Calculate prediction statistics
        mean_predictions = np.mean(base_predictions, axis=0)
        std_predictions = np.std(base_predictions, axis=0)

        # Identify unstable predictions
        unstable_indices = np.where(std_predictions > 0.2)[0]

        # Get ensemble predictions
        predictions = self.model.predict_proba(X_scaled)

        # Adjust confidence for unstable predictions
        confidence_adjustments = 1 - np.clip(std_predictions, 0, 0.5)
        adjusted_predictions = predictions * confidence_adjustments.reshape(-1, 1)

        return adjusted_predictions, unstable_indices

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with stability consideration"""
        predictions, unstable_indices = self.predict_with_stability(X)
        return (predictions[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions with stability adjustment"""
        return self.predict_with_stability(X)[0]

    def get_prediction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Calculate confidence scores considering stability"""
        X_scaled = self.scaler.transform(X)
        base_predictions = self._get_base_predictions(X_scaled)

        mean_preds = np.mean(base_predictions, axis=0)
        std_preds = np.std(base_predictions, axis=0)

        # Calculate multiple confidence factors
        boundary_distance = np.abs(mean_preds - 0.5)
        model_agreement = 1 - std_preds
        prediction_strength = np.where(mean_preds >= 0.5, mean_preds, 1 - mean_preds)

        # Combine factors with weights
        confidence = (
            boundary_distance * 0.4 + model_agreement * 0.4 + prediction_strength * 0.2
        )

        return np.clip(confidence, 0, 1)

    def add_cross_validation_stability(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5
    ) -> Dict[str, float]:
        """Measure prediction stability across CV folds"""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_predictions = np.zeros((len(X), n_splits))
        all_predictions[:] = np.nan

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model on this fold
            temp_model = VotingClassifier(
                estimators=[
                    (name, clone(model))
                    for name, model in self.calibrated_models.items()
                ]
                + [("meta_learner", clone(self.meta_learner))],
                voting="soft",
                weights=self.weights,
            )
            temp_model.fit(X_train, y_train)
            fold_proba = temp_model.predict_proba(X_val)[:, 1]

            all_predictions[val_idx, fold_idx] = fold_proba

        # Calculate stability metrics
        sample_variances = np.nanvar(all_predictions, axis=1)
        sample_ranges = np.nanmax(all_predictions, axis=1) - np.nanmin(
            all_predictions, axis=1
        )

        return {
            "mean_prediction_variance": np.mean(sample_variances),
            "max_prediction_variance": np.max(sample_variances),
            "mean_prediction_range": np.mean(sample_ranges),
            "max_prediction_range": np.max(sample_ranges),
            "stable_prediction_percentage": np.mean(sample_variances < 0.1),
        }

    def _build_stacked_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build stacked model using calibrated predictions"""
        base_preds = self._get_base_predictions(X)
        meta_features = np.column_stack([base_preds.T, X])

        self.meta_learner = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            num_leaves=8,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42,
            verbose=-1,
        )
        self.meta_learner.fit(meta_features, y)

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models"""
        predictions = []
        for model in self.calibrated_models.values():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        return np.array(predictions)

    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Optimize ensemble weights using scipy minimize"""
        self.logger.info("Optimizing ensemble weights...")
        from scipy.optimize import minimize

        def objective(weights):
            # Ensure weights sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)

            # Get predictions from all models including meta-learner
            all_preds = []
            for model in self.calibrated_models.values():
                all_preds.append(model.predict_proba(X)[:, 1])

            # Add meta-learner predictions
            meta_features = np.column_stack([np.array(all_preds).T, X])
            meta_preds = self.meta_learner.predict_proba(meta_features)[:, 1]
            all_preds.append(meta_preds)

            # Calculate weighted predictions
            weighted_preds = np.zeros(X.shape[0])
            for weight, preds in zip(weights, all_preds):
                weighted_preds += weight * preds

            # Return negative ROC AUC (for minimization)
            return -roc_auc_score(y, weighted_preds)

        n_models = len(self.calibrated_models) + 1  # +1 for meta-learner
        initial_weights = np.array([1.0 / n_models] * n_models)
        bounds = [(0, 1)] * n_models
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            optimized_weights = result.x / np.sum(result.x)
            self.logger.info(f"Optimized weights: {optimized_weights}")
            return optimized_weights.tolist()
        else:
            self.logger.warning("Weight optimization failed. Using equal weights.")
            return initial_weights.tolist()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from base models"""
        importance_dict = {}
        for name, model in self.base_models.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                for feat, imp in zip(self.feature_names, importances):
                    importance_dict[feat] = importance_dict.get(feat, 0) + imp

        if importance_dict:
            # Average and normalize importances
            num_models = len(
                [
                    m
                    for m in self.base_models.values()
                    if hasattr(m, "feature_importances_")
                ]
            )
            importance_dict = {k: v / num_models for k, v in importance_dict.items()}

            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

        return importance_dict

    def explain_prediction(self, instance: np.ndarray) -> Dict[str, float]:
        """Get SHAP explanations using uncalibrated base models"""
        if not isinstance(instance, np.ndarray):
            instance = np.array(instance)

        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Scale instance
        instance_scaled = self.scaler.transform(instance)

        explanations = {}
        for name, explainer in self.shap_explainers.items():
            if explainer is not None:
                shap_values = explainer.shap_values(instance_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                explanations[name] = dict(zip(self.feature_names, shap_values[0]))

        # Aggregate explanations across models
        if explanations:
            aggregated = {}
            for feature in self.feature_names:
                values = [exp[feature] for exp in explanations.values()]
                aggregated[feature] = float(np.mean(values))

            # Sort by absolute contribution
            aggregated = dict(
                sorted(aggregated.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            return aggregated

        return {}

    def analyze_feature_interactions(
        self, X: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Analyze feature interactions using SHAP"""
        interactions = []

        # Use XGBoost model for interaction analysis if available
        base_model = self.base_models.get("xgb")
        if base_model is None:
            base_model = next(iter(self.base_models.values()))

        try:
            X_scaled = self.scaler.transform(X)
            explainer = shap.TreeExplainer(base_model)
            shap_interaction_values = explainer.shap_interaction_values(X_scaled)

            n_features = len(self.feature_names)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    strength = np.abs(shap_interaction_values[:, i, j]).mean()
                    interactions.append(
                        (self.feature_names[i], self.feature_names[j], float(strength))
                    )

            # Sort by interaction strength
            interactions.sort(key=lambda x: x[2], reverse=True)
            return interactions[:top_k]

        except Exception as e:
            self.logger.error(f"Error analyzing feature interactions: {str(e)}")
            return []

    def get_prediction_diagnostics(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get detailed prediction diagnostics"""
        X_scaled = self.scaler.transform(X)
        base_predictions = self._get_base_predictions(X_scaled)

        mean_preds = np.mean(base_predictions, axis=0)
        std_preds = np.std(base_predictions, axis=0)

        # Calculate various diagnostic metrics
        model_agreement = 1 - std_preds
        prediction_strength = np.abs(mean_preds - 0.5)
        confidence_scores = self.get_prediction_confidence(X)

        unstable_mask = std_preds > 0.2

        return {
            "mean_probability": mean_preds,
            "std_probability": std_preds,
            "model_agreement": model_agreement,
            "prediction_strength": prediction_strength,
            "confidence_scores": confidence_scores,
            "unstable_predictions": unstable_mask,
        }

    def select_important_features(
        self, X: np.ndarray, y: np.ndarray, threshold: float = 0.01
    ) -> List[str]:
        """Select important features using SHAP values"""
        X_scaled = self.scaler.transform(X)

        # Use XGBoost for feature selection
        selector = xgb.XGBClassifier(n_estimators=100, random_state=42)
        selector.fit(X_scaled, y)

        explainer = shap.TreeExplainer(selector)
        shap_values = explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        importance_vals = np.abs(shap_values).mean(0)
        importance = dict(zip(self.feature_names, importance_vals))

        # Select features above threshold
        max_importance = max(importance.values())
        selected_features = [
            f for f, imp in importance.items() if imp > threshold * max_importance
        ]

        self.logger.info(
            f"\nSelected {len(selected_features)}/{len(self.feature_names)} features"
        )
        self.logger.info(
            "Top 10 features: %s",
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10],
        )

        return selected_features
