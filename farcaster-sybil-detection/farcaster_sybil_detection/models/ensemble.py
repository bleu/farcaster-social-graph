from typing import Dict, List, Optional, Any
import numpy as np
import logging
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


class OptimizedEnsemble:
    """Optimized ensemble model with stability tracking and confidence estimates"""

    def __init__(self, random_state: int = 42, n_trials: int = 50):
        self.random_state = random_state
        self.n_trials = n_trials
        self.base_models: Dict[str, Any] = {}
        self.calibrated_models: Dict[str, Any] = {}
        self.model: Optional[VotingClassifier] = None
        self.feature_names: List[str] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
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

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all calibrated base models"""
        predictions = []
        for model in self.calibrated_models.values():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        return np.array(predictions)

    def _get_trial_params(self, model_name: str, trial: optuna.Trial) -> Dict:
        """Get hyperparameter search space for each model type"""
        if model_name == "xgb":
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        elif model_name == "lgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            }
        else:  # RandomForest
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> None:
        """Train ensemble with optimized base models"""
        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(X.shape[1])
        ]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Initialize base models
        base_models = {
            "xgb": xgb.XGBClassifier(
                eval_metric="auc",
                random_state=self.random_state,
                scale_pos_weight=np.sum(y == 0) / np.sum(y == 1),
            ),
            "rf": RandomForestClassifier(
                n_jobs=-1, random_state=self.random_state, class_weight="balanced"
            ),
            "lgbm": LGBMClassifier(
                n_jobs=-1,
                random_state=self.random_state,
                class_weight="balanced",
                verbose=-1,
            ),
        }

        # Train and calibrate base models
        for name, model in base_models.items():
            self.logger.info(f"Optimizing {name}...")
            study = optuna.create_study(
                direction="maximize", study_name=f"optuna_{name}"
            )

            def objective(trial):
                params = self._get_trial_params(name, trial)
                model.set_params(**params)
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring="average_precision", n_jobs=-1
                )
                return scores.mean()

            study.optimize(objective, n_trials=self.n_trials)

            # Train and calibrate best model
            best_model = type(model)(**study.best_params)
            best_model.fit(X, y)
            self.base_models[name] = best_model

            calibrated = CalibratedClassifierCV(best_model, cv=5)
            calibrated.fit(X, y)
            self.calibrated_models[name] = calibrated

            self.logger.info(f"{name} best score: {study.best_value:.4f}")

        # Create final ensemble
        self.model = VotingClassifier(
            estimators=[
                (name, model) for name, model in self.calibrated_models.items()
            ],
            voting="soft",
            weights=[1.0 / len(self.calibrated_models)] * len(self.calibrated_models),
        )
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions with stability adjustment"""
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Get base model predictions
        base_predictions = self._get_base_predictions(X)
        std_predictions = np.std(base_predictions, axis=0)

        # Get ensemble predictions
        predictions = self.model.predict_proba(X)

        # Adjust predictions based on model agreement
        confidence_adjustments = 1 - np.clip(std_predictions, 0, 0.5)

        # Move predictions toward 0.5 based on uncertainty while maintaining normalization
        adjusted_class1_probs = predictions[:, 1] * confidence_adjustments + 0.5 * (
            1 - confidence_adjustments
        )

        adjusted_predictions = np.zeros_like(predictions)
        adjusted_predictions[:, 1] = adjusted_class1_probs
        adjusted_predictions[:, 0] = 1 - adjusted_class1_probs

        return adjusted_predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions"""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

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
            return dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

        return {}

    def get_prediction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions"""
        base_predictions = self._get_base_predictions(X)
        mean_preds = np.mean(base_predictions, axis=0)
        std_preds = np.std(base_predictions, axis=0)

        # Combine distance from decision boundary and model agreement
        boundary_distance = np.abs(mean_preds - 0.5) * 2  # Scale to [0,1]
        model_agreement = 1 - std_preds  # Already in [0,1]

        confidence = (boundary_distance + model_agreement) / 2
        return np.clip(confidence, 0, 1)
