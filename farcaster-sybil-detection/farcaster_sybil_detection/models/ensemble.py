import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import shap

from .base import BaseModel, ModelConfig

N_TRIALS = 50

class OptimizedEnsemble(BaseModel):
    """Ensemble model with hyperparameter optimization and stability tracking"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_models: Dict[str, Any] = {}
        self.weights: Optional[List[float]] = None
        self.shap_explainers: Dict[str, Any] = {}
        self.meta_learner = None
        self.cross_val_metrics = {}
        self._setup_logging()

    
    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            # Prevent adding multiple handlers in interactive environments
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def get_hyperparameters(self, model_name: str, trial: optuna.Trial) -> Dict:
        """Model-specific hyperparameter spaces"""
        if model_name == 'xgb':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
            }
        elif model_name == 'lgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30)
            }
        else:  # RandomForest
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
        """Train optimized ensemble with cross-validation"""
        self.feature_names = feature_names or []
                
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Define and optimize base models
        base_model_configs = {
            'xgb': xgb.XGBClassifier(eval_metric='auc', random_state=42),
            'rf': RandomForestClassifier(n_jobs=-1, random_state=42, class_weight='balanced'),
            'lgbm': LGBMClassifier(n_jobs=-1, random_state=42, class_weight='balanced', verbose=-1)
        }
        
        for name, model in base_model_configs.items():
            self.logger.info(f"Optimizing {name}...")
            study = optuna.create_study(direction='maximize', study_name=f'optuna_{name}')
            
            def objective(trial):
                params = self.get_hyperparameters(name, trial)
                model.set_params(**params)
                cv_scores = cross_val_score(
                    model, X, y, cv=cv, scoring='average_precision', n_jobs=-1
                )
                return cv_scores.mean()
            
            study.optimize(objective, n_trials=N_TRIALS, timeout=600)
            
            # Train and calibrate with best params
            best_model = type(model)(**study.best_params)
            best_model.fit(X, y)
            
            # Create SHAP explainer
            try:
                self.shap_explainers[name] = shap.TreeExplainer(best_model)
            except Exception as e:
                self.logger.warning(f"SHAP explainer failed for {name}: {e}")
            
            # Calibrate probabilities
            calibrated = CalibratedClassifierCV(best_model, cv=5)
            calibrated.fit(X, y)
            self.base_models[name] = calibrated
            
            self.logger.info(f"{name} best score: {study.best_value:.4f}")
            
        # Build meta-learner
        self._build_stacked_model(X, y)
        
        # Create final ensemble
        self.model = VotingClassifier(
            estimators=[
                (name, model) for name, model in self.base_models.items()
            ] + [('meta_learner', self.meta_learner)],
            voting='soft',
            weights=self.weights if self.weights is not None else [1.0] * (len(self.base_models) + 1)
        )

        
        # Optimize weights
        self.weights = self._optimize_weights(X, y)
        self.model.fit(X, y)
        
    def _build_stacked_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build stacked model using base predictions"""
        base_preds = np.zeros((len(self.base_models), len(X)))
        for i, (name, model) in enumerate(self.base_models.items()):
            base_preds[i] = model.predict_proba(X)[:, 1]
            
        meta_features = np.column_stack([base_preds.T, X])
        
        self.meta_learner = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            num_leaves=8,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42,
            verbose=-1
        )
        self.meta_learner.fit(meta_features, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.model:
            raise ValueError("Model has not been trained.")
        return self.model.predict_proba(X)[:, 1]

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimation"""
        # Get base model predictions
        base_predictions = []
        for model in self.base_models.values():
            pred = model.predict_proba(X)[:, 1]
            base_predictions.append(pred)
            
        base_predictions = np.array(base_predictions)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(base_predictions, axis=0)
        std_pred = np.std(base_predictions, axis=0)
        
        return mean_pred, std_pred


    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance"""
        importance_dict = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                for feat, imp in zip(self.feature_names, model.feature_importances_):
                    importance_dict[feat] = importance_dict.get(feat, 0) + imp
                    
        # Average across models
        num_models = len(self.base_models)
        if num_models > 0:
            importance_dict = {k: v/num_models for k, v in importance_dict.items()}
            
        return importance_dict

        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the positive class."""
        if not self.model:
            raise ValueError("Model has not been trained.")
        return self.model.predict_proba(X)

    def get_prediction_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get confidence scores for each prediction."""
        # Get base model predictions
        base_predictions = []
        for model in self.base_models.values():
            pred = model.predict_proba(X)[:, 1]
            base_predictions.append(pred)
            
        base_predictions = np.array(base_predictions)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(base_predictions, axis=0)
        std_pred = np.std(base_predictions, axis=0)
        
        # Convert to confidence scores
        confidence = 1 - (std_pred / (mean_pred + 1e-6))
        
        return confidence  # Return array instead of single float
    
    def explain_prediction(self, instance: np.ndarray) -> Dict[str, float]:
        """Get SHAP explanations for predictions."""
        explanations = {}
        for name, explainer in self.shap_explainers.items():
            shap_values = explainer.shap_values(instance.reshape(1, -1))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            explanations[name] = dict(zip(self.feature_names, shap_values[0]))
        return explanations

    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Optimize ensemble weights using cross-validation"""
        self.logger.info("Optimizing ensemble weights...")
        from sklearn.model_selection import cross_val_score
        from scipy.optimize import minimize

        def objective(weights):
            # Ensure weights sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            # Weighted average predictions
            preds = np.zeros(X.shape[0])
            for weight, model in zip(weights, self.base_models.values()):
                preds += weight * model.predict_proba(X)[:, 1]
            # Negative ROC AUC
            score = roc_auc_score(y, preds)
            return -score

        initial_weights = np.array([1.0 / len(self.base_models)] * len(self.base_models))
        bounds = [(0, 1)] * len(self.base_models)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            optimized_weights = result.x / np.sum(result.x)
            self.logger.info(f"Optimized weights: {optimized_weights}")
            return optimized_weights.tolist()
        else:
            self.logger.warning("Weight optimization failed. Using equal weights.")
            return initial_weights.tolist()
