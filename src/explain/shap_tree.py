"""
TreeSHAP Explainer Module
=========================

SHAP explanations for tree-based models (CatBoost, LightGBM).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


@dataclass
class SHAPResult:
    """Container for SHAP analysis results."""
    shap_values: np.ndarray  # (n_samples, n_features, n_classes) or (n_samples, n_features)
    base_values: np.ndarray
    feature_names: List[str]
    global_importance: pd.DataFrame
    top_features: List[str]


@dataclass
class TreeSHAPExplainer:
    """
    SHAP explainer for tree-based models.

    Uses TreeExplainer for efficient SHAP value computation
    on CatBoost and LightGBM models.
    """

    max_samples: int = 1000
    top_k_features: int = 10
    focus_class: Optional[int] = None  # Index of class to focus on (e.g., high-risk)

    def explain(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> SHAPResult:
        """
        Compute SHAP values for a model.

        Args:
            model: Fitted tree model (CatBoost or LightGBM)
            X: Data to explain
            feature_names: Optional feature names

        Returns:
            SHAPResult with SHAP values and importance
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")

        # Subsample if too many samples
        if len(X) > self.max_samples:
            indices = np.random.choice(len(X), self.max_samples, replace=False)
            X = X[indices]

        # Create explainer
        if hasattr(model, '_model'):
            # Our wrapper models
            internal_model = model._model
        else:
            internal_model = model

        # XGBoost 3.x stores base_score as a multi-class vector string which
        # SHAP 0.49.x cannot parse in TreeExplainer.  When that happens, fall back
        # to a fast LightGBM surrogate trained on model predictions so the rest of
        # the SHAP pipeline (figures, importance table) works unchanged.
        try:
            explainer = shap.TreeExplainer(internal_model)
            use_surrogate = False
        except (ValueError, TypeError, Exception) as tree_err:
            logger.debug(f"TreeExplainer failed ({tree_err}) — fitting LightGBM surrogate for SHAP")
            use_surrogate = True

        if use_surrogate:
            explainer, X = self._build_lgb_surrogate_explainer(model, X, feature_names)

        # Compute SHAP values — handle both legacy and SHAP 0.46+ Explanation API
        raw = explainer(X)  # returns Explanation object in SHAP 0.46+

        if hasattr(raw, 'values'):
            # SHAP 0.46+: Explanation object with .values ndarray
            shap_values = raw.values
            base_values = raw.base_values if hasattr(raw, 'base_values') else None
        else:
            # Legacy: direct ndarray or list
            shap_values = raw
            base_values = None

        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class: list of (n_samples, n_features) arrays
            shap_values = np.stack(shap_values, axis=-1)

        # Ensure float array (XGBoost 3.x can return object arrays in some versions)
        try:
            shap_values = np.array(shap_values, dtype=np.float64)
        except (ValueError, TypeError):
            raise ValueError(
                f"Could not convert SHAP values to float array. "
                f"Got type {type(shap_values)}, shape-like {getattr(shap_values, 'shape', 'N/A')}"
            )

        # shap_values shape for multi-class XGBoost/LightGBM via Explanation:
        # (n_samples, n_features, n_classes) — already correct 3-D format
        # For 2-D binary output: (n_samples, n_features) — also fine

        # Get base values
        if base_values is not None:
            if isinstance(base_values, (list, np.ndarray)):
                base_values = np.array(base_values, dtype=np.float64)
        elif hasattr(explainer, 'expected_value'):
            ev = explainer.expected_value
            base_values = np.array(ev, dtype=np.float64) if isinstance(ev, (list, np.ndarray)) else np.array([ev], dtype=np.float64)
        else:
            base_values = np.zeros(shap_values.shape[-1] if shap_values.ndim > 2 else 1)

        # Set feature names
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Compute global importance
        global_importance = self._compute_global_importance(shap_values, feature_names)

        # Get top features
        top_features = global_importance.head(self.top_k_features)['feature'].tolist()

        return SHAPResult(
            shap_values=shap_values,
            base_values=base_values,
            feature_names=feature_names,
            global_importance=global_importance,
            top_features=top_features
        )

    def _build_lgb_surrogate_explainer(self, model, X, feature_names):
        """
        Fit a small LightGBM model on the predictions of `model` and return a
        TreeExplainer wrapping it.  Used as a fast fallback when TreeExplainer
        cannot parse the primary model (e.g. XGBoost 3.x base_score format).
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM required as SHAP surrogate when XGBoost 3.x is used")

        y_pred = model.predict(X)
        n_classes = len(np.unique(y_pred))
        n_classes = max(n_classes, 2)

        params = {
            'objective': 'multiclass' if n_classes > 2 else 'binary',
            'num_class': n_classes if n_classes > 2 else None,
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'verbose': -1,
            'force_col_wise': True,
        }
        if params['num_class'] is None:
            del params['num_class']

        surrogate = lgb.LGBMClassifier(**params)
        surrogate.fit(X, y_pred)
        surrogate_explainer = shap.TreeExplainer(surrogate)
        return surrogate_explainer, X

    def _compute_global_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Compute global feature importance from SHAP values."""
        # For multi-class, focus on specific class or aggregate
        if len(shap_values.shape) == 3:
            if self.focus_class is not None:
                # Use specific class
                values = shap_values[:, :, self.focus_class]
            else:
                # Aggregate across classes (mean absolute)
                values = np.mean(np.abs(shap_values), axis=2)
        else:
            values = shap_values

        # Mean absolute SHAP value per feature
        importance = np.mean(np.abs(values), axis=0)

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)

        return df

    def explain_high_risk_predictions(
        self,
        model: Any,
        X: np.ndarray,
        y_pred: np.ndarray,
        high_risk_indices: List[int],
        feature_names: Optional[List[str]] = None
    ) -> SHAPResult:
        """
        Focus SHAP analysis on high-risk predictions.

        Args:
            model: Fitted model
            X: Full data
            y_pred: Predictions
            high_risk_indices: Indices of high-risk classes
            feature_names: Feature names

        Returns:
            SHAPResult focused on high-risk cases
        """
        # Filter to high-risk predictions
        high_risk_mask = np.isin(y_pred, high_risk_indices)
        X_high_risk = X[high_risk_mask]

        if len(X_high_risk) == 0:
            logger.warning("No high-risk predictions found")
            return self.explain(model, X, feature_names)

        logger.info(f"Explaining {len(X_high_risk)} high-risk predictions")
        return self.explain(model, X_high_risk, feature_names)

    def create_summary_plot(
        self,
        result: SHAPResult,
        output_path: Optional[Path] = None,
        plot_type: str = 'bar'
    ):
        """Create SHAP summary plot."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for plotting")
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_type == 'bar':
            # Bar plot of importance
            top_n = min(self.top_k_features, len(result.feature_names))
            importance = result.global_importance.head(top_n)

            ax.barh(range(top_n), importance['importance'].values[::-1])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(importance['feature'].values[::-1])
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title('Feature Importance (SHAP)')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP plot to {output_path}")

        plt.close()

    def get_feature_attribution_table(
        self,
        result: SHAPResult,
        n_samples: int = 5
    ) -> pd.DataFrame:
        """
        Create feature attribution table for sample predictions.

        Returns DataFrame showing SHAP contributions for each feature.
        """
        records = []

        for i in range(min(n_samples, len(result.shap_values))):
            for j, feature in enumerate(result.feature_names):
                if len(result.shap_values.shape) == 3:
                    # Multi-class: sum across classes or pick one
                    value = np.mean(np.abs(result.shap_values[i, j, :]))
                else:
                    value = result.shap_values[i, j]

                records.append({
                    'sample': i,
                    'feature': feature,
                    'shap_value': value
                })

        return pd.DataFrame(records)
