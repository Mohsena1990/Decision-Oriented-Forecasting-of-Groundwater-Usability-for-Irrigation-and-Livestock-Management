"""
Gaussian Process Classifier for groundwater quality forecasting.

Rationale
---------
Hydrochemical data is spatially structured: neighbouring wells share geology,
aquifer type, and agricultural pressure.  A Gaussian Process with a spatial
kernel captures this correlation explicitly and provides calibrated posterior
probabilities — no threshold tuning required.

Unlike LSTM/GRU (which expect sequential input) and tree models (which ignore
spatial distance), GPC marginalises over aleatoric + epistemic uncertainty,
giving a natural "confidence" signal for each prediction.  This is valuable
for a safety-critical irrigation advisory system.

Design choices
--------------
* Matérn-3/2 kernel — smooth but not infinitely differentiable, appropriate
  for hydrochemistry where abrupt compositional shifts occur at geological
  boundaries.
* WhiteKernel noise term — handles measurement noise in chemical assays.
* multi_class='one_vs_rest' — consistent with the other classifiers; avoids
  the pairwise coupling approximation of one_vs_one.
* Kernel hyperparameters are optimised by maximising the log marginal
  likelihood (sklearn default), so no PSO-GWO loop is needed.
* copy_X_train=False — reduces peak RAM usage on the training fit.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import (
        ConstantKernel, Matern, WhiteKernel,
    )
    GPC_AVAILABLE = True
except ImportError:
    GPC_AVAILABLE = False
    GaussianProcessClassifier = None

from .base import BaseForecaster, ForecasterConfig

logger = logging.getLogger(__name__)


class GPCForecaster(BaseForecaster):
    """
    Gaussian Process Classifier wrapper for groundwater quality forecasting.

    Provides calibrated posterior probabilities and built-in uncertainty
    quantification.  Works well on small datasets (n < 1 000) with spatial
    or chemistry-driven structure.

    Hyperparameters (set via ForecasterConfig.params):
        length_scale  : Matérn kernel length scale initial value  (default 1.0)
        noise_level   : WhiteKernel noise level initial value     (default 0.1)
        nu            : Matérn smoothness parameter (0.5 | 1.5 | 2.5)  (default 1.5)
        n_restarts    : number of optimiser restarts               (default 3)
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        if not GPC_AVAILABLE:
            raise ImportError(
                "scikit-learn GaussianProcessClassifier not available. "
                "Install with: pip install scikit-learn>=1.0"
            )
        super().__init__(config)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
    ) -> "GPCForecaster":
        X = self.apply_feature_mask(X)
        if X_val is not None:
            X_val = self.apply_feature_mask(X_val)

        self._n_features = X.shape[1]
        self._n_classes = len(np.unique(y))
        self._feature_names = feature_names

        length_scale = float(self.config.get_param("length_scale", 1.0))
        noise_level  = float(self.config.get_param("noise_level", 0.1))
        nu           = float(self.config.get_param("nu", 1.5))
        n_restarts   = int(self.config.get_param("n_restarts", 3))

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=length_scale,
                     length_scale_bounds=(1e-3, 1e3),
                     nu=nu)
            + WhiteKernel(noise_level=noise_level,
                          noise_level_bounds=(1e-5, 1e1))
        )

        self._model = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            random_state=self.config.random_state,
            multi_class="one_vs_rest",
            max_iter_predict=100,
            copy_X_train=False,
            n_jobs=-1,
        )

        logger.info(f"GPC fitting on {X.shape[0]} samples, {X.shape[1]} features")
        self._model.fit(X, y)

        # Log optimised kernel for interpretability
        try:
            logger.info(f"GPC optimised kernel: {self._model.kernel_}")
        except Exception:
            pass

        self._is_fitted = True
        logger.info(f"GPC fitted with {self._n_features} features, {self._n_classes} classes")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model not fitted.")
        X = self.apply_feature_mask(X)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model not fitted.")
        X = self.apply_feature_mask(X)
        return self._model.predict_proba(X)

    def get_model_complexity(self) -> Dict[str, Any]:
        n_kernel = 0
        if self._is_fitted and self._model is not None:
            try:
                n_kernel = len(self._model.kernel_.theta)
            except Exception:
                pass
        return {
            "n_kernel_params": n_kernel,
            "n_train_samples": getattr(self._model, "X_train_", np.array([])).shape[0]
            if self._is_fitted else 0,
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """GPC has no direct feature importance; return None."""
        return None

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"GPC saved to {path}")

    def load(self, path: Union[str, Path]) -> "GPCForecaster":
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        return self

    @classmethod
    def get_model_name(cls) -> str:
        return "GPC"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "length_scale": 1.0,
            "noise_level": 0.1,
            "nu": 1.5,
            "n_restarts": 3,
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            "length_scale": (0.1, 10.0),
            "noise_level": (1e-4, 1.0),
        }
