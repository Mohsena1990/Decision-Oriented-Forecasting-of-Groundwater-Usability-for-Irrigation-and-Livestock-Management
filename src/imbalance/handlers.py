"""
Imbalance Handling Module
=========================

Advanced strategies for handling class imbalance in groundwater classification.

Supported strategies:
    - smote_tomek    : SMOTE oversampling + Tomek links undersampling (recommended)
    - borderline_smote: SMOTE on borderline minority samples only
    - adasyn         : Adaptive synthetic sampling
    - smote          : Basic SMOTE oversampling
    - ctgan          : Conditional Tabular GAN (deep generative augmentation)
    - class_weights  : Weights only, no resampling
    - none           : No imbalance handling

Class weights are always computed and returned regardless of strategy,
so they can be passed to model loss functions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_class_weights(
    y: np.ndarray,
    method: str = "balanced"
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced classification.

    Args:
        y: Target array
        method: "balanced", "sqrt", or "none"

    Returns:
        Dictionary mapping class index to weight
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    if method == "balanced":
        weights = n_samples / (n_classes * counts)
    elif method == "sqrt":
        weights = np.sqrt(n_samples / (n_classes * counts))
    else:
        weights = np.ones(n_classes)

    return {int(cls): float(w) for cls, w in zip(unique_classes, weights)}


@dataclass
class ImbalanceHandler:
    """
    Handle class imbalance through combined class weighting and resampling.

    Class weights are ALWAYS computed (for model loss functions).
    Resampling is applied based on the chosen strategy.
    """

    strategy: str = "smote_tomek"
    weight_method: str = "balanced"
    smote_k_neighbors: int = 5
    gan_epochs: int = 300
    random_state: int = 42

    _class_weights: Optional[Dict[int, float]] = field(default=None, init=False)

    def fit(self, y: np.ndarray) -> 'ImbalanceHandler':
        """Fit handler — always compute class weights from original distribution."""
        self._class_weights = compute_class_weights(y, self.weight_method)
        logger.info(f"Class weights ({self.weight_method}): {self._class_weights}")
        return self

    def get_class_weights(self) -> Optional[Dict[int, float]]:
        """Return computed class weights (always available after fit)."""
        return self._class_weights

    def get_sample_weights(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Get per-sample weights for training."""
        if self._class_weights is None:
            return None
        return np.array([self._class_weights.get(int(label), 1.0) for label in y])

    def resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply resampling strategy to training data.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Resampled (X, y). Returns originals if strategy is 'class_weights' or 'none'.
        """
        strategy = self.strategy.lower()

        if strategy in ('class_weights', 'none'):
            logger.info(f"No resampling (strategy='{strategy}'). Using class weights only.")
            return X, y

        logger.info(f"Applying resampling strategy: {strategy}")
        logger.info(f"  Before: {len(y)} samples | dist={dict(zip(*np.unique(y, return_counts=True)))}")

        if strategy == 'smote_tomek':
            X_res, y_res = self._smote_tomek(X, y)
        elif strategy == 'borderline_smote':
            X_res, y_res = self._borderline_smote(X, y)
        elif strategy == 'adasyn':
            X_res, y_res = self._adasyn(X, y)
        elif strategy == 'smote':
            X_res, y_res = self._smote(X, y)
        elif strategy == 'ctgan':
            X_res, y_res = self._ctgan(X, y)
        else:
            logger.warning(f"Unknown strategy '{strategy}', skipping resampling.")
            return X, y

        logger.info(f"  After : {len(y_res)} samples | dist={dict(zip(*np.unique(y_res, return_counts=True)))}")
        return X_res, y_res

    # ------------------------------------------------------------------
    # Private resampling implementations
    # ------------------------------------------------------------------

    def _smote_tomek(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE

            resampler = SMOTETomek(
                smote=SMOTE(k_neighbors=self.smote_k_neighbors, random_state=self.random_state),
                random_state=self.random_state
            )
            return resampler.fit_resample(X, y)
        except ImportError:
            logger.warning("imbalanced-learn not installed. Falling back to SMOTE only.")
            return self._smote(X, y)
        except Exception as e:
            logger.warning(f"SMOTETomek failed ({e}). Falling back to SMOTE.")
            return self._smote(X, y)

    def _borderline_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from imblearn.over_sampling import BorderlineSMOTE

            resampler = BorderlineSMOTE(
                k_neighbors=self.smote_k_neighbors,
                random_state=self.random_state,
                kind='borderline-1'
            )
            return resampler.fit_resample(X, y)
        except ImportError:
            logger.warning("imbalanced-learn not installed. Falling back to SMOTE.")
            return self._smote(X, y)
        except Exception as e:
            logger.warning(f"BorderlineSMOTE failed ({e}). Falling back to SMOTE.")
            return self._smote(X, y)

    def _adasyn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from imblearn.over_sampling import ADASYN

            resampler = ADASYN(
                n_neighbors=self.smote_k_neighbors,
                random_state=self.random_state
            )
            return resampler.fit_resample(X, y)
        except ImportError:
            logger.warning("imbalanced-learn not installed. Falling back to SMOTE.")
            return self._smote(X, y)
        except Exception as e:
            logger.warning(f"ADASYN failed ({e}). Falling back to SMOTE.")
            return self._smote(X, y)

    def _smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from imblearn.over_sampling import SMOTE

            # Guard: SMOTE needs at least k_neighbors+1 samples per class
            unique, counts = np.unique(y, return_counts=True)
            safe_k = min(self.smote_k_neighbors, int(counts.min()) - 1)
            safe_k = max(1, safe_k)

            resampler = SMOTE(k_neighbors=safe_k, random_state=self.random_state)
            return resampler.fit_resample(X, y)
        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
            return X, y
        except Exception as e:
            logger.warning(f"SMOTE failed ({e}). Returning original data.")
            return X, y

    def _ctgan(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conditional Tabular GAN augmentation.

        Uses SDV's CTGAN to synthesize samples for minority classes,
        then appends them to the original data.  Falls back to SMOTE-Tomek
        if sdv is not installed or training fails.
        """
        try:
            import pandas as pd
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata

            logger.info(f"Training CTGAN for {self.gan_epochs} epochs ...")

            # Build a temporary DataFrame
            col_names = [f'f{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=col_names)
            df['__label__'] = y.astype(int)

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)
            # Force label column to categorical
            metadata.update_column('__label__', sdtype='categorical')

            synthesizer = CTGANSynthesizer(
                metadata,
                epochs=self.gan_epochs,
                verbose=False
            )
            synthesizer.fit(df)

            # Determine how many synthetic samples to generate per minority class
            unique_classes, counts = np.unique(y, return_counts=True)
            majority_count = int(counts.max())

            synthetic_frames = []
            for cls, cnt in zip(unique_classes, counts):
                n_needed = majority_count - cnt
                if n_needed <= 0:
                    continue
                conditions = [
                    {'__label__': int(cls)} for _ in range(n_needed)
                ]
                try:
                    from sdv.sampling import Condition
                    cond = Condition(num_rows=n_needed, column_values={'__label__': int(cls)})
                    syn = synthesizer.sample_from_conditions([cond])
                    synthetic_frames.append(syn)
                except Exception as inner:
                    logger.warning(f"  CTGAN condition sample failed for class {cls}: {inner}")

            if not synthetic_frames:
                logger.warning("CTGAN generated no synthetic samples. Falling back to SMOTETomek.")
                return self._smote_tomek(X, y)

            syn_all = pd.concat(synthetic_frames, ignore_index=True)
            X_syn = syn_all[col_names].values
            y_syn = syn_all['__label__'].values.astype(int)

            X_augmented = np.vstack([X, X_syn])
            y_augmented = np.concatenate([y, y_syn])

            logger.info(f"CTGAN generated {len(y_syn)} synthetic samples.")
            return X_augmented, y_augmented

        except ImportError:
            logger.warning("sdv not installed (pip install sdv). Falling back to SMOTETomek.")
            return self._smote_tomek(X, y)
        except Exception as e:
            logger.warning(f"CTGAN augmentation failed ({e}). Falling back to SMOTETomek.")
            return self._smote_tomek(X, y)
