"""
CORAL Ordinal Neural Network
============================

COnsistent RAnk Logits (CORAL) for ordinal classification.

Reference:
    Cao et al., "Rank Consistent Ordinal Regression for Neural Networks with
    Application to Age Estimation", Pattern Recognition Letters, 2020.
    https://arxiv.org/abs/1901.07884

Architecture:
    - Shared MLP feature extractor: Linear → BatchNorm → ReLU → Dropout → ...
    - CORAL output: K-1 binary neurons (K = n_classes), shared weights,
      individual biases → guarantees P(rank≥1) ≥ P(rank≥2) ≥ ...
    - Loss: sum of binary cross-entropy over all K-1 rank thresholds
    - Prediction: cumulative probabilities → class probabilities → argmax

CORAL loss applied with optional focal weighting per threshold.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..base import BaseForecaster, ForecasterConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CORAL network
# ---------------------------------------------------------------------------

class CORALNet(nn.Module):
    """
    CORAL ordinal network.

    Shared MLP + CORAL output layer (K-1 binary outputs with shared weights
    but individual biases).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_thresholds = n_classes - 1  # K-1 binary classifiers

        # Shared MLP feature extractor
        layers: List["nn.Module"] = []
        prev_dim = n_features
        for _ in range(n_layers):
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)

        # CORAL output: shared weight vector (hidden_dim → 1) + K-1 individual biases
        self.coral_weight = nn.Linear(prev_dim, 1, bias=False)
        self.coral_biases = nn.Parameter(torch.zeros(self.n_thresholds))

        # Initialise biases in ascending order so ordinal constraint starts satisfied
        with torch.no_grad():
            for k in range(self.n_thresholds):
                self.coral_biases[k] = float(k) - self.n_thresholds / 2.0

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: (B, n_features)

        Returns:
            cum_logits: (B, K-1)  raw logits for each rank threshold
        """
        h = self.feature_extractor(x)       # (B, hidden_dim)
        w = self.coral_weight(h)             # (B, 1)
        cum_logits = w + self.coral_biases   # (B, K-1)  broadcast
        return cum_logits

    def predict_cum_proba(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: (B, n_features)
        Returns:
            cum_proba: (B, K-1) P(rank >= k) for k=1..K-1
        """
        cum_logits = self.forward(x)
        return torch.sigmoid(cum_logits)

    def predict_class_proba(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Convert cumulative probabilities to per-class probabilities.

        P(y=0) = 1 - P(rank>=1)
        P(y=k) = P(rank>=k) - P(rank>=k+1)   for k=1..K-2
        P(y=K-1) = P(rank>=K-1)

        Probabilities are clipped to (eps, 1-eps) before differencing to
        avoid degenerate values, then renormalised.
        """
        eps = 1e-6
        cum = self.predict_cum_proba(x).clamp(eps, 1 - eps)   # (B, K-1)

        # Build (B, K) class proba tensor
        p0 = 1.0 - cum[:, :1]                     # (B, 1)  P(y=0)
        p_mid = cum[:, :-1] - cum[:, 1:]           # (B, K-2)
        p_last = cum[:, -1:]                        # (B, 1)  P(y=K-1)

        proba = torch.cat([p0, p_mid, p_last], dim=1)   # (B, K)
        # Clamp negatives caused by floating point
        proba = proba.clamp(min=0.0)
        proba = proba / proba.sum(dim=1, keepdim=True).clamp(min=eps)
        return proba


# ---------------------------------------------------------------------------
# CORAL loss
# ---------------------------------------------------------------------------

def coral_loss(
    cum_logits: "torch.Tensor",
    targets: "torch.Tensor",
    class_weights: Optional["torch.Tensor"] = None,
    gamma: float = 2.0,
) -> "torch.Tensor":
    """
    CORAL loss: sum of focal-binary cross-entropy over each rank threshold.

    For each threshold k, the binary label is 1 if y > k else 0.
    Focal weighting reduces the contribution of easy (well-separated) examples.

    Args:
        cum_logits: (B, K-1) raw logits from CORALNet.forward()
        targets: (B,) integer class labels [0..K-1]
        class_weights: (K,) optional class weights — mapped to binary weights
                       as the mean of class weights for classes above threshold k
        gamma: focal loss exponent

    Returns:
        Scalar loss
    """
    n_thresholds = cum_logits.size(1)
    n_classes = n_thresholds + 1

    # Binary labels: rank_labels[b, k] = 1 iff targets[b] > k
    rank_labels = torch.zeros_like(cum_logits)
    for k in range(n_thresholds):
        rank_labels[:, k] = (targets > k).float()

    # Binary focal cross-entropy per threshold
    bce = F.binary_cross_entropy_with_logits(
        cum_logits, rank_labels, reduction='none'
    )   # (B, K-1)

    pt = torch.exp(-bce)
    focal = ((1.0 - pt) ** gamma * bce)   # (B, K-1)

    # Optional class-level weighting: weight for threshold k = weight of the
    # positive class (i.e., classes > k)
    if class_weights is not None:
        threshold_weights = torch.zeros(n_thresholds, device=cum_logits.device)
        for k in range(n_thresholds):
            above = class_weights[k + 1:].mean() if k + 1 < n_classes else class_weights[-1]
            threshold_weights[k] = above
        focal = focal * threshold_weights.unsqueeze(0)

    return focal.mean()


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------

class CORALForecaster(BaseForecaster):
    """
    CORAL Ordinal Neural Network forecaster.

    Enforces ordinal consistency P(T1)≥P(T2)≥P(T3) by construction.
    Directly exploits the ordinal T1<T2<T3 risk structure.

    PSO-GWO optimizable hyperparameters:
        hidden_dim, n_layers, dropout, learning_rate, batch_size
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        super().__init__(config)
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': []
        }
        self._net: Optional["CORALNet"] = None
        self._device: Optional["torch.device"] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
    ) -> 'CORALForecaster':
        """Fit the CORAL network."""
        X = self.apply_feature_mask(X)
        if X_val is not None:
            X_val = self.apply_feature_mask(X_val)

        self._n_features = X.shape[1]
        self._n_classes = len(np.unique(y))
        self._feature_names = feature_names

        p = self.config.params
        hidden_dim = int(p.get('hidden_dim', 64))
        n_layers = int(p.get('n_layers', 2))
        dropout = float(p.get('dropout', 0.1))
        lr = float(p.get('learning_rate', 0.001))
        batch_size = int(p.get('batch_size', 32))
        max_epochs = int(p.get('max_epochs', 200))
        patience = int(p.get('patience', 20))

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(self.config.random_state)

        self._net = CORALNet(
            n_features=self._n_features,
            n_classes=self._n_classes,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self._device)

        class_weight_tensor = self._build_class_weight_tensor()

        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
            drop_last=len(X_t) % batch_size == 1
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self._device)
            y_val_t = torch.tensor(y_val, dtype=torch.long).to(self._device)

        self.training_history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            self._net.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_b, y_b in train_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                optimizer.zero_grad()

                cum_logits = self._net(X_b)
                loss = coral_loss(cum_logits, y_b,
                                  class_weights=class_weight_tensor, gamma=2.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train = epoch_loss / max(n_batches, 1)
            self.training_history['train_loss'].append(avg_train)

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_cum = self._net(X_val_t)
                    val_loss = coral_loss(val_cum, y_val_t,
                                         class_weights=class_weight_tensor, gamma=2.0).item()
                self.training_history['val_loss'].append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k_: v_.cpu().clone()
                                  for k_, v_ in self._net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.debug(f"CORAL early stop at epoch {epoch + 1}")
                        break
            else:
                self.training_history['val_loss'].append(avg_train)

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        logger.info(
            f"CORAL fitted: {self._n_features} features, {self._n_classes} classes, "
            f"hidden_dim={hidden_dim}, n_layers={n_layers}, epochs={epoch + 1}"
        )
        return self

    def _build_class_weight_tensor(self) -> Optional["torch.Tensor"]:
        if self.config.class_weights and self._device is not None:
            n_cls = self._n_classes or self.config.n_classes
            weights = np.ones(n_cls, dtype=np.float32)
            for cls_idx, w in self.config.class_weights.items():
                if int(cls_idx) < n_cls:
                    weights[int(cls_idx)] = float(w)
            return torch.tensor(weights, dtype=torch.float32).to(self._device)
        return None

    # ------------------------------------------------------------------
    # predict / predict_proba
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ordinal class probabilities.

        Returns:
            proba: (N, K) array with valid probability distribution.
                   Ordinal structure is enforced by construction.
        """
        if not self._is_fitted or self._net is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)
        self._net.eval()

        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
            proba = self._net.predict_class_proba(X_t).cpu().numpy()

        return proba

    # ------------------------------------------------------------------
    # Complexity
    # ------------------------------------------------------------------

    def get_model_complexity(self) -> Dict[str, Any]:
        if not self._is_fitted or self._net is None:
            return {'n_params': 0}
        n_params = sum(p.numel() for p in self._net.parameters() if p.requires_grad)
        return {
            'n_params': n_params,
            'hidden_dim': self.config.params.get('hidden_dim', 64),
            'n_layers': self.config.params.get('n_layers', 2),
            'n_features': self.get_n_selected_features(),
        }

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self._net.state_dict(), path.with_suffix('.pt'))

        with open(path.with_suffix('.config.pkl'), 'wb') as f:
            pickle.dump({
                'config': self.config,
                'n_features': self._n_features,
                'n_classes': self._n_classes,
                'feature_names': self._feature_names,
                'training_history': self.training_history,
            }, f)

        logger.info(f"CORAL model saved to {path}")

    def load(self, path: Union[str, Path]) -> 'CORALForecaster':
        path = Path(path)

        with open(path.with_suffix('.config.pkl'), 'rb') as f:
            state = pickle.load(f)

        self.config = state['config']
        self._n_features = state['n_features']
        self._n_classes = state['n_classes']
        self._feature_names = state['feature_names']
        self.training_history = state.get('training_history', {'train_loss': [], 'val_loss': []})

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        p = self.config.params

        self._net = CORALNet(
            n_features=self._n_features,
            n_classes=self._n_classes,
            hidden_dim=int(p.get('hidden_dim', 64)),
            n_layers=int(p.get('n_layers', 2)),
            dropout=float(p.get('dropout', 0.1)),
        ).to(self._device)

        self._net.load_state_dict(
            torch.load(path.with_suffix('.pt'), map_location=self._device)
        )
        self._net.eval()
        self._is_fitted = True

        logger.info(f"CORAL model loaded from {path}")
        return self

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_model_name(cls) -> str:
        return "CORAL"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'hidden_dim': 32,
            'n_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 300,
            'patience': 40,
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            'hidden_dim': (16, 64),
            'n_layers': (1, 2),
            'dropout': (0.0, 0.3),
            'learning_rate': (1e-4, 1e-2),
            'batch_size': (16, 32),
        }
