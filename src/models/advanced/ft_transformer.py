"""
Feature Tokenization Transformer (FT-Transformer)
==================================================

Pure PyTorch implementation of the FT-Transformer for tabular classification.

Reference:
    Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
    NeurIPS 2021. https://arxiv.org/abs/2106.11959

Architecture:
    - Each scalar feature → Linear(1, d_token) + bias → d_token-dim token
    - CLS token prepended (learnable)
    - N × TransformerEncoderLayer (multi-head self-attention + FFN)
    - CLS token output → MLP head → n_classes logits
    - Focal loss for class imbalance
"""

import logging
import math
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
# Focal loss
# ---------------------------------------------------------------------------

def weighted_ce(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    class_weights: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Weighted cross-entropy loss — directly aligned with macro-F1 optimisation."""
    return F.cross_entropy(logits, targets, weight=class_weights)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """
    Maps each scalar feature to a d_token-dimensional embedding.

    x_i (scalar) → Linear(1, d_token) + bias_i
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        # One weight vector per feature (shape: n_features × d_token)
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: (B, n_features)
        Returns:
            tokens: (B, n_features, d_token)
        """
        # x: (B, F) → unsqueeze → (B, F, 1) * weight (F, d_token) → (B, F, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerNet(nn.Module):
    """
    Full FT-Transformer network.

    Processes:
        1. Tokenise features
        2. Prepend CLS token
        3. Apply N transformer layers
        4. Extract CLS token
        5. MLP head → logits
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Clamp n_heads to valid divisor of d_token
        n_heads = _valid_n_heads(d_token, n_heads)

        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,         # Pre-LN as in the original paper
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # MLP head: CLS repr → hidden → logits
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: (B, n_features)
        Returns:
            logits: (B, n_classes)
        """
        # Tokenise
        tokens = self.tokenizer(x)           # (B, F, d_token)

        # Prepend CLS
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, F+1, d_token)

        # Transformer
        out = self.transformer(tokens)        # (B, F+1, d_token)

        # Extract CLS output (position 0)
        cls_out = out[:, 0]                   # (B, d_token)

        return self.head(cls_out)             # (B, n_classes)


def _valid_n_heads(d_token: int, n_heads: int) -> int:
    """
    Find largest divisor of d_token that is <= n_heads.
    Ensures d_token % n_heads == 0.
    """
    for h in range(n_heads, 0, -1):
        if d_token % h == 0:
            return h
    return 1


# ---------------------------------------------------------------------------
# Forecaster wrapper
# ---------------------------------------------------------------------------

class FTTransformerForecaster(BaseForecaster):
    """
    Feature Tokenization Transformer forecaster for tabular classification.

    PSO-GWO optimizable hyperparameters (via get_param_space):
        d_token, n_heads, n_layers, dropout, learning_rate, batch_size
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch"
            )
        super().__init__(config)
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': []
        }
        self._net: Optional["FTTransformerNet"] = None
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
    ) -> 'FTTransformerForecaster':
        """Fit the FT-Transformer."""
        X = self.apply_feature_mask(X)
        if X_val is not None:
            X_val = self.apply_feature_mask(X_val)

        self._n_features = X.shape[1]
        self._n_classes = len(np.unique(y))
        self._feature_names = feature_names

        p = self.config.params
        d_token = int(p.get('d_token', 128))
        n_heads = int(p.get('n_heads', 8))
        n_layers = int(p.get('n_layers', 3))
        dropout = float(p.get('dropout', 0.1))
        lr = float(p.get('learning_rate', 0.001))
        batch_size = int(p.get('batch_size', 32))
        max_epochs = int(p.get('max_epochs', 200))
        patience = int(p.get('patience', 20))

        # Device
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build network
        self._net = FTTransformerNet(
            n_features=self._n_features,
            n_classes=self._n_classes,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self._device)

        # Class weights tensor
        class_weight_tensor = self._build_class_weight_tensor()

        # Optimiser + LR schedule
        rng_state = self.config.random_state
        torch.manual_seed(rng_state)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )

        # DataLoaders
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(X_t) % batch_size == 1,
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self._device)
            y_val_t = torch.tensor(y_val, dtype=torch.long).to(self._device)

        # Reset history
        self.training_history = {'train_loss': [], 'val_loss': []}

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            # Training
            self._net.train()
            epoch_train_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad()
                logits = self._net(X_batch)
                loss = weighted_ce(logits, y_batch, class_weights=class_weight_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_train_loss / max(n_batches, 1)
            self.training_history['train_loss'].append(avg_train_loss)

            # Validation
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_logits = self._net(X_val_t)
                    val_loss = weighted_ce(
                        val_logits, y_val_t, class_weights=class_weight_tensor
                    ).item()
                self.training_history['val_loss'].append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.debug(f"FT-Transformer early stop at epoch {epoch + 1}")
                        break
            else:
                self.training_history['val_loss'].append(avg_train_loss)

        # Restore best weights
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        logger.info(
            f"FT-Transformer fitted: {self._n_features} features, "
            f"{self._n_classes} classes, d_token={d_token}, "
            f"n_layers={n_layers}, epochs={epoch + 1}"
        )
        return self

    def _build_class_weight_tensor(self) -> Optional["torch.Tensor"]:
        """Build class weight tensor for loss computation."""
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
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (softmax of logits)."""
        if not self._is_fitted or self._net is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)
        self._net.eval()

        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
            logits = self._net(X_t)
            proba = F.softmax(logits, dim=1).cpu().numpy()

        return proba

    # ------------------------------------------------------------------
    # Complexity
    # ------------------------------------------------------------------

    def get_model_complexity(self) -> Dict[str, Any]:
        """Get model complexity metrics."""
        if not self._is_fitted or self._net is None:
            return {'n_params': 0, 'd_token': 0, 'n_layers': 0}

        n_params = sum(p.numel() for p in self._net.parameters() if p.requires_grad)
        return {
            'n_params': n_params,
            'd_token': self.config.params.get('d_token', 128),
            'n_layers': self.config.params.get('n_layers', 3),
            'n_features': self.get_n_selected_features(),
        }

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
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

        logger.info(f"FT-Transformer model saved to {path}")

    def load(self, path: Union[str, Path]) -> 'FTTransformerForecaster':
        """Load model from disk."""
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

        self._net = FTTransformerNet(
            n_features=self._n_features,
            n_classes=self._n_classes,
            d_token=int(p.get('d_token', 128)),
            n_heads=int(p.get('n_heads', 8)),
            n_layers=int(p.get('n_layers', 3)),
            dropout=float(p.get('dropout', 0.1)),
        ).to(self._device)

        self._net.load_state_dict(
            torch.load(path.with_suffix('.pt'), map_location=self._device)
        )
        self._net.eval()
        self._is_fitted = True

        logger.info(f"FT-Transformer model loaded from {path}")
        return self

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_model_name(cls) -> str:
        return "FT-Transformer"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'd_token': 32,
            'n_heads': 2,
            'n_layers': 1,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 300,
            'patience': 40,
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            'd_token': (16, 64),
            'n_heads': (1, 4),
            'n_layers': (1, 2),
            'dropout': (0.0, 0.3),
            'learning_rate': (1e-4, 1e-2),
            'batch_size': (16, 32),
        }
