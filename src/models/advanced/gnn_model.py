"""
Spatial Graph Neural Network (SpatialGNN)
==========================================

K-nearest-neighbor spatial graph from lat/lon coordinates,
followed by 2-hop mean aggregation (simplified GraphSAGE style)
and an MLP classifier.

No torch_geometric dependency — uses scipy + numpy for graph construction
and pure PyTorch for the learnable MLP.

Design:
    1. fit(X, y):
       - Extract lat/lon columns from feature_names (looks for 'lat_gis' / 'long_gis')
       - Build k-NN spatial adjacency on training nodes
       - Aggregate: for each node i, compute mean of neighbor features
       - Concatenate [X_i, mean_neighbor_X_i] → double-width feature vector
       - Train MLP on concatenated features with focal loss + early stopping
       - Store training coordinates and features for inductive inference

    2. predict_proba(X_test):
       - For each test node, find k nearest training nodes by lat/lon
       - Aggregate their features → concatenate → MLP forward pass

    Fallback: if lat/lon columns not found, operates as a plain MLP on X.
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

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..base import BaseForecaster, ForecasterConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal loss (same helper as ft_transformer)
# ---------------------------------------------------------------------------

def weighted_ce(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    class_weights: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Weighted cross-entropy — directly aligned with macro-F1 optimisation."""
    return F.cross_entropy(logits, targets, weight=class_weights)


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple ReLU MLP with dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        layers: List["nn.Module"] = []
        prev_dim = in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------

class SpatialGNNForecaster(BaseForecaster):
    """
    Spatial GNN forecaster using k-NN graph + mean aggregation + MLP.

    PSO-GWO optimizable hyperparameters:
        k_neighbors, hidden_dim, n_layers, dropout, learning_rate
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        super().__init__(config)

        self.training_history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': []
        }
        self._net: Optional["MLP"] = None
        self._device: Optional["torch.device"] = None

        # Stored training data for inductive inference
        self._train_coords: Optional[np.ndarray] = None   # (N, 2) lat/lon
        self._train_X: Optional[np.ndarray] = None        # (N, n_features)

        # Indices of lat/lon in feature vector (-1 = not found)
        self._lat_idx: int = -1
        self._lon_idx: int = -1
        self._spatial_available: bool = False

        # kd-tree for fast neighbor lookup
        self._kdtree: Optional["cKDTree"] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_spatial_indices(self, feature_names: Optional[List[str]]) -> Tuple[int, int]:
        """Locate lat/lon column indices in feature_names."""
        lat_candidates = ['lat_gis', 'lat', 'latitude', 'Lat_GIS', 'LAT']
        lon_candidates = ['long_gis', 'lon', 'longitude', 'Long_GIS', 'LON', 'lng']

        if feature_names is None:
            return -1, -1

        lat_idx = -1
        lon_idx = -1

        for name in lat_candidates:
            if name in feature_names:
                lat_idx = feature_names.index(name)
                break

        for name in lon_candidates:
            if name in feature_names:
                lon_idx = feature_names.index(name)
                break

        return lat_idx, lon_idx

    def _aggregate_neighbors(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        ref_coords: np.ndarray,
        ref_X: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        For each row in coords, find k nearest neighbors in ref_coords and
        return the mean of their ref_X feature vectors.

        Args:
            X: (N, F) query features (unused for aggregation but needed for shape)
            coords: (N, 2) query lat/lon
            ref_coords: (M, 2) reference lat/lon (training nodes)
            ref_X: (M, F) reference features
            k: number of neighbors

        Returns:
            agg: (N, F) aggregated neighbor features
        """
        if not SCIPY_AVAILABLE:
            # Fallback: brute force
            agg = np.zeros((len(coords), ref_X.shape[1]), dtype=np.float32)
            for i, c in enumerate(coords):
                dists = np.sum((ref_coords - c) ** 2, axis=1)
                nbr_idx = np.argsort(dists)[:k]
                agg[i] = ref_X[nbr_idx].mean(axis=0)
            return agg

        tree = cKDTree(ref_coords)
        k_eff = min(k, len(ref_coords))
        _, idx = tree.query(coords, k=k_eff, workers=-1)
        if k_eff == 1:
            idx = idx[:, np.newaxis]
        agg = ref_X[idx].mean(axis=1)      # (N, F)
        return agg.astype(np.float32)

    def _build_augmented_features(
        self, X: np.ndarray, agg: np.ndarray
    ) -> np.ndarray:
        """Concatenate original features with aggregated neighbor features."""
        return np.concatenate([X, agg], axis=1).astype(np.float32)

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
    ) -> 'SpatialGNNForecaster':
        """Fit the Spatial GNN."""
        X = self.apply_feature_mask(X)
        if X_val is not None:
            X_val = self.apply_feature_mask(X_val)

        self._n_features = X.shape[1]
        self._n_classes = len(np.unique(y))
        self._feature_names = feature_names

        p = self.config.params
        k = int(p.get('k_neighbors', 5))
        hidden_dim = int(p.get('hidden_dim', 64))
        n_layers = int(p.get('n_layers', 2))
        dropout = float(p.get('dropout', 0.1))
        lr = float(p.get('learning_rate', 0.001))
        max_epochs = int(p.get('max_epochs', 200))
        patience = int(p.get('patience', 20))
        batch_size = 32

        # Detect spatial columns
        self._lat_idx, self._lon_idx = self._find_spatial_indices(
            feature_names or self.config.params.get('feature_names')
        )
        self._spatial_available = (self._lat_idx >= 0 and self._lon_idx >= 0)

        if not self._spatial_available:
            logger.warning(
                "SpatialGNN: lat/lon columns not found in feature_names. "
                "Falling back to plain MLP without spatial aggregation."
            )

        # Build aggregated features
        if self._spatial_available:
            coords = np.column_stack([X[:, self._lat_idx], X[:, self._lon_idx]])
            self._train_coords = coords.astype(np.float32)
            self._train_X = X.astype(np.float32)

            if SCIPY_AVAILABLE:
                self._kdtree = cKDTree(self._train_coords)

            agg_train = self._aggregate_neighbors(X, coords, self._train_coords, self._train_X, k)
            X_aug = self._build_augmented_features(X, agg_train)

            if X_val is not None:
                val_coords = np.column_stack([X_val[:, self._lat_idx], X_val[:, self._lon_idx]])
                agg_val = self._aggregate_neighbors(X_val, val_coords, self._train_coords, self._train_X, k)
                X_val_aug = self._build_augmented_features(X_val, agg_val)
            else:
                X_val_aug = None
        else:
            X_aug = X.astype(np.float32)
            X_val_aug = X_val.astype(np.float32) if X_val is not None else None

        in_dim = X_aug.shape[1]

        # Device
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(self.config.random_state)

        # Build MLP
        self._net = MLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=self._n_classes,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self._device)

        # Class weights
        class_weight_tensor = self._build_class_weight_tensor()

        # Optimiser
        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )

        X_t = torch.tensor(X_aug, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
            drop_last=len(X_t) % batch_size == 1
        )

        has_val = X_val_aug is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val_aug, dtype=torch.float32).to(self._device)
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
                logits = self._net(X_b)
                loss = weighted_ce(logits, y_b, class_weights=class_weight_tensor)
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
                    val_logits = self._net(X_val_t)
                    val_loss = weighted_ce(val_logits, y_val_t,
                                          class_weights=class_weight_tensor).item()
                self.training_history['val_loss'].append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k_: v_.cpu().clone()
                                  for k_, v_ in self._net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.debug(f"SpatialGNN early stop at epoch {epoch + 1}")
                        break
            else:
                self.training_history['val_loss'].append(avg_train)

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        logger.info(
            f"SpatialGNN fitted: {self._n_features} features, "
            f"{self._n_classes} classes, k={k}, "
            f"spatial={'yes' if self._spatial_available else 'no (plain MLP)'}"
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
        """Inductive prediction: aggregate train neighbors, then MLP."""
        if not self._is_fitted or self._net is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)
        X = X.astype(np.float32)

        p = self.config.params
        k = int(p.get('k_neighbors', 5))

        if self._spatial_available and self._train_coords is not None:
            test_coords = np.column_stack([X[:, self._lat_idx], X[:, self._lon_idx]])
            agg = self._aggregate_neighbors(
                X, test_coords, self._train_coords, self._train_X, k
            )
            X_aug = self._build_augmented_features(X, agg)
        else:
            X_aug = X

        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_aug, dtype=torch.float32).to(self._device)
            logits = self._net(X_t)
            proba = F.softmax(logits, dim=1).cpu().numpy()

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
            'k_neighbors': self.config.params.get('k_neighbors', 5),
            'hidden_dim': self.config.params.get('hidden_dim', 64),
            'n_layers': self.config.params.get('n_layers', 2),
            'n_features': self.get_n_selected_features(),
            'spatial': self._spatial_available,
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
                'lat_idx': self._lat_idx,
                'lon_idx': self._lon_idx,
                'spatial_available': self._spatial_available,
                'train_coords': self._train_coords,
                'train_X': self._train_X,
                'training_history': self.training_history,
                'net_in_dim': self._net.net[0].in_features if self._net else 0,
            }, f)

        logger.info(f"SpatialGNN model saved to {path}")

    def load(self, path: Union[str, Path]) -> 'SpatialGNNForecaster':
        path = Path(path)

        with open(path.with_suffix('.config.pkl'), 'rb') as f:
            state = pickle.load(f)

        self.config = state['config']
        self._n_features = state['n_features']
        self._n_classes = state['n_classes']
        self._feature_names = state['feature_names']
        self._lat_idx = state['lat_idx']
        self._lon_idx = state['lon_idx']
        self._spatial_available = state['spatial_available']
        self._train_coords = state['train_coords']
        self._train_X = state['train_X']
        self.training_history = state.get('training_history', {'train_loss': [], 'val_loss': []})

        if self._train_coords is not None and SCIPY_AVAILABLE:
            self._kdtree = cKDTree(self._train_coords)

        p = self.config.params
        in_dim = state.get('net_in_dim', self._n_features * 2 if self._spatial_available else self._n_features)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._net = MLP(
            in_dim=in_dim,
            hidden_dim=int(p.get('hidden_dim', 64)),
            out_dim=self._n_classes,
            n_layers=int(p.get('n_layers', 2)),
            dropout=float(p.get('dropout', 0.1)),
        ).to(self._device)

        self._net.load_state_dict(
            torch.load(path.with_suffix('.pt'), map_location=self._device)
        )
        self._net.eval()
        self._is_fitted = True

        logger.info(f"SpatialGNN model loaded from {path}")
        return self

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_model_name(cls) -> str:
        return "SpatialGNN"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'k_neighbors': 5,
            'hidden_dim': 32,
            'n_layers': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'max_epochs': 300,
            'patience': 40,
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            'k_neighbors': (3, 10),
            'hidden_dim': (16, 64),
            'n_layers': (1, 2),
            'dropout': (0.0, 0.3),
            'learning_rate': (1e-4, 1e-2),
        }
