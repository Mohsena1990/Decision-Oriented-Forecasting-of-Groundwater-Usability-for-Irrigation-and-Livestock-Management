"""
LSTM Forecaster Module
======================

LSTM-based neural network for groundwater quality forecasting.
Similar architecture to GRU with LSTM cells.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..base import BaseForecaster, ForecasterConfig

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class LSTMNetwork(nn.Module):
        """LSTM-based network with categorical embeddings."""

        def __init__(
            self,
            n_numeric: int,
            n_classes: int,
            categorical_cardinalities: Dict[int, int],
            hidden_size: int = 64,
            num_layers: int = 1,
            dropout: float = 0.2,
            embedding_dim: int = 16
        ):
            super().__init__()

            self.n_numeric = n_numeric
            self.categorical_cardinalities = categorical_cardinalities

            # Categorical embeddings
            self.embeddings = nn.ModuleDict()
            total_embedding_dim = 0

            for idx, cardinality in categorical_cardinalities.items():
                emb_dim = min(embedding_dim, cardinality // 2 + 1)
                self.embeddings[str(idx)] = nn.Embedding(cardinality + 1, emb_dim)
                total_embedding_dim += emb_dim

            # Input size
            input_size = n_numeric + total_embedding_dim

            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

            # Output layers
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, n_classes)

        def forward(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
            batch_size = x_numeric.size(0)

            # Process categorical features
            embedded = []
            for i, (idx, _) in enumerate(self.categorical_cardinalities.items()):
                cat_values = x_categorical[:, i].long()
                emb = self.embeddings[str(idx)](cat_values)
                embedded.append(emb)

            # Concatenate features
            if embedded:
                cat_embedded = torch.cat(embedded, dim=1)
                x = torch.cat([x_numeric, cat_embedded], dim=1)
            else:
                x = x_numeric

            # Add sequence dimension
            x = x.unsqueeze(1)

            # LSTM forward
            output, (h_n, c_n) = self.lstm(x)

            # Take last hidden state
            output = h_n[-1]

            # Classification head
            output = self.dropout(output)
            output = self.fc(output)

            return output


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based forecaster for groundwater quality.

    Features:
    - Categorical embeddings for location features
    - Configurable architecture
    - Early stopping support
    """

    def __init__(self, config: Optional[ForecasterConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        super().__init__(config)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._categorical_cardinalities: Dict[int, int] = {}
        self._n_numeric: int = 0
        self._network: Optional[nn.Module] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None
    ) -> 'LSTMForecaster':
        """Fit the LSTM model."""
        # Apply feature mask
        X = self.apply_feature_mask(X)
        if X_val is not None:
            X_val = self.apply_feature_mask(X_val)

        self._n_features = X.shape[1]
        self._n_classes = len(np.unique(y))
        self._feature_names = feature_names

        # Determine categorical features
        categorical_features = categorical_features or []
        self._categorical_cardinalities = {}

        for idx in categorical_features:
            if idx < X.shape[1]:
                cardinality = int(np.max(X[:, idx])) + 1
                self._categorical_cardinalities[idx] = cardinality

        # Split features
        numeric_indices = [i for i in range(X.shape[1]) if i not in categorical_features]
        self._n_numeric = len(numeric_indices)

        # Prepare data
        X_numeric = X[:, numeric_indices].astype(np.float32)
        X_categorical = X[:, categorical_features].astype(np.int64) if categorical_features else np.zeros((len(X), 0), dtype=np.int64)

        # Create network
        self._network = LSTMNetwork(
            n_numeric=self._n_numeric,
            n_classes=self._n_classes,
            categorical_cardinalities=self._categorical_cardinalities,
            hidden_size=self.config.params.get('hidden_size', 64),
            num_layers=self.config.params.get('num_layers', 1),
            dropout=self.config.params.get('dropout', 0.2),
            embedding_dim=self.config.params.get('embedding_dim', 16)
        ).to(self._device)

        # Training setup
        criterion = nn.CrossEntropyLoss(
            weight=self._get_class_weights(y) if self.config.class_weights else None
        )
        optimizer = optim.Adam(
            self._network.parameters(),
            lr=self.config.params.get('learning_rate', 0.001)
        )

        # Create data loaders
        batch_size = self.config.params.get('batch_size', 32)
        train_dataset = TensorDataset(
            torch.from_numpy(X_numeric),
            torch.from_numpy(X_categorical),
            torch.from_numpy(y.astype(np.int64))
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_numeric = X_val[:, numeric_indices].astype(np.float32)
            X_val_categorical = X_val[:, categorical_features].astype(np.int64) if categorical_features else np.zeros((len(X_val), 0), dtype=np.int64)

            val_dataset = TensorDataset(
                torch.from_numpy(X_val_numeric),
                torch.from_numpy(X_val_categorical),
                torch.from_numpy(y_val.astype(np.int64))
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        max_epochs = self.config.params.get('max_epochs', 200)
        patience = self.config.params.get('patience', 20)
        best_val_loss = float('inf')
        patience_counter = 0
        self.training_history: Dict[str, list] = {'train_loss': [], 'val_loss': []}

        for epoch in range(max_epochs):
            self._network.train()
            train_loss = 0.0

            for x_num, x_cat, targets in train_loader:
                x_num = x_num.to(self._device)
                x_cat = x_cat.to(self._device)
                targets = targets.to(self._device)

                optimizer.zero_grad()
                outputs = self._network(x_num, x_cat)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.training_history['train_loss'].append(train_loss / len(train_loader))

            # Validation
            if val_loader is not None:
                self._network.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for x_num, x_cat, targets in val_loader:
                        x_num = x_num.to(self._device)
                        x_cat = x_cat.to(self._device)
                        targets = targets.to(self._device)

                        outputs = self._network(x_num, x_cat)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                self.training_history['val_loss'].append(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

        self._is_fitted = True
        self._numeric_indices = numeric_indices
        self._categorical_indices = categorical_features
        logger.info(f"LSTM fitted with {self._n_features} features, {self._n_classes} classes")

        return self

    def _get_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute class weights tensor."""
        if self.config.class_weights:
            weights = [self.config.class_weights.get(i, 1.0) for i in range(self._n_classes)]
            return torch.tensor(weights, dtype=torch.float32).to(self._device)
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.apply_feature_mask(X)

        X_numeric = X[:, self._numeric_indices].astype(np.float32)
        X_categorical = X[:, self._categorical_indices].astype(np.int64) if self._categorical_indices else np.zeros((len(X), 0), dtype=np.int64)

        self._network.eval()
        with torch.no_grad():
            x_num = torch.from_numpy(X_numeric).to(self._device)
            x_cat = torch.from_numpy(X_categorical).to(self._device)

            outputs = self._network(x_num, x_cat)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def get_model_complexity(self) -> Dict[str, Any]:
        """Get model complexity metrics."""
        if not self._is_fitted:
            return {'n_params': 0}

        n_params = sum(p.numel() for p in self._network.parameters())
        return {
            'n_params': n_params,
            'hidden_size': self.config.params.get('hidden_size', 64),
            'num_layers': self.config.params.get('num_layers', 1),
            'n_features': self.get_n_selected_features()
        }

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self._network.state_dict(), path.with_suffix('.pth'))

        with open(path.with_suffix('.config.pkl'), 'wb') as f:
            pickle.dump({
                'config': self.config,
                'n_features': self._n_features,
                'n_classes': self._n_classes,
                'n_numeric': self._n_numeric,
                'categorical_cardinalities': self._categorical_cardinalities,
                'numeric_indices': self._numeric_indices,
                'categorical_indices': self._categorical_indices,
                'feature_names': self._feature_names
            }, f)

        logger.info(f"LSTM model saved to {path}")

    def load(self, path: Union[str, Path]) -> 'LSTMForecaster':
        """Load model from disk."""
        path = Path(path)

        with open(path.with_suffix('.config.pkl'), 'rb') as f:
            state = pickle.load(f)

        self.config = state['config']
        self._n_features = state['n_features']
        self._n_classes = state['n_classes']
        self._n_numeric = state['n_numeric']
        self._categorical_cardinalities = state['categorical_cardinalities']
        self._numeric_indices = state['numeric_indices']
        self._categorical_indices = state['categorical_indices']
        self._feature_names = state['feature_names']

        self._network = LSTMNetwork(
            n_numeric=self._n_numeric,
            n_classes=self._n_classes,
            categorical_cardinalities=self._categorical_cardinalities,
            hidden_size=self.config.params.get('hidden_size', 64),
            num_layers=self.config.params.get('num_layers', 1),
            dropout=self.config.params.get('dropout', 0.2),
            embedding_dim=self.config.params.get('embedding_dim', 16)
        ).to(self._device)

        self._network.load_state_dict(torch.load(path.with_suffix('.pth'), map_location=self._device))
        self._is_fitted = True

        logger.info(f"LSTM model loaded from {path}")
        return self

    @classmethod
    def get_model_name(cls) -> str:
        return "LSTM"

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'embedding_dim': 16,
            'max_epochs': 200,
            'patience': 20
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, Tuple[Any, Any]]:
        return {
            'hidden_size': (32, 128),
            'num_layers': (1, 2),
            'dropout': (0.1, 0.4),
            'learning_rate': (0.0001, 0.01),
            'batch_size': (16, 64),
            'embedding_dim': (8, 32)
        }
