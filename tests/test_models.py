"""
Tests for model implementations.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import ForecasterConfig, compute_class_weights


class TestForecasterConfig:
    """Tests for ForecasterConfig."""

    def test_config_creation(self):
        """Test creating a config."""
        config = ForecasterConfig(
            params={'learning_rate': 0.1, 'depth': 6},
            n_classes=5,
            random_state=42
        )

        assert config.get_param('learning_rate') == 0.1
        assert config.get_param('depth') == 6
        assert config.get_param('missing', default=0) == 0

    def test_config_serialization(self):
        """Test config to/from dict."""
        config = ForecasterConfig(
            params={'lr': 0.1},
            n_classes=3,
            feature_mask=np.array([1, 0, 1, 1])
        )

        d = config.to_dict()
        restored = ForecasterConfig.from_dict(d)

        assert restored.params == config.params
        assert restored.n_classes == config.n_classes
        np.testing.assert_array_equal(restored.feature_mask, config.feature_mask)


class TestClassWeights:
    """Tests for class weight computation."""

    def test_balanced_weights(self):
        """Test balanced class weights."""
        y = np.array([0, 0, 0, 0, 1, 1, 2])  # Imbalanced

        weights = compute_class_weights(y, method='balanced')

        # More weight for minority classes
        assert weights[2] > weights[0]
        assert weights[1] > weights[0]

    def test_equal_classes(self):
        """Test weights with equal classes."""
        y = np.array([0, 0, 1, 1, 2, 2])

        weights = compute_class_weights(y, method='balanced')

        # All weights should be equal
        assert abs(weights[0] - weights[1]) < 0.01
        assert abs(weights[1] - weights[2]) < 0.01


class TestModelInterface:
    """Tests for model interface compliance."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        return X, y

    def test_catboost_interface(self, sample_data):
        """Test CatBoost follows interface."""
        try:
            from src.models.trees import CatBoostForecaster

            X, y = sample_data
            config = ForecasterConfig(
                params={'iterations': 10, 'depth': 3},
                n_classes=3
            )

            model = CatBoostForecaster(config)
            model.fit(X, y)

            # Check predictions
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)

            assert y_pred.shape == (100,)
            assert y_proba.shape == (100, 3)
            assert np.allclose(y_proba.sum(axis=1), 1.0)

            # Check complexity
            complexity = model.get_model_complexity()
            assert 'n_trees' in complexity

        except ImportError:
            pytest.skip("CatBoost not installed")

    def test_lightgbm_interface(self, sample_data):
        """Test LightGBM follows interface."""
        try:
            from src.models.trees import LightGBMForecaster

            X, y = sample_data
            config = ForecasterConfig(
                params={'n_estimators': 10, 'max_depth': 3},
                n_classes=3
            )

            model = LightGBMForecaster(config)
            model.fit(X, y)

            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)

            assert y_pred.shape == (100,)
            assert y_proba.shape == (100, 3)

        except ImportError:
            pytest.skip("LightGBM not installed")

    def test_feature_mask_application(self, sample_data):
        """Test feature mask is applied correctly."""
        from src.models.base import BaseForecaster

        X, y = sample_data
        config = ForecasterConfig(
            feature_mask=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        )

        class DummyForecaster(BaseForecaster):
            def fit(self, X, y, **kwargs): return self
            def predict(self, X): return np.zeros(len(X))
            def predict_proba(self, X): return np.zeros((len(X), 2))
            def get_model_complexity(self): return {}
            def save(self, path): pass
            def load(self, path): return self
            @classmethod
            def get_model_name(cls): return "Dummy"
            @classmethod
            def get_default_params(cls): return {}
            @classmethod
            def get_param_space(cls): return {}

        model = DummyForecaster(config)
        X_masked = model.apply_feature_mask(X)

        assert X_masked.shape == (100, 5)  # Only 5 features selected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
