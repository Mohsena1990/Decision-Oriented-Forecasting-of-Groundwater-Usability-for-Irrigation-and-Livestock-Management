"""
Tests for optimization and decision-making modules.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.pareto import ParetoArchive, dominates
from src.decision.vikor import VIKOR


class TestParetoDominance:
    """Tests for Pareto dominance."""

    def test_dominates_basic(self):
        """Test basic dominance relationship."""
        # a dominates b (better in all objectives)
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 3.0])
        assert dominates(a, b) == True
        assert dominates(b, a) == False

    def test_non_dominated(self):
        """Test non-dominated solutions."""
        # Neither dominates the other (trade-off)
        a = np.array([1.0, 3.0])
        b = np.array([2.0, 2.0])
        assert dominates(a, b) == False
        assert dominates(b, a) == False

    def test_equal_solutions(self):
        """Test equal solutions don't dominate."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        assert dominates(a, b) == False
        assert dominates(b, a) == False


class TestParetoArchive:
    """Tests for ParetoArchive."""

    def setup_method(self):
        self.archive = ParetoArchive(max_size=10)

    def test_add_non_dominated(self):
        """Test adding non-dominated solutions."""
        pos1 = np.array([0.5, 0.5])
        obj1 = np.array([1.0, 3.0])
        self.archive.add(pos1, obj1)

        pos2 = np.array([0.3, 0.7])
        obj2 = np.array([2.0, 2.0])
        self.archive.add(pos2, obj2)

        assert len(self.archive) == 2

    def test_remove_dominated(self):
        """Test that dominated solutions are removed."""
        # Add a solution
        pos1 = np.array([0.5, 0.5])
        obj1 = np.array([3.0, 3.0])
        self.archive.add(pos1, obj1)

        # Add a better solution that dominates the first
        pos2 = np.array([0.3, 0.7])
        obj2 = np.array([1.0, 1.0])
        self.archive.add(pos2, obj2)

        # Only the dominating solution should remain
        assert len(self.archive) == 1

    def test_get_best(self):
        """Test getting best by objective."""
        self.archive.add(np.array([0.1]), np.array([1.0, 3.0]))
        self.archive.add(np.array([0.2]), np.array([2.0, 1.0]))

        # Best by first objective
        pos, obj = self.archive.get_best_by_objective(0)
        assert obj[0] == 1.0

        # Best by second objective
        pos, obj = self.archive.get_best_by_objective(1)
        assert obj[1] == 1.0


class TestVIKOR:
    """Tests for VIKOR decision-making."""

    def setup_method(self):
        self.vikor = VIKOR(v=0.5)

    def test_rank_alternatives(self):
        """Test ranking alternatives."""
        # 4 alternatives, 3 criteria (all minimize)
        objectives = np.array([
            [0.2, 0.3, 0.5],  # Alt 0
            [0.3, 0.2, 0.4],  # Alt 1
            [0.4, 0.4, 0.3],  # Alt 2
            [0.5, 0.5, 0.2],  # Alt 3
        ])

        weights = np.array([0.4, 0.3, 0.3])

        result = self.vikor.rank(objectives, weights)

        # Check result structure
        assert len(result.rankings) == 4
        assert len(result.Q) == 4
        assert len(result.S) == 4
        assert len(result.R) == 4
        assert result.best_idx in [0, 1, 2, 3]

    def test_equal_weights(self):
        """Test with equal weights."""
        objectives = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 1.5],
        ])

        result = self.vikor.rank(objectives)

        # All should have finite Q values
        assert np.all(np.isfinite(result.Q))

    def test_compromise_set(self):
        """Test compromise set identification."""
        objectives = np.array([
            [1.0, 4.0],
            [1.1, 3.9],  # Close to best
            [4.0, 1.0],
        ])

        result = self.vikor.rank(objectives)

        # Compromise set should include close alternatives
        assert result.best_idx in result.compromise_set
        assert len(result.compromise_set) >= 1


class TestObjectiveCalculator:
    """Tests for objective calculation."""

    def test_ordinal_distance(self):
        """Test ordinal distance calculation."""
        from src.objectives import ObjectiveCalculator
        from src.objectives.definitions import ORDINAL_ENCODING

        calc = ObjectiveCalculator(
            label_to_ordinal=ORDINAL_ENCODING,
            idx_to_label={0: 'C2S1', 1: 'C3S2', 2: 'C4S1'},
            high_risk_indices=[2]
        )

        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 0, 1])  # Some mismatches

        objectives = calc.compute_objectives(y_true, y_pred, n_features=5)

        assert len(objectives) == 4
        assert objectives[0] >= 0  # Ordinal distance
        assert 0 <= objectives[1] <= 1  # Severe FNR
        assert 0 <= objectives[2] <= 1  # 1 - MacroF1
        assert objectives[3] >= 0  # Complexity


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
