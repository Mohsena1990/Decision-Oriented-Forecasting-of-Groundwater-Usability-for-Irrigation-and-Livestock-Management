"""
Pareto Archive Module
=====================

Implements Pareto dominance and archive management for
multi-objective optimization.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """
    Check if solution a dominates solution b (for minimization).

    a dominates b if:
    - a is at least as good as b in all objectives
    - a is strictly better than b in at least one objective

    Args:
        obj_a: Objective values of solution a
        obj_b: Objective values of solution b

    Returns:
        True if a dominates b
    """
    at_least_as_good = np.all(obj_a <= obj_b)
    strictly_better = np.any(obj_a < obj_b)
    return at_least_as_good and strictly_better


def is_non_dominated(
    objectives: np.ndarray,
    archive_objectives: List[np.ndarray]
) -> bool:
    """
    Check if a solution is non-dominated by any solution in the archive.

    Args:
        objectives: Objective values to check
        archive_objectives: List of objective values in archive

    Returns:
        True if non-dominated
    """
    for arch_obj in archive_objectives:
        if dominates(arch_obj, objectives):
            return False
    return True


@dataclass
class ParetoArchive:
    """
    Archive of non-dominated solutions.

    Maintains a set of Pareto-optimal solutions discovered
    during optimization.
    """

    max_size: int = 100
    use_crowding: bool = True

    _positions: List[np.ndarray] = field(default_factory=list, init=False)
    _objectives: List[np.ndarray] = field(default_factory=list, init=False)

    def add(self, position: np.ndarray, objectives: np.ndarray) -> bool:
        """
        Add a solution to the archive if non-dominated.

        Args:
            position: Decision variable vector
            objectives: Objective value vector

        Returns:
            True if added to archive
        """
        # Check if dominated by any existing solution
        if not is_non_dominated(objectives, self._objectives):
            return False

        # Remove solutions dominated by the new one
        indices_to_keep = []
        for i, arch_obj in enumerate(self._objectives):
            if not dominates(objectives, arch_obj):
                indices_to_keep.append(i)

        self._positions = [self._positions[i] for i in indices_to_keep]
        self._objectives = [self._objectives[i] for i in indices_to_keep]

        # Add new solution
        self._positions.append(position.copy())
        self._objectives.append(objectives.copy())

        # Trim if over capacity
        if len(self._positions) > self.max_size:
            self._trim_archive()

        return True

    def _trim_archive(self):
        """Trim archive to max size using crowding distance."""
        if not self.use_crowding or len(self._positions) <= self.max_size:
            return

        # Compute crowding distances
        distances = self._compute_crowding_distances()

        # Keep solutions with highest crowding distance
        indices = np.argsort(distances)[::-1][:self.max_size]

        self._positions = [self._positions[i] for i in indices]
        self._objectives = [self._objectives[i] for i in indices]

    def _compute_crowding_distances(self) -> np.ndarray:
        """Compute crowding distance for each solution."""
        n = len(self._objectives)
        if n == 0:
            return np.array([])

        objectives = np.array(self._objectives)
        n_objectives = objectives.shape[1]

        distances = np.zeros(n)

        for m in range(n_objectives):
            # Sort by objective m
            sorted_indices = np.argsort(objectives[:, m])
            obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]

            if obj_range == 0:
                continue

            # Boundary points get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Interior points
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]

                distances[idx] += (objectives[next_idx, m] - objectives[prev_idx, m]) / obj_range

        return distances

    def get_front(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get the Pareto front as list of (position, objectives) tuples."""
        return list(zip(self._positions, self._objectives))

    def get_best_by_objective(
        self,
        objective_idx: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the solution with best value for a specific objective."""
        if not self._objectives:
            return None

        best_idx = np.argmin([obj[objective_idx] for obj in self._objectives])
        return self._positions[best_idx], self._objectives[best_idx]

    def get_objectives_array(self) -> np.ndarray:
        """Get all objectives as 2D array."""
        if not self._objectives:
            return np.array([]).reshape(0, 0)
        return np.array(self._objectives)

    def get_positions_array(self) -> np.ndarray:
        """Get all positions as 2D array."""
        if not self._positions:
            return np.array([]).reshape(0, 0)
        return np.array(self._positions)

    def __len__(self) -> int:
        return len(self._positions)

    def clear(self):
        """Clear the archive."""
        self._positions = []
        self._objectives = []

    def to_dataframe(self, position_names: Optional[List[str]] = None, objective_names: Optional[List[str]] = None):
        """Convert archive to pandas DataFrame."""
        import pandas as pd

        if not self._positions:
            return pd.DataFrame()

        data = {}

        # Add positions
        positions = self.get_positions_array()
        for i in range(positions.shape[1]):
            name = position_names[i] if position_names and i < len(position_names) else f'x{i}'
            data[name] = positions[:, i]

        # Add objectives
        objectives = self.get_objectives_array()
        for i in range(objectives.shape[1]):
            name = objective_names[i] if objective_names and i < len(objective_names) else f'f{i}'
            data[name] = objectives[:, i]

        return pd.DataFrame(data)


def compute_hypervolume(
    objectives: np.ndarray,
    reference_point: np.ndarray
) -> float:
    """
    Compute hypervolume indicator (approximate for >2 objectives).

    Args:
        objectives: 2D array of objective values (n_solutions, n_objectives)
        reference_point: Reference point (worst possible values)

    Returns:
        Hypervolume value
    """
    if len(objectives) == 0:
        return 0.0

    n_objectives = objectives.shape[1]

    if n_objectives == 2:
        # Exact 2D hypervolume
        return _hypervolume_2d(objectives, reference_point)
    else:
        # Monte Carlo approximation for higher dimensions
        return _hypervolume_monte_carlo(objectives, reference_point, n_samples=10000)


def _hypervolume_2d(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Exact 2D hypervolume calculation."""
    # Sort by first objective
    sorted_idx = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_idx]

    hv = 0.0
    prev_y = reference_point[1]

    for point in sorted_obj:
        if point[0] < reference_point[0] and point[1] < reference_point[1]:
            width = reference_point[0] - point[0]
            height = prev_y - point[1]
            if height > 0:
                hv += width * height
            prev_y = point[1]

    return hv


def _hypervolume_monte_carlo(
    objectives: np.ndarray,
    reference_point: np.ndarray,
    n_samples: int = 10000
) -> float:
    """Monte Carlo hypervolume approximation."""
    rng = np.random.default_rng(42)

    # Sample random points in the hyperbox
    ideal = np.min(objectives, axis=0)
    samples = rng.uniform(ideal, reference_point, size=(n_samples, len(reference_point)))

    # Count dominated samples
    dominated = 0
    for sample in samples:
        for obj in objectives:
            if np.all(obj <= sample):
                dominated += 1
                break

    # Estimate hypervolume
    box_volume = np.prod(reference_point - ideal)
    return box_volume * dominated / n_samples
