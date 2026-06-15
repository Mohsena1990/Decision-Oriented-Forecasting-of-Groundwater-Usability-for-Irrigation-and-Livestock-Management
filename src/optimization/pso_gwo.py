"""
PSO-GWO Hybrid Optimization Module
==================================

Implements hybrid Particle Swarm Optimization - Grey Wolf Optimizer
for hyperparameter and feature selection optimization.
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .pareto import ParetoArchive, dominates

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_position: np.ndarray
    best_objectives: np.ndarray
    pareto_front: List[Tuple[np.ndarray, np.ndarray]]
    history: List[Dict[str, Any]]
    n_evaluations: int


@dataclass
class Particle:
    """Particle in the swarm."""
    position: np.ndarray
    velocity: np.ndarray
    personal_best_position: np.ndarray
    personal_best_objectives: np.ndarray


@dataclass
class PSOGWO:
    """
    Hybrid PSO-GWO optimizer for multi-objective optimization.

    Combines exploration capabilities of PSO with exploitation
    of Grey Wolf Optimizer for balanced search.
    """

    # Problem dimensions
    n_dimensions: int = 0
    bounds_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    bounds_upper: np.ndarray = field(default_factory=lambda: np.array([]))

    # Algorithm parameters
    population_size: int = 30
    max_iterations: int = 50

    # PSO parameters
    w: float = 0.7  # Inertia weight
    c1: float = 1.5  # Cognitive parameter
    c2: float = 1.5  # Social parameter

    # GWO parameters
    a_start: float = 2.0
    a_end: float = 0.0

    # Hybrid weight (0=pure PSO, 1=pure GWO)
    hybrid_weight: float = 0.5

    # Parallel workers for particle evaluation (threading backend — GIL released
    # by numpy/sklearn/PyTorch during compute, so threads give real concurrency)
    n_workers: int = 1

    # Random state
    random_state: int = 42

    # Internal state
    _rng: np.random.Generator = field(default=None, init=False)
    _particles: List[Particle] = field(default_factory=list, init=False)
    _pareto_archive: ParetoArchive = field(default=None, init=False)
    _iteration: int = field(default=0, init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.random_state)
        self._pareto_archive = ParetoArchive(max_size=100)

    def initialize(
        self,
        bounds_lower: np.ndarray,
        bounds_upper: np.ndarray,
        warm_start: Optional[np.ndarray] = None
    ):
        """
        Initialize the swarm.

        Args:
            bounds_lower: Lower bounds for each dimension
            bounds_upper: Upper bounds for each dimension
            warm_start: Optional initial positions (e.g., from filter selection)
        """
        self.bounds_lower = bounds_lower
        self.bounds_upper = bounds_upper
        self.n_dimensions = len(bounds_lower)

        self._particles = []

        for i in range(self.population_size):
            if warm_start is not None and i < len(warm_start):
                # Use warm start position
                position = np.clip(warm_start[i], bounds_lower, bounds_upper)
            else:
                # Random initialization
                position = self._rng.uniform(bounds_lower, bounds_upper)

            velocity = self._rng.uniform(
                -(bounds_upper - bounds_lower) * 0.1,
                (bounds_upper - bounds_lower) * 0.1
            )

            particle = Particle(
                position=position,
                velocity=velocity,
                personal_best_position=position.copy(),
                personal_best_objectives=np.array([np.inf])  # Will be updated
            )
            self._particles.append(particle)

        self._iteration = 0
        logger.info(f"Initialized swarm with {self.population_size} particles")

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray],
        verbose: bool = True,
        checkpoint_path: Optional[str] = None,
        n_features: int = 0,
    ) -> OptimizationResult:
        """
        Run the optimization.

        Args:
            objective_fn: Function that takes position and returns objective vector
            verbose: Whether to log progress
            checkpoint_path: Path to save/resume checkpoint pickle (None = no checkpointing)

        Returns:
            OptimizationResult with best solutions
        """
        history: List[Dict[str, Any]] = []
        n_evaluations = 0
        start_iteration = 0

        # --- Resume from checkpoint if one exists ---
        if checkpoint_path and Path(checkpoint_path).exists():
            state = self._load_checkpoint(checkpoint_path, n_features=n_features)
            if state is not None:
                self._particles = state['particles']
                self._pareto_archive = state['pareto_archive']
                history = state['history']
                n_evaluations = state['n_evaluations']
                start_iteration = state['start_from_iteration']
                rng_state = state.get('rng_state')
                if rng_state is not None:
                    try:
                        self._rng.bit_generator.state = rng_state
                    except Exception:
                        pass
                logger.info(f"Resumed from checkpoint — starting at iteration {start_iteration} "
                            f"({n_evaluations} evaluations already done)")

        # --- Evaluate initial population (skipped when resuming past init) ---
        if start_iteration == 0:
            logger.info(f"Evaluating initial population ({self.population_size} particles)...")
            init_positions = [p.position for p in self._particles]
            objectives_init = self._eval_parallel(objective_fn, init_positions)
            n_evaluations += len(init_positions)
            for i, (particle, objectives) in enumerate(zip(self._particles, objectives_init)):
                particle.personal_best_objectives = objectives.copy()
                self._pareto_archive.add(particle.position.copy(), objectives.copy())
                if verbose and (i + 1) % 5 == 0:
                    logger.info(f"  Initial evaluation: {i + 1}/{self.population_size} particles done")
            if checkpoint_path:
                self._save_checkpoint(checkpoint_path, 0, history, n_evaluations, n_features)

        # --- Main optimization loop ---
        logger.info(f"Starting optimization ({self.max_iterations} iterations)...")
        for iteration in range(start_iteration, self.max_iterations):
            self._iteration = iteration

            if verbose:
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations} — "
                            f"computing {self.population_size} particle positions...")

            a = self.a_start - (self.a_start - self.a_end) * (iteration / self.max_iterations)
            w = self.w * (1 - 0.5 * iteration / self.max_iterations)

            alpha, beta, gamma = self._get_wolves()

            # Phase 1: compute all new positions sequentially (uses _rng — not thread-safe)
            updates: List[Tuple[np.ndarray, np.ndarray]] = []
            for particle in self._particles:
                r1 = self._rng.random(self.n_dimensions)
                r2 = self._rng.random(self.n_dimensions)
                cognitive = self.c1 * r1 * (particle.personal_best_position - particle.position)
                social = self.c2 * r2 * (alpha - particle.position)
                pso_velocity = w * particle.velocity + cognitive + social

                A1 = 2 * a * self._rng.random(self.n_dimensions) - a
                A2 = 2 * a * self._rng.random(self.n_dimensions) - a
                A3 = 2 * a * self._rng.random(self.n_dimensions) - a
                C1 = 2 * self._rng.random(self.n_dimensions)
                C2 = 2 * self._rng.random(self.n_dimensions)
                C3 = 2 * self._rng.random(self.n_dimensions)

                D_alpha = np.abs(C1 * alpha - particle.position)
                D_beta = np.abs(C2 * beta - particle.position)
                D_gamma = np.abs(C3 * gamma - particle.position)
                gwo_position = (alpha - A1 * D_alpha + beta - A2 * D_beta + gamma - A3 * D_gamma) / 3

                new_velocity = (
                    (1 - self.hybrid_weight) * pso_velocity
                    + self.hybrid_weight * (gwo_position - particle.position)
                )
                new_position = np.clip(
                    particle.position + new_velocity, self.bounds_lower, self.bounds_upper
                )
                updates.append((new_velocity, new_position))

            # Phase 2: evaluate all new positions in parallel
            new_positions = [pos for _, pos in updates]
            objectives_list = self._eval_parallel(objective_fn, new_positions)
            n_evaluations += len(new_positions)

            if verbose:
                logger.info(f"  Iteration {iteration + 1}: {len(new_positions)} evaluations complete")

            # Phase 3: update particles with results
            for particle, (new_velocity, new_position), objectives in zip(
                self._particles, updates, objectives_list
            ):
                particle.velocity = new_velocity
                particle.position = new_position

                if dominates(objectives, particle.personal_best_objectives):
                    particle.personal_best_position = particle.position.copy()
                    particle.personal_best_objectives = objectives.copy()
                elif not dominates(particle.personal_best_objectives, objectives):
                    if self._rng.random() < 0.3:
                        particle.personal_best_position = particle.position.copy()
                        particle.personal_best_objectives = objectives.copy()

                self._pareto_archive.add(particle.position.copy(), objectives.copy())

            best = self._pareto_archive.get_best_by_objective(0)
            if best:
                history.append({
                    'iteration': iteration,
                    'best_objectives': best[1].copy(),
                    'archive_size': len(self._pareto_archive),
                    'n_evaluations': n_evaluations,
                })
                if verbose:
                    logger.info(
                        f"Iteration {iteration + 1}/{self.max_iterations} complete: "
                        f"Best obj = {best[1][0]:.4f}, Archive size = {len(self._pareto_archive)}"
                    )

            # Checkpoint after each completed iteration
            if checkpoint_path:
                self._save_checkpoint(checkpoint_path, iteration + 1, history, n_evaluations, n_features)

        best = self._pareto_archive.get_best_by_objective(0)
        if best:
            best_position, best_objectives = best
        else:
            best_position = self._particles[0].position
            best_objectives = self._particles[0].personal_best_objectives

        return OptimizationResult(
            best_position=best_position,
            best_objectives=best_objectives,
            pareto_front=self._pareto_archive.get_front(),
            history=history,
            n_evaluations=n_evaluations,
        )

    def _eval_parallel(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray],
        positions: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Evaluate positions with optional threading parallelism.

        Threading releases the GIL during numpy/sklearn/PyTorch compute, giving
        real concurrency for both CPU tree models and GPU neural models.
        """
        n_jobs = min(self.n_workers, len(positions))
        if n_jobs > 1:
            try:
                from joblib import Parallel, delayed
                return Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(objective_fn)(pos) for pos in positions
                )
            except Exception as exc:
                logger.warning(f"Parallel evaluation failed ({exc}), falling back to sequential")
        return [objective_fn(pos) for pos in positions]

    def _save_checkpoint(
        self, path: str, start_from_iteration: int, history: list, n_evaluations: int,
        n_features: int = 0,
    ) -> None:
        """Atomically write optimizer state to a pickle checkpoint file."""
        state = {
            'particles': self._particles,
            'pareto_archive': self._pareto_archive,
            'history': history,
            'n_evaluations': n_evaluations,
            'start_from_iteration': start_from_iteration,
            'rng_state': self._rng.bit_generator.state,
            'n_features': n_features,
        }
        tmp = path + '.tmp'
        try:
            with open(tmp, 'wb') as fh:
                pickle.dump(state, fh, protocol=5)
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning(f"Checkpoint save failed: {exc}")

    def _load_checkpoint(self, path: str, n_features: int = 0) -> Optional[dict]:
        """Load checkpoint state dict.

        Returns None (triggering a fresh run) if the checkpoint was written with a
        different number of features — stale checkpoints cause the optimiser to reuse
        hyperparameters tuned for a different search-space dimensionality.
        """
        try:
            with open(path, 'rb') as fh:
                state = pickle.load(fh)
            stored_nf = state.get('n_features', 0)
            if n_features > 0 and stored_nf > 0 and stored_nf != n_features:
                logger.warning(
                    f"Checkpoint n_features mismatch ({stored_nf} stored vs {n_features} current) "
                    f"— discarding stale checkpoint and restarting optimisation."
                )
                return None
            return state
        except Exception as exc:
            logger.warning(f"Checkpoint load failed: {exc}")
            return None

    def _get_wolves(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get alpha, beta, gamma wolves from Pareto archive."""
        front = self._pareto_archive.get_front()

        if len(front) >= 3:
            # Sort by first objective and take top 3
            sorted_front = sorted(front, key=lambda x: x[1][0])
            alpha = sorted_front[0][0]
            beta = sorted_front[1][0]
            gamma = sorted_front[2][0]
        elif len(front) == 2:
            alpha = front[0][0]
            beta = front[1][0]
            gamma = (front[0][0] + front[1][0]) / 2
        elif len(front) == 1:
            alpha = beta = gamma = front[0][0]
        else:
            # Use random particles
            alpha = self._particles[0].position
            beta = self._particles[min(1, len(self._particles)-1)].position
            gamma = self._particles[min(2, len(self._particles)-1)].position

        return alpha, beta, gamma

    def get_pareto_archive(self) -> ParetoArchive:
        """Get the Pareto archive."""
        return self._pareto_archive


def create_search_space(
    param_bounds: Dict[str, Tuple[float, float]],
    n_features: int,
    include_feature_mask: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Create search space from parameter bounds.

    Args:
        param_bounds: Dictionary of parameter name to (min, max) bounds
        n_features: Number of features (for feature mask)
        include_feature_mask: Whether to include feature selection

    Returns:
        Tuple of (lower_bounds, upper_bounds, dimension_mapping)
    """
    lower = []
    upper = []
    mapping = {}
    idx = 0

    for name, (lb, ub) in param_bounds.items():
        lower.append(lb)
        upper.append(ub)
        mapping[name] = idx
        idx += 1

    if include_feature_mask:
        # Add binary feature mask
        for i in range(n_features):
            lower.append(0)
            upper.append(1)
            mapping[f'feature_{i}'] = idx
            idx += 1

    return np.array(lower), np.array(upper), mapping


def decode_position(
    position: np.ndarray,
    mapping: Dict[str, int],
    param_types: Dict[str, str]
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Decode position vector into parameters and feature mask.

    Args:
        position: Position vector from optimizer
        mapping: Dimension mapping
        param_types: Dictionary of parameter name to type ('int', 'float', 'log')

    Returns:
        Tuple of (parameters dict, feature mask array)
    """
    params = {}
    feature_mask = []

    for name, idx in mapping.items():
        if name.startswith('feature_'):
            # Binary feature selection (threshold at 0.5)
            feature_mask.append(1 if position[idx] > 0.5 else 0)
        else:
            value = position[idx]
            if name in param_types:
                if param_types[name] == 'int':
                    value = int(round(value))
                elif param_types[name] == 'log':
                    value = 10 ** value
            params[name] = value

    return params, np.array(feature_mask)
