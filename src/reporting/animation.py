"""
Training Animation Module
=========================

Generates animated GIFs showing model training progress:
  - Per-model: animated loss + accuracy curves growing epoch by epoch
  - Multi-model: animated comparison of validation loss across all models

Usage (from pipeline):
    from src.reporting.animation import TrainingAnimator
    animator = TrainingAnimator(output_dir=paper_output_dir / 'figures')
    animator.animate_training(model.training_history, model_name='LSTM')
    animator.animate_multi_model_comparison(model_histories, metric='val_loss')
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class TrainingAnimator:
    """
    Create animated GIFs of model training and validation progress.

    Requires matplotlib (always available in this project).
    Outputs are saved as .gif files using the Pillow writer.
    """

    output_dir: Path = field(default_factory=lambda: Path("outputs/paper_outputs/figures"))
    dpi: int = 100
    fps: int = 8
    style: str = "seaborn-v0_8-whitegrid"

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(self.style)
            except Exception:
                plt.style.use('default')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def animate_training(
        self,
        history: Dict[str, List[float]],
        model_name: str = "Model"
    ) -> Optional[Path]:
        """
        Animate training and validation loss/accuracy for a single model.

        Expected keys in history: 'train_loss', 'val_loss', and optionally
        'train_acc' / 'val_acc'.

        Args:
            history   : dict with lists of per-epoch metrics
            model_name: displayed in title and used for filename

        Returns:
            Path to saved .gif, or None on failure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available — skipping animation")
            return None

        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        train_acc = history.get('train_acc', history.get('train_accuracy', []))
        val_acc = history.get('val_acc', history.get('val_accuracy', []))

        if not train_loss:
            logger.warning(f"No training loss data for {model_name} — skipping animation")
            return None

        n_epochs = len(train_loss)
        has_acc = bool(train_acc)
        n_panels = 2 if has_acc else 1

        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]
        fig.suptitle(f"Training Progress — {model_name}", fontsize=13)

        ax_loss = axes[0]
        ax_loss.set_xlim(1, n_epochs)
        y_min = min(
            min(train_loss),
            min(val_loss) if val_loss else min(train_loss)
        ) * 0.95
        y_max = max(
            max(train_loss),
            max(val_loss) if val_loss else max(train_loss)
        ) * 1.05
        ax_loss.set_ylim(y_min, y_max)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Cross-Entropy Loss")
        ax_loss.grid(alpha=0.3)

        line_tl, = ax_loss.plot([], [], 'b-', lw=2, label='Train')
        line_vl, = ax_loss.plot([], [], 'r--', lw=2, label='Val')
        vline_l = ax_loss.axvline(0, color='green', lw=1, linestyle=':', alpha=0)
        ax_loss.legend(loc='upper right')

        if has_acc:
            ax_acc = axes[1]
            ax_acc.set_xlim(1, n_epochs)
            all_acc = list(train_acc) + (list(val_acc) if val_acc else [])
            ax_acc.set_ylim(max(0, min(all_acc) - 0.05), min(1.0, max(all_acc) + 0.05))
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.set_title("Accuracy")
            ax_acc.grid(alpha=0.3)
            line_ta, = ax_acc.plot([], [], 'b-', lw=2, label='Train')
            line_va, = ax_acc.plot([], [], 'r--', lw=2, label='Val')
            ax_acc.legend(loc='lower right')
        else:
            line_ta = line_va = None

        epochs_range = list(range(1, n_epochs + 1))

        # Best val epoch
        best_epoch = int(np.argmin(val_loss)) + 1 if val_loss else n_epochs

        def _init():
            line_tl.set_data([], [])
            line_vl.set_data([], [])
            if line_ta:
                line_ta.set_data([], [])
                line_va.set_data([], [])
            return [line_tl, line_vl]

        def _update(frame):
            ep = frame + 1
            xs = epochs_range[:ep]

            line_tl.set_data(xs, train_loss[:ep])
            if val_loss:
                line_vl.set_data(xs, val_loss[:ep])
                if ep >= best_epoch:
                    vline_l.set_xdata([best_epoch, best_epoch])
                    vline_l.set_alpha(0.7)

            if line_ta and train_acc:
                line_ta.set_data(xs, train_acc[:ep])
            if line_va and val_acc:
                line_va.set_data(xs, val_acc[:ep])

            return [line_tl, line_vl]

        ani = animation.FuncAnimation(
            fig, _update,
            frames=n_epochs,
            init_func=_init,
            blit=False,
            interval=1000 // self.fps
        )

        plt.tight_layout()
        safe_name = model_name.replace(' ', '_')
        output_path = self.output_dir / f"F_animation_{safe_name}_training.gif"
        try:
            ani.save(str(output_path), writer='pillow', fps=self.fps, dpi=self.dpi)
            logger.info(f"Saved training animation to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save GIF ({e}) — trying MP4")
            output_path = self.output_dir / f"F_animation_{safe_name}_training.mp4"
            try:
                ani.save(str(output_path), writer='ffmpeg', fps=self.fps, dpi=self.dpi)
                logger.info(f"Saved training animation to {output_path}")
            except Exception as e2:
                logger.warning(f"Animation save failed: {e2}")
                plt.close()
                return None

        plt.close()
        return output_path

    def animate_multi_model_comparison(
        self,
        model_histories: Dict[str, Dict[str, List[float]]],
        metric: str = 'val_loss'
    ) -> Optional[Path]:
        """
        Animate a comparison of a chosen metric across multiple models.

        Args:
            model_histories: dict of model_name -> history dict
            metric         : key to plot (e.g. 'val_loss', 'train_loss')

        Returns:
            Path to saved .gif, or None on failure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Filter models that actually have the metric
        valid = {
            name: hist[metric]
            for name, hist in model_histories.items()
            if metric in hist and hist[metric]
        }

        if not valid:
            logger.warning(f"No models with '{metric}' — skipping multi-model animation")
            return None

        max_epochs = max(len(v) for v in valid.values())

        # Pad shorter series by holding their final value so all models are
        # visible on the same x-axis for the full animation duration.
        valid = {
            name: (vals + [vals[-1]] * (max_epochs - len(vals)))
            for name, vals in valid.items()
        }

        colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))

        fig, ax = plt.subplots(figsize=(9, 5))

        all_vals = [v for vals in valid.values() for v in vals]
        y_min = min(all_vals) * 0.95
        y_max = max(all_vals) * 1.05
        ax.set_xlim(1, max_epochs)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f"Multi-Model Comparison — {metric.replace('_', ' ').title()}")
        ax.grid(alpha=0.3)

        lines = {}
        for (name, vals), color in zip(valid.items(), colors):
            line, = ax.plot([], [], lw=2, color=color, label=name)
            lines[name] = (line, vals)

        ax.legend(loc='upper right')

        def _init():
            for line, _ in lines.values():
                line.set_data([], [])
            return [l for l, _ in lines.values()]

        def _update(frame):
            ep = frame + 1
            for name, (line, vals) in lines.items():
                xs = list(range(1, min(ep, len(vals)) + 1))
                line.set_data(xs, vals[:ep])
            return [l for l, _ in lines.values()]

        ani = animation.FuncAnimation(
            fig, _update,
            frames=max_epochs,
            init_func=_init,
            blit=False,
            interval=1000 // self.fps
        )

        plt.tight_layout()
        output_path = self.output_dir / f"F_animation_multi_model_{metric}.gif"
        try:
            ani.save(str(output_path), writer='pillow', fps=self.fps, dpi=self.dpi)
            logger.info(f"Saved multi-model animation to {output_path}")
        except Exception as e:
            logger.warning(f"Multi-model animation save failed: {e}")
            plt.close()
            return None

        plt.close()
        return output_path

    def animate_cv_folds(
        self,
        fold_metrics: List[Dict[str, float]],
        metric_keys: List[str],
        model_name: str = "Model"
    ) -> Optional[Path]:
        """
        Animate per-fold cross-validation metric evolution (bar chart animation).

        Args:
            fold_metrics: list of dicts, one per CV fold
            metric_keys : which metric keys to show
            model_name  : model name for title/filename

        Returns:
            Path to saved .gif, or None on failure
        """
        if not MATPLOTLIB_AVAILABLE or not fold_metrics:
            return None

        n_folds = len(fold_metrics)
        available = [k for k in metric_keys if k in fold_metrics[0]]
        if not available:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"Cross-Validation Progress — {model_name}", fontsize=13)

        x = np.arange(len(available))
        colors = plt.cm.tab10(np.linspace(0, 0.7, n_folds))
        bars = ax.bar(x, [fold_metrics[0].get(k, 0) for k in available],
                      color=colors[0], edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('_', ' ').title() for k in available], rotation=30, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Metric Value")
        ax.grid(axis='y', alpha=0.3)

        fold_text = ax.text(0.02, 0.95, 'Fold 1', transform=ax.transAxes,
                            fontsize=11, va='top', color='navy')

        def _update(frame):
            fold_data = fold_metrics[frame]
            for bar, key in zip(bars, available):
                bar.set_height(fold_data.get(key, 0))
                bar.set_color(colors[frame])
            fold_text.set_text(f'Fold {frame + 1}')
            return list(bars) + [fold_text]

        ani = animation.FuncAnimation(
            fig, _update,
            frames=n_folds,
            blit=False,
            interval=800
        )

        plt.tight_layout()
        safe_name = model_name.replace(' ', '_')
        output_path = self.output_dir / f"F_animation_{safe_name}_cv_folds.gif"
        try:
            ani.save(str(output_path), writer='pillow', fps=2, dpi=self.dpi)
            logger.info(f"Saved CV fold animation to {output_path}")
        except Exception as e:
            logger.warning(f"CV fold animation save failed: {e}")
            plt.close()
            return None

        plt.close()
        return output_path
