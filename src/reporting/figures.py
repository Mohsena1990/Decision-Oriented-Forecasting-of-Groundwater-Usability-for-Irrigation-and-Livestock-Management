"""
Figure Generator Module
=======================

Generates publication-ready figures for the paper.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import shap as _shap_lib
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    _shap_lib = None

logger = logging.getLogger(__name__)

# Consistent colour palette for all 6 models + ensemble
MODEL_COLORS: Dict[str, str] = {
    'CatBoost': '#2196F3',       # blue
    'LightGBM': '#4CAF50',       # green
    'XGBoost': '#FF9800',        # orange
    'FT-Transformer': '#9C27B0', # purple
    'SpatialGNN': '#F44336',     # red
    'CORAL': '#00BCD4',          # cyan
    'Ensemble': '#607D8B',       # blue-grey
}


def _model_color(model_name: str, fallback_idx: int = 0) -> str:
    """Return the canonical colour for a model, or a fallback from tab10."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    # Graceful fallback for unknown model names
    palette = list(MODEL_COLORS.values())
    return palette[fallback_idx % len(palette)]


@dataclass
class FigureGenerator:
    """
    Generate publication-ready figures for JEM paper.

    Creates all required figures with consistent styling
    and saves to specified output directory.
    """

    output_dir: Path = field(default_factory=lambda: Path("outputs/paper_outputs/figures"))
    dpi: int = 300
    figsize: Tuple[int, int] = (10, 8)
    style: str = "seaborn-v0_8-whitegrid"

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(self.style)
            except:
                plt.style.use('default')

    def _savefig(self, path: Path) -> None:
        """Save the current matplotlib figure as both PNG (raster, dpi=self.dpi)
        and PDF (vector, LaTeX-ready) at the same base filename."""
        path = Path(path)
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')

    def f2_class_distribution(
        self,
        data: Dict[int, pd.DataFrame],
        target_col: str = 'Classification'
    ) -> Path:
        """
        F2: Class distribution per year.

        Args:
            data: Dictionary mapping year to DataFrame
            target_col: Target column name

        Returns:
            Path to saved figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return None

        fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 6))
        if len(data) == 1:
            axes = [axes]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 16))

        for ax, (year, df) in zip(axes, sorted(data.items())):
            counts = df[target_col].value_counts().sort_index()

            bars = ax.bar(range(len(counts)), counts.values, color=colors[:len(counts)])
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha='right')
            ax.set_xlabel('Classification')
            ax.set_ylabel('Count')
            ax.set_title(f'Year {year}')

            # Add count labels
            for bar, count in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       str(count), ha='center', va='bottom', fontsize=8)

        plt.suptitle('Groundwater Quality Class Distribution by Year', fontsize=14)
        plt.tight_layout()

        output_path = self.output_dir / "F2_class_distribution.png"
        self._savefig(output_path)
        plt.close()

        logger.info(f"Saved F2 to {output_path}")
        return output_path

    def f3_temporal_forecasting_schematic(self) -> Path:
        """
        F3: Temporal forecasting schematic showing train/test splits.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 4))

        # Draw timeline
        years = [2018, 2019, 2020]
        y_level = 0.5

        # Year boxes
        for i, year in enumerate(years):
            rect = mpatches.FancyBboxPatch(
                (i * 2, y_level - 0.2), 1.5, 0.4,
                boxstyle="round,pad=0.05",
                facecolor='lightblue' if year < 2020 else 'lightgreen',
                edgecolor='black'
            )
            ax.add_patch(rect)
            ax.text(i * 2 + 0.75, y_level, str(year), ha='center', va='center', fontsize=12)

        # Arrows for transitions
        # Train transition
        ax.annotate('', xy=(2, y_level), xytext=(1.5, y_level),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.text(1.75, y_level + 0.3, 'Train\n2018→2019', ha='center', fontsize=10, color='blue')

        # Test transition
        ax.annotate('', xy=(4, y_level), xytext=(3.5, y_level),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.text(3.75, y_level + 0.3, 'Test\n2019→2020', ha='center', fontsize=10, color='green')

        # Labels
        ax.text(0.75, y_level - 0.4, 'Features\n(X_t)', ha='center', fontsize=9)
        ax.text(2.75, y_level - 0.4, 'Target\n(y_{t+1})', ha='center', fontsize=9)
        ax.text(4.75, y_level - 0.4, 'Target\n(y_{t+1})', ha='center', fontsize=9)

        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-0.1, 1.2)
        ax.axis('off')
        ax.set_title('Temporal Forward Forecasting Setup', fontsize=14)

        output_path = self.output_dir / "F3_temporal_schematic.png"
        self._savefig(output_path)
        plt.close()

        logger.info(f"Saved F3 to {output_path}")
        return output_path

    def f4_model_comparison(
        self,
        results: pd.DataFrame,
        metrics: List[str] = ['macro_f1', 'severe_fnr', 'ordinal_distance_mean'],
        best_model: Optional[str] = None,
    ) -> Path:
        """
        F4: Model comparison bar chart with VIKOR winner highlighted.

        Args:
            results: DataFrame with model results (sorted by Rank ascending)
            metrics: Metrics to compare
            best_model: Name of the VIKOR-selected best model (auto-detected from Rank=1 if None)
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Auto-detect best model from Rank column
        if best_model is None and 'Rank' in results.columns:
            best_model = results.loc[results['Rank'] == 1, 'Model'].iloc[0] \
                if 'Model' in results.columns else None

        # Filter to metrics that actually exist
        available_metrics = [m for m in metrics if m in results.columns]
        if not available_metrics:
            available_metrics = [m for m in ['macro_f1', 'accuracy'] if m in results.columns]

        n_metrics = len(available_metrics)
        n_models = len(results)

        # Widen figure for 6 models
        fig_w = max(4 * n_metrics, n_metrics * max(4, n_models * 0.9))
        fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, 6))
        if n_metrics == 1:
            axes = [axes]

        models = results['Model'].values if 'Model' in results.columns else results.index.astype(str)

        # Colour: gold for VIKOR winner, MODEL_COLORS for others
        bar_colors = []
        for i, m in enumerate(models):
            if m == best_model:
                bar_colors.append('#FFD700')   # gold for VIKOR winner
            else:
                bar_colors.append(_model_color(m, fallback_idx=i))

        # Metric display names and directionality labels
        metric_labels = {
            'macro_f1': ('Macro F1', 'higher is better'),
            'severe_fnr': ('Severe FNR', 'lower is better ↓'),
            'ordinal_distance_mean': ('Ordinal Distance', 'lower is better ↓'),
            'accuracy': ('Accuracy', 'higher is better'),
            'severe_recall': ('Severe Recall', 'higher is better'),
        }

        for ax, metric in zip(axes, available_metrics):
            values = results[metric].values
            display_name, direction = metric_labels.get(metric, (metric.replace('_', ' ').title(), ''))

            bars = ax.bar(range(n_models), values, color=bar_colors)
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(display_name)
            ax.set_title(f'{display_name}\n({direction})', fontsize=10)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

            # Star annotation on winner bar
            if best_model in list(models):
                winner_idx = list(models).index(best_model)
                winner_val = values[winner_idx]
                ax.annotate(
                    '★ VIKOR #1',
                    xy=(winner_idx, winner_val),
                    xytext=(winner_idx, winner_val + max(values) * 0.08),
                    ha='center', fontsize=8, color='#B8860B', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#B8860B', lw=1.2)
                )

        plt.suptitle('Model Performance Comparison\n(gold bar = VIKOR-selected best model)', fontsize=13)
        plt.tight_layout()

        output_path = self.output_dir / "F4_model_comparison.png"
        self._savefig(output_path)
        plt.close()

        logger.info(f"Saved F4 to {output_path}")
        return output_path

    def f5_pareto_fronts(
        self,
        optimization_results: Dict[str, Any],
        objective_names: List[str] = ['Ordinal Distance', 'Severe FNR']
    ) -> Path:
        """
        F5: Pareto fronts per model.

        Args:
            optimization_results: Dict mapping model name to OptimizationResult
            objective_names: Names of objectives for axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Use MODEL_COLORS; fall back to tab10 for unknown names
        tab10 = plt.cm.tab10(np.linspace(0, 1, max(len(optimization_results), 1)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

        for i, (model_name, opt_result) in enumerate(optimization_results.items()):
            color = MODEL_COLORS.get(model_name, tab10[i % len(tab10)])
            marker = markers[i % len(markers)]
            # Extract objectives from Pareto front
            if hasattr(opt_result, 'pareto_front'):
                front = opt_result.pareto_front
                if front and len(front) > 0:
                    objectives = np.array([obj for _, obj in front])
                    ax.scatter(
                        objectives[:, 0], objectives[:, 1],
                        c=[color], marker=marker, s=100, alpha=0.7,
                        label=model_name, edgecolors='black'
                    )
            elif isinstance(opt_result, np.ndarray) and len(opt_result) > 0:
                ax.scatter(
                    opt_result[:, 0], opt_result[:, 1],
                    c=[color], marker=marker, s=100, alpha=0.7,
                    label=model_name, edgecolors='black'
                )

        ax.set_xlabel(objective_names[0] if len(objective_names) > 0 else 'Objective 1')
        ax.set_ylabel(objective_names[1] if len(objective_names) > 1 else 'Objective 2')
        ax.set_title('Pareto Fronts by Model')
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / "F5_pareto_fronts.png"
        self._savefig(output_path)
        plt.close()

        logger.info(f"Saved F5 to {output_path}")
        return output_path

    def f6_vikor_rankings(
        self,
        comparison_df: pd.DataFrame
    ) -> Path:
        """
        F6: VIKOR rankings visualization.

        Args:
            comparison_df: DataFrame with model comparison from VIKOR
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        n_models = len(comparison_df)
        fig_h = max(6, n_models * 0.6)
        fig, axes = plt.subplots(1, 2, figsize=(max(14, n_models * 2), fig_h))

        # Plot 1: Q Score comparison (horizontal bars)
        ax1 = axes[0]
        models = comparison_df['Model'].values
        q_scores = comparison_df['Q_Score'].values if 'Q_Score' in comparison_df.columns else np.zeros(len(models))

        bar_colors = [_model_color(m, i) for i, m in enumerate(models)]
        bars = ax1.barh(range(len(models)), q_scores, color=bar_colors, alpha=0.85)
        ax1.set_yticks(range(len(models)))
        ax1.set_yticklabels(models)
        ax1.set_xlabel('VIKOR Q Score (lower is better)')
        ax1.set_title('Model Rankings by VIKOR Q Score')

        for bar, val in zip(bars, q_scores):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='left', va='center', fontsize=10)

        # Plot 2: Objectives grouped bar chart
        ax2 = axes[1]
        obj_cols = [c for c in comparison_df.columns if c not in ['Model', 'Rank', 'Q_Score', 'n_features', 'In_Compromise_Set']]

        if obj_cols:
            x = np.arange(len(obj_cols))
            width = max(0.1, 0.8 / max(len(models), 1))

            for i, (_, row) in enumerate(comparison_df.iterrows()):
                values = [row[c] for c in obj_cols if c in row]
                color = _model_color(row['Model'], i)
                ax2.bar(x + i * width, values, width,
                        label=row['Model'], alpha=0.85, color=color)

            ax2.set_xticks(x + width * (len(models) - 1) / 2)
            ax2.set_xticklabels([c.replace('_', '\n') for c in obj_cols], rotation=0)
            ax2.set_ylabel('Objective Value')
            ax2.set_title('Objective Values by Model')
            ax2.legend(loc='upper right', fontsize=8)

        plt.tight_layout()

        output_path = self.output_dir / "F6_vikor_rankings.png"
        self._savefig(output_path)
        plt.close()

        logger.info(f"Saved F6 to {output_path}")
        return output_path

    def f7_shap_summary(
        self,
        importance: Dict[str, float],
        feature_names: List[str],
        top_k: int = 15
    ) -> Path:
        """
        F7: SHAP feature importance summary.

        Args:
            importance: Dict mapping feature name to importance value
            feature_names: List of feature names
            top_k: Number of top features to show
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Sort by importance and take top k
        sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

        features = [f[0] for f in sorted_importance]
        values = [f[1] for f in sorted_importance]

        # Reverse for horizontal bar chart
        features = features[::-1]
        values = values[::-1]

        colors = plt.cm.RdBu_r(np.linspace(0.2, 0.8, len(features)))
        bars = ax.barh(range(len(features)), values, color=colors)

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title(f'Top {top_k} Feature Importance (SHAP)')
        ax.axvline(0, color='black', linewidth=0.5)

        plt.tight_layout()

        output_path = self.output_dir / "F7_shap_summary.png"
        self._savefig(output_path)
        plt.close()

        logger.info(f"Saved F7 to {output_path}")
        return output_path

    def f8_scenario_impact(
        self,
        scenario_report: pd.DataFrame
    ) -> Path:
        """
        F8: Scenario simulation impact plot.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        scenarios = scenario_report['scenario'].values
        changes = scenario_report['high_risk_rate_change'].values * 100  # Convert to percentage

        colors = ['red' if c > 0 else 'green' for c in changes]
        bars = ax.barh(range(len(scenarios)), changes, color=colors, alpha=0.7)

        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios)
        ax.set_xlabel('Change in High-Risk Rate (%)')
        ax.set_title('Impact of Scenarios on High-Risk Groundwater Classification')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, changes):
            xpos = bar.get_width() + 0.5 if val >= 0 else bar.get_width() - 0.5
            ha = 'left' if val >= 0 else 'right'
            ax.text(xpos, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', ha=ha, va='center', fontsize=9)

        plt.tight_layout()

        output_path = self.output_dir / "F8_scenario_impact.png"
        self._savefig(output_path)
        plt.close()

        logger.info(f"Saved F8 to {output_path}")
        return output_path

    def f9_spatial_risk_map(
        self,
        df: pd.DataFrame,
        lat_col: str = 'lat_gis',
        long_col: str = 'long_gis',
        risk_col: str = 'predicted_risk'
    ) -> Path:
        """
        F9: Spatial risk map (scatter plot).
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create risk categories
        if risk_col in df.columns:
            scatter = ax.scatter(
                df[long_col], df[lat_col],
                c=df[risk_col], cmap='RdYlGn_r',
                s=50, alpha=0.7, edgecolors='black', linewidth=0.5
            )
            plt.colorbar(scatter, label='Risk Level', ax=ax)
        else:
            ax.scatter(
                df[long_col], df[lat_col],
                s=50, alpha=0.7, edgecolors='black', linewidth=0.5
            )

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Spatial Distribution of Groundwater Quality Risk')

        output_path = self.output_dir / "F9_spatial_risk_map.png"
        self._savefig(output_path)
        plt.close()

        logger.info(f"Saved F9 to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Enhanced SHAP figures
    # ------------------------------------------------------------------

    def f7b_shap_beeswarm(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        top_k: int = 15,
    ) -> Optional[Path]:
        """
        F7b: Beeswarm (dot-strip) plot of SHAP values.

        Each dot represents one sample; x-position is its SHAP value for
        that feature; colour encodes the original feature value (red=high,
        blue=low), matching the standard SHAP beeswarm convention.

        Parameters
        ----------
        shap_values : np.ndarray
            Shape (n_samples, n_features) or (n_samples, n_features, n_classes).
            For multi-class arrays the mean absolute SHAP across classes is used.
        X : np.ndarray
            Original (unscaled if possible) feature matrix — used for colouring.
        feature_names : list of str
        top_k : int
            Number of top features to display.

        Returns
        -------
        Path to saved figure, or None if matplotlib unavailable.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Collapse multi-class SHAP to 2-D
        if shap_values.ndim == 3:
            sv2d = np.mean(np.abs(shap_values), axis=2)   # (n, n_feat)
            sv_signed = np.mean(shap_values, axis=2)       # for sign
        else:
            sv2d = shap_values
            sv_signed = shap_values

        # Select top-k by mean |SHAP|
        importance = np.mean(np.abs(sv2d), axis=0)
        top_idx = np.argsort(importance)[-top_k:]          # ascending → last = most important
        top_idx_rev = top_idx[::-1]                        # most important first (top of plot)

        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.5)))

        rng = np.random.default_rng(42)
        cmap = plt.cm.RdBu_r

        for plot_row, feat_idx in enumerate(top_idx):      # plot_row 0 = least important (bottom)
            sv_col = sv_signed[:, feat_idx]
            feat_col = X[:, feat_idx] if feat_idx < X.shape[1] else np.zeros(len(sv_col))

            # Normalise feature values to [0, 1] for colour
            f_min, f_max = feat_col.min(), feat_col.max()
            if f_max > f_min:
                colour_vals = (feat_col - f_min) / (f_max - f_min)
            else:
                colour_vals = np.full_like(feat_col, 0.5)

            # Jitter on y so points do not overlap
            y_jitter = plot_row + rng.uniform(-0.35, 0.35, len(sv_col))
            ax.scatter(sv_col, y_jitter, c=colour_vals, cmap=cmap,
                       alpha=0.55, s=12, vmin=0, vmax=1)

        feat_labels = [feature_names[i] for i in top_idx]
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value  (impact on model output)")
        ax.set_title(f"SHAP Beeswarm — Top {len(top_idx)} Features")

        # Colour bar indicating feature value magnitude
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Feature value (low → high)", fontsize=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])

        plt.tight_layout()
        output_path = self.output_dir / "F7b_shap_beeswarm.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved F7b to {output_path}")
        return output_path

    def f7c_shap_class_comparison(
        self,
        global_importance: pd.DataFrame,
        high_risk_importance: pd.DataFrame,
        top_k: int = 12,
    ) -> Optional[Path]:
        """
        F7c: Side-by-side bar chart comparing global vs high-risk SHAP.

        Shows which features drive predictions *in general* versus
        which features specifically drive high-risk classifications,
        enabling identification of hazard-specific chemical drivers.

        Parameters
        ----------
        global_importance : pd.DataFrame
            Columns: ['feature', 'importance'].  Global mean |SHAP|.
        high_risk_importance : pd.DataFrame
            Same structure but computed on high-risk predictions only.
        top_k : int
            Number of top global features to include.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Align both DataFrames on top-k global features
        top_features = global_importance.head(top_k)["feature"].tolist()

        global_vals = (
            global_importance.set_index("feature")["importance"]
            .reindex(top_features).fillna(0).values
        )
        hr_vals = (
            high_risk_importance.set_index("feature")["importance"]
            .reindex(top_features).fillna(0).values
        )

        x = np.arange(len(top_features))
        width = 0.38

        fig, ax = plt.subplots(figsize=(12, 5))
        bars_g = ax.bar(x - width / 2, global_vals, width,
                        label="Global (all classes)", color="#4C72B0", alpha=0.85)
        bars_h = ax.bar(x + width / 2, hr_vals, width,
                        label="High-risk class", color="#DD8452", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance: Global vs High-Risk Class")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "F7c_shap_class_comparison.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved F7c to {output_path}")
        return output_path

    def f7d_shap_dependence(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        top_features: Optional[List[str]] = None,
        n_plots: int = 4,
    ) -> Optional[Path]:
        """
        F7d: SHAP dependence plots for the most important features.

        Each sub-plot shows SHAP value (y) vs raw feature value (x) for
        one feature, with colour representing the most correlated other
        feature (interaction indicator).

        Parameters
        ----------
        shap_values : np.ndarray
            (n_samples, n_features) or (n_samples, n_features, n_classes).
        X : np.ndarray
            Feature matrix (same ordering as feature_names).
        feature_names : list of str
        top_features : list of str, optional
            Ordered list of features to plot.  If None, inferred from
            mean |SHAP|.
        n_plots : int
            Number of dependence sub-plots (max 4).
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        n_plots = min(n_plots, 4)

        # Collapse multi-class
        if shap_values.ndim == 3:
            sv = np.mean(shap_values, axis=2)
        else:
            sv = shap_values

        # Determine which features to plot
        if top_features is None:
            importance = np.mean(np.abs(sv), axis=0)
            top_idx = np.argsort(importance)[-n_plots:][::-1]
        else:
            top_idx = []
            for f in top_features[:n_plots]:
                if f in feature_names:
                    top_idx.append(feature_names.index(f))
            if not top_idx:
                importance = np.mean(np.abs(sv), axis=0)
                top_idx = list(np.argsort(importance)[-n_plots:][::-1])

        ncols = min(2, n_plots)
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 4.5 * nrows),
                                 squeeze=False)

        for plot_i, feat_idx in enumerate(top_idx[:n_plots]):
            row, col = divmod(plot_i, ncols)
            ax = axes[row][col]

            feat_name = feature_names[feat_idx]
            shap_col = sv[:, feat_idx]
            x_vals = X[:, feat_idx]

            # Find most correlated feature for colour (simple approach)
            corrs = [
                abs(np.corrcoef(X[:, j], shap_col)[0, 1])
                if j != feat_idx else 0.0
                for j in range(X.shape[1])
            ]
            color_feat_idx = int(np.argmax(corrs))
            color_vals = X[:, color_feat_idx]
            color_name = feature_names[color_feat_idx] if color_feat_idx < len(feature_names) else "other"

            # Normalise colour
            c_min, c_max = color_vals.min(), color_vals.max()
            if c_max > c_min:
                c_norm = (color_vals - c_min) / (c_max - c_min)
            else:
                c_norm = np.full_like(color_vals, 0.5)

            sc = ax.scatter(x_vals, shap_col, c=c_norm, cmap="RdBu_r",
                            alpha=0.6, s=15, vmin=0, vmax=1)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel(f"SHAP({feat_name})", fontsize=9)
            ax.set_title(f"Dependence: {feat_name}", fontsize=10)
            plt.colorbar(sc, ax=ax).set_label(color_name, fontsize=8)

        # Hide unused axes
        for plot_i in range(len(top_idx), nrows * ncols):
            row, col = divmod(plot_i, ncols)
            axes[row][col].set_visible(False)

        plt.suptitle("SHAP Dependence Plots — Top Features", fontsize=13, y=1.01)
        plt.tight_layout()
        output_path = self.output_dir / "F7d_shap_dependence.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved F7d to {output_path}")
        return output_path

    def f7e_shap_class_heatmap(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        top_k: int = 12,
    ) -> Optional[Path]:
        """
        F7e: Heatmap of mean |SHAP| per feature × class.

        Reveals which features are most diagnostic for each water quality
        class, including the critical high-risk classes.

        Parameters
        ----------
        shap_values : np.ndarray
            Shape (n_samples, n_features, n_classes).
        feature_names : list of str
        class_names : list of str, optional
        top_k : int
            Number of top features (by global importance) to display.
        """
        if not MATPLOTLIB_AVAILABLE or shap_values.ndim != 3:
            return None

        n_classes = shap_values.shape[2]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        # Global importance → select top-k features
        global_imp = np.mean(np.abs(shap_values), axis=(0, 2))  # (n_features,)
        top_idx = np.argsort(global_imp)[-top_k:][::-1]

        # Build matrix: (top_k, n_classes)
        heatmap_data = np.zeros((len(top_idx), n_classes), dtype=np.float32)
        for row_i, f_idx in enumerate(top_idx):
            for c_idx in range(n_classes):
                heatmap_data[row_i, c_idx] = float(
                    np.mean(np.abs(shap_values[:, f_idx, c_idx]))
                )

        feat_labels = [feature_names[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.2), max(6, top_k * 0.5)))

        if SEABORN_AVAILABLE:
            import seaborn as sns
            df_heat = pd.DataFrame(heatmap_data, index=feat_labels, columns=class_names)
            sns.heatmap(
                df_heat, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.4, cbar_kws={"label": "Mean |SHAP|"},
            )
        else:
            im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(feat_labels)))
            ax.set_yticklabels(feat_labels, fontsize=9)
            plt.colorbar(im, ax=ax, label="Mean |SHAP|")

        ax.set_title("SHAP Feature Importance per Class", fontsize=13)
        plt.tight_layout()
        output_path = self.output_dir / "F7e_shap_class_heatmap.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved F7e to {output_path}")
        return output_path

    def generate_all_shap_figures(
        self,
        shap_result: Any,
        X: np.ndarray,
        global_importance: Optional[pd.DataFrame] = None,
        high_risk_importance: Optional[pd.DataFrame] = None,
        class_names: Optional[List[str]] = None,
        top_k: int = 15,
    ) -> Dict[str, Optional[Path]]:
        """
        Convenience wrapper: generate all SHAP analysis figures (F7 – F7e).

        Parameters
        ----------
        shap_result : SHAPResult
            Object with .shap_values, .feature_names, .global_importance.
        X : np.ndarray
            Feature matrix (used for beeswarm & dependence colouring).
        global_importance : pd.DataFrame, optional
            Falls back to shap_result.global_importance.
        high_risk_importance : pd.DataFrame, optional
            If None, F7c is skipped.
        class_names : list of str, optional
        top_k : int

        Returns
        -------
        dict mapping figure code → Path (or None).
        """
        sv = shap_result.shap_values
        feat_names = shap_result.feature_names
        if global_importance is None:
            global_importance = shap_result.global_importance

        paths: Dict[str, Optional[Path]] = {}

        # F7 — bar chart (existing)
        imp_dict = dict(zip(
            global_importance["feature"], global_importance["importance"]
        ))
        paths["F7"] = self.f7_shap_summary(imp_dict, feat_names, top_k=top_k)

        # F7b — beeswarm
        paths["F7b"] = self.f7b_shap_beeswarm(sv, X, feat_names, top_k=top_k)

        # F7c — global vs high-risk comparison
        if high_risk_importance is not None:
            paths["F7c"] = self.f7c_shap_class_comparison(
                global_importance, high_risk_importance, top_k=min(top_k, 12)
            )
        else:
            paths["F7c"] = None

        # F7d — dependence plots for top features
        top_feats = global_importance.head(4)["feature"].tolist()
        paths["F7d"] = self.f7d_shap_dependence(sv, X, feat_names,
                                                  top_features=top_feats)

        # F7e — per-class heatmap (only for 3-D SHAP arrays)
        if sv.ndim == 3:
            paths["F7e"] = self.f7e_shap_class_heatmap(
                sv, feat_names, class_names=class_names, top_k=min(top_k, 12)
            )
        else:
            paths["F7e"] = None

        saved = [k for k, v in paths.items() if v is not None]
        logger.info(f"Generated SHAP figures: {saved}")
        return paths

    def f4b_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        model_name: str = "Best Model",
    ) -> Optional[Path]:
        """
        F4b: Confusion matrix heatmap for one model.

        Parameters
        ----------
        y_true, y_pred : np.ndarray
            Integer-encoded ground-truth and predicted labels.
        class_names : list of str, optional
            Label for each class index.
        model_name : str
            Used in the figure title.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        from sklearn.metrics import confusion_matrix as _cm

        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        cm = _cm(y_true, y_pred, labels=labels)

        if class_names is None:
            class_names = [str(l) for l in labels]
        else:
            class_names = [class_names[l] if l < len(class_names) else str(l) for l in labels]

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.8)))

        if SEABORN_AVAILABLE:
            import seaborn as sns
            df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
            sns.heatmap(
                df_cm, ax=ax, annot=True, fmt="d", cmap="Blues",
                linewidths=0.4, cbar_kws={"label": "Count"},
            )
        else:
            im = ax.imshow(cm, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names, fontsize=8)
            plt.colorbar(im, ax=ax, label="Count")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)

        plt.tight_layout()
        output_path = self.output_dir / "F4b_confusion_matrix.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved F4b to {output_path}")
        return output_path

    def f4c_per_class_metrics(
        self,
        per_class_dict: Dict[str, Dict[str, float]],
        model_name: str = "Best Model",
    ) -> Optional[Path]:
        """
        F4c: Grouped bar chart of per-class precision / recall / F1.

        Parameters
        ----------
        per_class_dict : dict
            Output of MetricsCalculator.get_per_class_metrics():
            { class_label: {'precision': float, 'recall': float, 'f1': float,
                            'support': int} }
        model_name : str
            Used in the figure title.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        classes = list(per_class_dict.keys())
        precision = [per_class_dict[c]["precision"] for c in classes]
        recall    = [per_class_dict[c]["recall"]    for c in classes]
        f1        = [per_class_dict[c]["f1"]        for c in classes]
        support   = [per_class_dict[c]["support"]   for c in classes]

        x = np.arange(len(classes))
        width = 0.26

        fig, ax = plt.subplots(figsize=(max(9, len(classes) * 1.2), 5))
        bars_p = ax.bar(x - width, precision, width, label="Precision", color="#4C72B0", alpha=0.85)
        bars_r = ax.bar(x,          recall,    width, label="Recall",    color="#DD8452", alpha=0.85)
        bars_f = ax.bar(x + width,  f1,        width, label="F1",        color="#55A868", alpha=0.85)

        # Support counts as text above recall bar
        for bar, sup in zip(bars_r, support):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={sup}", ha="center", va="bottom", fontsize=7, color="gray",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(f"Per-Class Metrics — {model_name}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "F4c_per_class_metrics.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved F4c to {output_path}")
        return output_path

    def f4d_severity_comparison(
        self,
        results_df: pd.DataFrame,
    ) -> Optional[Path]:
        """
        F4d: Severity-focused metric comparison across all evaluated models.

        Plots severe_fnr, severe_recall, severe_precision, and severe_f1
        side-by-side for each model so the trade-offs are visible at a glance.

        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame with columns ['Model', 'severe_fnr', 'severe_recall',
            'severe_precision', 'severe_f1'] plus optionally others.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        severity_cols = ["severe_fnr", "severe_recall", "severe_precision", "severe_f1"]
        avail = [c for c in severity_cols if c in results_df.columns]
        if not avail:
            logger.warning("No severity columns in results_df — skipping F4d")
            return None

        models = results_df["Model"].tolist() if "Model" in results_df.columns else list(results_df.index)
        n_models = len(models)
        n_metrics = len(avail)

        x = np.arange(n_models)
        width = 0.8 / n_metrics
        colors = plt.cm.Set2(np.linspace(0, 0.9, n_metrics))

        fig, ax = plt.subplots(figsize=(max(8, n_models * 1.8), 5))

        for i, col in enumerate(avail):
            vals = results_df[col].values
            bars = ax.bar(x + i * width - (n_metrics - 1) * width / 2, vals,
                          width, label=col.replace("_", " ").title(),
                          color=colors[i], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Rate / Score")
        ax.set_title("Severity-Focused Metrics by Model\n"
                     "(lower FNR = fewer missed high-risk cases)")
        ax.legend(loc="upper right", fontsize=8)
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "F4d_severity_comparison.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved F4d to {output_path}")
        return output_path

    def create_mermaid_flowchart(self) -> Path:
        """
        F1: Create Mermaid flowchart file.
        """
        mermaid_content = """
```mermaid
flowchart TB
    subgraph Stage_A["Stage A: Data Ingestion"]
        A1[Load CSV Files<br>2018, 2019, 2020]
        A2[Column Harmonization]
        A1 --> A2
    end

    subgraph Stage_B["Stage B: Transition Building"]
        B1[Location Matching]
        B2[Build t→t+1 Pairs]
        B1 --> B2
    end

    subgraph Stage_C["Stage C: Data Quality"]
        C1[Schema Validation]
        C2[Missingness Analysis]
        C3[Outlier Detection]
        C4[Class Distribution]
        C1 --> C2 --> C3 --> C4
    end

    subgraph Stage_D["Stage D: Preprocessing"]
        D1[Imputation<br>Train-fit only]
        D2[Scaling<br>Deep models]
        D3[Encoding<br>Categorical]
        D1 --> D2 --> D3
    end

    subgraph Stage_E["Stage E: Model Candidates (6)"]
        E1[CatBoost]
        E2[LightGBM]
        E3[XGBoost]
        E4[FT-Transformer]
        E5[SpatialGNN]
        E6[CORAL]
    end

    subgraph Stage_F["Stage F: Optimization"]
        F1[PSO-GWO]
        F2[Feature Selection]
        F3[Hyperparameter Tuning]
        F4[Pareto Archive]
        F1 --> F2 --> F3 --> F4
    end

    subgraph Stage_G["Stage G: MCDM Selection"]
        G1[VIKOR per Model]
        G2[VIKOR across Models]
        G3[Final Model Selection]
        G1 --> G2 --> G3
    end

    subgraph Stage_H["Stage H: Explainability"]
        H1[TreeSHAP<br>Tree Models]
        H2[Surrogate SHAP<br>Deep Models]
        H3[Feature Attribution]
        H1 --> H3
        H2 --> H3
    end

    subgraph Stage_I["Stage I: Scenarios"]
        I1[TDS Perturbation]
        I2[SAR Perturbation]
        I3[RSC Thresholds]
        I4[Impact Analysis]
        I1 & I2 & I3 --> I4
    end

    subgraph Stage_J["Stage J: Paper Outputs"]
        J1[Figures]
        J2[Tables]
        J3[Managerial Insights]
    end

    Stage_A --> Stage_B --> Stage_C --> Stage_D
    Stage_D --> Stage_E
    Stage_E --> Stage_F --> Stage_G
    Stage_G --> Stage_H --> Stage_I --> Stage_J
```
"""

        output_path = self.output_dir / "F1_flowchart.md"
        with open(output_path, 'w') as f:
            f.write(mermaid_content)

        logger.info(f"Saved F1 (Mermaid) to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Classification result figures: ROC, PR, learning curves
    # ------------------------------------------------------------------

    def f_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        model_name: str = "Best Model",
    ) -> Optional[Path]:
        """
        Multi-class ROC curves (one-vs-rest).

        Plots AUC-ROC for each class and the macro-average.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        n_classes = y_proba.shape[1]
        classes = list(range(n_classes))
        if class_names is None:
            class_names = [str(c) for c in classes]

        y_bin = label_binarize(y_true, classes=classes)
        if n_classes == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        macro_tpr_interp = np.zeros(100)
        base_fpr = np.linspace(0, 1, 100)
        valid_classes = 0

        for i, (cls_name, color) in enumerate(zip(class_names, colors)):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{cls_name} (AUC={roc_auc:.2f})")
            macro_tpr_interp += np.interp(base_fpr, fpr, tpr)
            valid_classes += 1

        if valid_classes > 0:
            macro_tpr_interp /= valid_classes
            macro_auc = auc(base_fpr, macro_tpr_interp)
            ax.plot(base_fpr, macro_tpr_interp, 'k--', lw=2.5,
                    label=f"Macro avg (AUC={macro_auc:.2f})")

        ax.plot([0, 1], [0, 1], 'gray', lw=1, linestyle=':')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves (One-vs-Rest) — {model_name}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "F_roc_curves.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved ROC curves to {output_path}")
        return output_path

    def f_pr_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        model_name: str = "Best Model",
    ) -> Optional[Path]:
        """
        Multi-class Precision-Recall curves (one-vs-rest).
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize

        n_classes = y_proba.shape[1]
        classes = list(range(n_classes))
        if class_names is None:
            class_names = [str(c) for c in classes]

        y_bin = label_binarize(y_true, classes=classes)
        if n_classes == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for i, (cls_name, color) in enumerate(zip(class_names, colors)):
            if y_bin[:, i].sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ap = average_precision_score(y_bin[:, i], y_proba[:, i])
            ax.plot(recall, precision, color=color, lw=1.5,
                    label=f"{cls_name} (AP={ap:.2f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curves (One-vs-Rest) — {model_name}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "F_pr_curves.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved PR curves to {output_path}")
        return output_path

    def f_learning_curves(
        self,
        training_history: Dict[str, list],
        model_name: str = "Best Model",
    ) -> Optional[Path]:
        """
        Training / validation loss learning curves over epochs.

        Parameters
        ----------
        training_history : dict
            {'train_loss': [...], 'val_loss': [...]} from model.training_history
        model_name : str
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        train_loss = training_history.get('train_loss', [])
        val_loss = training_history.get('val_loss', [])

        if not train_loss:
            logger.warning("No training history to plot")
            return None

        fig, ax = plt.subplots(figsize=(9, 5))
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, 'b-', lw=2, label='Training loss')

        if val_loss:
            val_epochs = range(1, len(val_loss) + 1)
            ax.plot(val_epochs, val_loss, 'r--', lw=2, label='Validation loss')
            # Mark early-stopping point
            best_epoch = int(np.argmin(val_loss)) + 1
            ax.axvline(best_epoch, color='green', lw=1, linestyle=':', alpha=0.8,
                       label=f'Best epoch ({best_epoch})')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title(f"Learning Curves — {model_name}")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "F_learning_curves.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved learning curves to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # F8: Scenario Simulation
    # ------------------------------------------------------------------

    def f8_scenario_simulation(
        self,
        scenario_results: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        idx_to_label: Optional[Dict[int, str]] = None,
        high_risk_indices: Optional[List[int]] = None
    ) -> List[Path]:
        """
        F8: Scenario simulation impact figures.

        Generates three sub-figures:
          F8a – Risk change bar chart (mean high-risk probability change per scenario)
          F8b – Class shift heat-map  (% of samples that changed class per scenario)
          F8c – Transition matrix for the scenario with the largest risk change

        Args:
            scenario_results : dict of scenario_name -> ScenarioResult
            class_names      : ordered list of class label strings
            idx_to_label     : mapping from int index to label string
            high_risk_indices: list of high-risk class indices

        Returns:
            List of saved figure paths
        """
        if not MATPLOTLIB_AVAILABLE or not scenario_results:
            logger.warning("Cannot generate F8 — matplotlib unavailable or no scenarios")
            return []

        paths = []

        # Sort scenarios: TDS group first (ascending %), then SAR, then others.
        def _scenario_sort_key(name: str) -> tuple:
            n = name.upper()
            if n.startswith('TDS'):
                return (0, n)
            if n.startswith('SAR'):
                return (1, n)
            if n.startswith('RSC'):
                return (2, n)
            return (3, n)

        scenario_names = sorted(scenario_results.keys(), key=_scenario_sort_key)

        # ---- F8a + F8b data -------------------------------------------
        risk_changes = []
        escalation_pcts = []   # F8b: safe→T3 escalation (decision-relevant)
        n_samples_list = []

        for name in scenario_names:
            result = scenario_results[name]
            risk_changes.append(float(np.mean(result.high_risk_prob_change)))

            # Safe-to-risky escalation: fraction of non-T3 wells that flip to T3
            n_total = len(result.baseline_predictions)
            n_samples_list.append(n_total)
            if high_risk_indices:
                baseline_safe = ~np.isin(result.baseline_predictions, high_risk_indices)
                n_safe = int(baseline_safe.sum())
                if n_safe > 0:
                    escalated = baseline_safe & np.isin(result.scenario_predictions, high_risk_indices)
                    esc_pct = float(escalated.sum() / n_safe * 100)
                    n_esc = int(escalated.sum())
                else:
                    esc_pct, n_esc = 0.0, 0
            else:
                esc_pct = float(
                    np.mean(result.baseline_predictions != result.scenario_predictions) * 100
                )
                n_esc = int(np.sum(result.baseline_predictions != result.scenario_predictions))
            escalation_pcts.append((esc_pct, n_esc, n_safe if high_risk_indices else n_total))

        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(scenario_names) * 0.8 + 1.5)))

        # ---- F8a: Risk change bar chart --------------------------------
        colors_risk = ['#d62728' if v > 0 else '#2ca02c' for v in risk_changes]
        bars = axes[0].barh(
            scenario_names, risk_changes,
            color=colors_risk, edgecolor='black', linewidth=0.5
        )
        axes[0].axvline(0, color='black', lw=0.8, linestyle='--')
        axes[0].set_xlabel("Mean Δ High-Risk Probability")
        axes[0].set_title("F8a — High-Risk Probability Change per Scenario")
        axes[0].grid(axis='x', alpha=0.3)

        # Add 15% padding on the dominant side so labels don't clip
        x_lo, x_hi = axes[0].get_xlim()
        max_abs = max(abs(v) for v in risk_changes) if risk_changes else 1e-4
        axes[0].set_xlim(
            min(x_lo, -max_abs * 0.25),
            max(x_hi, max_abs * 0.25)
        )
        for i, v in enumerate(risk_changes):
            pad = max_abs * 0.08
            if v >= 0:
                axes[0].text(v + pad, i, f"+{v:.4f}", va='center', ha='left', fontsize=8)
            else:
                axes[0].text(v - pad, i, f"{v:.4f}", va='center', ha='right', fontsize=8)

        # ---- F8b: Safe→T3 escalation bar chart -------------------------
        esc_values = [e[0] for e in escalation_pcts]
        esc_counts = [(e[1], e[2]) for e in escalation_pcts]  # (escalated, n_safe)
        max_esc = max(esc_values) if esc_values else 1e-4
        colors_esc = plt.cm.OrRd(np.array(esc_values) / (max_esc + 1e-9))
        axes[1].barh(
            scenario_names, esc_values,
            color=colors_esc, edgecolor='black', linewidth=0.5
        )
        axes[1].set_xlabel("% Previously-Safe Wells Escalating to T3")
        axes[1].set_title("F8b — Safe→T3 Escalation Rate per Scenario")
        axes[1].grid(axis='x', alpha=0.3)
        x_lo1, x_hi1 = axes[1].get_xlim()
        axes[1].set_xlim(x_lo1, max(x_hi1, max_esc * 1.35))
        for i, (pct, (n_esc, n_safe)) in enumerate(zip(esc_values, esc_counts)):
            label = f"{pct:.1f}%  ({n_esc}/{n_safe})"
            axes[1].text(pct + max_esc * 0.03, i, label,
                         va='center', ha='left', fontsize=8)

        plt.tight_layout()
        p = self.output_dir / "F8a_scenario_risk_change.png"
        self._savefig(p)
        plt.close()
        paths.append(p)
        logger.info(f"Saved F8a to {p}")

        # ---- F8c: Transition matrix for highest-impact scenario -------
        if scenario_results:
            max_idx = int(np.argmax(np.abs(risk_changes)))
            max_name = scenario_names[max_idx]
            max_result = scenario_results[max_name]
            tm = max_result.transition_matrix

            if tm is not None and tm.size > 0:
                n_cls = tm.shape[0]
                labels = class_names if class_names and len(class_names) == n_cls else [str(i) for i in range(n_cls)]

                fig, ax = plt.subplots(figsize=(max(6, n_cls), max(5, n_cls - 1)))
                # Row-normalise
                row_sums = tm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                tm_norm = tm / row_sums

                if SEABORN_AVAILABLE:
                    import seaborn as sns
                    sns.heatmap(
                        tm_norm, annot=tm, fmt='d',
                        xticklabels=labels, yticklabels=labels,
                        cmap='YlOrRd', ax=ax,
                        linewidths=0.5, linecolor='gray'
                    )
                else:
                    im = ax.imshow(tm_norm, cmap='YlOrRd', aspect='auto')
                    plt.colorbar(im, ax=ax)
                    ax.set_xticks(range(n_cls))
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_yticks(range(n_cls))
                    ax.set_yticklabels(labels)
                    for i in range(n_cls):
                        for j in range(n_cls):
                            ax.text(j, i, str(tm[i, j]), ha='center', va='center', fontsize=8)

                ax.set_xlabel("Scenario Prediction")
                ax.set_ylabel("Baseline Prediction")
                ax.set_title(f"F8c — Class Transition Matrix\n(Scenario: {max_name})")
                plt.tight_layout()
                p = self.output_dir / "F8c_transition_matrix.png"
                self._savefig(p)
                plt.close()
                paths.append(p)
                logger.info(f"Saved F8c to {p}")

        return paths

    # ------------------------------------------------------------------
    # F9: Imbalance handling (before / after class distribution)
    # ------------------------------------------------------------------

    def f9_imbalance_handling(
        self,
        y_before: np.ndarray,
        y_after: np.ndarray,
        idx_to_label: Optional[Dict[int, str]] = None,
        strategy: str = ""
    ) -> Path:
        """
        F9: Side-by-side class distribution before and after resampling.

        Args:
            y_before     : original training labels
            y_after      : resampled training labels
            idx_to_label : mapping from int index to label string
            strategy     : name of the resampling strategy used

        Returns:
            Path to saved figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        def _count(y):
            cls, cnt = np.unique(y, return_counts=True)
            return {int(c): int(n) for c, n in zip(cls, cnt)}

        before = _count(y_before)
        after = _count(y_after)
        all_classes = sorted(set(before) | set(after))
        labels = [idx_to_label.get(c, str(c)) if idx_to_label else str(c) for c in all_classes]

        b_counts = [before.get(c, 0) for c in all_classes]
        a_counts = [after.get(c, 0) for c in all_classes]

        x = np.arange(len(all_classes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(all_classes) * 0.8), 5))
        bars_b = ax.bar(x - width / 2, b_counts, width, label='Before resampling', color='#4c72b0', alpha=0.85)
        bars_a = ax.bar(x + width / 2, a_counts, width, label='After resampling', color='#dd8452', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Sample Count")
        ax.set_title(f"F9 — Class Distribution Before vs After Resampling\n(Strategy: {strategy})")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Annotate bars
        for bar in bars_b:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(h),
                        ha='center', va='bottom', fontsize=7)
        for bar in bars_a:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(h),
                        ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        output_path = self.output_dir / "F9_imbalance_handling.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved F9 to {output_path}")
        return output_path

    def f_era5_analysis(
        self,
        data: Dict[int, pd.DataFrame],
        tier_col: str = 'Classification',
        tier_order: Optional[List[str]] = None,
    ) -> Path:
        """
        ERA5 climate feature analysis figure — 4-panel layout.

        Panel A: Annual precipitation (mm) by year — violin + strip.
        Panel B: Annual soil moisture (m³/m³) by year — violin + strip.
        Panel C: ERA5 annual precipitation split by quality tier (test year).
        Panel D: ERA5 annual soil moisture split by quality tier (test year).

        Requires era5_precip_annual_mm, era5_precip_monsoon_mm,
        era5_soil_moisture_annual, era5_soil_moisture_pre columns.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        era5_precip_col = 'era5_precip_annual_mm'
        era5_sm_col = 'era5_soil_moisture_annual'
        era5_mon_col = 'era5_precip_monsoon_mm'
        era5_pre_col = 'era5_precip_premonsoon_mm'
        era5_sm_mon = 'era5_soil_moisture_monsoon'
        era5_sm_pre = 'era5_soil_moisture_pre'

        # Build a flat DataFrame with all years
        frames = []
        for yr, df in sorted(data.items()):
            if df is None or len(df) == 0:
                continue
            row = df.copy()
            row['year'] = str(yr)
            frames.append(row)
        if not frames:
            logger.warning("ERA5 figure: no data available.")
            return None
        combined = pd.concat(frames, ignore_index=True)

        # Check at least one ERA5 column is present
        era5_cols_present = [c for c in [era5_precip_col, era5_sm_col] if c in combined.columns]
        if not era5_cols_present:
            logger.warning("ERA5 figure: ERA5 columns not found in data — skipping.")
            return None

        years = sorted(combined['year'].unique())
        has_tier = tier_col in combined.columns

        TIER_PALETTE = {
            'T1_Safe': '#27AE60',
            'T2_Marginal': '#F39C12',
            'T3_Restricted': '#E74C3C',
        }
        if tier_order is None:
            tier_order = ['T1_Safe', 'T2_Marginal', 'T3_Restricted']

        # Use the latest year for the tier-breakdown panels
        latest_year = years[-1]
        df_latest = combined[combined['year'] == latest_year].copy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ERA5-Land Climate Features — Temporal Trends & Water-Quality Association',
                     fontsize=13, fontweight='bold', y=1.01)

        year_palette = plt.cm.Blues(np.linspace(0.4, 0.85, len(years)))

        # ---- Panel A: Annual precipitation by year ----
        ax = axes[0, 0]
        if era5_precip_col in combined.columns:
            precip_by_year = [
                combined.loc[combined['year'] == yr, era5_precip_col].dropna().values
                for yr in years
            ]
            parts = ax.violinplot(precip_by_year, positions=range(len(years)),
                                  showmedians=True, showextrema=True)
            for i, (pc, yr_data) in enumerate(zip(parts['bodies'], precip_by_year)):
                pc.set_facecolor(year_palette[i])
                pc.set_alpha(0.75)
                jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(yr_data))
                ax.scatter(np.full(len(yr_data), i) + jitter, yr_data,
                           s=12, alpha=0.5, color=year_palette[i], zorder=3)
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels(years)
            ax.set_ylabel('Annual Precipitation (mm)')
            ax.set_title('(A) Annual Precipitation by Year')
            ax.grid(axis='y', alpha=0.3)

            # Add monsoon vs pre-monsoon breakdown as stacked bar inset
            if era5_mon_col in combined.columns and era5_pre_col in combined.columns:
                means_mon = [combined.loc[combined['year'] == yr, era5_mon_col].mean() for yr in years]
                means_pre = [combined.loc[combined['year'] == yr, era5_pre_col].mean() for yr in years]
                ax2 = ax.inset_axes([0.62, 0.60, 0.36, 0.35])
                xpos = np.arange(len(years))
                ax2.bar(xpos, means_mon, color='#1565C0', alpha=0.8, label='Monsoon', width=0.5)
                ax2.bar(xpos, means_pre, bottom=means_mon, color='#90CAF9', alpha=0.8,
                        label='Pre-monsoon', width=0.5)
                ax2.set_xticks(xpos)
                ax2.set_xticklabels(years, fontsize=6)
                ax2.set_ylabel('mm', fontsize=6)
                ax2.tick_params(labelsize=6)
                ax2.set_title('Seasonal', fontsize=7)
                ax2.legend(fontsize=5, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'ERA5 precipitation\nnot available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='grey')
            ax.set_title('(A) Annual Precipitation by Year')

        # ---- Panel B: Annual soil moisture by year ----
        ax = axes[0, 1]
        if era5_sm_col in combined.columns:
            sm_by_year = [
                combined.loc[combined['year'] == yr, era5_sm_col].dropna().values
                for yr in years
            ]
            parts = ax.violinplot(sm_by_year, positions=range(len(years)),
                                  showmedians=True, showextrema=True)
            for i, (pc, yr_data) in enumerate(zip(parts['bodies'], sm_by_year)):
                pc.set_facecolor(year_palette[i])
                pc.set_alpha(0.75)
                jitter = np.random.RandomState(43).uniform(-0.15, 0.15, len(yr_data))
                ax.scatter(np.full(len(yr_data), i) + jitter, yr_data,
                           s=12, alpha=0.5, color=year_palette[i], zorder=3)
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels(years)
            ax.set_ylabel('Soil Moisture (m³/m³)')
            ax.set_title('(B) Annual Soil Moisture by Year')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'ERA5 soil moisture\nnot available', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='grey')
            ax.set_title('(B) Annual Soil Moisture by Year')

        # ---- Panel C: Precipitation by quality tier (latest year) ----
        ax = axes[1, 0]
        if era5_precip_col in df_latest.columns and has_tier:
            tier_vals = {
                t: df_latest.loc[df_latest[tier_col] == t, era5_precip_col].dropna().values
                for t in tier_order if t in df_latest[tier_col].values
            }
            tiers_present = [t for t in tier_order if t in tier_vals and len(tier_vals[t]) > 0]
            if tiers_present:
                positions = range(len(tiers_present))
                parts = ax.violinplot(
                    [tier_vals[t] for t in tiers_present],
                    positions=positions, showmedians=True, showextrema=True
                )
                for pc, t in zip(parts['bodies'], tiers_present):
                    pc.set_facecolor(TIER_PALETTE.get(t, '#888'))
                    pc.set_alpha(0.75)
                for i, t in enumerate(tiers_present):
                    jitter = np.random.RandomState(44 + i).uniform(-0.12, 0.12, len(tier_vals[t]))
                    ax.scatter(np.full(len(tier_vals[t]), i) + jitter, tier_vals[t],
                               s=12, alpha=0.55, color=TIER_PALETTE.get(t, '#888'), zorder=3)
                # Annotate medians
                for i, t in enumerate(tiers_present):
                    med = float(np.median(tier_vals[t]))
                    ax.text(i, med, f'{med:.0f}', ha='center', va='bottom', fontsize=7,
                            fontweight='bold', color=TIER_PALETTE.get(t, '#333'))
                short_labels = [t.replace('T1_', 'T1\n').replace('T2_', 'T2\n').replace('T3_', 'T3\n')
                                for t in tiers_present]
                ax.set_xticks(positions)
                ax.set_xticklabels(short_labels, fontsize=9)
            ax.set_ylabel('Annual Precipitation (mm)')
            ax.set_title(f'(C) Precipitation by Quality Tier ({latest_year})')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No tier data\nfor {latest_year}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='grey')
            ax.set_title('(C) Precipitation by Quality Tier')

        # ---- Panel D: Soil moisture by quality tier (latest year) ----
        ax = axes[1, 1]
        if era5_sm_col in df_latest.columns and has_tier:
            tier_vals_sm = {
                t: df_latest.loc[df_latest[tier_col] == t, era5_sm_col].dropna().values
                for t in tier_order if t in df_latest[tier_col].values
            }
            tiers_present_sm = [t for t in tier_order if t in tier_vals_sm and len(tier_vals_sm[t]) > 0]
            if tiers_present_sm:
                positions = range(len(tiers_present_sm))
                parts = ax.violinplot(
                    [tier_vals_sm[t] for t in tiers_present_sm],
                    positions=positions, showmedians=True, showextrema=True
                )
                for pc, t in zip(parts['bodies'], tiers_present_sm):
                    pc.set_facecolor(TIER_PALETTE.get(t, '#888'))
                    pc.set_alpha(0.75)
                for i, t in enumerate(tiers_present_sm):
                    jitter = np.random.RandomState(47 + i).uniform(-0.12, 0.12, len(tier_vals_sm[t]))
                    ax.scatter(np.full(len(tier_vals_sm[t]), i) + jitter, tier_vals_sm[t],
                               s=12, alpha=0.55, color=TIER_PALETTE.get(t, '#888'), zorder=3)
                for i, t in enumerate(tiers_present_sm):
                    med = float(np.median(tier_vals_sm[t]))
                    ax.text(i, med, f'{med:.3f}', ha='center', va='bottom', fontsize=7,
                            fontweight='bold', color=TIER_PALETTE.get(t, '#333'))
                short_labels = [t.replace('T1_', 'T1\n').replace('T2_', 'T2\n').replace('T3_', 'T3\n')
                                for t in tiers_present_sm]
                ax.set_xticks(positions)
                ax.set_xticklabels(short_labels, fontsize=9)
            ax.set_ylabel('Soil Moisture (m³/m³)')
            ax.set_title(f'(D) Soil Moisture by Quality Tier ({latest_year})')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No tier data\nfor {latest_year}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='grey')
            ax.set_title('(D) Soil Moisture by Quality Tier')

        # Add a shared legend for tiers
        handles = [mpatches.Patch(color=TIER_PALETTE.get(t, '#888'),
                                  label=t.replace('_', ' ')) for t in tier_order]
        fig.legend(handles=handles, loc='lower center', ncol=len(tier_order),
                   fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        output_path = self.output_dir / "F_era5_climate_analysis.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved ERA5 figure to {output_path}")
        return output_path

    def f_era5_shap_spatial(
        self,
        nc_path: str,
        data: Dict[int, "pd.DataFrame"],
        shap_result: Any,
        feature_names: List[str],
        X_test: "np.ndarray",
        y_test: "np.ndarray",
        idx_to_label: Dict[int, str],
        source_year: int = 2019,
        importance_df: Optional["pd.DataFrame"] = None,
    ) -> Path:
        """
        Four-panel ERA5 + SHAP spatial figure.

        Panel A: ERA5 annual precipitation raster (pcolormesh) + well quality tier overlay.
        Panel B: ERA5 annual soil moisture raster + well quality tier overlay.
        Panel C: Mean |SHAP| bar chart for ERA5 features (from shap_result or importance_df).
        Panel D: SHAP dependence scatter when shap_result available; otherwise ERA5 feature
                 value distribution by quality tier (violin + strip) from the well data.

        Parameters
        ----------
        importance_df : optional pd.DataFrame with columns ['Feature','Importance'] or
                        ['feature','importance'] — used as Panel C fallback when shap_result
                        is None (e.g. ensemble best model where per-sample SHAP is unavailable).
        """
        import matplotlib.gridspec as gridspec

        TIER_PALETTE = {
            'T1_Safe':       '#27AE60',
            'T2_Marginal':   '#F39C12',
            'T3_Restricted': '#E74C3C',
        }
        tier_order = ['T1_Safe', 'T2_Marginal', 'T3_Restricted']

        # ── 1. Load ERA5 raster from netCDF ──────────────────────────────────
        raster_ok = False
        precip_raster = sm_raster = LON = LAT = None
        try:
            import netCDF4 as nc4
            ds = nc4.Dataset(nc_path, 'r')
            lats_raw = np.asarray(ds.variables['latitude'][:], dtype=float)
            lons_raw = np.asarray(ds.variables['longitude'][:], dtype=float)
            time_var = ds.variables['valid_time']
            try:
                import cftime
                times = nc4.num2date(time_var[:], units=time_var.units,
                                     calendar=getattr(time_var, 'calendar', 'standard'))
                year_arr = np.array([t.year for t in times], dtype=int)
            except Exception:
                # Fallback: assume uniform daily steps Jan 2018 – Dec 2020 (1095 days)
                n = len(time_var[:])
                import datetime as _dt
                base = _dt.date(2018, 1, 1)
                year_arr = np.array([(base + _dt.timedelta(days=i)).year for i in range(n)], dtype=int)

            year_mask = year_arr == source_year
            tp_raw  = np.asarray(ds.variables['tp'][year_mask], dtype=float)   # (n_days, lat, lon) m/day
            sm_raw  = np.asarray(ds.variables['swvl1'][year_mask], dtype=float) # (n_days, lat, lon) m³/m³
            ds.close()

            # Sort lats ascending (south → north) so pcolormesh puts south at bottom
            if lats_raw[0] > lats_raw[-1]:
                order = np.argsort(lats_raw)
                lats_raw = lats_raw[order]
                tp_raw  = tp_raw[:, order, :]
                sm_raw  = sm_raw[:, order, :]

            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter('ignore', RuntimeWarning)
                precip_raster = np.nanmean(tp_raw, axis=0) * 1000 * 365   # mm/yr
                sm_raster     = np.nanmean(sm_raw, axis=0)                 # m³/m³

            LON, LAT = np.meshgrid(lons_raw, lats_raw)
            raster_ok = True
        except Exception as exc:
            logger.warning(f"ERA5 raster load failed: {exc}")

        # ── 2. Well data for source_year ──────────────────────────────────────
        wells_df = data.get(source_year, pd.DataFrame())
        has_wells = (
            len(wells_df) > 0
            and 'lat_gis' in wells_df.columns
            and 'long_gis' in wells_df.columns
        )
        tier_col_name = 'Classification' if 'Classification' in (wells_df.columns if has_wells else []) else None

        # ── 3. SHAP analysis for ERA5 features ────────────────────────────────
        era5_feat_names = [f for f in feature_names if 'era5' in f.lower()]
        era5_feat_idx   = [feature_names.index(f) for f in era5_feat_names if f in feature_names]

        shap_ok = (shap_result is not None and hasattr(shap_result, 'shap_values')
                   and len(era5_feat_idx) > 0)

        # Normalise importance_df column names to lowercase
        _imp_df: Optional[pd.DataFrame] = None
        if importance_df is not None and len(importance_df) > 0:
            _imp_df = importance_df.copy()
            _imp_df.columns = [c.lower() for c in _imp_df.columns]
            if 'feature' not in _imp_df.columns or 'importance' not in _imp_df.columns:
                _imp_df = None

        sv2d = sv_t3 = None
        era5_imp = pd.DataFrame()
        if shap_ok:
            sv = shap_result.shap_values   # (n, n_feat) or (n, n_feat, n_classes)
            if sv.ndim == 3:
                sv2d  = np.mean(np.abs(sv), axis=2)          # mean-abs across classes
                t3_cls = max(idx_to_label.keys())
                sv_t3  = sv[:, :, t3_cls]                    # T3 class only
            else:
                sv2d  = np.abs(sv)
                sv_t3 = sv

            # Build ERA5 importance table from global_importance or sv2d
            gi = shap_result.global_importance
            era5_mask = gi['feature'].str.contains('era5', case=False, na=False)
            era5_imp = gi[era5_mask].copy()
            if len(era5_imp) == 0:
                era5_imp = pd.DataFrame({
                    'feature':    era5_feat_names,
                    'importance': [float(sv2d[:, i].mean()) for i in era5_feat_idx],
                })
        elif _imp_df is not None:
            # Full shap_result not available (e.g. ensemble winner) but importance CSV loaded
            era5_mask = _imp_df['feature'].str.contains('era5', case=False, na=False)
            era5_imp = _imp_df[era5_mask][['feature', 'importance']].copy()

        # ── 4. Figure layout ──────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 11))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
        ax_prec = fig.add_subplot(gs[0, 0])
        ax_sm   = fig.add_subplot(gs[0, 1])
        ax_bar  = fig.add_subplot(gs[1, 0])
        ax_dep  = fig.add_subplot(gs[1, 1])

        # helper: scatter wells on an axis
        def _plot_wells(ax):
            if not has_wells:
                return
            for tier in tier_order:
                sub = wells_df[wells_df[tier_col_name] == tier] if tier_col_name else wells_df
                if len(sub) == 0:
                    continue
                ax.scatter(sub['long_gis'], sub['lat_gis'],
                           c=TIER_PALETTE[tier], s=32, alpha=0.88,
                           edgecolors='white', linewidths=0.45, zorder=3,
                           label=tier.replace('_', ' '))

        # ── Panel A: Precipitation raster ─────────────────────────────────────
        if raster_ok:
            pcm_p = ax_prec.pcolormesh(LON, LAT, precip_raster,
                                       cmap='Blues', shading='auto')
            cb_p = plt.colorbar(pcm_p, ax=ax_prec, shrink=0.82, pad=0.02)
            cb_p.set_label('mm / yr', fontsize=8)
        else:
            ax_prec.set_facecolor('#ddeeff')
            ax_prec.text(0.5, 0.5, 'ERA5 raster\nnot available',
                         transform=ax_prec.transAxes, ha='center', va='center',
                         fontsize=10, color='grey')
        _plot_wells(ax_prec)
        ax_prec.set_xlabel('Longitude (°E)', fontsize=9)
        ax_prec.set_ylabel('Latitude (°N)', fontsize=9)
        ax_prec.set_title(f'(A) Annual Precipitation – {source_year}\n'
                          'ERA5-Land raster  ·  Well quality overlay', fontsize=9)
        ax_prec.tick_params(labelsize=8)

        # ── Panel B: Soil moisture raster ─────────────────────────────────────
        if raster_ok:
            pcm_s = ax_sm.pcolormesh(LON, LAT, sm_raster,
                                     cmap='YlGnBu', shading='auto')
            cb_s = plt.colorbar(pcm_s, ax=ax_sm, shrink=0.82, pad=0.02)
            cb_s.set_label('m³ / m³', fontsize=8)
        else:
            ax_sm.set_facecolor('#eeffee')
            ax_sm.text(0.5, 0.5, 'ERA5 raster\nnot available',
                       transform=ax_sm.transAxes, ha='center', va='center',
                       fontsize=10, color='grey')
        _plot_wells(ax_sm)
        ax_sm.set_xlabel('Longitude (°E)', fontsize=9)
        ax_sm.set_ylabel('Latitude (°N)', fontsize=9)
        ax_sm.set_title(f'(B) Annual Soil Moisture – {source_year}\n'
                        'ERA5-Land raster  ·  Well quality overlay', fontsize=9)
        ax_sm.tick_params(labelsize=8)

        # ── Panel C: ERA5 SHAP importance bar ─────────────────────────────────
        def _friendly(name: str) -> str:
            return (name.replace('era5_', '')
                        .replace('_annual_mm', ' (ann. mm)')
                        .replace('_monsoon_mm', ' (monsoon mm)')
                        .replace('_premonsoon_mm', ' (pre-mns. mm)')
                        .replace('_annual', ' (annual)')
                        .replace('_monsoon', ' (monsoon)')
                        .replace('_pre', ' (pre-mns.)'))

        if len(era5_imp) > 0:
            era5_sorted = era5_imp.sort_values('importance', ascending=True)
            labels = [_friendly(f) for f in era5_sorted['feature'].tolist()]
            vals   = era5_sorted['importance'].values

            cmap_b = plt.get_cmap('viridis', len(vals))
            bar_colors = [cmap_b(i / max(1, len(vals) - 1)) for i in range(len(vals))]

            bars = ax_bar.barh(range(len(vals)), vals, color=bar_colors,
                               edgecolor='white', linewidth=0.5)
            ax_bar.set_yticks(range(len(vals)))
            ax_bar.set_yticklabels(labels, fontsize=8)
            _c_xlabel = 'Mean |SHAP value|' if shap_ok else 'Feature Importance'
            _c_title  = ('(C) ERA5 Feature Importance\n(SHAP – mean |value|)' if shap_ok
                         else '(C) ERA5 Feature Importance\n(model feature importance)')
            ax_bar.set_xlabel(_c_xlabel, fontsize=9)
            ax_bar.set_title(_c_title, fontsize=9)
            ax_bar.grid(axis='x', alpha=0.3)
            max_val = vals.max() if vals.max() > 0 else 1.0
            for bar, v in zip(bars, vals):
                ax_bar.text(v + max_val * 0.015,
                            bar.get_y() + bar.get_height() / 2,
                            f'{v:.4f}', va='center', fontsize=7)
        else:
            ax_bar.text(0.5, 0.5, 'SHAP not\navailable',
                        transform=ax_bar.transAxes, ha='center', va='center',
                        fontsize=11, color='grey')
            ax_bar.set_title('(C) ERA5 Feature Importance', fontsize=9)

        # ── Panel D: SHAP dependence (top ERA5 feature → T3 class) ───────────
        if shap_ok and len(era5_imp) > 0 and X_test is not None:
            top_feat = era5_imp.sort_values('importance', ascending=False).iloc[0]['feature']
            if top_feat in feature_names:
                fidx   = feature_names.index(top_feat)
                x_vals = X_test[:, fidx]
                y_vals = sv_t3[:, fidx]

                for cls_idx, label in sorted(idx_to_label.items()):
                    mask = y_test == cls_idx
                    if mask.sum() == 0:
                        continue
                    ax_dep.scatter(x_vals[mask], y_vals[mask],
                                   c=TIER_PALETTE.get(label, '#888'),
                                   s=26, alpha=0.78,
                                   edgecolors='white', linewidths=0.35,
                                   label=label.replace('_', ' '), zorder=3)

                ax_dep.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.45)

                # Quadratic trend
                try:
                    z = np.polyfit(x_vals, y_vals, 2)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 120)
                    ax_dep.plot(x_line, np.polyval(z, x_line),
                                'k-', linewidth=1.6, alpha=0.55, label='Trend', zorder=4)
                except Exception:
                    pass

                ax_dep.set_xlabel(f'{_friendly(top_feat)} (standardised)', fontsize=9)
                ax_dep.set_ylabel('SHAP value → T3_Restricted', fontsize=9)
                ax_dep.set_title(f'(D) SHAP Dependence: {_friendly(top_feat)}\n'
                                 'Contribution to T3_Restricted prediction', fontsize=9)
                ax_dep.legend(fontsize=8, loc='best', framealpha=0.85)
                ax_dep.grid(alpha=0.3)
        else:
            # Fallback: violin + strip of top ERA5 feature value by quality tier
            # (from the source-year well data which has ERA5 columns merged in)
            _top_era5 = (era5_imp.sort_values('importance', ascending=False).iloc[0]['feature']
                         if len(era5_imp) > 0 else (era5_feat_names[0] if era5_feat_names else None))
            _dep_drawn = False
            if _top_era5 and has_wells and _top_era5 in wells_df.columns:
                _plot_df = wells_df[['Classification', _top_era5]].dropna()
                if len(_plot_df) > 5:
                    _tiers_present = [t for t in tier_order if t in _plot_df['Classification'].values]
                    positions = list(range(len(_tiers_present)))
                    for pos, tier in zip(positions, _tiers_present):
                        vals_t = _plot_df.loc[_plot_df['Classification'] == tier, _top_era5].values
                        # Violin body
                        if len(vals_t) >= 3:
                            try:
                                vp = ax_dep.violinplot([vals_t], positions=[pos],
                                                       widths=0.5, showmedians=True,
                                                       showextrema=True)
                                for pc in vp['bodies']:
                                    pc.set_facecolor(TIER_PALETTE[tier])
                                    pc.set_alpha(0.55)
                                for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
                                    if part in vp:
                                        vp[part].set_color(TIER_PALETTE[tier])
                                        vp[part].set_linewidth(1.5)
                            except Exception:
                                pass
                        # Strip
                        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals_t))
                        ax_dep.scatter(np.full(len(vals_t), pos) + jitter, vals_t,
                                       c=TIER_PALETTE[tier], s=18, alpha=0.60,
                                       edgecolors='white', linewidths=0.3, zorder=3)
                    short = [t.replace('T1_', 'T1\n').replace('T2_', 'T2\n').replace('T3_', 'T3\n')
                             for t in _tiers_present]
                    ax_dep.set_xticks(positions)
                    ax_dep.set_xticklabels(short, fontsize=9)
                    ax_dep.set_ylabel(_friendly(_top_era5), fontsize=9)
                    ax_dep.set_title(f'(D) {_friendly(_top_era5)} by Quality Tier\n'
                                     f'({source_year} wells — ERA5 merged values)', fontsize=9)
                    ax_dep.grid(axis='y', alpha=0.3)
                    _dep_drawn = True
            if not _dep_drawn:
                ax_dep.text(0.5, 0.5, 'SHAP not\navailable',
                            transform=ax_dep.transAxes, ha='center', va='center',
                            fontsize=11, color='grey')
                ax_dep.set_title('(D) SHAP Dependence (top ERA5 feature)', fontsize=9)

        # ── Shared tier legend ────────────────────────────────────────────────
        legend_handles = [
            mpatches.Patch(color=TIER_PALETTE[t], label=t.replace('_', ' '))
            for t in tier_order
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=3,
                   fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01),
                   title='Well Quality Tier', title_fontsize=9)

        fig.suptitle(
            f'ERA5-Land Climate Context and SHAP Attribution ({source_year})\n'
            'Spatial Groundwater Quality Distribution + Climate Feature Contributions',
            fontsize=11, fontweight='bold', y=1.02
        )

        output_path = self.output_dir / "F_era5_shap_spatial.png"
        self._savefig(output_path)
        plt.close()
        logger.info(f"Saved ERA5+SHAP spatial figure to {output_path}")
        return output_path
