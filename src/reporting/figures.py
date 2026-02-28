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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved F3 to {output_path}")
        return output_path

    def f4_model_comparison(
        self,
        results: pd.DataFrame,
        metrics: List[str] = ['macro_f1', 'severe_fnr', 'ordinal_distance']
    ) -> Path:
        """
        F4: Model comparison bar chart.

        Args:
            results: DataFrame with model results
            metrics: Metrics to compare
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        n_metrics = len(metrics)
        n_models = len(results)

        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for ax, metric in zip(axes, metrics):
            if metric not in results.columns:
                continue

            values = results[metric].values
            models = results['Model'].values if 'Model' in results else results.index

            bars = ax.bar(range(n_models), values, color=colors)
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()}')

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Model Performance Comparison', fontsize=14)
        plt.tight_layout()

        output_path = self.output_dir / "F4_model_comparison.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
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

        colors = plt.cm.tab10(np.linspace(0, 1, len(optimization_results)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

        for (model_name, opt_result), color, marker in zip(
            optimization_results.items(), colors, markers
        ):
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
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

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Q Score comparison
        ax1 = axes[0]
        models = comparison_df['Model'].values
        q_scores = comparison_df['Q_Score'].values if 'Q_Score' in comparison_df.columns else np.zeros(len(models))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
        bars = ax1.barh(range(len(models)), q_scores, color=colors)
        ax1.set_yticks(range(len(models)))
        ax1.set_yticklabels(models)
        ax1.set_xlabel('VIKOR Q Score (lower is better)')
        ax1.set_title('Model Rankings by VIKOR Q Score')

        for bar, val in zip(bars, q_scores):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='left', va='center', fontsize=10)

        # Plot 2: Objectives radar chart (simplified as bar chart)
        ax2 = axes[1]
        obj_cols = [c for c in comparison_df.columns if c not in ['Model', 'Rank', 'Q_Score', 'n_features', 'In_Compromise_Set']]

        if obj_cols:
            x = np.arange(len(obj_cols))
            width = 0.8 / len(models)

            for i, (_, row) in enumerate(comparison_df.iterrows()):
                values = [row[c] for c in obj_cols if c in row]
                ax2.bar(x + i * width, values, width, label=row['Model'], alpha=0.8)

            ax2.set_xticks(x + width * (len(models) - 1) / 2)
            ax2.set_xticklabels([c.replace('_', '\n') for c in obj_cols], rotation=0)
            ax2.set_ylabel('Objective Value')
            ax2.set_title('Objective Values by Model')
            ax2.legend(loc='upper right')

        plt.tight_layout()

        output_path = self.output_dir / "F6_vikor_rankings.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
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
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
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

    subgraph Stage_E["Stage E: Model Candidates"]
        E1[CatBoost]
        E2[LightGBM]
        E3[GRU]
        E4[LSTM]
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
