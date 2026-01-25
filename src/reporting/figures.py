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
        pareto_fronts: Dict[str, np.ndarray],
        objective_names: List[str] = ['Ordinal Distance', 'Severe FNR']
    ) -> Path:
        """
        F5: Pareto fronts per model.

        Args:
            pareto_fronts: Dict mapping model name to objectives array
            objective_names: Names of objectives for axes
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(pareto_fronts)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

        for (model_name, objectives), color, marker in zip(
            pareto_fronts.items(), colors, markers
        ):
            if len(objectives) > 0:
                ax.scatter(
                    objectives[:, 0], objectives[:, 1],
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
