"""
Table Generator Module
======================

Generates publication-ready tables for the paper.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TableGenerator:
    """
    Generate publication-ready tables for JEM paper.
    """

    output_dir: Path = field(default_factory=lambda: Path("outputs/paper_outputs/tables"))

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def t1_dataset_overview(
        self,
        data: Dict[int, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Path]:
        """
        T1: Dataset columns + missingness per year.
        """
        records = []

        # Get all columns across years
        all_columns = set()
        for df in data.values():
            all_columns.update(df.columns)

        for col in sorted(all_columns):
            row = {'Column': col}
            for year, df in sorted(data.items()):
                if col in df.columns:
                    missing_rate = df[col].isna().mean()
                    row[f'{year}_Missing%'] = f"{missing_rate*100:.1f}"
                    row[f'{year}_Present'] = "Yes"
                else:
                    row[f'{year}_Missing%'] = "N/A"
                    row[f'{year}_Present'] = "No"
            records.append(row)

        df_table = pd.DataFrame(records)

        # Save
        output_path = self.output_dir / "T1_dataset_overview.csv"
        df_table.to_csv(output_path, index=False)

        # Also save LaTeX version
        latex_path = self.output_dir / "T1_dataset_overview.tex"
        df_table.to_latex(latex_path, index=False, escape=True)

        logger.info(f"Saved T1 to {output_path}")
        return df_table, output_path

    def t2_label_mapping(
        self,
        label_encoder: Dict[str, int],
        high_risk_classes: List[str]
    ) -> Tuple[pd.DataFrame, Path]:
        """
        T2: Label mapping + ordinal encoding + high-risk definition.
        """
        import re

        records = []
        for label, idx in sorted(label_encoder.items(), key=lambda x: x[1]):
            # Parse C and S components
            match = re.match(r'C(\d)S(\d)', label)
            if match:
                c_val, s_val = int(match.group(1)), int(match.group(2))
            else:
                c_val, s_val = None, None

            records.append({
                'Label': label,
                'Index': idx,
                'C_Component': c_val,
                'S_Component': s_val,
                'High_Risk': "Yes" if label in high_risk_classes else "No"
            })

        df_table = pd.DataFrame(records)

        output_path = self.output_dir / "T2_label_mapping.csv"
        df_table.to_csv(output_path, index=False)

        logger.info(f"Saved T2 to {output_path}")
        return df_table, output_path

    def t3_hyperparameter_bounds(
        self,
        model_configs: Dict[str, Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, Path]:
        """
        T3: Hyperparameter bounds per model.
        """
        records = []

        for model_name, config in model_configs.items():
            params = config.get('params', {})
            for param_name, bounds in params.items():
                if isinstance(bounds, (list, tuple)):
                    records.append({
                        'Model': model_name,
                        'Parameter': param_name,
                        'Lower_Bound': min(bounds),
                        'Upper_Bound': max(bounds),
                        'Type': 'discrete' if all(isinstance(x, int) for x in bounds) else 'continuous'
                    })

        df_table = pd.DataFrame(records)

        output_path = self.output_dir / "T3_hyperparameter_bounds.csv"
        df_table.to_csv(output_path, index=False)

        logger.info(f"Saved T3 to {output_path}")
        return df_table, output_path

    def t4_best_configurations(
        self,
        model_results: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Path]:
        """
        T4: Best configuration per model + objective values.

        Args:
            model_results: Dict mapping model name to ConfigurationResult or dict
        """
        records = []

        for model_name, result in model_results.items():
            # Handle both ConfigurationResult dataclass and dict
            if hasattr(result, 'params'):
                # ConfigurationResult dataclass
                row = {'Model': model_name}

                # Add objectives (CV or test)
                objectives = result.test_objectives if result.test_objectives is not None else result.cv_objectives
                if objectives is not None:
                    obj_names = ['ordinal_distance', 'severe_fnr', 'macro_f1_complement', 'complexity']
                    for i, name in enumerate(obj_names):
                        if i < len(objectives):
                            row[name] = objectives[i]

                # Add key parameters
                params = result.params if hasattr(result, 'params') else {}
                for param_name, value in params.items():
                    row[f'param_{param_name}'] = value

                # Add feature count
                if result.feature_mask is not None:
                    row['n_features'] = int(result.feature_mask.sum())
                else:
                    row['n_features'] = 'N/A'

            else:
                # Dict format
                row = {
                    'Model': model_name,
                    'VIKOR_Q': result.get('vikor_q', np.nan)
                }

                # Add objectives
                objectives = result.get('objectives', {})
                for obj_name, value in objectives.items():
                    row[obj_name] = value

                # Add key parameters
                params = result.get('params', {})
                for param_name, value in params.items():
                    row[f'param_{param_name}'] = value

                row['n_features'] = result.get('n_features', 'N/A')

            records.append(row)

        df_table = pd.DataFrame(records)

        output_path = self.output_dir / "T4_best_configurations.csv"
        df_table.to_csv(output_path, index=False)

        logger.info(f"Saved T4 to {output_path}")
        return df_table, output_path

    def t5_test_performance(
        self,
        results: pd.DataFrame,
        bootstrap_ci: Optional[Dict[str, Dict[str, Tuple]]] = None
    ) -> Tuple[pd.DataFrame, Path]:
        """
        T5: Final temporal-forward test performance with CIs.
        """
        df_table = results.copy()

        # Add confidence intervals if provided
        if bootstrap_ci:
            for model in df_table['Model'].values:
                if model in bootstrap_ci:
                    for metric, (point, lower, upper) in bootstrap_ci[model].items():
                        col_name = f'{metric}_CI'
                        df_table.loc[df_table['Model'] == model, col_name] = f"[{lower:.3f}, {upper:.3f}]"

        output_path = self.output_dir / "T5_test_performance.csv"
        df_table.to_csv(output_path, index=False)

        logger.info(f"Saved T5 to {output_path}")
        return df_table, output_path

    def t6_shap_drivers(
        self,
        global_importance: pd.DataFrame,
        high_risk_importance: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Path]:
        """
        T6: SHAP top drivers (global + high-risk).
        """
        df_table = global_importance[['feature', 'importance', 'rank']].copy()
        df_table.columns = ['Feature', 'Global_Importance', 'Global_Rank']

        if high_risk_importance is not None:
            hr_cols = high_risk_importance[['feature', 'importance', 'rank']].copy()
            hr_cols.columns = ['Feature', 'HighRisk_Importance', 'HighRisk_Rank']
            df_table = pd.merge(df_table, hr_cols, on='Feature', how='outer')

        output_path = self.output_dir / "T6_shap_drivers.csv"
        df_table.to_csv(output_path, index=False)

        logger.info(f"Saved T6 to {output_path}")
        return df_table, output_path

    def t7_scenario_outcomes(
        self,
        district_impacts: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Path]:
        """
        T7: Scenario outcomes by district ranking.
        """
        # Get top vulnerable districts per scenario
        df_table = district_impacts.sort_values(
            ['scenario', 'vulnerability_rank']
        ).groupby('scenario').head(10)

        output_path = self.output_dir / "T7_scenario_outcomes.csv"
        df_table.to_csv(output_path, index=False)

        logger.info(f"Saved T7 to {output_path}")
        return df_table, output_path

    def generate_summary_statistics(
        self,
        data: Dict[int, pd.DataFrame],
        numeric_cols: List[str]
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Generate summary statistics table for numeric columns.
        """
        records = []

        for year, df in sorted(data.items()):
            for col in numeric_cols:
                if col not in df.columns:
                    continue

                values = df[col].dropna()
                if len(values) == 0:
                    continue

                records.append({
                    'Year': year,
                    'Variable': col,
                    'N': len(values),
                    'Mean': values.mean(),
                    'Std': values.std(),
                    'Min': values.min(),
                    'Q25': values.quantile(0.25),
                    'Median': values.median(),
                    'Q75': values.quantile(0.75),
                    'Max': values.max()
                })

        df_table = pd.DataFrame(records)

        output_path = self.output_dir / "summary_statistics.csv"
        df_table.to_csv(output_path, index=False)

        logger.info(f"Saved summary statistics to {output_path}")
        return df_table, output_path
