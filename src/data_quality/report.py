"""
Data Quality Report Module
==========================

Generates comprehensive data quality reports for documentation
and paper outputs.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """
    Generate and export data quality reports.

    Combines validation results and cleaning logs into
    comprehensive reports suitable for paper supplementary materials.
    """

    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        validation_results: Dict[str, Any],
        cleaning_log: pd.DataFrame,
        data_summary: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate full data quality report.

        Args:
            validation_results: Results from DataQualityValidator
            cleaning_log: Log from DataCleaner
            data_summary: Summary statistics from DataIngestion

        Returns:
            Dictionary of report DataFrames
        """
        reports = {}

        # Table 1: Dataset overview
        reports['dataset_overview'] = self._create_overview_table(data_summary)

        # Table 2: Missingness report
        reports['missingness'] = self._create_missingness_table(validation_results)

        # Table 3: Class distribution
        reports['class_distribution'] = self._create_class_table(validation_results)

        # Table 4: Cleaning actions
        reports['cleaning_actions'] = cleaning_log

        # Table 5: Location coverage
        reports['location_coverage'] = self._create_location_table(validation_results)

        # Table 6: Outlier summary
        reports['outliers'] = self._create_outlier_table(validation_results)

        # Export if output directory specified
        if self.output_dir:
            self._export_reports(reports)

        return reports

    def _create_overview_table(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Create dataset overview table."""
        if summary is None or summary.empty:
            return pd.DataFrame()

        overview = summary.copy()
        overview.columns = [c.replace('_', ' ').title() for c in overview.columns]
        return overview

    def _create_missingness_table(self, validation_results: Dict) -> pd.DataFrame:
        """Create missingness analysis table."""
        if 'missingness' not in validation_results:
            return pd.DataFrame()

        result = validation_results['missingness']
        rows = []

        for year, details in result.details.get('years', {}).items():
            for col, rate in details.get('missing_rates', {}).items():
                rows.append({
                    'year': year,
                    'column': col,
                    'missing_rate': rate,
                    'missing_pct': f"{rate*100:.1f}%"
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            # Pivot for better readability
            pivot = df.pivot(index='column', columns='year', values='missing_pct')
            return pivot.reset_index()

        return df

    def _create_class_table(self, validation_results: Dict) -> pd.DataFrame:
        """Create class distribution table."""
        if 'class_distribution' not in validation_results:
            return pd.DataFrame()

        result = validation_results['class_distribution']
        rows = []

        for year, details in result.details.get('years', {}).items():
            for cls, count in details.get('counts', {}).items():
                prop = details.get('proportions', {}).get(cls, 0)
                rows.append({
                    'year': year,
                    'class': cls,
                    'count': count,
                    'proportion': prop,
                    'proportion_pct': f"{prop*100:.1f}%"
                })

        return pd.DataFrame(rows)

    def _create_location_table(self, validation_results: Dict) -> pd.DataFrame:
        """Create location coverage table."""
        if 'location_coverage' not in validation_results:
            return pd.DataFrame()

        result = validation_results['location_coverage']
        details = result.details

        rows = []
        rows.append({
            'metric': 'Total unique locations',
            'value': details.get('total_unique_locations', 'N/A')
        })
        rows.append({
            'metric': 'Common across all years',
            'value': details.get('common_across_all_years', 'N/A')
        })

        for year, count in details.get('locations_per_year', {}).items():
            rows.append({
                'metric': f'Locations in {year}',
                'value': count
            })

        for key, value in details.items():
            if key.startswith('overlap_'):
                rows.append({
                    'metric': f'Overlap: {key.replace("overlap_", "").replace("_", " -> ")}',
                    'value': value
                })

        return pd.DataFrame(rows)

    def _create_outlier_table(self, validation_results: Dict) -> pd.DataFrame:
        """Create outlier summary table."""
        if 'outliers' not in validation_results:
            return pd.DataFrame()

        result = validation_results['outliers']
        rows = []

        for year, year_details in result.details.get('years', {}).items():
            for col, stats in year_details.items():
                rows.append({
                    'year': year,
                    'column': col,
                    'n_outliers': stats.get('n_outliers', 0),
                    'outlier_rate': f"{stats.get('outlier_rate', 0)*100:.1f}%",
                    'min': stats.get('min'),
                    'max': stats.get('max'),
                    'median': stats.get('median')
                })

        return pd.DataFrame(rows)

    def _export_reports(self, reports: Dict[str, pd.DataFrame]):
        """Export all reports to CSV files."""
        for name, df in reports.items():
            if df is not None and not df.empty:
                filepath = self.output_dir / f"dq_{name}.csv"
                df.to_csv(filepath, index=False)
                logger.info(f"Exported report: {filepath}")

    def generate_latex_tables(self, reports: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Generate LaTeX versions of tables for paper."""
        latex_tables = {}

        for name, df in reports.items():
            if df is None or df.empty:
                continue

            latex = df.to_latex(index=False, escape=True)
            latex_tables[name] = latex

            if self.output_dir:
                filepath = self.output_dir / f"dq_{name}.tex"
                with open(filepath, 'w') as f:
                    f.write(latex)
                logger.info(f"Exported LaTeX table: {filepath}")

        return latex_tables

    def generate_summary_stats(
        self,
        data: Dict[int, pd.DataFrame],
        numeric_cols: List[str]
    ) -> pd.DataFrame:
        """Generate summary statistics for numeric columns."""
        all_stats = []

        for year, df in data.items():
            for col in numeric_cols:
                if col not in df.columns:
                    continue

                values = df[col].dropna()
                if len(values) == 0:
                    continue

                stats = {
                    'year': year,
                    'column': col,
                    'count': len(values),
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'q25': values.quantile(0.25),
                    'median': values.median(),
                    'q75': values.quantile(0.75),
                    'max': values.max()
                }
                all_stats.append(stats)

        return pd.DataFrame(all_stats)
