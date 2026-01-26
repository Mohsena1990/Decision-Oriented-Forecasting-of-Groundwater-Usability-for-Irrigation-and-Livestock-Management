"""
Managerial Insights Module
==========================

Generate actionable insights for water resource managers.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ManagerialInsights:
    """
    Generate managerial insights from model results.

    Translates technical findings into actionable recommendations
    for water resource managers.
    """

    output_dir: Path = field(default_factory=lambda: Path("outputs/paper_outputs"))

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_key_findings(
        self,
        model_results: Dict[str, Any],
        shap_results: Optional[Dict[str, Any]] = None,
        scenario_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate key findings summary.

        Returns:
            Markdown-formatted key findings
        """
        findings = ["# Key Findings\n"]

        # Model performance findings
        findings.append("## Model Performance\n")
        if model_results:
            best_model = model_results.get('best_model', 'N/A')
            findings.append(f"- **Best Model**: {best_model}\n")

            if 'test_metrics' in model_results:
                metrics = model_results['test_metrics']
                findings.append(f"- **Macro F1**: {self._fmt_float(metrics.get('macro_f1'))}\n")
                findings.append(f"- **Severe FNR**: {self._fmt_float(metrics.get('severe_fnr'))}\n")
                findings.append(f"- **Ordinal Distance**: {self._fmt_float(metrics.get('ordinal_distance'))}\n")

                metrics = model_results['test_metrics']
                # findings.append(f"- **Macro F1**: {metrics.get('macro_f1', 'N/A'):.3f}\n")
                # findings.append(f"- **Severe FNR**: {metrics.get('severe_fnr', 'N/A'):.3f}\n")
                # findings.append(f"- **Ordinal Distance**: {metrics.get('ordinal_distance', 'N/A'):.3f}\n")

        # SHAP findings
        if shap_results:
            findings.append("\n## Key Drivers of Water Quality Risk\n")
            top_features = shap_results.get('top_features', [])
            for i, feature in enumerate(top_features[:5], 1):
                findings.append(f"{i}. **{feature}**\n")

        # Scenario findings
        if scenario_results:
            findings.append("\n## Scenario Analysis Insights\n")
            for scenario_name, result in scenario_results.items():
                change = result.get('high_risk_rate_change', 0) * 100
                direction = "increase" if change > 0 else "decrease"
                findings.append(f"- **{scenario_name}**: {abs(change):.1f}% {direction} in high-risk rate\n")

        return "\n".join(findings)

    def generate_recommendations(
        self,
        shap_top_features: List[str],
        vulnerable_districts: List[str],
        high_impact_scenarios: List[str]
    ) -> str:
        """
        Generate actionable recommendations.
        """
        recommendations = ["# Recommendations for Water Resource Management\n"]

        # Monitoring recommendations
        recommendations.append("## Priority Monitoring\n")
        recommendations.append("Based on the analysis, the following parameters should be prioritized for monitoring:\n")
        for feature in shap_top_features[:5]:
            recommendations.append(f"- **{feature}**: High influence on water quality classification\n")

        # Geographic prioritization
        if vulnerable_districts:
            recommendations.append("\n## Geographic Prioritization\n")
            recommendations.append("The following districts show highest vulnerability to water quality degradation:\n")
            for district in vulnerable_districts[:10]:
                recommendations.append(f"- {district}\n")

        # Scenario-based recommendations
        if high_impact_scenarios:
            recommendations.append("\n## Risk Mitigation Scenarios\n")
            recommendations.append("Scenarios with highest impact on water quality risk:\n")
            for scenario in high_impact_scenarios:
                recommendations.append(f"- {scenario}\n")

        # General recommendations
        recommendations.append("\n## General Recommendations\n")
        recommendations.append("""
1. **Early Warning System**: Implement the forecasting model for 1-year ahead predictions to enable proactive management.

2. **Targeted Interventions**: Focus resources on high-risk areas identified by the model, particularly districts showing consistent degradation trends.

3. **Parameter Thresholds**: Establish monitoring thresholds for key drivers (TDS, SAR, RSC) based on model sensitivity analysis.

4. **Data Collection**: Ensure consistent annual data collection to maintain forecasting capability and track long-term trends.

5. **Stakeholder Communication**: Use the classification system (C1-C4, S1-S4) to communicate water quality status to farmers and water users.
""")

        return "\n".join(recommendations)

    def generate_executive_summary(
        self,
        findings: str,
        recommendations: str
    ) -> str:
        """
        Generate executive summary combining findings and recommendations.
        """
        summary = [
            "# Executive Summary: Groundwater Quality Forecasting Framework\n",
            "## Overview\n",
            "This research implements a decision-ready spatio-temporal forecasting framework ",
            "for predicting next-year groundwater usability risk classes. The framework uses ",
            "machine learning to forecast water quality classifications (C#S#) based on ",
            "hydrochemical parameters.\n",
            "\n",
            findings,
            "\n",
            recommendations
        ]

        return "".join(summary)

    def save_insights(
        self,
        findings: str,
        recommendations: str
    ) -> Dict[str, Path]:
        """
        Save all insights to files.
        """
        paths = {}

        # Save findings
        findings_path = self.output_dir / "key_findings.md"
        with open(findings_path, 'w') as f:
            f.write(findings)
        paths['findings'] = findings_path

        # Save recommendations
        recs_path = self.output_dir / "recommendations.md"
        with open(recs_path, 'w') as f:
            f.write(recommendations)
        paths['recommendations'] = recs_path

        # Save executive summary
        summary = self.generate_executive_summary(findings, recommendations)
        summary_path = self.output_dir / "executive_summary.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
        paths['executive_summary'] = summary_path

        logger.info(f"Saved insights to {self.output_dir}")
        return paths
    
    @staticmethod
    def _fmt_float(value: Any, decimals: int = 3, na: str = "N/A") -> str:
        """Format a value as a float with fixed decimals; return N/A if missing/not numeric."""
        if value is None:
            return na
        # Handle common "N/A" strings
        if isinstance(value, str) and value.strip().lower() in {"n/a", "na", "none", ""}:
            return na
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return na

