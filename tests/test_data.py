"""
Tests for data ingestion and processing modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataIngestion, DataHarmonizer, TransitionBuilder
from src.preprocessing.encoders import LabelParser, OrdinalEncoder


class TestLabelParser:
    """Tests for LabelParser."""

    def setup_method(self):
        self.parser = LabelParser()

    def test_parse_valid_labels(self):
        """Test parsing valid C#S# labels."""
        assert self.parser.parse('C2S1') == (2, 1)
        assert self.parser.parse('C4S3') == (4, 3)
        assert self.parser.parse('c3s2') == (3, 2)  # Case insensitive

    def test_parse_invalid_labels(self):
        """Test parsing invalid labels."""
        assert self.parser.parse('Invalid') is None
        assert self.parser.parse('C5S1') is None  # Out of range
        assert self.parser.parse('') is None
        assert self.parser.parse(None) is None

    def test_ordinal_distance(self):
        """Test ordinal distance calculation."""
        dist = self.parser.compute_ordinal_distance('C2S1', 'C4S3')
        assert dist == 4  # |2-4| + |1-3| = 2 + 2 = 4

        dist = self.parser.compute_ordinal_distance('C2S1', 'C2S1')
        assert dist == 0

    def test_high_risk_detection(self):
        """Test high-risk class detection."""
        assert self.parser.is_high_risk('C4S3') == True
        assert self.parser.is_high_risk('C4S1') == True
        assert self.parser.is_high_risk('C2S1') == False
        assert self.parser.is_high_risk('C3S2') == False


class TestOrdinalEncoder:
    """Tests for OrdinalEncoder."""

    def setup_method(self):
        self.encoder = OrdinalEncoder()
        self.labels = pd.Series(['C2S1', 'C3S2', 'C4S1', 'C2S1', 'C3S1'])

    def test_fit_and_transform(self):
        """Test fitting and transforming labels."""
        self.encoder.fit(self.labels)

        encoded = self.encoder.transform(self.labels)
        assert len(encoded) == len(self.labels)
        assert encoded.dtype == np.int64 or encoded.dtype == int

    def test_inverse_transform(self):
        """Test inverse transformation."""
        self.encoder.fit(self.labels)
        encoded = self.encoder.transform(self.labels)
        decoded = self.encoder.inverse_transform(encoded)

        assert decoded == self.labels.tolist()

    def test_class_ordering(self):
        """Test that classes are ordered by C then S."""
        self.encoder.fit(self.labels)
        classes = self.encoder.classes

        # C2S1 should come before C3S1, C3S2, C4S1
        assert classes.index('C2S1') < classes.index('C3S1')
        assert classes.index('C3S1') < classes.index('C3S2')
        assert classes.index('C3S2') < classes.index('C4S1')


class TestTransitionBuilder:
    """Tests for TransitionBuilder."""

    def setup_method(self):
        self.builder = TransitionBuilder(
            location_keys=['district', 'village'],
            tolerance=0.001
        )

    def test_build_transitions(self):
        """Test building transition pairs."""
        data = {
            2018: pd.DataFrame({
                'district': ['A', 'B', 'C'],
                'village': ['V1', 'V2', 'V3'],
                'lat_gis': [10.0, 11.0, 12.0],
                'long_gis': [75.0, 76.0, 77.0],
                'TDS': [100, 200, 300],
                'Classification': ['C2S1', 'C3S1', 'C4S1']
            }),
            2019: pd.DataFrame({
                'district': ['A', 'B', 'D'],  # C is missing, D is new
                'village': ['V1', 'V2', 'V4'],
                'lat_gis': [10.0, 11.0, 13.0],
                'long_gis': [75.0, 76.0, 78.0],
                'TDS': [110, 210, 400],
                'Classification': ['C2S2', 'C3S2', 'C4S2']
            })
        }

        transitions = self.builder.build_transitions(
            data, 2018, 2019, target_col='Classification'
        )

        # Should have 2 matched locations (A-V1 and B-V2)
        assert len(transitions) == 2
        assert 'Classification_target' in transitions.columns


class TestDataLeakage:
    """Tests to ensure no data leakage."""

    def test_temporal_split_no_leakage(self):
        """Ensure no future data leaks into training."""
        from src.evaluation import TemporalSplitter

        transitions = {
            '2018_2019': pd.DataFrame({
                'from_year': [2018] * 10,
                'to_year': [2019] * 10,
                'features': range(10)
            }),
            '2019_2020': pd.DataFrame({
                'from_year': [2019] * 10,
                'to_year': [2020] * 10,
                'features': range(10, 20)
            })
        }

        splitter = TemporalSplitter(
            train_transitions=['2018_2019'],
            test_transitions=['2019_2020']
        )

        train_df, test_df = splitter.split(transitions)

        # All train data should be from 2018->2019
        assert all(train_df['from_year'] == 2018)
        assert all(train_df['to_year'] == 2019)

        # All test data should be from 2019->2020
        assert all(test_df['from_year'] == 2019)
        assert all(test_df['to_year'] == 2020)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
