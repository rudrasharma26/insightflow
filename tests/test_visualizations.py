"""
Unit tests for src/visualizations.py
"""

import io
import pandas as pd
import pytest
import plotly.graph_objects as go
from src.visualizations import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_scatter,
    plot_categorical_counts,
    plot_missing_values_bar,
)
from src.data_processor import get_missing_values_summary


@pytest.fixture
def sample_df():
    csv_data = """id,name,salary,experience,department
101,Alice,85000,4.5,Engineering
102,Bob,72000,6.0,Marketing
103,Charlie,110000,8.0,Engineering
104,Diana,65000,3.0,Sales
105,Evan,92000,,Engineering
"""
    return pd.read_csv(io.StringIO(csv_data))


def test_plot_distribution(sample_df):
    fig_hist = plot_distribution(sample_df, "salary", plot_type="Histogram")
    assert isinstance(fig_hist, go.Figure)
    
    fig_box = plot_distribution(sample_df, "salary", plot_type="Box Plot")
    assert isinstance(fig_box, go.Figure)

    fig_violin = plot_distribution(sample_df, "salary", plot_type="Violin Plot")
    assert isinstance(fig_violin, go.Figure)


def test_plot_correlation_heatmap(sample_df):
    fig = plot_correlation_heatmap(sample_df, ["salary", "experience"])
    assert isinstance(fig, go.Figure)


def test_plot_scatter(sample_df):
    fig = plot_scatter(sample_df, "experience", "salary", color_col="department", add_trendline=False)
    assert isinstance(fig, go.Figure)


def test_plot_categorical_counts(sample_df):
    fig_bar = plot_categorical_counts(sample_df, "department", chart_type="Bar")
    assert isinstance(fig_bar, go.Figure)

    fig_donut = plot_categorical_counts(sample_df, "department", chart_type="Donut")
    assert isinstance(fig_donut, go.Figure)


def test_plot_missing_values_bar(sample_df):
    missing_summary = get_missing_values_summary(sample_df)
    fig = plot_missing_values_bar(missing_summary)
    assert isinstance(fig, go.Figure)


def test_visualizations_small_dataframe():
    df_small = pd.DataFrame({"x": [10, 20, 30], "y": [1.5, 2.5, 3.5], "cat": ["A", "B", "A"]})
    assert isinstance(plot_distribution(df_small, "x"), go.Figure)
    assert isinstance(plot_correlation_heatmap(df_small, ["x", "y"]), go.Figure)
    assert isinstance(plot_scatter(df_small, "x", "y", "cat", add_trendline=True), go.Figure)
    assert isinstance(plot_categorical_counts(df_small, "cat"), go.Figure)
    missing_summary = get_missing_values_summary(df_small)
    assert isinstance(plot_missing_values_bar(missing_summary), go.Figure)

