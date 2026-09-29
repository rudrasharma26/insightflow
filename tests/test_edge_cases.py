"""
Edge-case validation tests for arbitrary CSV datasets:
- Numeric-only datasets
- Categorical-heavy datasets
- Datasets with missing values and constant columns
- Small datasets (1 row, 3 rows)
- Chart generators and AI copilot under each extreme
"""

import io
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.data_processor import (
    load_csv,
    get_dataset_summary,
    get_numeric_stats,
    get_categorical_stats,
    get_missing_values_summary,
    apply_filters,
    get_data_quality_report,
    detect_smart_insights,
)
from src.visualizations import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_scatter,
    plot_line,
    plot_area,
    plot_box,
    plot_violin,
    plot_categorical_counts,
    plot_bar_advanced,
    plot_donut,
    plot_missing_values_bar,
)
from src.mock_ai_engine import analyze_question


def test_numeric_only_dataset():
    """Verify application modules when there are zero categorical columns."""
    df_numeric = pd.DataFrame({
        "feature_1": [1.2, 3.4, 5.6, 7.8],
        "feature_2": [10, 20, 30, 40],
        "feature_3": [100.5, 200.5, 300.5, 400.5],
    })

    summary = get_dataset_summary(df_numeric)
    assert len(summary["categorical_columns"]) == 0
    assert len(summary["numeric_columns"]) == 3

    insights = detect_smart_insights(df_numeric)
    assert isinstance(insights, list)

    quality = get_data_quality_report(df_numeric)
    assert quality["quality_score"] == 100.0

    # Test charts
    assert isinstance(plot_distribution(df_numeric, "feature_1"), go.Figure)
    assert isinstance(plot_correlation_heatmap(df_numeric), go.Figure)
    assert isinstance(plot_scatter(df_numeric, "feature_1", "feature_2"), go.Figure)
    assert isinstance(plot_line(df_numeric, "feature_1", "feature_2"), go.Figure)

    # Test AI
    res = analyze_question("What is the average feature_1?", df_numeric)
    assert "average" in res["answer"].lower()


def test_categorical_heavy_dataset():
    """Verify application modules when there are zero numeric columns."""
    df_cat = pd.DataFrame({
        "country": ["USA", "Canada", "UK", "Germany", "USA"],
        "tier": ["Gold", "Silver", "Platinum", "Gold", "Silver"],
        "status": ["Active", "Pending", "Active", "Inactive", "Active"],
    })

    summary = get_dataset_summary(df_cat)
    assert len(summary["numeric_columns"]) == 0
    assert len(summary["categorical_columns"]) == 3

    cat_stats = get_categorical_stats(df_cat)
    assert len(cat_stats) == 3

    # Test charts
    assert isinstance(plot_categorical_counts(df_cat, "country"), go.Figure)
    assert isinstance(plot_bar_advanced(df_cat, "tier"), go.Figure)
    assert isinstance(plot_donut(df_cat, "status"), go.Figure)

    # Test AI
    res = analyze_question("Tell me about country", df_cat)
    assert "country" in res["answer"].lower() or "usa" in res["answer"].lower()


def test_missing_heavy_and_constant_dataset():
    """Verify application modules when columns have heavy missingness or constant values."""
    df_messy = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "constant_col": ["Fixed", "Fixed", "Fixed", "Fixed"],
        "mostly_missing": [None, None, "Present", None],
        "metric": [10.0, None, 30.0, 40.0],
    })

    quality = get_data_quality_report(df_messy)
    assert "constant_col" in quality["constant_columns"]
    assert quality["missing_cells"] == 4
    assert len(quality["suggestions"]) > 0

    missing_df = get_missing_values_summary(df_messy)
    fig_miss = plot_missing_values_bar(missing_df)
    assert isinstance(fig_miss, go.Figure)

    # Filter with is null
    filtered = apply_filters(df_messy, [{"column": "mostly_missing", "operator": "is null", "value": None}])
    assert len(filtered) == 3


def test_tiny_single_row_dataset():
    """Verify dataset with exactly 1 row does not break analytics or visual builders."""
    df_single = pd.DataFrame({"age": [42], "category": ["Lone"]})
    summary = get_dataset_summary(df_single)
    assert summary["num_rows"] == 1

    fig_dist = plot_distribution(df_single, "age")
    assert isinstance(fig_dist, go.Figure)

    fig_cat = plot_categorical_counts(df_single, "category")
    assert isinstance(fig_cat, go.Figure)

    res = analyze_question("How many rows are in the dataset?", df_single)
    assert "1 rows" in res["answer"] or "1 row" in res["answer"]
