"""
Unit tests for new advanced features: filter engine, data quality diagnostics,
smart auto-insights, and Chart Studio visual builders.
"""

import io
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.data_processor import (
    apply_filters,
    get_data_quality_report,
    detect_smart_insights,
    detect_column_types_extended,
)
from src.visualizations import (
    plot_line,
    plot_area,
    plot_box,
    plot_violin,
    plot_bar_advanced,
    plot_donut,
)


@pytest.fixture
def complex_df():
    data = """id,name,department,salary,experience,bonus,is_active
101,Alice,Engineering,85000,4.5,5000,True
102,Bob,Marketing,72000,6.0,3000,True
103,Charlie,Engineering,110000,8.0,,False
104,Diana,Sales,65000,3.0,2000,True
105,Evan,Engineering,92000,,4000,True
106,Fiona,Sales,68000,2.5,1500,False
"""
    return pd.read_csv(io.StringIO(data))


def test_apply_filters_numeric(complex_df):
    # Greater than
    f1 = [{"column": "salary", "operator": "greater than (>)", "value": 80000}]
    res1 = apply_filters(complex_df, f1)
    assert len(res1) == 3
    assert set(res1["name"]) == {"Alice", "Charlie", "Evan"}

    # Between
    f2 = [{"column": "salary", "operator": "between", "value": [70000, 95000]}]
    res2 = apply_filters(complex_df, f2)
    assert len(res2) == 3
    assert set(res2["name"]) == {"Alice", "Bob", "Evan"}

    # Is null
    f3 = [{"column": "experience", "operator": "is null", "value": None}]
    res3 = apply_filters(complex_df, f3)
    assert len(res3) == 1
    assert res3.iloc[0]["name"] == "Evan"


def test_apply_filters_categorical(complex_df):
    # Equals
    f1 = [{"column": "department", "operator": "equals", "value": "Engineering"}]
    res1 = apply_filters(complex_df, f1)
    assert len(res1) == 3

    # Contains
    f2 = [{"column": "name", "operator": "contains", "value": "li"}]
    res2 = apply_filters(complex_df, f2)
    assert len(res2) == 2  # Alice, Charlie

    # Is in
    f3 = [{"column": "department", "operator": "is in", "value": ["Marketing", "Sales"]}]
    res3 = apply_filters(complex_df, f3)
    assert len(res3) == 3


def test_apply_filters_combined(complex_df):
    filters = [
        {"column": "department", "operator": "equals", "value": "Engineering"},
        {"column": "salary", "operator": "greater than (>)", "value": 90000},
    ]
    res = apply_filters(complex_df, filters)
    assert len(res) == 2
    assert set(res["name"]) == {"Charlie", "Evan"}


def test_get_data_quality_report(complex_df):
    rep = get_data_quality_report(complex_df)
    assert "quality_score" in rep
    assert "completeness_pct" in rep
    assert "duplicate_rows" in rep
    assert rep["duplicate_rows"] == 0
    assert rep["missing_cells"] == 2  # bonus in row 3, experience in row 5
    assert isinstance(rep["audit_df"], pd.DataFrame)
    assert len(rep["audit_df"]) == len(complex_df.columns)
    assert len(rep["suggestions"]) > 0


def test_detect_smart_insights(complex_df):
    insights = detect_smart_insights(complex_df)
    assert isinstance(insights, list)
    assert len(insights) >= 2
    for ins in insights:
        assert "title" in ins
        assert "description" in ins
        assert "badge" in ins


def test_detect_column_types_extended(complex_df):
    types = detect_column_types_extended(complex_df)
    assert "salary" in types["numeric"]
    assert "department" in types["categorical"]
    assert "is_active" in types["boolean"]


def test_new_chart_studio_visualizations(complex_df):
    # Line
    fig_line = plot_line(complex_df, "experience", "salary", group_col="department")
    assert isinstance(fig_line, go.Figure)

    # Area
    fig_area = plot_area(complex_df, "experience", "salary")
    assert isinstance(fig_area, go.Figure)

    # Box
    fig_box = plot_box(complex_df, "salary", cat_col="department")
    assert isinstance(fig_box, go.Figure)

    # Violin
    fig_violin = plot_violin(complex_df, "salary", cat_col="department")
    assert isinstance(fig_violin, go.Figure)

    # Advanced Bar
    fig_bar = plot_bar_advanced(complex_df, "department", metric="Mean", val_col="salary")
    assert isinstance(fig_bar, go.Figure)

    # Donut
    fig_donut = plot_donut(complex_df, "department")
    assert isinstance(fig_donut, go.Figure)
