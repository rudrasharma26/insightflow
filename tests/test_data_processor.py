"""
Unit tests for src/data_processor.py
"""

import io
import pandas as pd
import pytest
from src.data_processor import (
    load_csv,
    get_dataset_summary,
    get_numeric_stats,
    get_categorical_stats,
    get_missing_values_summary,
)


@pytest.fixture
def sample_df():
    csv_data = """id,name,age,salary,department
1,Alice,25,50000,Engineering
2,Bob,30,60000,Marketing
3,Charlie,,75000,Engineering
4,David,40,,Sales
5,Eve,35,80000,
"""
    return pd.read_csv(io.StringIO(csv_data))


def test_load_csv_from_stringio():
    data = "a,b,c\n1,2,3\n4,5,6"
    df = load_csv(io.StringIO(data))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_get_dataset_summary(sample_df):
    summary = get_dataset_summary(sample_df)
    assert summary["num_rows"] == 5
    assert summary["num_cols"] == 5
    assert summary["total_cells"] == 25
    assert summary["total_missing_cells"] == 3
    assert summary["missing_percentage"] == 12.0
    assert "age" in summary["numeric_columns"]
    assert "salary" in summary["numeric_columns"]
    assert "name" in summary["categorical_columns"]


def test_get_dataset_summary_empty():
    empty_df = pd.DataFrame()
    summary = get_dataset_summary(empty_df)
    assert summary["num_rows"] == 0
    assert summary["num_cols"] == 0
    assert summary["numeric_columns"] == []


def test_get_numeric_stats(sample_df):
    stats = get_numeric_stats(sample_df)
    assert "age" in stats.index
    assert "salary" in stats.index
    assert "mean" in stats.columns
    assert "min" in stats.columns
    assert "max" in stats.columns
    assert "missing_count" in stats.columns
    assert stats.loc["age", "missing_count"] == 1
    assert stats.loc["salary", "missing_count"] == 1
    assert stats.loc["age", "min"] == 25.0
    assert stats.loc["age", "max"] == 40.0


def test_get_categorical_stats(sample_df):
    cat_stats = get_categorical_stats(sample_df)
    assert "department" in cat_stats.index
    assert "Top Value" in cat_stats.columns
    assert cat_stats.loc["department", "Top Value"] == "Engineering"
    assert cat_stats.loc["department", "Top Value Count"] == 2
    assert cat_stats.loc["department", "Missing Count"] == 1


def test_get_missing_values_summary(sample_df):
    missing_df = get_missing_values_summary(sample_df)
    assert isinstance(missing_df, pd.DataFrame)
    assert "Column" in missing_df.columns
    assert "Missing Count" in missing_df.columns
    assert "Missing %" in missing_df.columns
    assert "Status" in missing_df.columns
    
    # Check that columns with missing values are identified
    missing_cols = missing_df[missing_df["Missing Count"] > 0]["Column"].tolist()
    assert set(missing_cols) == {"age", "salary", "department"}


def test_small_datasets_1_and_3_rows():
    # 3-row dataset
    csv_3_rows = "id,name,score\n1,Alice,90\n2,Bob,80\n3,Charlie,85"
    df3 = load_csv(io.StringIO(csv_3_rows))
    summary3 = get_dataset_summary(df3)
    assert summary3["num_rows"] == 3
    assert summary3["num_cols"] == 3
    stats3 = get_numeric_stats(df3)
    assert stats3.loc["score", "mean"] == 85.0
    missing3 = get_missing_values_summary(df3)
    assert len(missing3) == 3

    # 1-row dataset
    csv_1_row = "id,name,score\n1,Solo,99"
    df1 = load_csv(io.StringIO(csv_1_row))
    summary1 = get_dataset_summary(df1)
    assert summary1["num_rows"] == 1
    stats1 = get_numeric_stats(df1)
    assert stats1.loc["score", "mean"] == 99.0

