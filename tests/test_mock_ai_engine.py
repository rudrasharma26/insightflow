"""
Unit tests for src/mock_ai_engine.py
"""

import io
import pandas as pd
import pytest
from src.mock_ai_engine import analyze_question


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


def test_analyze_question_structure(sample_df):
    result = analyze_question("What is the dataset overview?", sample_df)
    assert isinstance(result, dict)
    assert "query" in result
    assert "answer" in result
    assert "insights" in result
    assert "suggested_questions" in result
    assert isinstance(result["insights"], list)
    assert isinstance(result["suggested_questions"], list)
    assert len(result["insights"]) > 0
    assert len(result["suggested_questions"]) > 0


def test_analyze_empty_dataframe():
    empty_df = pd.DataFrame()
    result = analyze_question("How many rows?", empty_df)
    assert "No dataset is currently loaded" in result["answer"]


def test_analyze_row_count_query(sample_df):
    result = analyze_question("How many rows are in the dataset?", sample_df)
    assert "5 rows" in result["answer"]
    assert "5 columns" in result["answer"]


def test_analyze_average_query(sample_df):
    result = analyze_question("What is the average salary?", sample_df)
    # Expected mean of [85000, 72000, 110000, 65000, 92000] is 84800.00
    assert "84,800.00" in result["answer"]
    assert "salary" in result["answer"].lower()


def test_analyze_max_query(sample_df):
    result = analyze_question("What is the highest salary?", sample_df)
    assert "110,000.00" in result["answer"]


def test_analyze_missing_query(sample_df):
    result = analyze_question("Which column has missing values?", sample_df)
    assert "experience" in result["answer"]
    assert "missing" in result["answer"].lower()


def test_analyze_categorical_query(sample_df):
    result = analyze_question("Tell me about department", sample_df)
    assert "Engineering" in result["answer"]


def test_analyze_small_datasets():
    df_3_rows = pd.DataFrame({"id": [1, 2, 3], "age": [20, 30, 40], "city": ["NYC", "LA", "NYC"]})
    res_rows = analyze_question("How many rows?", df_3_rows)
    assert "3 rows" in res_rows["answer"]
    res_avg = analyze_question("What is the average age?", df_3_rows)
    assert "30.00" in res_avg["answer"]
    res_cat = analyze_question("Tell me about city", df_3_rows)
    assert "NYC" in res_cat["answer"]

    df_1_row = pd.DataFrame({"salary": [50000]})
    res_1 = analyze_question("What is the average salary?", df_1_row)
    assert "50,000.00" in res_1["answer"]

