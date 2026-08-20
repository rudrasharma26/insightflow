"""
Data processing utilities for the CSV Data Explorer.
Provides clean functions for loading datasets, extracting overview metadata,
calculating descriptive statistics, and identifying missing values.
"""

from typing import Any, Dict, List, Union
import io
import pandas as pd
import numpy as np


def load_csv(file_or_path: Union[str, io.BytesIO, io.StringIO, Any]) -> pd.DataFrame:
    """
    Load a CSV file from a path or file-like object into a pandas DataFrame.
    """
    try:
        if isinstance(file_or_path, str):
            df = pd.read_csv(file_or_path)
        else:
            df = pd.read_csv(file_or_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}") from e


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute high-level overview statistics and metadata for the dataset.
    """
    if df is None or df.empty:
        return {
            "num_rows": 0,
            "num_cols": 0,
            "total_cells": 0,
            "total_missing_cells": 0,
            "missing_percentage": 0.0,
            "memory_usage_kb": 0.0,
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "column_types": {},
            "duplicate_rows": 0,
        }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)
    duplicates = int(df.duplicated().sum())

    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "total_cells": total_cells,
        "total_missing_cells": total_missing,
        "missing_percentage": missing_pct,
        "memory_usage_kb": memory_kb,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "column_types": column_types,
        "duplicate_rows": duplicates,
    }


def get_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate detailed descriptive statistics for numerical columns in the dataset.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    stats = numeric_df.describe().transpose()
    
    # Add median, missing count, and missing percentage
    medians = []
    missing_counts = []
    missing_pcts = []
    
    for col in stats.index:
        medians.append(numeric_df[col].median())
        m_count = numeric_df[col].isna().sum()
        missing_counts.append(int(m_count))
        missing_pcts.append(round((m_count / len(numeric_df)) * 100, 2) if len(numeric_df) > 0 else 0.0)

    stats["median"] = medians
    stats["missing_count"] = missing_counts
    stats["missing_%"] = missing_pcts

    # Reorder and round columns
    cols_order = ["count", "mean", "std", "min", "25%", "50%", "median", "75%", "max", "missing_count", "missing_%"]
    existing_cols = [c for c in cols_order if c in stats.columns]
    stats = stats[existing_cols]
    
    return stats.round(2)


def get_categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for categorical/text columns in the dataset.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    if cat_df.empty:
        return pd.DataFrame()

    records = []
    for col in cat_df.columns:
        series = cat_df[col]
        total = len(series)
        missing = int(series.isna().sum())
        non_null = series.dropna()
        unique_cnt = int(series.nunique())
        
        if not non_null.empty:
            top_val = str(non_null.mode().iloc[0]) if len(non_null.mode()) > 0 else "N/A"
            top_freq = int(series.value_counts().iloc[0]) if len(series.value_counts()) > 0 else 0
            top_pct = round((top_freq / total) * 100, 2) if total > 0 else 0.0
        else:
            top_val = "N/A"
            top_freq = 0
            top_pct = 0.0

        records.append({
            "Column": col,
            "Data Type": str(series.dtype),
            "Total Count": total,
            "Unique Values": unique_cnt,
            "Top Value": top_val,
            "Top Value Count": top_freq,
            "Top Value %": top_pct,
            "Missing Count": missing,
            "Missing %": round((missing / total) * 100, 2) if total > 0 else 0.0,
        })

    return pd.DataFrame(records).set_index("Column")


def get_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a clean summary table of missing values for all columns in the DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Column", "Data Type", "Total Values", "Missing Count", "Missing %", "Status"])

    total_rows = len(df)
    missing_series = df.isna().sum()
    
    summary_data = []
    for col, missing_count in missing_series.items():
        pct = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        status = "Clean (0% missing)" if missing_count == 0 else f"Has Missing ({pct}%)"
        summary_data.append({
            "Column": str(col),
            "Data Type": str(df[col].dtype),
            "Total Values": total_rows,
            "Missing Count": int(missing_count),
            "Missing %": pct,
            "Status": status,
        })

    summary_df = pd.DataFrame(summary_data)
    # Sort descending by missing count so columns needing attention are at the top
    summary_df = summary_df.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)
    return summary_df
