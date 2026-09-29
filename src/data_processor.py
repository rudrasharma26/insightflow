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


def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply a list of user-defined filter rules sequentially to a DataFrame.
    Each filter is a dict: {'column': str, 'operator': str, 'value': Any}
    """
    if df is None or df.empty or not filters:
        return df

    filtered_df = df.copy()

    for f in filters:
        col = f.get("column")
        op = f.get("operator", "equals")
        val = f.get("value")

        if col not in filtered_df.columns:
            continue

        series = filtered_df[col]

        try:
            if op == "is null":
                filtered_df = filtered_df[series.isna()]
            elif op == "is not null":
                filtered_df = filtered_df[series.notna()]
            elif op == "equals":
                if pd.api.types.is_numeric_dtype(series):
                    filtered_df = filtered_df[series == float(val)]
                else:
                    filtered_df = filtered_df[series.astype(str) == str(val)]
            elif op == "does not equal":
                if pd.api.types.is_numeric_dtype(series):
                    filtered_df = filtered_df[series != float(val)]
                else:
                    filtered_df = filtered_df[series.astype(str) != str(val)]
            elif op == "greater than (>)" or op == ">":
                filtered_df = filtered_df[series > float(val)]
            elif op == "less than (<)" or op == "<":
                filtered_df = filtered_df[series < float(val)]
            elif op == "greater or equal (>=)" or op == ">=":
                filtered_df = filtered_df[series >= float(val)]
            elif op == "less or equal (<=)" or op == "<=":
                filtered_df = filtered_df[series <= float(val)]
            elif op == "between":
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    low, high = float(val[0]), float(val[1])
                    filtered_df = filtered_df[(series >= low) & (series <= high)]
            elif op == "contains":
                filtered_df = filtered_df[series.astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "does not contain":
                filtered_df = filtered_df[~series.astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "is in":
                if isinstance(val, (list, tuple, set)):
                    str_vals = [str(v) for v in val]
                    filtered_df = filtered_df[series.astype(str).isin(str_vals)]
            elif op == "is not in":
                if isinstance(val, (list, tuple, set)):
                    str_vals = [str(v) for v in val]
                    filtered_df = filtered_df[~series.astype(str).isin(str_vals)]
        except Exception:
            # Skip invalid filter rule gracefully rather than crashing
            continue

    return filtered_df


def detect_column_types_extended(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify column archetypes: numeric, categorical, boolean, datetime, and identifier candidates.
    """
    if df is None or df.empty:
        return {"numeric": [], "categorical": [], "boolean": [], "datetime": [], "identifier": []}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # Identify object columns and try checking for dates
    remaining_objs = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in bool_cols]
    
    cat_cols = []
    id_cols = []

    for c in remaining_objs:
        clean_s = df[c].dropna()
        if len(clean_s) > 0 and clean_s.nunique() == len(clean_s) and len(clean_s) > 5:
            # High chance of being an ID/key column
            id_cols.append(c)
        cat_cols.append(c)

    return {
        "numeric": numeric_cols,
        "categorical": cat_cols,
        "boolean": bool_cols,
        "datetime": datetime_cols,
        "identifier": id_cols,
    }


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an in-depth data quality audit including completeness score,
    duplicate analysis, constant columns, and actionable suggestions.
    """
    if df is None or df.empty:
        return {
            "quality_score": 0.0,
            "completeness_pct": 0.0,
            "total_cells": 0,
            "missing_cells": 0,
            "duplicate_rows": 0,
            "duplicate_pct": 0.0,
            "constant_columns": [],
            "audit_df": pd.DataFrame(),
            "suggestions": ["Upload a CSV file to evaluate data quality."],
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0
    completeness = round(100.0 - missing_pct, 1)

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

    constant_cols = []
    for c in df.columns:
        clean_s = df[c].dropna()
        if len(clean_s) > 0 and clean_s.nunique() <= 1:
            constant_cols.append(c)
        elif len(clean_s) > 1 and pd.api.types.is_numeric_dtype(clean_s):
            try:
                if np.isclose(float(clean_s.std(ddof=0)), 0.0, atol=1e-12):
                    constant_cols.append(c)
            except Exception:
                pass

    # Calculate overall health score (0 to 100)
    # Starts at 100, penalized by missingness and duplicate rate
    score = 100.0 - (missing_pct * 0.7) - (duplicate_pct * 0.5) - (len(constant_cols) * 2.0)
    quality_score = max(0.0, min(100.0, round(score, 1)))

    # Build detailed audit table
    audit_rows = []
    for col in df.columns:
        m_cnt = int(df[col].isna().sum())
        m_pct = round((m_cnt / total_rows) * 100, 2) if total_rows > 0 else 0.0
        u_cnt = int(df[col].nunique())
        u_pct = round((u_cnt / total_rows) * 100, 1) if total_rows > 0 else 0.0

        if total_rows == 0 or u_cnt == 0:
            status = "Empty (100% Missing)"
        elif col in constant_cols:
            status = "Constant (Zero Variance)"
        elif m_cnt == 0:
            status = "Excellent"
        elif m_pct < 5.0:
            status = "Good (<5% missing)"
        elif m_pct < 20.0:
            status = "Moderate (5-20% missing)"
        else:
            status = "High Missing (>20%)"

        audit_rows.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Populated": total_rows - m_cnt,
            "Missing": m_cnt,
            "Missing %": m_pct,
            "Unique Count": u_cnt,
            "Distinct %": u_pct,
            "Quality Status": status,
        })

    audit_df = pd.DataFrame(audit_rows).sort_values(by="Missing", ascending=False).reset_index(drop=True)

    # Actionable suggestions
    suggestions = []
    if missing_pct == 0.0 and duplicate_rows == 0:
        suggestions.append("🎉 Clean dataset! No missing cells or duplicate records detected.")
    if missing_pct > 0.0:
        high_miss = [c for c in df.columns if df[c].isna().sum() / total_rows > 0.2]
        if high_miss:
            suggestions.append(f"Consider evaluating columns with >20% missingness: {', '.join(high_miss[:3])}.")
        else:
            suggestions.append(f"Impute or handle {total_missing:,} missing values before predictive modeling.")
    if duplicate_rows > 0:
        suggestions.append(f"Found {duplicate_rows:,} duplicate row(s) ({duplicate_pct}%). Consider deduplicating.")
    if constant_cols:
        suggestions.append(f"Columns with zero variance (single value): {', '.join(constant_cols)}. These provide no analytical value.")
    if not suggestions:
        suggestions.append("Dataset is in ready-to-analyze condition.")

    return {
        "quality_score": quality_score,
        "completeness_pct": completeness,
        "total_cells": total_cells,
        "missing_cells": total_missing,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "constant_columns": constant_cols,
        "audit_df": audit_df,
        "suggestions": suggestions,
    }


def detect_smart_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Extract smart, automatic analytical insights for the dataset overview hero cards.
    """
    if df is None or df.empty:
        return []

    insights = []
    total_rows = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # 1. Completeness insight
    total_missing = int(df.isna().sum().sum())
    total_cells = total_rows * len(df.columns)
    if total_missing == 0:
        insights.append({
            "title": "High Data Completeness",
            "description": f"All {total_cells:,} data cells across {len(df.columns)} columns are 100% populated with zero missing values.",
            "badge": "100% Full",
            "category": "Quality",
        })
    else:
        miss_col = df.isna().sum().idxmax()
        miss_val = int(df.isna().sum().max())
        miss_pct = round((miss_val / total_rows) * 100, 1)
        insights.append({
            "title": f"Missing Value Concentration",
            "description": f"Column '{miss_col}' has the highest missingness ({miss_val} cells, {miss_pct}% of total records).",
            "badge": f"{miss_pct}% Missing",
            "category": "Quality",
        })

    # 2. Correlation insight
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                v = corr_matrix.loc[c1, c2]
                if not np.isnan(v):
                    pairs.append((c1, c2, float(v)))
        if pairs:
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_pair = pairs[0]
            direction = "positive" if top_pair[2] > 0 else "inverse"
            strength = "strong" if abs(top_pair[2]) > 0.6 else "moderate"
            insights.append({
                "title": f"Key Correlation Detected",
                "description": f"'{top_pair[0]}' and '{top_pair[1]}' exhibit a {strength} {direction} correlation of {top_pair[2]:.2f}.",
                "badge": f"r = {top_pair[2]:.2f}",
                "category": "Correlation",
            })

    # 3. Categorical distribution insight
    if cat_cols:
        # Pick category with reasonable cardinality
        valid_cats = [c for c in cat_cols if 1 < df[c].nunique() <= 30]
        chosen_cat = valid_cats[0] if valid_cats else cat_cols[0]
        vc = df[chosen_cat].value_counts()
        if not vc.empty:
            top_val = str(vc.index[0])
            top_pct = round((vc.iloc[0] / total_rows) * 100, 1)
            insights.append({
                "title": f"Dominant Segment: {chosen_cat}",
                "description": f"'{top_val}' is the leading category, representing {top_pct}% ({vc.iloc[0]:,} records) of the dataset.",
                "badge": f"{top_pct}% Share",
                "category": "Distribution",
            })

    # 4. Outlier / Numeric spread insight
    if numeric_cols:
        target_num = numeric_cols[0]
        series = df[target_num].dropna()
        if len(series) > 0 and series.mean() != 0:
            std_ratio = series.std() / (abs(series.mean()) + 1e-9)
            if std_ratio > 0.8:
                insights.append({
                    "title": f"High Variance in {target_num}",
                    "description": f"The metric ranges from {series.min():,.1f} to {series.max():,.1f} (std dev: {series.std():,.1f}), indicating substantial spread.",
                    "badge": "High Spread",
                    "category": "Variance",
                })
            else:
                insights.append({
                    "title": f"Uniformity in {target_num}",
                    "description": f"Mean is {series.mean():,.1f} with tight standard deviation of {series.std():,.1f}.",
                    "badge": "Consistent",
                    "category": "Metric",
                })

    return insights

