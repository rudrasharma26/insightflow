"""
Heuristic-based Mock AI Dataset Assistant.
Analyzes natural language questions against a loaded pandas DataFrame locally
without requiring an external API key or internet connection.
Returns structured answers, data insights, and suggested follow-up questions.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import pandas as pd
import numpy as np


def _find_matching_column(query: str, columns: List[str]) -> Optional[str]:
    """Find a column in DataFrame that matches a token in the user query."""
    clean_query = query.lower()
    # Check exact name match first
    for col in columns:
        col_clean = col.lower().strip()
        if col_clean in clean_query:
            return col
    # Check token overlap
    for col in columns:
        tokens = re.findall(r"\w+", col.lower())
        if any(tok in clean_query for tok in tokens if len(tok) > 2):
            return col
    return None


def analyze_question(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a user's natural language question and return a structured response.

    Returns:
        dict with keys:
            - query: Original query string
            - answer: Formatted markdown answer string
            - insights: List of bullet points with data-driven insights
            - suggested_questions: List of relevant follow-up questions
    """
    if df is None or df.empty:
        return {
            "query": query,
            "answer": "No dataset is currently loaded. Please upload a CSV file or load the sample dataset to explore data.",
            "insights": ["Upload a CSV file to begin analysis."],
            "suggested_questions": [
                "What is the dataset overview?",
                "How many rows are in the dataset?",
                "What columns are available?",
            ],
        }

    q = query.strip().lower()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    all_cols = df.columns.tolist()

    total_rows = len(df)
    total_cols = len(all_cols)
    total_missing = int(df.isna().sum().sum())

    matched_num_col = _find_matching_column(q, numeric_cols)
    matched_cat_col = _find_matching_column(q, cat_cols)
    matched_col = matched_num_col or matched_cat_col or _find_matching_column(q, all_cols)

    # 1. Row count / Dimensions / Size
    if any(k in q for k in ["how many rows", "row count", "number of rows", "dataset size", "dimensions", "shape", "how big"]):
        answer = f"The dataset contains **{total_rows:,} rows** and **{total_cols} columns** across **{total_rows * total_cols:,} total data cells**."
        insights = [
            f"Numerical columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'None'}",
            f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'None'}",
            f"Overall missing values: {total_missing:,} cells ({round((total_missing / (total_rows * total_cols)) * 100, 2)}% of dataset)",
        ]
        suggested = [
            "What is the average of the numerical columns?",
            "Which column has the highest missing values?",
            "Give me an executive summary of this dataset.",
        ]
        return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}

    # 2. Column names / list
    if any(k in q for k in ["what columns", "list columns", "column names", "all columns", "which columns"]):
        cols_formatted = ", ".join([f"`{c}`" for c in all_cols])
        answer = f"This dataset includes **{total_cols} columns**:\n\n{cols_formatted}"
        insights = [
            f"**{len(numeric_cols)}** numeric features available for statistical analysis and correlation.",
            f"**{len(cat_cols)}** categorical features available for segment grouping.",
        ]
        suggested = [
            f"What is the average {numeric_cols[0]}?" if numeric_cols else "How many rows are in the dataset?",
            "Which column has missing values?",
            "What is the correlation between numeric columns?",
        ]
        return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}

    # 3. Missing values analysis
    if any(k in q for k in ["missing", "null", "nan", "na ", "empty", "incomplete", "unfilled"]):
        missing_counts = df.isna().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        
        if cols_with_missing.empty:
            answer = "🎉 **Great news!** This dataset has **0 missing values**. All rows and columns are completely populated."
            insights = [
                "100% data completeness across all fields.",
                "No imputation or null-value handling is required for modeling or charting.",
            ]
        else:
            top_missing_col = cols_with_missing.idxmax()
            top_missing_val = cols_with_missing.max()
            top_missing_pct = round((top_missing_val / total_rows) * 100, 2)
            
            missing_list = "\n".join([
                f"- **`{col}`**: {cnt} missing values ({round((cnt / total_rows) * 100, 2)}%)"
                for col, cnt in cols_with_missing.items()
            ])
            
            answer = (
                f"There are **{total_missing} missing values** across **{len(cols_with_missing)} columns**.\n\n"
                f"**Columns with missing data:**\n{missing_list}"
            )
            insights = [
                f"Column with the most missing data is **`{top_missing_col}`** with {top_missing_val} missing entries ({top_missing_pct}%).",
                f"{len(all_cols) - len(cols_with_missing)} out of {total_cols} columns are completely full.",
            ]
        suggested = [
            "Show summary statistics for numeric columns",
            f"What is the distribution of {numeric_cols[0]}?" if numeric_cols else "List all columns",
            "What are the most frequent categories?",
        ]
        return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}

    # 4. Average / Mean query
    if any(k in q for k in ["average", "mean", "avg"]):
        if matched_num_col:
            series = df[matched_num_col].dropna()
            avg_val = round(series.mean(), 2)
            min_val = round(series.min(), 2)
            max_val = round(series.max(), 2)
            med_val = round(series.median(), 2)
            std_val = round(series.std(), 2)
            
            answer = (
                f"The average **`{matched_num_col}`** is **{avg_val:,.2f}** "
                f"(Median: {med_val:,.2f}, Range: {min_val:,.2f} to {max_val:,.2f})."
            )
            insights = [
                f"Standard deviation for `{matched_num_col}` is **{std_val:,.2f}**, indicating the spread of the data.",
                f"Calculated from {len(series):,} valid entries ({df[matched_num_col].isna().sum()} missing).",
            ]
            suggested = [
                f"What is the maximum {matched_num_col}?",
                f"What is the minimum {matched_num_col}?",
                "What is the correlation between numeric columns?",
            ]
            return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}
        elif numeric_cols:
            means = {col: round(df[col].mean(), 2) for col in numeric_cols}
            summary_lines = "\n".join([f"- **`{col}`**: {val:,.2f}" for col, val in means.items()])
            answer = f"Here are the average values for all numeric columns:\n\n{summary_lines}"
            insights = [
                f"Calculated across {len(numeric_cols)} numeric columns.",
                f"Median values may differ if distributions have skewness.",
            ]
            suggested = [
                f"What is the maximum {numeric_cols[0]}?",
                f"What is the distribution of {numeric_cols[0]}?",
                "Which column has missing values?",
            ]
            return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}

    # 5. Maximum / Highest queries
    if any(k in q for k in ["maximum", "max", "highest", "top", "greatest", "peak"]):
        if matched_num_col:
            series = df[matched_num_col].dropna()
            max_val = series.max()
            max_idx = series.idxmax()
            
            # Find identifier if name or ID exists
            id_col = next((c for c in ["Name", "name", "ID", "EmployeeID", "id"] if c in df.columns), None)
            id_str = f" for `{df.loc[max_idx, id_col]}`" if id_col and id_col in df.columns else ""
            
            answer = f"The maximum **`{matched_num_col}`** is **{max_val:,.2f}**{id_str} (at row index {max_idx})."
            insights = [
                f"Minimum `{matched_num_col}` in the dataset is {series.min():,.2f}.",
                f"Average `{matched_num_col}` is {series.mean():,.2f}.",
            ]
            suggested = [
                f"What is the minimum {matched_num_col}?",
                f"What is the average {matched_num_col}?",
                "Show summary statistics",
            ]
            return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}

    # 6. Minimum / Lowest queries
    if any(k in q for k in ["minimum", "min", "lowest", "smallest", "bottom"]):
        if matched_num_col:
            series = df[matched_num_col].dropna()
            min_val = series.min()
            min_idx = series.idxmin()
            
            id_col = next((c for c in ["Name", "name", "ID", "EmployeeID", "id"] if c in df.columns), None)
            id_str = f" for `{df.loc[min_idx, id_col]}`" if id_col and id_col in df.columns else ""
            
            answer = f"The minimum **`{matched_num_col}`** is **{min_val:,.2f}**{id_str} (at row index {min_idx})."
            insights = [
                f"Maximum `{matched_num_col}` in the dataset is {series.max():,.2f}.",
                f"Average `{matched_num_col}` is {series.mean():,.2f}.",
            ]
            suggested = [
                f"What is the maximum {matched_num_col}?",
                f"What is the average {matched_num_col}?",
                "Show missing values summary",
            ]
            return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}

    # 7. Correlation queries
    if any(k in q for k in ["correlation", "correlated", "relationship", "relate"]):
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Find highest positive correlation off-diagonal
            corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    c1, c2 = numeric_cols[i], numeric_cols[j]
                    val = corr_matrix.loc[c1, c2]
                    if not np.isnan(val):
                        corr_pairs.append((c1, c2, val))
            
            if corr_pairs:
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                top_c1, top_c2, top_val = corr_pairs[0]
                
                pair_lines = "\n".join([
                    f"- **`{c1}` & `{c2}`**: {round(v, 2)} ({'Strong Positive' if v > 0.7 else 'Moderate Positive' if v > 0.3 else 'Strong Negative' if v < -0.7 else 'Weak / Neutral'})"
                    for c1, c2, v in corr_pairs[:3]
                ])
                
                answer = f"Top correlated pairs among numerical features:\n\n{pair_lines}"
                insights = [
                    f"Strongest relationship is between **`{top_c1}`** and **`{top_c2}`** with a correlation coefficient of **{round(top_val, 2)}**.",
                    "Values close to 1.0 indicate strong positive co-movement, while values near 0 imply little linear correlation.",
                ]
                suggested = [
                    f"What is the average {top_c1}?",
                    f"What is the distribution of {top_c2}?",
                    "Show dataset summary",
                ]
                return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}

    # 8. Categorical distribution / breakdown queries
    if matched_cat_col:
        vc = df[matched_cat_col].value_counts()
        if not vc.empty:
            top_cat = vc.index[0]
            top_cnt = vc.iloc[0]
            top_pct = round((top_cnt / total_rows) * 100, 1)
            breakdown = "\n".join([f"- **`{cat}`**: {cnt} ({round((cnt / total_rows) * 100, 1)}%)" for cat, cnt in vc.head(5).items()])
            
            answer = (
                f"**`{matched_cat_col}`** has **{df[matched_cat_col].nunique()} unique categories**. "
                f"The most common category is **`{top_cat}`** with **{top_cnt}** occurrences ({top_pct}% of total).\n\n"
                f"**Top Categories Breakdown:**\n{breakdown}"
            )
            insights = [
                f"`{matched_cat_col}` contains {df[matched_cat_col].isna().sum()} missing values.",
                f"Top 5 categories represent {round((vc.head(5).sum() / total_rows) * 100, 1)}% of all records.",
            ]
            suggested = [
                f"Which column has missing values?",
                "What is the dataset summary?",
                f"What is the average {numeric_cols[0]}?" if numeric_cols else "List all columns",
            ]
            return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}

    # 9. General Summary / Default fallback
    summary_lines = [
        f"- **Records**: {total_rows:,} rows and {total_cols} columns",
        f"- **Data Completeness**: {round(((total_rows * total_cols - total_missing) / (total_rows * total_cols)) * 100, 1)}%",
    ]
    if numeric_cols:
        summary_lines.append(f"- **Key Numeric Metrics**: {', '.join(numeric_cols[:4])}")
    if cat_cols:
        summary_lines.append(f"- **Key Categories**: {', '.join(cat_cols[:4])}")

    answer = (
        f"### Dataset Snapshot\n\n"
        + "\n".join(summary_lines)
        + f"\n\n*Ask a specific question like **'What is the average {numeric_cols[0]}?'** or **'Which column has missing values?'** for tailored insights.*"
        if numeric_cols
        else f"### Dataset Snapshot\n\n" + "\n".join(summary_lines)
    )
    
    insights = [
        f"The dataset is loaded with {total_rows} entries and {total_cols} features.",
        f"{len(numeric_cols)} numerical and {len(cat_cols)} categorical columns detected.",
    ]
    suggested = [
        f"What is the average {numeric_cols[0]}?" if numeric_cols else "How many rows are in the dataset?",
        "Which column has the highest missing values?",
        "What are the top correlations?",
    ]
    return {"query": query, "answer": answer, "insights": insights, "suggested_questions": suggested}
