"""
Interactive visualization generators using Plotly with dark mode styling.
Provides chart creators for distributions, correlations, scatter plots,
categorical breakdowns, and missing value indicators.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Common dark layout styling
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(26, 30, 41, 0.7)",
    plot_bgcolor="rgba(14, 17, 23, 0.9)",
    font=dict(color="#F3F4F6", family="sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    plot_type: str = "Histogram",
    bins: int = 20,
) -> go.Figure:
    """
    Generate a distribution chart (Histogram, Box Plot, or Violin Plot) for a numerical column.
    """
    if column not in df.columns or df[column].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for selected column", **DARK_LAYOUT)
        return fig

    clean_series = df[column].dropna()

    if plot_type == "Histogram":
        fig = px.histogram(
            df,
            x=column,
            nbins=bins,
            marginal="box",
            color_discrete_sequence=["#6366F1"],
            title=f"Distribution of {column}",
        )
    elif plot_type == "Box Plot":
        fig = px.box(
            df,
            y=column,
            points="all",
            color_discrete_sequence=["#A855F7"],
            title=f"Box Plot of {column}",
        )
    elif plot_type == "Violin Plot":
        fig = px.violin(
            df,
            y=column,
            box=True,
            points="all",
            color_discrete_sequence=["#EC4899"],
            title=f"Violin Plot of {column}",
        )
    else:
        fig = px.histogram(df, x=column, color_discrete_sequence=["#6366F1"])

    fig.update_layout(**DARK_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
) -> go.Figure:
    """
    Generate an interactive correlation matrix heatmap for numeric columns.
    """
    if numeric_cols is None or len(numeric_cols) == 0:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[numeric_cols].select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Need at least 2 numerical columns to calculate correlation",
            **DARK_LAYOUT,
        )
        return fig

    corr = numeric_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Pearson Correlation Heatmap",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    add_trendline: bool = False,
) -> go.Figure:
    """
    Generate a 2D Scatter Plot with optional color category and trendline.
    """
    trendline_mode = "ols" if add_trendline else None
    
    try:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col if color_col and color_col != "None" else None,
            trendline=trendline_mode,
            color_discrete_sequence=px.colors.qualitative.Prism,
            title=f"{y_col} vs {x_col}",
        )
    except Exception:
        # Fallback to standard scatter plot if trendline fitting fails
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col if color_col and color_col != "None" else None,
            color_discrete_sequence=px.colors.qualitative.Prism,
            title=f"{y_col} vs {x_col}",
        )
    fig.update_layout(**DARK_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
    return fig


def plot_categorical_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
    chart_type: str = "Bar",
) -> go.Figure:
    """
    Generate a bar or donut chart representing category counts.
    """
    if column not in df.columns or df[column].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for selected column", **DARK_LAYOUT)
        return fig

    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = [column, "Count"]

    if chart_type == "Bar":
        fig = px.bar(
            counts,
            x=column,
            y="Count",
            text="Count",
            color="Count",
            color_continuous_scale="Viridis",
            title=f"Top {top_n} Categories in {column}",
        )
        fig.update_traces(textposition="outside")
    elif chart_type == "Donut":
        fig = px.pie(
            counts,
            names=column,
            values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title=f"Category Distribution in {column}",
        )
    else:
        fig = px.bar(counts, x=column, y="Count", color_discrete_sequence=["#10B981"])

    fig.update_layout(**DARK_LAYOUT)
    return fig


def plot_missing_values_bar(missing_summary_df: pd.DataFrame) -> go.Figure:
    """
    Generate a bar chart showing missing value percentages across columns.
    """
    if missing_summary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No columns to display", **DARK_LAYOUT)
        return fig

    # Filter to only columns with missing values if any, else show all
    has_missing = missing_summary_df[missing_summary_df["Missing Count"] > 0]
    display_df = has_missing if not has_missing.empty else missing_summary_df

    fig = px.bar(
        display_df,
        x="Column",
        y="Missing %",
        text="Missing %",
        color="Missing %",
        color_continuous_scale="Reds",
        title="Missing Values by Column (%)",
        labels={"Missing %": "Missing Percentage (%)"},
    )
    max_pct = float(display_df["Missing %"].max()) if not display_df["Missing %"].empty and not pd.isna(display_df["Missing %"].max()) else 0.0
    fig.update_layout(**DARK_LAYOUT, yaxis_range=[0, max(100.0, max_pct + 10.0)])
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
    return fig
