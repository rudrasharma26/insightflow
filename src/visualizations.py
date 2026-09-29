"""
Interactive visualization generators using Plotly with modern light analytical canvas styling.
Provides chart creators for distributions, correlations, scatter plots,
lines, areas, box plots, violins, categorical breakdowns, donuts, and missing value indicators.
"""

from typing import List, Optional, Union
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Light, crisp analytical workspace layout for Plotly
LIGHT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFAFC",
    font=dict(
        color="#0F172A",
        family="-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif",
        size=12,
    ),
    margin=dict(l=45, r=30, t=50, b=45),
    hoverlabel=dict(
        bgcolor="#0F172A",
        font_size=12,
        font_family="-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif",
        font_color="#FFFFFF",
        bordercolor="rgba(0, 0, 0, 0)",
    ),
)

# Retain DARK_LAYOUT variable as alias for backwards-compatibility with existing tests/imports
DARK_LAYOUT = LIGHT_LAYOUT

# Sophisticated, controlled SaaS color palette
COLOR_SEQUENCE = [
    "#4F46E5",  # Indigo (Primary)
    "#06B6D4",  # Cyan
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#EC4899",  # Rose
    "#8B5CF6",  # Violet
    "#3B82F6",  # Blue
    "#14B8A6",  # Teal
]


def _apply_light_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent light gridlines, axis colors, and high-contrast typography."""
    fig.update_layout(
        **LIGHT_LAYOUT,
        title_font=dict(color="#0F172A", size=14, family="Inter, -apple-system, sans-serif"),
        legend=dict(
            font=dict(color="#0F172A", size=11, family="Inter, -apple-system, sans-serif"),
            title_font=dict(color="#0F172A", size=11, family="Inter, -apple-system, sans-serif"),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
        tickfont=dict(color="#334155", size=11, family="Inter, -apple-system, sans-serif"),
        title_font=dict(color="#0F172A", size=12, family="Inter, -apple-system, sans-serif"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
        tickfont=dict(color="#334155", size=11, family="Inter, -apple-system, sans-serif"),
        title_font=dict(color="#0F172A", size=12, family="Inter, -apple-system, sans-serif"),
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color="#334155", size=10, family="Inter, -apple-system, sans-serif"),
            title_font=dict(color="#0F172A", size=11, family="Inter, -apple-system, sans-serif"),
        )
    )
    return fig


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
        fig.update_layout(title="No data available for selected column", **LIGHT_LAYOUT)
        return fig

    if plot_type == "Histogram":
        fig = px.histogram(
            df,
            x=column,
            nbins=bins,
            marginal="box",
            color_discrete_sequence=["#4F46E5"],
            title=f"Distribution of {column}",
        )
    elif plot_type == "Box Plot":
        fig = px.box(
            df,
            y=column,
            points="all",
            color_discrete_sequence=["#8B5CF6"],
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
        fig = px.histogram(df, x=column, color_discrete_sequence=["#4F46E5"])

    return _apply_light_theme(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    method: str = "pearson",
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
            **LIGHT_LAYOUT,
        )
        return fig

    corr = numeric_df.corr(method=method).round(2)

    # Use a professional continuous color scale on light canvas
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"{method.capitalize()} Correlation Heatmap",
        zmin=-1,
        zmax=1,
    )
    fig = _apply_light_theme(fig)
    fig.update_xaxes(tickfont=dict(color="#0F172A", size=11, family="Inter, -apple-system, sans-serif"))
    fig.update_yaxes(tickfont=dict(color="#0F172A", size=11, family="Inter, -apple-system, sans-serif"))
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    add_trendline: bool = False,
    opacity: float = 0.8,
) -> go.Figure:
    """
    Generate a 2D Scatter Plot with optional color category, size dimension, and trendline.
    """
    trendline_mode = "ols" if add_trendline else None
    color_arg = color_col if color_col and color_col != "None" else None
    size_arg = size_col if size_col and size_col != "None" and pd.api.types.is_numeric_dtype(df[size_col]) else None

    try:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_arg,
            size=size_arg,
            opacity=opacity,
            trendline=trendline_mode,
            color_discrete_sequence=COLOR_SEQUENCE,
            title=f"{y_col} vs {x_col}",
        )
    except Exception:
        # Fallback to standard scatter plot if trendline or size fitting fails
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_arg,
            opacity=opacity,
            color_discrete_sequence=COLOR_SEQUENCE,
            title=f"{y_col} vs {x_col}",
        )
    return _apply_light_theme(fig)


def plot_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    show_markers: bool = True,
) -> go.Figure:
    """
    Generate a line chart with optional categorical grouping.
    """
    color_arg = group_col if group_col and group_col != "None" else None
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_arg,
        markers=show_markers,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"{y_col} across {x_col}",
    )
    return _apply_light_theme(fig)


def plot_area(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    stacked: bool = False,
) -> go.Figure:
    """
    Generate an area chart with optional grouping.
    """
    color_arg = group_col if group_col and group_col != "None" else None
    fig = px.area(
        df,
        x=x_col,
        y=y_col,
        color=color_arg,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"Area Plot: {y_col} by {x_col}",
    )
    if not stacked:
        fig.for_each_trace(lambda trace: trace.update(fill='tozeroy'))
    return _apply_light_theme(fig)


def plot_box(
    df: pd.DataFrame,
    num_col: str,
    cat_col: Optional[str] = None,
    points: str = "outliers",
) -> go.Figure:
    """
    Generate a Box Plot for a numerical column, optionally segmented by a categorical column.
    """
    color_arg = cat_col if cat_col and cat_col != "None" else None
    fig = px.box(
        df,
        x=color_arg,
        y=num_col,
        color=color_arg,
        points=points,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"Box Distribution of {num_col}" + (f" by {cat_col}" if cat_col and cat_col != "None" else ""),
    )
    return _apply_light_theme(fig)


def plot_violin(
    df: pd.DataFrame,
    num_col: str,
    cat_col: Optional[str] = None,
    show_box: bool = True,
) -> go.Figure:
    """
    Generate a Violin Plot for a numerical column, optionally segmented.
    """
    color_arg = cat_col if cat_col and cat_col != "None" else None
    fig = px.violin(
        df,
        x=color_arg,
        y=num_col,
        color=color_arg,
        box=show_box,
        points="all",
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"Violin Distribution: {num_col}" + (f" by {cat_col}" if cat_col and cat_col != "None" else ""),
    )
    return _apply_light_theme(fig)


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
        fig.update_layout(title="No data available for selected column", **LIGHT_LAYOUT)
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
            color_continuous_scale="Purples",
            title=f"Top {top_n} Categories in {column}",
        )
        fig.update_traces(textposition="outside")
    elif chart_type == "Donut":
        fig = px.pie(
            counts,
            names=column,
            values="Count",
            hole=0.45,
            color_discrete_sequence=COLOR_SEQUENCE,
            title=f"Category Distribution in {column}",
        )
    else:
        fig = px.bar(counts, x=column, y="Count", color_discrete_sequence=["#4F46E5"])

    return _apply_light_theme(fig)


def plot_bar_advanced(
    df: pd.DataFrame,
    cat_col: str,
    metric: str = "Count",
    val_col: Optional[str] = None,
    top_n: int = 10,
    orientation: str = "v",
    sort_order: str = "desc",
) -> go.Figure:
    """
    Generate a flexible, advanced bar chart (count, sum, mean) with sorting and orientation controls.
    """
    if cat_col not in df.columns or df[cat_col].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for category column", **LIGHT_LAYOUT)
        return fig

    if metric == "Count" or not val_col or val_col not in df.columns:
        grouped = df[cat_col].value_counts().reset_index()
        grouped.columns = [cat_col, "Value"]
        y_title = "Record Count"
    elif metric == "Sum":
        grouped = df.groupby(cat_col)[val_col].sum().reset_index()
        grouped.columns = [cat_col, "Value"]
        y_title = f"Sum of {val_col}"
    elif metric == "Mean":
        grouped = df.groupby(cat_col)[val_col].mean().round(2).reset_index()
        grouped.columns = [cat_col, "Value"]
        y_title = f"Average {val_col}"
    else:
        grouped = df[cat_col].value_counts().reset_index()
        grouped.columns = [cat_col, "Value"]
        y_title = "Count"

    # Sorting
    ascending = True if sort_order == "asc" else False
    grouped = grouped.sort_values(by="Value", ascending=ascending).head(top_n)

    if orientation == "h":
        fig = px.bar(
            grouped,
            x="Value",
            y=cat_col,
            text="Value",
            orientation="h",
            color="Value",
            color_continuous_scale="Viridis",
            title=f"{y_title} by {cat_col} (Top {top_n})",
        )
        fig.update_traces(textposition="outside")
    else:
        fig = px.bar(
            grouped,
            x=cat_col,
            y="Value",
            text="Value",
            orientation="v",
            color="Value",
            color_continuous_scale="Viridis",
            title=f"{y_title} by {cat_col} (Top {top_n})",
        )
        fig.update_traces(textposition="outside")

    return _apply_light_theme(fig)


def plot_donut(
    df: pd.DataFrame,
    cat_col: str,
    val_col: Optional[str] = None,
    top_n: int = 8,
    hole: float = 0.45,
) -> go.Figure:
    """
    Generate an elegant donut breakdown chart.
    """
    if cat_col not in df.columns or df[cat_col].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="No data available", **LIGHT_LAYOUT)
        return fig

    if val_col and val_col in df.columns and pd.api.types.is_numeric_dtype(df[val_col]):
        grouped = df.groupby(cat_col)[val_col].sum().reset_index()
        values_name = val_col
    else:
        grouped = df[cat_col].value_counts().reset_index()
        grouped.columns = [cat_col, "Count"]
        values_name = "Count"

    grouped = grouped.sort_values(by=values_name, ascending=False).head(top_n)

    fig = px.pie(
        grouped,
        names=cat_col,
        values=values_name,
        hole=hole,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"Distribution of {cat_col}",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_light_theme(fig)


def plot_missing_values_bar(missing_summary_df: pd.DataFrame) -> go.Figure:
    """
    Generate a bar chart showing missing value percentages across columns.
    """
    if missing_summary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No columns to display", **LIGHT_LAYOUT)
        return fig

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
    max_pct = (
        float(display_df["Missing %"].max())
        if not display_df["Missing %"].empty and not pd.isna(display_df["Missing %"].max())
        else 0.0
    )
    fig = _apply_light_theme(fig)
    fig.update_layout(yaxis_range=[0, max(100.0, max_pct + 10.0)])
    fig.update_traces(textposition="outside")
    return fig
