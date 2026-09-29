"""
InsightFlow | AI-Powered CSV Data Intelligence Workspace
A modern, production-quality data exploration platform with dark shell chrome,
light analytical surfaces, interactive chart studio, multi-clause filter engine,
data quality center, and an offline heuristic AI copilot.
"""

from typing import Any, Dict, List, Optional
import inspect
import io
import os
import numpy as np
import pandas as pd
import streamlit as st

from src.data_processor import (
    load_csv,
    get_dataset_summary,
    get_numeric_stats,
    get_categorical_stats,
    get_missing_values_summary,
    apply_filters,
    get_data_quality_report,
    detect_smart_insights,
    detect_column_types_extended,
)
from src.visualizations import (
    LIGHT_LAYOUT,
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

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME HOOKS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="InsightFlow | Data Intelligence Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Detect runtime feature availability for backward/forward compatibility
HAS_SEGMENTED_CONTROL = hasattr(st, "segmented_control")
HAS_PILLS = hasattr(st, "pills")
HAS_POPOVER = hasattr(st, "popover")
HAS_ON_SELECT = "on_select" in inspect.signature(st.plotly_chart).parameters

# -----------------------------------------------------------------------------
# 2. DESIGN SYSTEM & CSS (Dark Shell + Light Analytical Canvas)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global Reset & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Branded Header Eyebrow */
    .brand-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #818CF8;
        margin-bottom: 2px;
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 0;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .brand-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* Light Analytical Workspace Card */
    .analytical-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 6px 12px -2px rgba(0, 0, 0, 0.02);
        color: #0F172A;
    }
    .analytical-card h3, .analytical-card h4 {
        color: #0F172A;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 8px;
    }
    .analytical-card p {
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }

    /* Light Metric KPI Box */
    .kpi-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .kpi-subtext {
        font-size: 0.75rem;
        color: #4F46E5;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Auto Insight Card */
    .insight-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #4F46E5;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .insight-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .insight-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0F172A;
    }
    .insight-badge {
        background-color: #EEF2FF;
        color: #4F46E5;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .insight-desc {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.45;
        margin: 0;
    }

    /* Status Pills */
    .status-pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-pill-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Filter Active Chip */
    .filter-chip {
        display: inline-flex;
        align-items: center;
        background-color: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #3730A3;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* Copilot Structured Cards */
    .copilot-response {
        background: #FFFFFF;
        border: 1px solid #E0E7FF;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.05);
    }
    .copilot-answer {
        font-size: 1.05rem;
        font-weight: 500;
        color: #1E293B;
        line-height: 1.6;
        margin-bottom: 16px;
    }

    /* Sidebar Dark Finish */
    [data-testid="stSidebar"] {
        background-color: #0A0D14;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #64748B !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
    }

    /* Button Polish */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.25);
    }

    /* Horizontal Segmented Control / Radio container alignment */
    div[data-testid="stRadio"] > div {
        background-color: #131B2E;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & CACHED PIPELINE
# -----------------------------------------------------------------------------
def get_sample_dataset_path() -> str:
    """Return the absolute path to the bundled sample dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data", "sample_employees.csv")


def init_session_state():
    """Initialize persistent session state variables."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None
    if "filters" not in st.session_state:
        st.session_state.filters = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Overview"
    if "ai_question" not in st.session_state:
        st.session_state.ai_question = ""
    if "last_ai_result" not in st.session_state:
        st.session_state.last_ai_result = None
    if "chart_selection_df" not in st.session_state:
        st.session_state.chart_selection_df = None


def load_dataset(file_or_path, name: str):
    """Load dataset into session state, resetting dependent UI caches."""
    try:
        df = load_csv(file_or_path)
        st.session_state.df = df
        st.session_state.dataset_name = name
        st.session_state.filters = []
        st.session_state.last_ai_result = None
        st.session_state.chart_selection_df = None
        st.success(f"Loaded **{name}** ({len(df):,} rows, {len(df.columns)} columns)")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def render_interactive_chart(fig, key: str = "chart", selection_mode=("points", "box", "lasso")):
    """
    Renders Plotly chart with interactive selection if supported by the Streamlit environment,
    falling back seamlessly to standard interactive display.
    """
    if HAS_ON_SELECT:
        try:
            return st.plotly_chart(
                fig,
                on_select="rerun",
                selection_mode=list(selection_mode),
                use_container_width=True,
                key=key,
            )
        except Exception:
            return st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        return st.plotly_chart(fig, use_container_width=True, key=key)


# -----------------------------------------------------------------------------
# 4. MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    init_session_state()

    # --- SIDEBAR (DATA CONTROL ROOM) ---
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #818CF8; letter-spacing: 0.1em; text-transform: uppercase;">PLATFORM</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">⚡ InsightFlow</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Data Intelligence Workspace</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Data Ingestion")
        uploaded_file = st.file_uploader(
            "Upload any CSV file",
            type=["csv"],
            help="Drop any CSV file to inspect schema, analyze distributions, audit quality, and query via AI.",
        )

        if uploaded_file is not None:
            if st.session_state.dataset_name != uploaded_file.name:
                load_dataset(uploaded_file, uploaded_file.name)

        sample_path = get_sample_dataset_path()
        if st.button("📁 Load Sample Employee Data", use_container_width=True):
            if os.path.exists(sample_path):
                load_dataset(sample_path, "sample_employees.csv")
                st.rerun()
            else:
                st.error("Sample dataset file not found.")

        # Active Dataset Inspector Card in Sidebar
        if st.session_state.df is not None:
            st.markdown("---")
            st.subheader("Active Dataset")
            df_raw = st.session_state.df
            num_filters = len(st.session_state.filters)

            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 14px; margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC; word-break: break-all;">{st.session_state.dataset_name}</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">
                        <span>{len(df_raw):,} rows</span> • <span>{len(df_raw.columns)} cols</span>
                    </div>
                    <div style="font-size: 0.75rem; color: {'#818CF8' if num_filters > 0 else '#64748B'}; margin-top: 4px;">
                        {'⚡ ' + str(num_filters) + ' active filter(s)' if num_filters > 0 else 'No active filters'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("🔄 Reset", use_container_width=True, help="Clear active filters"):
                    st.session_state.filters = []
                    st.session_state.chart_selection_df = None
                    st.rerun()
            with col_s2:
                if st.button("🗑️ Clear", use_container_width=True, help="Unload dataset"):
                    st.session_state.df = None
                    st.session_state.dataset_name = None
                    st.session_state.filters = []
                    st.session_state.last_ai_result = None
                    st.session_state.chart_selection_df = None
                    st.rerun()

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #64748B; line-height: 1.5;">
                <strong>Engine:</strong> Offline Heuristic AI<br>
                <strong>Security:</strong> 100% Private (No API Keys)<br>
                <strong>Version:</strong> 2.0 Production
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- TOP BRANDING & WORKSPACE BAR ---
    df_raw: Optional[pd.DataFrame] = st.session_state.df

    # Header Row
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.markdown('<div class="brand-eyebrow">DATA INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="brand-title">InsightFlow Analytics</h1>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-subtitle">Interactive exploratory analytics, context-aware visual studio, data quality diagnostics, and private copilot reasoning.</div>',
            unsafe_allow_html=True,
        )
    with top_col2:
        if df_raw is not None:
            sum_raw = get_dataset_summary(df_raw)
            if sum_raw["total_missing_cells"] == 0:
                st.markdown(
                    f'<div style="text-align: right; margin-top: 10px;"><span class="status-pill-green">● 100% Complete • {st.session_state.dataset_name}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="text-align: right; margin-top: 10px;"><span class="status-pill-amber">▲ {sum_raw["missing_percentage"]}% Missing • {st.session_state.dataset_name}</span></div>',
                    unsafe_allow_html=True,
                )

    # Empty State Onboarding if no dataset loaded
    if df_raw is None:
        st.markdown(
            """
            <div class="analytical-card" style="text-align: center; padding: 50px 30px; margin-top: 20px;">
                <div style="font-size: 2.8rem; margin-bottom: 12px;">📊</div>
                <h2 style="color: #0F172A; font-weight: 800; margin-bottom: 8px;">Welcome to InsightFlow</h2>
                <p style="color: #64748B; max-width: 580px; margin: 0 auto 24px auto; font-size: 0.95rem; line-height: 1.6;">
                    Upload any CSV dataset in the sidebar to start instant exploration, or load our curated sample dataset to test interactive filtering, the chart studio, quality audit, and heuristic copilot.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c1, col_c2, col_c3 = st.columns([1, 1.5, 1])
        with col_c2:
            if st.button("🚀 Load Sample Dataset & Begin Exploring", use_container_width=True, type="primary"):
                sample_path = get_sample_dataset_path()
                if os.path.exists(sample_path):
                    load_dataset(sample_path, "sample_employees.csv")
                    st.rerun()
                else:
                    st.error("Sample dataset not found.")
        return

    # --- FILTER PIPELINE ---
    df: pd.DataFrame = apply_filters(df_raw, st.session_state.filters)
    total_records = len(df_raw)
    active_records = len(df)
    is_filtered = len(st.session_state.filters) > 0

    # Filter Alert Banner if active
    if is_filtered:
        pct_kept = round((active_records / total_records) * 100, 1) if total_records > 0 else 0
        st.markdown(
            f"""
            <div style="background: rgba(79, 70, 229, 0.1); border: 1px solid rgba(79, 70, 229, 0.3); border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 0.85rem; color: #C7D2FE;">
                    <strong>Filter Active:</strong> Showing <strong>{active_records:,}</strong> of <strong>{total_records:,}</strong> records ({pct_kept}%).
                    All charts, workbench tables, quality metrics, and AI queries reflect this active slice.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- PRIMARY WORKSPACE NAVIGATION ---
    NAV_OPTIONS = [
        "📊 Overview",
        "📈 Chart Studio",
        "🎛️ Filter Lab",
        "📋 Data Workbench",
        "🛡️ Data Quality",
        "🤖 AI Copilot",
    ]

    if HAS_SEGMENTED_CONTROL:
        active_view = st.segmented_control("Navigation", NAV_OPTIONS, default=NAV_OPTIONS[0], label_visibility="collapsed")
    elif HAS_PILLS:
        active_view = st.pills("Navigation", NAV_OPTIONS, default=NAV_OPTIONS[0], label_visibility="collapsed")
    else:
        active_view = st.radio("Navigation", NAV_OPTIONS, index=0, horizontal=True, label_visibility="collapsed")

    summary = get_dataset_summary(df)
    arch = detect_column_types_extended(df)

    # -------------------------------------------------------------------------
    # SECTION 1: OVERVIEW
    # -------------------------------------------------------------------------
    if active_view == "📊 Overview":
        st.write("")
        # 6 KPI Metric Cards
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.metric("Total Records", f"{summary['num_rows']:,}")
        with k2:
            st.metric("Total Features", f"{summary['num_cols']}")
        with k3:
            st.metric("Completeness", f"{round(100 - summary['missing_percentage'], 1)}%")
        with k4:
            st.metric("Numeric Fields", f"{len(summary['numeric_columns'])}")
        with k5:
            st.metric("Categorical", f"{len(summary['categorical_columns'])}")
        with k6:
            st.metric("Memory", f"{summary['memory_usage_kb']} KB")

        st.write("")

        # Split Layout: Automatic Insights + Flagship Chart
        col_left, col_right = st.columns([1.1, 1.9])

        with col_left:
            st.markdown('<div class="analytical-card">', unsafe_allow_html=True)
            st.markdown("### What Stands Out?")
            st.markdown("<p>Automated statistical detections across the dataset.</p>", unsafe_allow_html=True)

            insights = detect_smart_insights(df)
            if insights:
                for ins in insights:
                    st.markdown(
                        f"""
                        <div class="insight-card">
                            <div class="insight-header">
                                <span class="insight-title">{ins['title']}</span>
                                <span class="insight-badge">{ins['badge']}</span>
                            </div>
                            <p class="insight-desc">{ins['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No significant anomalies or correlations detected.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="analytical-card">', unsafe_allow_html=True)
            st.markdown("### Flagship Signal")
            st.markdown("<p>Recommended primary visual for this dataset configuration.</p>", unsafe_allow_html=True)

            # Pick intelligent flagship chart
            if len(summary["numeric_columns"]) >= 2:
                fig_flagship = plot_scatter(
                    df,
                    summary["numeric_columns"][0],
                    summary["numeric_columns"][1],
                    color_col=summary["categorical_columns"][0] if summary["categorical_columns"] else None,
                    add_trendline=True,
                )
            elif summary["numeric_columns"]:
                fig_flagship = plot_distribution(df, summary["numeric_columns"][0], plot_type="Histogram")
            elif summary["categorical_columns"]:
                fig_flagship = plot_categorical_counts(df, summary["categorical_columns"][0], top_n=8)
            else:
                fig_flagship = plot_missing_values_bar(get_missing_values_summary(df))

            st.plotly_chart(fig_flagship, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 2: CHART STUDIO (PROFESSIONAL VISUAL BUILDER)
    # -------------------------------------------------------------------------
    elif active_view == "📈 Chart Studio":
        st.write("")
        st.markdown('<div class="analytical-card">', unsafe_allow_html=True)
        st.markdown("### Interactive Chart Studio")
        st.markdown("<p>Context-aware visualization builder with interactive point and box selection linked to data inspection.</p>", unsafe_allow_html=True)

        CHART_TYPES = [
            "Scatter",
            "Line",
            "Bar",
            "Area",
            "Histogram",
            "Box",
            "Violin",
            "Heatmap",
            "Donut",
        ]

        if HAS_SEGMENTED_CONTROL:
            chosen_chart = st.segmented_control("Select Chart Type", CHART_TYPES, default="Scatter")
        elif HAS_PILLS:
            chosen_chart = st.pills("Select Chart Type", CHART_TYPES, default="Scatter")
        else:
            chosen_chart = st.selectbox("Select Chart Type", CHART_TYPES, index=0)

        st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin: 16px 0 20px 0;'>", unsafe_allow_html=True)

        fig_studio = None

        # 1. SCATTER PLOT
        if chosen_chart == "Scatter":
            if len(summary["numeric_columns"]) >= 2:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    x_c = st.selectbox("X Axis (Numeric)", summary["numeric_columns"], index=0)
                with c2:
                    y_c = st.selectbox("Y Axis (Numeric)", summary["numeric_columns"], index=1 if len(summary["numeric_columns"]) > 1 else 0)
                with c3:
                    col_opts = ["None"] + summary["categorical_columns"] + summary["numeric_columns"]
                    c_c = st.selectbox("Color Dimension", col_opts, index=0)
                with c4:
                    size_opts = ["None"] + summary["numeric_columns"]
                    sz_c = st.selectbox("Size Dimension", size_opts, index=0)

                with st.popover("⚙️ Advanced Settings"):
                    trend_flag = st.checkbox("Fit Trendline (OLS)", value=False)
                    opacity_val = st.slider("Point Opacity", 0.1, 1.0, 0.8, 0.05)

                fig_studio = plot_scatter(
                    df,
                    x_col=x_c,
                    y_col=y_c,
                    color_col=c_c,
                    size_col=sz_c,
                    add_trendline=trend_flag,
                    opacity=opacity_val,
                )
            else:
                st.warning("At least 2 numeric columns are required for a Scatter Plot.")

        # 2. LINE PLOT
        elif chosen_chart == "Line":
            if summary["numeric_columns"]:
                all_possible_x = df.columns.tolist()
                c1, c2, c3 = st.columns(3)
                with c1:
                    x_l = st.selectbox("X Axis", all_possible_x, index=0)
                with c2:
                    y_l = st.selectbox("Y Axis (Numeric)", summary["numeric_columns"], index=0)
                with c3:
                    grp_opts = ["None"] + summary["categorical_columns"]
                    grp_l = st.selectbox("Group By", grp_opts, index=0)

                with st.popover("⚙️ Advanced Settings"):
                    show_markers = st.checkbox("Show Data Markers", value=True)

                fig_studio = plot_line(df, x_col=x_l, y_col=y_l, group_col=grp_l, show_markers=show_markers)
            else:
                st.warning("At least one numerical column is needed for a Line Chart.")

        # 3. BAR CHART
        elif chosen_chart == "Bar":
            if summary["categorical_columns"]:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    cat_b = st.selectbox("Category Column", summary["categorical_columns"], index=0)
                with c2:
                    metric_b = st.selectbox("Aggregation Metric", ["Count", "Sum", "Mean"], index=0)
                with c3:
                    val_b = None
                    if metric_b in ["Sum", "Mean"]:
                        if summary["numeric_columns"]:
                            val_b = st.selectbox("Value Column", summary["numeric_columns"], index=0)
                        else:
                            st.info("No numeric column for aggregation; reverting to Count.")
                            metric_b = "Count"
                    else:
                        st.text_input("Value Column", value="Record Frequency", disabled=True)
                with c4:
                    top_n_b = st.slider("Top Categories", 3, 30, 10)

                with st.popover("⚙️ Advanced Settings"):
                    orient_b = st.radio("Orientation", ["Vertical", "Horizontal"], horizontal=True)
                    sort_b = st.radio("Sort Order", ["Descending", "Ascending"], horizontal=True)

                fig_studio = plot_bar_advanced(
                    df,
                    cat_col=cat_b,
                    metric=metric_b,
                    val_col=val_b,
                    top_n=top_n_b,
                    orientation="v" if orient_b == "Vertical" else "h",
                    sort_order="desc" if sort_b == "Descending" else "asc",
                )
            else:
                st.warning("No categorical columns available for a Bar Chart.")

        # 4. AREA PLOT
        elif chosen_chart == "Area":
            if summary["numeric_columns"]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    x_a = st.selectbox("X Axis", df.columns.tolist(), index=0)
                with c2:
                    y_a = st.selectbox("Y Axis (Numeric)", summary["numeric_columns"], index=0)
                with c3:
                    grp_a = st.selectbox("Group By", ["None"] + summary["categorical_columns"], index=0)

                with st.popover("⚙️ Advanced Settings"):
                    stacked_a = st.checkbox("Stack Area Series", value=False)

                fig_studio = plot_area(df, x_col=x_a, y_col=y_a, group_col=grp_a, stacked=stacked_a)
            else:
                st.warning("Numeric columns required for Area Plot.")

        # 5. HISTOGRAM
        elif chosen_chart == "Histogram":
            if summary["numeric_columns"]:
                c1, c2 = st.columns([2, 1])
                with c1:
                    num_h = st.selectbox("Numerical Column", summary["numeric_columns"], index=0)
                with c2:
                    bins_h = st.slider("Bin Count", 5, 80, 25)

                fig_studio = plot_distribution(df, column=num_h, plot_type="Histogram", bins=bins_h)
            else:
                st.warning("No numeric columns found for Histogram.")

        # 6. BOX PLOT
        elif chosen_chart == "Box":
            if summary["numeric_columns"]:
                c1, c2 = st.columns(2)
                with c1:
                    num_box = st.selectbox("Metric Column (Numeric)", summary["numeric_columns"], index=0)
                with c2:
                    cat_box = st.selectbox("Group By (Optional)", ["None"] + summary["categorical_columns"], index=0)

                with st.popover("⚙️ Advanced Settings"):
                    pts = st.selectbox("Display Points", ["outliers", "all", "none"], index=0)

                fig_studio = plot_box(df, num_col=num_box, cat_col=cat_box, points=pts)
            else:
                st.warning("No numeric columns available for Box Plot.")

        # 7. VIOLIN PLOT
        elif chosen_chart == "Violin":
            if summary["numeric_columns"]:
                c1, c2 = st.columns(2)
                with c1:
                    num_v = st.selectbox("Metric Column", summary["numeric_columns"], index=0)
                with c2:
                    cat_v = st.selectbox("Segment By", ["None"] + summary["categorical_columns"], index=0)

                with st.popover("⚙️ Advanced Settings"):
                    box_overlay = st.checkbox("Overlay Box Inside", value=True)

                fig_studio = plot_violin(df, num_col=num_v, cat_col=cat_v, show_box=box_overlay)
            else:
                st.warning("No numeric columns available for Violin Plot.")

        # 8. CORRELATION HEATMAP
        elif chosen_chart == "Heatmap":
            if len(summary["numeric_columns"]) >= 2:
                sel_corr = st.multiselect(
                    "Select Numeric Columns",
                    options=summary["numeric_columns"],
                    default=summary["numeric_columns"][: min(10, len(summary["numeric_columns"]))],
                )
                with st.popover("⚙️ Advanced Settings"):
                    method_corr = st.selectbox("Correlation Method", ["pearson", "spearman"], index=0)

                if len(sel_corr) >= 2:
                    fig_studio = plot_correlation_heatmap(df, numeric_cols=sel_corr, method=method_corr)
                else:
                    st.info("Please select at least 2 numerical columns.")
            else:
                st.warning("At least 2 numeric columns are required for a Correlation Heatmap.")

        # 9. DONUT BREAKDOWN
        elif chosen_chart == "Donut":
            if summary["categorical_columns"]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    cat_d = st.selectbox("Category Field", summary["categorical_columns"], index=0)
                with c2:
                    val_d = st.selectbox("Value Field (Optional)", ["None (Count)"] + summary["numeric_columns"], index=0)
                with c3:
                    top_n_d = st.slider("Top Slices", 3, 15, 8)

                fig_studio = plot_donut(
                    df,
                    cat_col=cat_d,
                    val_col=val_d if val_d != "None (Count)" else None,
                    top_n=top_n_d,
                )
            else:
                st.warning("No categorical columns available for Donut chart.")

        # Render Chart and Link Interactive Selection
        if fig_studio is not None:
            chart_event = render_interactive_chart(fig_studio, key="main_studio_chart")

            # Check if user made a box/lasso/point selection
            if chart_event and isinstance(chart_event, dict) and "selection" in chart_event:
                sel_points = chart_event["selection"].get("points", [])
                if sel_points:
                    point_indices = [p["point_index"] for p in sel_points if "point_index" in p]
                    if point_indices and max(point_indices) < len(df):
                        st.session_state.chart_selection_df = df.iloc[point_indices]

            # Interactive Selection Panel Drawer
            if st.session_state.chart_selection_df is not None and not st.session_state.chart_selection_df.empty:
                st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin: 20px 0;'>", unsafe_allow_html=True)
                sel_df = st.session_state.chart_selection_df
                c_head1, c_head2 = st.columns([3, 1])
                with c_head1:
                    st.markdown(f"#### 🎯 Selected Records ({len(sel_df)} points)")
                    st.caption("Active cross-filter selection from chart points/box.")
                with c_head2:
                    if st.button("Clear Chart Selection", use_container_width=True):
                        st.session_state.chart_selection_df = None
                        st.rerun()

                st.dataframe(sel_df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 3: FILTER LAB (REAL MULTI-CLAUSE FILTER ENGINE)
    # -------------------------------------------------------------------------
    elif active_view == "🎛️ Filter Lab":
        st.write("")
        st.markdown('<div class="analytical-card">', unsafe_allow_html=True)
        st.markdown("### Multi-Clause Filter Builder")
        st.markdown("<p>Define precise filtering rules across numeric, text, and date columns. Active filters propagate across all views.</p>", unsafe_allow_html=True)

        # Active Filter Chips
        if st.session_state.filters:
            st.markdown("##### Active Filters:")
            chip_cols = st.columns([4, 1])
            with chip_cols[0]:
                for idx, flt in enumerate(st.session_state.filters):
                    st.markdown(
                        f"""<span class="filter-chip">🏷️ <strong>{flt['column']}</strong> &nbsp;{flt['operator']}&nbsp; <em>{flt['value']}</em></span>""",
                        unsafe_allow_html=True,
                    )
            with chip_cols[1]:
                if st.button("🗑️ Clear All Filters", use_container_width=True):
                    st.session_state.filters = []
                    st.rerun()
        else:
            st.info("No active filters. Showing 100% of the loaded dataset.")

        st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin: 16px 0;'>", unsafe_allow_html=True)

        # Add Filter Rule Form
        st.markdown("##### Add Filter Rule")
        col_f1, col_f2, col_f3, col_f4 = st.columns([1.5, 1.5, 2, 1])

        with col_f1:
            filt_col = st.selectbox("Select Column", df_raw.columns.tolist(), key="filt_col_sel")

        is_num = pd.api.types.is_numeric_dtype(df_raw[filt_col])

        with col_f2:
            if is_num:
                op_options = ["greater than (>)", "less than (<)", "greater or equal (>=)", "less or equal (<=)", "equals", "does not equal", "is null", "is not null"]
            else:
                op_options = ["contains", "equals", "does not equal", "is in", "is null", "is not null"]
            filt_op = st.selectbox("Operator", op_options, key="filt_op_sel")

        with col_f3:
            filt_val = None
            if filt_op not in ["is null", "is not null"]:
                if is_num:
                    s_clean = df_raw[filt_col].dropna()
                    min_v = float(s_clean.min()) if not s_clean.empty else 0.0
                    max_v = float(s_clean.max()) if not s_clean.empty else 100.0
                    mean_v = float(s_clean.mean()) if not s_clean.empty else 50.0
                    filt_val = st.number_input("Value", value=round(mean_v, 2), key="filt_val_num")
                elif filt_op == "is in":
                    u_vals = df_raw[filt_col].dropna().unique().tolist()
                    filt_val = st.multiselect("Select Categories", u_vals[:50], key="filt_val_multi")
                else:
                    filt_val = st.text_input("Match String", placeholder="e.g. Sales or tech", key="filt_val_text")

        with col_f4:
            st.write("")
            st.write("")
            if st.button("➕ Add Rule", type="primary", use_container_width=True):
                if filt_op in ["is null", "is not null"] or filt_val is not None:
                    st.session_state.filters.append({
                        "column": filt_col,
                        "operator": filt_op,
                        "value": filt_val,
                    })
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Filtered Result Summary Box
        st.markdown('<div class="analytical-card">', unsafe_allow_html=True)
        st.markdown(f"#### Active Dataset Slice ({len(df):,} of {len(df_raw):,} records)")
        st.dataframe(df.head(50), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 4: DATA WORKBENCH (DATA TABLE)
    # -------------------------------------------------------------------------
    elif active_view == "📋 Data Workbench":
        st.write("")
        st.markdown('<div class="analytical-card">', unsafe_allow_html=True)
        st.markdown("### Data Workbench")
        st.markdown("<p>Inspect rows with customizable presets, random sampling, column selection, and CSV export.</p>", unsafe_allow_html=True)

        # Controls Toolbar
        c_tb1, c_tb2, c_tb3, c_tb4 = st.columns([1.5, 1.5, 2.5, 1.5])
        with c_tb1:
            preset_rows = st.selectbox("Display Rows", ["25", "50", "100", "250", "All", "Custom"], index=1)
            if preset_rows == "All":
                slice_n = len(df)
            elif preset_rows == "Custom":
                slice_n = st.slider("Custom Count", min_value=1, max_value=min(1000, len(df)), value=min(50, len(df)))
            else:
                slice_n = min(int(preset_rows), len(df))

        with c_tb2:
            view_mode = st.selectbox("Slice Mode", ["First Rows", "Random Sample"], index=0)

        with c_tb3:
            all_cols = df.columns.tolist()
            visible_cols = st.multiselect("Visible Columns", options=all_cols, default=all_cols)

        with c_tb4:
            st.write("")
            csv_bytes = df[visible_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export CSV",
                data=csv_bytes,
                file_name=f"{st.session_state.dataset_name or 'dataset'}_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if not visible_cols:
            st.warning("Please select at least one column to display.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Prepare table slice
        display_df = df[visible_cols]
        if view_mode == "Random Sample" and len(display_df) > slice_n:
            display_slice = display_df.sample(n=slice_n, random_state=42)
        else:
            display_slice = display_df.head(slice_n)

        # Quick In-Table Search
        search_query = st.text_input("🔍 Quick Search Slice", placeholder="Filter rows in current slice by any keyword...")
        if search_query:
            mask = display_slice.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            display_slice = display_slice[mask]

        st.dataframe(display_slice, use_container_width=True)

        st.caption(f"Displaying {len(display_slice):,} rows across {len(visible_cols)} columns. Total active dataset size: {len(df):,} records.")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 5: DATA QUALITY CENTER
    # -------------------------------------------------------------------------
    elif active_view == "🛡️ Data Quality":
        st.write("")
        quality_rep = get_data_quality_report(df)

        st.markdown('<div class="analytical-card">', unsafe_allow_html=True)
        st.markdown("### Data Quality Diagnostics")
        st.markdown("<p>Comprehensive audit of completeness, duplicate records, zero-variance columns, and missingness severity.</p>", unsafe_allow_html=True)

        # Quality KPI Row
        q1, q2, q3, q4, q5 = st.columns(5)
        with q1:
            st.metric("Health Score", f"{quality_rep['quality_score']}/100")
        with q2:
            st.metric("Completeness", f"{quality_rep['completeness_pct']}%")
        with q3:
            st.metric("Missing Cells", f"{quality_rep['missing_cells']:,}")
        with q4:
            st.metric("Duplicate Rows", f"{quality_rep['duplicate_rows']:,}")
        with q5:
            st.metric("Constant Columns", f"{len(quality_rep['constant_columns'])}")

        st.write("")

        # Missingness Chart + Recommendations
        c_q_left, c_q_right = st.columns([1.5, 1])
        with c_q_left:
            st.markdown("##### Missingness Percentage by Column")
            missing_summary = get_missing_values_summary(df)
            fig_miss = plot_missing_values_bar(missing_summary)
            st.plotly_chart(fig_miss, use_container_width=True)

        with c_q_right:
            st.markdown("##### Actionable Diagnostic Insights")
            for sug in quality_rep["suggestions"]:
                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; font-size: 0.85rem; color: #1E293B;">
                        {sug}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Full Column Audit Table
        st.markdown("##### Detailed Feature Integrity Audit")
        st.dataframe(quality_rep["audit_df"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 6: AI COPILOT (DATASET INTELLIGENCE ASSISTANT)
    # -------------------------------------------------------------------------
    elif active_view == "🤖 AI Copilot":
        st.write("")
        st.markdown('<div class="analytical-card">', unsafe_allow_html=True)
        st.markdown("### Dataset Copilot (Offline & Private)")
        st.markdown("<p>Ask natural-language questions about averages, maximums, distributions, missing values, and correlations. Powered by a deterministic local engine.</p>", unsafe_allow_html=True)

        # Quick Prompt Chips
        st.markdown("##### Suggested Inquiries:")
        p1 = f"What is the average {summary['numeric_columns'][0]}?" if summary["numeric_columns"] else "How many rows are in the dataset?"
        p2 = "Which column has missing values?"
        p3 = "What is the correlation between numeric columns?" if len(summary["numeric_columns"]) >= 2 else "What is the dataset overview?"
        p4 = f"What is the highest {summary['numeric_columns'][0]}?" if summary["numeric_columns"] else "List all columns"

        q_cols = st.columns(4)
        with q_cols[0]:
            if st.button(f"💡 {p1}", key="p_btn1", use_container_width=True):
                st.session_state.ai_question = p1
                st.session_state.last_ai_result = analyze_question(p1, df)
        with q_cols[1]:
            if st.button(f"🔍 {p2}", key="p_btn2", use_container_width=True):
                st.session_state.ai_question = p2
                st.session_state.last_ai_result = analyze_question(p2, df)
        with q_cols[2]:
            if st.button(f"📈 {p3}", key="p_btn3", use_container_width=True):
                st.session_state.ai_question = p3
                st.session_state.last_ai_result = analyze_question(p3, df)
        with q_cols[3]:
            if st.button(f"📋 {p4}", key="p_btn4", use_container_width=True):
                st.session_state.ai_question = p4
                st.session_state.last_ai_result = analyze_question(p4, df)

        st.write("")

        # Natural Language Question Form
        with st.form("copilot_query_form"):
            user_input = st.text_input(
                "Your Dataset Question:",
                value=st.session_state.ai_question,
                placeholder="e.g. What is the average Salary? or Which columns have missing data?",
            )
            ask_submitted = st.form_submit_button("Ask Copilot", type="primary", use_container_width=False)

        if ask_submitted and user_input:
            st.session_state.ai_question = user_input
            st.session_state.last_ai_result = analyze_question(user_input, df)

        # Structured Copilot Response
        if st.session_state.last_ai_result:
            res = st.session_state.last_ai_result
            st.markdown(
                f"""
                <div class="copilot-response">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #4F46E5; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">DIRECT ANSWER</div>
                    <div class="copilot-answer">{res['answer']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Evidence / Analyzed Fields
            if res.get("evidence_cols"):
                st.markdown("##### 📌 Evidence & Features Evaluated")
                ev_html = "".join([f'<span class="filter-chip">📊 {c}</span>' for c in res["evidence_cols"]])
                st.markdown(ev_html, unsafe_allow_html=True)

            # Key Insights
            if res.get("insights"):
                st.markdown("##### 💡 Analytical Insights")
                for ins in res["insights"]:
                    st.markdown(f"- {ins}")

            # Suggested Follow-Up Questions
            if res.get("suggested_questions"):
                st.markdown("##### ❓ Suggested Next Inquiries")
                for sq in res["suggested_questions"]:
                    st.markdown(f"- `{sq}`")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
