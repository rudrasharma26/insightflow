"""
InsightFlow | Production-Grade Exploratory Data Intelligence Workspace
Features:
- Dark application chrome with layered local background image & atmospheric gradients
- Dark glass container surfaces (rgba(10, 18, 34, 0.72) with backdrop blur)
- Light analytical canvases for Plotly charts and dataframes
- Strict contrast & typography hierarchy
- Compact 4-KPI Overview with Dataset Pulse & What's Interesting?
- Chart Studio with compact builder & live preview focus
- Query builder Filter Lab with [Column] [Operator] [Value] [Remove] rules
- Dense Data Workbench with instant search, presets, sampling, and export
- Data Quality Diagnostics center with healthy-state indicator & corrected zero-variance logic
- Private, 100% offline heuristic AI Copilot
"""

from typing import Any, Dict, List, Optional
import base64
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
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="InsightFlow | Data Intelligence Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

HAS_SEGMENTED_CONTROL = hasattr(st, "segmented_control")
HAS_PILLS = hasattr(st, "pills")
HAS_POPOVER = hasattr(st, "popover")
HAS_ON_SELECT = "on_select" in inspect.signature(st.plotly_chart).parameters


# -----------------------------------------------------------------------------
# 2. LOCAL ASSET & BACKGROUND PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_background_base64() -> Optional[str]:
    """Load local background image assets/insightflow_bg.png as base64 string."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bg_path = os.path.join(base_dir, "assets", "insightflow_bg.png")
    if os.path.exists(bg_path):
        try:
            with open(bg_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None
    return None


bg_base64 = load_background_base64()

if bg_base64:
    bg_css_rule = f"""
    .stApp {{
        background-image: 
            radial-gradient(ellipse at 18% 15%, rgba(99, 102, 241, 0.16) 0%, transparent 45%),
            radial-gradient(ellipse at 82% 80%, rgba(6, 182, 212, 0.12) 0%, transparent 45%),
            linear-gradient(180deg, rgba(8, 12, 22, 0.88) 0%, rgba(10, 15, 28, 0.94) 100%),
            url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    """
else:
    bg_css_rule = """
    .stApp {
        background: 
            radial-gradient(ellipse at 18% 15%, rgba(99, 102, 241, 0.16) 0%, transparent 45%),
            radial-gradient(ellipse at 82% 80%, rgba(6, 182, 212, 0.12) 0%, transparent 45%),
            linear-gradient(180deg, #080C16 0%, #0A0F1C 100%);
        background-attachment: fixed;
    }
    """

# -----------------------------------------------------------------------------
# 3. GLOBAL STYLING: DARK SHELL + DARK GLASS CARDS + LIGHT ANALYTICAL CANVASES
# -----------------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    {bg_css_rule}

    /* Subtle Faint Grid Texture Overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: linear-gradient(rgba(255, 255, 255, 0.012) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255, 255, 255, 0.012) 1px, transparent 1px);
        background-size: 48px 48px;
        pointer-events: none;
        z-index: 0;
    }}

    /* Premium Dark Glass Container Cards (Replaces solid white boxes) */
    .dark-glass-card {{
        background: rgba(10, 18, 34, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.32);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .dark-glass-card:hover {{
        border-color: rgba(99, 102, 241, 0.28);
        box-shadow: 0 8px 32px 0 rgba(79, 70, 229, 0.1);
    }}
    .dark-glass-card h3, .dark-glass-card h4 {{
        color: #F8FAFC !important;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }}
    .dark-glass-card p {{
        color: #94A3B8;
        font-size: 0.9rem;
        margin-bottom: 14px;
        line-height: 1.5;
    }}

    /* Light Analytical Canvas Card (Charts & Tables stay LIGHT for contrast) */
    .light-canvas-card {{
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
        color: #0F172A;
    }}
    .light-canvas-card h3, .light-canvas-card h4 {{
        color: #0F172A !important;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 6px;
    }}
    .light-canvas-card p {{
        color: #475569;
        font-size: 0.88rem;
        margin-bottom: 14px;
    }}

    /* Compact Dark KPI Box */
    .dark-kpi-card {{
        background: rgba(10, 18, 34, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 16px 20px;
        backdrop-filter: blur(14px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .dark-kpi-card:hover {{
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-1px);
    }}
    .dark-kpi-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 4px;
    }}
    .dark-kpi-val {{
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }}
    .dark-kpi-sub {{
        font-size: 0.74rem;
        color: #818CF8;
        font-weight: 600;
        margin-top: 2px;
    }}

    /* Dataset Pulse Compact Panel */
    .dataset-pulse-bar {{
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        padding: 14px 20px;
        display: flex;
        flex-wrap: wrap;
        gap: 24px;
        align-items: center;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }}
    .pulse-item {{
        display: flex;
        flex-direction: column;
    }}
    .pulse-title {{
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .pulse-data {{
        font-size: 1.05rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 2px;
    }}

    /* Compact Dark Insight Cards */
    .dark-insight-card {{
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-left: 4px solid #6366F1;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}
    .dark-insight-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }}
    .dark-insight-title {{
        font-size: 0.88rem;
        font-weight: 700;
        color: #F8FAFC;
    }}
    .dark-insight-badge {{
        background-color: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }}
    .dark-insight-desc {{
        font-size: 0.82rem;
        color: #94A3B8;
        line-height: 1.45;
        margin: 0;
    }}

    /* Header Eyebrow & Hero Hierarchy */
    .brand-eyebrow {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #818CF8;
        margin-bottom: 4px;
    }}
    .brand-title {{
        font-size: 2.3rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 6px 0;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }}
    .brand-subtitle {{
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 20px;
        line-height: 1.5;
        max-width: 750px;
    }}

    /* Filter Active Chip */
    .filter-chip {{
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #C7D2FE;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }}

    /* Sidebar Dark Glass Finish */
    [data-testid="stSidebar"] {{
        background-color: rgba(8, 12, 22, 0.88) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
        backdrop-filter: blur(16px);
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: #F8FAFC !important;
    }}

    /* Streamlit Metric Overrides to Dark Glass */
    div[data-testid="stMetric"] {{
        background: rgba(10, 18, 34, 0.72) !important;
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        backdrop-filter: blur(14px) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #94A3B8 !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        color: #F8FAFC !important;
    }}

    /* Button Polish */
    .stButton button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    .stButton button:hover {{
        border-color: #6366F1 !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.35);
    }}
    .stButton button[kind="primary"] {{
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
    }}

    /* Navigation Bar Container (Pill-Style) */
    div[data-testid="stRadio"] > div {{
        background: rgba(10, 18, 34, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
        border-radius: 14px !important;
        padding: 5px 8px !important;
        gap: 6px !important;
        backdrop-filter: blur(12px) !important;
    }}
    div[data-testid="stRadio"] label {{
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        transition: all 0.15s ease !important;
    }}
    div[data-testid="stRadio"] label:hover {{
        color: #FFFFFF !important;
        background: rgba(99, 102, 241, 0.15) !important;
    }}

    /* Micro-interactions: prefers-reduced-motion */
    @media (prefers-reduced-motion: reduce) {{
        * {{
            transition: none !important;
            animation: none !important;
        }}
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
def get_sample_dataset_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data", "sample_employees.csv")


def init_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None
    if "filters" not in st.session_state:
        st.session_state.filters = []
    if "ai_question" not in st.session_state:
        st.session_state.ai_question = ""
    if "last_ai_result" not in st.session_state:
        st.session_state.last_ai_result = None
    if "chart_selection_df" not in st.session_state:
        st.session_state.chart_selection_df = None


def load_dataset(file_or_path, name: str):
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
# 5. MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    init_session_state()

    # --- SIDEBAR (DATA CONTROL ROOM) ---
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #818CF8; letter-spacing: 0.12em; text-transform: uppercase;">PLATFORM</div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">⚡ InsightFlow</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Data Intelligence Workspace</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### DATA SOURCE")
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            label_visibility="collapsed",
            help="Drop any CSV file to inspect schema, analyze distributions, audit quality, and query via AI.",
        )

        if uploaded_file is not None:
            if st.session_state.dataset_name != uploaded_file.name:
                load_dataset(uploaded_file, uploaded_file.name)

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("##### QUICK START")
        sample_path = get_sample_dataset_path()
        if st.button("📁 Load Sample Dataset", use_container_width=True):
            if os.path.exists(sample_path):
                load_dataset(sample_path, "sample_employees.csv")
                st.rerun()
            else:
                st.error("Sample dataset file not found.")

        # Active Dataset Inspector
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("##### ACTIVE DATASET")
            df_raw = st.session_state.df
            num_filters = len(st.session_state.filters)

            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC; word-break: break-all;">{st.session_state.dataset_name}</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">
                        <span>{len(df_raw):,} rows</span> • <span>{len(df_raw.columns)} cols</span>
                    </div>
                    <div style="font-size: 0.74rem; color: {'#A5B4FC' if num_filters > 0 else '#64748B'}; margin-top: 4px; font-weight: 600;">
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
                <strong>Engine:</strong> Deterministic Heuristic AI<br>
                <strong>Security:</strong> 100% Private (No External APIs)<br>
                <strong>Workspace:</strong> Production
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- TOP WORKSPACE HEADER ---
    df_raw: Optional[pd.DataFrame] = st.session_state.df

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.markdown('<div class="brand-eyebrow">DATA INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="brand-title">InsightFlow</h1>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-subtitle">Interactive exploration, visual analytics, diagnostics and private AI reasoning.</div>',
            unsafe_allow_html=True,
        )
    with top_col2:
        if df_raw is not None:
            sum_raw = get_dataset_summary(df_raw)
            if sum_raw["total_missing_cells"] == 0:
                st.markdown(
                    f'<div style="text-align: right; margin-top: 10px;"><span style="background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">● 100% Complete • {st.session_state.dataset_name}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="text-align: right; margin-top: 10px;"><span style="background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">▲ {sum_raw["missing_percentage"]}% Missing • {st.session_state.dataset_name}</span></div>',
                    unsafe_allow_html=True,
                )

    # Empty State Hero Card
    if df_raw is None:
        st.markdown(
            """
            <div class="dark-glass-card" style="text-align: center; padding: 48px 30px; margin-top: 10px;">
                <div style="font-size: 2.8rem; margin-bottom: 12px;">📊</div>
                <h2 style="color: #F8FAFC; font-weight: 800; margin-bottom: 8px;">Welcome to InsightFlow</h2>
                <p style="color: #94A3B8; max-width: 580px; margin: 0 auto 24px auto; font-size: 0.95rem; line-height: 1.6;">
                    Upload any CSV dataset in the sidebar to start instant exploration, or load our curated sample dataset to test interactive filtering, the chart studio, quality diagnostics, and the heuristic copilot.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_e1, c_e2, c_e3 = st.columns([1, 1.5, 1])
        with c_e2:
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

    if is_filtered:
        pct_kept = round((active_records / total_records) * 100, 1) if total_records > 0 else 0
        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 10px 16px; margin-bottom: 16px;">
                <span style="font-size: 0.85rem; color: #C7D2FE;">
                    <strong>Active Filter:</strong> Displaying <strong>{active_records:,}</strong> of <strong>{total_records:,}</strong> records ({pct_kept}%).
                    All charts, workbench tables, diagnostics, and AI queries reflect this active slice.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- NAVIGATION BAR ---
    NAV_OPTIONS = [
        "Overview",
        "Chart Studio",
        "Filter Lab",
        "Data Workbench",
        "Data Quality",
        "AI Copilot",
    ]

    if HAS_SEGMENTED_CONTROL:
        active_view = st.segmented_control("Navigation", NAV_OPTIONS, default=NAV_OPTIONS[0], label_visibility="collapsed")
    elif HAS_PILLS:
        active_view = st.pills("Navigation", NAV_OPTIONS, default=NAV_OPTIONS[0], label_visibility="collapsed")
    else:
        active_view = st.radio("Navigation", NAV_OPTIONS, index=0, horizontal=True, label_visibility="collapsed")

    summary = get_dataset_summary(df)

    # -------------------------------------------------------------------------
    # SECTION 1: OVERVIEW
    # -------------------------------------------------------------------------
    if active_view == "Overview":
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # 4 Compact KPI Cards Maximum
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Records", f"{summary['num_rows']:,}")
        with k2:
            st.metric("Columns", f"{summary['num_cols']}")
        with k3:
            st.metric("Completeness", f"{round(100 - summary['missing_percentage'], 1)}%")
        with k4:
            st.metric("Memory", f"{summary['memory_usage_kb']} KB")

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

        # DATASET PULSE
        st.markdown(
            f"""
            <div class="dataset-pulse-bar">
                <div class="pulse-item">
                    <span class="pulse-title">RECORD COUNT</span>
                    <span class="pulse-data">{summary['num_rows']:,}</span>
                </div>
                <div class="pulse-item">
                    <span class="pulse-title">FIELD COUNT</span>
                    <span class="pulse-data">{summary['num_cols']}</span>
                </div>
                <div class="pulse-item">
                    <span class="pulse-title">COMPLETENESS</span>
                    <span class="pulse-data">{round(100 - summary['missing_percentage'], 1)}%</span>
                </div>
                <div class="pulse-item">
                    <span class="pulse-title">NUMERIC FIELDS</span>
                    <span class="pulse-data">{len(summary['numeric_columns'])}</span>
                </div>
                <div class="pulse-item">
                    <span class="pulse-title">CATEGORICAL FIELDS</span>
                    <span class="pulse-data">{len(summary['categorical_columns'])}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Split Layout: WHAT'S INTERESTING? + PRIMARY SIGNAL
        col_ov_left, col_ov_right = st.columns([1.1, 1.9])

        with col_ov_left:
            st.markdown('<div class="dark-glass-card">', unsafe_allow_html=True)
            st.markdown("### WHAT'S INTERESTING?")
            st.markdown("<p>Automated statistical detections across the dataset.</p>", unsafe_allow_html=True)

            insights = detect_smart_insights(df)
            if insights:
                for ins in insights:
                    st.markdown(
                        f"""
                        <div class="dark-insight-card">
                            <div class="dark-insight-header">
                                <span class="dark-insight-title">{ins['title']}</span>
                                <span class="dark-insight-badge">{ins['badge']}</span>
                            </div>
                            <p class="dark-insight-desc">{ins['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No significant anomalies or correlations detected.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_ov_right:
            st.markdown('<div class="light-canvas-card">', unsafe_allow_html=True)
            st.markdown("### PRIMARY SIGNAL")
            st.markdown("<p>Recommended flagship visualization based on dataset topology.</p>", unsafe_allow_html=True)

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
    # SECTION 2: CHART STUDIO
    # -------------------------------------------------------------------------
    elif active_view == "Chart Studio":
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        col_studio_ctrl, col_studio_preview = st.columns([1.1, 2.1])

        with col_studio_ctrl:
            st.markdown('<div class="dark-glass-card">', unsafe_allow_html=True)
            st.markdown("### CHART BUILDER")
            st.markdown("<p>Configure parameters for interactive analysis.</p>", unsafe_allow_html=True)

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

            chosen_chart = st.selectbox("Chart Type", CHART_TYPES, index=0)
            fig_studio = None

            # 1. SCATTER
            if chosen_chart == "Scatter":
                if len(summary["numeric_columns"]) >= 2:
                    x_c = st.selectbox("X Axis (Numeric)", summary["numeric_columns"], index=0)
                    y_c = st.selectbox("Y Axis (Numeric)", summary["numeric_columns"], index=1 if len(summary["numeric_columns"]) > 1 else 0)
                    col_opts = ["None"] + summary["categorical_columns"] + summary["numeric_columns"]
                    c_c = st.selectbox("Color Dimension", col_opts, index=0)
                    size_opts = ["None"] + summary["numeric_columns"]
                    sz_c = st.selectbox("Size Dimension", size_opts, index=0)

                    with st.expander("⚙️ Advanced Settings"):
                        trend_flag = st.checkbox("Fit Trendline (OLS)", value=False)
                        opacity_val = st.slider("Point Opacity", 0.1, 1.0, 0.8, 0.05)

                    fig_studio = plot_scatter(
                        df, x_col=x_c, y_col=y_c, color_col=c_c, size_col=sz_c, add_trendline=trend_flag, opacity=opacity_val
                    )
                else:
                    st.warning("At least 2 numeric columns are required for a Scatter Plot.")

            # 2. LINE
            elif chosen_chart == "Line":
                if summary["numeric_columns"]:
                    x_l = st.selectbox("X Axis", df.columns.tolist(), index=0)
                    y_l = st.selectbox("Y Axis (Numeric)", summary["numeric_columns"], index=0)
                    grp_l = st.selectbox("Group By", ["None"] + summary["categorical_columns"], index=0)
                    with st.expander("⚙️ Advanced Settings"):
                        show_markers = st.checkbox("Show Data Markers", value=True)
                    fig_studio = plot_line(df, x_col=x_l, y_col=y_l, group_col=grp_l, show_markers=show_markers)
                else:
                    st.warning("At least one numerical column is needed for a Line Chart.")

            # 3. BAR
            elif chosen_chart == "Bar":
                if summary["categorical_columns"]:
                    cat_b = st.selectbox("Category Column", summary["categorical_columns"], index=0)
                    metric_b = st.selectbox("Aggregation Metric", ["Count", "Sum", "Mean"], index=0)
                    val_b = None
                    if metric_b in ["Sum", "Mean"]:
                        if summary["numeric_columns"]:
                            val_b = st.selectbox("Value Column", summary["numeric_columns"], index=0)
                        else:
                            metric_b = "Count"
                    top_n_b = st.slider("Top Categories", 3, 30, 10)
                    with st.expander("⚙️ Advanced Settings"):
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

            # 4. AREA
            elif chosen_chart == "Area":
                if summary["numeric_columns"]:
                    x_a = st.selectbox("X Axis", df.columns.tolist(), index=0)
                    y_a = st.selectbox("Y Axis (Numeric)", summary["numeric_columns"], index=0)
                    grp_a = st.selectbox("Group By", ["None"] + summary["categorical_columns"], index=0)
                    with st.expander("⚙️ Advanced Settings"):
                        stacked_a = st.checkbox("Stack Area Series", value=False)
                    fig_studio = plot_area(df, x_col=x_a, y_col=y_a, group_col=grp_a, stacked=stacked_a)
                else:
                    st.warning("Numeric columns required for Area Plot.")

            # 5. HISTOGRAM
            elif chosen_chart == "Histogram":
                if summary["numeric_columns"]:
                    num_h = st.selectbox("Numerical Column", summary["numeric_columns"], index=0)
                    bins_h = st.slider("Bin Count", 5, 80, 25)
                    fig_studio = plot_distribution(df, column=num_h, plot_type="Histogram", bins=bins_h)
                else:
                    st.warning("No numeric columns found for Histogram.")

            # 6. BOX
            elif chosen_chart == "Box":
                if summary["numeric_columns"]:
                    num_box = st.selectbox("Metric Column (Numeric)", summary["numeric_columns"], index=0)
                    cat_box = st.selectbox("Group By (Optional)", ["None"] + summary["categorical_columns"], index=0)
                    with st.expander("⚙️ Advanced Settings"):
                        pts = st.selectbox("Display Points", ["outliers", "all", "none"], index=0)
                    fig_studio = plot_box(df, num_col=num_box, cat_col=cat_box, points=pts)
                else:
                    st.warning("No numeric columns available for Box Plot.")

            # 7. VIOLIN
            elif chosen_chart == "Violin":
                if summary["numeric_columns"]:
                    num_v = st.selectbox("Metric Column", summary["numeric_columns"], index=0)
                    cat_v = st.selectbox("Segment By", ["None"] + summary["categorical_columns"], index=0)
                    with st.expander("⚙️ Advanced Settings"):
                        box_overlay = st.checkbox("Overlay Box Inside", value=True)
                    fig_studio = plot_violin(df, num_col=num_v, cat_col=cat_v, show_box=box_overlay)
                else:
                    st.warning("No numeric columns available for Violin Plot.")

            # 8. HEATMAP
            elif chosen_chart == "Heatmap":
                if len(summary["numeric_columns"]) >= 2:
                    sel_corr = st.multiselect(
                        "Select Numeric Columns",
                        options=summary["numeric_columns"],
                        default=summary["numeric_columns"][: min(10, len(summary["numeric_columns"]))],
                    )
                    with st.expander("⚙️ Advanced Settings"):
                        method_corr = st.selectbox("Correlation Method", ["pearson", "spearman"], index=0)
                    if len(sel_corr) >= 2:
                        fig_studio = plot_correlation_heatmap(df, numeric_cols=sel_corr, method=method_corr)
                    else:
                        st.info("Please select at least 2 numerical columns.")
                else:
                    st.warning("At least 2 numeric columns are required for a Correlation Heatmap.")

            # 9. DONUT
            elif chosen_chart == "Donut":
                if summary["categorical_columns"]:
                    cat_d = st.selectbox("Category Field", summary["categorical_columns"], index=0)
                    val_d = st.selectbox("Value Field (Optional)", ["None (Count)"] + summary["numeric_columns"], index=0)
                    top_n_d = st.slider("Top Slices", 3, 15, 8)
                    fig_studio = plot_donut(
                        df, cat_col=cat_d, val_col=val_d if val_d != "None (Count)" else None, top_n=top_n_d
                    )
                else:
                    st.warning("No categorical columns available for Donut chart.")

            st.markdown("</div>", unsafe_allow_html=True)

        with col_studio_preview:
            st.markdown('<div class="light-canvas-card">', unsafe_allow_html=True)
            st.markdown("### LIVE PREVIEW")
            st.markdown("<p>Interactive analytical surface with box, lasso, and point cross-filtering.</p>", unsafe_allow_html=True)

            if fig_studio is not None:
                chart_event = render_interactive_chart(fig_studio, key="main_studio_chart")

                if chart_event and isinstance(chart_event, dict) and "selection" in chart_event:
                    sel_points = chart_event["selection"].get("points", [])
                    if sel_points:
                        point_indices = [p["point_index"] for p in sel_points if "point_index" in p]
                        if point_indices and max(point_indices) < len(df):
                            st.session_state.chart_selection_df = df.iloc[point_indices]

                if st.session_state.chart_selection_df is not None and not st.session_state.chart_selection_df.empty:
                    st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin: 18px 0;'>", unsafe_allow_html=True)
                    sel_df = st.session_state.chart_selection_df
                    c_sh1, c_sh2 = st.columns([3, 1])
                    with c_sh1:
                        st.markdown(f"#### 🎯 Selected Records ({len(sel_df)} points)")
                    with c_sh2:
                        if st.button("Clear Selection", use_container_width=True):
                            st.session_state.chart_selection_df = None
                            st.rerun()
                    st.dataframe(sel_df, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 3: FILTER LAB
    # -------------------------------------------------------------------------
    elif active_view == "Filter Lab":
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="dark-glass-card">', unsafe_allow_html=True)
        st.markdown("### FILTER LAB")
        st.markdown("<p>Construct multi-clause query rules to isolate targeted segments across the dataset.</p>", unsafe_allow_html=True)

        if st.session_state.filters:
            st.markdown("##### ACTIVE RULES")
            c_chip1, c_chip2 = st.columns([4, 1])
            with c_chip1:
                for idx, flt in enumerate(st.session_state.filters):
                    st.markdown(
                        f"""<span class="filter-chip">🏷️ <strong>{flt['column']}</strong> &nbsp;{flt['operator']}&nbsp; <em>{flt['value']}</em></span>""",
                        unsafe_allow_html=True,
                    )
            with c_chip2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.filters = []
                    st.rerun()
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

        # Rule Builder Row: [Column] [Operator] [Value] [Add Rule]
        col_f1, col_f2, col_f3, col_f4 = st.columns([1.5, 1.5, 2, 1])
        with col_f1:
            filt_col = st.selectbox("Column", df_raw.columns.tolist(), key="filt_col_sel")

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
                    mean_v = float(s_clean.mean()) if not s_clean.empty else 50.0
                    filt_val = st.number_input("Value", value=round(mean_v, 2), key="filt_val_num")
                elif filt_op == "is in":
                    u_vals = df_raw[filt_col].dropna().unique().tolist()
                    filt_val = st.multiselect("Select Categories", u_vals[:50], key="filt_val_multi")
                else:
                    filt_val = st.text_input("Match Value", placeholder="e.g. Sales or Engineering", key="filt_val_text")

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

        # Filtered Dataset Result Section
        st.markdown('<div class="light-canvas-card">', unsafe_allow_html=True)
        st.markdown("### FILTERED DATASET")
        pct_left = round((len(df) / len(df_raw)) * 100, 1) if len(df_raw) > 0 else 0.0

        c_fm1, c_fm2, c_fm3 = st.columns(3)
        with c_fm1:
            st.metric("Rows Remaining", f"{len(df):,}")
        with c_fm2:
            st.metric("% Remaining", f"{pct_left}%")
        with c_fm3:
            st.metric("Columns", f"{len(df.columns)}")

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 4: DATA WORKBENCH
    # -------------------------------------------------------------------------
    elif active_view == "Data Workbench":
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="dark-glass-card">', unsafe_allow_html=True)
        st.markdown("### DATA WORKBENCH")
        st.markdown("<p>Inspect, sample, search, and export data records with high density.</p>", unsafe_allow_html=True)

        c_wb1, c_wb2, c_wb3, c_wb4 = st.columns(4)
        with c_wb1:
            st.metric("Rows", f"{len(df):,}")
        with c_wb2:
            st.metric("Columns", f"{len(df.columns)}")

        all_cols = df.columns.tolist()

        with c_wb3:
            visible_cols = st.multiselect("Visible Columns", options=all_cols, default=all_cols)
        with c_wb4:
            preset_rows = st.selectbox("Display Limit", ["25", "50", "100", "250", "All"], index=1)
            slice_n = len(df) if preset_rows == "All" else min(int(preset_rows), len(df))

        c_wbc1, c_wbc2, c_wbc3 = st.columns([1.5, 2.5, 1.2])
        with c_wbc1:
            view_mode = st.selectbox("Sampling Mode", ["First Rows", "Random Sample"], index=0)
        with c_wbc2:
            search_query = st.text_input("Quick In-Table Search", placeholder="Type keyword to filter current records...")
        with c_wbc3:
            st.write("")
            if visible_cols:
                csv_bytes = df[visible_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export CSV",
                    data=csv_bytes,
                    file_name=f"{st.session_state.dataset_name or 'dataset'}_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

        if visible_cols:
            display_df = df[visible_cols]
            if view_mode == "Random Sample" and len(display_df) > slice_n:
                display_slice = display_df.sample(n=slice_n, random_state=42)
            else:
                display_slice = display_df.head(slice_n)

            if search_query:
                mask = display_slice.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
                display_slice = display_slice[mask]

            st.markdown('<div class="light-canvas-card">', unsafe_allow_html=True)
            st.dataframe(display_slice, use_container_width=True)
            st.caption(f"Showing {len(display_slice):,} of {len(df):,} active rows ({len(visible_cols)} visible features).")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Select at least one column to display.")

    # -------------------------------------------------------------------------
    # SECTION 5: DATA QUALITY
    # -------------------------------------------------------------------------
    elif active_view == "Data Quality":
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        quality_rep = get_data_quality_report(df)

        # Top 4 Cards
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Quality Score", f"{quality_rep['quality_score']}/100")
        with q2:
            st.metric("Completeness", f"{quality_rep['completeness_pct']}%")
        with q3:
            st.metric("Missing Cells", f"{quality_rep['missing_cells']:,}")
        with q4:
            st.metric("Duplicate Rows", f"{quality_rep['duplicate_rows']:,}")

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

        # QUALITY DIAGNOSTICS: Two-Column Layout
        c_diag_left, c_diag_right = st.columns([1.5, 1])

        with c_diag_left:
            if quality_rep["missing_cells"] == 0:
                st.markdown(
                    """
                    <div class="dark-glass-card" style="text-align: center; padding: 40px 20px;">
                        <div style="font-size: 2.4rem; margin-bottom: 8px;">🎉</div>
                        <h3 style="color: #34D399 !important;">100% Data Completeness</h3>
                        <p style="color: #94A3B8; max-width: 450px; margin: 0 auto;">
                            Zero missing values detected across all columns. Dataset integrity is clean and ready for analysis.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="light-canvas-card">', unsafe_allow_html=True)
                st.markdown("### MISSINGNESS BY COLUMN")
                missing_summary = get_missing_values_summary(df)
                fig_miss = plot_missing_values_bar(missing_summary)
                st.plotly_chart(fig_miss, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with c_diag_right:
            st.markdown('<div class="dark-glass-card">', unsafe_allow_html=True)
            st.markdown("### ACTIONABLE FINDINGS")
            for sug in quality_rep["suggestions"]:
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; font-size: 0.85rem; color: #E2E8F0;">
                        {sug}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Full Column Integrity Audit
        st.markdown('<div class="light-canvas-card">', unsafe_allow_html=True)
        st.markdown("### FEATURE INTEGRITY AUDIT")
        st.dataframe(quality_rep["audit_df"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 6: AI COPILOT
    # -------------------------------------------------------------------------
    elif active_view == "AI Copilot":
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="dark-glass-card">', unsafe_allow_html=True)
        st.markdown("### AI COPILOT")
        st.markdown("<p>Ask questions about the active dataset. Evaluated locally with zero external API calls.</p>", unsafe_allow_html=True)

        # Quick Prompt Chips
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

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        with st.form("copilot_query_form"):
            user_input = st.text_input(
                "Your Dataset Question:",
                value=st.session_state.ai_question,
                placeholder="e.g. What is the average Salary? or Which columns have missing values?",
            )
            ask_submitted = st.form_submit_button("Ask Copilot", type="primary", use_container_width=False)

        if ask_submitted and user_input:
            st.session_state.ai_question = user_input
            st.session_state.last_ai_result = analyze_question(user_input, df)

        # Structured Copilot Output (Dark Glass Message Card)
        if st.session_state.last_ai_result:
            res = st.session_state.last_ai_result
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 14px; padding: 20px 22px; margin-top: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #818CF8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">DIRECT ANSWER</div>
                    <div style="font-size: 1.05rem; font-weight: 500; color: #F1F5F9; line-height: 1.6; margin-bottom: 14px;">{res['answer']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if res.get("evidence_cols"):
                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                st.markdown("##### 📌 Evidence & Features Evaluated")
                ev_html = "".join([f'<span class="filter-chip">📊 {c}</span>' for c in res["evidence_cols"]])
                st.markdown(ev_html, unsafe_allow_html=True)

            if res.get("insights"):
                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                st.markdown("##### 💡 Analytical Insights")
                for ins in res["insights"]:
                    st.markdown(f"- {ins}")

            if res.get("suggested_questions"):
                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                st.markdown("##### ❓ Suggested Next Inquiries")
                for sq in res["suggested_questions"]:
                    st.markdown(f"- `{sq}`")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
