"""
AI-Powered CSV Data Explorer
A modern, beginner-friendly Streamlit web application for exploring CSV datasets,
viewing descriptive statistics, detecting missing values, creating interactive charts,
and querying the dataset using a local mock AI assistant.
"""

from typing import Optional
import os
import pandas as pd
import streamlit as st

from src.data_processor import (
    load_csv,
    get_dataset_summary,
    get_numeric_stats,
    get_categorical_stats,
    get_missing_values_summary,
)
from src.visualizations import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_scatter,
    plot_categorical_counts,
    plot_missing_values_bar,
)
from src.mock_ai_engine import analyze_question

# Page Configuration
st.set_page_config(
    page_title="AI-Powered CSV Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark UI Styling
CUSTOM_CSS = """
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1A1E29;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #9CA3AF !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #F3F4F6 !important;
    }
    /* Section card */
    .section-card {
        background-color: #1A1E29;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    /* AI response box */
    .ai-answer-card {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.12) 0%, rgba(30, 27, 75, 0.25) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        background-color: #312E81;
        color: #C7D2FE;
        border: 1px solid #4338CA;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_sample_dataset_path() -> str:
    """Return the absolute path to the bundled sample dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data", "sample_employees.csv")


def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None
    if "ai_question" not in st.session_state:
        st.session_state.ai_question = ""
    if "last_ai_result" not in st.session_state:
        st.session_state.last_ai_result = None


def load_dataset(file_or_path, name: str):
    """Load dataset into session state."""
    try:
        df = load_csv(file_or_path)
        st.session_state.df = df
        st.session_state.dataset_name = name
        st.session_state.last_ai_result = None
        st.success(f"Successfully loaded **{name}** ({len(df):,} rows, {len(df.columns)} columns)!")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def main():
    init_session_state()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Data Source")
        st.caption("Upload your dataset or start with a sample file.")

        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Drag and drop or browse to upload any CSV file.",
        )

        if uploaded_file is not None:
            # Check if this is a newly uploaded file
            if st.session_state.dataset_name != uploaded_file.name:
                load_dataset(uploaded_file, uploaded_file.name)

        st.markdown("---")
        st.subheader("💡 Quick Start")
        sample_path = get_sample_dataset_path()
        
        if st.button("📁 Load Sample Dataset", use_container_width=True):
            if os.path.exists(sample_path):
                load_dataset(sample_path, "sample_employees.csv")
            else:
                st.error("Sample dataset file not found.")

        if st.session_state.df is not None:
            st.markdown("---")
            st.subheader("📌 Active Dataset")
            st.info(f"**File:** `{st.session_state.dataset_name}`\n\n"
                    f"**Rows:** {len(st.session_state.df):,}\n\n"
                    f"**Columns:** {len(st.session_state.df.columns)}")
            
            if st.button("🗑️ Clear Dataset", use_container_width=True):
                st.session_state.df = None
                st.session_state.dataset_name = None
                st.session_state.last_ai_result = None
                st.rerun()

        st.markdown("---")
        st.caption("AI-Powered CSV Data Explorer | Dark UI Mode")

    # --- MAIN CONTENT ---
    st.title("📊 AI-Powered CSV Data Explorer")
    st.write(
        "Explore your datasets interactively: view metadata, examine descriptive statistics, "
        "detect missing data, generate dark-themed charts, and query the data with a mock AI assistant."
    )

    df: Optional[pd.DataFrame] = st.session_state.df

    # If no dataset is loaded, show onboarding card
    if df is None:
        st.markdown("""
        <div class="section-card" style="text-align: center; padding: 40px 20px;">
            <h2 style="color: #818CF8; margin-bottom: 10px;">Welcome to CSV Data Explorer</h2>
            <p style="color: #9CA3AF; max-width: 600px; margin: 0 auto 25px auto;">
                Get started by uploading your own CSV file in the left sidebar, or click the button below to load our pre-built employee dataset with sample metrics and intentional missing values.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
        with col_c2:
            if st.button("🚀 Explore with Sample Dataset", use_container_width=True, type="primary"):
                sample_path = get_sample_dataset_path()
                if os.path.exists(sample_path):
                    load_dataset(sample_path, "sample_employees.csv")
                    st.rerun()
                else:
                    st.error("Sample dataset not found.")
        return

    # Dataset Summary Metrics Row
    summary = get_dataset_summary(df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", f"{summary['num_rows']:,}")
    with col2:
        st.metric("Total Columns", f"{summary['num_cols']}")
    with col3:
        st.metric("Missing Cells", f"{summary['total_missing_cells']:,}", f"{summary['missing_percentage']}%")
    with col4:
        st.metric("Memory Usage", f"{summary['memory_usage_kb']} KB")
    with col5:
        completeness = round(100 - summary['missing_percentage'], 1)
        st.metric("Completeness", f"{completeness}%")

    st.write("")

    # Navigation Tabs
    tab_preview, tab_stats, tab_missing, tab_charts, tab_ai = st.tabs([
        "📋 Dataset Preview",
        "📊 Descriptive Statistics",
        "🔍 Missing Values",
        "📈 Visualizations",
        "🤖 AI Assistant (Mock)",
    ])

    # ----------------------------------------------------
    # TAB 1: DATASET PREVIEW & SCHEMA
    # ----------------------------------------------------
    with tab_preview:
        st.subheader("Data Preview")
        total_rows = len(df)
        if total_rows <= 5:
            preview_rows = total_rows
        else:
            preview_rows = st.slider(
                "Rows to preview",
                min_value=5,
                max_value=min(100, total_rows),
                value=min(10, total_rows),
                step=5 if total_rows >= 10 else 1,
            )
        st.dataframe(df.head(preview_rows), use_container_width=True)

        st.subheader("Column Schema & Data Types")
        schema_cols = st.columns(2)
        
        with schema_cols[0]:
            st.write(f"**Numerical Columns ({len(summary['numeric_columns'])}):**")
            if summary['numeric_columns']:
                badges_html = "".join([f'<span class="badge">🔢 {col}</span>' for col in summary['numeric_columns']])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.caption("No numeric columns detected.")

        with schema_cols[1]:
            st.write(f"**Categorical / Text Columns ({len(summary['categorical_columns'])}):**")
            if summary['categorical_columns']:
                badges_html = "".join([f'<span class="badge">🔤 {col}</span>' for col in summary['categorical_columns']])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.caption("No categorical columns detected.")

        with st.expander("🔍 View Full Column Data Types Table"):
            type_df = pd.DataFrame(list(summary["column_types"].items()), columns=["Column Name", "Data Type"])
            st.dataframe(type_df, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: DESCRIPTIVE STATISTICS
    # ----------------------------------------------------
    with tab_stats:
        st.subheader("Numeric Columns Statistics")
        numeric_stats_df = get_numeric_stats(df)
        if not numeric_stats_df.empty:
            st.dataframe(numeric_stats_df, use_container_width=True)
        else:
            st.info("No numerical columns found in this dataset.")

        st.markdown("---")
        st.subheader("Categorical Columns Statistics")
        cat_stats_df = get_categorical_stats(df)
        if not cat_stats_df.empty:
            st.dataframe(cat_stats_df, use_container_width=True)
        else:
            st.info("No categorical columns found in this dataset.")

    # ----------------------------------------------------
    # TAB 3: MISSING VALUES
    # ----------------------------------------------------
    with tab_missing:
        st.subheader("Missing Value Audit")
        missing_df = get_missing_values_summary(df)

        if summary["total_missing_cells"] == 0:
            st.success("🎉 No missing values detected! Your dataset is 100% complete.")
        else:
            st.warning(f"⚠️ Found {summary['total_missing_cells']} missing values across {len(missing_df[missing_df['Missing Count'] > 0])} column(s).")

        col_m1, col_m2 = st.columns([3, 2])
        with col_m1:
            st.markdown("##### Missing Values Breakdown by Column")
            st.dataframe(missing_df, use_container_width=True)

        with col_m2:
            st.markdown("##### Missingness Percentage Chart")
            fig_missing = plot_missing_values_bar(missing_df)
            st.plotly_chart(fig_missing, use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: VISUALIZATIONS
    # ----------------------------------------------------
    with tab_charts:
        st.subheader("Interactive Data Visualizations")
        chart_mode = st.selectbox(
            "Select Exploration Chart",
            [
                "1. Distribution & Outlier Analysis (Histogram / Box Plot)",
                "2. Correlation Heatmap",
                "3. 2D Scatter Plot & Relationships",
                "4. Categorical Breakdown (Bar / Donut)",
            ],
        )

        if "1. Distribution" in chart_mode:
            if summary["numeric_columns"]:
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    num_col = st.selectbox("Select Numerical Column", summary["numeric_columns"])
                with c2:
                    p_type = st.selectbox("Plot Type", ["Histogram", "Box Plot", "Violin Plot"])
                with c3:
                    bins = st.slider("Bins (Histogram only)", 5, 50, 20) if p_type == "Histogram" else 20
                
                fig = plot_distribution(df, num_col, plot_type=p_type, bins=bins)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for distribution plots.")

        elif "2. Correlation Heatmap" in chart_mode:
            if len(summary["numeric_columns"]) >= 2:
                selected_num = st.multiselect(
                    "Select Numeric Columns for Correlation Matrix",
                    options=summary["numeric_columns"],
                    default=summary["numeric_columns"][: min(8, len(summary["numeric_columns"]))],
                )
                if len(selected_num) >= 2:
                    fig = plot_correlation_heatmap(df, selected_num)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least 2 numerical columns.")
            else:
                st.warning("At least 2 numeric columns are required to calculate correlations.")

        elif "3. 2D Scatter Plot" in chart_mode:
            if len(summary["numeric_columns"]) >= 2:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    x_axis = st.selectbox("X Axis (Numeric)", summary["numeric_columns"], index=0)
                with c2:
                    y_axis_idx = 1 if len(summary["numeric_columns"]) > 1 else 0
                    y_axis = st.selectbox("Y Axis (Numeric)", summary["numeric_columns"], index=y_axis_idx)
                with c3:
                    color_options = ["None"] + summary["categorical_columns"] + summary["numeric_columns"]
                    color_by = st.selectbox("Color By (Optional)", color_options, index=0)
                with c4:
                    trend = st.checkbox("Add Trendline", value=False)

                fig = plot_scatter(df, x_axis, y_axis, color_col=color_by if color_by != "None" else None, add_trendline=trend)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("At least 2 numeric columns are required for a scatter plot.")

        elif "4. Categorical Breakdown" in chart_mode:
            if summary["categorical_columns"]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    cat_col = st.selectbox("Select Categorical Column", summary["categorical_columns"])
                with c2:
                    c_type = st.selectbox("Chart Type", ["Bar", "Donut"])
                with c3:
                    top_n = st.slider("Top N Categories", 3, 20, 10)

                fig = plot_categorical_counts(df, cat_col, top_n=top_n, chart_type=c_type)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns available for categorical breakdown.")

    # ----------------------------------------------------
    # TAB 5: NATURAL LANGUAGE AI ASSISTANT (MOCK)
    # ----------------------------------------------------
    with tab_ai:
        st.subheader("🤖 Ask Questions About Your Dataset")
        st.caption("Type a question about rows, averages, maximums, missing values, or correlations. The local analyzer generates answers and insights instantly without requiring an API key.")

        # Quick Suggestion Chips
        st.write("**Quick Prompts:**")
        chip_cols = st.columns(4)
        
        sample_q1 = f"What is the average {summary['numeric_columns'][0]}?" if summary['numeric_columns'] else "How many rows are in the dataset?"
        sample_q2 = "Which column has missing values?"
        sample_q3 = "What is the dataset overview?"
        sample_q4 = f"What is the highest {summary['numeric_columns'][0]}?" if summary['numeric_columns'] else "List all columns"

        def set_question(q_text):
            st.session_state.ai_question = q_text

        with chip_cols[0]:
            if st.button(f"💡 {sample_q1}", key="btn_q1", use_container_width=True):
                set_question(sample_q1)
        with chip_cols[1]:
            if st.button(f"🔍 {sample_q2}", key="btn_q2", use_container_width=True):
                set_question(sample_q2)
        with chip_cols[2]:
            if st.button(f"📋 {sample_q3}", key="btn_q3", use_container_width=True):
                set_question(sample_q3)
        with chip_cols[3]:
            if st.button(f"📈 {sample_q4}", key="btn_q4", use_container_width=True):
                set_question(sample_q4)

        # Question input form
        with st.form("ai_question_form"):
            user_query = st.text_input(
                "Your Question:",
                value=st.session_state.ai_question,
                placeholder="e.g. What is the average Salary? or Which column has missing values?",
            )
            submitted = st.form_submit_button("Ask Assistant", type="primary", use_container_width=False)

        if submitted and user_query:
            st.session_state.ai_question = user_query
            st.session_state.last_ai_result = analyze_question(user_query, df)

        # Display AI Result if present
        if st.session_state.last_ai_result:
            res = st.session_state.last_ai_result
            
            st.markdown("### 💬 Assistant Response")
            
            # Answer Box
            st.markdown(
                f"""
                <div class="ai-answer-card">
                    <h4 style="color: #818CF8; margin-top: 0;">Direct Answer</h4>
                    <p style="font-size: 1.05rem; color: #F3F4F6; margin-bottom: 0;">{res['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Key Insights
            if res.get("insights"):
                st.markdown("#### 💡 Key Data Insights")
                for item in res["insights"]:
                    st.markdown(f"- {item}")

            # Suggested Follow-Up Questions
            if res.get("suggested_questions"):
                st.markdown("#### ❓ Suggested Follow-Up Questions")
                for sq in res["suggested_questions"]:
                    st.markdown(f"- `{sq}`")


if __name__ == "__main__":
    main()
