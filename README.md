# InsightFlow | AI-Powered CSV Data Intelligence Workspace

An interactive, production-grade exploratory data analytics platform built with Python and Streamlit, featuring dark shell chrome, light analytical canvases, a context-aware chart studio, multi-clause filter engine, data quality diagnostics, and a private offline heuristic AI copilot.

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://insightflow-csv-explorer.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rudrasharma26/insightflow)
[![Tests Passing](https://img.shields.io/badge/Tests-32%20Passed-10B981?style=for-the-badge&logo=pytest&logoColor=white)](#testing)

---

## 📌 Project Links
- **Live Application:** [https://insightflow-csv-explorer.streamlit.app/](https://insightflow-csv-explorer.streamlit.app/)
- **GitHub Repository:** [https://github.com/rudrasharma26/insightflow](https://github.com/rudrasharma26/insightflow)

---

## 📸 Workspace Preview

<!-- SCREENSHOT PLACEHOLDER: Add workspace preview images here -->
```
+---------------------------------------------------------------------------------------+
|  INSIGHTFLOW | DATA INTELLIGENCE WORKSPACE                                            |
|  [Overview]  [Chart Studio]  [Filter Lab]  [Data Workbench]  [Data Quality]  [Copilot] |
+---------------------------------------------------------------------------------------+
|  [KPIs: 500 Rows | 12 Features | 98.5% Complete | 6 Numeric | 6 Categorical | 48 KB]  |
|                                                                                       |
|  +-- Automatic Insights ----------+  +-- Light Canvas Interactive Visual ------------+|
|  | - Dominant Segment Detected    |  |  (Plotly Light Chart Canvas with Crisp Grid)   ||
|  | - Significant r=0.82 Pearson   |  |                                               ||
|  | - Missingness Concentration    |  |  [Points / Box Selection -> Cross-Filter Table]||
|  +--------------------------------+  +-----------------------------------------------+|
+---------------------------------------------------------------------------------------+
```

---

## 🚀 Key Capabilities

### 1. 📊 Executive Overview & Auto-Insights
- **KPI Metrics:** Immediate visibility into total rows, columns, completeness score, numeric fields, categorical features, and memory footprint.
- **Smart Insights Engine:** Automated statistical heuristics detect leading category dominance, high-variance metrics, feature correlations, and missingness concentrations.
- **Flagship Visual:** Dynamically selects the most informative chart type based on the loaded dataset's schema.

### 2. 📈 Context-Aware Chart Studio
- **Dynamic Visual Selector:** Line, Bar, Area, Scatter, Histogram, Box Plot, Violin Plot, Correlation Heatmap, and Donut Breakdown.
- **Context-Aware Controls:** Inputs adapt dynamically to only display valid columns and relevant aggregation options for the selected chart.
- **Advanced Popover Controls:** Secondary parameters (trendlines, opacity, sorting order, orientation, marginal plots, correlation methods) stay organized without visual clutter.
- **Interactive Cross-Filter Selection:** Box, lasso, and point selections on Plotly figures link directly to an inspection drawer for slicing data subsets.

### 3. 🎛️ Multi-Clause Filter Engine
- **Type-Aware Operators:** Supports comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=`, `between`), text matching (`contains`, `equals`, `does not equal`, `is in`), and null checks (`is null`, `is not null`).
- **Removable Filter Badges:** Active filter chips with individual removal and one-click global reset.
- **Global Propagation:** Filtered subsets automatically drive the Chart Studio, Data Workbench, Quality Audit, and Copilot context.

### 4. 📋 High-Performance Data Workbench
- **Display Presets:** Quick row toggles (`25`, `50`, `100`, `250`, `All`, or custom slider).
- **Sampling Modes:** Toggle between chronological first rows and reproducible random samples.
- **Column Visibility:** Interactive multi-select for customizing visible columns.
- **In-Table Search:** Instant multi-column keyword filtering.
- **One-Click Export:** Download active filtered slices as standard CSV files.

### 5. 🛡️ Data Quality Diagnostics
- **Comprehensive Quality Score:** Computed health score (0–100) based on completeness, duplicate rows, and single-value columns.
- **Feature-Level Audit:** Column-by-column breakdown of populated rows, missing percentage, unique values, cardinality ratio, and data integrity status.
- **Actionable Suggestions:** Clear heuristics identifying data cleaning priorities (imputation, deduplication, constant-column removal).

### 6. 🤖 Dataset AI Copilot (100% Offline & Private)
- **Local Heuristic NLP Engine:** Interprets natural-language queries regarding averages, extremes, correlations, missing data, and distributions.
- **Zero API Keys Required:** Runs entirely in-process with pandas/numpy algorithms; no paid cloud APIs or internet connection required.
- **Structured Copilot Outputs:**
  - Direct, clear factual answer.
  - Evidence & features evaluated.
  - Data-driven analytical takeaways.
  - Dynamic follow-up inquiry suggestions.

---

## 🛠️ Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| **Core Runtime** | Python 3.9+ | Standard analytical language ecosystem |
| **Interface Framework** | Streamlit (>=1.30.0) | Reactive analytical web workspace |
| **Data Processing** | Pandas (>=2.0.0), NumPy (>=1.24.0) | High-performance tabular computations |
| **Interactive Visualizations** | Plotly (>=5.18.0) | Light canvas, customizable SVG/WebGL charts |
| **Automated Testing** | Pytest (>=7.0.0) | Comprehensive unit and edge-case test coverage |

---

## 🏛️ Project Architecture

```text
insightflow/
├── .streamlit/
│   └── config.toml             # Dark chrome shell theme & server settings
├── data/
│   └── sample_employees.csv    # Bundled sample dataset with mixed types
├── src/
│   ├── __init__.py
│   ├── data_processor.py       # Summaries, stats, filter engine, quality diagnostics
│   ├── mock_ai_engine.py       # Offline heuristic NLP assistant & reasoning engine
│   └── visualizations.py       # Plotly chart builders with light analytical canvas
├── tests/
│   ├── __init__.py
│   ├── test_data_processor.py  # Unit tests for core data calculations
│   ├── test_mock_ai_engine.py  # Unit tests for natural language query handler
│   ├── test_visualizations.py  # Unit tests for chart generation functions
│   ├── test_new_features.py    # Tests for filter engine, quality score, & studio charts
│   └── test_edge_cases.py      # Tests for single-row, numeric-only, & messy CSVs
├── app.py                      # Production Streamlit application & layout
├── requirements.txt            # Compatible minimum dependencies
├── .gitignore
└── README.md                   # Project documentation & reference
```

---

## 🧪 Testing

The test suite validates data loading, calculations, edge cases, chart generation, and query handling across **32 automated tests**:

```bash
python -m pytest -v
```

Current test status:
```text
tests/test_data_processor.py::test_load_csv_from_stringio PASSED
tests/test_data_processor.py::test_get_dataset_summary PASSED
tests/test_data_processor.py::test_get_dataset_summary_empty PASSED
tests/test_data_processor.py::test_get_numeric_stats PASSED
tests/test_data_processor.py::test_get_categorical_stats PASSED
tests/test_data_processor.py::test_get_missing_values_summary PASSED
tests/test_data_processor.py::test_small_datasets_1_and_3_rows PASSED
tests/test_edge_cases.py::test_numeric_only_dataset PASSED
tests/test_edge_cases.py::test_categorical_heavy_dataset PASSED
tests/test_edge_cases.py::test_missing_heavy_and_constant_dataset PASSED
tests/test_edge_cases.py::test_tiny_single_row_dataset PASSED
tests/test_mock_ai_engine.py::test_analyze_question_structure PASSED
tests/test_mock_ai_engine.py::test_analyze_empty_dataframe PASSED
tests/test_mock_ai_engine.py::test_analyze_row_count_query PASSED
tests/test_mock_ai_engine.py::test_analyze_average_query PASSED
tests/test_mock_ai_engine.py::test_analyze_max_query PASSED
tests/test_mock_ai_engine.py::test_analyze_missing_query PASSED
tests/test_mock_ai_engine.py::test_analyze_categorical_query PASSED
tests/test_mock_ai_engine.py::test_analyze_small_datasets PASSED
tests/test_new_features.py::test_apply_filters_numeric PASSED
tests/test_new_features.py::test_apply_filters_categorical PASSED
tests/test_new_features.py::test_apply_filters_combined PASSED
tests/test_new_features.py::test_get_data_quality_report PASSED
tests/test_new_features.py::test_detect_smart_insights PASSED
tests/test_new_features.py::test_detect_column_types_extended PASSED
tests/test_new_features.py::test_new_chart_studio_visualizations PASSED
tests/test_visualizations.py::test_plot_distribution PASSED
tests/test_visualizations.py::test_plot_correlation_heatmap PASSED
tests/test_visualizations.py::test_plot_scatter PASSED
tests/test_visualizations.py::test_plot_categorical_counts PASSED
tests/test_visualizations.py::test_plot_missing_values_bar PASSED
tests/test_visualizations.py::test_visualizations_small_dataframe PASSED
============================== 32 passed in 2.81s ==============================
```

---

## 💻 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/rudrasharma26/insightflow.git
cd insightflow
```

### 2. Create and Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
python -m streamlit run app.py
```
Access the application at `http://localhost:8501`.

---

## ☁️ Deployment

The application is deployed to **Streamlit Community Cloud** directly from the `main` branch:
1. Connect GitHub repository to Streamlit Cloud.
2. Set the main file path to `app.py`.
3. Dependencies are resolved automatically from `requirements.txt`.
4. Theme settings are loaded from `.streamlit/config.toml`.

Live link: [https://insightflow-csv-explorer.streamlit.app/](https://insightflow-csv-explorer.streamlit.app/)

---

## 🔮 Future Enhancements
- Optional toggle for external LLM API providers (e.g., Anthropic Claude / Google Gemini) alongside the default offline heuristic engine.
- Automated hypothesis testing (t-tests, ANOVA, chi-square) in the Data Quality center.
- Support for Parquet and JSON ingestion.
- One-click PDF / Markdown automated exploratory data report generation.
