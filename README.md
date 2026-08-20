# 📊 AI-Powered CSV Data Explorer

An interactive, beginner-friendly CSV Data Explorer built with **Python**, **Streamlit**, **Pandas**, and **Plotly**. 

Upload any CSV dataset (or click one button to load the included sample dataset) to instantly inspect data schemas, descriptive statistics, missing value audits, dark-themed interactive charts, and natural-language dataset querying powered by a local heuristic AI assistant (no external API keys required).

---

## ✨ Features

- **📁 Instant CSV Upload & Sample Loader**: Drag-and-drop any `.csv` file or explore immediately with the built-in employee dataset.
- **📋 Dataset Preview & Schema**: View interactive tables, toggle preview row counts, and inspect column data types (numeric vs categorical).
- **📊 Descriptive Statistics**: Comprehensive statistical summaries for numerical columns (mean, std, min, 25%, 50%, 75%, max, median, missing counts) and categorical columns (unique counts, mode, top frequency).
- **🔍 Missing Value Detection**: Automated audit reporting missing counts, percentages, data types, and a missingness severity bar chart.
- **📈 Interactive Dark Visualizations**:
  - **Distribution Analysis**: Histograms, Box Plots, and Violin Plots with configurable bin counts.
  - **Correlation Heatmap**: Interactive Pearson correlation matrix with multi-column selector.
  - **2D Scatter Plots**: Feature relationship scatter with optional category color hues and trendlines.
  - **Categorical Breakdown**: Top N frequency Bar and Donut charts.
- **🤖 Natural Language AI Assistant (Mock)**: Ask questions about the dataset (e.g. *"What is the average salary?"*, *"Which column has missing values?"*, *"How many rows?"*) and receive structured answers, key data insights, and suggested follow-up questions without needing an API key.
- **🌙 Modern Dark UI**: Dark theme styling configured for high readability and clean presentation.

---

## 🗂️ Project Structure

```text
build-with-ai-prep/
├── .streamlit/
│   └── config.toml           # Streamlit dark theme settings
├── data/
│   └── sample_employees.csv  # Bundled sample dataset with mixed types & missing values
├── src/
│   ├── __init__.py
│   ├── data_processor.py     # CSV loading, overview metrics, statistics, and missing value logic
│   ├── visualizations.py     # Plotly dark-themed interactive chart builders
│   └── mock_ai_engine.py     # Local heuristic-based natural language Q&A engine
├── tests/
│   ├── __init__.py
│   ├── test_data_processor.py # Unit tests for data loading & calculations
│   └── test_mock_ai_engine.py # Unit tests for natural language query engine
├── app.py                    # Main Streamlit web application
├── requirements.txt          # Compatible minimum Python dependencies
└── README.md                 # Documentation & setup guide
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+** installed on your system.

### 2. Set Up a Virtual Environment (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required packages (compatible minimum versions):
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Application
Start the Streamlit development server:
```bash
streamlit run app.py
```
Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 🧪 Running Unit Tests

Run the test suite using `pytest`:
```bash
pytest -v
```

All unit tests in `tests/test_data_processor.py` and `tests/test_mock_ai_engine.py` will execute and validate the core calculation and AI query components.

---

## 💡 Example AI Questions to Try

Once you load `sample_employees.csv` (or your own CSV), try asking:
- `What is the average Salary?`
- `What is the highest Salary?`
- `Which column has missing values?`
- `How many rows are in the dataset?`
- `Tell me about Department`
- `What is the correlation between numeric columns?`
- `Give me a summary of this dataset`
