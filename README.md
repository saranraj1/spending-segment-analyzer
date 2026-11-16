# Spending Segment Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/jdkgcprg6k/badge.svg)](https://ask.thili.ai/b/jdkgcprg6k)

A Streamlit application for financial planners to record client transactions, analyze spending patterns using RFM feature engineering, and segment habits using K-Means clustering implemented from scratch.

## Project Scope

**Goal**: Build a Streamlit app for financial planners to track client expenses, segment spending habits, and improve budgeting.

**Done when**: The application successfully tracks and saves client transactions, loads them for analysis, clusters spending patterns using K-Means, and displays a budget summary with identified spending segments.

## Milestones

- [x] **1. Implement transaction tracking and saving to JSON.**
      Hint: Use the object-oriented class hierarchy and JSON persistence from the budget drill.
- [x] **2. Add functionality to load transactions and preprocess for RFM features.**
      Hint: Load the transaction data and engineer RFM features as demonstrated in the customer segmentation drill.
- [x] **3. Implement K-Means from scratch to cluster spending patterns.**
      Hint: Implement the K-Means clustering algorithm from scratch, as covered in the customer segmentation drill.
- [x] **4. Display budget summary and identified spending segments in Streamlit.**
      Hint: Integrate the budget summary and identified segments into a user-friendly Streamlit interface.

## Stretch Features

- [x] Transaction ledger management with row deletion.
- [x] Cumulative and chronological spending trends over time.

## Overview of Implementation

The codebase is split into core financial logic and presentation:

1. **Transaction Domain Model and Storage (`app/core.py`)**
   - Implements `Transaction`, `RecurringTransaction`, and `Account` classes.
   - Handles input validation (dates, non-empty categories, signed amounts).
   - Uses atomic file writes to protect data persistence.

2. **RFM Feature Engineering (`app/core.py`)**
   - Aggregates expense records by category.
   - Computes Recency (days since last transaction relative to dataset reference date), Frequency (transaction count), and Monetary value (total spend).
   - Applies Z-score standardization across RFM dimensions.

3. **K-Means Clustering from Scratch (`app/core.py`)**
   - Pure NumPy implementation of the Expectation-Maximization clustering loop.
   - Includes K-Means++ initialization to choose initial centroid positions.
   - Calculates cluster inertia and tracks convergence against centroid movement thresholds.
   - Assigns spending habit personas based on cluster median metrics (for example, High Volume, Major Outflow, Frequent Micro-Spender, and Low Spend).

4. **Elbow Method and Counterfactual Rebalancer (`app/core.py`)**
   - `compute_elbow_curve`: Computes inertia across k values from 1 to 5 and uses secant chord distance to recommend an optimal cluster count.
   - `simulate_counterfactual_intervention`: Calculates changes in cluster assignment and persona when spending in a selected category is reduced. Computes projected savings and multi-year compounded future value using annuity formulas.

5. **Streamlit User Interface (`app/main.py`)**
   - KPI metrics for balance, inflows, outflows, and budget status.
   - Category budget utilization meters and warning indicators for overages.
   - Scatter visualization of clusters (Recency vs. Monetary spend, sized by Frequency).
   - What-if simulation controls for category spending adjustments.
   - Interactive transaction ledger with deletion support.
   - Export options for Markdown planner memos and CSV datasets.
   - Optional AI advisor integration using configured LLM endpoints.

Detailed derivations and justifications for all formulas are documented in [MATHEMATICAL_SPECIFICATIONS.md](MATHEMATICAL_SPECIFICATIONS.md).

## Project Structure

```
spending-segment-analyzer/
├── app/
│   ├── core.py         # Domain models, persistence, RFM, KMeansScratch, simulation
│   ├── main.py         # Streamlit dashboard interface
│   ├── llm.py          # LLM client configuration
│   └── storage.py      # Storage helpers
├── data/
│   └── app.json        # Default transaction data store
├── tests/
│   ├── test_core.py    # Milestone, edge case, and algorithmic unit tests
│   └── test_smoke.py   # Baseline verification tests
├── pytest.ini          # Pytest configuration
├── requirements.txt    # Project dependencies
├── MATHEMATICAL_SPECIFICATIONS.md # Detailed mathematical formulations and derivations
└── README.md
```

## Running the Project

### Requirements
- Python 3.11 or Python 3.12
- Git

### Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment settings (optional for LLM advisory):

```bash
cp .env.example .env
```

4. Run the Streamlit application:

```bash
streamlit run app/main.py
```

5. Run test suite:

```bash
pytest
```

## Settings and Secrets

Model settings live in `.env`, which is ignored by git. Do not commit `.env` or hardcode API keys into source files.
