# Spending Segment Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/jdkgcprg6k/badge.svg)](https://ask.thili.ai/b/jdkgcprg6k)

This project creates a Spending Segment Analyzer, a Streamlit application designed for financial planners. It helps them track client expenses, identify spending habits through segmentation, and provide better budgeting advice.

## What you are building

**Goal.** Build a Streamlit app for financial planners to track client expenses, segment spending habits, and improve budgeting.

**Done when.** The application successfully tracks and saves client transactions, loads them for analysis, clusters spending patterns using K-Means, and displays a budget summary with identified spending segments.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Implement transaction tracking and saving to JSON.**  
      Hint: Use the object-oriented class hierarchy and JSON persistence from the budget drill.
- [ ] **2. Add functionality to load transactions and preprocess for RFM features.**  
      Hint: Load the transaction data and engineer RFM features as demonstrated in the customer segmentation drill.
- [ ] **3. Implement K-Means from scratch to cluster spending patterns.**  
      Hint: Implement the K-Means clustering algorithm from scratch, as covered in the customer segmentation drill.
- [ ] **4. Display budget summary and identified spending segments in Streamlit.**  
      Hint: Integrate the budget summary and identified segments into a user-friendly Streamlit interface.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow financial planners to edit or delete existing transactions.
- Implement a feature to visualize spending trends over different time periods.
