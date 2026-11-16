"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import datetime as dt
from decimal import Decimal
import pandas as pd
import pytest

from app.core import (
    Account,
    InvalidTransactionError,
    KMeansScratch,
    RecurringTransaction,
    Transaction,
    calculate_rfm,
    delete_transaction,
    generate_planner_report,
    get_budget_summary,
    get_spending_trends,
    load,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    save,
)

# Milestone 1: Implement transaction tracking and saving to JSON.
def test_milestone_1(tmp_path):
    store_file = tmp_path / "test_budget.json"

    # 1. Verify default milestone_1 execution and JSON generation
    account = milestone_1(path=store_file)
    assert len(account.transactions) > 0
    assert store_file.exists()

    # 2. Verify JSON loading matches saved data
    result = load(store_file)
    assert len(result.account.transactions) == len(account.transactions)
    assert result.account.balance == account.balance

    # 3. Test OOP Transaction validation
    valid_txn = Transaction(dt.date(2026, 3, 1), Decimal("-45.50"), "dining", "Lunch")
    assert valid_txn.kind == "expense"
    account.add(valid_txn)
    assert valid_txn in account.transactions

    # 4. Test zero amount raises validation error
    with pytest.raises(InvalidTransactionError):
        Transaction(dt.date(2026, 3, 1), Decimal("0"), "dining")

    # 5. Test RecurringTransaction occurrences
    recurring = RecurringTransaction(dt.date(2026, 1, 1), Decimal("-100"), "gym", frequency="monthly")
    occs = recurring.occurrences(until=dt.date(2026, 3, 1))
    assert len(occs) == 3


# Milestone 2: Add functionality to load transactions and preprocess for RFM features.
def test_milestone_2(tmp_path):
    store_file = tmp_path / "test_rfm_budget.json"
    milestone_1(path=store_file)

    # 1. Run milestone_2 on the saved file
    df_rfm = milestone_2(path=store_file)

    # 2. Assert RFM dataframe structure and schema
    assert isinstance(df_rfm, pd.DataFrame)
    assert not df_rfm.empty
    expected_cols = {"category", "recency", "frequency", "monetary", "recency_scaled", "frequency_scaled", "monetary_scaled"}
    assert expected_cols.issubset(set(df_rfm.columns))

    # 3. Assert values are logically bounded
    assert (df_rfm["recency"] >= 0).all()
    assert (df_rfm["frequency"] >= 1).all()
    assert (df_rfm["monetary"] > 0).all()

    # 4. Verify specific known aggregated metrics from default dataset
    housing_row = df_rfm[df_rfm["category"] == "housing"]
    assert not housing_row.empty
    assert housing_row["frequency"].iloc[0] == 2
    assert housing_row["monetary"].iloc[0] == 3000.00


# Milestone 3: Implement K-Means from scratch to cluster spending patterns.
def test_milestone_3(tmp_path):
    store_file = tmp_path / "test_cluster_budget.json"
    milestone_1(path=store_file)
    df_rfm = milestone_2(path=store_file)

    # 1. Run milestone_3 clustering from scratch
    model, clustered_df = milestone_3(df_rfm, k=3, random_state=42)

    # 2. Check model fit and properties
    assert isinstance(model, KMeansScratch)
    assert model.centroids is not None
    assert len(model.centroids) <= 3
    assert model.inertia_ >= 0.0
    assert model.n_iter_ > 0

    # 3. Check clustered dataframe schema and contents
    assert "cluster" in clustered_df.columns
    assert "segment_label" in clustered_df.columns
    assert set(clustered_df["cluster"].unique()).issubset({0, 1, 2})
    assert all(isinstance(lbl, str) and len(lbl) > 0 for lbl in clustered_df["segment_label"])

    # 4. Mathematical check on KMeansScratch with known synthetic clusters
    import numpy as np
    synthetic_x = np.array([[1.0, 1.0], [1.1, 1.2], [10.0, 10.0], [10.2, 9.8]])
    toy_kmeans = KMeansScratch(k=2, random_state=42).fit(synthetic_x)
    toy_labels = toy_kmeans.labels_
    assert toy_labels[0] == toy_labels[1]
    assert toy_labels[2] == toy_labels[3]
    assert toy_labels[0] != toy_labels[2]


# Milestone 4: Display budget summary and identified spending segments in Streamlit.
def test_milestone_4(tmp_path):
    store_file = tmp_path / "test_summary_budget.json"
    account = milestone_1(path=store_file)

    # 1. Run milestone_4 integration
    summary = milestone_4(account=account, path=store_file, k=3)

    # 2. Check contract keys required by Streamlit
    assert "account_name" in summary
    assert "balance" in summary
    assert "total_income" in summary
    assert "total_expenses" in summary
    assert "category_summaries" in summary
    assert "segments_df" in summary
    assert "spending_trends" in summary

    # 3. Check category breakdown calculations
    cat_items = summary["category_summaries"]
    assert len(cat_items) > 0
    dining_summary = next(c for c in cat_items if c["category"] == "dining")
    assert dining_summary["spent"] > 0
    assert dining_summary["budget"] == 250.00
    assert "segment_label" in dining_summary

    # 4. Check over budget logic
    # Add an expense with enforce_limits=False (simulating an actual statement overrun)
    over_txn = Transaction(dt.date(2026, 3, 5), Decimal("-500.00"), "dining", "Expensive celebration")
    account.add_many([over_txn], enforce_limits=False)
    updated_summary = milestone_4(account=account, path=store_file, k=3)
    updated_dining = next(c for c in updated_summary["category_summaries"] if c["category"] == "dining")
    assert updated_dining["over_budget"] is True
    assert updated_summary["over_budget_count"] >= 1


# Stretch features test
def test_stretch_features(tmp_path):
    store_file = tmp_path / "test_stretch.json"
    account = milestone_1(path=store_file)
    initial_count = len(account.transactions)
    initial_balance = account.balance

    # Test Stretch 1: Delete transaction
    first_txn_amt = account.transactions[0].amount
    deleted = delete_transaction(account, 0)
    assert deleted is True
    assert len(account.transactions) == initial_count - 1
    assert account.balance == initial_balance - first_txn_amt

    # Test Stretch 2: Spending trends timeline
    df_trends = get_spending_trends(account)
    assert isinstance(df_trends, pd.DataFrame)
    assert not df_trends.empty
    assert "cumulative_spend" in df_trends.columns
    assert (df_trends["cumulative_spend"] > 0).all()


def test_edge_cases_and_empty_scenarios():
    # 1. Empty account handling
    empty_acc = Account("Empty Test")
    assert empty_acc.balance == Decimal(0)
    assert len(empty_acc.transactions) == 0

    # 2. RFM preprocessing on empty account
    empty_rfm = calculate_rfm(empty_acc)
    assert isinstance(empty_rfm, pd.DataFrame)
    assert empty_rfm.empty
    assert "monetary" in empty_rfm.columns

    # 3. K-Means clustering on empty DataFrame
    model, clustered_empty = milestone_3(df_rfm=empty_rfm, k=3)
    assert isinstance(model, KMeansScratch)
    assert "cluster" in clustered_empty.columns
    assert "segment_label" in clustered_empty.columns

    # 4. Milestone 4 summary on empty account
    summary = milestone_4(account=empty_acc)
    assert summary["balance"] == 0.0
    assert summary["total_expenses"] == 0.0
    assert summary["transaction_count"] == 0
    assert summary["over_budget_count"] == 0

    # 5. Empty spending trends
    empty_trends = get_spending_trends(empty_acc)
    assert empty_trends.empty

    # 6. Delete transaction invalid index handling
    assert delete_transaction(empty_acc, 99) is False
    assert delete_transaction(empty_acc, -1) is False


def test_kmeans_plus_plus_and_report_generation():
    # 1. Verify KMeans++ initialization
    import numpy as np
    data = np.array([[1.0, 1.0], [1.1, 1.2], [50.0, 50.0], [51.0, 49.5]])
    km_pp = KMeansScratch(k=2, init="k-means++", random_state=42).fit(data)
    assert km_pp.labels_[0] == km_pp.labels_[1]
    assert km_pp.labels_[2] == km_pp.labels_[3]
    assert km_pp.labels_[0] != km_pp.labels_[2]

    # 2. Verify planner memo report generation
    acc = milestone_1()
    summary = milestone_4(account=acc, k=3)
    memo = generate_planner_report(acc, summary)
    assert isinstance(memo, str)
    assert "Executive Cashflow Overview" in memo
    assert "Category Budget Utilization" in memo
    assert "Spending Habit Archetypes" in memo
    assert f"${summary['balance']:,.2f}" in memo

