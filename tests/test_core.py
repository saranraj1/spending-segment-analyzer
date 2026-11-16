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
    load,
    milestone_1,
    milestone_2,
    milestone_3,
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
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

