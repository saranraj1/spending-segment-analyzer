"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import datetime as dt
from decimal import Decimal
import pytest

from app.core import (
    Account,
    InvalidTransactionError,
    RecurringTransaction,
    Transaction,
    load,
    milestone_1,
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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Implement K-Means from scratch to cluster spending patterns.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Display budget summary and identified spending segments in Streamlit.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

