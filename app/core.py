"""Core business logic for Spending Segment Analyzer.

Builds on the budget drill object-oriented hierarchy, RFM feature engineering,
and pure NumPy K-Means clustering from scratch.
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Object-Oriented Class Hierarchy & Persistence (Budget Drill)
# ---------------------------------------------------------------------------

try:
    from thilibudget import (
        Account,
        BudgetError,
        BudgetExceededError,
        InsufficientFundsError,
        InvalidTransactionError,
        LoadResult,
        RecurringTransaction,
        Transaction,
        load,
        save,
    )
except ImportError:
    # Self-contained fallback identical to thilibudget drill
    class BudgetError(Exception):
        """Base class for budget drill exceptions."""

    class InvalidTransactionError(BudgetError, ValueError):
        """Raised when a transaction field fails validation."""

    class InsufficientFundsError(BudgetError):
        """Raised when an account overdraws without permission."""

    class BudgetExceededError(BudgetError):
        """Raised when spending exceeds monthly budget."""

    FREQUENCIES = ("weekly", "monthly")

    @dataclass(frozen=True)
    class Transaction:
        date: dt.date
        amount: Decimal
        category: str
        description: str = ""

        def __post_init__(self) -> None:
            if not isinstance(self.date, dt.date):
                raise InvalidTransactionError(f"date must be a date, got {self.date!r}")
            if not isinstance(self.amount, Decimal) or not self.amount.is_finite():
                raise InvalidTransactionError(f"amount must be a finite Decimal, got {self.amount!r}")
            if self.amount == 0:
                raise InvalidTransactionError("amount must not be zero")
            if not self.category or not self.category.strip():
                raise InvalidTransactionError("category must not be empty")

        @property
        def kind(self) -> str:
            return "income" if self.amount > 0 else "expense"

        def occurrences(self, until: dt.date | None = None) -> list[Transaction]:
            return [self]

        @classmethod
        def parse(cls, date: str, amount: str, category: str, description: str = "", **extra) -> Transaction:
            try:
                parsed_date = dt.date.fromisoformat(date.strip())
            except ValueError:
                raise InvalidTransactionError(f"date must look like 2026-03-15, got {date!r}") from None
            try:
                parsed_amount = Decimal(amount.strip())
            except InvalidOperation:
                raise InvalidTransactionError(f"amount must be a number, got {amount!r}") from None
            return cls(parsed_date, parsed_amount, category.strip().lower(), description.strip(), **extra)

    @dataclass(frozen=True)
    class RecurringTransaction(Transaction):
        frequency: str = "monthly"

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.frequency not in FREQUENCIES:
                raise InvalidTransactionError(f"frequency must be one of {FREQUENCIES}, got {self.frequency!r}")

        def _nth(self, n: int) -> dt.date:
            if self.frequency == "weekly":
                return self.date + dt.timedelta(weeks=n)
            year_offset, month_index = divmod(self.date.month - 1 + n, 12)
            year, month = self.date.year + year_offset, month_index + 1
            day = min(self.date.day, calendar.monthrange(year, month)[1])
            return dt.date(year, month, day)

        def occurrences(self, until: dt.date | None = None) -> list[Transaction]:
            until = until or self.date
            out: list[Transaction] = []
            n = 0
            while (when := self._nth(n)) <= until:
                out.append(Transaction(when, self.amount, self.category, self.description))
                n += 1
            return out

    class Account:
        def __init__(self, name: str = "Main", allow_overdraft: bool = False):
            self.name = name
            self.allow_overdraft = allow_overdraft
            self._transactions: list[Transaction] = []
            self._budgets: dict[str, Decimal] = {}

        @property
        def balance(self) -> Decimal:
            return sum((t.amount for t in self._transactions), Decimal(0))

        @property
        def transactions(self) -> tuple[Transaction, ...]:
            return tuple(self._transactions)

        @property
        def budgets(self) -> dict[str, Decimal]:
            return dict(self._budgets)

        def spent(self, category: str, year: int, month: int) -> Decimal:
            return sum(
                (-t.amount for t in self._transactions
                 if t.category == category and t.amount < 0 and (t.date.year, t.date.month) == (year, month)),
                Decimal(0),
            )

        def set_budget(self, category: str, monthly_limit: Decimal) -> None:
            if monthly_limit <= 0:
                raise InvalidTransactionError("a budget limit must be positive")
            self._budgets[category.strip().lower()] = monthly_limit

        def add(self, transaction: Transaction, until: dt.date | None = None) -> list[Transaction]:
            return self.add_many(transaction.occurrences(until))

        def add_many(self, transactions, *, enforce_limits: bool = True) -> list[Transaction]:
            batch = sorted(transactions, key=lambda t: t.date)
            if enforce_limits:
                self._check(batch)
            self._transactions.extend(batch)
            return batch

        def _check(self, batch: list[Transaction]) -> None:
            balance = self.balance
            for t in batch:
                balance += t.amount
                if t.amount > 0:
                    continue
                if not self.allow_overdraft and balance < 0:
                    raise InsufficientFundsError(balance - t.amount, -t.amount)

    @dataclass
    class LoadResult:
        account: Account
        problems: list[str] = field(default_factory=list)

    def save(account: Account, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "version": 1,
            "name": account.name,
            "allow_overdraft": account.allow_overdraft,
            "budgets": {c: str(v) for c, v in account.budgets.items()},
            "transactions": [
                {"date": t.date.isoformat(), "amount": str(t.amount), "category": t.category, "description": t.description}
                for t in account.transactions
            ],
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def load(path: str | Path) -> LoadResult:
        path = Path(path)
        if not path.exists():
            return LoadResult(Account())
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("top level is not an object")
        except (ValueError, UnicodeDecodeError) as exc:
            quarantine = path.with_name(path.name + ".corrupt")
            os.replace(path, quarantine)
            return LoadResult(Account(), [f"{path.name} was unreadable ({exc}); kept as {quarantine.name}"])

        account = Account(name=str(data.get("name", "Main")), allow_overdraft=bool(data.get("allow_overdraft", False)))
        problems: list[str] = []
        for category, limit in (data.get("budgets") or {}).items():
            try:
                account.set_budget(category, Decimal(limit))
            except (InvalidOperation, TypeError, BudgetError) as exc:
                problems.append(f"budget {category!r} skipped: {exc}")
        good: list[Transaction] = []
        for index, record in enumerate(data.get("transactions") or []):
            try:
                good.append(Transaction.parse(record["date"], record["amount"], record["category"], record.get("description", "")))
            except (KeyError, TypeError, AttributeError, BudgetError) as exc:
                problems.append(f"transaction #{index} skipped: {exc!r}")
        account.add_many(good, enforce_limits=False)
        return LoadResult(account, problems)


def delete_transaction(account: Account, index: int) -> bool:
    """Helper for Stretch Idea 1: remove a transaction by 0-based index."""
    if hasattr(account, "_transactions") and 0 <= index < len(account._transactions):
        del account._transactions[index]
        return True
    return False


# --- Milestone 1: Implement transaction tracking and saving to JSON.
def milestone_1(transactions: list[Transaction] | None = None, path: str | Path | None = None) -> Account:
    """Implement transaction tracking and saving to JSON.

    Uses the OOP class hierarchy (Transaction, RecurringTransaction, Account)
    and atomic JSON persistence (save, load).
    """
    target_path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
    target_path.parent.mkdir(parents=True, exist_ok=True)

    account = Account("Client Expense Portfolio", allow_overdraft=True)

    if transactions is not None:
        account.add_many(transactions, enforce_limits=False)
    else:
        # Default sample transactions across common financial planner categories
        sample_txns = [
            Transaction(dt.date(2026, 1, 5), Decimal("5000.00"), "income", "Monthly salary"),
            Transaction(dt.date(2026, 1, 7), Decimal("-1500.00"), "housing", "Monthly rent"),
            Transaction(dt.date(2026, 1, 10), Decimal("-145.20"), "groceries", "Supermarket run"),
            Transaction(dt.date(2026, 1, 14), Decimal("-65.00"), "dining", "Dinner with friends"),
            Transaction(dt.date(2026, 1, 18), Decimal("-80.00"), "utilities", "Electricity & internet"),
            Transaction(dt.date(2026, 1, 24), Decimal("-210.00"), "groceries", "Whole foods groceries"),
            Transaction(dt.date(2026, 2, 2), Decimal("-1500.00"), "housing", "Monthly rent"),
            Transaction(dt.date(2026, 2, 8), Decimal("-75.50"), "dining", "Brunch outing"),
            Transaction(dt.date(2026, 2, 12), Decimal("-160.00"), "shopping", "Winter clothes"),
            Transaction(dt.date(2026, 2, 18), Decimal("-90.00"), "entertainment", "Concert tickets"),
            Transaction(dt.date(2026, 2, 25), Decimal("-180.00"), "groceries", "Weekly groceries"),
            Transaction(dt.date(2026, 3, 2), Decimal("-120.00"), "dining", "Family dinner"),
        ]
        account.add_many(sample_txns, enforce_limits=False)
        account.set_budget("dining", Decimal("250.00"))
        account.set_budget("groceries", Decimal("600.00"))
        account.set_budget("housing", Decimal("1600.00"))
        account.set_budget("shopping", Decimal("200.00"))
        account.set_budget("entertainment", Decimal("150.00"))

    save(account, target_path)
    return account


# ---------------------------------------------------------------------------
# Milestone 2: RFM Feature Engineering (Customer Segmentation Drill)
# ---------------------------------------------------------------------------

def calculate_rfm(
    transactions_or_account: Account | list[Transaction] | Path | str | None = None,
    group_by: str = "category",
    reference_date: dt.date | None = None,
) -> pd.DataFrame:
    """Preprocess transaction records into Recency, Frequency, and Monetary (RFM) features.

    - Recency: Days elapsed since the last transaction in this segment.
    - Frequency: Total count of transactions in this segment.
    - Monetary: Total spending volume (positive magnitude).
    """
    txns: list[Transaction] = []

    if isinstance(transactions_or_account, Account):
        txns = list(transactions_or_account.transactions)
    elif isinstance(transactions_or_account, (list, tuple)):
        txns = list(transactions_or_account)
    elif isinstance(transactions_or_account, (str, Path)):
        load_res = load(transactions_or_account)
        txns = list(load_res.account.transactions)
    elif transactions_or_account is None:
        target_path = Path(os.environ.get("DATA_PATH", "data/app.json"))
        if target_path.exists():
            txns = list(load(target_path).account.transactions)
        if not txns:
            account = milestone_1()
            txns = list(account.transactions)

    # Filter for expense transactions (spending analysis focuses on outflows)
    expense_txns = [t for t in txns if t.amount < 0]
    if not expense_txns:
        expense_txns = txns

    if not expense_txns:
        return pd.DataFrame(columns=[group_by, "recency", "frequency", "monetary"])

    # Determine reference date (defaults to most recent transaction date across dataset)
    if reference_date is None:
        reference_date = max(t.date for t in expense_txns)

    # Aggregate by group (default: category)
    records: dict[str, list[Transaction]] = {}
    for t in expense_txns:
        key = getattr(t, group_by, t.category)
        records.setdefault(key, []).append(t)

    rows = []
    for key, group_txns in records.items():
        latest_txn_date = max(t.date for t in group_txns)
        recency = max(0, (reference_date - latest_txn_date).days)
        frequency = len(group_txns)
        monetary = float(sum(abs(t.amount) for t in group_txns))
        rows.append({
            group_by: key,
            "recency": int(recency),
            "frequency": int(frequency),
            "monetary": round(monetary, 2),
        })

    df = pd.DataFrame(rows)

    # Calculate standardized features for clustering (Z-score normalization)
    for col in ["recency", "frequency", "monetary"]:
        std = df[col].std()
        if std and not np.isnan(std) and std != 0:
            df[f"{col}_scaled"] = (df[col] - df[col].mean()) / std
        else:
            df[f"{col}_scaled"] = 0.0

    return df.sort_values(by="monetary", ascending=False).reset_index(drop=True)


# --- Milestone 2: Add functionality to load transactions and preprocess for RFM features.
def milestone_2(
    path: str | Path | None = None,
    transactions: list[Transaction] | None = None,
    group_by: str = "category",
    reference_date: dt.date | None = None,
) -> pd.DataFrame:
    """Load transactions and engineer RFM features as demonstrated in the customer segmentation drill."""
    source = transactions if transactions is not None else path
    return calculate_rfm(source, group_by=group_by, reference_date=reference_date)


# --- Milestone 3: Implement K-Means from scratch to cluster spending patterns.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Display budget summary and identified spending segments in Streamlit.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
