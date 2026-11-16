"""Streamlit entry point: `streamlit run app/main.py`

Spending Segment Analyzer for financial planners:
- Tracks client expenses with crash-resilient JSON persistence.
- Preprocesses transactions for RFM (Recency, Frequency, Monetary) features.
- Clusters spending patterns using K-Means from scratch (K-Means++ initialization).
- Visualizes budget summaries, spending archetypes, 2D scatter cluster maps, and temporal trends.
- Exports formatted Markdown decision memos and CSV reports for client advisory.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.core import (
    Account,
    BudgetExceededError,
    Transaction,
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

st.set_page_config(
    page_title="Spending Segment Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(os.environ.get("DATA_PATH", "data/app.json"))


def display_dataframe(df: pd.DataFrame, **kwargs) -> None:
    """Helper to render DataFrames cleanly across both modern and legacy Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def display_line_chart(data: Any, **kwargs) -> None:
    """Helper to render line charts cleanly across Streamlit versions."""
    try:
        st.line_chart(data, width="stretch", **kwargs)
    except TypeError:
        st.line_chart(data, use_container_width=True, **kwargs)


def load_current_account() -> Account:
    """Load account from disk or initialize with baseline data."""
    if DATA_FILE.exists():
        result = load(DATA_FILE)
        if result.account.transactions:
            return result.account
    return milestone_1(path=DATA_FILE)


# Initialize or retrieve session state
if "account" not in st.session_state:
    st.session_state.account = load_current_account()

account: Account = st.session_state.account

# --- Header ---
st.title("💳 Spending Segment Analyzer")
st.caption(
    "Financial planners use a Streamlit app to track expenses, then get segmented spending habits "
    "(e.g., 'Foodie', 'Saver') to improve budgeting for clients."
)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Planning Controls")

k_clusters = st.sidebar.slider(
    "Number of Spending Clusters (k)",
    min_value=2,
    max_value=5,
    value=3,
    help="Target clusters for K-Means algorithm implemented from scratch with K-Means++ initialization.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Add Transaction")

with st.sidebar.form("add_txn_form", clear_on_submit=True):
    txn_date = st.date_input("Date", value=dt.date.today())
    txn_type = st.radio("Type", ["Expense", "Income"], horizontal=True)
    txn_category = st.text_input("Category", value="dining", placeholder="e.g., dining, groceries, rent")
    txn_amount = st.number_input("Amount ($)", min_value=0.01, value=50.00, step=5.0)
    txn_desc = st.text_input("Description", placeholder="e.g. Client dinner")
    submitted = st.form_submit_button("Record Transaction")

    if submitted:
        cat_clean = txn_category.strip().lower()
        if not cat_clean:
            st.sidebar.error("Category name cannot be empty.")
        else:
            amt = -Decimal(str(txn_amount)) if txn_type == "Expense" else Decimal(str(txn_amount))
            try:
                new_txn = Transaction(
                    date=txn_date,
                    amount=amt,
                    category=cat_clean,
                    description=txn_desc.strip(),
                )
                try:
                    account.add(new_txn)
                except BudgetExceededError as budget_err:
                    # Still record historical outflow and warn planner
                    account.add_many([new_txn], enforce_limits=False)
                    st.sidebar.warning(f"Recorded expense, but note: {budget_err}")

                save(account, DATA_FILE)
                st.session_state.account = account
                st.sidebar.success(f"Recorded {cat_clean} transaction!")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Failed to record transaction: {exc}")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Set Category Budget")
with st.sidebar.form("budget_form", clear_on_submit=True):
    b_category = st.text_input("Category", placeholder="e.g., dining")
    b_limit = st.number_input("Monthly Limit ($)", min_value=1.0, value=300.0, step=25.0)
    b_submit = st.form_submit_button("Set Budget")
    if b_submit:
        b_cat_clean = b_category.strip().lower()
        if not b_cat_clean:
            st.sidebar.error("Category cannot be empty.")
        else:
            try:
                account.set_budget(b_cat_clean, Decimal(str(b_limit)))
                save(account, DATA_FILE)
                st.session_state.account = account
                st.sidebar.success(f"Budget for '{b_cat_clean}' set to ${b_limit:,.2f}")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Error setting budget: {exc}")

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export & Reset")
if st.sidebar.button("🔄 Reset to Sample Data"):
    account = milestone_1(path=DATA_FILE)
    st.session_state.account = account
    st.sidebar.info("Sample statement reloaded!")
    st.rerun()

# --- Compute Analytics via Milestones ---
summary_data = milestone_4(account=account, path=DATA_FILE, k=k_clusters)
df_segments: pd.DataFrame = summary_data["segments_df"]
df_trends: pd.DataFrame = summary_data["spending_trends"]

# Sidebar quick download for client decision memo
memo_text = summary_data.get("memo_markdown", generate_planner_report(account, summary_data))
st.sidebar.download_button(
    "📄 Download Client Memo",
    data=memo_text,
    file_name=f"{account.name.lower().replace(' ', '_')}_memo.md",
    mime="text/markdown",
)

# --- Top Level Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Net Balance", f"${summary_data['balance']:,.2f}")
col2.metric("Total Expenses", f"${summary_data['total_expenses']:,.2f}")
col3.metric("Total Inflow", f"${summary_data['total_income']:,.2f}")
col4.metric(
    "Budget Status",
    f"{summary_data['over_budget_count']} Exceeded" if summary_data['over_budget_count'] > 0 else "All Within Budget",
    delta=f"-{summary_data['over_budget_count']}" if summary_data['over_budget_count'] > 0 else "Optimal",
    delta_color="inverse",
)

st.markdown("---")

# --- Tabs for Dashboard Views ---
tab_budget, tab_segments, tab_trends, tab_txns = st.tabs([
    "📊 Budget Summary (M1 & M4)",
    "🎯 RFM & Habit Segments (M2 & M3)",
    "📈 Spending Trends (Stretch 2)",
    "📝 Transaction Ledger (M1 & Stretch 1)",
])

# --- TAB 1: Budget Summary ---
with tab_budget:
    st.subheader("Category Budget Allocation vs. Actual Spending")
    cat_summaries = summary_data["category_summaries"]

    if cat_summaries:
        summary_df = pd.DataFrame(cat_summaries)
        display_df = summary_df[[
            "category", "budget", "spent", "remaining", "utilization_pct", "segment_label", "over_budget"
        ]].copy()
        display_df.columns = [
            "Category", "Monthly Budget ($)", "Actual Spent ($)", "Remaining ($)", "Utilization (%)", "Assigned Persona", "Over Budget?"
        ]
        display_dataframe(display_df)

        st.markdown("#### Budget Utilization Meters")
        meter_cols = st.columns(min(max(len(cat_summaries), 1), 3))
        for idx, item in enumerate(cat_summaries):
            col = meter_cols[idx % 3]
            with col:
                cat_name = item["category"].capitalize()
                spent_val = item["spent"]
                budget_val = item["budget"]
                util_pct = min(100.0, item["utilization_pct"])
                st.write(f"**{cat_name}**: ${spent_val:,.2f} of ${budget_val:,.2f} limit")
                st.progress(util_pct / 100.0)
                if item["over_budget"]:
                    st.caption(f"🚨 :red[Over budget by ${spent_val - budget_val:,.2f}]")
                elif budget_val > 0:
                    st.caption(f"✅ {item['utilization_pct']}% utilized (${item['remaining']:,.2f} remaining)")
                else:
                    st.caption("ℹ️ No explicit budget limit set.")
    else:
        st.info("No expense categories found yet.")

# --- TAB 2: RFM & Habit Segments ---
with tab_segments:
    st.subheader("Customer Segmentation Drill: RFM Features & K-Means from Scratch")
    st.write(
        "Transactions are preprocessed into **Recency** (days since last purchase), "
        "**Frequency** (count of transactions), and **Monetary** (aggregate spend), "
        "then clustered using our pure NumPy **K-Means from Scratch** algorithm (with K-Means++ centroid initialization)."
    )

    if not df_segments.empty and "cluster" in df_segments.columns:
        col_table, col_chart = st.columns([1, 1])

        with col_table:
            st.markdown("#### RFM Spending Habit Matrix")
            show_cols = [c for c in ["category", "recency", "frequency", "monetary", "cluster", "segment_label"] if c in df_segments.columns]
            table_view = df_segments[show_cols].copy()
            table_view.columns = [c.replace("_", " ").title() for c in show_cols]
            display_dataframe(table_view)

            # Export button for CSV
            st.download_button(
                "📥 Export RFM Segments (CSV)",
                data=df_segments.to_csv(index=False).encode("utf-8"),
                file_name="rfm_spending_segments.csv",
                mime="text/csv",
            )

        with col_chart:
            st.markdown("#### 2D Cluster Space: Recency vs. Monetary Outflow")
            st.caption("Point color indicates assigned Persona; point size scales with transaction Frequency.")
            st.scatter_chart(
                df_segments,
                x="recency",
                y="monetary",
                color="segment_label",
                size="frequency",
            )

        st.markdown("#### Identified Client Spending Personas")
        for cluster_id in sorted(df_segments["cluster"].unique()):
            sub = df_segments[df_segments["cluster"] == cluster_id]
            if not sub.empty:
                persona_name = sub["segment_label"].iloc[0]
                cats = ", ".join(sub["category"].tolist())
                avg_spend = sub["monetary"].mean()
                avg_freq = sub["frequency"].mean()

                with st.expander(f"Cluster {cluster_id}: {persona_name}", expanded=True):
                    st.write(f"**Categories in this habit group:** `{cats}`")
                    st.write(f"**Average Outflow:** `${avg_spend:,.2f}` | **Average Transaction Count:** `{avg_freq:.1f}`")
                    if "Foodie" in persona_name or "Micro-Spender" in persona_name:
                        st.info("💡 **Financial Planner Recommendation:** Frequent low-ticket spending. Recommend setting weekly discretionary envelopes.")
                    elif "Major Outflow" in persona_name or "Fixed" in persona_name:
                        st.warning("💡 **Financial Planner Recommendation:** High-ticket fixed expenses. Rebalance with automated savings buffers on payday.")
                    elif "Saver" in persona_name or "Low" in persona_name:
                        st.success("💡 **Financial Planner Recommendation:** High savings adherence. Safe to allocate surplus toward investment goals.")
                    else:
                        st.info("💡 **Financial Planner Recommendation:** Core operational spending. Review quarterly for efficiency.")
    else:
        st.info("No expense data available for RFM clustering.")

# --- TAB 3: Spending Trends ---
with tab_trends:
    st.subheader("Historical Spending Trends Over Time (Stretch Goal 2)")
    if not df_trends.empty:
        st.write("Visualizing cumulative outflows and transaction chronology across time periods.")
        trend_plot_df = df_trends.set_index("date")[["cumulative_spend"]]
        display_line_chart(trend_plot_df)

        st.markdown("#### Outflows by Date")
        display_dataframe(df_trends[["date", "category", "amount", "cumulative_spend", "description"]])
    else:
        st.info("No expense history recorded yet to build trends.")

# --- TAB 4: Transaction Ledger & Deletion ---
with tab_txns:
    st.subheader("Transaction Ledger & Management (Stretch Goal 1)")
    txns = list(account.transactions)

    if txns:
        txn_records = []
        for idx, t in enumerate(txns):
            txn_records.append({
                "Index": idx,
                "Date": t.date.isoformat(),
                "Type": t.kind.capitalize(),
                "Category": t.category,
                "Amount ($)": float(t.amount),
                "Description": t.description,
            })
        df_ledger = pd.DataFrame(txn_records)
        display_dataframe(df_ledger)

        st.download_button(
            "📥 Export Transaction Ledger (CSV)",
            data=df_ledger.to_csv(index=False).encode("utf-8"),
            file_name="transactions_ledger.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("#### 🗑️ Delete Transaction")
        del_idx = st.selectbox(
            "Select transaction to remove",
            options=range(len(txns)),
            format_func=lambda i: f"#{i} | {txns[i].date} | {txns[i].category} | ${float(txns[i].amount):,.2f} ({txns[i].description})",
        )
        if st.button("Delete Selected Transaction", type="primary"):
            if delete_transaction(account, del_idx):
                save(account, DATA_FILE)
                st.session_state.account = account
                st.success(f"Deleted transaction #{del_idx}.")
                st.rerun()
    else:
        st.info("No transactions recorded yet.")
