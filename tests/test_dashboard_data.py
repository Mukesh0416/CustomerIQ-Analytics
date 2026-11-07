"""
tests/test_dashboard_data.py
-----------------------------
Pytest suite for Customer 360 Revenue Intelligence dashboard.

Covers:
  - Column normalization
  - Required-column validation
  - Missing customer identifier
  - Duplicate normalized column names
  - Data loading from parquet
  - Customer-level aggregation
  - Dashboard data preparation (filters, churn probability guard)
  - Optional return columns
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Replicate the two pure-Python helpers from streamlit_app.py ──────────────
# (avoids importing Streamlit at test time)

OPTIONAL_RETURN_COLS = [
    "has_returned", "total_returns", "return_rate", "return_quantity_abs"
]

REQUIRED_COLS = [
    "customer_id", "total_revenue", "health_tier", "rfm_segment",
    "action_priority", "customer_health_score", "churn_risk_tier",
    "final_recommended_action", "model_scored", "churn_probability",
    "country_mode", "vip_flag", "retention_target_flag",
    "customer_value_tier", "r_score", "f_score", "m_score",
    "rfm_score", "total_invoices", "recency_days",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_names = [
        re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")
        for col in df.columns
    ]
    if len(new_names) != len(set(new_names)):
        seen, dupes = set(), []
        for n in new_names:
            if n in seen:
                dupes.append(n)
            seen.add(n)
        raise ValueError(f"Column normalization produced duplicate names: {dupes}")
    df = df.copy()
    df.columns = new_names
    return df


def validate_required_columns(df: pd.DataFrame, required: list) -> list:
    return [c for c in required if c not in df.columns]


# ── Minimal sample DataFrame factory ─────────────────────────────────────────

def make_sample_df(n=10, include_returns=True, customer_id_col="customer_id"):
    rng = np.random.default_rng(42)
    data = {
        customer_id_col:            list(range(1000, 1000 + n)),
        "total_revenue":            rng.uniform(50, 5000, n).tolist(),
        "health_tier":              rng.choice(["Excellent", "Healthy", "Watchlist", "At Risk", "Critical"], n).tolist(),
        "rfm_segment":              rng.choice(["Champions", "Loyal Customers", "Hibernating"], n).tolist(),
        "action_priority":          rng.choice(["Priority 1", "Priority 2", "Priority 3"], n).tolist(),
        "customer_health_score":    rng.uniform(10, 99, n).tolist(),
        "churn_risk_tier":          rng.choice(["Low Risk", "High Risk", "Critical Risk", "Not Scored"], n).tolist(),
        "final_recommended_action": ["Take action"] * n,
        "model_scored":             rng.integers(0, 2, n).tolist(),
        "churn_probability":        rng.uniform(0, 1, n).tolist(),
        "country_mode":             rng.choice(["United Kingdom", "Germany", "France"], n).tolist(),
        "vip_flag":                 rng.integers(0, 2, n).tolist(),
        "retention_target_flag":    rng.integers(0, 2, n).tolist(),
        "customer_value_tier":      rng.choice(["Top 10% Value", "High Value", "Mid Value", "Low Value"], n).tolist(),
        "r_score":                  rng.integers(1, 6, n).tolist(),
        "f_score":                  rng.integers(1, 6, n).tolist(),
        "m_score":                  rng.integers(1, 6, n).tolist(),
        "rfm_score":                ["555"] * n,
        "total_invoices":           rng.integers(1, 50, n).tolist(),
        "recency_days":             rng.integers(1, 400, n).tolist(),
    }
    if include_returns:
        data["has_returned"]        = rng.integers(0, 2, n).tolist()
        data["total_returns"]       = rng.integers(0, 5, n).astype(float).tolist()
        data["return_rate"]         = rng.uniform(0, 0.3, n).tolist()
        data["return_quantity_abs"] = rng.integers(0, 10, n).astype(float).tolist()
    return pd.DataFrame(data)


# ════════════════════════════════════════════════════════════════════════════
# 1. Column normalization
# ════════════════════════════════════════════════════════════════════════════

class TestNormalizeColumns:

    def test_lowercase_conversion(self):
        df = pd.DataFrame({"CustomerID": [1], "TotalRevenue": [100.0]})
        result = normalize_columns(df)
        assert list(result.columns) == ["customerid", "totalrevenue"]

    def test_spaces_replaced_with_underscore(self):
        df = pd.DataFrame({"Customer ID": [1], "Total Revenue": [100.0]})
        result = normalize_columns(df)
        assert "customer_id" in result.columns
        assert "total_revenue" in result.columns

    def test_mixed_case_and_spaces(self):
        df = pd.DataFrame({"Customer  ID": [1], "Unit Price": [9.99]})
        result = normalize_columns(df)
        assert "customer_id" in result.columns
        assert "unit_price" in result.columns

    def test_special_characters_stripped(self):
        df = pd.DataFrame({"Revenue (GBP)": [1.0], "Qty#": [2]})
        result = normalize_columns(df)
        assert "revenue_gbp" in result.columns
        assert "qty" in result.columns

    def test_already_snake_case_unchanged(self):
        df = pd.DataFrame({"customer_id": [1], "total_revenue": [100.0]})
        result = normalize_columns(df)
        assert list(result.columns) == ["customer_id", "total_revenue"]

    def test_original_df_not_mutated(self):
        df = pd.DataFrame({"Customer ID": [1]})
        original_cols = list(df.columns)
        normalize_columns(df)
        assert list(df.columns) == original_cols

    def test_duplicate_after_normalization_raises(self):
        df = pd.DataFrame({"Customer ID": [1], "customer_id": [2]})
        with pytest.raises(ValueError, match="duplicate names"):
            normalize_columns(df)

    def test_duplicate_spaces_normalize_to_same_raises(self):
        df = pd.DataFrame({"Total  Revenue": [1.0], "Total Revenue": [2.0]})
        with pytest.raises(ValueError, match="duplicate names"):
            normalize_columns(df)


# ════════════════════════════════════════════════════════════════════════════
# 2. Required-column validation
# ════════════════════════════════════════════════════════════════════════════

class TestValidateRequiredColumns:

    def test_all_required_present_returns_empty(self):
        df = make_sample_df()
        missing = validate_required_columns(df, REQUIRED_COLS)
        assert missing == [], f"Unexpected missing columns: {missing}"

    def test_missing_customer_id_detected(self):
        df = make_sample_df().drop(columns=["customer_id"])
        missing = validate_required_columns(df, REQUIRED_COLS)
        assert "customer_id" in missing

    def test_missing_total_revenue_detected(self):
        df = make_sample_df().drop(columns=["total_revenue"])
        missing = validate_required_columns(df, REQUIRED_COLS)
        assert "total_revenue" in missing

    def test_multiple_missing_all_reported(self):
        df = make_sample_df().drop(columns=["customer_id", "health_tier", "rfm_segment"])
        missing = validate_required_columns(df, REQUIRED_COLS)
        assert set(missing) == {"customer_id", "health_tier", "rfm_segment"}

    def test_extra_columns_do_not_cause_failure(self):
        df = make_sample_df()
        df["extra_col"] = 0
        missing = validate_required_columns(df, REQUIRED_COLS)
        assert missing == []

    def test_empty_dataframe_reports_all_missing(self):
        df = pd.DataFrame()
        missing = validate_required_columns(df, REQUIRED_COLS)
        assert set(missing) == set(REQUIRED_COLS)


# ════════════════════════════════════════════════════════════════════════════
# 3. Optional return columns
# ════════════════════════════════════════════════════════════════════════════

class TestOptionalReturnColumns:

    def test_return_cols_present_when_included(self):
        df = make_sample_df(include_returns=True)
        for col in OPTIONAL_RETURN_COLS:
            assert col in df.columns

    def test_return_cols_absent_when_excluded(self):
        df = make_sample_df(include_returns=False)
        for col in OPTIONAL_RETURN_COLS:
            assert col not in df.columns

    def test_adding_nan_return_cols_does_not_break_required_validation(self):
        df = make_sample_df(include_returns=False)
        for col in OPTIONAL_RETURN_COLS:
            df[col] = np.nan
        missing = validate_required_columns(df, REQUIRED_COLS)
        assert missing == []

    def test_nan_return_cols_are_float(self):
        df = make_sample_df(include_returns=False)
        for col in OPTIONAL_RETURN_COLS:
            df[col] = np.nan
        for col in OPTIONAL_RETURN_COLS:
            assert df[col].dtype == float

    def test_has_returned_nan_display_logic(self):
        val = np.nan
        result = (
            "Yes" if val == 1
            else "No" if val == 0
            else "N/A (returns data unavailable)"
        )
        assert result == "N/A (returns data unavailable)"

    def test_return_rate_nan_display_logic(self):
        val = np.nan
        result = f"{val:.1%}" if pd.notna(val) else "N/A"
        assert result == "N/A"

    def test_total_returns_nan_display_logic(self):
        val = np.nan
        result = f"{int(val):,}" if pd.notna(val) else "N/A"
        assert result == "N/A"


# ════════════════════════════════════════════════════════════════════════════
# 4. Data loading from parquet
# ════════════════════════════════════════════════════════════════════════════

PARQUET_PATH = (
    Path(__file__).parent.parent / "data" / "processed" / "customer_360.parquet"
)


@pytest.mark.skipif(not PARQUET_PATH.exists(), reason="customer_360.parquet not found")
class TestParquetLoading:

    def test_parquet_loads_without_error(self):
        df = pd.read_parquet(PARQUET_PATH)
        assert isinstance(df, pd.DataFrame)

    def test_parquet_has_rows(self):
        df = pd.read_parquet(PARQUET_PATH)
        assert len(df) > 0

    def test_parquet_customer_id_is_unique(self):
        df = pd.read_parquet(PARQUET_PATH)
        assert df["customer_id"].is_unique

    def test_parquet_required_cols_present_after_normalize(self):
        df = pd.read_parquet(PARQUET_PATH)
        df = normalize_columns(df)
        missing = validate_required_columns(df, REQUIRED_COLS)
        assert missing == [], f"Required columns missing after normalize: {missing}"

    def test_parquet_customer_id_dtype_is_numeric(self):
        df = pd.read_parquet(PARQUET_PATH)
        assert pd.api.types.is_numeric_dtype(df["customer_id"])

    def test_parquet_total_revenue_non_negative(self):
        df = pd.read_parquet(PARQUET_PATH)
        assert (df["total_revenue"] >= 0).all()

    def test_parquet_health_tier_valid_values(self):
        valid = {"Excellent", "Healthy", "Watchlist", "At Risk", "Critical"}
        df = pd.read_parquet(PARQUET_PATH)
        actual = set(df["health_tier"].dropna().unique())
        assert actual.issubset(valid), f"Unexpected health_tier values: {actual - valid}"

    def test_parquet_model_scored_binary(self):
        df = pd.read_parquet(PARQUET_PATH)
        assert set(df["model_scored"].unique()).issubset({0, 1})

    def test_parquet_churn_probability_range(self):
        df = pd.read_parquet(PARQUET_PATH)
        scored = df[df["model_scored"] == 1]["churn_probability"]
        assert scored.between(0.0, 1.0).all()

    def test_parquet_optional_return_cols_absent(self):
        df = pd.read_parquet(PARQUET_PATH)
        for col in OPTIONAL_RETURN_COLS:
            assert col not in df.columns, (
                f"Expected {col} to be absent — raw Excel was not processed"
            )


# ════════════════════════════════════════════════════════════════════════════
# 5. Customer-level aggregation
# ════════════════════════════════════════════════════════════════════════════

class TestCustomerAggregation:

    def test_groupby_rfm_segment_count(self):
        df = make_sample_df(n=30)
        result = df.groupby("rfm_segment").size().reset_index(name="count")
        assert "rfm_segment" in result.columns
        assert result["count"].sum() == 30

    def test_groupby_rfm_segment_revenue_sum_matches_total(self):
        df = make_sample_df(n=20)
        result = df.groupby("rfm_segment")["total_revenue"].sum()
        assert abs(result.sum() - df["total_revenue"].sum()) < 1e-6

    def test_agg_customer_count_uses_customer_id(self):
        df = make_sample_df(n=15)
        result = df.groupby("action_priority").agg(
            customer_count=("customer_id", "count"),
            total_revenue=("total_revenue", "sum"),
        ).reset_index()
        assert result["customer_count"].sum() == 15

    def test_top_n_by_revenue_is_sorted_descending(self):
        df = make_sample_df(n=20)
        top5 = df.nlargest(5, "total_revenue")
        assert len(top5) == 5
        assert top5["total_revenue"].is_monotonic_decreasing

    def test_vip_flag_sum_within_bounds(self):
        df = make_sample_df(n=50)
        vip_count = df["vip_flag"].sum()
        assert 0 <= vip_count <= 50

    def test_segment_summary_agg_shape(self):
        df = make_sample_df(n=30)
        summary = df.groupby("rfm_segment").agg(
            customer_count=("customer_id", "count"),
            total_revenue=("total_revenue", "sum"),
            avg_churn_probability=("churn_probability", "mean"),
        ).reset_index()
        assert "customer_count" in summary.columns
        assert summary["customer_count"].sum() == 30


# ════════════════════════════════════════════════════════════════════════════
# 6. Dashboard data preparation — filter & churn probability guard
# ════════════════════════════════════════════════════════════════════════════

class TestDashboardDataPrep:

    def _apply_churn_prob_filter(self, df, cp_range=(0.0, 1.0)):
        """Replicate the fixed filter logic from streamlit_app.py."""
        scored_in_df = df[df["model_scored"] == 1]
        if len(scored_in_df) > 0:
            mask_cp = scored_in_df["churn_probability"].between(cp_range[0], cp_range[1])
            keep_ids = (
                set(scored_in_df.loc[mask_cp, "customer_id"])
                | set(df.loc[df["model_scored"] == 0, "customer_id"])
            )
            df = df[df["customer_id"].isin(keep_ids)]
        return df

    def test_no_crash_when_all_customers_unscored(self):
        """Root cause: KeyError on customer_id when scored_in_df is empty."""
        df = make_sample_df(n=10)
        df["model_scored"] = 0
        df["churn_probability"] = np.nan
        result = self._apply_churn_prob_filter(df, cp_range=(0.2, 0.8))
        assert len(result) == 10

    def test_unscored_customers_always_preserved(self):
        df = make_sample_df(n=20)
        df.loc[:9, "model_scored"] = 0
        df.loc[:9, "churn_probability"] = np.nan
        df.loc[10:, "model_scored"] = 1
        df.loc[10:, "churn_probability"] = 0.9
        result = self._apply_churn_prob_filter(df, cp_range=(0.0, 0.5))
        unscored_ids = set(df.loc[:9, "customer_id"])
        assert unscored_ids.issubset(set(result["customer_id"]))

    def test_full_range_keeps_all_scored(self):
        df = make_sample_df(n=15)
        df["model_scored"] = 1
        result = self._apply_churn_prob_filter(df, cp_range=(0.0, 1.0))
        assert len(result) == 15

    def test_narrow_range_reduces_rows(self):
        df = make_sample_df(n=20)
        df["model_scored"] = 1
        df["churn_probability"] = np.linspace(0.0, 1.0, 20)
        result = self._apply_churn_prob_filter(df, cp_range=(0.4, 0.6))
        assert len(result) < 20

    def test_impossible_range_returns_empty(self):
        df = make_sample_df(n=10)
        df["model_scored"] = 1
        df["churn_probability"] = 0.1
        result = self._apply_churn_prob_filter(df, cp_range=(0.5, 1.0))
        assert len(result) == 0

    def test_health_tier_filter(self):
        df = make_sample_df(n=30)
        filtered = df[df["health_tier"].isin(["Excellent"])]
        assert all(filtered["health_tier"] == "Excellent")

    def test_revenue_range_filter_bounds(self):
        df = make_sample_df(n=50)
        lo = df["total_revenue"].quantile(0.25)
        hi = df["total_revenue"].quantile(0.75)
        filtered = df[(df["total_revenue"] >= lo) & (df["total_revenue"] <= hi)]
        assert (filtered["total_revenue"] >= lo).all()
        assert (filtered["total_revenue"] <= hi).all()

    def test_customer_lookup_by_id_returns_correct_row(self):
        df = make_sample_df(n=10)
        target_id = df["customer_id"].iloc[3]
        cust = df[df["customer_id"] == target_id].iloc[0]
        assert cust["customer_id"] == target_id

    def test_country_filter(self):
        df = make_sample_df(n=40)
        filtered = df[df["country_mode"].isin(["United Kingdom"])]
        assert all(filtered["country_mode"] == "United Kingdom")
