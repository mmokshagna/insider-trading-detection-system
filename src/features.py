"""Feature engineering utilities for insider trading anomaly detection."""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from src.ingest import load_insider_csvs


def _get_date_column(df: pd.DataFrame) -> str:
    """Return the name of the primary date column.

    The function searches for a column named ``date`` first, then the first
    column containing the substring ``date``. Raises a ``ValueError`` if no
    date-like column is found.
    """

    if "date" in df.columns:
        return "date"
    for col in df.columns:
        if "date" in col:
            return col
    raise ValueError("No date column found in dataframe.")


def _get_transaction_column(df: pd.DataFrame) -> str:
    """Return the transaction column name, prioritizing ``transaction``."""

    if "transaction" in df.columns:
        return "transaction"
    for col in df.columns:
        if "transaction" in col:
            return col
    raise ValueError("Transaction column missing from dataframe.")


def _get_insider_column(df: pd.DataFrame) -> Optional[str]:
    """Return the insider identifier column if one exists."""

    for candidate in ("insider", "insider_name", "reporting_owner", "owner"):
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if "insider" in col:
            return col
    return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw insider trading data.

    Steps:
    - Lowercase column names.
    - Remove rows missing essential fields (date, shares, price).
    - Convert numeric columns safely.
    - Convert date columns to datetime.
    - Keep only buy or sale transactions.
    """

    cleaned = df.copy()
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]

    essential = [col for col in ("date", "shares", "price") if col in cleaned.columns]
    if len(essential) < 3:
        missing = {"date", "shares", "price"} - set(essential)
        raise ValueError(f"Missing essential columns: {missing}")

    cleaned = cleaned.dropna(subset=essential)

    for num_col in ("shares", "price", "value"):
        if num_col in cleaned.columns:
            cleaned[num_col] = pd.to_numeric(cleaned[num_col], errors="coerce")

    date_columns = [col for col in cleaned.columns if "date" in col]
    for col in date_columns:
        cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=essential)

    transaction_col = _get_transaction_column(cleaned)
    cleaned[transaction_col] = cleaned[transaction_col].astype(str).str.strip().str.title()
    cleaned = cleaned[cleaned[transaction_col].isin(["Buy", "Sale"])]

    return cleaned.reset_index(drop=True)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple, deterministic features to the dataset."""

    enriched = df.copy()
    transaction_col = _get_transaction_column(enriched)

    enriched["trade_value"] = enriched["shares"] * enriched["price"]
    enriched["is_buy"] = enriched[transaction_col] == "Buy"
    enriched["is_sell"] = enriched[transaction_col] == "Sale"

    relationship_col = "relationship" if "relationship" in enriched.columns else None
    if relationship_col is None:
        for col in enriched.columns:
            if "relationship" in col:
                relationship_col = col
                break

    def _extract_role(text: str) -> str:
        normalized = str(text).lower()
        if "ceo" in normalized:
            return "CEO"
        if "cfo" in normalized:
            return "CFO"
        if "director" in normalized:
            return "Director"
        if "officer" in normalized:
            return "Officer"
        return "Other"

    if relationship_col:
        enriched["insider_role"] = enriched[relationship_col].apply(_extract_role).astype("category")

    if "ticker" in enriched.columns:
        enriched["ticker"] = enriched["ticker"].astype("category")

    return enriched


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and percentile-based statistical features."""

    stats_df = df.copy()
    date_col = _get_date_column(stats_df)
    transaction_col = _get_transaction_column(stats_df)

    # Ensure sorting for deterministic rolling calculations
    stats_df = stats_df.sort_values(["ticker", date_col])

    rolling_7d = (
        stats_df.groupby("ticker")
        .rolling(window="7D", on=date_col)["trade_value"]
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_30d = (
        stats_df.groupby("ticker")
        .rolling(window="30D", on=date_col)["trade_value"]
        .mean()
        .reset_index(level=0, drop=True)
    )

    stats_df["trade_value_7d_mean"] = rolling_7d
    stats_df["trade_value_30d_mean"] = rolling_30d

    rolling_mean = (
        stats_df.groupby("ticker")
        .rolling(window="30D", on=date_col)["trade_value"]
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_std = (
        stats_df.groupby("ticker")
        .rolling(window="30D", on=date_col)["trade_value"]
        .std()
        .reset_index(level=0, drop=True)
    ).replace(0, pd.NA)

    stats_df["trade_value_zscore"] = (stats_df["trade_value"] - rolling_mean) / rolling_std

    stats_df["trade_value_pct_rank"] = stats_df.groupby("ticker")["trade_value"].rank(pct=True)
    percentile_95 = stats_df.groupby("ticker")["trade_value"].transform(lambda x: x.quantile(0.95))
    stats_df["unusual_volume"] = (stats_df["trade_value"] > percentile_95).astype(int)

    return stats_df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived features to the dataset."""

    dated_df = df.copy()
    date_col = _get_date_column(dated_df)
    insider_col = _get_insider_column(dated_df)

    dated_df["year"] = dated_df[date_col].dt.year
    dated_df["month"] = dated_df[date_col].dt.month
    dated_df["day"] = dated_df[date_col].dt.day
    dated_df["weekday"] = dated_df[date_col].dt.weekday

    if insider_col and "ticker" in dated_df.columns:
        dated_df = dated_df.sort_values([insider_col, "ticker", date_col])
        dated_df["days_since_last_trade"] = (
            dated_df.groupby([insider_col, "ticker"])[date_col].diff().dt.days
        )
    else:
        dated_df["days_since_last_trade"] = pd.NA

    return dated_df


def build_feature_matrix() -> pd.DataFrame:
    """Load insider trading data and produce an ML-ready feature matrix."""

    raw_df = load_insider_csvs()
    if raw_df is None or raw_df.empty:
        return raw_df if raw_df is not None else pd.DataFrame()

    df = clean_data(raw_df)
    df = add_basic_features(df)
    df = add_statistical_features(df)
    df = add_date_features(df)

    drop_columns: Iterable[str] = (
        "relationship",
        "insider",
        "insider_name",
        "reporting_owner",
        "owner",
        "remarks",
    )
    existing_drop_cols = [col for col in drop_columns if col in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)

    return df


if __name__ == "__main__":
    feature_df = build_feature_matrix()
    print("Feature matrix shape:", feature_df.shape)
    print(feature_df.head())
