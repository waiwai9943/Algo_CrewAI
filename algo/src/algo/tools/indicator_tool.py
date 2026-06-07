"""
indicator_tool.py
-----------------
CrewAI custom tool: computes VWAP, VWAP bands (±1σ, ±2σ),
and RSI from a JSON-serialised OHLCV DataFrame produced by MarketDataTool.

Returns a JSON string with the original bars plus computed indicator columns.
"""

from __future__ import annotations

import json
import math
from typing import Type

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Input Schema
# ─────────────────────────────────────────────

class IndicatorInput(BaseModel):
    """Input schema for IndicatorTool."""

    market_data_json: str = Field(
        ...,
        description=(
            "JSON string returned by MarketDataTool. Must contain a 'full_data' key "
            "with a list of OHLCV bar records including 'typical_price' and 'session_date'."
        ),
    )
    rsi_period: int = Field(
        default=14,
        description="RSI lookback period. Common values: 7, 9, 14. Default: 14.",
    )
    vwap_std_window: int = Field(
        default=20,
        description=(
            "Rolling window (in bars) used to compute standard deviation of "
            "(close - VWAP) for VWAP band construction. Default: 20 bars = 100 minutes."
        ),
    )
    vwap_band_multipliers: list[float] = Field(
        default=[1.0, 2.0],
        description="Std-dev multipliers for VWAP bands. Default: [1.0, 2.0] → ±1σ and ±2σ.",
    )


# ─────────────────────────────────────────────
# Pure Computation Helpers
# ─────────────────────────────────────────────

def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute anchored intraday VWAP, resetting at the start of each session_date.

    Formula:
        VWAP_t = Σ(typical_price_i * volume_i) / Σ(volume_i)
        where i runs from session open to bar t.

    Args:
        df: DataFrame with columns ['typical_price', 'volume', 'session_date'].

    Returns:
        pd.Series of VWAP values aligned to df.index.
    """
    vwap_values = np.empty(len(df))
    cum_tp_vol = 0.0
    cum_vol = 0.0
    prev_date = None

    for i, (idx, row) in enumerate(df.iterrows()):
        current_date = row["session_date"]
        if current_date != prev_date:
            # New session — reset accumulators
            cum_tp_vol = 0.0
            cum_vol = 0.0
            prev_date = current_date

        cum_tp_vol += float(row["typical_price"]) * float(row["volume"])
        cum_vol += float(row["volume"])

        vwap_values[i] = cum_tp_vol / cum_vol if cum_vol > 0 else float(row["typical_price"])

    return pd.Series(vwap_values, index=df.index, name="vwap")


def compute_vwap_zscore(
    close: pd.Series,
    vwap: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute the VWAP z-score:
        Z_vwap = (Close - VWAP) / StdDev(Close - VWAP, window)

    Args:
        close: Series of closing prices.
        vwap: Series of VWAP values.
        window: Rolling standard deviation window in bars.

    Returns:
        pd.Series of z-scores (NaN for first `window` bars).
    """
    deviation = close - vwap
    rolling_std = deviation.rolling(window=window, min_periods=window).std()
    zscore = deviation / rolling_std.replace(0, np.nan)
    zscore.name = "vwap_zscore"
    return zscore


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EMA with alpha=1/period).

    Formula:
        RS = AvgGain / AvgLoss
        RSI = 100 - (100 / (1 + RS))

    Args:
        close: Series of closing prices.
        period: RSI lookback period (e.g. 14).

    Returns:
        pd.Series of RSI values in [0, 100], NaN for first `period` bars.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing: EMA with span = period
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = "rsi"
    return rsi


# ─────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────

class IndicatorTool(BaseTool):
    """
    Computes VWAP, VWAP z-score, VWAP bands (±1σ, ±2σ), and RSI from
    OHLCV data provided as JSON (output of MarketDataTool).

    Returns JSON with all original bar columns plus:
      - vwap          : Session-anchored VWAP (resets each trading day)
      - vwap_zscore   : (Close - VWAP) / StdDev(Close - VWAP, N)
      - vwap_upper_1  : VWAP + 1σ
      - vwap_lower_1  : VWAP - 1σ
      - vwap_upper_2  : VWAP + 2σ
      - vwap_lower_2  : VWAP - 2σ
      - rsi           : RSI(period)
    """

    name: str = "Indicator Calculator"
    description: str = (
        "Computes VWAP (session-anchored, resets daily), VWAP z-score, "
        "VWAP upper/lower bands (±1σ and ±2σ), and RSI from raw OHLCV data. "
        "Input must be the JSON output from MarketDataTool. "
        "Returns an enriched JSON with all indicator columns appended. "
        "Use this after MarketDataTool and before BacktestTool."
    )
    args_schema: Type[BaseModel] = IndicatorInput

    def _run(
        self,
        market_data_json: str,
        rsi_period: int = 14,
        vwap_std_window: int = 20,
        vwap_band_multipliers: list[float] | None = None,
    ) -> str:
        """
        Compute all indicators and return enriched bar data as JSON.

        Args:
            market_data_json: JSON from MarketDataTool.
            rsi_period: RSI period (default 14).
            vwap_std_window: Rolling std window for VWAP bands (default 20).
            vwap_band_multipliers: Std multipliers (default [1.0, 2.0]).

        Returns:
            JSON string with indicators appended to each bar record.
        """
        if vwap_band_multipliers is None:
            vwap_band_multipliers = [1.0, 2.0]

        # ── Parse input ─────────────────────────────────────────
        try:
            payload: dict = json.loads(market_data_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON input: {exc}"})

        if "error" in payload:
            return market_data_json  # pass-through upstream errors

        records: list[dict] = payload.get("full_data", [])
        if not records:
            return json.dumps({"error": "full_data is empty."})

        # ── Build DataFrame ──────────────────────────────────────
        df = pd.DataFrame(records)
        dt_col = "datetime" if "datetime" in df.columns else df.columns[0]
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col).sort_index()

        for col in ["open", "high", "low", "close", "volume", "typical_price"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["close", "volume", "typical_price"], inplace=True)

        if len(df) < max(rsi_period, vwap_std_window) + 5:
            return json.dumps(
                {"error": f"Insufficient bars ({len(df)}) for indicator computation."}
            )

        # ── Compute indicators ───────────────────────────────────
        df["vwap"] = compute_session_vwap(df)
        df["vwap_zscore"] = compute_vwap_zscore(df["close"], df["vwap"], vwap_std_window)
        df["rsi"] = compute_rsi(df["close"], rsi_period)

        # Rolling std of (close - vwap) for bands
        deviation = df["close"] - df["vwap"]
        rolling_std = deviation.rolling(window=vwap_std_window, min_periods=vwap_std_window).std()

        for mult in vwap_band_multipliers:
            label = str(mult).replace(".", "_")
            df[f"vwap_upper_{label}"] = df["vwap"] + mult * rolling_std
            df[f"vwap_lower_{label}"] = df["vwap"] - mult * rolling_std

        # ── Compute summary statistics ───────────────────────────
        valid = df.dropna(subset=["rsi", "vwap_zscore"])
        summary = {
            "total_bars": int(len(df)),
            "valid_bars_with_indicators": int(len(valid)),
            "rsi_period": rsi_period,
            "vwap_std_window": vwap_std_window,
            "vwap_band_multipliers": vwap_band_multipliers,
            "rsi_stats": {
                "mean": round(float(valid["rsi"].mean()), 2),
                "min":  round(float(valid["rsi"].min()), 2),
                "max":  round(float(valid["rsi"].max()), 2),
                "pct_oversold_30":   round(float((valid["rsi"] < 30).mean() * 100), 2),
                "pct_overbought_70": round(float((valid["rsi"] > 70).mean() * 100), 2),
            },
            "vwap_zscore_stats": {
                "mean": round(float(valid["vwap_zscore"].mean()), 4),
                "std":  round(float(valid["vwap_zscore"].std()), 4),
                "pct_below_neg1":  round(float((valid["vwap_zscore"] < -1).mean() * 100), 2),
                "pct_above_pos1":  round(float((valid["vwap_zscore"] > 1).mean() * 100), 2),
                "pct_below_neg2":  round(float((valid["vwap_zscore"] < -2).mean() * 100), 2),
                "pct_above_pos2":  round(float((valid["vwap_zscore"] > 2).mean() * 100), 2),
            },
        }

        # ── Serialise output ─────────────────────────────────────
        df_out = df.copy()
        df_out.index = df_out.index.astype(str)
        df_out = df_out.replace({float("nan"): None, float("inf"): None, float("-inf"): None})

        enriched_records = df_out.reset_index().rename(
            columns={"index": "datetime"}
        ).to_dict(orient="records")

        result = {
            "ticker": payload.get("ticker", "N/A"),
            "interval": payload.get("interval", "N/A"),
            "summary": summary,
            "indicator_columns": [
                "vwap", "vwap_zscore", "rsi",
                *[f"vwap_upper_{str(m).replace('.','_')}" for m in vwap_band_multipliers],
                *[f"vwap_lower_{str(m).replace('.','_')}" for m in vwap_band_multipliers],
            ],
            "full_data": enriched_records,
            "sample_tail": enriched_records[-10:],
        }

        return json.dumps(result, default=str)
