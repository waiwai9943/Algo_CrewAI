"""
market_data_tool.py
-------------------
CrewAI custom tool: fetches 5-minute OHLCV bars for NQ or CL futures
using yfinance, then returns JSON-serialized data for downstream agents.

Supported tickers:
  NQ Futures  → "NQ=F"
  CL Futures  → "CL=F"
"""

from __future__ import annotations

import json
from typing import Type

import pandas as pd
import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Input Schema
# ─────────────────────────────────────────────

class MarketDataInput(BaseModel):
    """Input schema for MarketDataTool."""

    ticker: str = Field(
        default="NQ=F",
        description=(
            "Futures ticker symbol. Use 'NQ=F' for E-mini Nasdaq-100 or "
            "'CL=F' for WTI Crude Oil. Other examples: 'ES=F', 'GC=F'."
        ),
    )
    period: str = Field(
        default="60d",
        description=(
            "Lookback period for yfinance. yfinance supports intraday data "
            "up to 60 days back. Examples: '30d', '60d'."
        ),
    )
    interval: str = Field(
        default="5m",
        description=(
            "Bar interval. For intraday, use '1m', '2m', '5m', '15m', '30m', '60m'. "
            "Note: intervals <1h are only available for the last 60 days."
        ),
    )
    session_open_hour: int = Field(
        default=9,
        description="Session open hour (ET) for VWAP anchor reset. NQ RTH = 9, CL = 9.",
    )
    session_open_minute: int = Field(
        default=30,
        description="Session open minute (ET). NQ RTH starts at 09:30.",
    )
    output_file: str = Field(
        default="market_data.json",
        description="Local JSON file path where the full OHLCV data will be saved.",
    )


# ─────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────

class MarketDataTool(BaseTool):
    """
    Fetches 5-minute (or other interval) OHLCV bars for NQ or CL futures
    from Yahoo Finance, saves the full dataset to a local JSON file,
    and returns a summary for quantitative analysis.
    """

    name: str = "Market Data Fetcher"
    description: str = (
        "Fetches historical OHLCV (Open, High, Low, Close, Volume) bar data "
        "for NQ (E-mini Nasdaq-100 Futures) or CL (Crude Oil Futures) from "
        "Yahoo Finance. Saves the full data to a local file (e.g. 'market_data.json') "
        "and returns a summary JSON. Use this tool first to obtain raw price data."
    )
    args_schema: Type[BaseModel] = MarketDataInput

    def _run(
        self,
        ticker: str = "NQ=F",
        period: str = "60d",
        interval: str = "5m",
        session_open_hour: int = 9,
        session_open_minute: int = 30,
        output_file: str = "market_data.json",
    ) -> str:
        """
        Download OHLCV data via yfinance, save to file, and return summary JSON.
        """
        try:
            raw: pd.DataFrame = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            return json.dumps({"error": f"yfinance download failed: {exc}"})

        if raw.empty:
            return json.dumps({"error": f"No data returned for {ticker}."})

        # Flatten MultiIndex columns if present (yfinance sometimes returns them)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Normalise column names to lowercase
        raw.columns = [c.lower() for c in raw.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(raw.columns)):
            return json.dumps(
                {"error": f"Missing columns. Got: {list(raw.columns)}"}
            )

        df = raw[["open", "high", "low", "close", "volume"]].copy()
        df.dropna(inplace=True)

        # Tag each bar with its session date (for VWAP reset)
        df.index = pd.to_datetime(df.index)
        df["session_date"] = df.index.date

        # Compute intrabar typical price (used for VWAP in indicator tool)
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

        # Serialise index to string for JSON compatibility
        df_out = df.copy()
        df_out.index = df_out.index.astype(str)
        df_out["session_date"] = df_out["session_date"].astype(str)

        records = df_out.reset_index().rename(
            columns={"Datetime": "datetime", "index": "datetime"}
        ).to_dict(orient="records")

        # Save full data to local file to avoid LLM context size limits
        payload = {
            "ticker": ticker,
            "interval": interval,
            "period": period,
            "session_open": f"{session_open_hour:02d}:{session_open_minute:02d} ET",
            "total_bars": len(records),
            "date_range": {
                "first": str(df.index[0]),
                "last": str(df.index[-1]),
            },
            "columns": ["datetime", "open", "high", "low", "close", "volume",
                        "session_date", "typical_price"],
            "full_data": records,
        }
        
        try:
            with open(output_file, "w") as f:
                json.dump(payload, f, default=str)
        except Exception as e:
            return json.dumps({"error": f"Failed to save data to {output_file}: {e}"})

        result = {
            "ticker": ticker,
            "interval": interval,
            "period": period,
            "session_open": f"{session_open_hour:02d}:{session_open_minute:02d} ET",
            "total_bars": len(records),
            "date_range": {
                "first": str(df.index[0]),
                "last": str(df.index[-1]),
            },
            "saved_to": output_file,
            "sample_tail": records[-10:],   # last 10 bars for quick inspection
        }

        return json.dumps(result, default=str)
