"""
backtester.py
Silnik backtestu strategii Pyramid Long.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Struktury danych
# ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    level:        int
    entry_date:   object
    entry_price:  float
    tp_price:     float
    exit_date:    object = None
    exit_price:   object = None
    pnl:          float  = 0.0
    days_open:    int    = 0
    closed:       bool   = False


@dataclass
class BacktestResult:
    trades:               List[Trade]
    total_pnl:            float
    total_trades:         int
    winning_trades:       int
    losing_trades:        int
    open_trades:          int
    max_concurrent:       int
    max_capital_needed:   float
    win_rate:             float
    avg_days_open:        float
    equity_curve:         pd.Series
    daily_open_contracts: pd.Series
    params:               dict


# ─────────────────────────────────────────────────────────────
# Lista surowcow
# ─────────────────────────────────────────────────────────────

COMMODITY_SYMBOLS = {
    "Corn (ZC)":        "ZC=F",
    "Wheat (ZW)":       "ZW=F",
    "Soybeans (ZS)":    "ZS=F",
    "Crude Oil (CL)":   "CL=F",
    "Natural Gas (NG)": "NG=F",
    "Gold (GC)":        "GC=F",
    "Silver (SI)":      "SI=F",
    "Copper (HG)":      "HG=F",
    "Cotton (CT)":      "CT=F",
    "Sugar (SB)":       "SB=F",
    "Coffee (KC)":      "KC=F",
    "S&P 500 (ES)":     "ES=F",
    "Nasdaq (NQ)":      "NQ=F",
}


# ─────────────────────────────────────────────────────────────
# Pobieranie danych
# ─────────────────────────────────────────────────────────────

def fetch_data(symbol: str, start, end) -> pd.DataFrame:
    try:
        df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Blad pobierania danych dla {symbol}: {e}")


# ─────────────────────────────────────────────────────────────
# Silnik backtestu
# ─────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    entry_threshold: float,
    pyramid_step: float,
    take_profit: float,
    margin_per_contract: float,
    qty_per_entry: int = 1,
) -> BacktestResult:

    trades: List[Trade] = []
    open_trades: List[Trade] = []
    last_entry_price: Optional[float] = None
    total_pnl: float = 0.0
    max_concurrent: int = 0
    cumulative_pnl: float = 0.0

    daily_open = pd.Series(0, index=df.index, dtype=int)
    equity_curve = pd.Series(0.0, index=df.index)

    for i in range(len(df)):
        date = df.index[i]
        bar_high  = float(df["High"].iloc[i])
        bar_low   = float(df["Low"].iloc[i])
        bar_close = float(df["Close"].iloc[i])

        # Sprawdz TP dla otwartych pozycji
        still_open: List[Trade] = []
        for trade in open_trades:
            if bar_high >= trade.tp_price:
                trade.exit_date  = date
                trade.exit_price = trade.tp_price
                trade.pnl        = (trade.tp_price - trade.entry_price) * qty_per_entry
                trade.days_open  = (date - trade.entry_date).days
                trade.closed     = True
                cumulative_pnl  += trade.pnl
                total_pnl       += trade.pnl
            else:
                still_open.append(trade)

        open_trades = still_open

        if len(open_trades) == 0:
            last_entry_price = None

        # Pierwsze wejscie
        if len(open_trades) == 0 and bar_low < entry_threshold:
            tp_px = bar_close + take_profit
            trade = Trade(
                level       = 1,
                entry_date  = date,
                entry_price = bar_close,
                tp_price    = tp_px,
            )
            open_trades.append(trade)
            trades.append(trade)
            last_entry_price = bar_close

        # Kolejne wejscia (pyramiding)
        elif len(open_trades) > 0 and last_entry_price is not None:
            next_trigger = last_entry_price - pyramid_step
            if bar_low <= next_trigger:
                tp_px = bar_close + take_profit
                trade = Trade(
                    level       = len(open_trades) + 1,
                    entry_date  = date,
                    entry_price = bar_close,
                    tp_price    = tp_px,
                )
                open_trades.append(trade)
                trades.append(trade)
                last_entry_price = bar_close

        # Statystyki dzienne
        current_open = len(open_trades)
        daily_open.iloc[i] = current_open
        if current_open > max_concurrent:
            max_concurrent = current_open
        equity_curve.iloc[i] = cumulative_pnl

    # Statystyki koncowe
    closed_trades = [t for t in trades if t.closed]
    open_remaining = [t for t in trades if not t.closed]
    winning = [t for t in closed_trades if t.pnl > 0]
    losing  = [t for t in closed_trades if t.pnl <= 0]

    win_rate  = len(winning) / len(closed_trades) * 100 if closed_trades else 0.0
    days_list = [t.days_open for t in closed_trades]
    avg_days  = float(np.mean(days_list)) if days_list else 0.0
    max_capital = float(max_concurrent) * margin_per_contract

    return BacktestResult(
        trades               = trades,
        total_pnl            = total_pnl,
        total_trades         = len(closed_trades),
        winning_trades       = len(winning),
        losing_trades        = len(losing),
        open_trades          = len(open_remaining),
        max_concurrent       = max_concurrent,
        max_capital_needed   = max_capital,
        win_rate             = win_rate,
        avg_days_open        = avg_days,
        equity_curve         = equity_curve,
        daily_open_contracts = daily_open,
        params               = {
            "entry_threshold":     entry_threshold,
            "pyramid_step":        pyramid_step,
            "take_profit":         take_profit,
            "margin_per_contract": margin_per_contract,
            "qty_per_entry":       qty_per_entry,
        },
    )


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "Poziom":        t.level,
            "Data wejscia":  str(t.entry_date)[:10] if t.entry_date else "",
            "Cena wejscia":  round(t.entry_price, 4),
            "TP":            round(t.tp_price, 4),
            "Data wyjscia":  str(t.exit_date)[:10] if t.exit_date else "OTWARTE",
            "Cena wyjscia":  round(t.exit_price, 4) if isinstance(t.exit_price, (int, float)) else None,
            "PnL ($)":       round(t.pnl, 2) if t.closed else None,
            "Dni otwarcia":  t.days_open if t.closed else None,
            "Status":        "Zamkniete" if t.closed else "Otwarte",
        })
    return pd.DataFrame(rows)
