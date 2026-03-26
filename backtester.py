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
    exit_date:         object = None
    exit_price:        object = None
    pnl:               float  = 0.0
    days_open:         int    = 0
    closed:            bool   = False
    entry_commission:  float  = 0.0   # commission paid on entry side
    exit_commission:   float  = 0.0   # commission paid on exit side


@dataclass
class BacktestResult:
    trades:                       List[Trade]
    total_pnl:                    float
    total_trades:                 int
    winning_trades:               int
    losing_trades:                int
    open_trades:                  int
    max_concurrent:               int
    max_capital_needed:           float
    max_capital_with_unrealized:  float
    win_rate:                     float
    avg_days_open:                float
    equity_curve:                 pd.Series   # realized PnL only (cumulative)
    balance_curve:                pd.Series   # realized + unrealized (mark-to-market at Close)
    daily_open_contracts:         pd.Series
    total_commission:             float
    params:                       dict


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

# Mapowanie surowca -> prefiks symbolu + gielda dla konkretnych kontraktow
COMMODITY_CONTRACT_INFO = {
    "Corn (ZC)":        {"prefix": "ZC", "exchange": "CBT",  "months": "FHKNUZ",  "unit": "cents/bushel",  "contract_size": 5000,  "tick": 0.25, "tick_value": 12.50},
    "Wheat (ZW)":       {"prefix": "ZW", "exchange": "CBT",  "months": "HKNUZ",   "unit": "cents/bushel",  "contract_size": 5000,  "tick": 0.25, "tick_value": 12.50},
    "Soybeans (ZS)":    {"prefix": "ZS", "exchange": "CBT",  "months": "FHKNQUX", "unit": "cents/bushel",  "contract_size": 5000,  "tick": 0.25, "tick_value": 12.50},
    "Crude Oil (CL)":   {"prefix": "CL", "exchange": "NYM",  "months": "FGHJKMNQUVXZ", "unit": "USD/bbl", "contract_size": 1000,  "tick": 0.01, "tick_value": 10.00},
    "Natural Gas (NG)": {"prefix": "NG", "exchange": "NYM",  "months": "FGHJKMNQUVXZ", "unit": "USD/MMBtu","contract_size": 10000, "tick": 0.001,"tick_value": 10.00},
    "Gold (GC)":        {"prefix": "GC", "exchange": "CMX",  "months": "GJMQVZ",  "unit": "USD/oz",        "contract_size": 100,   "tick": 0.10, "tick_value": 10.00},
    "Silver (SI)":      {"prefix": "SI", "exchange": "CMX",  "months": "HKNUZ",   "unit": "cents/oz",      "contract_size": 5000,  "tick": 0.005,"tick_value": 25.00},
    "Copper (HG)":      {"prefix": "HG", "exchange": "CMX",  "months": "HKNUZ",   "unit": "cents/lb",      "contract_size": 25000, "tick": 0.0005,"tick_value":12.50},
    "S&P 500 (ES)":     {"prefix": "ES", "exchange": "CME",  "months": "HMUZ",    "unit": "USD/index",     "contract_size": 50,    "tick": 0.25, "tick_value": 12.50},
    "Nasdaq (NQ)":      {"prefix": "NQ", "exchange": "CME",  "months": "HMUZ",    "unit": "USD/index",     "contract_size": 20,    "tick": 0.25, "tick_value":  5.00},
}

# Mapowanie litery miesiaca na nazwe
MONTH_CODES = {
    "F": ("Sty", 1), "G": ("Lut", 2), "H": ("Mar", 3), "J": ("Kwi", 4),
    "K": ("Maj", 5), "M": ("Cze", 6), "N": ("Lip", 7), "Q": ("Sie", 8),
    "U": ("Wrz", 9), "V": ("Paz",10), "X": ("Lis",11), "Z": ("Gru",12),
}


def get_available_contracts(commodity_name: str, years_ahead: int = 3) -> list:
    """
    Pobiera dostepne konkretne kontrakty dla danego surowca z Yahoo Finance.
    Uzywa batch download dla szybkosci zamiast pojedynczych requestow.
    """
    import yfinance as yf
    from datetime import date

    info = COMMODITY_CONTRACT_INFO.get(commodity_name)
    if not info:
        return []

    prefix   = info["prefix"]
    exchange = info["exchange"]
    months   = info["months"]
    today    = date.today()

    # Buduj liste kandydatow (tylko biezacy rok i naprzod)
    candidates = []
    for year_offset in range(0, years_ahead + 1):
        yr  = today.year + year_offset
        yr2 = str(yr)[-2:]
        for month_code in months:
            symbol = f"{prefix}{month_code}{yr2}.{exchange}"
            month_num = MONTH_CODES.get(month_code, ("?", 0))[1]
            # Pomin juz przeszle miesiace w biezacym roku
            if yr == today.year and month_num < today.month:
                continue
            month_name = MONTH_CODES.get(month_code, ("?", 0))[0]
            candidates.append({
                "symbol": symbol,
                "name":   f"{prefix} {month_name}-{yr} ({month_code}{yr2})",
            })

    if not candidates:
        return []

    # Batch download ostatnich 5 dni — jezeli symbol ma dane to jest aktywny
    symbols = [c["symbol"] for c in candidates]
    try:
        raw = yf.download(
            symbols,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception:
        return []

    results = []
    for c in candidates:
        sym = c["symbol"]
        try:
            # Pobierz ostatnia cene zamkniecia
            if len(symbols) == 1:
                closes = raw["Close"].dropna()
            else:
                closes = raw[sym]["Close"].dropna() if sym in raw else pd.Series(dtype=float)

            if closes.empty:
                continue
            price = float(closes.iloc[-1])
            if price <= 0:
                continue

            # Wylicz przyblizony expiry z symbolu (bez dodatkowego HTTP request)
            # Format: ZCK26.CBT -> month=K (Maj), year=26
            month_code = sym[len(prefix)]
            yr2_str    = sym[len(prefix)+1:len(prefix)+3]
            try:
                yr_full    = 2000 + int(yr2_str)
                month_num  = MONTH_CODES.get(month_code, ("?", 1))[1]
                expiry     = f"{yr_full}-{month_num:02d}-14"
            except Exception:
                expiry = ""

            results.append({
                "symbol":        sym,
                "name":          c["name"],
                "price":         round(price, 4),
                "expiry":        expiry,
                "open_interest": 0,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["expiry"])
    return results


# ─────────────────────────────────────────────────────────────
# Pobieranie danych
# ─────────────────────────────────────────────────────────────

TIMEFRAME_INTERVALS = {
    "1 hour (1h)":   "1h",
    "Daily (1d)":    "1d",
    "Weekly (1w)":   "1wk",
}

# Yahoo Finance limits for each interval
TIMEFRAME_LIMITS = {
    "1h":  "⚠ Yahoo Finance: max ~60 days of 1h data.",
    "1d":  None,
    "1wk": None,
}


def fetch_data(symbol: str, start, end, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.
    interval: '1h', '1d', '1wk'
    Note: Yahoo limits 1h data to ~60 days regardless of start/end.
    """
    try:
        df = yf.download(
            symbol, start=start, end=end,
            interval=interval, auto_adjust=True, progress=False
        )
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
    point_value: float = 1.0,
    commission_per_side: float = 0.0,
    direction: str = "long",
) -> BacktestResult:
    """
    direction: 'long' or 'short'

    LONG  — buy when price drops below threshold, add on further drops,
            TP when price rises +take_profit above each entry.

    SHORT — sell short when price rises above threshold, add on further rises,
            TP when price falls -take_profit below each entry.
            Mirror image of long strategy.

    commission_per_side: USD per contract per side. Round-trip = 2x.
    point_value: USD value of 1 price point.
    """

    is_long = direction == "long"

    trades: List[Trade] = []
    open_trades: List[Trade] = []
    last_entry_price: Optional[float] = None
    total_pnl: float = 0.0
    total_comm: float = 0.0
    max_concurrent: int = 0
    cumulative_pnl: float = 0.0

    ec_per_entry = commission_per_side * qty_per_entry  # entry commission, constant
    max_capital_with_unrealized: float = 0.0

    daily_open    = pd.Series(0,   index=df.index, dtype=int)
    equity_curve  = pd.Series(0.0, index=df.index)
    balance_curve = pd.Series(0.0, index=df.index)  # realized + unrealized at Close

    for i in range(len(df)):
        date      = df.index[i]
        bar_open  = float(df["Open"].iloc[i])
        bar_high  = float(df["High"].iloc[i])
        bar_low   = float(df["Low"].iloc[i])
        bar_close = float(df["Close"].iloc[i])

        # ── STEP 1a: Gap TP (open gaps through TP level) ─────────────────────
        still_open: List[Trade] = []
        for trade in open_trades:
            hit = (is_long and bar_open >= trade.tp_price) or (not is_long and bar_open <= trade.tp_price)
            if hit:
                fill_price            = bar_open
                trade.exit_date       = date
                trade.exit_price      = fill_price
                trade.exit_commission = ec_per_entry
                gross = (fill_price - trade.entry_price) * qty_per_entry * point_value if is_long else (trade.entry_price - fill_price) * qty_per_entry * point_value
                trade.pnl       = gross - trade.entry_commission - trade.exit_commission
                trade.days_open = (date - trade.entry_date).days
                trade.closed    = True
                cumulative_pnl += trade.pnl
                total_pnl      += trade.pnl
                total_comm     += trade.entry_commission + trade.exit_commission
            else:
                still_open.append(trade)
        open_trades = still_open

        # ── STEP 1b: Intrabar TP ─────────────────────────────────────────────
        still_open2: List[Trade] = []
        for trade in open_trades:
            hit = (is_long and bar_high >= trade.tp_price) or (not is_long and bar_low <= trade.tp_price)
            if hit:
                fill_price            = trade.tp_price
                trade.exit_date       = date
                trade.exit_price      = fill_price
                trade.exit_commission = ec_per_entry
                gross = (fill_price - trade.entry_price) * qty_per_entry * point_value if is_long else (trade.entry_price - fill_price) * qty_per_entry * point_value
                trade.pnl       = gross - trade.entry_commission - trade.exit_commission
                trade.days_open = (date - trade.entry_date).days
                trade.closed    = True
                cumulative_pnl += trade.pnl
                total_pnl      += trade.pnl
                total_comm     += trade.entry_commission + trade.exit_commission
            else:
                still_open2.append(trade)
        open_trades = still_open2

        # ── STEP 2: Update last_entry_price ──────────────────────────────────
        if len(open_trades) == 0:
            last_entry_price = None
        else:
            last_entry_price = min(t.entry_price for t in open_trades) if is_long else max(t.entry_price for t in open_trades)

        # ── STEP 3: New entries ───────────────────────────────────────────────
        if is_long:
            if bar_low < entry_threshold:
                if len(open_trades) == 0:
                    fill_px = bar_open if bar_open < entry_threshold else bar_close
                    t = Trade(level=1, entry_date=date, entry_price=fill_px, tp_price=fill_px + take_profit, entry_commission=ec_per_entry)
                    cumulative_pnl -= ec_per_entry; total_comm += ec_per_entry
                    open_trades.append(t); trades.append(t)
                    last_entry_price = fill_px

                if len(open_trades) > 0 and last_entry_price is not None:
                    if bar_open < last_entry_price - pyramid_step:
                        fill_px      = bar_open
                        next_trigger = last_entry_price - pyramid_step
                        while next_trigger >= bar_open:
                            t = Trade(level=len(open_trades)+1, entry_date=date, entry_price=fill_px, tp_price=fill_px + take_profit, entry_commission=ec_per_entry)
                            cumulative_pnl -= ec_per_entry; total_comm += ec_per_entry
                            open_trades.append(t); trades.append(t)
                            last_entry_price = next_trigger
                            next_trigger     = last_entry_price - pyramid_step
                    next_trigger = last_entry_price - pyramid_step
                    while bar_low <= next_trigger:
                        t = Trade(level=len(open_trades)+1, entry_date=date, entry_price=next_trigger, tp_price=next_trigger + take_profit, entry_commission=ec_per_entry)
                        cumulative_pnl -= ec_per_entry; total_comm += ec_per_entry
                        open_trades.append(t); trades.append(t)
                        last_entry_price = next_trigger
                        next_trigger     = last_entry_price - pyramid_step
        else:
            if bar_high > entry_threshold:
                if len(open_trades) == 0:
                    fill_px = bar_open if bar_open > entry_threshold else bar_close
                    t = Trade(level=1, entry_date=date, entry_price=fill_px, tp_price=fill_px - take_profit, entry_commission=ec_per_entry)
                    cumulative_pnl -= ec_per_entry; total_comm += ec_per_entry
                    open_trades.append(t); trades.append(t)
                    last_entry_price = fill_px

                if len(open_trades) > 0 and last_entry_price is not None:
                    if bar_open > last_entry_price + pyramid_step:
                        fill_px      = bar_open
                        next_trigger = last_entry_price + pyramid_step
                        while next_trigger <= bar_open:
                            t = Trade(level=len(open_trades)+1, entry_date=date, entry_price=fill_px, tp_price=fill_px - take_profit, entry_commission=ec_per_entry)
                            cumulative_pnl -= ec_per_entry; total_comm += ec_per_entry
                            open_trades.append(t); trades.append(t)
                            last_entry_price = next_trigger
                            next_trigger     = last_entry_price + pyramid_step
                    next_trigger = last_entry_price + pyramid_step
                    while bar_high >= next_trigger:
                        t = Trade(level=len(open_trades)+1, entry_date=date, entry_price=next_trigger, tp_price=next_trigger - take_profit, entry_commission=ec_per_entry)
                        cumulative_pnl -= ec_per_entry; total_comm += ec_per_entry
                        open_trades.append(t); trades.append(t)
                        last_entry_price = next_trigger
                        next_trigger     = last_entry_price + pyramid_step

        # ── Krok 4: Statystyki dzienne ────────────────────────────────────────
        current_open = len(open_trades)
        daily_open.iloc[i] = current_open
        if current_open > max_concurrent:
            max_concurrent = current_open
        equity_curve.iloc[i] = cumulative_pnl

        # Balance curve = realized PnL + unrealized PnL at bar Close
        # Unrealized = (close - entry) × qty × point_value for each open trade
        # For Short: unrealized = (entry - close) × qty × point_value
        unrealized_at_close = 0.0
        for ot in open_trades:
            if is_long:
                unrealized_at_close += (bar_close - ot.entry_price) * qty_per_entry * point_value
            else:
                unrealized_at_close += (ot.entry_price - bar_close) * qty_per_entry * point_value
        balance_curve.iloc[i] = cumulative_pnl + unrealized_at_close

        # ── Krok 5: Max capital including unrealized loss ─────────────────────
        # Margin capital + unrealized loss on open positions at worst bar price.
        # For LONG:  worst price on bar = Low  (positions losing value)
        # For SHORT: worst price on bar = High (positions losing value)
        # Unrealized loss per contract = abs(entry_price - worst_price) × point_value
        # This represents how much ADDITIONAL capital you need beyond margin
        # to cover the floating loss at the worst intrabar moment.
        if open_trades:
            worst_price = bar_low if is_long else bar_high
            unrealized_loss = 0.0
            for ot in open_trades:
                if is_long:
                    loss = max(0.0, (ot.entry_price - worst_price) * qty_per_entry * point_value)
                else:
                    loss = max(0.0, (worst_price - ot.entry_price) * qty_per_entry * point_value)
                unrealized_loss += loss
            capital_this_bar = current_open * margin_per_contract + unrealized_loss
            if capital_this_bar > max_capital_with_unrealized:
                max_capital_with_unrealized = capital_this_bar

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
        trades                      = trades,
        total_pnl                   = total_pnl,
        total_trades                = len(closed_trades),
        winning_trades              = len(winning),
        losing_trades               = len(losing),
        open_trades                 = len(open_remaining),
        max_concurrent              = max_concurrent,
        max_capital_needed          = max_capital,
        max_capital_with_unrealized = max_capital_with_unrealized,
        win_rate                    = win_rate,
        avg_days_open               = avg_days,
        total_commission            = total_comm,
        equity_curve                = equity_curve,
        balance_curve               = balance_curve,
        daily_open_contracts        = daily_open,
        params               = {
            "entry_threshold":     entry_threshold,
            "pyramid_step":        pyramid_step,
            "take_profit":         take_profit,
            "margin_per_contract": margin_per_contract,
            "qty_per_entry":       qty_per_entry,
            "point_value":         point_value,
        },
    )


# ─────────────────────────────────────────────────────────────
# VOLUME PROFILE — strefy wsparcia / oporu
# ─────────────────────────────────────────────────────────────

def compute_volume_profile(df: pd.DataFrame, bins: int = 100):
    """
    Liczy Volume Profile dla podanego DataFrame.

    Metoda:
      - Dzieli zakres cen (Low..High) na 'bins' poziomow cenowych
      - Dla kazdego bara rozklada wolumen proporcjonalnie na ceny
        w przedziale [Low, High] tego bara (uproszczenie TPO)
      - Zwraca DataFrame z kolumnami: price_level, volume, pct
      - POC = cena z najwyzszym wolumenem
      - Value Area = zakres zawierajacy 70% calkowitego wolumenu wokol POC
    """
    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())
    if price_min >= price_max:
        return pd.DataFrame(), price_min, price_min, price_min

    levels = np.linspace(price_min, price_max, bins + 1)
    mid_prices = (levels[:-1] + levels[1:]) / 2
    vol_at_level = np.zeros(bins)

    for _, row in df.iterrows():
        bar_low   = float(row["Low"])
        bar_high  = float(row["High"])
        bar_vol   = float(row["Volume"]) if float(row["Volume"]) > 0 else 1.0
        bar_range = bar_high - bar_low
        if bar_range < 1e-9:
            # zero-range bar — przypisz do najblizszego levelu
            idx = np.searchsorted(levels, bar_low, side="right") - 1
            idx = min(max(idx, 0), bins - 1)
            vol_at_level[idx] += bar_vol
            continue
        # znajdz ktore poziomy wchodza w zakres tego bara
        mask = (mid_prices >= bar_low) & (mid_prices <= bar_high)
        n_levels = mask.sum()
        if n_levels == 0:
            idx = np.searchsorted(levels, (bar_low + bar_high) / 2, side="right") - 1
            idx = min(max(idx, 0), bins - 1)
            vol_at_level[idx] += bar_vol
        else:
            vol_at_level[mask] += bar_vol / n_levels

    total_vol = vol_at_level.sum()
    pct = vol_at_level / total_vol * 100 if total_vol > 0 else vol_at_level

    vp_df = pd.DataFrame({
        "price_level": mid_prices,
        "volume":      vol_at_level,
        "pct":         pct,
    })

    # POC — Point of Control
    poc_idx = int(np.argmax(vol_at_level))
    poc_price = float(mid_prices[poc_idx])

    # Value Area (70% wolumenu wokol POC)
    target_vol = total_vol * 0.70
    va_low_idx  = poc_idx
    va_high_idx = poc_idx
    accumulated = vol_at_level[poc_idx]

    while accumulated < target_vol:
        can_go_down = va_low_idx > 0
        can_go_up   = va_high_idx < bins - 1
        if not can_go_down and not can_go_up:
            break
        add_down = vol_at_level[va_low_idx - 1] if can_go_down else -1
        add_up   = vol_at_level[va_high_idx + 1] if can_go_up   else -1
        if add_down >= add_up:
            va_low_idx  -= 1
            accumulated += vol_at_level[va_low_idx]
        else:
            va_high_idx += 1
            accumulated += vol_at_level[va_high_idx]

    va_low  = float(mid_prices[va_low_idx])
    va_high = float(mid_prices[va_high_idx])

    return vp_df, poc_price, va_low, va_high


def find_support_zones(df: pd.DataFrame, bins: int = 200, top_n: int = 5, min_gap_pct: float = 0.03):
    """
    Znajdz N stref wsparcia/oporu jako lokalnych maksimow Volume Profile.

    Zwraca liste slownikow:
      { price, volume_pct, zone_low, zone_high, strength }
    posortowanych od najwyzszego wolumenu.

    min_gap_pct: minimalna odleglosc miedzy strefami jako % ceny (domyslnie 3%)
    """
    vp_df, poc, va_low, va_high = compute_volume_profile(df, bins=bins)
    if vp_df.empty:
        return [], poc, va_low, va_high

    prices  = vp_df["price_level"].values
    volumes = vp_df["volume"].values
    pcts    = vp_df["pct"].values
    total   = volumes.sum()

    # Znajdz lokalne maksima (kazdy punkt wyzszy od swoich 5 sasiadow)
    window = 5
    local_max_idx = []
    for i in range(window, len(volumes) - window):
        if volumes[i] == max(volumes[i-window:i+window+1]):
            local_max_idx.append(i)

    # Posortuj po wolumenie malejaco
    local_max_idx.sort(key=lambda i: volumes[i], reverse=True)

    # Wybierz top N z minimalnym odstepem miedzy strefami
    zones = []
    for idx in local_max_idx:
        price = prices[idx]
        # Sprawdz czy nie za blisko juz wybranej strefy
        too_close = any(abs(price - z["price"]) / price < min_gap_pct for z in zones)
        if too_close:
            continue
        # Szerokosc strefy = polowa odleglosci do sasiednich poziomow
        step = prices[1] - prices[0] if len(prices) > 1 else price * 0.01
        strength = round(pcts[idx], 2)
        zones.append({
            "price":      round(price, 4),
            "zone_low":   round(price - step * 3, 4),
            "zone_high":  round(price + step * 3, 4),
            "volume_pct": strength,
            "label":      f"Strefa {round(price, 2)}$ ({strength:.1f}% vol)",
        })
        if len(zones) >= top_n:
            break

    # Posortuj strefy od najnizszej ceny
    zones.sort(key=lambda z: z["price"])
    return zones, poc, va_low, va_high


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
