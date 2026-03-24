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
    Zwraca liste slownikow: {symbol, name, price, expiry, open_interest}
    posortowanych po dacie wygasniecia.
    """
    import yfinance as yf
    from datetime import date, timedelta

    info = COMMODITY_CONTRACT_INFO.get(commodity_name)
    if not info:
        return []

    prefix   = info["prefix"]
    exchange = info["exchange"]
    months   = info["months"]

    today      = date.today()
    results    = []
    years_back = 1

    for year_offset in range(-years_back, years_ahead + 1):
        yr = today.year + year_offset
        yr2 = str(yr)[-2:]
        for month_code in months:
            symbol = f"{prefix}{month_code}{yr2}.{exchange}"
            try:
                t    = yf.Ticker(symbol)
                inf  = t.info
                price = inf.get("regularMarketPrice")
                expiry = inf.get("expireIsoDate", "")
                oi     = inf.get("openInterest", 0)
                if price and price > 0 and expiry:
                    month_name = MONTH_CODES.get(month_code, ("?", 0))[0]
                    results.append({
                        "symbol":        symbol,
                        "name":          f"{prefix} {month_name}-{yr} ({month_code}{yr2})",
                        "price":         price,
                        "expiry":        expiry[:10],
                        "open_interest": oi or 0,
                    })
            except Exception:
                pass

    results.sort(key=lambda x: x["expiry"])
    return results


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
        date      = df.index[i]
        bar_open  = float(df["Open"].iloc[i])
        bar_high  = float(df["High"].iloc[i])
        bar_low   = float(df["Low"].iloc[i])
        bar_close = float(df["Close"].iloc[i])

        # ══════════════════════════════════════════════════════════════════════
        # LOGIKA GAPOW
        #
        # Gap w gore (bar_open > prev_close):
        #   Cena otwiera sie wyzej niz zamknela — jezeli Open przebija przez
        #   poziomy TP, wszystkie takie TP realizuja sie po cenie Open (nie po
        #   swojej indywidualnej cenie TP). To realistyczne — order TP nie
        #   zdazyл sie wypelnic przed otwarciem nowej sesji.
        #
        # Gap w dol (bar_open < prev_close):
        #   Cena otwiera sie nizej — jezeli Open przeskakuje przez poziomy
        #   dokupowania, wszystkie aktywowane zakupy realizuja sie po cenie
        #   Open (jeden fill po gorzej cenie). Realistyczne odwzorowanie
        #   zlecen stop-limit przy gapie.
        #
        # Bez gapu: normalna logika intrabar (High dla TP, Low dla wejsc).
        # ══════════════════════════════════════════════════════════════════════

        # ── Krok 1a: TP przez gap w gore ─────────────────────────────────────
        # Jezeli Open > TP danego kontraktu — zamknij po cenie Open (nie TP)
        still_open: List[Trade] = []
        for trade in open_trades:
            if bar_open >= trade.tp_price:
                # Gap w gore przeskoczyl TP — fill po cenie otwarcia
                fill_price = bar_open
                trade.exit_date  = date
                trade.exit_price = fill_price
                trade.pnl        = (fill_price - trade.entry_price) * qty_per_entry
                trade.days_open  = (date - trade.entry_date).days
                trade.closed     = True
                cumulative_pnl  += trade.pnl
                total_pnl       += trade.pnl
            else:
                still_open.append(trade)

        open_trades = still_open

        # ── Krok 1b: TP normalny (intrabar — High dotyka TP) ─────────────────
        still_open2: List[Trade] = []
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
                still_open2.append(trade)

        open_trades = still_open2

        # ── Krok 2: Aktualizuj last_entry_price po TP ────────────────────────
        if len(open_trades) == 0:
            last_entry_price = None
        else:
            last_entry_price = min(t.entry_price for t in open_trades)

        # ── Krok 3: Wejscia — gap w dol + normalne intrabar ──────────────────
        #
        # Kolejnosc sprawdzania:
        #   1. Czy jest gap w dol? (Open < last_entry - step)
        #      -> policz ile poziomow przeskoczono, kup WSZYSTKIE po cenie Open
        #      -> ustaw last_entry_price na najnizszy aktywowany poziom
        #   2. Czy Low spada dalej ponizej last_entry - step? (normalne ruchy)
        #      -> kup kolejne poziomy po ich indywidualnych cenach
        #
        # Dzieki temu gap powoduje jeden fill po cenie Open dla wszystkich
        # przeskoczonych poziomow, a potem normalne dokupowanie przy dalszym
        # spadku w ciagu sesji.

        if bar_low < entry_threshold:

            # -- Pierwsze wejscie (brak pozycji) --
            if len(open_trades) == 0:
                fill_px = bar_open if bar_open < entry_threshold else bar_close
                tp_px   = fill_px + take_profit
                trade = Trade(level=1, entry_date=date, entry_price=fill_px, tp_price=tp_px)
                open_trades.append(trade)
                trades.append(trade)
                last_entry_price = fill_px

            # -- Dokupowanie (jestesmy juz w pozycji) --
            if len(open_trades) > 0 and last_entry_price is not None:

                # GAP W DOL: Open przeskoczyl przez co najmniej jeden poziom
                # Kup wszystkie przeskoczone poziomy po cenie Open
                if bar_open < last_entry_price - pyramid_step:
                    fill_px      = bar_open
                    tp_px        = fill_px + take_profit  # TP od rzeczywistej ceny kupna
                    next_trigger = last_entry_price - pyramid_step
                    while next_trigger >= bar_open:
                        trade = Trade(
                            level       = len(open_trades) + 1,
                            entry_date  = date,
                            entry_price = fill_px,   # fill po cenie Open
                            tp_price    = tp_px,     # TP = fill + take_profit
                        )
                        open_trades.append(trade)
                        trades.append(trade)
                        last_entry_price = next_trigger  # sledz logiczny poziom
                        next_trigger     = last_entry_price - pyramid_step

                # NORMALNE INTRABAR: Low spada przez kolejne poziomy
                # Kup kazdy poziom po jego indywidualnej cenie
                next_trigger = last_entry_price - pyramid_step
                while bar_low <= next_trigger:
                    tp_px = next_trigger + take_profit
                    trade = Trade(
                        level       = len(open_trades) + 1,
                        entry_date  = date,
                        entry_price = next_trigger,
                        tp_price    = tp_px,
                    )
                    open_trades.append(trade)
                    trades.append(trade)
                    last_entry_price = next_trigger
                    next_trigger     = last_entry_price - pyramid_step

        # ── Krok 4: Statystyki dzienne ────────────────────────────────────────
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
