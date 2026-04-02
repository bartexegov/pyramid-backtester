"""
app.py — Pyramid Long Backtester
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import collections

from backtester import fetch_data, run_backtest, trades_to_dataframe, COMMODITY_SYMBOLS, COMMODITY_CONTRACT_INFO, TIMEFRAME_INTERVALS, TIMEFRAME_LIMITS, get_available_contracts, find_support_zones, compute_volume_profile, fetch_coinbase_products, fetch_coinbase_candles, COINBASE_FUTURES, COINBASE_GRANULARITY

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pyramid Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }
.sidebar-header { background: #1e293b; padding: 20px 20px 16px 20px; border-bottom: 1px solid #334155; }
.sidebar-logo { font-size: 1.25rem; font-weight: 700; color: #38bdf8 !important; letter-spacing: -0.02em; }
.sidebar-logo span { color: #94a3b8 !important; font-weight: 400; font-size:0.9rem; }
.sidebar-section { padding: 16px 20px 8px 20px; border-bottom: 1px solid #1e293b; }
.sidebar-section-title { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b !important; margin-bottom: 12px; }
.strategy-tag { display: inline-block; background: #0c4a6e; color: #38bdf8 !important; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 6px; border: 1px solid #0369a1; margin-bottom: 12px; }
[data-testid="stTabs"] [role="tab"] { font-size: 0.85rem; font-weight: 600; color: #64748b; padding: 8px 16px; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #38bdf8; border-bottom: 2px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def fmt_date(d) -> str:
    """Convert any date/timestamp/string to US format MM/DD/YYYY."""
    try:
        s = str(d)[:10]          # get YYYY-MM-DD part
        y, m, day = s.split("-")
        return f"{m}/{day}/{y}"
    except Exception:
        return str(d)[:10]


def info(tooltip_text: str) -> str:
    """Info icon using title= attribute - works in all Streamlit versions."""
    safe = tooltip_text.replace('"', '').replace("'", "").replace('<', '').replace('>', '')
    style = "display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:50%;background:#334155;color:#94a3b8;font-size:9px;font-weight:700;font-style:normal;cursor:help;margin-left:5px;vertical-align:middle;border:1px solid #475569"
    return f'<span title="{safe}" style="{style}">i</span>'


# Tooltip text for each metric card
METRIC_TOOLTIPS = {
    "Total PnL":         "Net profit/loss after deducting all commissions (entry + exit sides). Formula: sum of (exit_price - entry_price) × point_value - round_trip_commission for all closed contracts.",
    "Total commission":  "Total broker fees paid. Counts every buy AND every sell separately. Formula: (entries + closed_sells) × commission_per_side. Includes fees for open positions (entry side already paid).",
    "Total operations":  "Total number of broker interactions = Buys + Sells (TP). Each buy costs commission, each sell (TP) costs commission. Important: even open positions already paid entry commission.",
    "Closed trades":     "Number of contracts that hit their Take Profit and were sold. Each contract bought = 1 trade. Win = PnL > 0 after commission. Loss = PnL ≤ 0 (commission exceeded gross profit).",
    "Avg PnL / trade":   "Average net PnL per closed contract. Formula: Total PnL ÷ Closed trades. Negative means average commission cost exceeds average gross profit per trade.",
    "Max contracts":     "Peak number of simultaneously open contracts during the entire backtest period. Occurs at the deepest price drawdown.",
    "Max capital req.":  "Margin only: Max contracts × margin per contract. Does NOT include floating losses on open positions.",
    "Max capital (real)": "Real capital needed = margin + max unrealized loss at the worst intrabar price. This is the true minimum account size to survive the worst moment.",
    "Open positions":    "Contracts still open (TP not yet hit) at the end of the backtest period. Entry commission already paid, exit commission not yet paid.",
    "Avg days to TP":    "Average number of calendar days from entry to TP hit. Measures how long you typically wait for a position to become profitable.",
    "Period HIGH":       "Highest High price of any bar in the selected date range. Sourced from Yahoo Finance daily OHLCV data.",
    "Period LOW":        "Lowest Low price of any bar in the selected date range. Sourced from Yahoo Finance daily OHLCV data.",
}

def metric_card(label, value, sub="", positive=None):
    val_color = "#34d399" if positive is True else "#f87171" if positive is False else "#f1f5f9"
    tooltip = METRIC_TOOLTIPS.get(label, "")
    info_html = info(tooltip) if tooltip else ""
    s_card  = "background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px;text-align:center"
    s_label = "font-size:0.75rem;color:#94a3b8;font-weight:600;margin-bottom:6px;display:flex;align-items:center;justify-content:center;gap:4px"
    s_value = f"font-size:1.5rem;font-weight:700;color:{val_color};letter-spacing:-0.02em"
    s_sub   = "font-size:0.82rem;color:#cbd5e1;margin-top:5px;font-weight:500"
    return f'<div style="{s_card}"><div style="{s_label}">{label}{info_html}</div><div style="{s_value}">{value}</div><div style="{s_sub}">{sub}</div></div>'


# Tooltip text for optimization table columns
OPT_COL_TIPS = {
    "Step":        "Pyramid step — price drop required to add one more contract. Smaller step = more contracts bought = higher commission cost.",
    "TP":          "Take Profit — how many $ above entry price each contract is sold. Must exceed 2 × commission/side to be profitable.",
    "PnL":         "Net PnL after all commissions for this Step/TP combination. Formula: gross profit − (Total ops × commission/side).",
    "Buys":        "Total number of contracts bought (entries). Each buy triggers entry commission immediately.",
    "Sells (TP)":  "Contracts closed by hitting Take Profit. Each sell triggers exit commission. Sells ≤ Buys (open positions not yet sold).",
    "Open":        "Contracts still open at end of period — bought and paid entry commission, but TP not yet hit.",
    "Total ops":   "Buys + Sells = total broker interactions. Formula: Entries + Closed (TP). Multiply by commission/side to get total cost.",
    "Win %":       "Percentage of closed trades with positive net PnL (after commission). 100% = every closed contract was profitable net of fees.",
    "Commission":  "Total commission cost. Formula: Total ops × commission/side. Red because it directly reduces your PnL.",
    "Max contr.":  "Peak simultaneous open contracts — occurs at deepest price drop. Determines max capital required.",
    "Max capital": "Real capital needed = margin + max unrealized floating loss at worst intrabar price. Same formula as 'Max capital (real)' in the metrics above.",
    "Bal@peak contr.": "Account balance (realized + unrealized at Close) on the day with the most contracts open. Same as 'Balance at peak contracts' metric card.",
    "Lowest bal.":     "Lowest account balance at Close across the entire backtest period. Same as 'Lowest balance (Close)' metric card.",
    "Total needed":    "Total capital needed on the worst day = margin for open contracts + loss cover. Formula: contracts × margin + max(0, -lowest_balance).",
    "Avg days":    "Average calendar days from entry to TP hit. Low = fast turnover. High = capital tied up for longer.",
}

def th(label: str, color: str = "#94a3b8") -> str:
    """Returns a table header cell with inline-style info tooltip."""
    tip = OPT_COL_TIPS.get(label, "")
    info_html = info(tip) if tip else ""
    s = f"background:#0f172a;color:{color};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;padding:10px 14px;text-align:right;border-bottom:1px solid #334155"
    return f'<th style="{s}">{label}{info_html}</th>'


def render_opt_table(df: pd.DataFrame, top_n: int = 20):
    # All styles inline — works in all Streamlit versions
    S_TABLE  = "width:100%;border-collapse:collapse;font-size:0.82rem"
    S_TD     = "padding:9px 14px;border-bottom:1px solid #1e293b;text-align:right;color:#cbd5e1"
    S_TD_CTR = "padding:9px 14px;border-bottom:1px solid #1e293b;text-align:center;font-weight:600;color:#cbd5e1"
    RANK_COLORS = {1: ("background:#78350f", "#fbbf24"), 2: ("background:#1e3a5f", "#94a3b8"), 3: ("background:#2d1b09", "#cd7c2f")}

    rows_html = ""
    for i, (_, row) in enumerate(df.head(top_n).iterrows()):
        rank = i + 1
        rb_bg, rb_col = RANK_COLORS.get(rank, ("background:#334155", "#94a3b8"))
        rb_style = f"display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;font-size:0.75rem;font-weight:700;{rb_bg};color:{rb_col}"
        pnl = row["PnL ($)"]
        pnl_col = "#34d399" if pnl >= 0 else "#f87171"
        comm_val  = row["Total comm ($)"]  if "Total comm ($)"  in row.index else 0.0
        bal_peak  = row["Bal@peak contr."] if "Bal@peak contr." in row.index else 0.0
        bal_low   = row["Lowest bal."]     if "Lowest bal."     in row.index else 0.0
        tot_need  = row["Total needed"]    if "Total needed"    in row.index else 0
        bal_peak_col = "#34d399" if bal_peak >= 0 else "#f87171"
        bal_low_col  = "#34d399" if bal_low  >= 0 else "#f87171"
        row_bg = {1: "background:rgba(251,191,36,0.08)", 2: "background:rgba(148,163,184,0.06)", 3: "background:rgba(205,124,47,0.06)"}.get(rank, "")
        rows_html += f"""<tr style="{row_bg}">
            <td style="{S_TD_CTR}"><span style="{rb_style}">{rank}</span></td>
            <td style="{S_TD_CTR}">{row['Step ($)']:.2f}</td>
            <td style="{S_TD_CTR}">{row['TP ($)']:.2f}</td>
            <td style="{S_TD};color:{pnl_col};font-weight:700">${pnl:,.2f}</td>
            <td style="{S_TD}">{row['Entries']}</td>
            <td style="{S_TD}">{row['Closed (TP)']}</td>
            <td style="{S_TD}">{row['Open']}</td>
            <td style="{S_TD};color:#f87171">${comm_val:,.2f}</td>
            <td style="{S_TD}">{row['Max contr.']}</td>
            <td style="{S_TD}">${row['Max capital ($)']:,}</td>
            <td style="{S_TD};color:{bal_peak_col};font-weight:700">${bal_peak:,.2f}</td>
            <td style="{S_TD};color:{bal_low_col};font-weight:700">${bal_low:,.2f}</td>
            <td style="{S_TD};color:#f87171">${tot_need:,}</td>
            <td style="{S_TD}">{row['Avg days']} d</td>
        </tr>"""
    html = f"""<table style="{S_TABLE}"><thead><tr>
            {th("#")}{th("Step")}{th("TP")}{th("PnL")}{th("Buys")}{th("Sells (TP)")}{th("Open")}
            {th("Commission","#f87171")}
            {th("Max contr.")}{th("Max capital")}
            {th("Bal@peak contr.")}{th("Lowest bal.")}{th("Total needed","#f87171")}{th("Avg days")}
        </tr></thead><tbody>{rows_html}</tbody></table>"""
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">
            📊 Pyramid Backtester
            <br><span>Test strategies on commodity futures</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Default values — overridden below by each data source path
    point_value         = 1.0
    symbol              = ""
    commodity_name      = ""
    coinbase_product_id = None
    coinbase_point_value = 1.0

    # ── Data source ─────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Data source</div>', unsafe_allow_html=True)
    data_source = st.radio("Source", ["Yahoo Finance (Commodities)", "Coinbase (Crypto Futures)"], horizontal=False, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Instrument ──────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Instrument</div>', unsafe_allow_html=True)

    # ── COINBASE path ────────────────────────────────────────
    coinbase_product_id = None
    coinbase_point_value = 1.0

    if data_source == "Coinbase (Crypto Futures)":
        # Load Coinbase API keys from Streamlit secrets
        try:
            cb_key    = st.secrets["coinbase"]["api_key"]
            cb_secret = st.secrets["coinbase"]["api_secret"]
        except Exception:
            cb_key = cb_secret = ""

        # Cache Coinbase product list
        if "coinbase_products" not in st.session_state:
            with st.spinner("Loading Coinbase futures..."):
                st.session_state["coinbase_products"] = fetch_coinbase_products(cb_key, cb_secret)

        cb_products = st.session_state["coinbase_products"]

        if not cb_products:
            st.error("Could not load Coinbase futures. Check API keys in Streamlit Secrets.")
        else:
            # Group by prefix for better UX
            prefixes = sorted(set(p["prefix"] for p in cb_products))
            known_prefixes = [p for p in prefixes if p in COINBASE_FUTURES]
            unknown_prefixes = [p for p in prefixes if p not in COINBASE_FUTURES]
            all_prefixes = known_prefixes + unknown_prefixes

            selected_prefix = st.selectbox(
                "Crypto asset",
                options=all_prefixes,
                format_func=lambda x: COINBASE_FUTURES.get(x, (x, 1.0))[0],
                label_visibility="collapsed",
                key="cb_prefix_sel",
            )

            # Show contracts for selected prefix
            prefix_products = [p for p in cb_products if p["prefix"] == selected_prefix]
            product_options = [p["product_id"] for p in prefix_products]
            product_labels  = [f"{p['product_id']}  exp {p['expiry']}" for p in prefix_products]

            prev_pid = st.session_state.get(f"cb_selected_{selected_prefix}", product_options[0] if product_options else "")
            default_idx = product_options.index(prev_pid) if prev_pid in product_options else 0

            selected_cb_idx = st.selectbox(
                "Contract",
                options=range(len(product_labels)),
                format_func=lambda i: product_labels[i],
                index=default_idx,
                label_visibility="collapsed",
                key=f"cb_contract_sel_{selected_prefix}",
            )
            coinbase_product_id  = product_options[selected_cb_idx]
            selected_product     = prefix_products[selected_cb_idx]
            # Read contract_size from API data (e.g. 0.1 for ET, 0.01 for BIT)
            coinbase_point_value = selected_product.get("point_value", 1.0)
            cb_contract_size     = selected_product.get("contract_size", 1.0)
            cb_contract_unit     = selected_product.get("contract_unit", selected_prefix)
            cb_group_desc        = selected_product.get("group_desc", "")
            st.session_state[f"cb_selected_{selected_prefix}"] = coinbase_product_id

            st.caption(f"Product: `{coinbase_product_id}`")
            if cb_group_desc:
                st.caption(f"{cb_group_desc} · Contract size: {cb_contract_size} {cb_contract_unit}")
            st.caption(f"Point value: **${coinbase_point_value}**/contract per $1 price move")

            if st.button("🔄 Refresh Coinbase products", key="refresh_cb", use_container_width=True):
                if "coinbase_products" in st.session_state:
                    del st.session_state["coinbase_products"]
                st.rerun()

        commodity_name = selected_prefix if cb_products else "Coinbase"
        symbol = coinbase_product_id or ""
        point_value = coinbase_point_value
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── YAHOO FINANCE path ───────────────────────────────
        commodity_name = st.selectbox("Commodity", options=list(COMMODITY_SYMBOLS.keys()), index=0, label_visibility="collapsed")

        contract_mode = st.radio("Contract type", ["Continuous (=F)", "Specific month"], horizontal=True, label_visibility="collapsed")

        if contract_mode == "Continuous (=F)":
            symbol = COMMODITY_SYMBOLS[commodity_name]
            st.caption(f"Symbol: `{symbol}` — auto rollover to front month")
        else:
            contract_info = COMMODITY_CONTRACT_INFO.get(commodity_name)
            if contract_info is None:
                st.warning("No contract data available for this instrument.")
                symbol = COMMODITY_SYMBOLS[commodity_name]
            else:
                cache_key = f"contracts_{commodity_name}"
                if cache_key not in st.session_state:
                    with st.spinner("Loading available contracts..."):
                        st.session_state[cache_key] = get_available_contracts(commodity_name, years_ahead=2)

                contracts = st.session_state[cache_key]

                if not contracts:
                    st.warning("No active contracts found. Using continuous contract.")
                    symbol = COMMODITY_SYMBOLS[commodity_name]
                else:
                    contract_symbols = [c["symbol"] for c in contracts]
                    contract_labels  = [f"{c['name']}  |  exp {fmt_date(c['expiry'])}" for c in contracts]

                    prev_sym = st.session_state.get(f"selected_contract_{commodity_name}", contract_symbols[0])
                    default_idx = contract_symbols.index(prev_sym) if prev_sym in contract_symbols else 0

                    selected_idx = st.selectbox(
                        "Select contract",
                        options=range(len(contract_labels)),
                        format_func=lambda i: contract_labels[i],
                        index=default_idx,
                        label_visibility="collapsed",
                        key=f"contract_sel_{commodity_name}",
                    )
                    selected = contracts[selected_idx]
                    symbol = selected["symbol"]
                    st.session_state[f"selected_contract_{commodity_name}"] = symbol

                    ci = COMMODITY_CONTRACT_INFO[commodity_name]
                    st.caption(f"Symbol: `{symbol}`")
                    st.caption(f"Expires: {fmt_date(selected['expiry'])}")
                    unit_part = ci['unit'].split('/')[1] if '/' in ci['unit'] else 'units'
                    st.caption(f"Tick: {ci['tick']} = ${ci['tick_value']}  |  Size: {ci['contract_size']:,} {unit_part}")

                if st.button("🔄 Refresh contract list", key="refresh_contracts", use_container_width=True):
                    if cache_key in st.session_state:
                        del st.session_state[cache_key]
                    st.rerun()

        # Auto-detect point_value — Yahoo Finance only
        _ci = COMMODITY_CONTRACT_INFO.get(commodity_name)
        if _ci:
            point_value = _ci["tick_value"] / _ci["tick"]
        else:
            point_value = 1.0
        st.caption(f"Point value: **{point_value:.0f} $/pt**")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Timeframe ────────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Timeframe</div>', unsafe_allow_html=True)
    tf_label = st.radio("Timeframe", list(TIMEFRAME_INTERVALS.keys()), index=1, horizontal=True, label_visibility="collapsed")
    tf_interval = TIMEFRAME_INTERVALS[tf_label]
    tf_warning  = TIMEFRAME_LIMITS.get(tf_interval)
    if tf_warning:
        st.warning(tf_warning)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Date range ──────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Backtest period</div>', unsafe_allow_html=True)
    # Default date range based on timeframe
    if tf_interval == "1h":
        default_start = date.today() - timedelta(days=59)
    elif tf_interval == "1wk":
        default_start = date.today() - timedelta(days=365*5)
    else:
        default_start = date.today() - timedelta(days=365*2)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=default_start, min_value=date(2000,1,1), max_value=date.today())
    with col2:
        end_date = st.date_input("To", value=date.today(), min_value=date(2000,1,2), max_value=date.today())
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Strategy parameters ─────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Strategy parameters</div>', unsafe_allow_html=True)
    direction = st.radio("Direction", ["Long", "Short"], horizontal=True,
        help="Long: buy when price drops, TP when rises. Short: sell when price rises, TP when falls.")
    if direction == "Long":
        thresh_help  = "Buy first contract when Low < this price"
        step_help    = "Add 1 contract every X$ DROP below last entry"
        tp_help      = "Close each contract X$ ABOVE its entry price"
        thresh_label = "Entry threshold ($) — buy below"
    else:
        thresh_help  = "Sell short when High > this price"
        step_help    = "Add 1 contract every X$ RISE above last entry"
        tp_help      = "Close each contract X$ BELOW its entry price"
        thresh_label = "Entry threshold ($) — sell above"
    entry_threshold = st.number_input(thresh_label, min_value=0.01, max_value=100000.0, value=470.0, step=1.0, format="%.2f", help=thresh_help)
    pyramid_step = st.number_input("Pyramid step ($)", min_value=0.01, max_value=10000.0, value=5.0, step=0.25, format="%.2f", help=step_help)
    take_profit = st.number_input("Take Profit ($)", min_value=0.01, max_value=10000.0, value=5.0, step=0.25, format="%.2f", help=tp_help)
    margin_per_contract = st.number_input("Margin / contract ($)", min_value=100.0, max_value=1000000.0, value=1500.0, step=100.0, format="%.0f", help="Broker margin requirement per contract")
    commission_per_side = st.number_input("Commission / contract ($)", min_value=0.0, max_value=1000.0, value=2.50, step=0.50, format="%.2f", help="Broker commission per contract per side (entry + exit = 2×)")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Optimization ────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Step optimization</div>', unsafe_allow_html=True)
    optimize_enabled = st.checkbox("Enable optimization", value=False, help="Test all step × TP combinations automatically")

    opt_step_min = 2.0
    opt_step_max = 10.0
    opt_step_inc = 1.0
    opt_tp_min   = 2.0
    opt_tp_max   = 10.0
    opt_tp_inc   = 1.0

    opt_max_capital = 0.0  # 0 = no filter

    if optimize_enabled:
        st.caption("**Pyramid step range**")
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            opt_step_min = st.number_input("From", value=2.0, step=0.25, format="%.2f", key="smin")
        with cs2:
            opt_step_max = st.number_input("To", value=10.0, step=0.25, format="%.2f", key="smax")
        with cs3:
            opt_step_inc = st.number_input("By", value=1.0, step=0.25, format="%.2f", key="sinc")
        st.caption("**Take Profit range**")
        ct1, ct2, ct3 = st.columns(3)
        with ct1:
            opt_tp_min = st.number_input("From", value=2.0, step=0.25, format="%.2f", key="tmin")
        with ct2:
            opt_tp_max = st.number_input("To", value=10.0, step=0.25, format="%.2f", key="tmax")
        with ct3:
            opt_tp_inc = st.number_input("By", value=1.0, step=0.25, format="%.2f", key="tinc")
        n_s = max(1, round((opt_step_max - opt_step_min) / opt_step_inc) + 1)
        n_t = max(1, round((opt_tp_max - opt_tp_min) / opt_tp_inc) + 1)
        st.caption(f"Total: **{n_s * n_t} combinations**")
        st.markdown("---", unsafe_allow_html=False)
        st.caption("**Max account capital ($)**")
        opt_max_capital = st.number_input(
            "Max capital I have ($)",
            min_value=0.0, value=0.0, step=1000.0, format="%.0f",
            key="opt_max_capital",
            help="Filter out combinations where 'Max capital (real)' exceeds this amount. Set to 0 to show all.",
        )
        if opt_max_capital > 0:
            st.caption(f"Only showing combos requiring ≤ **${opt_max_capital:,.0f}**")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    run_button = st.button("▶ Run Backtest", type="primary", use_container_width=True)
    st.markdown("<div style='padding:0 20px 20px 20px'><p style='font-size:0.7rem;color:#475569;text-align:center;margin-top:8px'>Data: Yahoo Finance · for educational purposes only</p></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <div class="main-title">Pyramid Backtester</div>
    <div class="main-subtitle">Test pyramid long strategies on commodity futures</div>
</div>
""", unsafe_allow_html=True)

with st.container():

    st.markdown('<div class="strategy-tag">Pyramid Long · Buy below threshold · TP per contract</div>', unsafe_allow_html=True)

    if run_button:
        df_new = None
        with st.spinner(f"Fetching data for {commodity_name}..."):
            try:
                if data_source == "Coinbase (Crypto Futures)" and coinbase_product_id:
                    gran = COINBASE_GRANULARITY.get(tf_interval, "ONE_DAY")
                    try:
                        cb_key    = st.secrets["coinbase"]["api_key"]
                        cb_secret = st.secrets["coinbase"]["api_secret"]
                    except Exception:
                        cb_key = cb_secret = ""
                    df_new = fetch_coinbase_candles(coinbase_product_id, start=start_date, end=end_date, granularity=gran, api_key=cb_key, api_secret=cb_secret)
                else:
                    df_new = fetch_data(symbol, start=start_date, end=end_date, interval=tf_interval)
            except Exception as e:
                st.error(f"Data fetch error: {e}")
        if df_new is None or df_new.empty:
            st.error("No data returned. Check symbol or adjust date range.")
            if data_source == "Coinbase (Crypto Futures)":
                st.info("💡 Coinbase futures contracts have limited history — only from their launch date. Try a shorter date range (e.g. last 30–60 days).")
        else:
            with st.spinner("Running backtest..."):
                result_new = run_backtest(
                    df=df_new,
                    entry_threshold=entry_threshold,
                    pyramid_step=pyramid_step,
                    take_profit=take_profit,
                    margin_per_contract=margin_per_contract,
                    qty_per_entry=1,
                    point_value=point_value,
                    commission_per_side=commission_per_side,
                    direction=direction.lower(),
                )
            st.session_state["bt_df"]              = df_new
            st.session_state["bt_result"]          = result_new
            st.session_state["bt_symbol"]          = commodity_name
            st.session_state["bt_start"]           = fmt_date(start_date)
            st.session_state["bt_end"]             = fmt_date(end_date)
            st.session_state["bt_margin"]          = margin_per_contract
            st.session_state["bt_threshold"]       = entry_threshold
            st.session_state["bt_step"]            = pyramid_step
            st.session_state["bt_tp"]              = take_profit
            st.session_state["bt_point_value"]     = point_value
            st.session_state["bt_commission"]      = commission_per_side
            st.session_state["bt_direction"]       = direction
            st.session_state["bt_tf_label"]        = tf_label
            st.session_state["bt_tf_interval"]     = tf_interval

    if "bt_result" not in st.session_state:
        st.markdown("""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:32px;text-align:center;margin-top:24px">
            <div style="font-size:2.5rem;margin-bottom:12px">👈</div>
            <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:8px">Set parameters and run the backtest</div>
            <div style="font-size:0.85rem;color:#64748b">Choose a commodity, date range and strategy parameters in the left panel,<br>then click <b>Run Backtest</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df     = st.session_state["bt_df"]
        result = st.session_state["bt_result"]
        commodity_name_disp      = st.session_state["bt_symbol"]
        margin_per_contract_disp = st.session_state["bt_margin"]
        entry_threshold_disp     = st.session_state["bt_threshold"]
        pv                       = st.session_state.get("bt_point_value", 1.0)
        tf_label_disp            = st.session_state.get("bt_tf_label", "Daily (1d)")
        tf_interval_disp         = st.session_state.get("bt_tf_interval", "1d")
        direction_disp           = st.session_state.get("bt_direction", "Long")
        is_long_disp             = direction_disp == "Long"

        # Check if point_value in session_state matches current instrument specs
        # If mismatch (e.g. instrument was added to CONTRACT_INFO after last run)
        # warn user to re-run backtest
        current_pv = point_value  # from sidebar, always up-to-date
        if abs(current_pv - pv) > 0.01 and pv == 1.0:
            st.warning(f"⚠️ Point value mismatch detected. Stored: {pv:.0f} $/pt, Current: {current_pv:.0f} $/pt. Please **Re-run backtest** to recalculate with correct point value.")
            pv = current_pv  # use correct value for display

        pnl_currency = "USD"
        pv_info = f"point value: {pv:.0f} $/pt (auto-detected)"
        comm = st.session_state.get("bt_commission", 0.0)
        comm_info = f"commission: ${comm:.2f}/side (${comm*2:.2f} round-trip per contract)" if comm > 0 else "no commission"
        dir_badge_color = "#34d399" if is_long_disp else "#f87171"
        dir_badge = f"<span style='background:{dir_badge_color};color:#0f172a;font-weight:700;padding:2px 8px;border-radius:4px;font-size:0.75rem'>{direction_disp}</span>"

        bar_label = {"1h": "bars (1h)", "1d": "sessions", "1wk": "weeks"}.get(tf_interval_disp, "bars")

        # Warn if Coinbase returned much less data than requested
        if data_source == "Coinbase (Crypto Futures)" and len(df) < 30:
            st.warning(f"⚠️ Only {len(df)} bars returned. Coinbase futures contracts have limited history — only data since contract launch date is available.")

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:0.82rem'>"
            f"{dir_badge}"
            f"<span style='color:#64748b'>{commodity_name_disp} · {tf_label_disp} · {len(df)} {bar_label}</span>"
            f"<span style='color:#334155'>|</span>"
            f"<span style='color:#475569'>{st.session_state['bt_start']} → {st.session_state['bt_end']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"<div style='font-size:0.75rem;color:#475569;margin-bottom:14px'>{pv_info} · {comm_info}</div>", unsafe_allow_html=True)

        # ── Metrics ────────────────────────────────────────────
        avg_pnl = result.total_pnl / result.total_trades if result.total_trades > 0 else 0
        pnl_pos = result.total_pnl >= 0
        period_high = float(df["High"].max())
        period_low  = float(df["Low"].min())
        high_date   = fmt_date(df["High"].idxmax())
        low_date    = fmt_date(df["Low"].idxmin())

        if True:
            # ── Active parameters banner — compact pill style ───
            _step = result.params.get("pyramid_step", "?")
            _tp   = result.params.get("take_profit", "?")
            _thr  = result.params.get("entry_threshold", "?")
            _comm = result.params.get("commission_per_side", 0.0)
            _pv   = result.params.get("point_value", 1.0)
            dir_col = "#34d399" if direction_disp == "Long" else "#f87171"
            dir_bg  = "rgba(52,211,153,0.12)" if direction_disp == "Long" else "rgba(248,113,113,0.12)"
            pill = "display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;font-size:0.78rem;font-weight:600;gap:4px"
            st.markdown(
                f"<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:16px;padding:12px 16px;"
                f"background:#0f172a;border:1px solid #1e293b;border-radius:10px'>"
                f"<span style='color:#475569;font-size:0.7rem;font-weight:700;letter-spacing:0.1em;margin-right:4px'>ACTIVE</span>"
                f"<span style='{pill};background:{dir_bg};color:{dir_col};border:1px solid {dir_col}33'>{direction_disp}</span>"
                f"<span style='{pill};background:rgba(248,113,113,0.1);color:#fca5a5;border:1px solid #f8717133'>Threshold {_thr}$</span>"
                f"<span style='{pill};background:rgba(56,189,248,0.1);color:#7dd3fc;border:1px solid #38bdf833'>Step {_step}$</span>"
                f"<span style='{pill};background:rgba(52,211,153,0.1);color:#86efac;border:1px solid #34d39933'>TP {_tp}$</span>"
                f"<span style='{pill};background:#1e293b;color:#94a3b8;border:1px solid #33415533'>Comm {_comm}$/side</span>"
                f"<span style='{pill};background:#1e293b;color:#94a3b8;border:1px solid #33415533'>PV {_pv:.0f}$/pt</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            # ── Section: Results ──────────────────────────────
            st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin:4px 0 8px 0'>Results</div>", unsafe_allow_html=True)
            total_comm_paid = getattr(result, "total_commission", 0.0)
            total_entries   = len(result.trades)
            total_exits     = result.total_trades
            total_ops       = total_entries + total_exits
            cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px">'
            cards_html += metric_card("Total PnL", f"${result.total_pnl:,.2f}", f"{'▲' if pnl_pos else '▼'} after commissions", positive=pnl_pos)
            cards_html += metric_card("Total commission", f"${total_comm_paid:,.2f}", f"{total_ops} ops × ${comm:.2f}")
            cards_html += metric_card("Total operations", str(total_ops), f"{total_entries} buys + {total_exits} sells (TP)")
            cards_html += metric_card("Closed trades", str(total_exits), f"{result.winning_trades}W / {result.losing_trades}L")
            cards_html += metric_card("Avg PnL / trade", f"${avg_pnl:.2f}", "per closed contract", positive=avg_pnl >= 0)
            cards_html += metric_card("Open positions", str(result.open_trades), "not closed at end of period")
            cards_html += metric_card("Avg days to TP", f"{result.avg_days_open:.0f}", "average hold time")
            cards_html += metric_card("Period HIGH", f"${period_high:,.2f}", f"highest price · {high_date}")
            cards_html += metric_card("Period LOW", f"${period_low:,.2f}", f"lowest price · {low_date}")
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            # ── Section: Capital & Risk ───────────────────────
            st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin:4px 0 8px 0'>Capital & Risk</div>", unsafe_allow_html=True)
            risk_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px">'
            risk_html += metric_card("Max contracts", str(result.max_concurrent), "open simultaneously")
            risk_html += metric_card("Max capital req.", f"${result.max_capital_needed:,.0f}", f"{result.max_concurrent} × ${margin_per_contract_disp:,.0f} (margin only)")
            unrealized_extra = result.max_capital_with_unrealized - result.max_capital_needed
            risk_html += metric_card("Max capital (real)", f"${result.max_capital_with_unrealized:,.0f}", f"margin + ${unrealized_extra:,.0f} unrealized loss", positive=False)
            max_contr_idx = result.daily_open_contracts.idxmax()
            max_contr_date = fmt_date(max_contr_idx)
            bal_at_max_contr = float(result.balance_curve.loc[max_contr_idx]) if max_contr_idx in result.balance_curve.index else 0.0
            risk_html += metric_card("Balance at peak contracts", f"${bal_at_max_contr:,.2f}", f"Close · {max_contr_date} ({result.max_concurrent} contr.)", positive=bal_at_max_contr >= 0)
            min_balance = float(result.balance_curve.min())
            min_balance_idx = result.balance_curve.idxmin()
            min_balance_date = fmt_date(min_balance_idx)
            min_balance_contracts = int(result.daily_open_contracts.loc[min_balance_idx]) if min_balance_idx in result.daily_open_contracts.index else 0
            risk_html += metric_card("Lowest balance (Close)", f"${min_balance:,.2f}", f"{min_balance_date} · {min_balance_contracts} contr.", positive=False)
            worst_margin = min_balance_contracts * margin_per_contract_disp
            total_needed = worst_margin + max(0.0, -min_balance)
            risk_html += metric_card("Total needed at worst day", f"${total_needed:,.2f}", f"margin ${worst_margin:,.0f} + loss ${max(0.0,-min_balance):,.0f}", positive=False)
            risk_html += '</div>'
            st.markdown(risk_html, unsafe_allow_html=True)

            # ── Charts ────────────────────────────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Price + signals", "Open contracts", "🤖 IBKR Script"])

            with chart_tab2:
                fig_open = go.Figure()
                fig_open.add_trace(go.Scatter(
                    x=result.daily_open_contracts.index, y=result.daily_open_contracts.values,
                    mode="lines", name="Contracts",
                    line=dict(color="#f87171", width=1.5),
                    fill="tozeroy", fillcolor="rgba(248,113,113,0.1)",
                ))
                fig_open.add_hline(
                    y=result.max_concurrent, line_dash="dash", line_color="#fbbf24",
                    annotation_text=f"Max: {result.max_concurrent}", annotation_position="right",
                )
                fig_open.update_layout(
                    template="plotly_dark", height=360,
                    margin=dict(l=0,r=0,t=24,b=0),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    title=dict(text=f"Open contracts over time ({tf_label_disp})", font=dict(size=13, color="#94a3b8")),
                    xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig_open, use_container_width=True)

            with chart_tab1:
                from plotly.subplots import make_subplots

                entry_groups = collections.defaultdict(list)
                for t in result.trades:
                    entry_groups[t.entry_date].append(t.entry_price)

                tp_groups = collections.defaultdict(list)
                tp_pnl_groups = collections.defaultdict(float)
                for t in result.trades:
                    if t.closed:
                        tp_groups[t.exit_date].append(t.exit_price)
                        tp_pnl_groups[t.exit_date] += t.pnl

                # Build hover text for each bar: date, OHLC, balance, open contracts
                open_contracts_series = result.daily_open_contracts
                balance_series        = result.balance_curve
                equity_series         = result.equity_curve

                hover_texts = []
                for idx_d in df.index:
                    o   = df.loc[idx_d, "Open"]
                    h   = df.loc[idx_d, "High"]
                    l   = df.loc[idx_d, "Low"]
                    c   = df.loc[idx_d, "Close"]
                    bal = balance_series.get(idx_d, 0.0)
                    eq  = equity_series.get(idx_d, 0.0)
                    oc  = open_contracts_series.get(idx_d, 0)
                    unr = bal - eq
                    bal_color_txt = "<span style='color:#34d399'>" if bal >= 0 else "<span style='color:#f87171'>"
                    # For 1h bars include time, for daily/weekly just date
                    if tf_interval_disp == "1h":
                        try:
                            time_str = f" {idx_d.strftime('%H:%M')}"
                        except Exception:
                            time_str = ""
                    else:
                        time_str = ""
                    hover_texts.append(
                        f"<b style='font-size:13px'>{fmt_date(idx_d)}{time_str}</b><br>"
                        f"<span style='font-size:12px'>"
                        f"Open: <b>{o:.2f}</b>  High: <b>{h:.2f}</b>  Low: <b>{l:.2f}</b>  Close: <b>{c:.2f}</b><br>"
                        f"Balance: {bal_color_txt}<b>{bal:+.2f}</b></span><br>"
                        f"<span style='font-size:11px;color:#94a3b8'>"
                        f"Realized: {eq:+.2f}  |  Unrealized: {unr:+.2f}<br>"
                        f"Open contracts: {int(oc)}</span>"
                    )

                # Subplot: price 60% / balance 40% (taller balance panel)
                fig_p = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.58, 0.42], vertical_spacing=0.04,
                    subplot_titles=["", "Balance at Close ($)"]
                )

                # ── Row 1: Candlestick ────────────────────────────────────────
                fig_p.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="Price",
                    increasing_line_color="#34d399", decreasing_line_color="#f87171",
                    text=hover_texts, hoverinfo="text",
                ), row=1, col=1)

                fig_p.add_hline(y=entry_threshold_disp, line_dash="dash",
                    line_color="#f87171", line_width=1.5,
                    annotation_text=f"Threshold {entry_threshold_disp}",
                    annotation_position="right", row=1, col=1)

                price_range = float(df["High"].max() - df["Low"].min())
                offset = price_range * 0.008

                # Build per-contract detail maps for hover
                # entry_details[date] = list of (entry_price) per contract
                entry_details = collections.defaultdict(list)
                for t in result.trades:
                    entry_details[t.entry_date].append(t.entry_price)

                # tp_details[exit_date] = list of (entry_price, exit_price, pnl)
                tp_details = collections.defaultdict(list)
                for t in result.trades:
                    if t.closed:
                        tp_details[t.exit_date].append((t.entry_price, t.exit_price, t.pnl))

                entry_label = "Buy" if is_long_disp else "Sell Short"
                entry_sym   = "triangle-up" if is_long_disp else "triangle-down"
                entry_col   = "#34d399" if is_long_disp else "#f87171"

                if entry_details:
                    ex = list(entry_details.keys())
                    ey = []
                    for d in ex:
                        if d in df.index:
                            ey.append(float(df.loc[d, "Low"]) - offset if is_long_disp else float(df.loc[d, "High"]) + offset)
                        else:
                            ey.append(float(entry_details[d][0]))

                    # Hover: list each contract with its own price and date
                    etxt = []
                    for d in ex:
                        prices = entry_details[d]
                        lines = [f"<b>{entry_label} — {fmt_date(d)}</b>"]
                        for idx_c, ep in enumerate(prices):
                            lines.append(f"  Contract {idx_c+1}: bought @ <b>{ep:.2f}</b>  TP → {ep + result.params.get('take_profit', 0):.2f}")
                        etxt.append("<br>".join(lines))

                    fig_p.add_trace(go.Scatter(
                        x=ex, y=ey, mode="markers", name=entry_label,
                        marker=dict(symbol=entry_sym, size=8, color=entry_col, line=dict(width=0)),
                        hovertext=etxt, hoverinfo="text",
                    ), row=1, col=1)

                if tp_details:
                    tx = list(tp_details.keys())
                    ty = []
                    for d in tx:
                        if d in df.index:
                            ty.append(float(df.loc[d, "High"]) + offset if is_long_disp else float(df.loc[d, "Low"]) - offset)
                        else:
                            ty.append(float(tp_details[d][0][1]))

                    # Hover: list each contract with its own entry, exit price and PnL
                    tp_sym = "triangle-down" if is_long_disp else "triangle-up"
                    ttxt = []
                    for d in tx:
                        contracts = tp_details[d]
                        total_pnl_day = sum(c[2] for c in contracts)
                        lines = [f"<b>TP — {fmt_date(d)}</b>  ({len(contracts)} contract{'s' if len(contracts)>1 else ''})"]
                        for idx_c, (ep, xp, pnl) in enumerate(contracts):
                            lines.append(f"  Contract {idx_c+1}: bought @ {ep:.2f}  sold @ <b>{xp:.2f}</b>  PnL <b style='color:{'#34d399' if pnl>=0 else '#f87171'}'>{pnl:+.2f}</b>")
                        lines.append(f"  <b>Total PnL: {total_pnl_day:+.2f}</b>")
                        ttxt.append("<br>".join(lines))

                    fig_p.add_trace(go.Scatter(
                        x=tx, y=ty, mode="markers", name="TP",
                        marker=dict(symbol=tp_sym, size=8, color="#fbbf24", line=dict(width=0)),
                        hovertext=ttxt, hoverinfo="text",
                    ), row=1, col=1)

                # ── Row 2: Balance curve — green above zero, red below ───────
                bal_vals = balance_series.values
                eq_vals  = equity_series.values
                bal_x    = list(balance_series.index)

                # Build colored segments: split at zero crossings
                # Use scatter with color per segment via separate traces per sign
                pos_y = [v if v >= 0 else 0.0 for v in bal_vals]
                neg_y = [v if v < 0  else 0.0 for v in bal_vals]

                # Positive balance — green fill
                fig_p.add_trace(go.Scatter(
                    x=bal_x, y=pos_y,
                    mode="lines", name="Balance (+)",
                    line=dict(color="#34d399", width=2),
                    fill="tozeroy", fillcolor="rgba(52,211,153,0.15)",
                    hoverinfo="skip", showlegend=True,
                ), row=2, col=1)

                # Negative balance — red fill
                fig_p.add_trace(go.Scatter(
                    x=bal_x, y=neg_y,
                    mode="lines", name="Balance (−)",
                    line=dict(color="#f87171", width=2),
                    fill="tozeroy", fillcolor="rgba(248,113,113,0.15)",
                    hoverinfo="skip", showlegend=True,
                ), row=2, col=1)

                # Invisible hover trace for balance — full values with tooltip
                fig_p.add_trace(go.Scatter(
                    x=bal_x, y=bal_vals,
                    mode="lines", name="Balance",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hovertemplate="%{x|%m/%d/%Y} Balance: <b>%{y:+.2f}</b><extra></extra>",
                ), row=2, col=1)

                # Realized PnL — thin dashed reference line
                fig_p.add_trace(go.Scatter(
                    x=list(equity_series.index), y=eq_vals,
                    mode="lines", name="Realized PnL",
                    line=dict(color="#94a3b8", width=1, dash="dot"),
                    hovertemplate="%{x|%m/%d/%Y} Realized: <b>%{y:+.2f}</b><extra></extra>",
                ), row=2, col=1)

                fig_p.add_hline(y=0, line_dash="solid", line_color="#475569",
                    line_width=1, row=2, col=1)

                fig_p.update_layout(
                    template="plotly_dark", height=700,
                    margin=dict(l=0, r=0, t=40, b=60),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    xaxis_rangeslider_visible=False,
                     title=dict(
                        text=f"{commodity_name_disp} · {tf_label_disp} · {direction_disp} · T:{entry_threshold_disp} S:{result.params.get('pyramid_step','?')} TP:{result.params.get('take_profit','?')}",
                        font=dict(size=12, color="#64748b"),
                    ),
                    legend=dict(orientation="h", y=-0.1, x=0, font=dict(color="#94a3b8", size=12)),
                    hovermode="x",
                    hoverdistance=100,
                    spikedistance=100,
                    hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(size=13, color="#e2e8f0"), namelength=-1),
                )
                fig_p.update_xaxes(gridcolor="#1e293b", showgrid=True, showspikes=True, spikecolor="#64748b", spikethickness=1, spikemode="across")
                fig_p.update_yaxes(gridcolor="#1e293b", row=1, col=1)
                fig_p.update_yaxes(
                    gridcolor="#1e293b", row=2, col=1,
                    title_text="Balance at Close ($)",
                    title_font=dict(color="#64748b", size=11),
                    zeroline=True, zerolinecolor="#475569", zerolinewidth=2,
                )
                st.plotly_chart(fig_p, use_container_width=True)

            # ── Volume Profile ──────────────────────────────────
            st.markdown("<div style='margin-top:24px;border-top:1px solid #1e293b;padding-top:20px'></div>", unsafe_allow_html=True)
            st.markdown("<div style='height:1px;background:#1e293b;margin:24px 0 20px 0'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Volume Profile — Historical Support Zones</div>", unsafe_allow_html=True)

            vp_min_date = date(1950, 1, 1)
            vp_dc1, vp_dc2 = st.columns(2)
            with vp_dc1:
                vp_start = st.date_input("From", value=date.today() - timedelta(days=365*10), key="vp_start2", min_value=vp_min_date, max_value=date.today())
            with vp_dc2:
                vp_end = st.date_input("To", value=date.today(), key="vp_end2", min_value=vp_min_date, max_value=date.today())

            with st.spinner("Computing Volume Profile..."):
                try:
                    vp_df_raw = fetch_data(symbol, start=vp_start, end=vp_end)
                    if vp_df_raw is not None and not vp_df_raw.empty:
                        zones, poc, va_low, va_high = find_support_zones(vp_df_raw, bins=300, top_n=6, min_gap_pct=0.025)
                        vp_data, _, _, _ = compute_volume_profile(vp_df_raw, bins=300)
                        vp_ok = True
                    else:
                        vp_ok = False
                except Exception:
                    vp_ok = False

            if not vp_ok:
                st.error("No VP data. Adjust date range.")
            else:
                vp_period_high = float(vp_df_raw["High"].max())
                vp_period_low  = float(vp_df_raw["Low"].min())
                st.markdown(f"<div style='font-size:0.8rem;color:#64748b;margin-bottom:12px'>✓ {len(vp_df_raw)} sessions · {fmt_date(vp_start)} → {fmt_date(vp_end)}</div>", unsafe_allow_html=True)

                kl = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0">'
                kl += metric_card("Point of Control", f"${poc:,.2f}", "price with highest volume")
                kl += metric_card("Value Area Low", f"${va_low:,.2f}", "lower bound of 70% vol ← entry signal", positive=True)
                kl += metric_card("Value Area High", f"${va_high:,.2f}", "upper bound of 70% vol")
                kl += metric_card("Period Low", f"${vp_period_low:,.2f}", "absolute minimum")
                kl += metric_card("Period High", f"${vp_period_high:,.2f}", "absolute maximum")
                kl += metric_card("Support zones", str(len(zones)), "detected VP levels")
                kl += '</div>'
                st.markdown(kl, unsafe_allow_html=True)

                from plotly.subplots import make_subplots

                # Use subplot: candlestick (75%) | VP histogram (25%) side by side
                fig_vp = make_subplots(
                    rows=1, cols=2,
                    column_widths=[0.75, 0.25],
                    horizontal_spacing=0.01,
                    shared_yaxes=True,
                )

                # ── Left: Candlestick ─────────────────────────────
                fig_vp.add_trace(go.Candlestick(
                    x=vp_df_raw.index, open=vp_df_raw["Open"], high=vp_df_raw["High"],
                    low=vp_df_raw["Low"], close=vp_df_raw["Close"], name="Price",
                    increasing_line_color="#34d399", decreasing_line_color="#f87171",
                ), row=1, col=1)

                zone_colors  = ["rgba(56,189,248,0.15)","rgba(251,191,36,0.12)","rgba(52,211,153,0.12)",
                                "rgba(248,113,113,0.12)","rgba(167,139,250,0.12)","rgba(249,115,22,0.12)"]
                border_colors = ["#38bdf8","#fbbf24","#34d399","#f87171","#a78bfa","#f97316"]

                # Zone rectangles — no annotations (labels go on right panel)
                for i, zone in enumerate(zones):
                    fig_vp.add_hrect(
                        y0=zone["zone_low"], y1=zone["zone_high"],
                        fillcolor=zone_colors[i % len(zone_colors)],
                        line=dict(color=border_colors[i % len(border_colors)], width=1, dash="dot"),
                        row=1, col=1,
                    )

                # POC and VA lines — no text annotations on chart
                for yval, col, width in [(poc, "#fbbf24", 2), (va_low, "#34d399", 2), (va_high, "#38bdf8", 1.5)]:
                    fig_vp.add_hline(y=yval, line_dash="dash", line_color=col, line_width=width, row=1, col=1)

                # ── Right: VP Histogram ───────────────────────────
                bar_colors = []
                zone_ranges = [(z["zone_low"], z["zone_high"]) for z in zones]
                for price in vp_data["price_level"]:
                    if abs(price - poc) < (vp_period_high - vp_period_low) / 300 * 4:
                        bar_colors.append("#fbbf24")
                    elif any(lo <= price <= hi for lo, hi in zone_ranges):
                        bar_colors.append("#38bdf8")
                    else:
                        bar_colors.append("#1e3a5f")

                # Custom hover for each bar in histogram
                vp_hover = []
                for price, pct in zip(vp_data["price_level"], vp_data["pct"]):
                    label = ""
                    if abs(price - poc) < (vp_period_high - vp_period_low) / 300 * 4:
                        label = " ← POC"
                    elif abs(price - va_low) < (vp_period_high - vp_period_low) / 300 * 4:
                        label = " ← VA Low"
                    elif abs(price - va_high) < (vp_period_high - vp_period_low) / 300 * 4:
                        label = " ← VA High"
                    vp_hover.append(f"Price: {price:.2f}{label}<br>Volume: {pct:.2f}%")

                fig_vp.add_trace(go.Bar(
                    x=vp_data["pct"], y=vp_data["price_level"],
                    orientation="h",
                    marker_color=bar_colors, marker_line_width=0,
                    name="Volume %",
                    hovertext=vp_hover, hoverinfo="text",
                    showlegend=False,
                ), row=1, col=2)

                # POC / VA lines on histogram too
                for yval, col, width in [(poc, "#fbbf24", 2), (va_low, "#34d399", 2), (va_high, "#38bdf8", 1.5)]:
                    fig_vp.add_hline(y=yval, line_dash="dash", line_color=col, line_width=width, row=1, col=2)

                # Zone fills on histogram
                for i, zone in enumerate(zones):
                    fig_vp.add_hrect(
                        y0=zone["zone_low"], y1=zone["zone_high"],
                        fillcolor=zone_colors[i % len(zone_colors)],
                        line=dict(width=0),
                        row=1, col=2,
                    )

                # ── Labels via scatter (right side of histogram) ──
                label_x_max = float(vp_data["pct"].max()) * 1.05
                label_items = [
                    (poc,    "#fbbf24", f"POC {poc:.2f}"),
                    (va_low, "#34d399", f"VA Low {va_low:.2f}"),
                    (va_high,"#38bdf8", f"VA High {va_high:.2f}"),
                ]
                for i, zone in enumerate(zones):
                    mid = (zone["zone_low"] + zone["zone_high"]) / 2
                    label_items.append((mid, border_colors[i % len(border_colors)], f"{zone['price']:.2f} ({zone['volume_pct']:.1f}%)"))

                for yval, col, txt in label_items:
                    fig_vp.add_trace(go.Scatter(
                        x=[label_x_max], y=[yval],
                        mode="text",
                        text=[txt],
                        textposition="middle right",
                        textfont=dict(color=col, size=10),
                        showlegend=False,
                        hoverinfo="skip",
                    ), row=1, col=2)

                fig_vp.update_layout(
                    template="plotly_dark", height=580,
                    margin=dict(l=0, r=140, t=32, b=0),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    title=dict(text=f"{commodity_name} — Volume Profile & support zones", font=dict(size=13, color="#94a3b8")),
                    barmode="overlay",
                    bargap=0,
                    hovermode="y unified",
                )
                fig_vp.update_xaxes(gridcolor="#1e293b", row=1, col=1, rangeslider_visible=False)
                fig_vp.update_xaxes(gridcolor="#1e293b", title_text="% vol", row=1, col=2, showticklabels=True)
                fig_vp.update_yaxes(gridcolor="#1e293b", row=1, col=1)
                fig_vp.update_yaxes(gridcolor="#1e293b", row=1, col=2, showticklabels=False)
                st.plotly_chart(fig_vp, use_container_width=True)

                zone_rows = ""
                for z in zones:
                    if z["price"] <= va_low:
                        sig = "<span style='color:#34d399;font-weight:700'>✓ Entry zone</span>"
                    elif z["price"] <= poc:
                        sig = "<span style='color:#fbbf24'>~ Below POC</span>"
                    else:
                        sig = "<span style='color:#64748b'>Above POC</span>"
                    zone_rows += f"<tr><td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:700'>${z['price']:,.2f}</td><td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#94a3b8'>${z['zone_low']:,.2f} – ${z['zone_high']:,.2f}</td><td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#38bdf8;font-weight:600'>{z['volume_pct']:.1f}%</td><td style='padding:8px 12px;border-bottom:1px solid #1e293b'>{sig}</td></tr>"
                st.markdown(f"<table style='width:100%;border-collapse:collapse;font-size:0.82rem'><thead><tr><th style='padding:9px 12px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>Price</th><th style='padding:9px 12px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>Zone range</th><th style='padding:9px 12px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>% Vol</th><th style='padding:9px 12px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>Signal</th></tr></thead><tbody>{zone_rows}</tbody></table>", unsafe_allow_html=True)
                st.markdown(f"<div style='background:#0c2541;border:1px solid #0369a1;border-radius:8px;padding:14px 18px;margin-top:16px;font-size:0.85rem;color:#93c5fd;line-height:1.7'><b style='color:#38bdf8'>💡 Tip:</b> Set <b>Entry threshold</b> in Strategy 1 to <b style='color:#34d399'>${va_low:,.2f}</b> (Value Area Low) — historically cheap zone for {commodity_name}.</div>", unsafe_allow_html=True)

            # ── IBKR Script tab ──────────────────────────────────
            with chart_tab3:
                _thr  = result.params.get("entry_threshold", entry_threshold_disp)
                _step = result.params.get("pyramid_step", st.session_state.get("bt_step", 5.0))
                _tp   = result.params.get("take_profit", st.session_state.get("bt_tp", 5.0))
                _pv   = result.params.get("point_value", 1.0)
                _dir  = st.session_state.get("bt_direction", "Long")
                _sym  = st.session_state.get("bt_symbol", commodity_name_disp)
                _tf   = st.session_state.get("bt_tf_label", "Daily (1d)")
                _comm = result.params.get("commission_per_side", 0.0)

                # Map commodity to IBKR symbol/exchange/currency
                ibkr_map = {
                    "Corn (ZC)":          ("ZC", "CBOT", "USD", "FUT", "cents/bushel"),
                    "Wheat (ZW)":         ("ZW", "CBOT", "USD", "FUT", "cents/bushel"),
                    "Soybeans (ZS)":      ("ZS", "CBOT", "USD", "FUT", "cents/bushel"),
                    "Oats (ZO)":          ("ZO", "CBOT", "USD", "FUT", "cents/bushel"),
                    "Rough Rice (ZR)":    ("ZR", "CBOT", "USD", "FUT", "cents/cwt"),
                    "Live Cattle (LE)":   ("LE", "CME",  "USD", "FUT", "cents/lb"),
                    "Lean Hogs (HE)":     ("HE", "CME",  "USD", "FUT", "cents/lb"),
                    "Feeder Cattle (FC)": ("GF", "CME",  "USD", "FUT", "cents/lb"),
                    "Sugar #11 (SB)":     ("SB", "NYBOT","USD", "FUT", "cents/lb"),
                    "Cotton (CT)":        ("CT", "NYBOT","USD", "FUT", "cents/lb"),
                    "Coffee (KC)":        ("KC", "NYBOT","USD", "FUT", "cents/lb"),
                    "Cocoa (CC)":         ("CC", "NYBOT","USD", "FUT", "USD/ton"),
                    "Orange Juice (OJ)":  ("OJ", "NYBOT","USD", "FUT", "cents/lb"),
                    "Crude Oil (CL)":     ("CL", "NYMEX","USD", "FUT", "USD/bbl"),
                    "Natural Gas (NG)":   ("NG", "NYMEX","USD", "FUT", "USD/MMBtu"),
                    "Gold (GC)":          ("GC", "COMEX","USD", "FUT", "USD/oz"),
                    "Silver (SI)":        ("SI", "COMEX","USD", "FUT", "cents/oz"),
                    "Palladium (PA)":     ("PA", "COMEX","USD", "FUT", "USD/oz"),
                    "Platinum (PL)":      ("PL", "NYMEX","USD", "FUT", "USD/oz"),
                    "Copper (HG)":        ("HG", "COMEX","USD", "FUT", "cents/lb"),
                    "S&P 500 (ES)":       ("ES", "CME",  "USD", "FUT", "USD/pt"),
                    "Nasdaq (NQ)":        ("NQ", "CME",  "USD", "FUT", "USD/pt"),
                }
                ibkr_sym, ibkr_exch, ibkr_ccy, ibkr_stype, ibkr_unit = ibkr_map.get(
                    _sym, ("ZC", "CBOT", "USD", "FUT", "cents/bushel")
                )
                action_entry = "BUY" if _dir == "Long" else "SELL"
                action_exit  = "SELL" if _dir == "Long" else "BUY"
                cmp_op       = "<" if _dir == "Long" else ">"
                add_cmp      = "<=" if _dir == "Long" else ">="
                price_move   = f"drops" if _dir == "Long" else "rises"

                # Build script using string substitution — avoids f-string conflicts
                # All {VAR} in the template are literal Python code, not app.py variables
                _script_template = (
                    '#!/usr/bin/env python3\n'
                    '"""\n'
                    'IBKR Pyramid __DIR__ Strategy — Auto-generated by Pyramid Backtester\n'
                    'Instrument : __SYM__ (__IBKR_SYM__) on __IBKR_EXCH__\n'
                    'Direction  : __DIR__\n'
                    'Timeframe  : __TF__\n'
                    'Parameters : Threshold=__THR__  Step=__STEP__  TP=__TP__  Commission=__COMM__/side\n'
                    '\n'
                    'HOW TO USE:\n'
                    '1. Install: pip install ib_insync\n'
                    '2. Open TWS or IB Gateway on your computer\n'
                    '3. Enable API: TWS -> Edit -> Global Config -> API -> Enable ActiveX and Socket Clients\n'
                    '4. Set port 7497 (paper) or 7496 (live)\n'
                    '5. Run: python ibkr_strategy.py\n'
                    '6. Press Ctrl+C to stop\n'
                    '\n'
                    'WARNING: Test with paper trading first! Set PAPER_TRADING = True below.\n'
                    '"""\n'
                    '\n'
                    'from ib_insync import IB, Future, MarketOrder, LimitOrder, util\n'
                    'import time\n'
                    'import logging\n'
                    '\n'
                    'logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")\n'
                    'log = logging.getLogger(__name__)\n'
                    '\n'
                    '# ─────────────────────────────────────────────\n'
                    '# SETTINGS — edit these if needed\n'
                    '# ─────────────────────────────────────────────\n'
                    'PAPER_TRADING   = True          # True = paper (port 7497), False = live (port 7496)\n'
                    'HOST            = "127.0.0.1"\n'
                    'PORT            = 7497 if PAPER_TRADING else 7496\n'
                    'CLIENT_ID       = 1\n'
                    '\n'
                    '# Contract\n'
                    'SYMBOL          = "__IBKR_SYM__"\n'
                    'EXCHANGE        = "__IBKR_EXCH__"\n'
                    'CURRENCY        = "__IBKR_CCY__"\n'
                    'EXPIRY          = ""            # e.g. "20260600" for Jun-2026, leave "" for front month\n'
                    '\n'
                    '# Strategy parameters (from backtester)\n'
                    'DIRECTION       = "__DIR__"     # "Long" or "Short"\n'
                    'ENTRY_THRESHOLD = __THR__       # First entry when price __CMP_OP__ this (__UNIT__)\n'
                    'PYRAMID_STEP    = __STEP__      # Add 1 contract every X pts drop/rise\n'
                    'TAKE_PROFIT     = __TP__        # TP per contract: X pts above/below entry\n'
                    'QTY_PER_ENTRY   = 1             # Contracts per entry level\n'
                    'CHECK_INTERVAL  = 10            # Seconds between price checks\n'
                    '\n'
                    '# ─────────────────────────────────────────────\n'
                    '# STRATEGY STATE\n'
                    '# ─────────────────────────────────────────────\n'
                    'open_entries     = []\n'
                    'last_entry_price = None\n'
                    '\n'
                    'def get_contract():\n'
                    '    c = Future(symbol=SYMBOL, exchange=EXCHANGE, currency=CURRENCY)\n'
                    '    if EXPIRY:\n'
                    '        c.lastTradeDateOrContractMonth = EXPIRY\n'
                    '    return c\n'
                    '\n'
                    'def get_current_price(ib, contract):\n'
                    '    ticker = ib.reqMktData(contract, "", False, False)\n'
                    '    ib.sleep(2)\n'
                    '    price = ticker.last or ticker.close or ticker.bid\n'
                    '    ib.cancelMktData(contract)\n'
                    '    return float(price) if price and price > 0 else None\n'
                    '\n'
                    'def place_entry(ib, contract, entry_price):\n'
                    '    global last_entry_price\n'
                    '    log.info(f"[{DIRECTION}] Placing entry #{len(open_entries)+1} @ ~{entry_price:.4f}")\n'
                    '    entry_order = MarketOrder(action="__ACTION_ENTRY__", totalQuantity=QTY_PER_ENTRY)\n'
                    '    ib.placeOrder(contract, entry_order)\n'
                    '    ib.sleep(2)\n'
                    '    tp_price = entry_price + TAKE_PROFIT if DIRECTION == "Long" else entry_price - TAKE_PROFIT\n'
                    '    tp_order = LimitOrder(action="__ACTION_EXIT__", totalQuantity=QTY_PER_ENTRY, lmtPrice=round(tp_price, 4))\n'
                    '    tp_trade = ib.placeOrder(contract, tp_order)\n'
                    '    ib.sleep(1)\n'
                    '    open_entries.append({"entry_price": entry_price, "tp_price": tp_price, "tp_order_id": tp_trade.order.orderId})\n'
                    '    last_entry_price = entry_price\n'
                    '    log.info(f"  TP set at {tp_price:.4f}")\n'
                    '\n'
                    'def check_tp_fills(ib):\n'
                    '    still_open = []\n'
                    '    open_order_ids = {t.order.orderId for t in ib.openTrades()}\n'
                    '    for e in open_entries:\n'
                    '        if e["tp_order_id"] in open_order_ids:\n'
                    '            still_open.append(e)\n'
                    '        else:\n'
                    '            pnl = e["tp_price"] - e["entry_price"] if DIRECTION == "Long" else e["entry_price"] - e["tp_price"]\n'
                    '            log.info(f"[TP HIT] {e[\'entry_price\']:.4f} -> {e[\'tp_price\']:.4f} | PnL ~{pnl:.4f} pts")\n'
                    '    return still_open\n'
                    '\n'
                    'def run_strategy():\n'
                    '    global open_entries, last_entry_price\n'
                    '    ib = IB()\n'
                    '    log.info(f"Connecting to IBKR ({"PAPER" if PAPER_TRADING else "LIVE"}) on {HOST}:{PORT}")\n'
                    '    ib.connect(HOST, PORT, clientId=CLIENT_ID)\n'
                    '    log.info("Connected!")\n'
                    '    contract = get_contract()\n'
                    '    ib.qualifyContracts(contract)\n'
                    '    log.info(f"Contract: {contract}")\n'
                    '    log.info("=" * 55)\n'
                    '    log.info(f"Strategy : Pyramid __DIR__")\n'
                    '    log.info(f"Symbol   : {SYMBOL} on {EXCHANGE}")\n'
                    '    log.info(f"Threshold: {ENTRY_THRESHOLD} (__UNIT__)")\n'
                    '    log.info(f"Step     : {PYRAMID_STEP}  TP: {TAKE_PROFIT}")\n'
                    '    log.info("=" * 55)\n'
                    '    log.info("Press Ctrl+C to stop")\n'
                    '    try:\n'
                    '        while True:\n'
                    '            price = get_current_price(ib, contract)\n'
                    '            if price is None:\n'
                    '                log.warning("No price, retrying...")\n'
                    '                time.sleep(CHECK_INTERVAL)\n'
                    '                continue\n'
                    '            log.info(f"Price: {price:.4f} | Open: {len(open_entries)} | Last entry: {last_entry_price}")\n'
                    '            open_entries = check_tp_fills(ib)\n'
                    '            if not open_entries:\n'
                    '                last_entry_price = None\n'
                    '            if DIRECTION == "Long":\n'
                    '                if not open_entries and price < ENTRY_THRESHOLD:\n'
                    '                    place_entry(ib, contract, price)\n'
                    '                elif open_entries and last_entry_price is not None:\n'
                    '                    if price <= last_entry_price - PYRAMID_STEP:\n'
                    '                        place_entry(ib, contract, price)\n'
                    '            else:\n'
                    '                if not open_entries and price > ENTRY_THRESHOLD:\n'
                    '                    place_entry(ib, contract, price)\n'
                    '                elif open_entries and last_entry_price is not None:\n'
                    '                    if price >= last_entry_price + PYRAMID_STEP:\n'
                    '                        place_entry(ib, contract, price)\n'
                    '            time.sleep(CHECK_INTERVAL)\n'
                    '    except KeyboardInterrupt:\n'
                    '        log.info("Stopped. Open entries:")\n'
                    '        for e in open_entries:\n'
                    '            log.info(f"  Entry {e[\'entry_price\']:.4f}  TP {e[\'tp_price\']:.4f}")\n'
                    '    finally:\n'
                    '        ib.disconnect()\n'
                    '        log.info("Disconnected.")\n'
                    '\n'
                    'if __name__ == "__main__":\n'
                    '    util.startLoop()\n'
                    '    run_strategy()\n'
                )
                # Substitute placeholders with actual values
                ibkr_script = (
                    _script_template
                    .replace("__DIR__",         str(_dir))
                    .replace("__SYM__",         str(_sym))
                    .replace("__IBKR_SYM__",    str(ibkr_sym))
                    .replace("__IBKR_EXCH__",   str(ibkr_exch))
                    .replace("__IBKR_CCY__",    str(ibkr_ccy))
                    .replace("__TF__",          str(_tf))
                    .replace("__THR__",         str(_thr))
                    .replace("__STEP__",        str(_step))
                    .replace("__TP__",          str(_tp))
                    .replace("__COMM__",        str(_comm))
                    .replace("__CMP_OP__",      str(cmp_op))
                    .replace("__ADD_CMP__",     str(add_cmp))
                    .replace("__UNIT__",        str(ibkr_unit))
                    .replace("__ACTION_ENTRY__",str(action_entry))
                    .replace("__ACTION_EXIT__", str(action_exit))
                )

                st.markdown("""
                <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px;margin-bottom:16px'>
                    <div style='font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:8px'>🤖 IBKR Auto-Trading Script</div>
                    <div style='font-size:0.82rem;color:#94a3b8;line-height:1.6'>
                        This script monitors live prices through Interactive Brokers and automatically places
                        buy/sell orders according to your backtest parameters.<br>
                        <b style='color:#38bdf8'>Requirements:</b> IBKR account + TWS or IB Gateway running on your computer.
                        <b style='color:#fbbf24'>Always test with paper trading first!</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Parameter summary
                p_cols = st.columns(4)
                with p_cols[0]:
                    st.metric("Direction", _dir)
                with p_cols[1]:
                    st.metric("Threshold", f"{_thr}")
                with p_cols[2]:
                    st.metric("Step", f"{_step}")
                with p_cols[3]:
                    st.metric("Take Profit", f"{_tp}")

                st.markdown("---")
                st.caption("Copy the script below, save as `ibkr_strategy.py` and run with Python.")
                st.code(ibkr_script, language="python")

                # Download button
                st.download_button(
                    label="⬇️ Download ibkr_strategy.py",
                    data=ibkr_script,
                    file_name="ibkr_strategy.py",
                    mime="text/plain",
                    use_container_width=True,
                )

                st.markdown("""
                <div style='background:#0c2541;border:1px solid #0369a1;border-radius:8px;padding:14px 18px;margin-top:8px;font-size:0.82rem;color:#93c5fd;line-height:1.7'>
                    <b style='color:#38bdf8'>Quick start:</b><br>
                    1. pip install ib_insync<br>
                    2. Open TWS → Edit → Global Config → API → Enable Socket Client → Port 7497<br>
                    3. python ibkr_strategy.py<br>
                    4. Watch the terminal — it prints every entry and TP fill<br>
                    5. Press Ctrl+C to stop at any time
                </div>
                """, unsafe_allow_html=True)

            # ── Optimization ────────────────────────────────────
            if optimize_enabled:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
                    <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9">🔬 Optimization results</div>
                    <div style="font-size:0.75rem;color:#64748b">{commodity_name} · threshold {entry_threshold}$</div>
                </div>
                """, unsafe_allow_html=True)

                steps = np.arange(float(opt_step_min), float(opt_step_max) + float(opt_step_inc)*0.5, float(opt_step_inc))
                tps   = np.arange(float(opt_tp_min),   float(opt_tp_max)   + float(opt_tp_inc)*0.5,   float(opt_tp_inc))
                total_runs = len(steps) * len(tps)

                if total_runs > 300:
                    st.warning(f"Too many combinations ({total_runs}). Narrow the range or increase the step.")
                else:
                    opt_results = []
                    prog = st.progress(0, text="Computing combinations...")
                    for idx_s, s in enumerate(steps):
                        for idx_t, tp_val in enumerate(tps):
                            r = run_backtest(
                                df=df, entry_threshold=entry_threshold_disp,
                                pyramid_step=round(float(s), 4),
                                take_profit=round(float(tp_val), 4),
                                margin_per_contract=margin_per_contract_disp,
                                point_value=st.session_state.get("bt_point_value", 1.0),
                                commission_per_side=st.session_state.get("bt_commission", 0.0),
                                direction=st.session_state.get("bt_direction", "Long").lower(),
                            )
                            r_entries  = len(r.trades)
                            r_exits    = r.total_trades
                            r_ops      = r_entries + r_exits
                            r_comm     = st.session_state.get("bt_commission", 0.0)
                            comm_cost  = r_comm * r_ops
                            opt_results.append({
                                "Step ($)":        round(float(s), 2),
                                "TP ($)":          round(float(tp_val), 2),
                                "PnL ($)":         round(r.total_pnl, 2),
                                "Entries":         r_entries,
                                "Closed (TP)":     r_exits,
                                "Open":            r.open_trades,
                                "Total ops":       r_ops,
                                "Win %":           round(r.win_rate, 1),
                                "Total comm ($)":  round(comm_cost, 2),
                                 "Max contr.":      r.max_concurrent,
                                 "Max capital ($)": int(r.max_capital_with_unrealized),
                                 "Bal@peak contr.": round(float(r.balance_curve.loc[r.daily_open_contracts.idxmax()]) if r.max_concurrent > 0 else 0.0, 2),
                                 "Lowest bal.":     round(float(r.balance_curve.min()), 2),
                                 "Total needed":    int(int(r.daily_open_contracts.loc[r.balance_curve.idxmin()]) * margin_per_contract_disp + max(0.0, -float(r.balance_curve.min()))),
                                 "Avg days":        int(r.avg_days_open),
                            })
                            done = idx_s * len(tps) + idx_t + 1
                            prog.progress(done / total_runs, text=f"{done}/{total_runs} combinations...")
                    prog.empty()

                    opt_df = pd.DataFrame(opt_results)

                    # Apply max capital filter if set
                    max_cap_filter = st.session_state.get("opt_max_capital", 0.0)
                    if max_cap_filter > 0:
                        opt_df = opt_df[opt_df["Max capital ($)"] <= max_cap_filter].copy()

                    opt_df_sorted = opt_df.sort_values("PnL ($)", ascending=False).reset_index(drop=True)

                    medals = [("🥇","gold"), ("🥈","silver"), ("🥉","bronze")]
                    c1, c2, c3 = st.columns(3)
                    for col, (medal, cls) in zip([c1, c2, c3], medals):
                        idx = medals.index((medal, cls))
                        if len(opt_df_sorted) > idx:
                            row = opt_df_sorted.iloc[idx]
                            pnl_cls = "pos" if row["PnL ($)"] >= 0 else "neg"
                            with col:
                                border_col = {"gold":"#f59e0b","silver":"#94a3b8","bronze":"#cd7c2f"}.get(cls,"#334155")
                                pnl_col = "#34d399" if row["PnL ($)"] >= 0 else "#f87171"
                                s_card   = f"background:#1e293b;border:2px solid {border_col};border-radius:12px;padding:20px;text-align:center;height:100%"
                                s_medal  = "font-size:2rem;margin-bottom:8px"
                                s_params = "font-size:1.1rem;font-weight:700;color:#38bdf8;margin-bottom:8px"
                                s_pnl    = f"font-size:1.4rem;font-weight:800;color:{pnl_col};margin-bottom:8px"
                                s_stats  = "font-size:0.8rem;color:#94a3b8;line-height:1.6"
                                st.markdown(f'<div style="{s_card}"><div style="{s_medal}">{medal}</div><div style="{s_params}">Step {row["Step ($)"]}$ / TP {row["TP ($)"]}$</div><div style="{s_pnl}">${row["PnL ($)"]:,.2f}</div><div style="{s_stats}">{row["Entries"]} entries · {row["Closed (TP)"]} closed · Win {row["Win %"]}%<br>Max {row["Max contr."]} contr. · ${row["Max capital ($)"]:,}</div></div>', unsafe_allow_html=True)

                    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                    st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#94a3b8;margin-bottom:8px'>PnL Heatmap — Step vs Take Profit</div>", unsafe_allow_html=True)
                    pivot = opt_df.pivot(index="Step ($)", columns="TP ($)", values="PnL ($)")
                    fig_heat = px.imshow(
                        pivot,
                        labels=dict(x="Take Profit ($)", y="Step ($)", color="PnL ($)"),
                        color_continuous_scale="RdYlGn",
                        aspect="auto",
                        text_auto=True,
                    )
                    fig_heat.update_layout(
                        height=380, margin=dict(l=0,r=0,t=8,b=0),
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#94a3b8"),
                        coloraxis_colorbar=dict(title="PnL ($)", tickfont=dict(color="#94a3b8")),
                    )
                    fig_heat.update_traces(textfont=dict(size=11))
                    st.plotly_chart(fig_heat, use_container_width=True)

                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                    # ── Sort & display controls ───────────────────────────────
                    sort_col1, sort_col2, sort_col3 = st.columns([3, 2, 1])
                    with sort_col1:
                        sort_by = st.selectbox(
                            "Sort by",
                            options=[
                                "PnL ($)",
                                "Bal@peak contr.",
                                "Lowest bal.",
                                "Total needed",
                                "Max capital ($)",
                                "Commission",
                                "Entries",
                                "Closed (TP)",
                                "Avg days",
                                "Max contr.",
                            ],
                            index=0,
                            key="opt_sort_by",
                        )
                    with sort_col2:
                        sort_asc = st.radio(
                            "Order",
                            ["Best first (↓)", "Worst first (↑)"],
                            index=0,
                            horizontal=True,
                            key="opt_sort_order",
                        )
                    with sort_col3:
                        top_n_sel = st.selectbox(
                            "Show top",
                            options=[10, 20, 30, 50],
                            index=1,
                            key="opt_top_n",
                        )

                    # Map display sort column to actual df column
                    sort_col_map = {
                        "PnL ($)":          "PnL ($)",
                        "Bal@peak contr.":  "Bal@peak contr.",
                        "Lowest bal.":      "Lowest bal.",
                        "Total needed":     "Total needed",
                        "Max capital ($)":  "Max capital ($)",
                        "Commission":       "Total comm ($)",
                        "Entries":          "Entries",
                        "Closed (TP)":      "Closed (TP)",
                        "Avg days":         "Avg days",
                        "Max contr.":       "Max contr.",
                    }
                    ascending_when_best = {
                        "PnL ($)":          False,
                        "Bal@peak contr.":  False,
                        "Lowest bal.":      False,
                        "Total needed":     True,
                        "Max capital ($)":  True,
                        "Total comm ($)":   True,
                        "Entries":          False,
                        "Closed (TP)":      False,
                        "Avg days":         True,
                        "Max contr.":       True,
                    }
                    actual_col  = sort_col_map[sort_by]
                    best_first  = sort_asc == "Best first (↓)"
                    ascending   = not ascending_when_best[actual_col] if not best_first else ascending_when_best[actual_col]

                    # Sort ALL combinations then show top N
                    with st.spinner("Sorting and filtering..."):
                        opt_df_view  = opt_df.sort_values(actual_col, ascending=ascending).reset_index(drop=True)
                        total_combos = len(opt_df_view)

                        filter_note = ""
                        if max_cap_filter > 0:
                            orig_count     = len(pd.DataFrame(opt_results))
                            filtered_count = total_combos
                            filter_note    = f" · <span style='color:#fbbf24'>filtered: {filtered_count}/{orig_count} within ${max_cap_filter:,.0f}</span>"

                    if total_combos == 0:
                        st.warning(f"No combinations found within max capital ${max_cap_filter:,.0f}. Try increasing the limit.")
                    else:
                        st.markdown(
                            f"<div style='font-size:0.9rem;font-weight:600;color:#94a3b8;margin:8px 0 10px 0'>"
                            f"Showing top {min(top_n_sel, total_combos)} of {total_combos} combinations · sorted by <b style='color:#38bdf8'>{sort_by}</b>{filter_note}</div>",
                            unsafe_allow_html=True
                        )
                        st.caption("👆 Click a row below to load its parameters and re-run the backtest")

                        # Build display rows for selectbox
                        view_slice = opt_df_view.head(top_n_sel)
                        row_labels = []
                        for idx_r, r_row in view_slice.iterrows():
                            row_labels.append(
                                f"#{idx_r+1}  Step {r_row['Step ($)']}$ / TP {r_row['TP ($)']}$"
                                f"  →  PnL ${r_row['PnL ($)']:,.0f}"
                                f"  |  Capital ${r_row['Max capital ($)']:,}"
                            )

                        selected_row_label = st.selectbox(
                            "Select row to apply",
                            options=["— (no selection)"] + row_labels,
                            index=0,
                            key="opt_selected_row",
                            label_visibility="collapsed",
                        )

                        # If user selected a row — apply its params and re-run backtest
                        if selected_row_label != "— (no selection)":
                            sel_idx = row_labels.index(selected_row_label)
                            sel_row = view_slice.iloc[sel_idx]
                            new_step = float(sel_row["Step ($)"])
                            new_tp   = float(sel_row["TP ($)"])

                            # Only re-run if params actually changed
                            if (new_step != st.session_state.get("bt_step") or
                                    new_tp != st.session_state.get("bt_tp")):
                                with st.spinner(f"Running backtest with Step={new_step}$ / TP={new_tp}$..."):
                                    r_new = run_backtest(
                                        df=df,
                                        entry_threshold=entry_threshold_disp,
                                        pyramid_step=new_step,
                                        take_profit=new_tp,
                                        margin_per_contract=margin_per_contract_disp,
                                        point_value=st.session_state.get("bt_point_value", 1.0),
                                        commission_per_side=st.session_state.get("bt_commission", 0.0),
                                        direction=st.session_state.get("bt_direction", "Long").lower(),
                                    )
                                st.session_state["bt_result"] = r_new
                                st.session_state["bt_step"]   = new_step
                                st.session_state["bt_tp"]     = new_tp
                                st.success(f"Loaded: Step={new_step}$ / TP={new_tp}$ — scroll up to see updated chart & metrics")
                                st.rerun()

                        render_opt_table(opt_df_view, top_n=top_n_sel)

            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
