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

from backtester import fetch_data, run_backtest, trades_to_dataframe, COMMODITY_SYMBOLS, COMMODITY_CONTRACT_INFO, get_available_contracts, find_support_zones, compute_volume_profile

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
.main-header { padding: 24px 0 8px 0; border-bottom: 1px solid #1e293b; margin-bottom: 24px; }
.main-title { font-size: 1.75rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.03em; }
.main-subtitle { font-size: 0.9rem; color: #64748b; }
.strategy-tag { display: inline-block; background: #0c4a6e; color: #38bdf8 !important; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 6px; border: 1px solid #0369a1; margin-bottom: 12px; }
/* ── Info tooltip ── */
.info-wrap { position: relative; display: inline-block; }
.info-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 14px; height: 14px; border-radius: 50%;
    background: #334155; color: #94a3b8;
    font-size: 9px; font-weight: 700; font-style: normal;
    cursor: help; margin-left: 5px; vertical-align: middle;
    border: 1px solid #475569; flex-shrink: 0; line-height: 1;
}
.info-icon:hover { background: #475569; color: #e2e8f0; }
.info-tooltip {
    visibility: hidden; opacity: 0;
    position: absolute; z-index: 9999;
    bottom: 125%; left: 50%; transform: translateX(-50%);
    background: #1e293b; color: #cbd5e1;
    font-size: 0.75rem; font-weight: 400; line-height: 1.5;
    padding: 8px 12px; border-radius: 8px;
    border: 1px solid #334155;
    width: 220px; text-align: left;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    transition: opacity 0.15s ease;
    pointer-events: none;
}
.info-tooltip::after {
    content: "";
    position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: #334155;
}
.info-wrap:hover .info-tooltip { visibility: visible; opacity: 1; }
</style>
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }
.metric-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; text-align: center; }
.metric-card-label { font-size: 0.75rem; color: #94a3b8; font-weight: 600; margin-bottom: 6px; }
.metric-card-value { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; }
.metric-card-value.positive { color: #34d399; }
.metric-card-value.negative { color: #f87171; }
.metric-card-sub { font-size: 0.82rem; color: #cbd5e1; margin-top: 5px; font-weight: 500; }
.combo-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; text-align: center; height: 100%; }
.combo-card.gold { border-color: #f59e0b; }
.combo-card.silver { border-color: #94a3b8; }
.combo-card.bronze { border-color: #cd7c2f; }
.combo-medal { font-size: 2rem; margin-bottom: 8px; }
.combo-params { font-size: 1.1rem; font-weight: 700; color: #38bdf8; margin-bottom: 8px; }
.combo-pnl { font-size: 1.4rem; font-weight: 800; margin-bottom: 8px; }
.combo-pnl.pos { color: #34d399; }
.combo-pnl.neg { color: #f87171; }
.combo-stats { font-size: 0.8rem; color: #94a3b8; line-height: 1.6; }
.opt-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.opt-table th { background: #0f172a; color: #94a3b8; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; padding: 10px 14px; text-align: right; border-bottom: 1px solid #334155; }
.opt-table th:first-child, .opt-table th:nth-child(2) { text-align: center; }
.opt-table td { padding: 9px 14px; border-bottom: 1px solid #1e293b; text-align: right; color: #cbd5e1; }
.opt-table td:first-child, .opt-table td:nth-child(2) { text-align: center; font-weight: 600; }
.opt-table tr:hover td { background: #1e293b; }
.opt-table tr:first-child td { background: rgba(251,191,36,0.08); }
.opt-table tr:nth-child(2) td { background: rgba(148,163,184,0.06); }
.opt-table tr:nth-child(3) td { background: rgba(205,124,47,0.06); }
.pnl-pos { color: #34d399 !important; font-weight: 700; }
.pnl-neg { color: #f87171 !important; font-weight: 700; }
.rank-badge { display: inline-block; width: 24px; height: 24px; border-radius: 50%; font-size: 0.75rem; font-weight: 700; line-height: 24px; text-align: center; background: #334155; color: #94a3b8; }
.rank-1 { background: #78350f; color: #fbbf24; }
.rank-2 { background: #1e3a5f; color: #94a3b8; }
.rank-3 { background: #2d1b09; color: #cd7c2f; }
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] select { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 6px !important; color: #e2e8f0 !important; }
.stButton button[kind="primary"] { background: linear-gradient(135deg, #0ea5e9, #6366f1) !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; letter-spacing: 0.02em !important; padding: 10px 0 !important; transition: opacity 0.2s !important; }
.stButton button[kind="primary"]:hover { opacity: 0.9 !important; }
[data-testid="stTabs"] [role="tab"] { font-size: 0.85rem; font-weight: 600; color: #64748b; padding: 8px 16px; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #38bdf8; border-bottom: 2px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

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
    "Max capital req.":  "Maximum margin capital needed at the worst moment. Formula: Max contracts × margin per contract. This is the minimum account size you need to survive the worst drawdown.",
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
    "Max capital": "Max contracts × margin/contract = minimum account size needed to survive the worst drawdown.",
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
        ops_val  = row["Total ops"]       if "Total ops"       in row.index else 0
        comm_val = row["Total comm ($)"]  if "Total comm ($)"  in row.index else 0.0
        row_bg = {1: "background:rgba(251,191,36,0.08)", 2: "background:rgba(148,163,184,0.06)", 3: "background:rgba(205,124,47,0.06)"}.get(rank, "")
        rows_html += f"""<tr style="{row_bg}">
            <td style="{S_TD_CTR}"><span style="{rb_style}">{rank}</span></td>
            <td style="{S_TD_CTR}">{row['Step ($)']:.2f}</td>
            <td style="{S_TD_CTR}">{row['TP ($)']:.2f}</td>
            <td style="{S_TD};color:{pnl_col};font-weight:700">${pnl:,.2f}</td>
            <td style="{S_TD}">{row['Entries']}</td>
            <td style="{S_TD}">{row['Closed (TP)']}</td>
            <td style="{S_TD}">{row['Open']}</td>
            <td style="{S_TD};color:#fbbf24;font-weight:700">{ops_val}</td>
            <td style="{S_TD}">{row['Win %']:.1f}%</td>
            <td style="{S_TD};color:#f87171">${comm_val:,.2f}</td>
            <td style="{S_TD}">{row['Max contr.']}</td>
            <td style="{S_TD}">${row['Max capital ($)']:,}</td>
            <td style="{S_TD}">{row['Avg days']} d</td>
        </tr>"""
    html = f"""<table style="{S_TABLE}"><thead><tr>
            {th("#")}{th("Step")}{th("TP")}{th("PnL")}{th("Buys")}{th("Sells (TP)")}{th("Open")}
            {th("Total ops","#fbbf24")}{th("Win %")}{th("Commission","#f87171")}
            {th("Max contr.")}{th("Max capital")}{th("Avg days")}
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

    # ── Instrument ──────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Instrument</div>', unsafe_allow_html=True)
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
            with st.spinner("Loading available contracts..."):
                contracts = get_available_contracts(commodity_name, years_ahead=2)
            if not contracts:
                st.warning("No active contracts found. Using continuous contract.")
                symbol = COMMODITY_SYMBOLS[commodity_name]
            else:
                contract_labels = []
                for c in contracts:
                    oi_str = f"{c['open_interest']:,}" if c['open_interest'] else "?"
                    contract_labels.append(f"{c['name']}  |  {c['price']:.2f}  |  exp {c['expiry']}  |  OI: {oi_str}")
                selected_label = st.selectbox("Select contract", contract_labels, label_visibility="collapsed")
                selected_idx = contract_labels.index(selected_label)
                selected = contracts[selected_idx]
                symbol = selected["symbol"]
                ci = COMMODITY_CONTRACT_INFO[commodity_name]
                st.caption(f"Symbol: `{symbol}`")
                st.caption(f"Expires: {selected['expiry']}  |  OI: {selected['open_interest']:,}")
                unit_part = ci['unit'].split('/')[1] if '/' in ci['unit'] else 'units'
                st.caption(f"Tick: {ci['tick']} = ${ci['tick_value']}  |  Size: {ci['contract_size']:,} {unit_part}")

    _ci = COMMODITY_CONTRACT_INFO.get(commodity_name)
    if _ci and contract_mode == "Specific month":
        point_value = _ci["tick_value"] / _ci["tick"]
    else:
        point_value = 1.0

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Date range ──────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Backtest period</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=365*2), min_value=date(2000,1,1), max_value=date.today())
    with col2:
        end_date = st.date_input("To", value=date.today(), min_value=date(2000,1,2), max_value=date.today())
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Strategy parameters ─────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Strategy parameters</div>', unsafe_allow_html=True)
    entry_threshold = st.number_input("Entry threshold ($)", min_value=0.01, max_value=100000.0, value=470.0, step=1.0, format="%.2f", help="Buy when Low < this price")
    pyramid_step = st.number_input("Pyramid step ($)", min_value=0.01, max_value=10000.0, value=5.0, step=0.25, format="%.2f", help="Add one contract every X$ drop from last entry")
    take_profit = st.number_input("Take Profit ($)", min_value=0.01, max_value=10000.0, value=5.0, step=0.25, format="%.2f", help="Close each contract X$ above its entry price")
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

strategy_tab1, strategy_tab2 = st.tabs([
    "📈 Strategy 1 — Pyramid Long",
    "➕ Add strategy (coming soon)",
])

with strategy_tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("More strategies coming soon — Pyramid Short, Mean Reversion, Breakout, etc.")

with strategy_tab1:

    st.markdown('<div class="strategy-tag">Pyramid Long · Buy below threshold · TP per contract</div>', unsafe_allow_html=True)

    if run_button:
        df_new = None
        with st.spinner(f"Fetching data for {commodity_name}..."):
            try:
                df_new = fetch_data(symbol, start=start_date, end=end_date)
            except Exception as e:
                st.error(f"Data fetch error: {e}")
        if df_new is None or df_new.empty:
            st.error("No data returned. Check symbol or adjust date range.")
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
                )
            st.session_state["bt_df"]              = df_new
            st.session_state["bt_result"]          = result_new
            st.session_state["bt_symbol"]          = commodity_name
            st.session_state["bt_start"]           = str(start_date)
            st.session_state["bt_end"]             = str(end_date)
            st.session_state["bt_margin"]          = margin_per_contract
            st.session_state["bt_threshold"]       = entry_threshold
            st.session_state["bt_point_value"]     = point_value
            st.session_state["bt_commission"]      = commission_per_side

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

        pnl_currency = "USD" if pv > 1.0 else "pts"
        pv_info = f"point value: {pv:.0f} $/pt" if pv > 1.0 else "continuous contract (PnL in price points)"
        comm = st.session_state.get("bt_commission", 0.0)
        comm_info = f"commission: ${comm:.2f}/side (${comm*2:.2f} round-trip per contract)" if comm > 0 else "no commission"

        st.markdown(f"<div style='font-size:0.8rem;color:#64748b;margin-bottom:4px'>✓ {len(df)} sessions · {commodity_name_disp} · {st.session_state['bt_start']} → {st.session_state['bt_end']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.78rem;color:#475569;margin-bottom:12px'>💡 {pv_info} · {comm_info} — PnL in <b>{pnl_currency}</b></div>", unsafe_allow_html=True)

        # ── Metrics ────────────────────────────────────────────
        avg_pnl = result.total_pnl / result.total_trades if result.total_trades > 0 else 0
        pnl_pos = result.total_pnl >= 0
        period_high = float(df["High"].max())
        period_low  = float(df["Low"].min())
        high_date   = str(df["High"].idxmax())[:10]
        low_date    = str(df["Low"].idxmin())[:10]

        if True:
            cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0">'
            total_comm_paid = getattr(result, "total_commission", 0.0)
            total_entries   = len(result.trades)
            total_exits     = result.total_trades
            total_ops       = total_entries + total_exits
            cards_html += metric_card("Total PnL", f"${result.total_pnl:,.2f}", f"{'▲' if pnl_pos else '▼'} after commissions", positive=pnl_pos)
            cards_html += metric_card("Total commission", f"${total_comm_paid:,.2f}", f"{total_ops} ops × ${comm:.2f}")
            cards_html += metric_card("Total operations", str(total_ops), f"{total_entries} buys + {total_exits} sells (TP)")
            cards_html += metric_card("Closed trades", str(total_exits), f"{result.winning_trades}W / {result.losing_trades}L")
            cards_html += metric_card("Avg PnL / trade", f"${avg_pnl:.2f}", "per closed contract", positive=avg_pnl >= 0)
            cards_html += metric_card("Max contracts", str(result.max_concurrent), "open simultaneously")
            cards_html += metric_card("Max capital req.", f"${result.max_capital_needed:,.0f}", f"{result.max_concurrent} × ${margin_per_contract_disp:,.0f}")
            cards_html += metric_card("Open positions", str(result.open_trades), "not closed at end of period")
            cards_html += metric_card("Avg days to TP", f"{result.avg_days_open:.0f}", "average holding time")
            cards_html += metric_card("Period HIGH", f"${period_high:,.2f}", f"highest price · {high_date}")
            cards_html += metric_card("Period LOW", f"${period_low:,.2f}", f"lowest price · {low_date}")
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            # ── Charts ────────────────────────────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Equity Curve", "Open contracts", "Price + signals"])

            with chart_tab1:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=result.equity_curve.index, y=result.equity_curve.values,
                    mode="lines", name="PnL",
                    line=dict(color="#38bdf8", width=2),
                    fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
                ))
                fig_eq.add_hline(y=0, line_dash="dot", line_color="#475569", opacity=0.6)
                fig_eq.update_layout(
                    template="plotly_dark", height=360,
                    margin=dict(l=0,r=0,t=24,b=0),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    title=dict(text="Equity Curve — cumulative PnL", font=dict(size=13, color="#94a3b8")),
                    xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig_eq, use_container_width=True)

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
                    title=dict(text="Number of open contracts over time", font=dict(size=13, color="#94a3b8")),
                    xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig_open, use_container_width=True)

            with chart_tab3:
                entry_groups = collections.defaultdict(list)
                for t in result.trades:
                    entry_groups[t.entry_date].append(t.entry_price)

                tp_groups = collections.defaultdict(list)
                tp_pnl_groups = collections.defaultdict(float)
                for t in result.trades:
                    if t.closed:
                        tp_groups[t.exit_date].append(t.exit_price)
                        tp_pnl_groups[t.exit_date] += t.pnl

                fig_p = go.Figure()
                fig_p.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="Price",
                    increasing_line_color="#34d399", decreasing_line_color="#f87171",
                ))
                fig_p.add_hline(y=entry_threshold_disp, line_dash="dash",
                    line_color="#f87171", line_width=1.5,
                    annotation_text=f"Threshold {entry_threshold_disp}$",
                    annotation_position="right")

                if entry_groups:
                    ex   = list(entry_groups.keys())
                    ecnt = [len(v) for v in entry_groups.values()]
                    eavg = [sum(v)/len(v) for v in entry_groups.values()]
                    ey = []
                    price_range = float(df["High"].max() - df["Low"].min())
                    offset = price_range * 0.008
                    for d in ex:
                        if d in df.index:
                            ey.append(float(df.loc[d, "Low"]) - offset)
                        else:
                            ey.append(eavg[ex.index(d)])
                    etxt = [f"Buy x{c} @ {p:.2f}" for c, p in zip(ecnt, eavg)]
                    fig_p.add_trace(go.Scatter(
                        x=ex, y=ey, mode="markers", name="Buy",
                        marker=dict(symbol="triangle-up", size=8, color="#34d399", line=dict(width=0)),
                        hovertext=etxt, hoverinfo="text",
                    ))

                if tp_groups:
                    tx   = list(tp_groups.keys())
                    tcnt = [len(v) for v in tp_groups.values()]
                    tavg = [sum(v)/len(v) for v in tp_groups.values()]
                    tpnl = [tp_pnl_groups[d] for d in tx]
                    ty = []
                    price_range = float(df["High"].max() - df["Low"].min())
                    offset = price_range * 0.008
                    for d in tx:
                        if d in df.index:
                            ty.append(float(df.loc[d, "High"]) + offset)
                        else:
                            ty.append(tavg[tx.index(d)])
                    ttxt = [f"TP x{c} @ {p:.2f} | PnL +{pnl:.2f}" for c, p, pnl in zip(tcnt, tavg, tpnl)]
                    fig_p.add_trace(go.Scatter(
                        x=tx, y=ty, mode="markers", name="TP",
                        marker=dict(symbol="triangle-down", size=8, color="#fbbf24", line=dict(width=0)),
                        hovertext=ttxt, hoverinfo="text",
                    ))

                fig_p.update_layout(
                    template="plotly_dark", height=500,
                    margin=dict(l=0, r=0, t=24, b=0),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    xaxis_rangeslider_visible=False,
                    title=dict(text=f"{commodity_name_disp} — price with signals (hover for details)", font=dict(size=13, color="#94a3b8")),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b"),
                    legend=dict(orientation="h", y=1.02, x=0, font=dict(color="#94a3b8")),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            # ── Volume Profile ──────────────────────────────────
            st.markdown("<div style='margin-top:24px;border-top:1px solid #1e293b;padding-top:20px'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:12px'>🔥 Volume Profile — support zones</div>", unsafe_allow_html=True)

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
                st.markdown(f"<div style='font-size:0.8rem;color:#64748b;margin-bottom:12px'>✓ {len(vp_df_raw)} sessions · {vp_start} → {vp_end}</div>", unsafe_allow_html=True)

                kl = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0">'
                kl += metric_card("Point of Control", f"${poc:,.2f}", "price with highest volume")
                kl += metric_card("Value Area Low", f"${va_low:,.2f}", "lower bound of 70% vol ← entry signal", positive=True)
                kl += metric_card("Value Area High", f"${va_high:,.2f}", "upper bound of 70% vol")
                kl += metric_card("Period Low", f"${vp_period_low:,.2f}", "absolute minimum")
                kl += metric_card("Period High", f"${vp_period_high:,.2f}", "absolute maximum")
                kl += metric_card("Support zones", str(len(zones)), "detected VP levels")
                kl += '</div>'
                st.markdown(kl, unsafe_allow_html=True)

                fig_vp = go.Figure()
                fig_vp.add_trace(go.Candlestick(
                    x=vp_df_raw.index, open=vp_df_raw["Open"], high=vp_df_raw["High"],
                    low=vp_df_raw["Low"], close=vp_df_raw["Close"], name="Price",
                    increasing_line_color="#34d399", decreasing_line_color="#f87171",
                ))
                zone_colors = ["rgba(56,189,248,0.15)","rgba(251,191,36,0.12)","rgba(52,211,153,0.12)",
                               "rgba(248,113,113,0.12)","rgba(167,139,250,0.12)","rgba(249,115,22,0.12)"]
                border_colors = ["#38bdf8","#fbbf24","#34d399","#f87171","#a78bfa","#f97316"]
                for i, zone in enumerate(zones):
                    fig_vp.add_hrect(
                        y0=zone["zone_low"], y1=zone["zone_high"],
                        fillcolor=zone_colors[i % len(zone_colors)],
                        line=dict(color=border_colors[i % len(border_colors)], width=1, dash="dot"),
                        annotation_text=f"  {zone['price']:.2f} ({zone['volume_pct']:.1f}%)",
                        annotation_position="right",
                        annotation=dict(font=dict(color=border_colors[i % len(border_colors)], size=11)),
                    )
                fig_vp.add_hline(y=poc, line_dash="dash", line_color="#fbbf24", line_width=2,
                    annotation_text=f"POC {poc:.2f}", annotation_position="left",
                    annotation=dict(font=dict(color="#fbbf24", size=11)))
                fig_vp.add_hline(y=va_low, line_dash="dash", line_color="#34d399", line_width=2,
                    annotation_text=f"VA Low {va_low:.2f} ← entry", annotation_position="left",
                    annotation=dict(font=dict(color="#34d399", size=11)))
                fig_vp.add_hline(y=va_high, line_dash="dash", line_color="#38bdf8", line_width=1.5,
                    annotation_text=f"VA High {va_high:.2f}", annotation_position="left",
                    annotation=dict(font=dict(color="#38bdf8", size=11)))
                fig_vp.update_layout(
                    template="plotly_dark", height=520,
                    margin=dict(l=0,r=120,t=24,b=0),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    xaxis_rangeslider_visible=False,
                    title=dict(text=f"{commodity_name} — Volume Profile & support zones", font=dict(size=13, color="#94a3b8")),
                    xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig_vp, use_container_width=True)

                bar_colors = []
                zone_ranges = [(z["zone_low"], z["zone_high"]) for z in zones]
                for price in vp_data["price_level"]:
                    if abs(price - poc) < (vp_period_high - vp_period_low) / 300 * 4:
                        bar_colors.append("#fbbf24")
                    elif any(lo <= price <= hi for lo, hi in zone_ranges):
                        bar_colors.append("#38bdf8")
                    else:
                        bar_colors.append("#334155")

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=vp_data["pct"], y=vp_data["price_level"],
                    orientation="h", marker_color=bar_colors, marker_line_width=0,
                ))
                fig_hist.add_hline(y=poc,    line_dash="dash", line_color="#fbbf24", line_width=1.5)
                fig_hist.add_hline(y=va_low, line_dash="dash", line_color="#34d399", line_width=1.5)
                fig_hist.add_hline(y=va_high,line_dash="dash", line_color="#38bdf8", line_width=1)
                fig_hist.update_layout(
                    template="plotly_dark", height=380,
                    margin=dict(l=0,r=0,t=8,b=0),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    xaxis=dict(title="% volume", gridcolor="#1e293b"),
                    yaxis=dict(title="Price ($)", gridcolor="#1e293b"),
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

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
                                "Max capital ($)": int(r.max_capital_needed),
                                "Avg days":        int(r.avg_days_open),
                            })
                            done = idx_s * len(tps) + idx_t + 1
                            prog.progress(done / total_runs, text=f"{done}/{total_runs} combinations...")
                    prog.empty()

                    opt_df = pd.DataFrame(opt_results)
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

                    st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#94a3b8;margin:16px 0 10px 0'>Top 20 combinations</div>", unsafe_allow_html=True)
                    render_opt_table(opt_df_sorted, top_n=20)

            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
