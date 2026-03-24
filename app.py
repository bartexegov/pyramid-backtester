"""
app.py — Pyramid Long Backtester
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

from backtester import fetch_data, run_backtest, trades_to_dataframe, COMMODITY_SYMBOLS, find_support_zones, compute_volume_profile

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
# GLOBAL CSS — styl Tailwind-inspired
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* ── Sidebar sekcje ── */
.sidebar-header {
    background: #1e293b;
    padding: 20px 20px 16px 20px;
    border-bottom: 1px solid #334155;
}
.sidebar-logo {
    font-size: 1.25rem;
    font-weight: 700;
    color: #38bdf8 !important;
    letter-spacing: -0.02em;
}
.sidebar-logo span { color: #94a3b8 !important; font-weight: 400; font-size:0.9rem; }
.sidebar-section {
    padding: 16px 20px 8px 20px;
    border-bottom: 1px solid #1e293b;
}
.sidebar-section-title {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b !important;
    margin-bottom: 12px;
}
.sidebar-badge {
    display: inline-block;
    background: #0ea5e9;
    color: #fff !important;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 9999px;
    margin-left: 6px;
}

/* ── Main area ── */
.main-header {
    padding: 24px 0 8px 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 24px;
}
.main-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.03em;
}
.main-subtitle { font-size: 0.9rem; color: #64748b; }
.strategy-tag {
    display: inline-block;
    background: #0c4a6e;
    color: #38bdf8 !important;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid #0369a1;
    margin-bottom: 12px;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-card-label {
    font-size: 0.75rem;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-card-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.metric-card-value.positive { color: #34d399; }
.metric-card-value.negative { color: #f87171; }
.metric-card-sub { font-size: 0.82rem; color: #cbd5e1; margin-top: 5px; font-weight: 500; }

/* ── Top combo cards ── */
.combo-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    height: 100%;
}
.combo-card.gold { border-color: #f59e0b; }
.combo-card.silver { border-color: #94a3b8; }
.combo-card.bronze { border-color: #cd7c2f; }
.combo-medal { font-size: 2rem; margin-bottom: 8px; }
.combo-params { font-size: 1.1rem; font-weight: 700; color: #38bdf8; margin-bottom: 8px; }
.combo-pnl { font-size: 1.4rem; font-weight: 800; margin-bottom: 8px; }
.combo-pnl.pos { color: #34d399; }
.combo-pnl.neg { color: #f87171; }
.combo-stats { font-size: 0.8rem; color: #94a3b8; line-height: 1.6; }

/* ── Optimization table ── */
.opt-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.opt-table th {
    background: #0f172a;
    color: #94a3b8;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 10px 14px;
    text-align: right;
    border-bottom: 1px solid #334155;
}
.opt-table th:first-child, .opt-table th:nth-child(2) { text-align: center; }
.opt-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #1e293b;
    text-align: right;
    color: #cbd5e1;
}
.opt-table td:first-child, .opt-table td:nth-child(2) { text-align: center; font-weight: 600; }
.opt-table tr:hover td { background: #1e293b; }
.opt-table tr:first-child td { background: rgba(251,191,36,0.08); }
.opt-table tr:nth-child(2) td { background: rgba(148,163,184,0.06); }
.opt-table tr:nth-child(3) td { background: rgba(205,124,47,0.06); }
.pnl-pos { color: #34d399 !important; font-weight: 700; }
.pnl-neg { color: #f87171 !important; font-weight: 700; }
.rank-badge {
    display: inline-block;
    width: 24px; height: 24px;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    line-height: 24px;
    text-align: center;
    background: #334155;
    color: #94a3b8;
}
.rank-1 { background: #78350f; color: #fbbf24; }
.rank-2 { background: #1e3a5f; color: #94a3b8; }
.rank-3 { background: #2d1b09; color: #cd7c2f; }

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 10px 0 !important;
    transition: opacity 0.2s !important;
}
.stButton button[kind="primary"]:hover { opacity: 0.9 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
    padding: 8px 16px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #38bdf8;
    border-bottom: 2px solid #38bdf8;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def metric_card(label, value, sub="", positive=None):
    cls = ""
    if positive is True:
        cls = "positive"
    elif positive is False:
        cls = "negative"
    return f"""
    <div class="metric-card">
        <div class="metric-card-label">{label}</div>
        <div class="metric-card-value {cls}">{value}</div>
        <div class="metric-card-sub">{sub}</div>
    </div>"""


def render_opt_table(df: pd.DataFrame, top_n: int = 20):
    rows_html = ""
    for i, (_, row) in enumerate(df.head(top_n).iterrows()):
        rank = i + 1
        rank_cls = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "")
        pnl = row["PnL ($)"]
        pnl_cls = "pnl-pos" if pnl >= 0 else "pnl-neg"
        rows_html += f"""
        <tr>
            <td><span class="rank-badge {rank_cls}">{rank}</span></td>
            <td>{row['Krok ($)']:.2f} $</td>
            <td>{row['TP ($)']:.2f} $</td>
            <td class="{pnl_cls}">${pnl:,.2f}</td>
            <td>{row['Transakcji']}</td>
            <td>{row['Win %']:.1f}%</td>
            <td>{row['Max kontr.']}</td>
            <td>${row['Max kapital ($)']:,}</td>
            <td>{row['Sr. dni']} dni</td>
        </tr>"""
    html = f"""
    <table class="opt-table">
        <thead><tr>
            <th>#</th><th>Krok</th><th>TP</th>
            <th>PnL</th><th>Transakcji</th><th>Win %</th>
            <th>Max kontr.</th><th>Max kapital</th><th>Sr. dni</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>"""
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">
            📊 Pyramid Backtester
            <br><span>Testuj strategie na surowcach</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Instrument ──────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Instrument</div>', unsafe_allow_html=True)
    commodity_name = st.selectbox("Surowiec", options=list(COMMODITY_SYMBOLS.keys()), index=0, label_visibility="collapsed")
    symbol = COMMODITY_SYMBOLS[commodity_name]
    st.caption(f"Symbol: `{symbol}`")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Okres ───────────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Okres backtestów</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Od", value=date.today() - timedelta(days=365*2), min_value=date(2000,1,1), max_value=date.today())
    with col2:
        end_date = st.date_input("Do", value=date.today(), min_value=date(2000,1,2), max_value=date.today())
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Strategia ───────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Parametry strategii</div>', unsafe_allow_html=True)
    entry_threshold = st.number_input("Cena wejścia ($)", min_value=0.01, max_value=100000.0, value=470.0, step=1.0, format="%.2f", help="Kupuj gdy Low < tej ceny")
    pyramid_step = st.number_input("Krok dokupowania ($)", min_value=0.01, max_value=10000.0, value=5.0, step=0.25, format="%.2f", help="Co ile $ w dół dokupujesz kolejny kontrakt")
    take_profit = st.number_input("Take Profit ($)", min_value=0.01, max_value=10000.0, value=5.0, step=0.25, format="%.2f", help="Ile $ powyżej ceny kupna zamykasz kontrakt")
    margin_per_contract = st.number_input("Margin / kontrakt ($)", min_value=100.0, max_value=1000000.0, value=1500.0, step=100.0, format="%.0f", help="Depozyt zabezpieczający u brokera")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Optymalizacja ────────────────────────────────────────
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Optymalizacja kroków</div>', unsafe_allow_html=True)
    optimize_enabled = st.checkbox("Włącz optymalizację", value=False, help="Przetestuj wszystkie kombinacje krok × TP automatycznie")

    opt_step_min = 2.0
    opt_step_max = 10.0
    opt_step_inc = 1.0
    opt_tp_min   = 2.0
    opt_tp_max   = 10.0
    opt_tp_inc   = 1.0

    if optimize_enabled:
        st.caption("**Krok dokupowania**")
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            opt_step_min = st.number_input("Od", value=2.0, step=0.25, format="%.2f", key="smin")
        with cs2:
            opt_step_max = st.number_input("Do", value=10.0, step=0.25, format="%.2f", key="smax")
        with cs3:
            opt_step_inc = st.number_input("Co", value=1.0, step=0.25, format="%.2f", key="sinc")

        st.caption("**Take Profit**")
        ct1, ct2, ct3 = st.columns(3)
        with ct1:
            opt_tp_min = st.number_input("Od", value=2.0, step=0.25, format="%.2f", key="tmin")
        with ct2:
            opt_tp_max = st.number_input("Do", value=10.0, step=0.25, format="%.2f", key="tmax")
        with ct3:
            opt_tp_inc = st.number_input("Co", value=1.0, step=0.25, format="%.2f", key="tinc")

        n_s = max(1, round((opt_step_max - opt_step_min) / opt_step_inc) + 1)
        n_t = max(1, round((opt_tp_max - opt_tp_min) / opt_tp_inc) + 1)
        st.caption(f"Łącznie: **{n_s * n_t} kombinacji**")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    run_button = st.button("▶ Uruchom backtest", type="primary", use_container_width=True)
    st.markdown("<div style='padding:0 20px 20px 20px'><p style='font-size:0.7rem;color:#475569;text-align:center;margin-top:8px'>Dane: Yahoo Finance · tylko edukacyjnie</p></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN — zakładki strategii
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <div class="main-title">Pyramid Backtester</div>
    <div class="main-subtitle">Analiza strategii pyramidingu na kontraktach futures</div>
</div>
""", unsafe_allow_html=True)

# Zakładki
strategy_tab1, strategy_tab2 = st.tabs([
    "📈 Strategia 1 — Pyramid Long",
    "➕ Dodaj strategię (wkrótce)",
])

with strategy_tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Tu pojawią się kolejne strategie. Np. Pyramid Short, Mean Reversion, Breakout itp.")

with strategy_tab1:

    st.markdown('<div class="strategy-tag">Pyramid Long · Kupuj poniżej progu · TP per kontrakt</div>', unsafe_allow_html=True)

    # Uruchom backtest i zapisz wyniki w session_state
    if run_button:
        df_new = None
        with st.spinner(f"Pobieranie danych {commodity_name}..."):
            try:
                df_new = fetch_data(symbol, start=start_date, end=end_date)
            except Exception as e:
                st.error(f"Błąd pobierania danych: {e}")

        if df_new is None or df_new.empty:
            st.error("Brak danych. Sprawdź symbol lub zmień zakres dat.")
        else:
            with st.spinner("Obliczanie..."):
                result_new = run_backtest(
                    df=df_new,
                    entry_threshold=entry_threshold,
                    pyramid_step=pyramid_step,
                    take_profit=take_profit,
                    margin_per_contract=margin_per_contract,
                    qty_per_entry=1,
                )
            st.session_state["bt_df"]     = df_new
            st.session_state["bt_result"] = result_new
            st.session_state["bt_symbol"] = commodity_name
            st.session_state["bt_start"]  = str(start_date)
            st.session_state["bt_end"]    = str(end_date)
            st.session_state["bt_margin"] = margin_per_contract
            st.session_state["bt_threshold"] = entry_threshold

    # Renderuj wyniki z session_state (przetrwają każdy rerun)
    if "bt_result" not in st.session_state:
        st.markdown("""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:32px;text-align:center;margin-top:24px">
            <div style="font-size:2.5rem;margin-bottom:12px">👈</div>
            <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:8px">Ustaw parametry i uruchom backtest</div>
            <div style="font-size:0.85rem;color:#64748b">Wybierz surowiec, daty i parametry strategii w panelu po lewej,<br>następnie kliknij <b>Uruchom backtest</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df     = st.session_state["bt_df"]
        result = st.session_state["bt_result"]
        commodity_name_disp  = st.session_state["bt_symbol"]
        margin_per_contract_disp = st.session_state["bt_margin"]
        entry_threshold_disp = st.session_state["bt_threshold"]

        st.markdown(f"<div style='font-size:0.8rem;color:#64748b;margin-bottom:16px'>✓ {len(df)} sesji · {commodity_name_disp} · {st.session_state['bt_start']} → {st.session_state['bt_end']}</div>", unsafe_allow_html=True)

        # ── Metryki ────────────────────────────────────────────
        avg_pnl = result.total_pnl / result.total_trades if result.total_trades > 0 else 0
        pnl_pos = result.total_pnl >= 0
        period_high = float(df["High"].max())
        period_low  = float(df["Low"].min())
        high_date   = str(df["High"].idxmax())[:10]
        low_date    = str(df["Low"].idxmin())[:10]

        if True:
            cards_html = '<div class="metric-grid">'
            cards_html += metric_card("Zysk / Strata", f"${result.total_pnl:,.2f}", f"{'▲' if pnl_pos else '▼'} całkowity PnL", positive=pnl_pos)
            cards_html += metric_card("Transakcji", str(result.total_trades), f"{result.winning_trades}W / {result.losing_trades}L")
            cards_html += metric_card("Win Rate", f"{result.win_rate:.1f}%", "procent wygranych", positive=result.win_rate >= 50)
            cards_html += metric_card("Śr. PnL / transakcję", f"${avg_pnl:.2f}", "per zamknięty kontrakt", positive=avg_pnl >= 0)
            cards_html += metric_card("Max kontraktów", str(result.max_concurrent), "jednocześnie otwartych")
            cards_html += metric_card("Max kapitał", f"${result.max_capital_needed:,.0f}", f"{result.max_concurrent} kontr. × ${margin_per_contract_disp:,.0f}")
            cards_html += metric_card("Otwarte pozycje", str(result.open_trades), "niezamknięte na koniec")
            cards_html += metric_card("Śr. dni do TP", f"{result.avg_days_open:.0f}", "średni czas trzymania")
            cards_html += metric_card("MAX HIGH okresu", f"${period_high:,.2f}", f"najwyższa cena · {high_date}")
            cards_html += metric_card("MIN LOW okresu", f"${period_low:,.2f}", f"najniższa cena · {low_date}")
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            # ── Wykresy ────────────────────────────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Equity Curve", "Otwarte kontrakty", "Cena + sygnały"])

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
                    title=dict(text="Equity Curve — skumulowany PnL", font=dict(size=13, color="#94a3b8")),
                    xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig_eq, use_container_width=True)

            with chart_tab2:
                fig_open = go.Figure()
                fig_open.add_trace(go.Scatter(
                    x=result.daily_open_contracts.index, y=result.daily_open_contracts.values,
                    mode="lines", name="Kontrakty",
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
                    title=dict(text="Liczba otwartych kontraktów w czasie", font=dict(size=13, color="#94a3b8")),
                    xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig_open, use_container_width=True)

            with chart_tab3:
                fig_p = go.Figure()
                fig_p.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    name="Cena",
                    increasing_line_color="#34d399", decreasing_line_color="#f87171",
                ))
                fig_p.add_hline(y=entry_threshold_disp, line_dash="dash", line_color="#f87171", line_width=1.5,
                    annotation_text=f"Próg: {entry_threshold_disp}", annotation_position="right")
                fig_p.add_trace(go.Scatter(
                    x=[t.entry_date for t in result.trades],
                    y=[t.entry_price for t in result.trades],
                    mode="markers", name="Kupno",
                    marker=dict(symbol="triangle-up", size=9, color="#34d399"),
                ))
                fig_p.add_trace(go.Scatter(
                    x=[t.exit_date for t in result.trades if t.closed],
                    y=[t.exit_price for t in result.trades if t.closed],
                    mode="markers", name="TP",
                    marker=dict(symbol="triangle-down", size=9, color="#fbbf24"),
                ))
                fig_p.update_layout(
                    template="plotly_dark", height=460,
                    margin=dict(l=0,r=0,t=24,b=0),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    xaxis_rangeslider_visible=False,
                    title=dict(text=f"{commodity_name} — wejścia i wyjścia", font=dict(size=13, color="#94a3b8")),
                    xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            # ── Volume Profile — sekcja pod wykresami (nie w tab, bo button resetuje tab) ──
            st.markdown("<div style='margin-top:24px;border-top:1px solid #1e293b;padding-top:20px'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:12px'>🔥 Volume Profile — strefy wsparcia</div>", unsafe_allow_html=True)

            if "vp_days" not in st.session_state:
                st.session_state["vp_days"] = 365 * 10

            st.markdown("<div style='font-size:0.75rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px'>Okres Volume Profile</div>", unsafe_allow_html=True)
            qc = st.columns(5)
            quick_labels = ["1 rok", "5 lat", "10 lat", "20 lat", "Max"]
            quick_days   = [365, 365*5, 365*10, 365*20, 365*40]
            for i, (qcol, qlabel, qday) in enumerate(zip(qc, quick_labels, quick_days)):
                with qcol:
                    if st.button(qlabel, key=f"vp_q_{i}", use_container_width=True):
                        st.session_state["vp_days"] = qday

            vp_start_default = date.today() - timedelta(days=st.session_state["vp_days"])
            vp_dc1, vp_dc2 = st.columns(2)
            with vp_dc1:
                vp_start = st.date_input("Od", value=vp_start_default, key="vp_start2", min_value=date(1990,1,1), max_value=date.today())
            with vp_dc2:
                vp_end = st.date_input("Do", value=date.today(), key="vp_end2", min_value=date(1990,1,2), max_value=date.today())

            with st.spinner("Obliczanie Volume Profile..."):
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
                st.error("Brak danych VP. Zmień zakres dat.")
            else:
                vp_period_high = float(vp_df_raw["High"].max())
                vp_period_low  = float(vp_df_raw["Low"].min())

                st.markdown(f"<div style='font-size:0.8rem;color:#64748b;margin-bottom:12px'>✓ {len(vp_df_raw)} sesji · {vp_start} → {vp_end}</div>", unsafe_allow_html=True)

                # Kluczowe poziomy
                kl = '<div class="metric-grid">'
                kl += metric_card("Point of Control", f"${poc:,.2f}", "cena z najwyższym wolumenem")
                kl += metric_card("Value Area Low", f"${va_low:,.2f}", "dolna granica 70% vol ← wejście", positive=True)
                kl += metric_card("Value Area High", f"${va_high:,.2f}", "górna granica 70% vol")
                kl += metric_card("Min Low okresu", f"${vp_period_low:,.2f}", "absolutne minimum")
                kl += metric_card("Max High okresu", f"${vp_period_high:,.2f}", "absolutne maksimum")
                kl += metric_card("Stref wsparcia", str(len(zones)), "wykrytych poziomów")
                kl += '</div>'
                st.markdown(kl, unsafe_allow_html=True)

                # Wykres ceny ze strefami
                fig_vp = go.Figure()
                fig_vp.add_trace(go.Candlestick(
                    x=vp_df_raw.index, open=vp_df_raw["Open"], high=vp_df_raw["High"],
                    low=vp_df_raw["Low"], close=vp_df_raw["Close"], name="Cena",
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
                        annotation_text=f"  {zone['price']:.2f}$ ({zone['volume_pct']:.1f}%)",
                        annotation_position="right",
                        annotation=dict(font=dict(color=border_colors[i % len(border_colors)], size=11)),
                    )
                fig_vp.add_hline(y=poc, line_dash="dash", line_color="#fbbf24", line_width=2,
                    annotation_text=f"POC {poc:.2f}$", annotation_position="left",
                    annotation=dict(font=dict(color="#fbbf24", size=11)))
                fig_vp.add_hline(y=va_low, line_dash="dash", line_color="#34d399", line_width=2,
                    annotation_text=f"VA Low {va_low:.2f}$ ← wejście", annotation_position="left",
                    annotation=dict(font=dict(color="#34d399", size=11)))
                fig_vp.add_hline(y=va_high, line_dash="dash", line_color="#38bdf8", line_width=1.5,
                    annotation_text=f"VA High {va_high:.2f}$", annotation_position="left",
                    annotation=dict(font=dict(color="#38bdf8", size=11)))
                fig_vp.update_layout(
                    template="plotly_dark", height=520,
                    margin=dict(l=0,r=120,t=24,b=0),
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    xaxis_rangeslider_visible=False,
                    title=dict(text=f"{commodity_name} — Volume Profile i strefy wsparcia", font=dict(size=13, color="#94a3b8")),
                    xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig_vp, use_container_width=True)

                # VP histogram poziomy
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
                    xaxis=dict(title="% wolumenu", gridcolor="#1e293b"),
                    yaxis=dict(title="Cena ($)", gridcolor="#1e293b"),
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # Tabela stref
                zone_rows = ""
                for z in zones:
                    if z["price"] <= va_low:
                        sig = "<span style='color:#34d399;font-weight:700'>✓ Strefa wejścia</span>"
                    elif z["price"] <= poc:
                        sig = "<span style='color:#fbbf24'>~ Poniżej POC</span>"
                    else:
                        sig = "<span style='color:#64748b'>Powyżej POC</span>"
                    zone_rows += f"<tr><td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:700'>${z['price']:,.2f}</td><td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#94a3b8'>${z['zone_low']:,.2f} – ${z['zone_high']:,.2f}</td><td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#38bdf8;font-weight:600'>{z['volume_pct']:.1f}%</td><td style='padding:8px 12px;border-bottom:1px solid #1e293b'>{sig}</td></tr>"

                st.markdown(f"<table style='width:100%;border-collapse:collapse;font-size:0.82rem'><thead><tr><th style='padding:9px 12px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>Cena</th><th style='padding:9px 12px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>Zakres</th><th style='padding:9px 12px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>% Vol</th><th style='padding:9px 12px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>Sygnał</th></tr></thead><tbody>{zone_rows}</tbody></table>", unsafe_allow_html=True)

                st.markdown(f"<div style='background:#0c2541;border:1px solid #0369a1;border-radius:8px;padding:14px 18px;margin-top:16px;font-size:0.85rem;color:#93c5fd;line-height:1.7'><b style='color:#38bdf8'>💡 Wskazówka:</b> Ustaw <b>Cena wejścia</b> w Strategii 1 na <b style='color:#34d399'>${va_low:,.2f}$</b> (Value Area Low) — to historycznie tania strefa dla {commodity_name}.</div>", unsafe_allow_html=True)

            # ── Optymalizacja ──────────────────────────────────
            if optimize_enabled:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
                    <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9">🔬 Optymalizacja kombinacji</div>
                    <div style="font-size:0.75rem;color:#64748b">{commodity_name} · próg {entry_threshold}$</div>
                </div>
                """, unsafe_allow_html=True)

                steps = np.arange(float(opt_step_min), float(opt_step_max) + float(opt_step_inc)*0.5, float(opt_step_inc))
                tps   = np.arange(float(opt_tp_min),   float(opt_tp_max)   + float(opt_tp_inc)*0.5,   float(opt_tp_inc))
                total_runs = len(steps) * len(tps)

                if total_runs > 300:
                    st.warning(f"Za dużo kombinacji ({total_runs}). Zawęź zakres lub zwiększ krok.")
                else:
                    opt_results = []
                    prog = st.progress(0, text="Obliczanie kombinacji...")
                    for idx_s, s in enumerate(steps):
                        for idx_t, tp_val in enumerate(tps):
                            r = run_backtest(
                                df=df, entry_threshold=entry_threshold_disp,
                                pyramid_step=round(float(s), 4),
                                take_profit=round(float(tp_val), 4),
                                margin_per_contract=margin_per_contract_disp,
                            )
                            opt_results.append({
                                "Krok ($)":        round(float(s), 2),
                                "TP ($)":          round(float(tp_val), 2),
                                "PnL ($)":         round(r.total_pnl, 2),
                                "Transakcji":      r.total_trades,
                                "Win %":           round(r.win_rate, 1),
                                "Max kontr.":      r.max_concurrent,
                                "Max kapital ($)": int(r.max_capital_needed),
                                "Sr. dni":         int(r.avg_days_open),
                            })
                            done = idx_s * len(tps) + idx_t + 1
                            prog.progress(done / total_runs, text=f"{done}/{total_runs} kombinacji...")
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
                                st.markdown(f"""
                                <div class="combo-card {cls}">
                                    <div class="combo-medal">{medal}</div>
                                    <div class="combo-params">Krok {row['Krok ($)']}$ / TP {row['TP ($)']}$</div>
                                    <div class="combo-pnl {pnl_cls}">${row['PnL ($)']:,.2f}</div>
                                    <div class="combo-stats">
                                        {row['Transakcji']} transakcji · Win {row['Win %']}%<br>
                                        Max {row['Max kontr.']} kontr. · ${row['Max kapital ($)']:,}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

                    st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#94a3b8;margin-bottom:8px'>Heatmapa PnL — Krok vs Take Profit</div>", unsafe_allow_html=True)
                    pivot = opt_df.pivot(index="Krok ($)", columns="TP ($)", values="PnL ($)")
                    fig_heat = px.imshow(
                        pivot,
                        labels=dict(x="Take Profit ($)", y="Krok ($)", color="PnL ($)"),
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

                    st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#94a3b8;margin:16px 0 10px 0'>Top 20 kombinacji</div>", unsafe_allow_html=True)
                    render_opt_table(opt_df_sorted, top_n=20)

            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

# (stara zakładka VP usunieta - VP przeniesione do chart_tab4)
if False:
    st.markdown('<div class="strategy-tag">Volume Profile · Point of Control · Value Area · Strefy wsparcia</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px;margin:12px 0 20px 0;font-size:0.85rem;color:#94a3b8;line-height:1.7">
        <b style="color:#e2e8f0">Jak to działa:</b> Volume Profile pokazuje na jakich poziomach cenowych skupił się największy wolumen w historii.
        Strefy z wysokim wolumenem to naturalne wsparcia i opory — cena ma tendencję do powracania do tych poziomów.
        <br><b style="color:#38bdf8">POC</b> (Point of Control) = cena z absolutnie najwyższym wolumenem.
        <br><b style="color:#34d399">Value Area Low</b> = dolna granica strefy obejmującej 70% wolumenu wokół POC — <b style="color:#fbbf24">to jest Twój sygnał do wejścia w strategię.</b>
    </div>
    """, unsafe_allow_html=True)

    # ── Wybor instrumentu i okresu VP ─────────────────────────
    vp_col1, vp_col2 = st.columns([2, 3])

    with vp_col1:
        vp_symbol_name = st.selectbox("Instrument", options=list(COMMODITY_SYMBOLS.keys()), index=0, key="vp_symbol")
        vp_symbol = COMMODITY_SYMBOLS[vp_symbol_name]

    with vp_col2:
        st.markdown("<div style='font-size:0.75rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px'>Okres historyczny</div>", unsafe_allow_html=True)
        quick_cols = st.columns(5)
        quick_labels = ["1 rok", "5 lat", "10 lat", "20 lat", "Max"]
        quick_days   = [365, 365*5, 365*10, 365*20, 365*40]
        vp_quick = None
        for i, (col, label) in enumerate(zip(quick_cols, quick_labels)):
            with col:
                if st.button(label, key=f"vp_quick_{i}", use_container_width=True):
                    vp_quick = quick_days[i]

        vp_date_col1, vp_date_col2 = st.columns(2)
        with vp_date_col1:
            default_vp_start = date.today() - timedelta(days=365*10)
            if vp_quick:
                default_vp_start = date.today() - timedelta(days=vp_quick)
            vp_start = st.date_input("Od", value=default_vp_start, key="vp_start", min_value=date(1990,1,1), max_value=date.today())
        with vp_date_col2:
            vp_end = st.date_input("Do", value=date.today(), key="vp_end", min_value=date(1990,1,2), max_value=date.today())

    vp_run = st.button("🔍 Analizuj strefy wsparcia", type="primary", use_container_width=False, key="vp_run")

    if vp_run:
        with st.spinner(f"Pobieranie danych {vp_symbol_name}..."):
            try:
                vp_df_raw = fetch_data(vp_symbol, start=vp_start, end=vp_end)
            except Exception as e:
                st.error(f"Błąd: {e}")
                vp_df_raw = None

        if vp_df_raw is None or vp_df_raw.empty:
            st.error("Brak danych. Zmień zakres dat lub instrument.")
        else:
            with st.spinner("Obliczanie Volume Profile..."):
                zones, poc, va_low, va_high = find_support_zones(vp_df_raw, bins=300, top_n=6, min_gap_pct=0.025)
                vp_data, _, _, _ = compute_volume_profile(vp_df_raw, bins=300)

            period_low  = float(vp_df_raw["Low"].min())
            period_high = float(vp_df_raw["High"].max())

            st.markdown(f"<div style='font-size:0.8rem;color:#64748b;margin-bottom:16px'>✓ {len(vp_df_raw)} sesji · {vp_symbol_name} · {vp_start} → {vp_end}</div>", unsafe_allow_html=True)

            # ── Kluczowe poziomy ──────────────────────────────
            kl_cards = '<div class="metric-grid">'
            kl_cards += metric_card("Point of Control", f"${poc:,.2f}", "cena z najwyższym wolumenem")
            kl_cards += metric_card("Value Area Low", f"${va_low:,.2f}", "dolna granica 70% wolumenu ← wejście", positive=True)
            kl_cards += metric_card("Value Area High", f"${va_high:,.2f}", "górna granica 70% wolumenu")
            kl_cards += metric_card("Min Low okresu", f"${period_low:,.2f}", "absolutne minimum w okresie")
            kl_cards += metric_card("Max High okresu", f"${period_high:,.2f}", "absolutne maksimum w okresie")
            kl_cards += metric_card("Stref wsparcia", str(len(zones)), "wykrytych poziomów VP")
            kl_cards += '</div>'
            st.markdown(kl_cards, unsafe_allow_html=True)

            # ── Wykres ceny z Volume Profile i strefami ───────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            fig_vp = go.Figure()

            # Wykres swiecowy ceny
            fig_vp.add_trace(go.Candlestick(
                x=vp_df_raw.index,
                open=vp_df_raw["Open"],
                high=vp_df_raw["High"],
                low=vp_df_raw["Low"],
                close=vp_df_raw["Close"],
                name="Cena",
                increasing_line_color="#34d399",
                decreasing_line_color="#f87171",
                xaxis="x", yaxis="y",
            ))

            # Strefy wsparcia jako prostokaty
            colors_zone = ["rgba(56,189,248,0.15)", "rgba(251,191,36,0.12)",
                           "rgba(52,211,153,0.12)", "rgba(248,113,113,0.12)",
                           "rgba(167,139,250,0.12)", "rgba(249,115,22,0.12)"]
            border_colors = ["#38bdf8", "#fbbf24", "#34d399", "#f87171", "#a78bfa", "#f97316"]

            for i, zone in enumerate(zones):
                col_bg  = colors_zone[i % len(colors_zone)]
                col_brd = border_colors[i % len(border_colors)]
                fig_vp.add_hrect(
                    y0=zone["zone_low"], y1=zone["zone_high"],
                    fillcolor=col_bg,
                    line=dict(color=col_brd, width=1, dash="dot"),
                    annotation_text=f"  {zone['price']:.2f}$ ({zone['volume_pct']:.1f}%vol)",
                    annotation_position="right",
                    annotation=dict(font=dict(color=col_brd, size=11)),
                )

            # POC linia
            fig_vp.add_hline(
                y=poc, line_dash="dash", line_color="#fbbf24", line_width=2,
                annotation_text=f"POC {poc:.2f}$",
                annotation_position="left",
                annotation=dict(font=dict(color="#fbbf24", size=11)),
            )

            # Value Area Low linia
            fig_vp.add_hline(
                y=va_low, line_dash="dash", line_color="#34d399", line_width=2,
                annotation_text=f"VA Low {va_low:.2f}$ ← wejście",
                annotation_position="left",
                annotation=dict(font=dict(color="#34d399", size=11)),
            )

            # Value Area High linia
            fig_vp.add_hline(
                y=va_high, line_dash="dash", line_color="#38bdf8", line_width=1.5,
                annotation_text=f"VA High {va_high:.2f}$",
                annotation_position="left",
                annotation=dict(font=dict(color="#38bdf8", size=11)),
            )

            fig_vp.update_layout(
                template="plotly_dark",
                height=580,
                margin=dict(l=0, r=120, t=24, b=0),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                xaxis_rangeslider_visible=False,
                title=dict(text=f"{vp_symbol_name} — cena z Volume Profile i strefami wsparcia", font=dict(size=13, color="#94a3b8")),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_vp, use_container_width=True)

            # ── Volume Profile histogram ───────────────────────
            st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#94a3b8;margin:4px 0 8px 0'>Volume Profile — rozkład wolumenu po cenie</div>", unsafe_allow_html=True)

            fig_hist = go.Figure()
            # Koloruj slupki - jasniejsze = strefa wsparcia
            bar_colors = []
            zone_ranges = [(z["zone_low"], z["zone_high"]) for z in zones]
            for price in vp_data["price_level"]:
                in_zone = any(lo <= price <= hi for lo, hi in zone_ranges)
                if abs(price - poc) < (period_high - period_low) / 300 * 4:
                    bar_colors.append("#fbbf24")
                elif in_zone:
                    bar_colors.append("#38bdf8")
                else:
                    bar_colors.append("#334155")

            fig_hist.add_trace(go.Bar(
                x=vp_data["pct"],
                y=vp_data["price_level"],
                orientation="h",
                name="Volume %",
                marker_color=bar_colors,
                marker_line_width=0,
            ))
            fig_hist.add_hline(y=poc,    line_dash="dash", line_color="#fbbf24", line_width=1.5)
            fig_hist.add_hline(y=va_low, line_dash="dash", line_color="#34d399", line_width=1.5)
            fig_hist.add_hline(y=va_high,line_dash="dash", line_color="#38bdf8", line_width=1)

            fig_hist.update_layout(
                template="plotly_dark",
                height=420,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                xaxis=dict(title="% wolumenu", gridcolor="#1e293b"),
                yaxis=dict(title="Cena ($)", gridcolor="#1e293b"),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # ── Tabela stref ───────────────────────────────────
            if zones:
                st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#94a3b8;margin:16px 0 10px 0'>Wykryte strefy wsparcia — rekomendowane poziomy wejścia</div>", unsafe_allow_html=True)
                zone_rows = ""
                for i, z in enumerate(zones):
                    signal = ""
                    if z["price"] <= va_low:
                        signal = "<span style='color:#34d399;font-weight:700'>✓ Strefa wejścia</span>"
                    elif z["price"] <= poc:
                        signal = "<span style='color:#fbbf24'>~ Poniżej POC</span>"
                    else:
                        signal = "<span style='color:#64748b'>Powyżej POC</span>"
                    zone_rows += f"""
                    <tr>
                        <td style='padding:9px 14px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:700'>${z['price']:,.2f}</td>
                        <td style='padding:9px 14px;border-bottom:1px solid #1e293b;color:#94a3b8'>${z['zone_low']:,.2f} – ${z['zone_high']:,.2f}</td>
                        <td style='padding:9px 14px;border-bottom:1px solid #1e293b;color:#38bdf8;font-weight:600'>{z['volume_pct']:.1f}%</td>
                        <td style='padding:9px 14px;border-bottom:1px solid #1e293b'>{signal}</td>
                    </tr>"""

                st.markdown(f"""
                <table style='width:100%;border-collapse:collapse;font-size:0.85rem'>
                    <thead><tr>
                        <th style='padding:10px 14px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>Cena strefy</th>
                        <th style='padding:10px 14px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>Zakres strefy</th>
                        <th style='padding:10px 14px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>% Wolumenu</th>
                        <th style='padding:10px 14px;background:#0f172a;color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;text-align:left;border-bottom:1px solid #334155'>Sygnał</th>
                    </tr></thead>
                    <tbody>{zone_rows}</tbody>
                </table>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:#0c2541;border:1px solid #0369a1;border-radius:8px;padding:14px 18px;margin-top:20px;font-size:0.85rem;color:#93c5fd;line-height:1.7">
                    <b style="color:#38bdf8">💡 Jak używać tych danych ze strategią Pyramid Long:</b><br>
                    Ustaw <b>Cena wejścia</b> w Strategii 1 na poziomie <b style="color:#34d399">Value Area Low = ${va_low:,.2f}$</b> lub na cenie najbliższej strefy wsparcia.
                    Gdy cena zbliży się do tego poziomu — to sygnał że jesteś w historycznie taniej strefie i czas aktywować strategię pyramidingu.
                </div>
                """, unsafe_allow_html=True)
