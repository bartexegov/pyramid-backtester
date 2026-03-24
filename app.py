"""
app.py
Pyramid Long Backtester — interfejs Streamlit.

Uruchomienie:
    cd PyramidBacktester
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta

from backtester import (
    fetch_data,
    run_backtest,
    trades_to_dataframe,
    COMMODITY_SYMBOLS,
)

# ─────────────────────────────────────────────────────────────
# Helper: renderuj DataFrame jako HTML bez pyarrow
# ─────────────────────────────────────────────────────────────

def show_table(df: pd.DataFrame, max_rows: int = 500):
    """Wyswietla DataFrame jako HTML - nie wymaga pyarrow."""
    if df.empty:
        st.info("Brak danych do wyswietlenia.")
        return
    display_df = df.head(max_rows).fillna("—")
    html = display_df.to_html(index=False, border=0, classes="custom-table")
    st.markdown(f"""
    <style>
    .custom-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        font-family: monospace;
    }}
    .custom-table th {{
        background: #1e2130;
        color: #fff;
        padding: 6px 10px;
        text-align: left;
        border-bottom: 2px solid #444;
    }}
    .custom-table td {{
        padding: 5px 10px;
        border-bottom: 1px solid #2a2a2a;
        color: #ddd;
    }}
    .custom-table tr:nth-child(even) td {{ background: #0e1117; }}
    .custom-table tr:hover td {{ background: #1a1f2e; }}
    </style>
    {html}
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Konfiguracja strony
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pyramid Long Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-box {
        background: #1e2130;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stMetric label { font-size: 0.8rem; color: #aaa; }
    div[data-testid="stSidebarContent"] { background: #0e1117; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PANEL LEWY — parametry strategii (Okno 1)
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Parametry strategii")
    st.markdown("---")

    # Wybor instrumentu
    st.subheader("📈 Instrument")
    commodity_name = st.selectbox(
        "Wybierz surowiec",
        options=list(COMMODITY_SYMBOLS.keys()),
        index=0,
    )
    symbol = COMMODITY_SYMBOLS[commodity_name]
    st.caption(f"Symbol Yahoo Finance: `{symbol}`")

    st.markdown("---")

    # Daty backtestu
    st.subheader("📅 Okres backtestów")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Od",
            value=date.today() - timedelta(days=365 * 2),
            min_value=date(2000, 1, 1),
            max_value=date.today(),
        )
    with col2:
        end_date = st.date_input(
            "Do",
            value=date.today(),
            min_value=date(2000, 1, 2),
            max_value=date.today(),
        )

    st.markdown("---")

    # Parametry strategii
    st.subheader("🎯 Strategia")

    entry_threshold = st.number_input(
        "Cena wejścia ($) — kupuj gdy Low < tej ceny",
        min_value=0.01,
        max_value=100000.0,
        value=470.0,
        step=1.0,
        format="%.2f",
        help="Pierwsze kupno następuje gdy Low baru spadnie poniżej tej ceny.",
    )

    pyramid_step = st.number_input(
        "Krok dokupowania ($) — co ile $ w dół kupujesz kolejny kontrakt",
        min_value=0.01,
        max_value=10000.0,
        value=5.0,
        step=0.25,
        format="%.2f",
        help="Każde kolejne wejście następuje gdy cena spadnie o tę wartość od poprzedniego kupna.",
    )

    take_profit = st.number_input(
        "Take Profit ($) — ile $ powyżej ceny kupna",
        min_value=0.01,
        max_value=10000.0,
        value=5.0,
        step=0.25,
        format="%.2f",
        help="Każdy kontrakt zamykany gdy High baru osiągnie cenę_kupna + TP.",
    )

    st.markdown("---")

    # Parametry kapitalowe
    st.subheader("💰 Kapitał")

    margin_per_contract = st.number_input(
        "Margin na 1 kontrakt ($) — depozyt u brokera",
        min_value=100.0,
        max_value=1000000.0,
        value=1500.0,
        step=100.0,
        format="%.0f",
        help="Wymagany depozyt zabezpieczający na 1 kontrakt. Użyty do obliczenia max wymaganego kapitału.",
    )

    st.markdown("---")

    # Optymalizacja — zakres kroków do porównania
    st.subheader("🔬 Optymalizacja kroków")
    st.caption("Przetestuj automatycznie wszystkie kombinacje krok × TP i znajdź najlepszą.")

    optimize_enabled = st.checkbox("Włącz optymalizację", value=False)

    opt_step_min = 2.0
    opt_step_max = 10.0
    opt_step_inc = 1.0
    opt_tp_min   = 2.0
    opt_tp_max   = 10.0
    opt_tp_inc   = 1.0

    if optimize_enabled:
        st.markdown("**Krok dokupowania** — co ile $ w dół dokupujesz kontrakt")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            opt_step_min = st.number_input("Zacznij od ($)", value=2.0, step=0.25, format="%.2f", key="smin", help="Najmniejszy krok do przetestowania. Np. 2 = dokupuj co 2$.")
        with col_s2:
            opt_step_max = st.number_input("Skończ na ($)", value=10.0, step=0.25, format="%.2f", key="smax", help="Największy krok do przetestowania. Np. 10 = dokupuj co 10$.")
        with col_s3:
            opt_step_inc = st.number_input("Co ($)", value=1.0, step=0.25, format="%.2f", key="sinc", help="Odstęp między testowanymi krokami. Np. 1 = testuj 2, 3, 4, ... 10.")

        st.markdown("**Take Profit** — ile $ powyżej ceny kupna sprzedajesz kontrakt")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            opt_tp_min = st.number_input("Zacznij od ($)", value=2.0, step=0.25, format="%.2f", key="tmin", help="Najmniejszy TP do przetestowania. Np. 2 = sprzedaj +2$ od kupna.")
        with col_t2:
            opt_tp_max = st.number_input("Skończ na ($)", value=10.0, step=0.25, format="%.2f", key="tmax", help="Największy TP do przetestowania. Np. 10 = sprzedaj +10$ od kupna.")
        with col_t3:
            opt_tp_inc = st.number_input("Co ($)", value=1.0, step=0.25, format="%.2f", key="tinc", help="Odstęp między testowanymi TP. Np. 1 = testuj 2, 3, 4, ... 10.")

        n_steps = max(1, round((opt_step_max - opt_step_min) / opt_step_inc) + 1)
        n_tps   = max(1, round((opt_tp_max   - opt_tp_min)   / opt_tp_inc)   + 1)
        st.caption(f"Liczba kombinacji: {n_steps} kroków × {n_tps} TP = **{n_steps * n_tps} testów**")

    st.markdown("---")
    run_button = st.button("▶️ Uruchom backtest", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# GLOWNY PANEL — wyniki (Okno 2)
# ─────────────────────────────────────────────────────────────

st.title("📊 Pyramid Long Backtester")
st.caption("Testuj strategie pyramidingu na surowcach. Dane: Yahoo Finance (dzienne).")

if not run_button:
    st.info("👈 Ustaw parametry w panelu po lewej i kliknij **Uruchom backtest**.")
    st.stop()

# ── Pobieranie danych ─────────────────────────────────────────
with st.spinner(f"Pobieranie danych {commodity_name} ({symbol})..."):
    try:
        df = fetch_data(symbol, start=start_date, end=end_date)
    except Exception as e:
        st.error(f"Błąd pobierania danych: {e}")
        st.stop()

if df.empty:
    st.error("Brak danych dla podanego symbolu i zakresu dat. Sprawdź symbol lub daty.")
    st.stop()

st.success(f"Pobrano {len(df)} sesji dla {commodity_name} ({start_date} → {end_date})")

# ── Backtest główny ───────────────────────────────────────────
with st.spinner("Obliczanie backtestu..."):
    result = run_backtest(
        df               = df,
        entry_threshold  = entry_threshold,
        pyramid_step     = pyramid_step,
        take_profit      = take_profit,
        margin_per_contract = margin_per_contract,
        qty_per_entry    = 1,
    )

# ─────────────────────────────────────────────────────────────
# SEKCJA 1: METRYKI GLOWNE
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📋 Wyniki backtestu")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    pnl_color = "normal" if result.total_pnl >= 0 else "inverse"
    st.metric(
        label="Zysk / Strata całkowita",
        value=f"${result.total_pnl:,.2f}",
        delta=f"{result.total_pnl:+,.2f} $",
    )

with col2:
    st.metric(
        label="Transakcji zamkniętych",
        value=str(result.total_trades),
        help="Liczba kontraktów które osiągnęły TP",
    )

with col3:
    st.metric(
        label="Win Rate",
        value=f"{result.win_rate:.1f}%",
        help="% transakcji zakończonych zyskiem",
    )

with col4:
    st.metric(
        label="Max kontraktów jednocześnie",
        value=str(result.max_concurrent),
        help="Największa liczba jednocześnie otwartych kontraktów w historii backtestu",
    )

with col5:
    st.metric(
        label="Max wymagany kapitał",
        value=f"${result.max_capital_needed:,.0f}",
        help=f"Max kontrakty ({result.max_concurrent}) × margin (${margin_per_contract:,.0f})",
    )

with col6:
    st.metric(
        label="Transakcji otwartych (niezamk.)",
        value=str(result.open_trades),
        help="Kontrakty które nie osiągnęły TP do końca okresu backtestu",
    )

# Dodatkowy rzad metryk
col7, col8, col9, col10 = st.columns(4)

with col7:
    st.metric("Wygrane transakcje", str(result.winning_trades))

with col8:
    st.metric("Przegrane transakcje", str(result.losing_trades))

with col9:
    avg_pnl = result.total_pnl / result.total_trades if result.total_trades > 0 else 0
    st.metric("Śr. PnL na transakcję", f"${avg_pnl:.2f}")

with col10:
    st.metric("Śr. dni do zamknięcia", f"{result.avg_days_open:.0f} dni")

# ─────────────────────────────────────────────────────────────
# SEKCJA 2: WYKRESY
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📈 Wykresy")

tab1, tab2, tab3 = st.tabs(["Equity Curve", "Otwarte kontrakty w czasie", "Cena + sygnały"])

with tab1:
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve.values,
        mode="lines",
        name="Equity",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
    ))
    fig_equity.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_equity.update_layout(
        title="Equity Curve — skumulowany PnL ($)",
        xaxis_title="Data",
        yaxis_title="PnL ($)",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_equity, use_container_width=True)

with tab2:
    fig_open = go.Figure()
    fig_open.add_trace(go.Scatter(
        x=result.daily_open_contracts.index,
        y=result.daily_open_contracts.values,
        mode="lines",
        name="Otwarte kontrakty",
        line=dict(color="#ff6b6b", width=1),
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.2)",
    ))
    fig_open.add_hline(
        y=result.max_concurrent,
        line_dash="dash",
        line_color="yellow",
        annotation_text=f"Max: {result.max_concurrent}",
        annotation_position="right",
    )
    fig_open.update_layout(
        title="Liczba otwartych kontraktów w czasie",
        xaxis_title="Data",
        yaxis_title="Liczba kontraktów",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_open, use_container_width=True)

with tab3:
    fig_price = go.Figure()

    # Wykres ceny (candlestick)
    fig_price.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Cena",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    # Linia progu wejscia
    fig_price.add_hline(
        y=entry_threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Próg wejścia: {entry_threshold}",
        annotation_position="right",
    )

    # Punkty wejsc (zielone strzalki)
    entry_dates  = [t.entry_date for t in result.trades]
    entry_prices = [t.entry_price for t in result.trades]
    fig_price.add_trace(go.Scatter(
        x=entry_dates,
        y=entry_prices,
        mode="markers",
        name="Wejście (kupno)",
        marker=dict(symbol="triangle-up", size=10, color="lime"),
    ))

    # Punkty wyjsc (TP trafione)
    exit_dates  = [t.exit_date  for t in result.trades if t.closed]
    exit_prices = [t.exit_price for t in result.trades if t.closed]
    fig_price.add_trace(go.Scatter(
        x=exit_dates,
        y=exit_prices,
        mode="markers",
        name="TP trafiony",
        marker=dict(symbol="triangle-down", size=10, color="orange"),
    ))

    fig_price.update_layout(
        title=f"{commodity_name} — cena z sygnałami wejść/wyjść",
        xaxis_title="Data",
        yaxis_title="Cena",
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_price, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SEKCJA 3: OPTYMALIZACJA
# ─────────────────────────────────────────────────────────────

if optimize_enabled:
    st.markdown("---")
    st.subheader("🔬 Optymalizacja — wyniki")
    st.caption(f"Wyniki dla: **{commodity_name}** | próg wejścia: **{entry_threshold}$** | margin: **{margin_per_contract:,.0f}$**")
    st.info("Optymalizacja jest robiona dla konkretnego surowca i okresu. Wyniki dla kukurydzy beda inne niz dla ropy czy zlota — kazdy instrument ma inna zmiennosc i zakres cenowyn.")

    steps = np.arange(float(opt_step_min), float(opt_step_max) + float(opt_step_inc) * 0.5, float(opt_step_inc))
    tps   = np.arange(float(opt_tp_min),   float(opt_tp_max)   + float(opt_tp_inc)   * 0.5, float(opt_tp_inc))

    total_runs = len(steps) * len(tps)

    if total_runs > 300:
        st.warning(f"Za duzo kombinacji ({total_runs}). Zawez zakres lub zwieksz krok 'Co ($)'.")
    else:
        opt_results = []
        progress = st.progress(0, text="Obliczanie...")
        run_count = 0

        for s in steps:
            for tp in tps:
                r = run_backtest(
                    df=df,
                    entry_threshold=entry_threshold,
                    pyramid_step=round(float(s), 4),
                    take_profit=round(float(tp), 4),
                    margin_per_contract=margin_per_contract,
                )
                opt_results.append({
                    "Krok ($)":        round(float(s), 2),
                    "TP ($)":          round(float(tp), 2),
                    "PnL ($)":         round(r.total_pnl, 2),
                    "Transakcji":      r.total_trades,
                    "Win %":           round(r.win_rate, 1),
                    "Max kontr.":      r.max_concurrent,
                    "Max kapital ($)": int(r.max_capital_needed),
                    "Sr. dni":         int(r.avg_days_open),
                })
                run_count += 1
                progress.progress(run_count / total_runs, text=f"Testuje {run_count}/{total_runs}...")

        progress.empty()

        opt_df = pd.DataFrame(opt_results)
        opt_df_sorted = opt_df.sort_values("PnL ($)", ascending=False).reset_index(drop=True)

        # ── Banery top 3 ─────────────────────────────────────────────────────
        st.markdown("### Najlepsze kombinacje")
        medals = ["🥇", "🥈", "🥉"]
        top_cols = st.columns(3)
        for idx, col in enumerate(top_cols):
            if idx < len(opt_df_sorted):
                row = opt_df_sorted.iloc[idx]
                with col:
                    st.markdown(f"""
                    <div style="background:#1e2130;border-radius:10px;padding:16px;text-align:center">
                    <div style="font-size:2rem">{medals[idx]}</div>
                    <div style="font-size:1.4rem;font-weight:bold;color:#00d4aa">
                        Krok {row['Krok ($)']}$ / TP {row['TP ($)']}$
                    </div>
                    <div style="font-size:1.1rem;color:#fff;margin-top:6px">
                        PnL: <b style="color:{'#00d4aa' if row['PnL ($)']>=0 else '#ff6b6b'}">${row['PnL ($)']:,.2f}</b>
                    </div>
                    <div style="color:#aaa;font-size:0.85rem;margin-top:4px">
                        {row['Transakcji']} transakcji &nbsp;|&nbsp; Win: {row['Win %']}%<br>
                        Max {row['Max kontr.']} kontr. &nbsp;|&nbsp; Kapital: ${row['Max kapital ($)']:,}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Heatmapa ─────────────────────────────────────────────────────────
        st.markdown("### Heatmapa PnL — Krok vs TP")
        st.caption("Zielony = wiekszy zysk. Czerwony = strata lub mniejszy zysk. Kliknij na komorke aby zobaczyc dokladna wartosc.")
        pivot = opt_df.pivot(index="Krok ($)", columns="TP ($)", values="PnL ($)")
        fig_heat = px.imshow(
            pivot,
            labels=dict(x="Take Profit ($)", y="Krok dokupowania ($)", color="PnL ($)"),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            template="plotly_dark",
            text_auto=".0f",
        )
        fig_heat.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_colorbar=dict(title="PnL ($)"),
            xaxis_title="Take Profit ($) — ile $ powyzej kupna sprzedajesz",
            yaxis_title="Krok ($) — co ile $ w dol dokupujesz",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Pelna tabela top 20 ───────────────────────────────────────────────
        st.markdown("### Tabela wynikow — Top 20 kombinacji (sortowane po PnL)")
        show_table(opt_df_sorted.head(20))

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("Pyramid Long Backtester | Dane: Yahoo Finance | Tylko do celów edukacyjnych — nie jest to porada inwestycyjna.")
