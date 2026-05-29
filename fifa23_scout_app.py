"""
FIFA 23 Scout Intelligence — Aplicación de Scouting con Streamlit
Detecta oportunidades de negocio en fichajes usando métricas personalizadas (rpp)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL DE LA APP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA 23 · Scout Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo principal */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .stSidebar { background-color: #161b22; }

    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card h2 { color: #58a6ff; font-size: 2.2rem; margin: 0; }
    .metric-card p  { color: #8b949e; font-size: 0.85rem; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Encabezados de sección */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f0f6fc;
        border-left: 4px solid #58a6ff;
        padding-left: 12px;
        margin: 30px 0 16px 0;
    }

    /* Badges de oportunidad */
    .badge-steal    { background: #0d4429; color: #3fb950; border: 1px solid #3fb950; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }
    .badge-overval  { background: #3d1515; color: #f85149; border: 1px solid #f85149; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }
    .badge-hidden   { background: #1a2c4a; color: #58a6ff; border: 1px solid #58a6ff; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }
    .badge-efficient{ background: #2d2208; color: #d29922; border: 1px solid #d29922; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }

    /* Tabla */
    .dataframe { background-color: #161b22 !important; }

    /* Sidebar labels */
    .stSlider label, .stMultiSelect label, .stSelectbox label { color: #8b949e !important; font-size: 0.82rem !important; }

    /* Separador */
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA Y CACHÉ DEL DATASET
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("fifa23_limpio.csv")

    # Normalización básica de columnas numéricas
    numeric_cols = [
        "overall", "value_eur", "wage_eur", "rpp",
        "pace", "shooting", "passing", "dribbling", "defending", "physic"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Score de oportunidad de fichaje (0–100)
    # Fórmula: combina rpp alto + value bajo + wage bajo
    df["value_pct"]    = df.groupby("player_positions")["value_eur"].rank(pct=True)
    df["wage_pct"]     = df.groupby("player_positions")["wage_eur"].rank(pct=True)
    df["rpp_pct"]      = df.groupby("player_positions")["rpp"].rank(pct=True)
    df["overall_pct"]  = df.groupby("player_positions")["overall"].rank(pct=True)

    df["opportunity_score"] = (
        df["rpp_pct"] * 0.45
        + (1 - df["value_pct"]) * 0.30
        + (1 - df["wage_pct"]) * 0.25
    ) * 100
    df["opportunity_score"] = df["opportunity_score"].round(1)

    # Ratio eficiencia rpp / valor
    df["rpp_value_ratio"] = (df["rpp"] / (df["value_eur"] / 1_000_000 + 0.001)).round(2)

    return df


df_raw = load_data()


# ─────────────────────────────────────────────
# SIDEBAR — FILTROS INTERACTIVOS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ FIFA 23 Scout")
    st.markdown("---")

    # Posición
    positions = sorted(df_raw["player_positions"].dropna().unique())
    sel_positions = st.multiselect(
        "Posición", positions, default=[], placeholder="Todas las posiciones"
    )

    # Nacionalidad
    nations = sorted(df_raw["nationality_name"].dropna().unique())
    sel_nations = st.multiselect(
        "Nacionalidad", nations, default=[], placeholder="Todas las nacionalidades"
    )

    # Rango de valor de mercado
    min_val = int(df_raw["value_eur"].min())
    max_val = int(df_raw["value_eur"].max())
    val_range = st.slider(
        "Valor de mercado (€)", min_val, max_val,
        (min_val, max_val), step=500_000, format="€%d"
    )

    # Rango de rpp
    min_rpp = float(df_raw["rpp"].min())
    max_rpp = float(df_raw["rpp"].max())
    rpp_range = st.slider("Rango de RPP", min_rpp, max_rpp, (min_rpp, max_rpp), step=0.5)

    # Rango de overall
    overall_range = st.slider("Rango de Overall", 40, 99, (40, 99))

    st.markdown("---")
    st.caption("📌 RPP = Rating Por Posición")


# ─────────────────────────────────────────────
# FILTRADO PRINCIPAL
# ─────────────────────────────────────────────
df = df_raw.copy()

if sel_positions:
    df = df[df["player_positions"].isin(sel_positions)]
if sel_nations:
    df = df[df["nationality_name"].isin(sel_nations)]

df = df[
    (df["value_eur"] >= val_range[0]) & (df["value_eur"] <= val_range[1]) &
    (df["rpp"]       >= rpp_range[0]) & (df["rpp"]       <= rpp_range[1]) &
    (df["overall"]   >= overall_range[0]) & (df["overall"] <= overall_range[1])
]


# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#58a6ff; margin-bottom:4px;'>⚽ FIFA 23 · Scout Intelligence</h1>"
    "<p style='color:#8b949e; margin-top:0;'>Herramienta de análisis de fichajes basada en RPP — Rating Por Posición</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ─────────────────────────────────────────────
# SECCIÓN 1: DASHBOARD GENERAL (KPIs)
# ─────────────────────────────────────────────
st.markdown("<div class='section-title'>📊 Dashboard General</div>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

kpis = [
    (col1, len(df), "Jugadores"),
    (col2, round(df["overall"].mean(), 1) if not df.empty else 0, "Overall Promedio"),
    (col3, round(df["rpp"].mean(), 1) if not df.empty else 0, "RPP Promedio"),
    (col4, f"€{df['value_eur'].median()/1e6:.1f}M" if not df.empty else "€0", "Valor Mediano"),
    (col5, f"€{df['wage_eur'].median()/1e3:.0f}K" if not df.empty else "€0", "Salario Mediano / semana"),
]

for col, val, label in kpis:
    with col:
        st.markdown(
            f"<div class='metric-card'><h2>{val}</h2><p>{label}</p></div>",
            unsafe_allow_html=True
        )

# Distribución por posición
if not df.empty:
    pos_dist = df["player_positions"].value_counts().reset_index()
    pos_dist.columns = ["Posición", "Cantidad"]
    fig_pos = px.bar(
        pos_dist, x="Posición", y="Cantidad",
        color="Cantidad", color_continuous_scale="Blues",
        template="plotly_dark", title="Distribución por Posición"
    )
    fig_pos.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        coloraxis_showscale=False, height=350
    )
    st.plotly_chart(fig_pos, use_container_width=True)


# ─────────────────────────────────────────────
# SECCIÓN 2: ANÁLISIS DE OPORTUNIDADES
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>🔥 Análisis de Oportunidades de Fichaje</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💎 Infravalorados",
    "📉 Sobrevalorados",
    "🌟 Talento Oculto",
    "💰 Eficiencia Salarial",
    "🏆 Top por Posición"
])


# ── TAB 1: JUGADORES INFRAVALORADOS ──────────────────────────────────────────
with tab1:
    st.markdown("**Criterio:** RPP > percentil 75 · Valor de mercado < percentil 50")

    if not df.empty:
        p75_rpp  = df["rpp"].quantile(0.75)
        p50_val  = df["value_eur"].quantile(0.50)

        underval = df[(df["rpp"] >= p75_rpp) & (df["value_eur"] <= p50_val)].copy()
        underval = underval.sort_values("opportunity_score", ascending=False)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_uv = px.scatter(
                underval, x="value_eur", y="rpp",
                hover_name="long_name",
                hover_data={"overall": True, "player_positions": True, "opportunity_score": True},
                color="opportunity_score",
                color_continuous_scale="Greens",
                size="overall", size_max=18,
                template="plotly_dark",
                labels={"value_eur": "Valor (€)", "rpp": "RPP"},
                title=f"💎 {len(underval)} Jugadores Infravalorados"
            )
            fig_uv.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
            st.plotly_chart(fig_uv, use_container_width=True)

        with col_b:
            st.markdown(f"**{len(underval)} jugadores encontrados**")
            cols_show = ["long_name", "player_positions", "overall", "rpp", "value_eur", "opportunity_score"]
            cols_show = [c for c in cols_show if c in underval.columns]
            st.dataframe(
                underval[cols_show].head(20).rename(columns={
                    "long_name": "Jugador", "player_positions": "Pos",
                    "overall": "OVR", "value_eur": "Valor €",
                    "opportunity_score": "Score"
                }),
                hide_index=True, use_container_width=True
            )


# ── TAB 2: JUGADORES SOBREVALORADOS ──────────────────────────────────────────
with tab2:
    st.markdown("**Criterio:** Overall > percentil 75 · RPP < percentil 40")

    if not df.empty:
        p75_ovr = df["overall"].quantile(0.75)
        p40_rpp = df["rpp"].quantile(0.40)

        overval = df[(df["overall"] >= p75_ovr) & (df["rpp"] <= p40_rpp)].copy()
        overval = overval.sort_values("value_eur", ascending=False)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_ov = px.scatter(
                overval, x="overall", y="rpp",
                hover_name="long_name",
                hover_data={"value_eur": True, "player_positions": True},
                color="value_eur",
                color_continuous_scale="Reds",
                size="value_eur", size_max=18,
                template="plotly_dark",
                labels={"overall": "Overall", "rpp": "RPP"},
                title=f"📉 {len(overval)} Jugadores Sobrevalorados"
            )
            fig_ov.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
            st.plotly_chart(fig_ov, use_container_width=True)

        with col_b:
            st.markdown(f"**{len(overval)} jugadores encontrados**")
            cols_show = ["long_name", "player_positions", "overall", "rpp", "value_eur"]
            cols_show = [c for c in cols_show if c in overval.columns]
            st.dataframe(
                overval[cols_show].head(20).rename(columns={
                    "long_name": "Jugador", "player_positions": "Pos",
                    "overall": "OVR", "value_eur": "Valor €"
                }),
                hide_index=True, use_container_width=True
            )


# ── TAB 3: TALENTO OCULTO ─────────────────────────────────────────────────────
with tab3:
    st.markdown("**Criterio:** RPP > percentil 70 en su posición · Overall < promedio de su posición")

    if not df.empty:
        pos_avg_ovr = df.groupby("player_positions")["overall"].transform("mean")
        pos_p70_rpp = df.groupby("player_positions")["rpp"].transform(lambda x: x.quantile(0.70))

        hidden = df[(df["rpp"] >= pos_p70_rpp) & (df["overall"] < pos_avg_ovr)].copy()
        hidden = hidden.sort_values("rpp", ascending=False)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_hid = px.scatter(
                hidden, x="overall", y="rpp",
                hover_name="long_name",
                hover_data={"value_eur": True, "player_positions": True},
                color="player_positions",
                size="rpp", size_max=16,
                template="plotly_dark",
                labels={"overall": "Overall", "rpp": "RPP"},
                title=f"🌟 {len(hidden)} Talentos Ocultos"
            )
            fig_hid.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
            st.plotly_chart(fig_hid, use_container_width=True)

        with col_b:
            st.markdown(f"**{len(hidden)} jugadores encontrados**")
            cols_show = ["long_name", "player_positions", "overall", "rpp", "value_eur"]
            cols_show = [c for c in cols_show if c in hidden.columns]
            st.dataframe(
                hidden[cols_show].head(20).rename(columns={
                    "long_name": "Jugador", "player_positions": "Pos",
                    "overall": "OVR", "value_eur": "Valor €"
                }),
                hide_index=True, use_container_width=True
            )


# ── TAB 4: EFICIENCIA SALARIAL ────────────────────────────────────────────────
with tab4:
    st.markdown("**Criterio:** RPP > percentil 70 · Salario < percentil 40")

    if not df.empty:
        p70_rpp  = df["rpp"].quantile(0.70)
        p40_wage = df["wage_eur"].quantile(0.40)

        efficient = df[(df["rpp"] >= p70_rpp) & (df["wage_eur"] <= p40_wage)].copy()
        efficient["rpp_wage_ratio"] = (efficient["rpp"] / (efficient["wage_eur"] / 1000 + 0.001)).round(2)
        efficient = efficient.sort_values("rpp_wage_ratio", ascending=False)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_eff = px.scatter(
                efficient, x="wage_eur", y="rpp",
                hover_name="long_name",
                hover_data={"overall": True, "player_positions": True, "rpp_wage_ratio": True},
                color="rpp_wage_ratio",
                color_continuous_scale="YlOrGn",
                size="rpp", size_max=16,
                template="plotly_dark",
                labels={"wage_eur": "Salario (€/sem)", "rpp": "RPP"},
                title=f"💰 {len(efficient)} Jugadores Eficientes en Salario"
            )
            fig_eff.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
            st.plotly_chart(fig_eff, use_container_width=True)

        with col_b:
            st.markdown(f"**{len(efficient)} jugadores encontrados**")
            cols_show = ["long_name", "player_positions", "overall", "rpp", "wage_eur", "rpp_wage_ratio"]
            cols_show = [c for c in cols_show if c in efficient.columns]
            st.dataframe(
                efficient[cols_show].head(20).rename(columns={
                    "long_name": "Jugador", "player_positions": "Pos",
                    "overall": "OVR", "wage_eur": "Salario €",
                    "rpp_wage_ratio": "RPP/Sal."
                }),
                hide_index=True, use_container_width=True
            )


# ── TAB 5: TOP POR POSICIÓN ───────────────────────────────────────────────────
with tab5:
    st.markdown("**Top 10 jugadores por RPP en cada posición**")

    if not df.empty:
        all_positions = sorted(df["player_positions"].dropna().unique())
        sel_pos_top = st.selectbox("Selecciona una posición", all_positions, key="top_pos")

        top10 = (
            df[df["player_positions"] == sel_pos_top]
            .sort_values("rpp", ascending=False)
            .head(10)
        )

        if not top10.empty:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                fig_top = px.bar(
                    top10.sort_values("rpp"), x="rpp", y="long_name",
                    orientation="h",
                    color="overall",
                    color_continuous_scale="Blues",
                    template="plotly_dark",
                    labels={"rpp": "RPP", "long_name": "Jugador"},
                    title=f"🏆 Top 10 · {sel_pos_top}"
                )
                fig_top.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
                st.plotly_chart(fig_top, use_container_width=True)

            with col_b:
                cols_show = ["long_name", "overall", "rpp", "value_eur", "wage_eur", "nationality_name"]
                cols_show = [c for c in cols_show if c in top10.columns]
                st.dataframe(
                    top10[cols_show].rename(columns={
                        "long_name": "Jugador", "overall": "OVR",
                        "value_eur": "Valor €", "wage_eur": "Salario €",
                        "nationality_name": "País"
                    }),
                    hide_index=True, use_container_width=True
                )


# ─────────────────────────────────────────────
# SECCIÓN 3: VISUALIZACIONES GENERALES
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>📈 Visualizaciones</div>", unsafe_allow_html=True)

if not df.empty:
    col_v1, col_v2 = st.columns(2)

    # Histograma de RPP
    with col_v1:
        fig_hist = px.histogram(
            df, x="rpp", nbins=40,
            color_discrete_sequence=["#58a6ff"],
            template="plotly_dark",
            title="Distribución de RPP"
        )
        fig_hist.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Scatter: value_eur vs rpp
    with col_v2:
        fig_sc1 = px.scatter(
            df.sample(min(2000, len(df))),
            x="value_eur", y="rpp",
            hover_name="long_name",
            color="player_positions",
            opacity=0.7,
            template="plotly_dark",
            labels={"value_eur": "Valor (€)", "rpp": "RPP"},
            title="Valor de Mercado vs RPP"
        )
        fig_sc1.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_sc1, use_container_width=True)

    col_v3, col_v4 = st.columns(2)

    # Scatter: overall vs rpp
    with col_v3:
        fig_sc2 = px.scatter(
            df.sample(min(2000, len(df))),
            x="overall", y="rpp",
            hover_name="long_name",
            color="player_positions",
            opacity=0.7,
            template="plotly_dark",
            labels={"overall": "Overall", "rpp": "RPP"},
            title="Overall vs RPP"
        )
        fig_sc2.add_shape(
            type="line", x0=df["overall"].min(), x1=df["overall"].max(),
            y0=df["overall"].min(), y1=df["overall"].max(),
            line=dict(color="white", dash="dot", width=1)
        )
        fig_sc2.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_sc2, use_container_width=True)

    # Boxplot de RPP por posición
    with col_v4:
        top_pos_for_box = (
            df["player_positions"].value_counts().head(12).index.tolist()
        )
        df_box = df[df["player_positions"].isin(top_pos_for_box)]
        fig_box = px.box(
            df_box, x="player_positions", y="rpp",
            color="player_positions",
            template="plotly_dark",
            title="Distribución RPP por Posición",
            labels={"player_positions": "Posición", "rpp": "RPP"}
        )
        fig_box.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            showlegend=False, height=350
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ─────────────────────────────────────────────
# SECCIÓN 4: TABLA INTERACTIVA
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>📋 Tabla de Jugadores Filtrados</div>", unsafe_allow_html=True)

if not df.empty:
    sort_col = st.selectbox(
        "Ordenar por",
        ["rpp", "overall", "value_eur", "wage_eur", "opportunity_score"],
        index=0
    )
    sort_asc = st.checkbox("Orden ascendente", value=False)

    display_cols = [
        "long_name", "player_positions", "nationality_name",
        "overall", "rpp", "value_eur", "wage_eur",
        "opportunity_score", "rpp_value_ratio",
        "pace", "shooting", "passing", "dribbling", "defending", "physic"
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    df_display = (
        df[display_cols]
        .sort_values(sort_col, ascending=sort_asc)
        .reset_index(drop=True)
    )

    st.dataframe(
        df_display.rename(columns={
            "long_name": "Jugador", "player_positions": "Posición",
            "nationality_name": "País", "overall": "OVR",
            "value_eur": "Valor €", "wage_eur": "Salario €/sem",
            "opportunity_score": "Oport. Score", "rpp_value_ratio": "RPP/Valor"
        }),
        use_container_width=True,
        height=420
    )

    st.caption(f"Mostrando {len(df_display):,} jugadores")


# ─────────────────────────────────────────────
# SECCIÓN 5: SCORE DE OPORTUNIDAD — RANKING GENERAL
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>🏅 Ranking de Oportunidades — Top 20 Gangas</div>", unsafe_allow_html=True)

if not df.empty:
    steal_threshold = df["opportunity_score"].quantile(0.90)
    steals = df[df["opportunity_score"] >= steal_threshold].sort_values("opportunity_score", ascending=False).head(20)

    fig_rank = px.bar(
        steals.sort_values("opportunity_score"),
        x="opportunity_score", y="long_name",
        orientation="h",
        color="opportunity_score",
        color_continuous_scale="Viridis",
        hover_data={"player_positions": True, "overall": True, "rpp": True, "value_eur": True},
        template="plotly_dark",
        labels={"opportunity_score": "Score de Oportunidad", "long_name": "Jugador"},
        title="💎 Top 20 Jugadores con Mayor Score de Oportunidad"
    )
    fig_rank.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        coloraxis_showscale=False, height=550
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # Tarjetas de las top 3 gangas
    st.markdown("#### 🔝 Las 3 Mejores Gangas")
    top3 = steals.head(3)
    card_cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with card_cols[i]:
            val_m = row["value_eur"] / 1_000_000 if pd.notna(row["value_eur"]) else 0
            st.markdown(
                f"""
                <div class='metric-card' style='text-align:left;'>
                    <span class='badge-steal'>⭐ GANGA</span><br><br>
                    <b style='font-size:1.1rem;color:#f0f6fc;'>{row['long_name']}</b><br>
                    <span style='color:#8b949e;'>Posición: <b style='color:#58a6ff;'>{row['player_positions']}</b></span><br><br>
                    <span style='color:#8b949e;'>Overall: </span><b>{int(row['overall'])}</b>&nbsp;&nbsp;
                    <span style='color:#8b949e;'>RPP: </span><b>{row['rpp']:.1f}</b><br>
                    <span style='color:#8b949e;'>Valor: </span><b style='color:#3fb950;'>€{val_m:.1f}M</b><br>
                    <span style='color:#8b949e;'>Score: </span><b style='color:#d29922;'>{row['opportunity_score']}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")
st.caption("FIFA 23 Scout Intelligence · Desarrollado con Streamlit + Plotly · Datos: FIFA 23")
