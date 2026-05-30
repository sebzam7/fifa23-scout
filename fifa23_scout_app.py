"""
FIFA 23 Scout Intelligence — Aplicación de Scouting con Streamlit
Detecta oportunidades de negocio en fichajes usando métricas personalizadas (rpp)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
# ESTILOS CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .stSidebar { background-color: #161b22; }
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
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f0f6fc;
        border-left: 4px solid #58a6ff;
        padding-left: 12px;
        margin: 30px 0 16px 0;
    }
    .badge-steal    { background: #0d4429; color: #3fb950; border: 1px solid #3fb950; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }
    .badge-overval  { background: #3d1515; color: #f85149; border: 1px solid #f85149; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }
    .badge-hidden   { background: #1a2c4a; color: #58a6ff; border: 1px solid #58a6ff; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }
    .badge-efficient{ background: #2d2208; color: #d29922; border: 1px solid #d29922; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }
    .badge-young    { background: #1a1a4a; color: #a371f7; border: 1px solid #a371f7; border-radius: 6px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }
    .dataframe { background-color: #161b22 !important; }
    /* Color de sliders: azul dashboard en lugar de rojo */
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] { color: #58a6ff !important; }
    .stSlider [data-baseweb="slider"] > div > div > div { background: #58a6ff !important; }
    .stSlider label, .stMultiSelect label, .stSelectbox label { color: #8b949e !important; font-size: 0.82rem !important; }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA Y CACHÉ DEL DATASET
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("fifa23_limpio.csv")

    numeric_cols = [
        "overall", "value_eur", "wage_eur", "rpp", "age",
        "pace", "shooting", "passing", "dribbling", "defending", "physic"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["value_pct"]   = df.groupby("player_positions")["value_eur"].rank(pct=True)
    df["wage_pct"]    = df.groupby("player_positions")["wage_eur"].rank(pct=True)
    df["rpp_pct"]     = df.groupby("player_positions")["rpp"].rank(pct=True)
    df["overall_pct"] = df.groupby("player_positions")["overall"].rank(pct=True)

    df["opportunity_score"] = (
        df["rpp_pct"] * 0.45
        + (1 - df["value_pct"]) * 0.30
        + (1 - df["wage_pct"]) * 0.25
    ) * 100
    df["opportunity_score"] = df["opportunity_score"].round(1)

    df["rpp_value_ratio"] = (df["rpp"] / (df["value_eur"] / 1_000_000 + 0.001)).round(2)

    def categorize_age(age):
        if pd.isna(age):
            return "Desconocido"
        elif age <= 21:
            return "Joven (≤21)"
        elif age <= 25:
            return "Promesa (22-25)"
        elif age <= 29:
            return "Prime (26-29)"
        elif age <= 32:
            return "Experimentado (30-32)"
        else:
            return "Veterano (33+)"

    df["age_category"] = df["age"].apply(categorize_age)

    return df


df_raw = load_data()

# Helper para formatear valores con separador de miles/millones
def fmt_eur(val, decimals=1):
    if pd.isna(val):
        return "€0"
    if val >= 1_000_000:
        return f"€{val/1_000_000:,.{decimals}f}M".replace(",", "X").replace(".", ",").replace("X", ".")
    elif val >= 1_000:
        return f"€{val/1_000:,.{decimals}f}K".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€{val:,.0f}"


# ─────────────────────────────────────────────
# SIDEBAR — FILTROS INTERACTIVOS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ FIFA 23 Scout")
    st.markdown("---")
    # ENCABEZADO DE FILTROS
    st.markdown("### Filtros globales")

    positions = sorted(df_raw["player_positions"].dropna().unique())
    sel_positions = st.multiselect("Posición", positions, default=[], placeholder="Todas las posiciones")

    nations = sorted(df_raw["nationality_name"].dropna().unique())
    sel_nations = st.multiselect("Nacionalidad", nations, default=[], placeholder="Todas las nacionalidades")

    min_age = int(df_raw["age"].min())
    max_age = int(df_raw["age"].max())
    age_range = st.slider("Rango de edad", min_age, max_age, (min_age, max_age))

    min_val = int(df_raw["value_eur"].min())
    max_val = int(df_raw["value_eur"].max())
    val_range = st.slider("Valor de mercado (€)", min_val, max_val, (min_val, max_val), step=500_000, format="€%d")

    min_rpp = float(df_raw["rpp"].min())
    max_rpp = float(df_raw["rpp"].max())
    rpp_range = st.slider("Rango de RPP", min_rpp, max_rpp, (min_rpp, max_rpp), step=0.5)

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
    (df["age"]       >= age_range[0])    & (df["age"]       <= age_range[1]) &
    (df["value_eur"] >= val_range[0])    & (df["value_eur"] <= val_range[1]) &
    (df["rpp"]       >= rpp_range[0])    & (df["rpp"]       <= rpp_range[1]) &
    (df["overall"]   >= overall_range[0]) & (df["overall"]  <= overall_range[1])
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

col1, col2, col3, col4, col5, col6 = st.columns(6)

valor_mediano_fmt = fmt_eur(df["value_eur"].median()) if not df.empty else "€0"
salario_mediano_fmt = fmt_eur(df["wage_eur"].median()) if not df.empty else "€0"

kpis = [
    (col1, f"{len(df):,}".replace(",", "."), "Jugadores"),
    (col2, round(df["overall"].mean(), 1) if not df.empty else 0, "Overall Promedio"),
    (col3, round(df["rpp"].mean(), 1) if not df.empty else 0, "RPP Promedio"),
    (col4, round(df["age"].mean(), 1) if not df.empty else 0, "Edad Promedio"),
    (col5, valor_mediano_fmt, "Valor Mediano"),
    (col6, salario_mediano_fmt, "Salario Mediano/sem"),
]

for col, val, label in kpis:
    with col:
        st.markdown(
            f"<div class='metric-card'><h2>{val}</h2><p>{label}</p></div>",
            unsafe_allow_html=True
        )

if not df.empty:
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        pos_dist = df["player_positions"].value_counts().reset_index()
        pos_dist.columns = ["Posición", "Cantidad"]
        fig_pos = px.bar(
            pos_dist, x="Posición", y="Cantidad",
            template="plotly_dark",
            title="Distribución por Posición"
        )
        fig_pos.update_traces(marker_color="#58a6ff")
        fig_pos.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            showlegend=False, height=350
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    with col_d2:
        age_dist = df["age_category"].value_counts().reset_index()
        age_dist.columns = ["Categoría", "Cantidad"]
        order = ["Joven (≤21)", "Promesa (22-25)", "Prime (26-29)", "Experimentado (30-32)", "Veterano (33+)"]
        age_dist["Categoría"] = pd.Categorical(age_dist["Categoría"], categories=order, ordered=True)
        age_dist = age_dist.sort_values("Categoría")
        fig_age = px.bar(
            age_dist, x="Categoría", y="Cantidad",
            template="plotly_dark",
            title="Distribución por Categoría de Edad"
        )
        fig_age.update_traces(marker_color="#58a6ff")
        fig_age.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            showlegend=False, height=350
        )
        st.plotly_chart(fig_age, use_container_width=True)


# ─────────────────────────────────────────────
# SECCIÓN 2: ANÁLISIS DE OPORTUNIDADES
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>🔥 Análisis de Oportunidades de Fichaje</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💎 Infravalorados",
    "📉 Sobrevalorados",
    "🌟 Talento Oculto",
    "💰 Eficiencia Salarial",
    "🔮 Jóvenes Promesas",
    "🏆 Top por Posición"
])


# ── TAB 1: JUGADORES INFRAVALORADOS ──────────────────────────────────────────
with tab1:
    st.markdown("**Criterio:** RPP > percentil 75 · Valor de mercado < percentil 50")

    if not df.empty:
        p75_rpp = df["rpp"].quantile(0.75)
        p50_val = df["value_eur"].quantile(0.50)

        underval = df[(df["rpp"] >= p75_rpp) & (df["value_eur"] <= p50_val)].copy()
        underval = underval.sort_values("opportunity_score", ascending=False)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_uv = px.scatter(
                underval, x="value_eur", y="rpp",
                hover_name="long_name",
                hover_data={"overall": True, "age": True, "player_positions": True, "opportunity_score": True},
                color="age_category",
                size="overall", size_max=18,
                template="plotly_dark",
                labels={"value_eur": "Valor (€)", "rpp": "RPP", "age_category": "Edad"},
                title=f"💎 {len(underval)} Jugadores Infravalorados"
            )
            fig_uv.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
            st.plotly_chart(fig_uv, use_container_width=True)

        with col_b:
            st.markdown(f"**{len(underval)} jugadores encontrados**")
            st.caption("Score de Oportunidad: RPP alto (45%) + valor bajo (30%) + salario bajo (25%)")
            cols_show = ["long_name", "player_positions", "age", "overall", "rpp", "value_eur", "opportunity_score"]
            cols_show = [c for c in cols_show if c in underval.columns]
            st.dataframe(
                underval[cols_show].head(20).rename(columns={
                    "long_name": "Jugador", "player_positions": "Pos", "age": "Edad",
                    "overall": "OVR", "value_eur": "Valor €", "opportunity_score": "Score"
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
                hover_data={"value_eur": True, "age": True, "player_positions": True},
                color="age_category",
                size="value_eur", size_max=18,
                template="plotly_dark",
                labels={"overall": "Overall", "rpp": "RPP", "age_category": "Edad"},
                title=f"📉 {len(overval)} Jugadores Sobrevalorados"
            )
            fig_ov.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
            st.plotly_chart(fig_ov, use_container_width=True)

        with col_b:
            st.markdown(f"**{len(overval)} jugadores encontrados**")
            cols_show = ["long_name", "player_positions", "age", "overall", "rpp", "value_eur"]
            cols_show = [c for c in cols_show if c in overval.columns]
            st.dataframe(
                overval[cols_show].head(20).rename(columns={
                    "long_name": "Jugador", "player_positions": "Pos", "age": "Edad",
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
                hover_data={"value_eur": True, "age": True, "player_positions": True},
                color="age_category",
                size="rpp", size_max=16,
                template="plotly_dark",
                labels={"overall": "Overall", "rpp": "RPP", "age_category": "Edad"},
                title=f"🌟 {len(hidden)} Talentos Ocultos"
            )
            fig_hid.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
            st.plotly_chart(fig_hid, use_container_width=True)

        with col_b:
            st.markdown(f"**{len(hidden)} jugadores encontrados**")
            cols_show = ["long_name", "player_positions", "age", "overall", "rpp", "value_eur"]
            cols_show = [c for c in cols_show if c in hidden.columns]
            st.dataframe(
                hidden[cols_show].head(20).rename(columns={
                    "long_name": "Jugador", "player_positions": "Pos", "age": "Edad",
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
                hover_data={"overall": True, "age": True, "player_positions": True, "rpp_wage_ratio": True},
                color="age_category",
                size="rpp", size_max=16,
                template="plotly_dark",
                labels={"wage_eur": "Salario (€/sem)", "rpp": "RPP", "age_category": "Edad"},
                title=f"💰 {len(efficient)} Jugadores Eficientes en Salario"
            )
            fig_eff.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
            st.plotly_chart(fig_eff, use_container_width=True)

        with col_b:
            st.markdown(f"**{len(efficient)} jugadores encontrados**")
            cols_show = ["long_name", "player_positions", "age", "overall", "rpp", "wage_eur", "rpp_wage_ratio"]
            cols_show = [c for c in cols_show if c in efficient.columns]
            st.dataframe(
                efficient[cols_show].head(20).rename(columns={
                    "long_name": "Jugador", "player_positions": "Pos", "age": "Edad",
                    "overall": "OVR", "wage_eur": "Salario €", "rpp_wage_ratio": "RPP/Sal."
                }),
                hide_index=True, use_container_width=True
            )


# ── TAB 5: JÓVENES PROMESAS ───────────────────────────────────────────────────
with tab5:
    st.markdown("**Criterio:** Edad ≤ 23 · RPP > percentil 60 en su posición · Valor < percentil 60")

    if not df.empty:
        pos_p60_rpp = df.groupby("player_positions")["rpp"].transform(lambda x: x.quantile(0.60))
        p60_val     = df["value_eur"].quantile(0.60)

        young = df[
            (df["age"] <= 23) &
            (df["rpp"] >= pos_p60_rpp) &
            (df["value_eur"] <= p60_val)
        ].copy()
        young = young.sort_values(["rpp", "age"], ascending=[False, True])

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_young = px.scatter(
                young, x="age", y="rpp",
                hover_name="long_name",
                hover_data={"overall": True, "value_eur": True, "player_positions": True},
                color="player_positions",
                size="overall", size_max=18,
                template="plotly_dark",
                labels={"age": "Edad", "rpp": "RPP", "player_positions": "Posición"},
                title=f"🔮 {len(young)} Jóvenes Promesas (≤23 años)"
            )
            fig_young.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400)
            st.plotly_chart(fig_young, use_container_width=True)

        with col_b:
            st.markdown(f"**{len(young)} jugadores encontrados**")
            cols_show = ["long_name", "player_positions", "age", "overall", "rpp", "value_eur"]
            cols_show = [c for c in cols_show if c in young.columns]
            st.dataframe(
                young[cols_show].head(20).rename(columns={
                    "long_name": "Jugador", "player_positions": "Pos", "age": "Edad",
                    "overall": "OVR", "value_eur": "Valor €"
                }),
                hide_index=True, use_container_width=True
            )

        # Gráfico: RPP promedio por edad
        st.markdown("**RPP promedio según edad**")
        rpp_by_age = df.groupby("age").agg(rpp_mean=("rpp", "mean"), n=("rpp", "count")).reset_index()
        rpp_by_age.columns = ["Edad", "RPP Promedio", "n"]

        fig_rpp_age = px.line(
            rpp_by_age, x="Edad", y="RPP Promedio",
            template="plotly_dark",
            title="Evolución del RPP Promedio por Edad",
            markers=True
        )
        fig_rpp_age.add_vline(x=23, line_dash="dot", line_color="#a371f7",
                              annotation_text="Corte joven (23)", annotation_position="top right")
        fig_rpp_age.add_vline(x=29, line_dash="dot", line_color="#3fb950",
                              annotation_text="Peak (29)", annotation_position="top right")

        # Anotación "n reducido" en el punto de mayor edad
        max_age_data = rpp_by_age[rpp_by_age["n"] < 5]["Edad"]
        if not max_age_data.empty:
            anno_age = int(max_age_data.min())
        else:
            anno_age = int(rpp_by_age["Edad"].max())
        anno_rpp = float(rpp_by_age[rpp_by_age["Edad"] == anno_age]["RPP Promedio"].values[0])

        fig_rpp_age.add_annotation(
            x=anno_age, y=anno_rpp,
            text="n reducido",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#f85149",
            font=dict(color="#f85149", size=11),
            bgcolor="#1a1a1a",
            bordercolor="#f85149",
            ax=40, ay=-30
        )

        fig_rpp_age.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_rpp_age, use_container_width=True)


# ── TAB 6: TOP POR POSICIÓN ───────────────────────────────────────────────────
with tab6:
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
                fig_top = go.Figure(go.Bar(
                    x=top10.sort_values("rpp")["rpp"],
                    y=top10.sort_values("rpp")["long_name"],
                    orientation="h",
                    marker=dict(
                        color=top10.sort_values("rpp")["overall"],
                        colorscale="Blues",
                        showscale=False
                    ),
                    hovertemplate="<b>%{y}</b><br>RPP: %{x}<extra></extra>"
                ))
                fig_top.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    title=f"🏆 Top 10 · {sel_pos_top}",
                    xaxis_title="RPP", yaxis_title="Jugador",
                    height=400
                )
                st.plotly_chart(fig_top, use_container_width=True)

            with col_b:
                cols_show = ["long_name", "age", "overall", "rpp", "value_eur", "wage_eur", "nationality_name"]
                cols_show = [c for c in cols_show if c in top10.columns]
                st.dataframe(
                    top10[cols_show].rename(columns={
                        "long_name": "Jugador", "age": "Edad", "overall": "OVR",
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

    with col_v1:
        fig_hist = px.histogram(
            df, x="rpp", nbins=40,
            color_discrete_sequence=["#58a6ff"],
            template="plotly_dark",
            title="Distribución de RPP",
            labels={"rpp": "RPP", "count": "Cantidad"}
        )
        fig_hist.update_yaxes(title_text="Cantidad")
        fig_hist.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_v2:
        fig_hist_age = px.histogram(
            df, x="age", nbins=30,
            color_discrete_sequence=["#a371f7"],
            template="plotly_dark",
            title="Distribución de Edad",
            labels={"age": "Edad", "count": "Cantidad"}
        )
        fig_hist_age.update_yaxes(title_text="Cantidad")
        fig_hist_age.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_hist_age, use_container_width=True)

    col_v3, col_v4 = st.columns(2)

    with col_v3:
        # Escala logarítmica en eje X para distribuir mejor los puntos
        fig_sc1 = px.scatter(
            df.sample(min(2000, len(df)), random_state=42),
            x="value_eur", y="rpp",
            hover_name="long_name",
            color="player_positions",
            opacity=0.7,
            template="plotly_dark",
            log_x=True,
            labels={"value_eur": "Valor (€) — escala log", "rpp": "RPP"},
            title="Valor de Mercado vs RPP (eje X logarítmico)"
        )
        fig_sc1.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_sc1, use_container_width=True)

    with col_v4:
        fig_sc2 = px.scatter(
            df.sample(min(2000, len(df)), random_state=42),
            x="age", y="rpp",
            hover_name="long_name",
            color="player_positions",
            opacity=0.7,
            template="plotly_dark",
            labels={"age": "Edad", "rpp": "RPP"},
            title="Edad vs RPP"
        )
        fig_sc2.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_sc2, use_container_width=True)

    col_v5, col_v6 = st.columns(2)

    with col_v5:
        fig_sc3 = px.scatter(
            df.sample(min(2000, len(df)), random_state=42),
            x="overall", y="rpp",
            hover_name="long_name",
            color="player_positions",
            opacity=0.7,
            template="plotly_dark",
            labels={"overall": "Overall", "rpp": "RPP"},
            title="Overall vs RPP"
        )
        fig_sc3.add_shape(
            type="line",
            x0=df["overall"].min(), x1=df["overall"].max(),
            y0=df["overall"].min(), y1=df["overall"].max(),
            line=dict(color="white", dash="dot", width=1)
        )
        fig_sc3.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_sc3, use_container_width=True)

    with col_v6:
        top_pos_for_box = df["player_positions"].value_counts().head(12).index.tolist()
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
st.caption("Oport. Score: combina RPP alto (45%) + valor de mercado bajo (30%) + salario bajo (25%), relativo a la posición del jugador. Escala 0–100.")

if not df.empty:
    sort_col = st.selectbox(
        "Ordenar por",
        ["rpp", "overall", "age", "value_eur", "wage_eur", "opportunity_score"],
        index=0
    )
    sort_asc = st.checkbox("Orden ascendente", value=False)

    display_cols = [
        "long_name", "player_positions", "nationality_name", "age",
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
            "nationality_name": "País", "age": "Edad", "overall": "OVR",
            "value_eur": "Valor €", "wage_eur": "Salario €/sem",
            "opportunity_score": "Oport. Score", "rpp_value_ratio": "RPP/Valor"
        }),
        use_container_width=True,
        height=420
    )
    st.caption(f"Mostrando {len(df_display):,} jugadores")


# ─────────────────────────────────────────────
# SECCIÓN 5: RANKING DE OPORTUNIDADES
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-title'>🏅 Ranking de Oportunidades — Top 20 Gangas</div>", unsafe_allow_html=True)
st.caption("Score de Oportunidad: RPP alto (45%) + valor de mercado bajo (30%) + salario bajo (25%), calculado por percentiles dentro de cada posición. Escala 0–100.")

if not df.empty:
    steal_threshold = df["opportunity_score"].quantile(0.90)
    steals = (
        df[df["opportunity_score"] >= steal_threshold]
        .sort_values("opportunity_score", ascending=False)
        .head(20)
    )

    steals_sorted = steals.sort_values("opportunity_score").reset_index(drop=True)
    n = len(steals_sorted)

    # Color base azul oscuro; top 3 (últimas 3 filas tras sort asc) en dorado
    bar_colors = ["#1c3a5e"] * n
    for i in range(max(0, n - 3), n):
        bar_colors[i] = "#d29922"

    fig_rank = go.Figure(go.Bar(
        x=steals_sorted["opportunity_score"],
        y=steals_sorted["long_name"],
        orientation="h",
        marker=dict(color=bar_colors),
        customdata=steals_sorted[["player_positions", "age", "overall", "rpp", "value_eur"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score: %{x:.1f}<br>"
            "Posición: %{customdata[0]}<br>"
            "Edad: %{customdata[1]}<br>"
            "Overall: %{customdata[2]}<br>"
            "RPP: %{customdata[3]:.1f}<br>"
            "Valor: €%{customdata[4]:,.0f}<extra></extra>"
        )
    ))
    fig_rank.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        title="💎 Top 20 Jugadores con Mayor Score de Oportunidad  ·  🟡 Top 3 destacados",
        xaxis_title="Score de Oportunidad",
        yaxis_title="Jugador",
        height=560
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # Tarjetas top 3
    st.markdown("#### 🔝 Las 3 Mejores Gangas")
    top3 = steals.head(3)
    card_cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with card_cols[i]:
            val_fmt = fmt_eur(row["value_eur"])
            age_val = int(row["age"]) if pd.notna(row["age"]) else "?"
            st.markdown(
                f"""
                <div class='metric-card' style='text-align:left;'>
                    <span class='badge-steal'>⭐ GANGA</span><br><br>
                    <b style='font-size:1.1rem;color:#f0f6fc;'>{row['long_name']}</b><br>
                    <span style='color:#8b949e;'>Posición: <b style='color:#58a6ff;'>{row['player_positions']}</b></span>&nbsp;
                    <span style='color:#8b949e;'>Edad: <b style='color:#a371f7;'>{age_val}</b></span><br><br>
                    <span style='color:#8b949e;'>Overall: </span><b>{int(row['overall'])}</b>&nbsp;&nbsp;
                    <span style='color:#8b949e;'>RPP: </span><b>{row['rpp']:.1f}</b><br>
                    <span style='color:#8b949e;'>Valor: </span><b style='color:#3fb950;'>{val_fmt}</b><br>
                    <span style='color:#8b949e;'>Score: </span><b style='color:#d29922;'>{row['opportunity_score']}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")
st.caption("FIFA 23 Scout Intelligence · Desarrollado con Streamlit + Plotly · Fuente de datos: FIFA 23 Complete Player Dataset — Kaggle (sofifa.com)")
