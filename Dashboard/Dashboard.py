import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuração da página ---
st.set_page_config(
    page_title="Dashboard Nutricional",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Leitura dos dados ---
df = pd.read_excel("C:/Users/leand/VS Code Projetos/food.xlsx")

# Converter colunas numéricas
for col in df.columns:
    if col.startswith("Data."):
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Paleta de cores personalizada ---
cores = {
    "Data.Kilocalories": "#F94144",   # vermelho (calorias)
    "Data.Protein": "#277DA1",        # azul (proteína)
    "Data.Carbohydrate": "#F9C74F",   # amarelo (carboidrato)
    "Data.Fat.Total Lipid": "#F9844A" # laranja (gordura)
}

# --- Tema escuro/claro ---
tema = st.radio("🌗 Escolha o tema do gráfico:", ["Automático", "Claro", "Escuro"], horizontal=True)

if tema == "Claro":
    template = "plotly_white"
elif tema == "Escuro":
    template = "plotly_dark"
else:
    template = "plotly"

# --- Título ---
st.title("🍎 Dashboard Nutricional Completo")

# --- Filtros globais ---
categorias = st.multiselect("Filtrar por categoria:", df["Category"].unique())
if categorias:
    df = df[df["Category"].isin(categorias)]

# --- Abas principais ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Métricas Gerais",
    "🥗 Comparação por Categoria",
    "🏆 Top 10 Alimentos",
    "🔍 Análise Individual"
])

# ==============================================================
# 📊 ABA 1 — MÉTRICAS GERAIS
# ==============================================================

with tab1:
    st.subheader("📊 Visão Geral dos Alimentos")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Qtd. Alimentos", len(df))
    col2.metric("Calorias Médias", f"{df['Data.Kilocalories'].mean():.1f}")
    col3.metric("Proteína Média", f"{df['Data.Protein'].mean():.1f} g")
    col4.metric("Gordura Média", f"{df['Data.Fat.Total Lipid'].mean():.1f} g")

    st.markdown("---")
    st.info("Use as abas acima para navegar entre as análises detalhadas.")

# ==============================================================
# 🥗 ABA 2 — COMPARAÇÃO POR CATEGORIA
# ==============================================================

with tab2:
    st.subheader("🥗 Nutrientes Médios por Categoria")

    # Lista de nutrientes disponíveis
    nutrientes = [
        "Data.Kilocalories",
        "Data.Protein",
        "Data.Carbohydrate",
        "Data.Fat.Total Lipid"
    ]

    # Seletor de nutriente
    nutriente = st.selectbox("Escolha o nutriente:", nutrientes, index=0)

    # Média por categoria
    cat_media = df.groupby("Category")[nutrientes].mean().reset_index()

    # Seleciona as 20 categorias com maior valor no nutriente escolhido
    top20 = cat_media.nlargest(20, nutriente).sort_values(by=nutriente)

    # --- Gráfico ---
    fig_cat = px.bar(
        top20,
        x=nutriente,
        y="Category",
        orientation="h",  # horizontal para melhor legibilidade
        text_auto=".1f",
        color=nutriente,
        title=f"Top 20 Categorias - {nutriente.split('.')[-1]}",
        template=template,
        color_continuous_scale="YlOrRd"
    )

    # Tooltip com valor detalhado
    hover_text = (
    "<b>Categoria:</b> %{y}<br>"
    f"<b>{nutriente.split('.')[-1]}:</b> %%{{x:.2f}}<extra></extra>"
)

    fig_cat.update_traces(
        hovertemplate=hover_text,
        textposition="outside"
    )

    fig_cat.update_layout(
        xaxis_title=nutriente.split(".")[-1],
        yaxis_title=None,
        showlegend=False,
        margin=dict(t=60, b=60, l=150),
        height=600
    )

    # Exibe o gráfico
    st.plotly_chart(fig_cat, use_container_width=True)

    # --- Tabela complementar ---
    st.markdown("### 📋 Detalhes das Categorias")
    st.dataframe(
        top20[["Category", nutriente]].rename(
            columns={
                "Category": "Categoria",
                nutriente: nutriente.split(".")[-1]
            }
        ),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================
# 🏆 ABA 3 — TOP 10 ALIMENTOS
# ==============================================================

with tab3:
    st.subheader("🏆 Top 10 Alimentos")

    coluna = st.selectbox("Escolha o nutriente:", nutrientes, index=1)

    # Verifica coluna de código
    if "Code" in df.columns:
        id_col = "Code"
    elif "ID" in df.columns:
        id_col = "ID"
    elif "Código" in df.columns:
        id_col = "Código"
    else:
        df = df.reset_index().rename(columns={"index": "Código"})
        id_col = "Código"

    # Seleciona top 10
    top10 = df.nlargest(10, coluna)[[id_col, "Description", coluna]].copy()

    # Ordena crescente para visual
    top10_sorted = top10.sort_values(coluna)

    # --- Gráfico ---
    fig_rank = px.bar(
        top10_sorted,
        x=top10_sorted[id_col].astype(str),  # usa só códigos no eixo X
        y=coluna,
        title=f"Top 10 Alimentos - {coluna.split('.')[-1]}",
        text_auto=".1f",
        color=coluna,
        orientation="v",
        template=template,
        color_continuous_scale="Agsunset"
    )

    # Tooltip com descrição completa
    hover_text = (
        "<b>Código:</b> %{x}<br>"
        "<b>Descrição:</b> %{customdata[0]}<br>"
        "<b>" + coluna.split(".")[-1] + ":</b> %{y:.2f}<extra></extra>"
    )

    fig_rank.update_traces(
        hovertemplate=hover_text,
        customdata=top10_sorted[["Description"]].values,
        textposition="outside"
    )

    fig_rank.update_layout(
        xaxis_title="Código do Alimento",
        yaxis_title=coluna.split(".")[-1],
        xaxis_tickangle=0,
        showlegend=False,
        margin=dict(t=60, b=60),
        height=450
    )

    st.plotly_chart(fig_rank, use_container_width=True)

    # --- Tabela detalhada ---
    st.markdown("### 🧾 Detalhes dos Alimentos")
    st.dataframe(
        top10_sorted[[id_col, "Description", coluna]].rename(
            columns={
                id_col: "Código",
                "Description": "Descrição",
                coluna: coluna.split(".")[-1]
            }
        ),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================
# 🔍 ABA 4 — ANÁLISE INDIVIDUAL
# ==============================================================

with tab4:
    st.subheader("🔍 Perfil Nutricional de um Alimento")

    alimento = st.selectbox("Selecione um alimento:", df["Description"].unique())
    item = df[df["Description"] == alimento].iloc[0]

    dados_item = {
        "Calorias (kcal)": item["Data.Kilocalories"],
        "Proteína (g)": item["Data.Protein"],
        "Carboidrato (g)": item["Data.Carbohydrate"],
        "Gordura (g)": item["Data.Fat.Total Lipid"]
    }

    df_item = pd.DataFrame(list(dados_item.items()), columns=["Nutriente", "Valor"])

    # Define cores consistentes com a paleta
    cores_item = [
        cores["Data.Kilocalories"],
        cores["Data.Protein"],
        cores["Data.Carbohydrate"],
        cores["Data.Fat.Total Lipid"]
    ]

    fig_item = px.bar(
        df_item,
        x="Nutriente",
        y="Valor",
        text_auto=".1f",
        title=f"Perfil Nutricional de {alimento}",
        color="Nutriente",
        orientation="v",
        template=template,
        color_discrete_sequence=cores_item
    )
    fig_item.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(t=60, b=60),
        height=400
    )
    st.plotly_chart(fig_item, use_container_width=True)
