import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Configuração da página ---
st.set_page_config(
    page_title="Dashboard Nutricional",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Leitura dos dados ---
df = pd.read_excel("/home/leandro/VS Code Projetos/food.xlsx")

# Converter colunas numéricas
for col in df.columns:
    if col.startswith("Data."):
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Paleta de cores ---
cores = {
    "Data.Kilocalories": "#F94144",
    "Data.Protein": "#277DA1",
    "Data.Carbohydrate": "#F9C74F",
    "Data.Fat.Total Lipid": "#F9844A"
}

# --- Tema ---
tema = st.radio("🌗 Escolha o tema do gráfico:", ["Automático", "Claro", "Escuro"], horizontal=True)
if tema == "Claro":
    template = "plotly_white"
elif tema == "Escuro":
    template = "plotly_dark"
else:
    template = "plotly"

# --- Título ---
st.title("🍎 Dashboard Nutricional Completo com Machine Learning")

# --- Filtros globais ---
categorias = st.multiselect("Filtrar por categoria:", df["Category"].unique())
if categorias:
    df = df[df["Category"].isin(categorias)]

# --- Abas principais ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Métricas Gerais",
    "🥗 Comparação por Categoria",
    "🏆 Top 10 Alimentos",
    "🔍 Análise Individual",
    "🤖 Machine Learning"
])

# ============================================================== #
# 📊 ABA 1 — MÉTRICAS GERAIS
# ============================================================== #
with tab1:
    st.subheader("📊 Visão Geral dos Alimentos")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Qtd. Alimentos", len(df))
    col2.metric("Calorias Médias", f"{df['Data.Kilocalories'].mean():.1f}")
    col3.metric("Proteína Média", f"{df['Data.Protein'].mean():.1f} g")
    col4.metric("Gordura Média", f"{df['Data.Fat.Total Lipid'].mean():.1f} g")
    st.markdown("---")
    st.info("Use as abas acima para navegar entre as análises detalhadas.")

# ============================================================== #
# 🥗 ABA 2 — COMPARAÇÃO POR CATEGORIA
# ============================================================== #
with tab2:
    st.subheader("🥗 Nutrientes Médios por Categoria")
    nutrientes = ["Data.Kilocalories", "Data.Protein", "Data.Carbohydrate", "Data.Fat.Total Lipid"]
    nutriente = st.selectbox("Escolha o nutriente:", nutrientes, index=0)
    cat_media = df.groupby("Category")[nutrientes].mean().reset_index()
    top20 = cat_media.nlargest(20, nutriente).sort_values(by=nutriente)
    fig_cat = px.bar(
        top20, x=nutriente, y="Category", orientation="h",
        text_auto=".1f", color=nutriente, title=f"Top 20 Categorias - {nutriente.split('.')[-1]}",
        template=template, color_continuous_scale="YlOrRd"
    )
    hover_text = ("<b>Categoria:</b> %{y}<br>" f"<b>{nutriente.split('.')[-1]}:</b> %%{{x:.2f}}<extra></extra>")
    fig_cat.update_traces(hovertemplate=hover_text, textposition="outside")
    fig_cat.update_layout(xaxis_title=nutriente.split(".")[-1], yaxis_title=None, showlegend=False,
                          margin=dict(t=60, b=60, l=150), height=600)
    st.plotly_chart(fig_cat, use_container_width=True)
    st.markdown("### 📋 Detalhes das Categorias")
    st.dataframe(
        top20[["Category", nutriente]].rename(columns={"Category": "Categoria", nutriente: nutriente.split(".")[-1]}),
        use_container_width=True, hide_index=True
    )

# ============================================================== #
# 🏆 ABA 3 — TOP 10 ALIMENTOS
# ============================================================== #
with tab3:
    st.subheader("🏆 Top 10 Alimentos")
    coluna = st.selectbox("Escolha o nutriente:", nutrientes, index=1)
    if "Code" in df.columns: id_col = "Code"
    elif "ID" in df.columns: id_col = "ID"
    elif "Código" in df.columns: id_col = "Código"
    else: df = df.reset_index().rename(columns={"index": "Código"}); id_col = "Código"
    top10 = df.nlargest(10, coluna)[[id_col, "Description", coluna]].copy()
    top10_sorted = top10.sort_values(coluna)
    fig_rank = px.bar(
        top10_sorted, x=top10_sorted[id_col].astype(str), y=coluna, title=f"Top 10 Alimentos - {coluna.split('.')[-1]}",
        text_auto=".1f", color=coluna, orientation="v", template=template, color_continuous_scale="Agsunset"
    )
    hover_text = ("<b>Código:</b> %{x}<br><b>Descrição:</b> %{customdata[0]}<br>" + "<b>" + coluna.split(".")[-1] + ":</b> %{y:.2f}<extra></extra>")
    fig_rank.update_traces(hovertemplate=hover_text, customdata=top10_sorted[["Description"]].values, textposition="outside")
    fig_rank.update_layout(xaxis_title="Código do Alimento", yaxis_title=coluna.split(".")[-1], xaxis_tickangle=0,
                           showlegend=False, margin=dict(t=60, b=60), height=450)
    st.plotly_chart(fig_rank, use_container_width=True)
    st.markdown("### 🧾 Detalhes dos Alimentos")
    st.dataframe(
        top10_sorted[[id_col, "Description", coluna]].rename(columns={id_col: "Código", "Description": "Descrição", coluna: coluna.split(".")[-1]}),
        use_container_width=True, hide_index=True
    )

# ============================================================== #
# 🔍 ABA 4 — ANÁLISE INDIVIDUAL
# ============================================================== #
with tab4:
    st.subheader("🔍 Perfil Nutricional de um Alimento")
    alimento = st.selectbox("Selecione um alimento:", df["Description"].unique())
    item = df[df["Description"] == alimento].iloc[0]
    dados_item = {
        "Calorias (kcal)": item["Data.Kilocalories"] if not np.isnan(item["Data.Kilocalories"]) else 0,
        "Proteína (g)": item["Data.Protein"] if not np.isnan(item["Data.Protein"]) else 0,
        "Carboidrato (g)": item["Data.Carbohydrate"] if not np.isnan(item["Data.Carbohydrate"]) else 0,
        "Gordura (g)": item["Data.Fat.Total Lipid"] if not np.isnan(item["Data.Fat.Total Lipid"]) else 0
    }
    df_item = pd.DataFrame(list(dados_item.items()), columns=["Nutriente", "Valor"])
    cores_item_map = {
        "Calorias (kcal)": cores["Data.Kilocalories"],
        "Proteína (g)": cores["Data.Protein"],
        "Carboidrato (g)": cores["Data.Carbohydrate"],
        "Gordura (g)": cores["Data.Fat.Total Lipid"]
    }
    fig_item = px.bar(df_item, x="Nutriente", y="Valor", text_auto=".1f", title=f"Perfil Nutricional de {alimento}",
                      color="Nutriente", orientation="v", template=template, color_discrete_map=cores_item_map)
    fig_item.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None, margin=dict(t=60, b=60), height=400)
    st.plotly_chart(fig_item, use_container_width=True)

# ============================================================== #
# 🤖 ABA 5 — MACHINE LEARNING INTUITIVO
# ============================================================== #
with tab5:
    st.subheader("🤖 Machine Learning Nutricional")
    st.markdown("""
    Este painel permite explorar previsões, agrupamentos e recomendações de alimentos """)

    # --- Previsão de Calorias ---
    st.markdown("### 🔮 Previsão de Calorias de um Alimento")
    st.markdown("Insira os valores de **proteínas, carboidratos e gorduras** de um alimento e veja a **estimativa de calorias**.")

    X = df[["Data.Protein","Data.Carbohydrate","Data.Fat.Total Lipid"]]
    y = df["Data.Kilocalories"]
    pipeline_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline_rf.fit(X, y)

    protein = st.number_input("Proteína (g):", min_value=0.0)
    carb = st.number_input("Carboidrato (g):", min_value=0.0)
    fat = st.number_input("Gordura (g):", min_value=0.0)
    if st.button("Prever Calorias"):
        cal_pred = pipeline_rf.predict([[protein, carb, fat]])[0]
        st.success(f"🍎 Calorias estimadas: **{cal_pred:.1f} kcal**")

        importances = pipeline_rf.named_steps["model"].feature_importances_
        fig_importance = px.bar(
            x=["Proteína", "Carboidrato", "Gordura"], 
            y=importances,
            text=[f"{i:.2f}" for i in importances],
            labels={"x":"Nutriente", "y":"Importância"},
            title="🔑 Importância das Variáveis no Modelo",
            template=template,
            color=["#277DA1","#F9C74F","#F9844A"]
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("---")



    st.markdown("### 🧩 Agrupamento de Alimentos (Clusters)")
    st.markdown("""
    Agrupamos os alimentos por perfis nutricionais semelhantes.
    Cada cor representa um grupo, e o **losango preto** mostra o centro médio de cada cluster.
    """)

    # --- Seleção das variáveis usadas no agrupamento ---
    X_cluster = df[["Data.Kilocalories", "Data.Protein", "Data.Carbohydrate", "Data.Fat.Total Lipid"]]

    # --- Escolha de número de clusters ---
    k = st.slider("Número de grupos (clusters):", 2, 10, 4)

    # --- Criação do pipeline de processamento ---
    pipeline_kmeans = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=42))
    ])
    pipeline_kmeans.fit(X_cluster)

    # --- Criação do DataFrame clusterizado ---
    df_clustered = df.copy()
    df_clustered["Cluster"] = pipeline_kmeans.named_steps["kmeans"].labels_

    # --- Resumo dos clusters ---
    cluster_summary = (
        df_clustered.groupby("Cluster")[["Data.Kilocalories", "Data.Protein", "Data.Carbohydrate", "Data.Fat.Total Lipid"]]
        .mean()
        .round(1)
    )
    cluster_summary["Qtd. Itens"] = df_clustered.groupby("Cluster").size()
    cluster_summary["Descrição Nutricional"] = cluster_summary.apply(
        lambda row: f"Cal: {row['Data.Kilocalories']}, Prot: {row['Data.Protein']}, Carb: {row['Data.Carbohydrate']}, Gord: {row['Data.Fat.Total Lipid']}",
        axis=1
    )

    # --- Descrição automática simplificada ---
    def gerar_descricao(row):
        if row["Data.Protein"] > row["Data.Carbohydrate"] and row["Data.Fat.Total Lipid"] > row["Data.Carbohydrate"]:
            return "Rico em proteína e gordura"
        elif row["Data.Carbohydrate"] > row["Data.Protein"] and row["Data.Carbohydrate"] > row["Data.Fat.Total Lipid"]:
            return "Fonte principal de carboidratos"
        elif row["Data.Kilocalories"] < 150:
            return "Alimentos leves e de baixo valor calórico"
        else:
            return "Perfil nutricional equilibrado"

    cluster_summary["Descrição Simplificada"] = cluster_summary.apply(gerar_descricao, axis=1)

    # --- Exibe resumo geral dos clusters ---
    st.markdown("#### 📋 Resumo Geral dos Clusters")
    st.dataframe(cluster_summary[["Qtd. Itens", "Descrição Nutricional", "Descrição Simplificada"]], use_container_width=True)

    # --- Seleção de cluster para explorar ---
    st.markdown("#### 🔍 Visualizar Detalhes de um Cluster Específico")
    selected_cluster = st.selectbox("Selecione um cluster para visualizar:", sorted(df_clustered["Cluster"].unique()))

    # --- Filtra dados do cluster selecionado ---
    cluster_data = df_clustered[df_clustered["Cluster"] == selected_cluster]
    st.info(f"**Cluster {selected_cluster}:** {cluster_summary.loc[selected_cluster, 'Descrição Simplificada']}")

    # --- Tabela de alimentos do cluster selecionado ---
    st.markdown(f"##### 🍽️ Alimentos pertencentes ao Cluster {selected_cluster}")
    st.dataframe(
        cluster_data[["Description", "Data.Kilocalories", "Data.Protein", "Data.Carbohydrate", "Data.Fat.Total Lipid"]],
        use_container_width=True, hide_index=True
    )

    # --- Escolha dinâmica dos eixos ---
    st.markdown("#### ⚙️ Escolha as variáveis para visualizar no gráfico")

    col1, col2 = st.columns(2)
    opcoes = ["Data.Kilocalories", "Data.Protein", "Data.Carbohydrate", "Data.Fat.Total Lipid"]

    x_var = col1.selectbox("Eixo X:", opcoes, index=1, format_func=lambda x: x.split('.')[-1])
    y_var = col2.selectbox("Eixo Y:", opcoes, index=2, format_func=lambda x: x.split('.')[-1])

    # --- Gráfico interativo baseado nas escolhas ---
    fig_cluster = px.scatter(
    df_clustered,
    x=x_var,
    y=y_var,
    color=df_clustered["Cluster"].astype(str),  # <- tratar como categórico
    hover_name="Description",
    size="Data.Kilocalories",
    title=f"📊 Clusters de Alimentos ({x_var.split('.')[-1]} vs {y_var.split('.')[-1]})",
    template="plotly_white"
)

    # --- Adiciona centros dos clusters ---
    centers = cluster_summary.reset_index()
    fig_cluster.add_scatter(
        x=centers[x_var],
        y=centers[y_var],
        mode="markers+text",
        marker=dict(size=15, color="black", symbol="diamond"),
        text=centers["Cluster"].astype(str),
        textposition="top center",
        name="Centros"
    )

    # --- Destaca cluster selecionado ---
    for trace in fig_cluster.data:
        if f"{selected_cluster}" not in trace.name:
            trace.opacity = 0.2  # desbota os demais

    fig_cluster.update_layout(
        xaxis_title=x_var.split('.')[-1],
        yaxis_title=y_var.split('.')[-1],
        legend_title_text="Cluster",
        title_x=0.05
    )

    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown(f"🔹 **Cluster {selected_cluster}** contém {len(cluster_data)} alimentos com perfis nutricionais semelhantes.")
    st.markdown("🔸 Use os menus acima para alterar as variáveis visualizadas e explorar diferentes relações nutricionais.")
    st.markdown("---")


    # --- Recomendação de alimentos ---
    st.markdown("### 🍏 Recomendação de Alimentos Semelhantes")
    st.markdown("Selecione um alimento e veja **5 alimentos com perfil nutricional parecido**.")

    alimento_rec = st.selectbox("Escolha um alimento para recomendações:", df["Description"].unique())

    def recomendar_alimentos(alimento, df, top_n=5):
        # Corrigido: flatten para item 1D
        item = df[df['Description']==alimento][["Data.Kilocalories","Data.Protein","Data.Carbohydrate","Data.Fat.Total Lipid"]].fillna(0).values.flatten()
        df_filled = df[["Data.Kilocalories","Data.Protein","Data.Carbohydrate","Data.Fat.Total Lipid"]].fillna(0)
        df_copy = df.copy()
        df_copy['dist'] = df_filled.apply(lambda x: np.linalg.norm(x-item), axis=1)
        return df_copy.nsmallest(top_n+1,'dist')['Description'].tolist()[1:]

    if st.button("Recomendar"):
        recomendados = recomendar_alimentos(alimento_rec, df)
        st.success("Alimentos semelhantes encontrados:")
        for r in recomendados:
            st.write(f"🍽️ {r}")
