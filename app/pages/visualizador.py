import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import zipfile
import tempfile
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

st.image("logo.png", use_container_width=True)
st.markdown("<h1 style='text-align: center; color: white;'>Visualizador de Resultados</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Envie o arquivo .zip com os resultados:", type="zip")

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        possible_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        if not possible_dirs:
            st.error("Nenhuma pasta encontrada dentro do .zip.")
        else:
            st.subheader("Selecione a Pasta de Avaliação")
            selected_root = st.selectbox("Pasta de Avaliação:", possible_dirs)
            base_dir = os.path.join(tmpdir, selected_root)

            is_hierarchical = "_hierarquicos_" in selected_root.lower()
            series = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
            selected_series = st.selectbox("Selecione a Série:", series)
            methods_path = os.path.join(base_dir, selected_series)
            methods = sorted(os.listdir(methods_path))
            methods.remove("scores.csv")

            time_series_dir = os.path.join("time_series")
            serie_txt_path = os.path.join(time_series_dir, f"{selected_series}.txt")
            if os.path.exists(serie_txt_path):
                with open(serie_txt_path, 'r') as f:
                    line = f.readline()
                    serie = np.array([float(val) for val in line.strip().split(',')])
                st.success(f"Série original carregada de: {serie_txt_path}")
            else:
                st.warning(f"Série '{selected_series}' não encontrada em: {serie_txt_path}")
                serie = None

            selected_methods = st.multiselect("Selecione os Métodos a Comparar:", methods)
            st.subheader(f"Série: {selected_series}")
            start_app = st.button("Iniciar")

            if selected_methods and start_app:
                if is_hierarchical:
                    st.subheader("Scores do Clustering Hierárquico")
                    scores_path = os.path.join(base_dir, selected_series, "scores.csv")
                    scores_df = pd.read_csv(scores_path)
                    st.dataframe(scores_df)

                    cluster_col = None
                    score_col = None
                    for col in scores_df.columns:
                        if "cluster" in col.lower():
                            cluster_col = col
                        if "score" in col.lower():
                            score_col = col

                    st.subheader("Dendrogramas dos Submétodos Selecionados")
                    for method in selected_methods:
                        output_dir = os.path.join(methods_path, method)
                        img_path = os.path.join(output_dir, 'dendrogram.png')
                        if os.path.exists(img_path):
                            st.image(Image.open(img_path), caption=f"{method} - dendrogram.png", use_container_width=True)
                            
                    # --- 1️⃣ Line plot: Silhouette Score vs Número de Clusters ---
                    st.subheader("Silhouette Score vs Número de Clusters")

                    fig1 = px.line(
                        scores_df,
                        x='param',
                        y='silhouette_score',
                        color='linkage',
                        line_dash='metric',
                        markers=True,
                        title='Silhouette Score vs Número de Clusters',
                        labels={
                            'param': 'Número de Clusters',
                            'silhouette_score': 'Silhouette Score',
                            'linkage': 'Linkage',
                            'metric': 'Metric'
                        }
                    )

                    st.plotly_chart(fig1, use_container_width=True)

                    # --- 2️⃣ Boxplot por Linkage ---
                    st.subheader("Distribuição do Silhouette Score por Linkage")

                    fig2 = px.box(
                        scores_df,
                        x='linkage',
                        y='silhouette_score',
                        color='linkage',
                        title='Distribuição do Silhouette Score por Linkage',
                        labels={
                            'linkage': 'Linkage',
                            'silhouette_score': 'Silhouette Score'
                        }
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                    # --- 3️⃣ Boxplot por Metric ---
                    st.subheader("Distribuição do Silhouette Score por Métrica de Distância")

                    fig3 = px.box(
                        scores_df,
                        x='metric',
                        y='silhouette_score',
                        color='metric',
                        title='Distribuição do Silhouette Score por Métrica de Distância',
                        labels={
                            'metric': 'Métrica de Distância',
                            'silhouette_score': 'Silhouette Score'
                        }
                    )

                    st.plotly_chart(fig3, use_container_width=True)

                    # --- 4️⃣ Heatmap Linkage vs Número de Clusters ---
                    st.subheader("Heatmap - Silhouette Score (Linkage vs Número de Clusters)")

                    pivot_df = scores_df.pivot_table(
                        index='linkage',
                        columns='param',
                        values='silhouette_score'
                    )

                    fig4 = ff.create_annotated_heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns.tolist(),
                        y=pivot_df.index.tolist(),
                        annotation_text=pivot_df.round(3).values,
                        colorscale='Viridis',
                        showscale=True
                    )

                    fig4.update_layout(
                        title='Heatmap - Silhouette Score (Linkage vs Número de Clusters)',
                        xaxis_title='Número de Clusters',
                        yaxis_title='Linkage'
                    )

                    st.plotly_chart(fig4, use_container_width=True)

                else:
                    fig_shapes = go.Figure()
                    fig_series = go.Figure()
                    if serie is not None:
                        fig_series.add_trace(go.Scatter(
                            y=serie,
                            mode='lines',
                            name='Série Original',
                            line=dict(color='lightgray'),
                            opacity=0.7
                        ))

                    metrics_list = []
                    for method in selected_methods:
                        output_dir = os.path.join(methods_path, method)
                        st.markdown(f"### Método: {method}")

                        for img_name in ['regime_bar.png', 'dendrograma.png', 'silhouette.png']:
                            img_path = os.path.join(output_dir, img_name)
                            if os.path.exists(img_path):
                                st.image(Image.open(img_path), caption=f"{method} - {img_name}", use_container_width=True)

                        snippets = None
                        snippets_json = os.path.join(output_dir, 'snippets.json')
                        if os.path.exists(snippets_json):
                            with open(snippets_json) as f:
                                snippets = json.load(f)
                        elif os.path.exists(os.path.join(output_dir, 'snippets.npy')):
                            snippets = np.load(os.path.join(output_dir, 'snippets.npy'), allow_pickle=True)
                            snippets = [{'index': int(s[0]), 'subsequence': s[1].tolist()} for s in snippets]

                        if snippets:
                            for i, snip in enumerate(snippets[:3]):
                                fig_shapes.add_trace(go.Scatter(
                                    y=snip['subsequence'],
                                    mode='lines',
                                    name=f'{method} - Snippet {i+1}'
                                ))
                            if serie is not None:
                                subseq_size = len(snippets[0]['subsequence'])
                                for i, snip in enumerate(snippets[:3]):
                                    idx = snip['index']
                                    x_vals = list(range(idx, idx + subseq_size))
                                    y_vals = serie[idx:idx + subseq_size]
                                    fig_series.add_trace(go.Scatter(
                                        x=x_vals,
                                        y=y_vals,
                                        mode='lines',
                                        name=f'{method} - Snippet {i+1}',
                                        line=dict(width=3)
                                    ))

                        metrics_path = os.path.join(output_dir, 'metrics.json')
                        if os.path.exists(metrics_path):
                            with open(metrics_path) as f:
                                metrics = json.load(f)
                            flat_metrics = {}
                            for k, v in metrics.items():
                                if isinstance(v, dict):
                                    for sub_k, sub_v in v.items():
                                        flat_metrics[f"{k}.{sub_k}"] = sub_v
                                else:
                                    flat_metrics[k] = v
                            for metric_name, metric_value in flat_metrics.items():
                                metrics_list.append({
                                    "Método": method,
                                    "Métrica": metric_name,
                                    "Valor": metric_value
                                })

                    st.subheader("Shapes dos Snippets Comparados")
                    st.plotly_chart(fig_shapes, use_container_width=True)

                    st.subheader("Snippets sobrepostos na Série Temporal")
                    st.plotly_chart(fig_series, use_container_width=True)

                    if metrics_list:
                        st.subheader("Métricas Comparativas")
                        metrics_df = pd.DataFrame(metrics_list)
                        st.dataframe(metrics_df)
                        st.subheader("Gráficos Comparativos das Métricas")
                        unique_metrics = metrics_df["Métrica"].unique()
                        for metric in unique_metrics:
                            metric_data = metrics_df[metrics_df["Métrica"] == metric].reset_index()
                            if isinstance(metric_data["Valor"].iloc[0], list):
                                continue
                            if metric_data['Métrica'].iloc[0] == 'cover_area_fractions':
                                fig_metric = go.Figure()
                                fig_metric.add_trace(go.Bar(
                                    x=metric_data["Método"],
                                    y=metric_data["Valor"].apply(lambda x: round(x*100, 2)),
                                    text=metric_data["Valor"],
                                    textposition='outside'
                                ))
                            else:
                                fig_metric = go.Figure()
                                fig_metric.add_trace(go.Bar(
                                    x=metric_data["Método"],
                                    y=metric_data["Valor"].apply(lambda x: round(x, 2)),
                                    text=metric_data["Valor"],
                                    textposition='outside'
                                ))
                            fig_metric.update_layout(
                                title=f"Comparação da Métrica: {metric}",
                                xaxis_title="Método",
                                yaxis_title="Valor",
                                bargap=0.4
                            )
                            st.plotly_chart(fig_metric, use_container_width=True)
