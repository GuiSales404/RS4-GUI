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


st.title("Visualizador de Resultados de Clusterização via ZIP")

uploaded_file = st.file_uploader("Envie o arquivo .zip com os resultados:", type="zip")

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Salva e extrai o zip
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Detecta a pasta base de resultados
        possible_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        if not possible_dirs:
            st.error("Nenhuma pasta encontrada dentro do .zip.")
        else:
            base_dir = os.path.join(tmpdir, possible_dirs[0])  # Assume primeira pasta
            series = sorted(os.listdir(base_dir))
            selected_series = st.selectbox("Selecione a Série:", series)

            methods_path = os.path.join(base_dir, selected_series)
            methods = sorted(os.listdir(methods_path))
            selected_method = st.selectbox("Selecione o Método:", methods)

            output_dir = os.path.join(methods_path, selected_method)

            st.subheader(f"Série: {selected_series} | Método: {selected_method}")

            # Caminho do .txt da série original
            serie_txt_path = os.path.join(base_dir, 'MixedBag', f'{selected_series}.txt')
            if os.path.exists(serie_txt_path):
                with open(serie_txt_path, 'r') as f:
                    line = f.readline()
                    serie = np.array([float(val) for val in line.strip().split(',')])
                st.success(f"Série original carregada de: {serie_txt_path}")
                st.write(f"Primeiros valores da série: {serie[:10]}")
            else:
                st.warning("Série original não encontrada no formato esperado (.txt em MixedBag).")
                serie = None

            # Mostrar imagens
            for img_name in ['regime_bar.png', 'dendrograma.png', 'silhouette.png']:
                img_path = os.path.join(output_dir, img_name)
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), caption=img_name, use_container_width=True)

            # Mostrar métricas
            metrics_path = os.path.join(output_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                st.subheader("Métricas")
                with open(metrics_path) as f:
                    metrics = json.load(f)
                st.json(metrics)

            # Carregar snippets
            snippets = None
            snippets_json = os.path.join(output_dir, 'snippets.json')
            if os.path.exists(snippets_json):
                with open(snippets_json) as f:
                    snippets = json.load(f)
            elif os.path.exists(os.path.join(output_dir, 'snippets.npy')):
                snippets = np.load(os.path.join(output_dir, 'snippets.npy'), allow_pickle=True)
                snippets = [{'index': int(s[0]), 'subsequence': s[1].tolist()} for s in snippets]


            if snippets:
                st.subheader("Visualização dos Snippets (Plotly)")

                max_snippets = len(snippets)
                num_snippets = st.slider(
                    "Número de snippets a visualizar:",
                    min_value=1,    
                    max_value=max_snippets,
                    value=min(5, max_snippets)
                )

                selected_snippets = snippets[:num_snippets]

                # === Gráfico 1: Shapes dos snippets ===
                fig_shapes = go.Figure()
                for i, snip in enumerate(selected_snippets):
                    fig_shapes.add_trace(go.Scatter(
                        y=snip['subsequence'],
                        mode='lines',
                        name=f'Snippet {i+1}'
                    ))
                fig_shapes.update_layout(
                    title="Shapes dos Snippets",
                    xaxis_title="Índice",
                    yaxis_title="Valor",
                    height=400
                )
                st.plotly_chart(fig_shapes, use_container_width=True)

                # === Gráfico 2: Snippets sobre a série original ===
                if serie is not None:
                    fig_series = go.Figure()

                    # Série completa em cinza
                    fig_series.add_trace(go.Scatter(
                        y=serie,
                        mode='lines',
                        name='Série Original',
                        line=dict(color='lightgray'),
                        opacity=0.7
                    ))

                    subseq_size = len(snippets[0]['subsequence'])

                    # Snippets sobrepostos
                    for i, snip in enumerate(selected_snippets):
                        idx = snip['index']
                        x_vals = list(range(idx, idx + subseq_size))
                        y_vals = serie[idx:idx + subseq_size]

                        fig_series.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='lines',
                            name=f'Snippet {i+1}',
                            line=dict(width=3)
                        ))

                    fig_series.update_layout(
                        title="Snippets na Série Temporal",
                        xaxis_title="Índice",
                        yaxis_title="Valor",
                        height=450
                    )
                    st.plotly_chart(fig_series, use_container_width=True)
                else:
                    st.warning("Série original não disponível para sobrepor os snippets.")