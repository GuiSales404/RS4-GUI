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

            selected_methods = st.multiselect("Selecione os Métodos a Comparar:", methods)

            # Exibe subheader com a série escolhida
            st.subheader(f"Série: {selected_series}")
            start_app = st.button("Iniciar")
            # Se pelo menos um método for selecionado
            if selected_methods and start_app:

                # PREPARA GRÁFICO SHAPES DOS SNIPPETS
                fig_shapes = go.Figure()

                # PREPARA GRÁFICO SNIPPETS NA SÉRIE
                fig_series = go.Figure()

                # Adiciona a série completa em cinza (se disponível)
                if serie is not None:
                    fig_series.add_trace(go.Scatter(
                        y=serie,
                        mode='lines',
                        name='Série Original',
                        line=dict(color='lightgray'),
                        opacity=0.7
                    ))

                # PREPARA MÉTRICAS COMPARATIVAS
                metrics_list = []

                # LOOP NOS MÉTODOS SELECIONADOS
                for method in selected_methods:
                    output_dir = os.path.join(methods_path, method)
                    
                    st.markdown(f"### Método: {method}")

                    # --- MOSTRAR IMAGENS ---
                    for img_name in ['regime_bar.png', 'dendrograma.png', 'silhouette.png']:
                        img_path = os.path.join(output_dir, img_name)
                        if os.path.exists(img_path):
                            st.image(Image.open(img_path), caption=f"{method} - {img_name}", use_container_width=True)

                    # --- CARREGA SNIPPETS ---
                    snippets = None
                    snippets_json = os.path.join(output_dir, 'snippets.json')
                    if os.path.exists(snippets_json):
                        with open(snippets_json) as f:
                            snippets = json.load(f)
                    elif os.path.exists(os.path.join(output_dir, 'snippets.npy')):
                        snippets = np.load(os.path.join(output_dir, 'snippets.npy'), allow_pickle=True)
                        snippets = [{'index': int(s[0]), 'subsequence': s[1].tolist()} for s in snippets]

                    # --- PLOTA SNIPPETS SHAPES ---
                    if snippets:
                        for i, snip in enumerate(snippets[:3]):  # Mostra até 3 por método (ou ajuste como quiser)
                            fig_shapes.add_trace(go.Scatter(
                                y=snip['subsequence'],
                                mode='lines',
                                name=f'{method} - Snippet {i+1}'
                            ))

                        # --- PLOTA SNIPPETS NA SÉRIE ---
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

                    # --- CARREGA E ARMAZENA MÉTRICAS ---
                    metrics_path = os.path.join(output_dir, 'metrics.json')
                    if os.path.exists(metrics_path):
                        with open(metrics_path) as f:
                            metrics = json.load(f)

                        # Flatten se necessário
                        flat_metrics = {}
                        for k, v in metrics.items():
                            if isinstance(v, dict):
                                for sub_k, sub_v in v.items():
                                    flat_metrics[f"{k}.{sub_k}"] = sub_v
                            else:
                                flat_metrics[k] = v

                        # Adiciona coluna "Método"
                        for metric_name, metric_value in flat_metrics.items():
                            metrics_list.append({
                                "Método": method,
                                "Métrica": metric_name,
                                "Valor": metric_value
                            })

                # --- EXIBE OS GRÁFICOS ---
                st.subheader("Shapes dos Snippets Comparados")
                st.plotly_chart(fig_shapes, use_container_width=True)

                st.subheader("Snippets sobrepostos na Série Temporal")
                st.plotly_chart(fig_series, use_container_width=True)

                # --- EXIBE TABELA DE MÉTRICAS COMPARATIVAS ---
                if metrics_list:
                    st.subheader("Métricas Comparativas")
                    metrics_df = pd.DataFrame(metrics_list)
                    st.dataframe(metrics_df)
