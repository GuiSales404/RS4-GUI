import streamlit as st
import os
import json
import numpy as np
import pandas as pd

st.image("logo.png", use_container_width=True)
st.markdown("<h1 style='text-align: center; color: white;'>RS4 - Clustering Parametrizável</h1>", unsafe_allow_html=True)

# --- Interface ---

# Subseq size
subseq_size = st.number_input("Tamanho da subsequência (subseq_size):", min_value=10, max_value=1000, value=150)

# K range
k_min = st.number_input("Valor mínimo de k:", min_value=2, max_value=100, value=2)
k_max = st.number_input("Valor máximo de k:", min_value=2, max_value=100, value=25)
k_range = (k_min, k_max)

# Métodos disponíveis
available_methods = ['agglomerative', 'hierarchical', 'hdbscan', 'minibatchkmeans', 'kshape']
selected_methods = st.multiselect("Selecione os métodos de clustering:", available_methods)

# Parâmetros adicionais (aparecem dinamicamente!)
method_params = {}

if 'agglomerative' in selected_methods:
    linkage_agglomerative = st.selectbox("Agglomerative - linkage:", ['ward', 'complete', 'average', 'single'])
    method_params['agglomerative'] = {'linkage': linkage_agglomerative}

if 'hierarchical' in selected_methods:
    linkage_hierarchical = st.selectbox("Hierarchical - linkage:", ['ward', 'complete', 'average', 'single'])
    method_params['hierarchical'] = {'linkage': linkage_hierarchical}

if 'hdbscan' in selected_methods:
    min_cluster_size = st.number_input("HDBSCAN - min_cluster_size:", min_value=2, max_value=100, value=5)
    method_params['hdbscan'] = {'min_cluster_size': min_cluster_size}

if 'minibatchkmeans' in selected_methods:
    batch_size = st.number_input("MiniBatchKMeans - batch_size:", min_value=10, max_value=500, value=50)
    method_params['minibatchkmeans'] = {'batch_size': batch_size}

if 'kshape' in selected_methods:
    st.info("KShape não possui parâmetros adicionais.")

# Simulação de MixedBag
base_path = 'time_series'
dataset = {}

for file in os.listdir(base_path):
    with open(os.path.join(base_path, file), 'r') as f:
        lines = f.read()
    cleaned = lines.replace('\n', ',').replace(' ', '')
    parts = [x for x in cleaned.split(',') if x]
    dataset[file] = [float(x) for x in parts]

# Seleção da(s) série(s)
series_available = list(dataset.keys())
selected_series = st.multiselect("Selecione as séries temporais a processar:", series_available)

# Rodar
if st.button("Rodar Clusterização"):
    st.write("Rodando com os seguintes parâmetros:")
    st.json({
        "subseq_size": subseq_size,
        "k_range": k_range,
        "selected_methods": selected_methods,
        "method_params": method_params,
        "selected_series": selected_series
    })

    # Aqui você chamaria sua função adaptada, ex:
    # run_selected_clusterings(selected_series, dataset, subseq_size, k_range, selected_methods, method_params)

    st.success("Execução finalizada! Resultados salvos na pasta ./resultados")
