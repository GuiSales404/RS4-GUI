import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from statistics import mean, median, stdev, mode, StatisticsError

def join_images_side_by_side(img1, img2):
    # Garante que ambas têm a mesma altura
    height = max(img1.height, img2.height)
    new_img = Image.new('RGB', (img1.width + img2.width, height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img

# Carregar o arquivo JSON
@st.cache_data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Função para carregar CSV
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Suponha que esta função já existe e retorna um numpy array com os valores do snippet
def load_snippet(path):
    return np.load(path)

def plot_csv_optimized(file_path, snippet, analysis, tolerance):
    df = load_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['acc_norm'] = np.sqrt(df['ACC_X(m/s^2)']**2 + df['ACC_Y']**2 + df['ACC_Z']**2)

    y_data = df['acc_norm'] if analysis == "motion" else df['HEART_RATE(bpm)']
    snippet_len = len(snippet)
    total_windows = len(df) - snippet_len + 1
    match_intervals = []

    if snippet_len > 0 and total_windows > 0:
        y_values = y_data.values

        for i in range(total_windows):
            window = y_values[i:i + snippet_len]
            if np.all(np.abs(window - snippet) <= tolerance):
                start_date = df['date'].iloc[i]
                end_date = df['date'].iloc[i + snippet_len - 1]
                match_intervals.append((start_date, end_date))

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Heart Rate (bpm)', color='tab:red')
    ax1.plot(df['date'], df['HEART_RATE(bpm)'], color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Norma da Aceleração', color='tab:blue')
    ax2.plot(df['date'], df['acc_norm'], color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Adiciona marcações no final, de uma vez
    for start, end in match_intervals:
        ax1.axvspan(start, end, color='yellow', alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)

    coverage_percent = (len(match_intervals) / total_windows) * 100 if total_windows > 0 else 0
    return coverage_percent


def plot_csv(file_path, snippet, analysis, tolerance):
    df = load_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['acc_norm'] = np.sqrt(df['ACC_X(m/s^2)']**2 + df['ACC_Y']**2 + df['ACC_Z']**2)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Heart Rate (bpm)', color='tab:red')
    ax1.plot(df['date'], df['HEART_RATE(bpm)'], color='tab:red', label='Heart Rate')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Norma da Aceleração', color='tab:blue')
    ax2.plot(df['date'], df['acc_norm'], color='tab:blue', label='Aceleração')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    match_count = 0
    total_windows = len(df) - len(snippet) + 1

    if len(snippet) > 0:
        y_data = df['acc_norm'] if analysis == "motion" else df['HEART_RATE(bpm)']
        
        for i in range(total_windows):
            if np.all(np.abs(y_data.iloc[i:i+len(snippet)].values - snippet) <= tolerance):
                ax1.axvspan(df['date'].iloc[i], df['date'].iloc[i+len(snippet)-1], color='yellow', alpha=0.3)
                match_count += 1

    fig.tight_layout()
    st.pyplot(fig)

    coverage_percent = (match_count / total_windows) * 100 if total_windows > 0 else 0
    return coverage_percent


def show_json_statistics(selected_data):
    st.header("Estatísticas do Arquivo Selecionado")
    for metric_key, analyses in selected_data.items():
        st.subheader(f"Métrica: {metric_key}")
        for analysis_key, weights in analyses.items():
            st.markdown(f"**Análise: {analysis_key}**")
            for weight_key, value in weights.items():
                st.markdown(f"- Ponderação: {weight_key}")
                values = value if isinstance(value, list) else [value]
                try:
                    stats = {
                        'Média': mean(values),
                        'Mediana': median(values),
                        'Desvio Padrão': stdev(values) if len(values) > 1 else 0
                    }
                except StatisticsError:
                    stats = {
                        'Média': mean(values),
                        'Mediana': median(values),
                        'Desvio Padrão': stdev(values) if len(values) > 1 else 0,
                    }
                st.write(stats)

# Interface Streamlit
def main():
    st.title("Avaliar Estatísticas de Snippets")

    # Caminho base para os arquivos CSV
    csv_base_path = "/home/guilherme-sales/insight_samsung/snippets/streamlit/files"  
    eval_methods_path = "/home/guilherme-sales/insight_samsung/snippets/new_eval_methods/results"
    # Selecionar ponderação
    weight_option = st.selectbox("Selecione a ponderação", [x.replace('euclidean_', '') for x in os.listdir(csv_base_path)])
    weight_option = 'euclidean_' + weight_option   
    # Selecionar a chave do arquivo CSV
    csv_keys = os.listdir(os.path.join(csv_base_path, weight_option))
    selected_csv = st.selectbox("Selecione o arquivo CSV", csv_keys)
    
    cut_keys = os.listdir(os.path.join(csv_base_path, weight_option, selected_csv))
    selected_cut = st.selectbox("Selecione o recorte", cut_keys)

    # Selecionar o tipo de análise
    analysis = st.selectbox("Selecione a análise", ["bpm", "motion"])

    # Nível de aceitação
    tolerance = st.number_input("Selecione o nível de aceitação", value=5)

    # Escolher snippet
    snippet_keys = [x for x in os.listdir(os.path.join(csv_base_path, weight_option, selected_csv, selected_cut, analysis)) if x.split('.')[1] == 'npy']
    selected_snippet = st.selectbox("Selecione o Snippet", snippet_keys)
    
    final_snippet_path = os.path.join(csv_base_path, weight_option, selected_csv, selected_cut, analysis, selected_snippet)

    # Carregar snippet
    snippet = load_snippet(final_snippet_path)

    gt_base_path = '/home/guilherme-sales/insight_samsung/snippets/streamlit/collect_gt'
    
    # Carregar snippet alternativo (para comparação)
    alt_snippet_path = os.path.join(csv_base_path, 'stumpy', selected_csv, selected_cut, analysis, selected_snippet)
    alt_snippet = load_snippet(alt_snippet_path) if os.path.exists(alt_snippet_path) else None

    # Atualizar snippet caso o toggle esteja ativado
    snippet_finder = st.toggle("Snippet Finder")
    if snippet_finder and alt_snippet is not None:
        snippet = alt_snippet
    
    csv_path = os.path.join(gt_base_path, selected_csv, 'time_cuts', selected_cut)
    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False
    st.subheader("Comparação dos Snippets")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Snippet ERB:**")
        st.write(load_snippet(final_snippet_path))

    with col2:
        st.markdown("**Snippet Stumpy (SnippetFinder):**")
        if alt_snippet is not None:
            st.write(alt_snippet)
        else:
            st.warning("Snippet alternativo não encontrado.")

    image_erb = Image.open(os.path.join(csv_base_path, weight_option, selected_csv, selected_cut, analysis, 'snippets_plot.png'))
    image_sf = Image.open(os.path.join(csv_base_path, 'stumpy', selected_csv, selected_cut, analysis, 'snippets_plot.png'))
        
    combined_image = join_images_side_by_side(image_erb, image_sf)
        
    st.image(combined_image, caption='Snippets ERB | Snippet Stumpy', use_container_width=True)
    
    images_eval_serie = os.path.join(eval_methods_path, weight_option, selected_csv, selected_cut, analysis, 'series_with_snippets.png')
    images_eval_tilled_plot = os.path.join(eval_methods_path, weight_option, selected_csv, selected_cut, analysis, 'snippet_tiled_with_error_basic_errorplot.png')
    images_eval_tilled_main = os.path.join(eval_methods_path, weight_option, selected_csv, selected_cut, analysis, 'snippet_tiled_with_error_basic_main.png')
    
    st.image(images_eval_serie, use_container_width=True)
    st.image(images_eval_tilled_plot, use_container_width=True)
    st.image(images_eval_tilled_main, use_container_width=True)
    
    if st.button("Calcular Cobertura"):
        st.session_state.run_analysis = True

    if st.session_state.run_analysis:
        try:
            coverage_percent = plot_csv(csv_path, snippet, analysis, tolerance)
            # coverage_percent = 0
            st.success(f"Cobertura do snippet: {coverage_percent:.2f}%")
        except Exception as e:
            st.error(f"Erro ao carregar o CSV: {e}")


if __name__ == "__main__":
    main()
