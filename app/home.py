import streamlit as st

st.image("logo.png", use_container_width=True)
st.markdown("<h1 style='text-align: center; color: white;'>RS4 - Ferramentas de Análise de Séries Temporais</h1>", unsafe_allow_html=True)

st.write("Bem-vindo! Selecione abaixo o serviço que deseja utilizar:")

st.page_link("pages/visualizador.py", label="Visualizador de Resultados", icon="📊")
st.page_link("pages/rs4-parametrizado.py", label="Executar RS4 Parametrizado", icon="⚙️")
