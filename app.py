import streamlit as st
import pandas as pd
import os
import joblib

from utils.processamento import mapear_valores
from utils.recomendacao import preparar_modelo, recomendar_material, texto_para_vetor

# Configuração
st.set_page_config(
    page_title="IA Matéria",
    layout="centered",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "materiais.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_model.joblib")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

# Estilo personalizado dark + roxo
st.markdown("""
    <style>

    body, .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    /* inputs */
    input, textarea, select {
        background-color: #111 !important;
        color: #fff !important;
        border: 1px solid #333 !important;
        border-radius: 6px !important;
    }

    /* botões */
    .stButton button {
        background-color: #7c3aed !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #9d4edd !important;
    }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        color: #fff;
    }

    </style>
""", unsafe_allow_html=True)


# Header

st.image("assets/logo.png", width=300)
st.write("Simulador inteligente para escolha e descoberta de materiais")
st.markdown("---")


# Carregar dados
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df_raw = load_data(DATA_PATH)
df_proc = mapear_valores(df_raw)

# Carregar ou treinar modelo
if os.path.exists(MODEL_PATH):
    modelo = joblib.load(MODEL_PATH)['modelo']
else:
    modelo = preparar_modelo(df_proc, n_neighbors=5)
    joblib.dump({'modelo': modelo}, MODEL_PATH)


# Sidebar
st.sidebar.header("Entrada do usuário")
modo = st.sidebar.selectbox("Modo de entrada", ["Descrição (texto livre)", "Entradas estruturadas"])

st.sidebar.markdown("---")
st.sidebar.write("Banco: **50 materiais embutidos**")
st.sidebar.write("Modelo: **KNN:K-Nearest Neighbors**")


# UI Principal
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if modo == "Descrição (texto livre)":
        st.subheader("Descreva o material desejado")
        texto = st.text_area(
            "Descrição",
            height=140,
            placeholder="Ex.: leve, barato, resistência alta, não biodegradável"
        )

        if st.button("Encontrar materiais semelhantes"):
            with st.spinner("Analisando descrição..."):
                vetor = texto_para_vetor(texto, df_proc)
                resultados = recomendar_material(modelo, df_raw, vetor, k=3)
                st.write("### Resultados")
                st.dataframe(resultados)

    else:
        st.subheader("Entradas estruturadas")
        cols = st.columns(2)

        with cols[0]:
            custo = st.slider("Custo (índice)", 0, 10)
            dens = st.number_input("Densidade (kg/m³)")
            tensile = st.number_input("Resistência à tração (MPa)")

        with cols[1]:
            cond = st.number_input("Condutividade térmica (W/m·K)")
            recicl = st.radio("Reciclável?", (1, 0))
            bio = st.radio("Biodegradável?", (0, 1))
            temp = st.number_input("Temperatura máx (°C)")

        if st.button("Buscar (estruturado)"):
            vetor = [float(custo), float(dens), float(tensile), float(cond), float(recicl), float(bio), float(temp)]
            resultados = recomendar_material(modelo, df_raw, vetor, k=3)
            st.dataframe(resultados)

    st.markdown('</div>', unsafe_allow_html=True)


# Banco completo
with st.expander("Mostrar banco de dados completo"):
    st.dataframe(df_raw)



