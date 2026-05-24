import streamlit as st

st.set_page_config(page_title="PAI TP", page_icon=":robot_face:", layout="wide")

PAGES = [
    "Visualizar imagem",
    "Carregar dataset",
    "Segmentação",
    "Aumento de dados",
    "Treinar modelo",
    "Classificação binária",
    "Classificação 4 classes",
    "Grad-CAM",
]

page = st.sidebar.selectbox("Navegação", PAGES)

if page == "Visualizar imagem":
    st.header("Visualizar imagem")
    st.info("WIP")
elif page == "Carregar dataset":
    st.header("Carregar dataset")
    st.info("WIP")
elif page == "Segmentação":
    st.header("Segmentação")
    st.info("WIP")
elif page == "Aumento de dados":
    st.header("Aumento de dados")
    st.info("WIP")
elif page == "Treinar modelo":
    st.header("Treinar modelo")
    st.info("WIP")
elif page == "Classificação binária":
    st.header("Classificação binária")
    st.info("WIP")
elif page == "Classificação 4 classes":
    st.header("Classificação 4 classes")
    st.info("WIP")
elif page == "Grad-CAM":
    st.header("Grad-CAM")
    st.info("WIP")
