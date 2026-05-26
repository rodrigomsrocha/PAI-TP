import io

import cv2
import numpy as np
import streamlit as st
import tifffile
from PIL import Image

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
    file = st.file_uploader("Abrir imagem", type=["png", "tif", "tiff"])

    if file:
        raw = file.read()
        arr = np.array(Image.open(io.BytesIO(raw)))

        st.caption(
            f"Resolução: {arr.shape[1]}×{arr.shape[0]} px — dtype: {arr.dtype} — range: {arr.min()}–{arr.max()}"
        )

        hist, bins = np.histogram(arr[arr > 0], bins=255)
        st.bar_chart(hist)

        col1, col2 = st.columns(2)
        with col1:
            zoom = st.slider("Zoom (%)", 10, 400, 100, step=10)
        with col2:
            window_mode = st.checkbox("Janelamento manual W/L")

        if window_mode:
            col3, col4 = st.columns(2)
            with col3:
                W = st.slider("W (largura)", 1, 255, int(arr.max()), step=1)
            with col4:
                L = st.slider("L (nível)", 0, 255, int(arr.max() // 2), step=1)

            low = L - W // 2
            high = L + W // 2

            arr_display = np.clip(arr.astype(np.float32), low, high)
            arr_display = ((arr_display - low) / (high - low) * 255).astype(np.uint8)
        else:
            lo, hi = arr.min(), arr.max()
            arr_display = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(
                np.uint8
            )

        h, w = arr_display.shape[:2]
        new_w = int(w * zoom / 100)
        new_h = int(h * zoom / 100)
        arr_zoom = cv2.resize(
            arr_display, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

        st.image(arr_zoom, caption=file.name, clamp=True, width="content")

        st.session_state["image_arr"] = arr
        st.session_state["image_name"] = file.name

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
