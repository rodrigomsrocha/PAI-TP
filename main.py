import io

import cv2
import numpy as np
import streamlit as st
from pandas.core.col import col
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

    diretory = st.text_input("Caminho do diretório raiz", placeholder="/dados/rmlo/")
    col1, col2 = st.columns(2)

    with col1:
        mammary = st.selectbox("Mama", ["right", "left"])
    with col2:
        orientation = st.selectbox("Orientação", ["CC", "MLO"])

    PREFIX = {"D": "BIRADS I", "E": "BIRADS II", "F": "BIRADS III", "G": "BIRADS IV"}

    if st.button("Carregar diretório"):
        import glob
        import os

        files = glob.glob(os.path.join(diretory, "**", "*.png"), recursive=True)

        test, train = [], []
        for path in files:
            filename = os.path.basename(path)
            prefix = filename[0].upper()

            if prefix not in PREFIX:
                st.warning(f"Arquivo ignorado (prefixo desconhecido): {filename}")
                continue

            digits = "".join(filter(str.isdigit, filename))
            number = int(digits) if digits else 0

            entry = {
                "path": path,
                "class": PREFIX[prefix],
                "name": filename,
            }

            if number % 4 == 0:
                test.append(entry)
            else:
                train.append(entry)

        st.session_state["dataset"] = {"train": train, "test": test}

        col1, col2, col3 = st.columns(3)
        col1.metric("Imagens de treino", len(train))
        col2.metric("Imagens de teste (múlt. 4)", len(test))
        col3.metric("Total", len(train) + len(test))

        import pandas as pd

        df = pd.DataFrame(train + test)
        df["split"] = ["treino"] * len(train) + ["teste"] * len(test)
        st.dataframe(df[["name", "class", "split"]], use_container_width=True)

elif page == "Segmentação":
    st.header("Segmentação")

    if "image_arr" not in st.session_state:
        st.warning("Abra uma imagem primeiro na aba 'Visualizar imagem'.")
        st.stop()

    arr = st.session_state["image_arr"]

    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Threshold", 0, 500, 20)
    with col2:
        kernel_size = st.selectbox("Kernel morfológico", [5, 7, 11, 15])

    def segment_image(arr, threshold, kernel_size):
        img = arr.copy()

        _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255

        segmented = cv2.bitwise_and(img, img, mask=mask)
        return mask, segmented

    mask, segmented = segment_image(arr, threshold, kernel_size)

    col1, col2, col3 = st.columns(3)
    col1.image(arr, caption="Original", clamp=True, width="content")
    col2.image(mask, caption="Máscara", clamp=True, width="content")
    col3.image(segmented, caption="Segmentada", clamp=True, width="content")

    if st.button("Aplicar ao dataset inteiro"):
        st.session_state["segmentation_params"] = {
            "threshold": threshold,
            "kernel_size": kernel_size,
        }
        st.success(
            "Parâmetros salvos. O dataset será processado durante o treinamento."
        )

    st.session_state["segmented_arr"] = segmented

elif page == "Aumento de dados":
    st.header("Aumento de dados")
    st.caption("Rotações de −20° a +20° em intervalos de 10° — 5 variações por imagem")

    if "segmented_arr" not in st.session_state:
        st.warning("Segmente uma imagem primeiro.")
        st.stop()

    arr = st.session_state["segmented_arr"]

    def rotate(arr, angle):
        img = arr.copy()
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        return rotated

    angles = [-20, -10, 0, 10, 20]
    cols = st.columns(len(angles))
    for col, angle in zip(cols, angles):
        rotated = rotate(arr, angle)
        col.image(rotated, caption=f"{angle:+d}°", clamp=True, width="content")

    if st.button("Gerar aumentos para todo o dataset de treino"):
        if "dataset" not in st.session_state:
            st.error("Carregue o dataset primeiro.")
        else:
            st.info(
                f"Serão geradas {len(st.session_state['dataset']['train']) * 5} imagens no total."
            )
            st.session_state["use_augmentation"] = True
            st.success(
                "Configurado. O aumento será aplicado durante o carregamento para treino."
            )

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
