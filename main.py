import io
import time
from typing import cast

import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as T
from pandas.core.col import col
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def segment_image(arr, threshold, kernel_size):
    img = arr.copy()

    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255

    segmented = cv2.bitwise_and(img, img, mask=mask)
    return mask, segmented


def rotate(arr, angle):
    img = arr.copy()
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated


def load_model(name: str, classes_num: int, device: torch.device):
    """
    name: 'densenet' or 'inception'
    classes_num: 2 for binary classification, 4 for multi-class
    device: 'cpu' or 'cuda'
    """

    if name == "densenet":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        for param in model.parameters():
            param.requires_grad = False

        feateures_num = model.classifier.in_features
        model.classifier = torch.nn.Linear(feateures_num, classes_num)
        img_size = 224

    elif name == "inception":
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True
        )

        for param in model.parameters():
            param.requires_grad = False

        if model.AuxLogits is not None:
            aux_features = cast(torch.nn.Linear, model.AuxLogits.fc)
            model.AuxLogits.fc = torch.nn.Linear(aux_features.in_features, classes_num)

        features_num = model.fc.in_features
        model.fc = torch.nn.Linear(features_num, classes_num)
        img_size = 299

    else:
        raise ValueError(
            f"Rede '{name}' não reconhecida. Use 'densenet' ou 'inception'."
        )

    return model.to(device), img_size


def train_model(
    network_name: str,
    classes_num: int,
    train_entries: list,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    epoch_callback=None,
):
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    best_loss = float("inf")
    epochs_no_improve = 0
    PATIENCE = 8

    model, img_size = load_model(network_name, classes_num, device)
    model.train()

    train_dataset = MammoDataset(
        train_entries,
        use_augment=st.session_state.get("use_augmentation", True),
        classes_num=classes_num,
        img_size=img_size,
    )
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criteria = torch.nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss, hits, total = 0.0, 0, 0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if network_name == "inception":
                output, aux_output = model(imgs)
                loss = criteria(output, labels) + 0.4 * criteria(aux_output, labels)
            else:
                output = model(imgs)
                loss = criteria(output, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            hits += (output.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)

        avg_loss = total_loss / total

        acuracy = hits / total
        time_span = time.time() - t0

        entry = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "acuracy": round(acuracy, 4),
            "time_s": round(time_span, 1),
        }
        history.append(entry)

        if epoch_callback:
            epoch_callback({**entry, "loss": avg_loss, "stop": True})

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_{network_name}_nc{classes_num}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    return model, history


class MammoDataset(Dataset):
    def __init__(self, entries, use_augment=False, img_size=224, classes_num=4):
        """
        entries: dicts list with keys 'path' and 'class'
                 generated by the dataset laoder
        use_augment: if True, expands each image in 5 rotations
        img_size: network entry size (224 for DenseNet and Inception)
        """
        self.img_size = img_size
        if classes_num == 2:
            self.class_id = {
                "BIRADS I": 0,
                "BIRADS II": 0,
                "BIRADS III": 1,
                "BIRADS IV": 1,
            }
        else:
            self.class_id = {
                "BIRADS I": 0,
                "BIRADS II": 1,
                "BIRADS III": 2,
                "BIRADS IV": 3,
            }

        angles = [-20, -10, 0, 10, 20] if use_augment else [0]
        self.samples = [
            {**entry, "angle": angle} for entry in entries for angle in angles
        ]

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        arr = np.array(Image.open(sample["path"]).convert("L"))

        params = st.session_state.get(
            "segmentation_params", {"threshold": 18, "kernel_size": 7}
        )
        _, arr = segment_image(arr, params["threshold"], params["kernel_size"])

        if sample["angle"] != 0:
            arr = rotate(arr, sample["angle"])

        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

        arr = cv2.resize(
            arr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )

        arr_rgb = np.stack([arr] * 3, axis=-1)

        tensor = self.transform(arr_rgb)

        label = self.class_id[sample["class"]]

        return tensor, label


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

    if "train" not in st.session_state["dataset"]:
        st.warning("Carregue o dataset primeiro.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        network_name = st.selectbox("Rede", ["densenet", "inception"])
        classes_num = st.selectbox(
            "Tarefa",
            [("Binária (I+II x III+IV)", 2), ("4 classes (IxIIxIIIxIV)", 4)],
            format_func=lambda x: x[0],
        )[1]
    with col2:
        epochs = st.number_input("Épocas", value=30, min_value=1)
        batch_size = st.number_input("Tamanho do btach", value=8, min_value=1)
        lr = st.number_input("Taxa de aprendizado", value=1e-4, format="%.5f")

    if st.button("Iniciar treino"):
        if "train" not in st.session_state["dataset"]:
            st.error("Carregue o dataset primeiro.")
        else:
            log = st.empty()
            progress_bar = st.progress(0)
            graph = st.empty()

            ui_history = []

            def update(entry):
                ui_history.append(entry)

                log.text(
                    "\n".join(
                        f"Época {e['epoch']:3d} | "
                        f"loss: {e['loss']:.4f} | "
                        f"acc: {e['acuracy']:.2%} | "
                        f"{e['time_s']}s"
                        for e in ui_history[-10:]
                    )
                )

                progress_bar.progress(entry["epoch"] / epochs)

                import pandas as pd

                df = pd.DataFrame(ui_history)
                graph.line_chart(df.set_index("epoch")[["loss", "acuracy"]])

            model, history = train_model(
                network_name=network_name,
                classes_num=classes_num,
                train_entries=st.session_state["dataset"]["train"],
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                epoch_callback=update,
            )

            key = f"{network_name}_nc{classes_num}"
            path = f"{key}_weights.pth"
            torch.save(model.state_dict(), path)

            st.session_state[f"model_{key}"] = model
            st.session_state[f"path_{path}"] = path
            st.success(f"Treino concluído. Pesos salvos em `{path}`")

    st.divider()
    st.subheader("Carregar pesos salvos")

    col3, col4 = st.columns(2)
    with col3:
        load_network = st.selectbox("Rede", ["densenet", "inception"], key="rl")
        load_classes_num = st.selectbox(
            "Tarefa",
            [2, 4],
            key="ncl",
            format_func=lambda x: "Binária" if x == 2 else "4 classes",
        )
    with col4:
        weights_file = st.file_uploader("Arquivo .pth", type=["pth"])

    if st.button("Carregar") and weights_file:
        device = torch.device("cuda")
        model, _ = load_model(load_network, load_classes_num, device)
        state = torch.load(io.BytesIO(weights_file.read()), map_location=device)
        model.load_state_dict(state)
        model.eval()

        key = f"{load_network}_nc{load_classes_num}"
        st.session_state[f"model_{key}"] = model
        st.session_state[f"path_{key}"] = weights_file.name
        st.success(f"Pesos carregados: {weights_file.name}")


elif page == "Classificação binária":
    st.header("Classificação binária")
    st.info("WIP")

elif page == "Classificação 4 classes":
    st.header("Classificação 4 classes")
    st.info("WIP")

elif page == "Grad-CAM":
    st.header("Grad-CAM")
    st.info("WIP")
