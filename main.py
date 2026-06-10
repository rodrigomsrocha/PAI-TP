# Nome                             Matricula       Curso                   Campus
# João Victor Martins dos Anjos    824604          Ciência da Computação   Lourdes
# Rodrigo Marques Rocha            826583          Ciência da Computação   Lourdes
# Rafael Coelho                    769774          Ciência da Computação   Lourdes

import io
import time
from typing import cast
import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
)
from torch.utils.data import DataLoader, Dataset


def apply_clahe(arr, clip_limit=2.0, tile_size=(8, 8)):
    if arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(arr)


def segment_and_crop_image(arr, kernel_size):
    img = arr.copy()

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255

    erosion_kernel = np.ones((15, 15), np.uint8)
    mask = cv2.erode(mask, erosion_kernel, iterations=1)

    segmented = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(mask)
    
    if w < 50 or h < 50:
        return mask, segmented
        
    cropped_mask = mask[y:y+h, x:x+w]
    cropped_segmented = segmented[y:y+h, x:x+w]

    return cropped_mask, cropped_segmented


def rotate(arr, angle):
    img = arr.copy()
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated


def load_model(name: str, classes_num: int, device: torch.device):
    if name == "densenet":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.features.denseblock3.parameters():
            param.requires_grad = True
            
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True

        feateures_num = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(feateures_num, classes_num)
        )
        img_size = 224

    elif name == "inception":
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.Mixed_7b.parameters():
            param.requires_grad = True
        for param in model.Mixed_7c.parameters():
            param.requires_grad = True

        if model.AuxLogits is not None:
            aux_features = cast(torch.nn.Linear, model.AuxLogits.fc)
            model.AuxLogits.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(aux_features.in_features, classes_num)
            )

        features_num = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(features_num, classes_num)
        )
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,      
        pin_memory=True,    
        prefetch_factor=2       
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

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


def evalute_model(model, test_entries, classes_num, network_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 299 if network_name == "inception" else 224
    model.eval()

    test_dataset = MammoDataset(
        test_entries,
        use_augment=False,
        classes_num=classes_num,
        img_size=img_size,
    )

    loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    all_labels = []
    all_preds = []

    t0 = time.time()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)

            if network_name == "inception":
                model.eval()

            output = model(imgs)
            preds = output.argmax(dim=1).cpu().tolist()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds)

    time_span = time.time() - t0

    return all_labels, all_preds, time_span


class MammoDataset(Dataset):
    def __init__(self, entries, use_augment=False, img_size=224, classes_num=4):
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

        if use_augment:
            self.transform = T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomResizedCrop(img_size, scale=(0.7, 0.9), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((img_size, img_size), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        arr = np.array(Image.open(sample["path"]).convert("L"))

        params = st.session_state.get(
            "segmentation_params", {"clip_limit": 2.0, "kernel_size": 7}
        )
        
        arr = apply_clahe(arr, clip_limit=params.get("clip_limit", 2.0))
        
        _, arr = segment_and_crop_image(arr, params["kernel_size"])

        if sample["angle"] != 0:
            arr = rotate(arr, sample["angle"])

        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

        arr_rgb = np.stack([arr] * 3, axis=-1)
        tensor = self.transform(arr_rgb)
        label = self.class_id[sample["class"]]

        return tensor, label

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = {}
        self.gradients = {}

        for param in target_layer.parameters():
            param.requires_grad = True

        target_layer.register_forward_hook(
            lambda module, input, output: self.activations.update({"feat": output})
        )

    def generate(self, tensor, class_id=None):
        self.model.eval()
        tensor = tensor.unsqueeze(0)

        output = self.model(tensor)
        if isinstance(output, tuple):
            output = output[0]

        id = class_id if class_id is not None else output.argmax(dim=1).item()
        prob = torch.nn.functional.softmax(output, dim=1)[0, id].item()

        self.model.zero_grad()
        output[0, id].backward(retain_graph=True)

        gradients = torch.autograd.grad(
            outputs=output[0, id],
            inputs=self.activations["feat"],
            retain_graph=True,
            allow_unused=True,
        )[0]

        if gradients is None:
            st.error("Gradiente não calculado — verifique a camada alvo.")
            return None, id, prob

        weights = gradients.mean(dim=[2, 3], keepdim=True)

        cam = (weights * self.activations["feat"]).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)

        cam = cam.squeeze().cpu().detach().numpy()
        min_cam, max_cam = cam.min(), cam.max()
        if max_cam > min_cam:
            cam = (cam - min_cam) / (max_cam - min_cam)
        else:
            cam = np.zeros_like(cam)

        return cam, id, prob


def get_target_layer(model, network_name):
    if network_name == "densenet":
        return model.features.denseblock4.denselayer16.conv2
    elif network_name == "inception":
        return model.Mixed_7c.branch_pool.conv


def apply_gradcam(normalized_cam, uint8_img, size):
    cam_resized = cv2.resize(normalized_cam, (size, size))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    rgb_img = np.stack([uint8_img] * 3, axis=-1)

    overlay = cv2.addWeighted(rgb_img, 0.5, heatmap_rgb, 0.5, 0)
    return heatmap_rgb, overlay


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
    st.header("Segmentação Avançada")

    if "image_arr" not in st.session_state:
        st.warning("Abra uma imagem primeiro na aba 'Visualizar imagem'.")
        st.stop()

    arr = st.session_state["image_arr"]

    col1, col2 = st.columns(2)
    with col1:
        clip_limit = st.slider("CLAHE Clip Limit (Contraste)", 0.1, 10.0, 2.0, step=0.1)
    with col2:
        kernel_size = st.selectbox("Kernel morfológico", [5, 7, 11, 15], index=1)

    arr_clahe = apply_clahe(arr, clip_limit=clip_limit)
    mask, cropped_segmented = segment_and_crop_image(arr_clahe, kernel_size)

    col1, col2, col3 = st.columns(3)
    col1.image(arr, caption="Original", clamp=True, width="content")
    col2.image(mask, caption="Máscara (Otsu + Erosão + Crop)", clamp=True, width="content")
    col3.image(cropped_segmented, caption="Realçada, Segmentada e Cortada", clamp=True, width="content")

    if st.button("Aplicar ao dataset inteiro"):
        st.session_state["segmentation_params"] = {
            "clip_limit": clip_limit,
            "kernel_size": kernel_size,
        }
        st.success(
            "Parâmetros salvos! O novo pipeline (CLAHE + Otsu + Erosão + Crop) será executado dinamicamente no Dataset."
        )

    st.session_state["segmented_arr"] = cropped_segmented

elif page == "Aumento de dados":
    st.header("Aumento de dados")
    st.caption("Rotações de −20° a +20° em intervalos de 10° — 5 variações por imagem")

    if "segmented_arr" not in st.session_state:
        st.warning("Acesse a aba 'Segmentação' primeiro para processar a imagem de exemplo.")
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
                f"Serão geradas {len(st.session_state['dataset']['train']) * 5} variações de imagens durante o loop de treino."
            )
            st.session_state["use_augmentation"] = True
            st.success(
                "Configurado. O aumento (Rotação + Random Crop) será aplicado de forma transparente durante o treino."
            )

elif page == "Treinar modelo":
    st.header("Treinar modelo")

    if "dataset" not in st.session_state or "train" not in st.session_state["dataset"]:
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    st.caption("Benigno (BIRADS I+II) vs Maligno (BIRADS III+IV)")

    if "dataset" not in st.session_state or "test" not in st.session_state["dataset"]:
        st.warning("Carregue o dataset primeiro.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        network_name = st.selectbox("Rede", ["densenet", "inception"])

    key = f"{network_name}_nc2"
    is_model_available = f"model_{key}" in st.session_state

    if not is_model_available:
        st.warning("Treine ou carregue os pesos para classificação binária primeiro.")
        st.stop()

    if st.button("Classificar conjunto de teste"):
        modelo = st.session_state[f"model_{key}"]
        labels, preds, time_span = evalute_model(
            modelo, st.session_state["dataset"]["test"], 2, network_name
        )

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        f1 = f1_score(labels, preds)

        conf_matrix = confusion_matrix(labels, preds)
        tn, fp, fn, tp = conf_matrix.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        st.metric("Tempo", f"{time_span:.1f}s")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sensibilidade", f"{sensitivity:.2%}")
        col2.metric("Especificidade", f"{specificity:.2%}")
        col3.metric("Precisão", f"{prec:.2%}")
        col4.metric("Acurácia", f"{acc:.2%}")
        col5.metric("F1-score", f"{f1:.4f}")

        import pandas as pd

        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=["Real Benigno", "Real Maligno"],
            columns=["Predito Benigno", "Predito Maligno"],
        )
        st.subheader("Matriz de confusão")
        st.dataframe(conf_matrix_df)

        st.session_state[f"{network_name}_bin_results"] = {
            "sens": sensitivity,
            "spec": specificity,
            "prec": prec,
            "acc": acc,
            "f1": f1,
            "time_s": time_span,
            "conf_matrix": conf_matrix,
        }

elif page == "Classificação 4 classes":
    st.header("Classificação 4 classes")
    st.caption("BIRADS I vs II vs III vs IV")

    if "dataset" not in st.session_state or "test" not in st.session_state["dataset"]:
        st.warning("Carregue o dataset primeiro.")
        st.stop()

    network_name = st.selectbox("Rede", ["densenet", "inception"])
    key = f"{network_name}_nc4"
    is_model_available = f"model_{key}" in st.session_state

    if not is_model_available:
        st.warning("Treine ou carregue os pesos para classificação 4 classes primeiro.")
        st.stop()

    if st.button("Classificar conjunto de teste"):
        model = st.session_state[f"model_{key}"]
        labels, preds, time_span = evalute_model(
            model, st.session_state["dataset"]["test"], 4, network_name
        )

        classes_names = ["BIRADS I", "BIRADS II", "BIRADS III", "BIRADS IV"]
        conf_matrix = confusion_matrix(labels, preds)
        acc = accuracy_score(labels, preds)

        sens_per_class, spec_per_class = [], []
        for i in range(4):
            tp = conf_matrix[i, i]
            fn = conf_matrix[i, :].sum() - tp
            fp = conf_matrix[:, i].sum() - tp
            tn = conf_matrix.sum() - (tp + fn + fp)

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            sens_per_class.append(sens)
            spec_per_class.append(spec)

        avg_sens = sum(sens_per_class) / len(sens_per_class)
        avg_spec = sum(spec_per_class) / len(spec_per_class)

        st.metric("Tempo", f"{time_span:.1f}s")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sensibilidade média", f"{avg_sens:.2%}")
        col2.metric("Especificidade média", f"{avg_spec:.2%}")
        col3.metric("Acurácia", f"{acc:.2%}")

        import pandas as pd

        conf_matrix_df = pd.DataFrame(
            conf_matrix, index=classes_names, columns=classes_names
        )
        st.subheader("Matriz de confusão")
        st.dataframe(conf_matrix_df)

        st.subheader("Sensibilidade por classe")
        for name, sens in zip(classes_names, sens_per_class):
            st.write(f"{name}: {sens:.2%}")

        st.subheader("Especificidade por classe")
        for name, spec in zip(classes_names, spec_per_class):
            st.write(f"{name}: {spec:.2%}")

        st.session_state[f"{network_name}_4c_results"] = {
            "avg_sens": avg_sens,
            "avg_spec": avg_spec,
            "acc": acc,
            "time_s": time_span,
            "conf_matrix": conf_matrix,
        }

elif page == "Grad-CAM":
    st.header("Grad-CAM")
    st.caption("Regiões que influenciaram a decisão da rede")

    col1, col2 = st.columns(2)
    with col1:
        network_name = st.selectbox("Rede", ["densenet", "inception"])
    with col2:
        task = st.selectbox(
            "Tarefa",
            [("Binária", 2), ("4 classes", 4)],
            format_func=lambda x: x[0],
        )
        classes_num = task[1]

    key = f"{network_name}_nc{classes_num}"
    if f"model_{key}" not in st.session_state:
        st.warning("Treine ou carregue os pesos para essa tarefa primeiro.")
        st.stop()

    file = st.file_uploader(
        "Selecionar imagem para análise", type=["png", "tif", "tiff"]
    )

    NAMES_4_CLASSES = ["BIRADS I", "BIRADS II", "BIRADS III", "BIRADS IV"]
    NAMES_BINARY = ["Benigno (I+II)", "Maligno (III+IV)"]

    if file and st.button("Gerar Grad-CAM"):
        model = st.session_state[f"model_{key}"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_size = 299 if network_name == "inception" else 224

        raw = file.read()
        arr = np.array(Image.open(io.BytesIO(raw)).convert("L"))

        params = st.session_state.get(
            "segmentation_params", {"clip_limit": 2.0, "kernel_size": 7}
        )
        
        arr_clahe = apply_clahe(arr, clip_limit=params.get("clip_limit", 2.0))
        _, cropped_segmented = segment_and_crop_image(
            arr_clahe, params["kernel_size"]
        )

        lo, hi = cropped_segmented.min(), cropped_segmented.max()
        if hi > lo:
            arr_8 = ((cropped_segmented.astype(np.float32) - lo) / (hi - lo) * 255).astype(
                np.uint8
            )
        else:
            arr_8 = np.zeros_like(cropped_segmented, dtype=np.uint8)

        resized_arr = cv2.resize(arr_8, (img_size, img_size))

        tensor = torch.from_numpy(resized_arr).to(device)
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)

        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
        elif tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor = normalize(tensor)

        layer = get_target_layer(model, network_name)
        gradcam = GradCAM(model, layer)
        cam, pred_id, prob = gradcam.generate(tensor, class_id=None)
        heatmap, overlay = apply_gradcam(cam, resized_arr, img_size)

        col1, col2, col3 = st.columns(3)
        col1.image(resized_arr, caption="Segmentada e Cortada", clamp=True, width="content")
        col2.image(heatmap, caption="Mapa de calor", clamp=True, width="content")
        col3.image(overlay, caption="Sobreposição", clamp=True, width="content")

        names = NAMES_BINARY if classes_num == 2 else NAMES_4_CLASSES
        st.divider()
        st.subheader("Resultado da classificação")

        col4, col5 = st.columns(2)
        col4.metric("Classe predita", names[pred_id])
        col5.metric("Confiança", f"{prob:.2%}")

        with torch.no_grad():
            output = model(tensor.unsqueeze(0))
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().tolist()

        st.caption("Probabilidade por classe:")

        for name, p in zip(names, probs):
            st.progress(p, text=f"{name}: {p:.2%}")
