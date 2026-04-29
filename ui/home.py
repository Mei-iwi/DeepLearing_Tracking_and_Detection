from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
from PIL import Image
import torch
import torch.optim as optim
import torchvision.transforms as T

from src.models.common.factory import build_model
from src.training.checkpoint import load_checkpoint


CLASS_NAMES = ['Person', 'Car', 'Motorcycle', 'Bus']
CKPT_ROOT = ROOT / "shared_storage" / "checkpoints_shared"


def get_checkpoint_path(model_name: str, package_name: str = "pkg_001") -> Path:
    return CKPT_ROOT / model_name / package_name / "best_model.pth"


@st.cache_resource
def load_model(model_name: str, package_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        model_name=model_name,
        n_classes=len(CLASS_NAMES),
        device=device
    )

    ckpt_path = get_checkpoint_path(model_name, package_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    dummy_optimizer = optim.Adam(model.parameters(), lr=1e-3) # type: ignore

    model, _, _, _ = load_checkpoint(
        model=model,
        optimizer=dummy_optimizer,
        checkpoint_path=str(ckpt_path),
        device=device
    )

    model.eval()
    return model, device, ckpt_path


def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])


def predict(model, device, image):
    x = get_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    topk = torch.topk(probs, k=min(3, len(CLASS_NAMES)))
    return topk.indices.tolist(), topk.values.tolist()


st.set_page_config(page_title="Hệ thống phân loại người và vật", layout="wide")

st.markdown("""
<style>
.stApp {
    background: black;
}
h1 {
    text-align: center;
    color: #1d4ed8;
    font-size: 42px;
    font-weight: 700;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}
.stProgress > div > div > div > div {
    background-color: #2563eb;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Cấu hình")
model_name = st.sidebar.selectbox(
    "Chọn model",
    ["model_1", "model_2", "model_3", "model_4"],
    index=0
)
package_name = st.sidebar.text_input("Package checkpoint", value="pkg_001")

st.markdown("<h1>Hệ thống phân loại người và vật</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#475569;'>Tải 1 ảnh để model dự đoán</p>",
    unsafe_allow_html=True
)

try:
    model, device, ckpt_path = load_model(model_name, package_name)
    st.sidebar.success(f"Đã nạp: {ckpt_path.name}")
    st.sidebar.caption(str(ckpt_path))
except Exception as e:
    st.error(f"Lỗi nạp model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, caption="Ảnh input", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        with st.spinner("Đang phân tích ảnh..."):
            indices, values = predict(model, device, image)

        st.subheader("Kết quả dự đoán")

        top1 = CLASS_NAMES[indices[0]]
        conf1 = values[0]

        st.markdown(f"""
        <h2 style='color:#2563eb'>
        {top1} ({conf1:.2%})
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("Tỉ lệ dự đoán")
        for i in range(len(indices)):
            label = CLASS_NAMES[indices[i]]
            prob = values[i]
            st.write(label)
            st.progress(float(prob))

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Tải lên một ảnh để bắt đầu")