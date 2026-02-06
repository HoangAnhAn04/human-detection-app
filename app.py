import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Config page
st.set_page_config(page_title="Human Detection", page_icon="🧑", layout="centered")

# Load model (cache để không load lại mỗi lần)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("human_classifier_mobilenet.h5")

model = load_model()

# Preprocess function (giống MobileNetV2)
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    # MobileNetV2 preprocess: scale to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)

# UI
st.title("🧑 Human Detection")
st.write("Upload ảnh để phát hiện có người hay không")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã upload", use_column_width=True)
    
    # Predict
    with st.spinner("Đang phân tích..."):
        img_array = preprocess_image(image)
        score = model.predict(img_array, verbose=0)[0][0]
    
    # Hiển thị kết quả
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    if score < 0.5:
        confidence = (1 - score) * 100
        col1.metric("Kết quả", "✅ HUMAN")
        col2.metric("Độ tin cậy", f"{confidence:.1f}%")
        st.success(f"Phát hiện: **CÓ NGƯỜI** trong ảnh!")
    else:
        confidence = score * 100
        col1.metric("Kết quả", "❌ NOT HUMAN")
        col2.metric("Độ tin cậy", f"{confidence:.1f}%")
        st.warning(f"Phát hiện: **KHÔNG CÓ NGƯỜI** trong ảnh!")
    
    # Progress bar
    st.write("**Confidence Score:**")
    st.progress(confidence / 100)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using MobileNetV2 + Streamlit")