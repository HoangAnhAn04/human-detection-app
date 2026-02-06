import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Config page - dùng wide layout để hiển thị 2 cột
st.set_page_config(page_title="Human Detection", page_icon="🧑", layout="wide")

# Load model (cache để không load lại mỗi lần)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("human_classifier_mobilenet.h5")

try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi load model: {e}")
    st.stop()

# Preprocess function (giống MobileNetV2)
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    # MobileNetV2 preprocess: scale to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)

# Hàm hiển thị kết quả (dùng chung cho cả 3 tabs)
def show_result(score):
    if score < 0.5:
        confidence = (1 - score) * 100
        st.success("✅ **HUMAN DETECTED**")
        st.metric("Độ tin cậy", f"{confidence:.1f}%")
    else:
        confidence = score * 100
        st.warning("❌ **NOT HUMAN**")
        st.metric("Độ tin cậy", f"{confidence:.1f}%")
    st.progress(confidence / 100)

# UI Header
st.title("🧑 Human Detection")
st.markdown("An Hoàng Anh - 223332813")
st.write("Upload ảnh để phát hiện có phải người hay không")
st.markdown("---")

# Tabs cho các phương thức input
tab_upload, tab_camera, tab_url = st.tabs(["📁 Upload file", "📷 Webcam", "🔗 URL"])

# ==================== TAB UPLOAD ====================
with tab_upload:
    col_upload, col_result_upload = st.columns([3, 2])
    
    with col_upload:
        st.markdown("##### 📁 Chọn ảnh từ máy tính")
        uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Ảnh đã chọn", use_container_width=True)
            
            if st.button("🔍 Submit", type="primary", use_container_width=True, key="submit_upload"):
                try:
                    with st.spinner("Đang phân tích..."):
                        img_array = preprocess_image(image)
                        score = float(model.predict(img_array, verbose=0)[0][0])
                        st.session_state.upload_result = score
                except Exception as e:
                    st.error(f"Lỗi: {e}")
    
    with col_result_upload:
        st.markdown("##### 📊 Kết quả phân tích")
        with st.container(border=True):
            if "upload_result" in st.session_state and st.session_state.upload_result is not None:
                show_result(st.session_state.upload_result)
            else:
                st.caption("Chọn ảnh và nhấn Submit để xem kết quả")

# ==================== TAB WEBCAM ====================
with tab_camera:
    # Session state cho webcam
    if "webcam_enabled" not in st.session_state:
        st.session_state.webcam_enabled = False
    if "camera_result" not in st.session_state:
        st.session_state.camera_result = None
    
    if not st.session_state.webcam_enabled:
        st.info("📷 Nhấn nút bên dưới để bật webcam")
        if st.button("🎥 Bật Webcam", type="primary"):
            st.session_state.webcam_enabled = True
            st.session_state.camera_result = None
            st.rerun()
    else:
        col_cam, col_result_cam = st.columns([3, 2])
        
        with col_cam:
            st.markdown("##### 📷 Chụp ảnh từ webcam")
            camera_photo = st.camera_input("Chụp ảnh", label_visibility="collapsed")
            
            if camera_photo is not None:
                camera_image = Image.open(camera_photo).convert("RGB")
                
                if st.button("🔍 Submit", type="primary", use_container_width=True, key="submit_camera"):
                    try:
                        with st.spinner("Đang phân tích..."):
                            img_array = preprocess_image(camera_image)
                            score = float(model.predict(img_array, verbose=0)[0][0])
                            st.session_state.camera_result = score
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
            
            if st.button("❌ Tắt Webcam"):
                st.session_state.webcam_enabled = False
                st.session_state.camera_result = None
                st.rerun()
        
        with col_result_cam:
            st.markdown("##### 📊 Kết quả phân tích")
            with st.container(border=True):
                if st.session_state.camera_result is not None:
                    show_result(st.session_state.camera_result)
                else:
                    st.caption("Chụp ảnh và nhấn Submit để xem kết quả")

# ==================== TAB URL ====================
with tab_url:
    col_url, col_result_url = st.columns([3, 2])
    
    with col_url:
        st.markdown("##### 🔗 Nhập URL ảnh từ internet")
        url_input = st.text_input("URL ảnh:", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
        st.caption("💡 Nhấn **Enter** để tải ảnh")
        
        if url_input:
            try:
                with st.spinner("Đang tải ảnh..."):
                    response = requests.get(url_input, timeout=10)
                    response.raise_for_status()
                    url_image = Image.open(BytesIO(response.content)).convert("RGB")
                    st.image(url_image, caption="Ảnh từ URL", use_container_width=True)
                    
                    if st.button("🔍 Submit", type="primary", use_container_width=True, key="submit_url"):
                        try:
                            with st.spinner("Đang phân tích..."):
                                img_array = preprocess_image(url_image)
                                score = float(model.predict(img_array, verbose=0)[0][0])
                                st.session_state.url_result = score
                        except Exception as e:
                            st.error(f"Lỗi: {e}")
            except Exception as e:
                st.error(f"Không thể tải ảnh từ URL: {e}")
    
    with col_result_url:
        st.markdown("##### 📊 Kết quả phân tích")
        with st.container(border=True):
            if "url_result" in st.session_state and st.session_state.url_result is not None:
                show_result(st.session_state.url_result)
            else:
                st.caption("Nhập URL và nhấn Submit để xem kết quả")

# Footer
st.markdown("---")
st.caption("MobileNetV2 + Streamlit | An Hoàng Anh - 223332813")