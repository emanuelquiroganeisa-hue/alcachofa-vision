import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Alcachofa Vision - Detector Pro",
    page_icon="🌱",
    layout="wide"
)

# Estilo visual
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #00c853; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- CARGAR MODELOS ---
@st.cache_resource
def load_models():
    m_alc = None
    # Intentar cargar el modelo si ya existe en la carpeta (best.pt)
    if os.path.exists("best.pt"):
        m_alc = YOLO("best.pt")
    elif os.path.exists("alcachofa_detector3/weights/best.pt"):
        m_alc = YOLO("alcachofa_detector3/weights/best.pt")
        
    m_seg = YOLO("yolov8n.pt") 
    return m_alc, m_seg

modelo_alc, modelo_seg = load_models()

# --- BARRA LATERAL ---
with st.sidebar:
    st.title("🌱 Panel de Control")
    
    if modelo_alc is None:
        st.warning("⚠️ Modelo 'best.pt' no detectado.")
        up_file = st.file_uploader("Sube tu modelo (.pt) aquí", type=['pt'])
        if up_file:
            with open("best.pt", "wb") as f:
                f.write(up_file.getbuffer())
            st.success("Modelo cargado. Reiniciando...")
            st.rerun()
    else:
        st.success("✅ Modelo Alcachofa listo")

    conf_val = st.sidebar.slider("Confianza", 0.01, 1.0, 0.25)
    use_clahe = st.sidebar.checkbox("Filtro CLAHE (Luz)", value=True)

# --- CUERPO PRINCIPAL ---
st.title("Identificador de Alcachofas y Maleza")
st.write("Sube una foto desde tu galería o cámara.")

upload_img = st.file_uploader("Selecciona una imagen...", type=['jpg', 'jpeg', 'png'])

if upload_img:
    file_bytes = np.asarray(bytearray(upload_img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1) # BGR
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Entrada")
        st.image(image, channels="BGR", use_container_width=True)

    if st.button("🚀 INICIAR IDENTIFICACIÓN"):
        if modelo_alc is None:
            st.error("Por favor, sube el modelo .pt en el menú lateral.")
        else:
            with st.spinner("Procesando..."):
                img_proc = image.copy()
                if use_clahe:
                    lab = cv2.cvtColor(img_proc, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    img_proc = cv2.cvtColor(cv2.merge((clahe.apply(l),a,b)), cv2.COLOR_LAB2BGR)

                results = modelo_alc(img_proc, conf=conf_val)
                res_sec = modelo_seg(img_proc, conf=0.35, classes=[0, 39]) # Persona y basura

                # Dibujo simplificado para velocidad
                img_draw = image.copy()
                
                # Maleza (Tinte Azul)
                mask = np.ones(img_draw.shape[:2], dtype=np.uint8) * 255
                for r in results:
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
                
                capa_azul = np.zeros_like(img_draw)
                capa_azul[:] = (255, 0, 0)
                img_tinted = cv2.addWeighted(img_draw, 0.6, capa_azul, 0.4, 0)
                img_draw[mask == 255] = img_tinted[mask == 255]

                # Cajas de Planta
                for r in results:
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        label = "Flor" if int(b.cls[0]) == 1 else "Hojas"
                        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img_draw, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                with col2:
                    st.subheader("Resultado")
                    st.image(img_draw, channels="BGR", use_container_width=True)
                    
                    _, buf = cv2.imencode(".jpg", img_draw)
                    st.download_button("💾 Guardar imagen", buf.tobytes(), "resultado.jpg", "image/jpeg")

