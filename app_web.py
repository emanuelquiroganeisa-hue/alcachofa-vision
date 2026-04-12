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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Diseño Premium con CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-color: #00c853; color: white; font-weight: bold;
        transition: 0.3s; border: none;
    }
    .stButton>button:hover { background-color: #00e676; transform: scale(1.02); }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf, #2e7bcf); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CARGAR MODELOS ---
@st.cache_resource
def load_models():
    # Ruta por defecto del modelo especializado
    ruta_modelo = r"alcachofa_detector3/weights/best.pt"
    if not os.path.exists(ruta_modelo):
        # Si no existe en la ruta local, se busca en la raíz
        ruta_modelo = "best.pt"
        
    m_alcachofa = None
    if os.path.exists(ruta_modelo):
        m_alcachofa = YOLO(ruta_modelo)
        
    m_seguridad = YOLO("yolov8n.pt") # Modelo estándar para intrusos
    return m_alcachofa, m_seguridad

modelo_alc, modelo_seg = load_models()

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1047/1047461.png", width=100)
    st.title("Configuración")
    
    if modelo_alc is None:
        st.error("⚠️ Modelo especializado no encontrado.")
        up_file = st.file_uploader("Sube tu archivo .pt", type=['pt'])
        if up_file:
            with open("best.pt", "wb") as f:
                f.write(up_file.getbuffer())
            st.rerun()
    else:
        st.success("✅ Modelo Alcachofa listo")

    st.subheader("Sensibilidad")
    conf_val = st.slider("Confianza Detección", 0.01, 1.0, 0.25)
    iou_val = st.slider("Traslape (IOU)", 0.01, 1.0, 0.45)
    
    st.subheader("Mejoras de Imagen")
    use_clahe = st.checkbox("Filtro CLAHE (Luz)", value=True)
    use_tta = st.checkbox("TTA (Máxima Precisión)", value=True)
    
    st.subheader("Servidor")
    save_in_server = st.checkbox("💾 Guardar en carpeta local del servidor", value=False)
    SAVE_PATH = "resultados_servidor"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

# --- CUERPO PRINCIPAL ---
st.title("🌱 Alcachofa Vision: Identificador Inteligente")
st.write("Carga una imagen desde tu galería o usa la cámara del celular para analizar malezas y plantas.")

# Selector de origen de imagen
source = st.radio("Origen de la imagen:", ["Subir Archivo / Galería", "Usar Cámara"], horizontal=True)

if source == "Subir Archivo / Galería":
    upload_img = st.file_uploader("Selecciona una imagen...", type=['jpg', 'jpeg', 'png'])
else:
    upload_img = st.camera_input("Toma una foto")

if upload_img:
    # Leer imagen
    file_bytes = np.asarray(bytearray(upload_img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1) # BGR
    
    col_orig, col_res = st.columns(2)
    
    with col_orig:
        st.subheader("Imagen de Entrada")
        st.image(image, channels="BGR", use_container_width=True)

    if st.button("🔍 INICIAR ANÁLISIS"):
        if modelo_alc is None:
            st.error("No se puede iniciar el análisis sin el archivo .pt")
        else:
            with st.spinner("Analizando con YOLO v8..."):
                # --- PROCESAMIENTO ---
                img_to_predict = image.copy()
                
                if use_clahe:
                    lab = cv2.cvtColor(img_to_predict, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    img_to_predict = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

                # Detección
                res_alc = modelo_alc(img_to_predict, augment=use_tta, conf=conf_val, iou=iou_val, imgsz=800)
                
                # Seguridad (Intrusos)
                clases_seg = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 39]
                res_sec = modelo_seg(img_to_predict, conf=0.35, classes=clases_seg)

                # --- DIBUJO ---
                img_draw = image.copy()
                
                # Clasificar cajas
                cajas_planta = []
                for r in res_alc:
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cls = int(b.cls[0])
                        conf = float(b.conf[0])
                        cajas_planta.append([x1, y1, x2, y2, cls, conf])

                detecciones_seg = []
                for r in res_sec:
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cls = int(b.cls[0])
                        conf = float(b.conf[0])
                        detecciones_seg.append([x1, y1, x2, y2, cls, conf])

                # Máscara maleza
                mask_maleza = np.ones(img_draw.shape[:2], dtype=np.uint8) * 255
                for c in cajas_planta:
                    cv2.rectangle(mask_maleza, (c[0], c[1]), (c[2], c[3]), 0, -1)
                for c in detecciones_seg:
                    cv2.rectangle(mask_maleza, (c[0], c[1]), (c[2], c[3]), 0, -1)

                # Tinte azul maleza
                capa_azul = np.zeros_like(img_draw)
                capa_azul[:] = (255, 0, 0)
                img_tinted = cv2.addWeighted(img_draw, 0.6, capa_azul, 0.4, 0)
                img_draw[mask_maleza == 255] = img_tinted[mask_maleza == 255]

                # Cajas Planta
                for c in cajas_planta:
                    text = "Flor" if c[4] == 1 else "Hojas"
                    cv2.rectangle(img_draw, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), 2)
                    cv2.putText(img_draw, f"{text} {c[5]:.2f}", (c[0], c[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Cajas Seguridad
                labels_coco = {0: "Persona", 39: "Basura/Botella"}
                for c in detecciones_seg:
                    label = labels_coco.get(c[4], "Intruso")
                    cv2.rectangle(img_draw, (c[0], c[1]), (c[2], c[3]), (0, 140, 255), 3)
                    cv2.putText(img_draw, f"{label} {c[5]:.2f}", (c[0], c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

                with col_res:
                    st.subheader("Resultado del Análisis")
                    st.image(img_draw, channels="BGR", use_container_width=True)
                    
                    # Botón Descargar
                    _, buf = cv2.imencode(".jpg", img_draw)
                    st.download_button("💾 Descargar Resultado (.jpg)", buf.tobytes(), "resultado_alcachofa.jpg", "image/jpeg")

                # Guardado en servidor
                if save_in_server:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = os.path.join(SAVE_PATH, f"detec_{ts}.jpg")
                    cv2.imwrite(fname, img_draw)
                    st.toast(f"Guardado localmente en: {fname}")

st.markdown("---")
st.caption("Desarrollado para identificación agrícola de precisión.")
