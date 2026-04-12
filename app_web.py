import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import datetime
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Alcachofa Vision",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em;
        background-color: #00c853; color: white; font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

HF_REPO_ID = "Emanuel1102/alcachofa-model"
HF_FILENAME = "best.pt"

# --- CARGA DE MODELOS ---
@st.cache_resource
def load_models():
    m_alc = None
    if os.path.exists("best.pt"):
        try:
            m_alc = YOLO("best.pt")
        except:
            pass

    if m_alc is None:
        with st.spinner("⬇️ Descargando modelo desde Hugging Face..."):
            ruta = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, local_dir=".")
            m_alc = YOLO(ruta)

    m_seg = YOLO("yolov8n.pt")
    return m_alc, m_seg

modelo_alc, modelo_seg = load_models()

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1047/1047461.png", width=80)
    st.title("🌱 Panel de Control")
    st.success("✅ Modelo listo")

    st.divider()
    st.subheader("⚙️ Parámetros")
    use_clahe = st.checkbox("Filtro CLAHE (Iluminación)", value=True)
    use_tta   = st.checkbox("Usar TTA (Mayor Precisión)", value=True)
    conf_val  = st.slider("Confianza (Mínima)", 0.01, 1.0, 0.25, step=0.01)
    iou_val   = st.slider("Sensibilidad (Solapado)", 0.01, 1.0, 0.45, step=0.01)

    st.divider()
    st.subheader("💾 Resultados")
    save_local = st.checkbox("Guardar en servidor automáticamente", value=False)
    SAVE_PATH = "resultados_servidor"
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    st.divider()
    st.subheader("🎨 Leyenda")
    st.markdown("🟥 Flor / Hojas (Cajas)")
    st.markdown("🟦 Maleza (Áreas Azules)")
    st.markdown("🟧 Seguridad (Naranja)")


# ─────────────────────────────────────────────
# UTILIDADES DE PROCESAMIENTO
# ─────────────────────────────────────────────

def aplicar_clahe(img_rgb):
    """Aplicación de CLAHE exacta a la interfaz original."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    res_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)

def check_overlap(b1, b2, margin=15):
    """Verifica solapamiento idéntico a interfaz_alcachofa.py."""
    return not (b1[2] + margin < b2[0] or b1[0] - margin > b2[2] or b1[3] + margin < b2[1] or b1[1] - margin > b2[3])

def fusionar_cajas(cajas_por_clase):
    """Algoritmo de fusión idéntico al de escritorio."""
    cajas_finales = []
    for cls, lista_cajas in cajas_por_clase.items():
        cambio = True
        while cambio:
            cambio = False
            nuevas_cajas = []
            while lista_cajas:
                caja = lista_cajas.pop(0)
                merged = False
                for i in range(len(lista_cajas)):
                    otra = lista_cajas[i]
                    if check_overlap(caja, otra):
                        nx1, ny1 = min(caja[0], otra[0]), min(caja[1], otra[1])
                        nx2, ny2 = max(caja[2], otra[2]), max(caja[3], otra[3])
                        nconf = max(caja[4], otra[4])
                        lista_cajas[i] = [nx1, ny1, nx2, ny2, nconf]
                        merged = True
                        cambio = True
                        break
                if not merged: nuevas_cajas.append(caja)
            lista_cajas = nuevas_cajas
        for cx1, cy1, cx2, cy2, cconf in lista_cajas:
            cajas_finales.append((cls, cx1, cy1, cx2, cy2, cconf))
    return cajas_finales

def main_process(imagen_pil):
    """Procesamiento optimizado."""
    img_rgb = np.array(imagen_pil.convert("RGB"))
    # Pasamos PIL a YOLO para evitar problemas de canales BGR/RGB
    img_para_yolo = Image.fromarray(aplicar_clahe(img_rgb)) if use_clahe else imagen_pil

    # Detecciones
    res_a = modelo_alc(img_para_yolo, augment=use_tta, conf=conf_val, iou=iou_val, imgsz=800)
    clases_seg = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 39]
    res_s = modelo_seg(img_para_yolo, conf=0.35, classes=clases_seg)

    detecciones_seg = []
    for r in res_s:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            detecciones_seg.append([x1, y1, x2, y2, int(b.cls[0]), float(b.conf[0])])

    cajas_por_clase = {}
    for r in res_a:
        if r.boxes:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cajas_por_clase.setdefault(int(b.cls[0]), []).append([x1, y1, x2, y2, float(b.conf[0])])

    cajas_planta = [c for c in fusionar_cajas(cajas_por_clase) if c[0] in [0, 1]]

    # Dibujo de maleza y protección de áreas
    img_dibujo = img_rgb.copy()
    mask_maleza = np.ones(img_dibujo.shape[:2], dtype=np.uint8) * 255
    for c in cajas_planta: cv2.rectangle(mask_maleza, (c[1], c[2]), (c[3], c[4]), 0, -1)
    for s in detecciones_seg: cv2.rectangle(mask_maleza, (s[0], s[1]), (s[2], s[3]), 0, -1)

    capa_azul = np.zeros_like(img_dibujo); capa_azul[:] = (0, 0, 255) # RGB azul
    img_tinte = np.clip(img_dibujo * 0.6 + capa_azul * 0.4, 0, 255).astype(np.uint8)
    img_dibujo[mask_maleza == 255] = img_tinte[mask_maleza == 255]

    # Dibujo de cajas finales (usando BGR para cv2 y luego volviendo a RGB)
    img_bgr = cv2.cvtColor(img_dibujo, cv2.COLOR_RGB2BGR)
    for c in cajas_planta:
        lbl = "Flor" if c[0] == 1 else "Hojas"
        cv2.rectangle(img_bgr, (c[1], c[2]), (c[3], c[4]), (0, 0, 255), 2)
        cv2.putText(img_bgr, f"{lbl} {c[5]:.2f}", (c[1], max(c[2]-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    nombres_s = {0: "Persona", 39: "Botella/Basura"}
    for s in detecciones_seg:
        lbl = nombres_s.get(s[4], "Intruso")
        cv2.rectangle(img_bgr, (s[0], s[1]), (s[2], s[3]), (0, 140, 255), 3) # Naranja
        cv2.putText(img_bgr, f"{lbl} {s[5]:.2f}", (s[0], max(s[1]-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), len(cajas_planta), len(detecciones_seg)

# --- INTERFAZ ---
st.title("🌱 Alcachofa Vision (Espejo Local)")
st.write("Sube una imagen o usa la cámara de tu dispositivo para identificar alcachofas y malezas.")

tab1, tab2 = st.tabs(["📁 Subir Imagen", "📷 Usar Cámara"])
upload_img = None

with tab1:
    up = st.file_uploader("Selecciona una imagen...", type=["jpg", "png", "jpeg"])
    if up: upload_img = up

with tab2:
    cam = st.camera_input("Captura una foto desde tu dispositivo")
    if cam: upload_img = cam

if upload_img:
    img_pil = Image.open(upload_img).convert("RGB")
    col1, col2 = st.columns(2)
    with col1: 
        st.subheader("📸 Imagen de Entrada")
        st.image(img_pil, use_container_width=True)
    
    if st.button("🚀 INICIAR IDENTIFICACIÓN"):
        with st.spinner("🔍 Analizando con YOLOv8..."):
            res_img, n_plant, n_seg = main_process(img_pil)
        
        with col2:
            st.subheader("✅ Resultado del Análisis")
            st.image(res_img, use_container_width=True)
            
            # Métricas
            m1, m2 = st.columns(2)
            m1.metric("🍃 Plantas Detectadas", n_plant)
            m2.metric("⚠️ Alertas Seguridad", n_seg)
            
            # Botón de descarga para el resultado
            buf = io.BytesIO()
            res_img.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="💾 Descargar Resultado",
                data=buf.getvalue(),
                file_name=f"deteccion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg"
            )


