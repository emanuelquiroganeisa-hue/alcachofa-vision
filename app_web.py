import streamlit as st
import numpy as np
import cv2
from PIL import Image
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

# Estilos Premium
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; background-color: #00c853; color: white; font-weight: bold; }
    .stSelectbox label { color: #00c853; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

HF_REPO_ID = "Emanuel1102/alcachofa-model"
HF_FILENAME = "best.pt"
SAVE_PATH = "resultados_servidor"
if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

# --- CARGA DE MODELOS ---
@st.cache_resource
def load_models():
    m_alc = None
    if os.path.exists("best.pt"):
        try: m_alc = YOLO("best.pt")
        except: pass
    if m_alc is None:
        with st.spinner("⬇️ Descargando modelo..."):
            ruta = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, local_dir=".")
            m_alc = YOLO(ruta)
    m_seg = YOLO("yolov8n.pt")
    return m_alc, m_seg

modelo_alc, modelo_seg = load_models()

# --- MENÚ DE NAVEGACIÓN ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1047/1047461.png", width=80)
    st.title("🌱 Menú Principal")
    opcion = st.radio("Ir a:", ["🚀 Identificador", "📂 Historial de Guardados"])
    
    st.divider()
    st.subheader("⚙️ Parámetros de Análisis")
    use_clahe = st.checkbox("Filtro CLAHE (Luz)", value=True)
    use_tta   = st.checkbox("Usar TTA (Precisión)", value=True)
    conf_val  = st.slider("Confianza", 0.01, 1.0, 0.25, step=0.01)
    iou_val   = st.slider("Solapado", 0.01, 1.0, 0.45, step=0.01)

    st.divider()
    save_local = st.checkbox("💾 Guardar automáticamente en servidor", value=True)


# ─────────────────────────────────────────────
# LOGICA DE PROCESAMIENTO
# ─────────────────────────────────────────────

def aplicar_clahe(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l); limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(cv2.cvtColor(limg, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

def fusionar_cajas(cajas_por_clase):
    cajas_finales = []
    for cls, lista_cajas in cajas_por_clase.items():
        cambio = True
        while cambio:
            cambio = False; nuevas_cajas = []
            while lista_cajas:
                caja = lista_cajas.pop(0); merged = False
                for i in range(len(lista_cajas)):
                    otra = lista_cajas[i]
                    if not (caja[2]+15 < otra[0] or caja[0]-15 > otra[2] or caja[3]+15 < otra[1] or caja[1]-15 > otra[3]):
                        lista_cajas[i] = [min(caja[0],otra[0]), min(caja[1],otra[1]), max(caja[2],otra[2]), max(caja[3],otra[3]), max(caja[4],otra[4])]
                        merged = True; cambio = True; break
                if not merged: nuevas_cajas.append(caja)
            lista_cajas = nuevas_cajas
        for c in lista_cajas: cajas_finales.append((cls, *c))
    return cajas_finales

def main_process(imagen_pil):
    img_rgb = np.array(imagen_pil.convert("RGB"))
    img_para_yolo = Image.fromarray(aplicar_clahe(img_rgb)) if use_clahe else imagen_pil
    
    res_a = modelo_alc(img_para_yolo, augment=use_tta, conf=conf_val, iou=iou_val, imgsz=800)
    res_s = modelo_seg(img_para_yolo, conf=0.35, classes=[0, 39]) # Simplificado para seguridad

    detecciones_seg = []
    for r in res_s:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            detecciones_seg.append([x1, y1, x2, y2, int(b.cls[0]), float(b.conf[0])])

    cajas_p = {}
    for r in res_a:
        if r.boxes:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cajas_p.setdefault(int(b.cls[0]), []).append([x1, y1, x2, y2, float(b.conf[0])])

    cp = [c for c in fusionar_cajas(cajas_p) if c[0] in [0, 1]]
    img_d = img_rgb.copy()
    mask = np.ones(img_d.shape[:2], dtype=np.uint8) * 255
    for c in cp: cv2.rectangle(mask, (int(c[1]), int(c[2])), (int(c[3]), int(c[4])), 0, -1)
    for s in detecciones_seg: cv2.rectangle(mask, (s[0], s[1]), (s[2], s[3]), 0, -1)

    capa_a = np.zeros_like(img_d); capa_a[:] = (0, 0, 255)
    img_t = np.clip(img_d * 0.6 + capa_a * 0.4, 0, 255).astype(np.uint8)
    img_d[mask == 255] = img_t[mask == 255]

    img_bgr = cv2.cvtColor(img_d, cv2.COLOR_RGB2BGR)
    for c in cp:
        cv2.rectangle(img_bgr, (int(c[1]), int(c[2])), (int(c[3]), int(c[4])), (0, 0, 255), 2)
        cv2.putText(img_bgr, f"{'Flor' if c[0]==1 else 'Hojas'} {c[5]:.2f}", (int(c[1]), max(int(c[2])-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    for s in detecciones_seg:
        cv2.rectangle(img_bgr, (s[0], s[1]), (s[2], s[3]), (0, 140, 255), 3)
        cv2.putText(img_bgr, f"{'Persona' if s[4]==0 else 'Basura'} {s[5]:.2f}", (s[0], max(s[1]-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,140,255), 2)

    res_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    if save_local:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        res_pil.save(f"{SAVE_PATH}/detec_{ts}.jpg", quality=95)
        
    return res_pil, len(cp), len(detecciones_seg)

# ─────────────────────────────────────────────
# VISTAS
# ─────────────────────────────────────────────

if opcion == "🚀 Identificador":
    st.title("🚀 Analizador de Plantaciones")
    tab1, tab2 = st.tabs(["📁 Subir", "📷 Cámara"])
    img_in = None
    with tab1:
        u = st.file_uploader("Imagen...", type=["jpg","jpeg","png"])
        if u: img_in = u
    with tab2:
        c = st.camera_input("Foto...")
        if c: img_in = c
    
    if img_in:
        pil_in = Image.open(img_in).convert("RGB")
        col1, col2 = st.columns(2)
        with col1: st.image(pil_in, caption="Original")
        if st.button("🔍 COMENZAR ANÁLISIS"):
            res, np, ns = main_process(pil_in)
            with col2:
                st.image(res, caption="Analizado")
                st.metric("Detecciones", f"{np} Plantas | {ns} Alertas")
                buf = io.BytesIO(); res.save(buf, format="JPEG")
                st.download_button("💾 Descargar", buf.getvalue(), "resultado.jpg", "image/jpeg")

elif opcion == "📂 Historial de Guardados":
    st.title("📂 Historial de Detecciones en el Servidor")
    archivos = sorted(os.listdir(SAVE_PATH), reverse=True)
    
    if not archivos:
        st.info("Aún no hay imágenes guardadas en el servidor.")
    else:
        st.write(f"Se han encontrado **{len(archivos)}** capturas guardadas.")
        for arc in archivos:
            with st.expander(f"🖼️ {arc}"):
                col_img, col_info = st.columns([3, 1])
                ruta = os.path.join(SAVE_PATH, arc)
                with col_img:
                    st.image(ruta, use_container_width=True)
                with col_info:
                    st.write(f"**Fecha:** {arc.split('_')[1]}")
                    with open(ruta, "rb") as f:
                        st.download_button(f"📥 Descargar", f.read(), arc, "image/jpeg", key=arc)
                    if st.button(f"🗑️ Eliminar", key=f"del_{arc}"):
                        os.remove(ruta)
                        st.rerun()



