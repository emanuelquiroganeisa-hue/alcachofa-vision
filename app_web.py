import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import datetime
import io
import zipfile

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
SAVE_PATH_CAM = "camara_originales"
SAVE_PATH_VID = "videos_originales"
SAVE_PATH_VID_OUT = "videos_analizados"

for path in [SAVE_PATH, SAVE_PATH_CAM, SAVE_PATH_VID, SAVE_PATH_VID_OUT]:
    if not os.path.exists(path): os.makedirs(path)

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
    opcion = st.radio("Ir a:", [
        "🚀 Identificador", 
        "📂 Historial Análisis", 
        "📸 Fotos de Cámara",
        "💾 Videos Originales",
        "🎬 Videos Analizados"
    ])
    
    st.divider()
    st.subheader("⚙️ Parámetros de Análisis")
    use_clahe = st.checkbox("Filtro CLAHE (Luz)", value=True)
    use_tta   = st.checkbox("Usar TTA (Precisión)", value=True)
    conf_val  = st.slider("Confianza", 0.01, 1.0, 0.25, step=0.01)
    iou_val   = st.slider("Solapado", 0.01, 1.0, 0.45, step=0.01)

    st.divider()
    save_local = st.checkbox("💾 Guardar auto. en servidor", value=True)


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

def main_process(imagen_pil, save_to_disk=False):
    img_rgb = np.array(imagen_pil.convert("RGB"))
    img_para_yolo = Image.fromarray(aplicar_clahe(img_rgb)) if use_clahe else imagen_pil
    
    res_a = modelo_alc(img_para_yolo, augment=use_tta, conf=conf_val, iou=iou_val, imgsz=800)
    # Clases COCO: 0:Persona, 15:Gato, 16:Perro, 17:Caballo, 18:Oveja, 19:Vaca, 39:Botella(Basura)
    nombres_extra = {0: "Persona", 15: "Gato", 16: "Perro", 17: "Caballo", 18: "Oveja", 19: "Vaca", 39: "Basura"}
    res_s = modelo_seg(img_para_yolo, conf=0.35, classes=list(nombres_extra.keys()))

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
        label_extra = nombres_extra.get(s[4], "Alerta")
        cv2.putText(img_bgr, f"{label_extra} {s[5]:.2f}", (s[0], max(s[1]-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

    res_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    if save_to_disk and save_local:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        res_pil.save(f"{SAVE_PATH}/detec_{ts}.jpg", quality=95)
        
        
    return res_pil, len(cp), len(detecciones_seg)

# --- PROCESAMIENTO DE VIDEO ---
def process_video(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Intentamos MP4V, si falla el usuario verá el error o podrá descargar
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    bar = st.progress(0, text="Procesando video...")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Procesar frame (se omite CLAHE en video para mayor velocidad si se desea, 
        # pero aquí respetamos la selección del usuario)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        res_pil, _, _ = main_process(pil_frame, save_to_disk=False)
        
        # Volver a BGR para OpenCV
        res_frame = cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR)
        out.write(res_frame)
        
        count += 1
        pct = count / total
        bar.progress(pct, text=f"Procesando: {int(pct*100)}% ({count}/{total})")
    
    cap.release()
    out.release()
    bar.success("¡Video procesado con éxito!")

# --- FUNCION PARA MOSTRAR HISTORIAL GENERAL ---
def render_historial(path, titulo, is_video=False):
    st.title(titulo)
    archivos = sorted(os.listdir(path), reverse=True)
    if not archivos:
        st.info("No hay archivos en esta sección.")
    else:
        st.write(f"Total: **{len(archivos)}** archivos.")
        buf_zip = io.BytesIO()
        with zipfile.ZipFile(buf_zip, "w") as zf:
            for arc in archivos: zf.write(os.path.join(path, arc), arc)
        
        c1, c2 = st.columns(2)
        with c1: st.download_button("📥 Descargar Todo (ZIP)", buf_zip.getvalue(), f"{path}_completo.zip", "application/zip")
        with c2: 
            if st.button("🗑️ Borrar Todo", key=f"clear_{path}"):
                for arc in archivos: os.remove(os.path.join(path, arc))
                st.rerun()
        st.divider()
        for arc in archivos:
            with st.expander(f"{'🎥' if is_video else '🖼️'} {arc}"):
                ruta = os.path.join(path, arc)
                col_media, col_btn = st.columns([3, 1])
                with col_media: 
                    if is_video: st.video(ruta)
                    else: st.image(ruta, use_container_width=True)
                with col_btn:
                    with open(ruta, "rb") as f: 
                        tipo = "video/mp4" if is_video else "image/jpeg"
                        st.download_button("📥 Descargar", f.read(), arc, tipo, key=f"d_{path}_{arc}")
                    if st.button("🗑️ Eliminar", key=f"del_{path}_{arc}"):
                        os.remove(ruta); st.rerun()

# ─────────────────────────────────────────────
# VISTAS PRINCIPALES
# ─────────────────────────────────────────────

if "img_file" not in st.session_state: st.session_state.img_file = None
if "vid_file" not in st.session_state: st.session_state.vid_file = None

if opcion == "🚀 Identificador":
    st.title("🚀 Analizador de Plantaciones")
    
    # Botón para limpiar todo
    if st.button("🚿 Limpiar Selección"):
        st.session_state.img_file = None
        st.session_state.vid_file = None
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["📁 Subir Imagen", "📷 Tomar Foto", "🎥 Tomar Video"])
    
    with tab1:
        u = st.file_uploader("Seleccionar imagen...", type=["jpg","jpeg","png"], key="uploader_img")
        if u: 
            st.session_state.img_file = u
            st.session_state.vid_file = None # Priorizar imagen si se sube

    with tab2:
        c = st.camera_input("Capturar foto...", key="cam_img")
        if c: 
            st.session_state.img_file = c
            st.session_state.vid_file = None
            # Guardamos la original inmediatamente si es de cámara
            if save_local:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                Image.open(c).save(f"{SAVE_PATH_CAM}/original_{ts}.jpg", quality=95)
    
    with tab3:
        st.markdown("### 🎥 Grabadora Directa")
        st.info("💡 **Consejo:** En celulares, pulsa el botón y elige 'Cámara' (Video). Al terminar, espera a que la barra azul de carga llegue al 100%.")
        v = st.file_uploader("🔴 GRABAR / SUBIR VIDEO", type=["mp4", "mov", "avi"], key="recorder_vid")
        if v: 
            st.session_state.vid_file = v
            st.session_state.img_file = None

    img_in = st.session_state.img_file
    vid_in = st.session_state.vid_file

    if img_in:
        pil_in = Image.open(img_in).convert("RGB")
        col1, col2 = st.columns(2)
        with col1: st.image(pil_in, caption="Entrada")
        if st.button("🔍 COMENZAR ANÁLISIS"):
            res, np, ns = main_process(pil_in, save_to_disk=True)
            with col2:
                st.image(res, caption="Resultado")
                st.metric("Detecciones", f"{np} Plantas | {ns} Alertas")
                buf = io.BytesIO(); res.save(buf, format="JPEG")
                st.download_button("💾 Descargar Resultado", buf.getvalue(), "resultado.jpg", "image/jpeg")

    elif vid_in:
        st.success("🎬 Video listo para análisis")
        st.subheader("📼 Previsualización")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        v_in_path = f"{SAVE_PATH_VID}/video_{ts}.mp4"
        v_out_path = f"{SAVE_PATH_VID_OUT}/analizado_{ts}.mp4"
        
        with open(v_in_path, "wb") as f: f.write(vid_in.read())
        
        col_v1, col_v2 = st.columns(2)
        with col_v1: st.video(v_in_path)
        
        if st.button("⚙️ PROCESAR VIDEO (Identificador)"):
            process_video(v_in_path, v_out_path)
            with col_v2:
                st.video(v_out_path)
                with open(v_out_path, "rb") as f:
                    st.download_button("💾 Descargar Video Analizado", f.read(), f"analizado_{ts}.mp4", "video/mp4")

elif opcion == "📂 Historial Análisis":
    render_historial(SAVE_PATH, "📂 Historial de Detecciones (Imágenes)")

elif opcion == "📸 Fotos de Cámara":
    render_historial(SAVE_PATH_CAM, "📸 Galería de Fotos Originales (Cámara)")

elif opcion == "💾 Videos Originales":
    render_historial(SAVE_PATH_VID, "💾 Historial de Videos Subidos", is_video=True)

elif opcion == "🎬 Videos Analizados":
    render_historial(SAVE_PATH_VID_OUT, "🎬 Galería de Videos Procesados", is_video=True)

