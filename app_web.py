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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading

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
    fast_video = st.checkbox("🔥 Video Turbo (Doble velocidad)", value=True)
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

# --- CALLBACK PARA VIDEO EN VIVO ---
last_frame_processed = {"img": None, "count": 0}
res_lock = threading.Lock()
shared_res = {"a": [], "s": []}

def run_inference_alc(img_rgb, size=384):
    res = modelo_alc(img_rgb, conf=0.35, iou=0.45, imgsz=size, verbose=False)
    with res_lock: shared_res["a"] = res

def run_inference_seg(img_rgb, size=384):
    res = modelo_seg(img_rgb, conf=0.5, classes=[0, 15, 16, 17, 18, 19, 39], imgsz=size, verbose=False)
    with res_lock: shared_res["s"] = res

def video_frame_callback(frame):
    last_frame_processed["count"] += 1
    
    # Procesamos 1 de cada 3 frames para balancear fluidez y calidad
    process_this_frame = (last_frame_processed["count"] % 3 == 0)

    img = frame.to_ndarray(format="bgr24")
    h, w = img.shape[:2]
    
    if process_this_frame:
        # Preparamos imagen optimizada para la IA (384px es un gran balance)
        img_small = cv2.resize(img, (384, int(384 * h / w)))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        
        # Ejecutamos ambos modelos en PARALELO para ganar velocidad
        t1 = threading.Thread(target=run_inference_alc, args=(img_rgb,))
        t2 = threading.Thread(target=run_inference_seg, args=(img_rgb,))
        t1.start(); t2.start()
        t1.join(); t2.join()

    # Reescalar coordenadas
    scale_x = w / 384
    scale_y = h / (384 * h / w)

    with res_lock:
        # Dibujar (Alcachofa)
        for r in shared_res["a"]:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                lx1, ly1, lx2, ly2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                cls = int(b.cls[0]); label = "Flor" if cls == 1 else "Hojas"
                cv2.rectangle(img, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
                cv2.putText(img, label, (lx1, ly1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # Dibujar (Extras)
        nombres_extra = {0: "Persona", 15: "Gato", 16: "Perro", 17: "Caballo", 18: "Oveja", 19: "Vaca", 39: "Basura"}
        for r in shared_res["s"]:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                lx1, ly1, lx2, ly2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                label = nombres_extra.get(int(b.cls[0]), "Alerta")
                cv2.rectangle(img, (lx1, ly1), (lx2, ly2), (0, 140, 255), 2)
                cv2.putText(img, label, (lx1, ly1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)

    last_frame_processed["img"] = img
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- PROCESAMIENTO DE VIDEO (Mejorado para web) ---
def process_video(in_path, out_path):
    try:
        input_container = av.open(in_path)
        output_container = av.open(out_path, mode='w', format='mp4')
        
        # Obtener stream original
        in_stream = input_container.streams.video[0]
        fps = in_stream.average_rate
        total_frames = in_stream.frames
        
        # Configurar stream de salida (H.264) con preset de alta velocidad
        out_stream = output_container.add_stream('libx264', rate=fps)
        out_stream.width = in_stream.width
        out_stream.height = in_stream.height
        out_stream.pix_fmt = 'yuv420p'
        out_stream.options = {'preset': 'ultrafast', 'crf': '28'} # Turbo encoding
        
        bar = st.progress(0, text="Procesando con motor de ALTA PRECISIÓN (Batch Inference)...")
        count = 0
        
        # Parámetros de calidad máximos
        inf_size = 384
        batch_size = 8 # Procesamos de 8 en 8 para máxima velocidad
        frame_buffer = []
        original_frames = []

        for frame in input_container.decode(video=0):
            # 1. Capturar frame
            img_pil = frame.to_image()
            original_frames.append(img_pil)
            
            # Preparar imagen escalada para la IA
            img_rgb = np.array(img_pil.convert("RGB"))
            img_small = cv2.resize(img_rgb, (inf_size, int(inf_size * in_stream.height / in_stream.width)))
            frame_buffer.append(img_small)
            
            # 2. Cuando el buffer está lleno, procesar en lote (BATCH)
            if len(frame_buffer) == batch_size:
                # Inferencia en lote: esto es MUCHO más rápido que uno por uno
                results_alc = modelo_alc(frame_buffer, imgsz=inf_size, conf=conf_val, iou=iou_val, verbose=False)
                results_seg = modelo_seg(frame_buffer, imgsz=inf_size, conf=conf_val, classes=[0, 15, 16, 17, 18, 19, 39], verbose=False)
                
                # Relación de escala
                scale_x = in_stream.width / inf_size
                scale_y = in_stream.height / (inf_size * in_stream.height / in_stream.width)
                
                # 3. Dibujar y guardar cada frame del lote
                for i in range(batch_size):
                    img_draw = np.array(original_frames[i])
                    
                    # Dibujar Alcachofas
                    for b in results_alc[i].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        lx1, ly1, lx2, ly2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                        cls = int(b.cls[0]); lbl = f"{'Flor' if cls==1 else 'Hojas'} {float(b.conf[0]):.2f}"
                        cv2.rectangle(img_draw, (lx1, ly1), (lx2, ly2), (0,0,255), 3)
                        cv2.putText(img_draw, lbl, (lx1, ly1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    
                    # Dibujar Extras
                    for b in results_seg[i].boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        lx1, ly1, lx2, ly2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                        lbl = f"{nombres_extra.get(int(b.cls[0]), 'Alerta')} {float(b.conf[0]):.2f}"
                        cv2.rectangle(img_draw, (lx1, ly1), (lx2, ly2), (0,140,255), 3)
                        cv2.putText(img_draw, lbl, (lx1, ly1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,140,255), 2)
                    
                    # Codificar frame
                    new_frame = av.VideoFrame.from_image(Image.fromarray(img_draw))
                    for packet in out_stream.encode(new_frame): output_container.mux(packet)
                    
                    count += 1
                
                # Limpiar buffers
                frame_buffer = []
                original_frames = []
                
                # Actualizar progreso
                if total_frames > 0:
                    pct = min(count / total_frames, 1.0)
                    bar.progress(pct, text=f"Procesando en Lote (Alta Precisión): {int(pct*100)}% ({count}/{total_frames})")

        # Procesar frames restantes si el total no es múltiplo del batch_size
        if frame_buffer:
            results_alc = modelo_alc(frame_buffer, imgsz=inf_size, conf=conf_val, iou=iou_val, verbose=False)
            results_seg = modelo_seg(frame_buffer, imgsz=inf_size, conf=conf_val, classes=[0, 15, 16, 17, 18, 19, 39], verbose=False)
            for i in range(len(frame_buffer)):
                img_draw = np.array(original_frames[i])
                for b in results_alc[i].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0]); lx1, ly1, lx2, ly2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                    cv2.rectangle(img_draw, (lx1, ly1), (lx2, ly2), (0,0,255), 3)
                for b in results_seg[i].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0]); lx1, ly1, lx2, ly2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                    cv2.rectangle(img_draw, (lx1, ly1), (lx2, ly2), (0,140,255), 3)
                new_frame = av.VideoFrame.from_image(Image.fromarray(img_draw))
                for packet in out_stream.encode(new_frame): output_container.mux(packet)
        
        # Finalizar codificación
        for packet in out_stream.encode():
            output_container.mux(packet)
        
        input_container.close()
        output_container.close()
        bar.success("¡Video analizado y codificado correctamente!")
        
    except Exception as e:
        st.error(f"Error procesando video: {str(e)}")
        if 'output_container' in locals(): output_container.close()

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

    tab1, tab2, tab3, tab4 = st.tabs(["📁 Subir Imagen", "📷 Tomar Foto", "📁 Cargar Video", "⚡ Análisis en Vivo"])
    
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
        st.markdown("### 📁 Analizador de Archivos de Video")
        st.info("📂 Selecciona un video de tu galería o archivos para procesar el análisis completo frame a frame.")
        v = st.file_uploader("Subir video para análisis...", type=["mp4", "mov", "avi"], key="uploader_vid_file")
        if v: 
            st.session_state.vid_file = v
            st.session_state.img_file = None

    with tab4:
        st.markdown("### ⚡ Identificación en Tiempo Real")
        st.warning("⚠️ El análisis en vivo consume más CPU. Úsalo en un área con buena iluminación.")
        
        rtc_config = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]}
        )
        
        webrtc_streamer(
            key="live-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640}, 
                    "height": {"ideal": 480},
                    "facingMode": "environment",
                    "frameRate": {"ideal": 20}
                }, 
                "audio": False
            },
            async_processing=True,
        )

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


