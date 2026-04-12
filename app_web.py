import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import datetime
import io

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    st.warning("⚠️ OpenCV no disponible — funciones CLAHE y dibujo en modo compatibilidad.")

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
        except Exception as e:
            st.sidebar.warning(f"Modelo local caído: {e}")

    if m_alc is None:
        try:
            with st.spinner("⬇️ Descargando modelo desde Hugging Face (solo la primera vez)..."):
                ruta = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=HF_FILENAME,
                    local_dir="."
                )
            m_alc = YOLO(ruta)
        except Exception as e:
            st.sidebar.error(f"❌ No se pudo descargar el modelo: {e}")

    m_seg = YOLO("yolov8n.pt")
    return m_alc, m_seg

modelo_alc, modelo_seg = load_models()

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1047/1047461.png", width=80)
    st.title("🌱 Panel de Control")

    if modelo_alc is None:
        st.error("❌ Modelo no disponible.")
        st.info("Sube manualmente el archivo si Hugging Face no responde:")
        up_file = st.file_uploader("Cargar modelo (.pt)", type=["pt"])
        if up_file:
            with open("best.pt", "wb") as f:
                f.write(up_file.getbuffer())
            st.success("✅ Modelo cargado. Actualizando...")
            st.rerun()
    else:
        st.success("✅ Modelo listo")

    st.divider()
    st.subheader("⚙️ Parámetros")
    use_clahe = st.checkbox("Filtro CLAHE (Iluminación)", value=True)
    use_tta   = st.checkbox("Usar TTA (Mayor Precisión)", value=True)
    conf_val  = st.slider("Confianza (Mínima)",       0.01, 1.0, 0.25, step=0.01)
    iou_val   = st.slider("Sensibilidad (Solapado)",  0.01, 1.0, 0.45, step=0.01)

    st.divider()
    st.subheader("💾 Resultados")
    save_local = st.checkbox("Guardar en servidor automáticamente", value=False)
    SAVE_PATH = "resultados_servidor"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    if save_local and os.path.exists(SAVE_PATH):
        archivos = os.listdir(SAVE_PATH)
        if archivos:
            st.caption(f"📂 {len(archivos)} archivo(s) guardado(s) en el servidor")

    st.divider()
    st.subheader("🎨 Leyenda")
    st.markdown("🟥 &nbsp; Flor / Hojas (Cajas)", unsafe_allow_html=True)
    st.markdown("🟦 &nbsp; Maleza (Áreas Azules)", unsafe_allow_html=True)
    st.markdown("🟧 &nbsp; Seguridad (Naranja)", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UTILIDADES DE PROCESAMIENTO  (espejo exacto de interfaz_alcachofa.py)
# ─────────────────────────────────────────────

def aplicar_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    CLAHE sobre imagen RGB numpy.
    - Con cv2: usa LAB + CLAHE real (idéntico a interfaz_alcachofa.py)
    - Sin cv2: ecualización de histograma adaptativa con numpy como fallback
    Devuelve siempre RGB numpy.
    """
    if CV2_OK:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        resultado = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
    else:
        # Fallback numpy: equalización de luminancia
        img_f = img_rgb.astype(np.float32)
        r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        hist, _ = np.histogram(lum.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        lut = np.round((cdf - cdf_min) / (lum.size - cdf_min) * 255).astype(np.uint8)
        factor = np.clip(lut[lum.astype(np.uint8)] / (lum + 1e-6), 0.5, 1.8)
        return np.clip(img_f * factor[:, :, np.newaxis], 0, 255).astype(np.uint8)


def check_overlap(b1, b2, margin=15):
    """Igual que en interfaz_alcachofa.py."""
    return not (b1[2] + margin < b2[0] or
                b1[0] - margin > b2[2] or
                b1[3] + margin < b2[1] or
                b1[1] - margin > b2[3])


def fusionar_cajas(cajas_por_clase: dict) -> list:
    """
    Fusiona cajas solapadas dentro de cada clase.
    Algoritmo idéntico al de interfaz_alcachofa.py.
    Devuelve [(cls, x1, y1, x2, y2, conf), ...]
    """
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
                        nx1 = min(caja[0], otra[0])
                        ny1 = min(caja[1], otra[1])
                        nx2 = max(caja[2], otra[2])
                        ny2 = max(caja[3], otra[3])
                        nconf = max(caja[4], otra[4])
                        lista_cajas[i] = [nx1, ny1, nx2, ny2, nconf]
                        merged = True
                        cambio = True
                        break
                if not merged:
                    nuevas_cajas.append(caja)
            lista_cajas = nuevas_cajas

        for cx1, cy1, cx2, cy2, cconf in lista_cajas:
            cajas_finales.append((cls, cx1, cy1, cx2, cy2, cconf))

    return cajas_finales


def _rect_numpy(arr: np.ndarray, x1, y1, x2, y2, color_rgb, thick=2):
    """Dibuja rectángulo sobre array RGB numpy sin cv2."""
    arr[y1:y1+thick, x1:x2] = color_rgb
    arr[y2-thick:y2, x1:x2] = color_rgb
    arr[y1:y2, x1:x1+thick] = color_rgb
    arr[y1:y2, x2-thick:x2] = color_rgb


def _text_pil(draw, x, y, texto, color_rgb):
    """Texto simple con PIL (fallback sin cv2)."""
    from PIL import ImageDraw
    draw.text((x, max(y - 14, 0)), texto, fill=tuple(color_rgb))


def procesar_imagen(imagen_pil: Image.Image,
                    modelo_a, modelo_s,
                    usar_clahe: bool, usar_tta: bool,
                    conf_thr: float, iou_thr: float):
    """
    Pipeline de detección idéntico a AppDeteccion.detectar_objetos().
    YOLO recibe PIL (RGB) para evitar confusión de canales BGR/RGB.
    El dibujo final se hace en numpy RGB.
    """
    img_rgb = np.array(imagen_pil.convert("RGB"))

    # 1. CLAHE → resultado como PIL (RGB) para pasárselo a YOLO correctamente
    if usar_clahe:
        img_clahe_rgb = aplicar_clahe(img_rgb)   # devuelve RGB numpy
        img_para_yolo = Image.fromarray(img_clahe_rgb)  # PIL RGB ✓
    else:
        img_para_yolo = imagen_pil.convert("RGB")       # PIL RGB ✓

    # YOLO acepta PIL images en RGB sin problema (internamente convierte).
    # NO pasar numpy BGR aquí porque en el fallback sin cv2 sería RGB mal interpretado.

    # 2. Detección alcachofa (con TTA e IoU)
    resultados = modelo_a(img_para_yolo,
                          augment=usar_tta,
                          conf=conf_thr,
                          iou=iou_thr,
                          imgsz=800)

    # 3. Detección seguridad — mismas clases que interfaz original
    clases_seg = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 39]
    res_sec = modelo_s(img_para_yolo, conf=0.35, classes=clases_seg)
    detecciones_seguridad = []
    for r in res_sec:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detecciones_seguridad.append(
                [x1, y1, x2, y2, int(box.cls[0]), float(box.conf[0])]
            )

    # 4. Agrupar por clase y fusionar solapadas (merge algorithm)
    cajas_por_clase: dict = {}
    for resultado in resultados:
        if resultado.boxes is not None:
            for caja in resultado.boxes:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                cls = int(caja.cls[0])
                conf = float(caja.conf[0])
                cajas_por_clase.setdefault(cls, []).append([x1, y1, x2, y2, conf])

    cajas_finales = fusionar_cajas(cajas_por_clase)
    cajas_planta = [c for c in cajas_finales if c[0] in [0, 1]]

    # 5. Dibujar sobre imagen ORIGINAL (no sobre la de CLAHE)
    img_dibujo = img_rgb.copy()  # RGB numpy

    # Máscara maleza: 255=teñir azul, 0=protegido
    mask_maleza = np.ones(img_dibujo.shape[:2], dtype=np.uint8) * 255
    for cls, x1, y1, x2, y2, conf in cajas_planta:
        mask_maleza[y1:y2, x1:x2] = 0
    for x1, y1, x2, y2, cls, conf in detecciones_seguridad:
        mask_maleza[y1:y2, x1:x2] = 0  # personas NO se tiñen de azul

    # Tinte azul a maleza (RGB: 0, 0, 255)
    capa_azul = np.zeros_like(img_dibujo)
    capa_azul[:] = (0, 0, 255)
    img_tinte = np.clip(img_dibujo * 0.6 + capa_azul * 0.4, 0, 255).astype(np.uint8)
    img_dibujo[mask_maleza == 255] = img_tinte[mask_maleza == 255]

    if CV2_OK:
        # Una sola conversión RGB→BGR, dibujar todo, luego BGR→RGB
        img_dibujo_bgr = cv2.cvtColor(img_dibujo, cv2.COLOR_RGB2BGR)

        # Cajas de planta (rojo en BGR = 0,0,255)
        for cls, x1, y1, x2, y2, conf in cajas_planta:
            texto = "Flor" if cls == 1 else "Hojas"
            cv2.rectangle(img_dibujo_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_dibujo_bgr, f"{texto} {conf:.2f}",
                        (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Cajas de seguridad (naranja en BGR = 0,140,255)
        nombres_seg = {0: "Persona", 39: "Botella/Basura"}
        for x1, y1, x2, y2, cls, conf in detecciones_seguridad:
            label = nombres_seg.get(cls, "Intruso/Animal")
            cv2.rectangle(img_dibujo_bgr, (x1, y1), (x2, y2), (0, 140, 255), 3)
            cv2.putText(img_dibujo_bgr, f"{label} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        img_dibujo = cv2.cvtColor(img_dibujo_bgr, cv2.COLOR_BGR2RGB)
    else:
        # — Fallback PIL sin cv2 —
        img_pil_draw = Image.fromarray(img_dibujo)
        draw = ImageDraw.Draw(img_pil_draw)
        nombres_seg = {0: "Persona", 39: "Botella/Basura"}
        for cls, x1, y1, x2, y2, conf in cajas_planta:
            texto = "Flor" if cls == 1 else "Hojas"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=2)
            lbl = f"{texto} {conf:.2f}"
            draw.rectangle([x1, max(y1 - 18, 0), x1 + len(lbl) * 7, y1], fill=(255, 50, 50))
            draw.text((x1 + 2, max(y1 - 17, 0)), lbl, fill=(255, 255, 255))
        for x1, y1, x2, y2, cls, conf in detecciones_seguridad:
            label = nombres_seg.get(cls, "Intruso/Animal")
            draw.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=3)
            lbl = f"{label} {conf:.2f}"
            draw.rectangle([x1, max(y1 - 18, 0), x1 + len(lbl) * 7, y1], fill=(255, 140, 0))
            draw.text((x1 + 2, max(y1 - 17, 0)), lbl, fill=(255, 255, 255))
        img_dibujo = np.array(img_pil_draw)

    return (Image.fromarray(img_dibujo), cajas_planta, detecciones_seguridad)


# ─────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────
st.title("🌱 Identificador de Alcachofas y Maleza")
st.write("Sube una imagen desde tu galería o toma una foto directamente con tu celular.")

tab1, tab2 = st.tabs(["📁 Subir Imagen", "📷 Cámara del Celular"])
upload_img = None

with tab1:
    upload_img = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

with tab2:
    cam_img = st.camera_input("Toma una foto con tu cámara")
    if cam_img:
        upload_img = cam_img

if upload_img:
    imagen_pil = Image.open(upload_img).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Imagen Original")
        st.image(imagen_pil, use_container_width=True)

    if st.button("🚀 INICIAR IDENTIFICACIÓN"):
        if modelo_alc is None:
            st.error("❌ Modelo no disponible. Sube el archivo .pt en el panel lateral.")
        else:
            with st.spinner("🔍 Analizando con YOLO v8..."):
                img_resultado, cajas_planta, detecciones_seg = procesar_imagen(
                    imagen_pil,
                    modelo_alc, modelo_seg,
                    usar_clahe=use_clahe,
                    usar_tta=use_tta,
                    conf_thr=conf_val,
                    iou_thr=iou_val
                )

                n_hojas  = sum(1 for c in cajas_planta if c[0] == 0)
                n_flores = sum(1 for c in cajas_planta if c[0] == 1)
                n_seg    = len(detecciones_seg)

                with col2:
                    st.subheader("✅ Resultado del Análisis")
                    st.image(img_resultado, use_container_width=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("🍃 Hojas",    n_hojas)
                    m2.metric("🌸 Flores",   n_flores)
                    m3.metric("⚠️ Alertas", n_seg)

                    buf = io.BytesIO()
                    img_resultado.save(buf, format="JPEG", quality=95)
                    st.download_button(
                        "💾 Descargar Resultado",
                        buf.getvalue(),
                        "resultado_alcachofa.jpg",
                        "image/jpeg"
                    )

                if save_local:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = os.path.join(SAVE_PATH, f"detec_{ts}.jpg")
                    img_resultado.save(fname, quality=95)
                    st.sidebar.success(f"💾 Guardado: detec_{ts}.jpg")

# --- PIE DE PÁGINA ---
st.markdown("---")
st.caption("Sistema de identificación agrícola de precisión con YOLO v8.")

