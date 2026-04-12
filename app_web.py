import streamlit as st
import numpy as np
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

# =====================================================================
# IMPORTANTE: Reemplaza "TU_USUARIO" con tu usuario real de Hugging Face
# Ejemplo: si tu usuario es "emanuelquiroga", queda:
#   repo_id="emanuelquiroga/alcachofa-model"
# =====================================================================
HF_REPO_ID = "Emanuel1102/alcachofa-model"
HF_FILENAME = "best.pt"

# --- CARGA DE MODELOS ---
@st.cache_resource
def load_models():
    m_alc = None

    # Si ya fue descargado antes en esta sesión, usarlo directamente
    if os.path.exists("best.pt"):
        try:
            m_alc = YOLO("best.pt")
        except Exception as e:
            st.sidebar.warning(f"Modelo local caído: {e}")

    # Si no existe, descargar desde Hugging Face
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

    # Modelo de seguridad (personas/basura) - se descarga automáticamente
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
    conf_val = st.slider("Confianza Mínima", 0.01, 1.0, 0.25)
    use_clahe = st.checkbox("Filtro CLAHE (Mejora de Luz)", value=True)

    st.divider()
    st.subheader("💾 Resultados")
    save_local = st.checkbox("Guardar en servidor automáticamente", value=False)
    SAVE_PATH = "resultados_servidor"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Mostrar archivos guardados
    if save_local and os.path.exists(SAVE_PATH):
        archivos = os.listdir(SAVE_PATH)
        if archivos:
            st.caption(f"📂 {len(archivos)} archivo(s) guardado(s) en el servidor")


# --- FUNCIÓN DE MEJORA CLAHE (sin cv2) ---
def aplicar_clahe_pil(imagen_pil):
    """Mejora de contraste adaptativo usando solo PIL y numpy."""
    img_array = np.array(imagen_pil).astype(np.float32)
    # Convertir a LAB manualmente (aproximación con numpy)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    luminancia = 0.299*r + 0.587*g + 0.114*b
    # Ecualización por histograma simple
    hist, bins = np.histogram(luminancia.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    total_pixels = luminancia.size
    lut = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255).astype(np.uint8)
    factor = np.clip(lut[luminancia.astype(np.uint8)] / (luminancia + 1e-6), 0.5, 1.8)
    img_mejorada = np.clip(img_array * factor[:,:,np.newaxis], 0, 255).astype(np.uint8)
    return Image.fromarray(img_mejorada)


# --- FUNCIÓN DE DIBUJO (Solo PIL, sin cv2) ---
def dibujar_resultado(imagen_pil, resultados_alc, resultados_seg):
    """Dibuja detecciones y tinte de maleza usando solo PIL."""
    img = imagen_pil.convert("RGB")

    # 1. Tinte Azul para Maleza (todo excepto las cajas de planta)
    capa_azul = Image.new("RGB", img.size, (30, 100, 255))
    mask = Image.new("L", img.size, 255)  # 255 = maleza (teñir)
    draw_mask = ImageDraw.Draw(mask)

    for r in resultados_alc:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            draw_mask.rectangle([x1, y1, x2, y2], fill=0)  # 0 = planta (no teñir)

    img_resultado = Image.composite(
        Image.blend(img, capa_azul, 0.4),
        img,
        mask
    )

    # 2. Dibujar cajas de Planta (Rojo)
    draw = ImageDraw.Draw(img_resultado)
    for r in resultados_alc:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            label = f"{'Flor' if cls == 1 else 'Hojas'} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=3)
            draw.rectangle([x1, max(y1 - 20, 0), x1 + len(label) * 8, y1], fill=(255, 50, 50))
            draw.text((x1 + 2, max(y1 - 18, 0)), label, fill=(255, 255, 255))

    # 3. Dibujar cajas de Seguridad (Naranja)
    labels_coco = {0: "Persona", 39: "Basura"}
    for r in resultados_seg:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            label = f"{labels_coco.get(cls, 'Intruso')} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=3)
            draw.rectangle([x1, max(y1 - 20, 0), x1 + len(label) * 8, y1], fill=(255, 140, 0))
            draw.text((x1 + 2, max(y1 - 18, 0)), label, fill=(255, 255, 255))

    return img_resultado


# --- INTERFAZ PRINCIPAL ---
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

                # Aplicar CLAHE si está activado
                img_para_analizar = aplicar_clahe_pil(imagen_pil) if use_clahe else imagen_pil

                # Predicciones
                res_alc = modelo_alc(img_para_analizar, conf=conf_val)
                res_seg = modelo_seg(img_para_analizar, conf=0.35, classes=[0, 39])

                # Dibujar resultado
                img_resultado = dibujar_resultado(imagen_pil, res_alc, res_seg)

                # Conteo de detecciones
                n_hojas = sum(1 for r in res_alc for b in r.boxes if int(b.cls[0]) == 0)
                n_flores = sum(1 for r in res_alc for b in r.boxes if int(b.cls[0]) == 1)
                n_seg = sum(len(r.boxes) for r in res_seg)

                with col2:
                    st.subheader("✅ Resultado del Análisis")
                    st.image(img_resultado, use_container_width=True)

                    # Métricas
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🍃 Hojas", n_hojas)
                    m2.metric("🌸 Flores", n_flores)
                    m3.metric("⚠️ Alertas", n_seg)

                    # Botón de descarga
                    buf = io.BytesIO()
                    img_resultado.save(buf, format="JPEG", quality=95)
                    st.download_button(
                        "💾 Descargar Resultado",
                        buf.getvalue(),
                        "resultado_alcachofa.jpg",
                        "image/jpeg"
                    )

                # Guardar en servidor automáticamente
                if save_local:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = os.path.join(SAVE_PATH, f"detec_{ts}.jpg")
                    img_resultado.save(fname, quality=95)
                    st.sidebar.success(f"💾 Guardado: detec_{ts}.jpg")

# --- PIE DE PÁGINA ---
st.markdown("---")
col_a, col_b, col_c = st.columns(3)
col_a.markdown("**Leyenda:**")
col_b.markdown("🟥 Flor / Hojas detectadas")
col_c.markdown("🟦 Área de Maleza")
st.caption("Sistema de identificación agrícola de precisión con YOLO v8.")


