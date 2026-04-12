import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import datetime
import io

# --- CONFIGURACIÓN ---
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

# --- CARGA DE MODELOS (Sin cv2) ---
@st.cache_resource
def load_models():
    m_alc = None
    if os.path.exists("best.pt"):
        m_alc = YOLO("best.pt")
    m_seg = YOLO("yolov8n.pt")
    return m_alc, m_seg

modelo_alc, modelo_seg = load_models()

# --- BARRA LATERAL ---
with st.sidebar:
    st.title("🌱 Panel de Control")

    if modelo_alc is None:
        st.warning("⚠️ No se encontró 'best.pt'")
        up_file = st.file_uploader("Sube tu modelo (.pt)", type=["pt"])
        if up_file:
            with open("best.pt", "wb") as f:
                f.write(up_file.getbuffer())
            st.success("Modelo guardado. Actualizando...")
            st.rerun()
    else:
        st.success("✅ Modelo listo")

    conf_val = st.slider("Confianza Mínima", 0.01, 1.0, 0.25)
    save_local = st.checkbox("💾 Guardar resultados en servidor", value=False)
    SAVE_PATH = "resultados_servidor"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

# --- FUNCIÓN DE DIBUJO (Solo PIL, sin cv2) ---
def dibujar_resultado(imagen_pil, resultados_alc, resultados_seg):
    """Dibuja las detecciones usando solo PIL, sin cv2."""
    img = imagen_pil.convert("RGBA")
    ancho, alto = img.size

    # 1. Tinte Azul para Maleza (todo el fondo)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Máscara: todo azul excepto las cajas de planta
    capa_azul = Image.new("RGBA", img.size, (30, 100, 255, 100))

    # Creamos una máscara blanca y "borramos" donde hay plantas
    mask = Image.new("L", img.size, 255)  # Todo blanco = maleza
    draw_mask = ImageDraw.Draw(mask)
    
    for r in resultados_alc:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            draw_mask.rectangle([x1, y1, x2, y2], fill=0)  # Negro = planta (no teñir)

    # Aplicar tinte azul solo donde hay maleza
    img_resultado = Image.composite(
        Image.blend(img.convert("RGBA"), capa_azul, 0.4),
        img,
        mask
    ).convert("RGB")

    # 2. Dibujar cajas de Planta
    draw = ImageDraw.Draw(img_resultado)
    for r in resultados_alc:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            label = f"{'Flor' if cls == 1 else 'Hojas'} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=3)
            draw.rectangle([x1, y1 - 20, x1 + len(label)*8, y1], fill=(255, 50, 50))
            draw.text((x1 + 2, y1 - 18), label, fill=(255, 255, 255))

    # 3. Dibujar cajas de Seguridad
    labels_coco = {0: "Persona", 39: "Basura"}
    for r in resultados_seg:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            label = f"{labels_coco.get(cls, 'Intruso')} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=3)
            draw.rectangle([x1, y1 - 20, x1 + len(label)*8, y1], fill=(255, 140, 0))
            draw.text((x1 + 2, y1 - 18), label, fill=(255, 255, 255))

    return img_resultado

# --- INTERFAZ PRINCIPAL ---
st.title("🌱 Identificador de Alcachofas y Maleza")
st.write("Sube una imagen desde tu galería o toma una foto directamente con tu celular.")

tab1, tab2 = st.tabs(["📁 Subir Imagen", "📷 Cámara"])
upload_img = None

with tab1:
    upload_img = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

with tab2:
    cam_img = st.camera_input("Toma una foto")
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
            st.error("Por favor sube el archivo 'best.pt' en el panel lateral.")
        else:
            with st.spinner("Analizando con YOLO..."):
                # YOLO acepta PIL directamente, sin necesitar cv2
                res_alc = modelo_alc(imagen_pil, conf=conf_val)
                res_seg = modelo_seg(imagen_pil, conf=0.35, classes=[0, 39])

                img_resultado = dibujar_resultado(imagen_pil, res_alc, res_seg)

                with col2:
                    st.subheader("✅ Resultado del Análisis")
                    st.image(img_resultado, use_container_width=True)

                    # Botón de descarga
                    buf = io.BytesIO()
                    img_resultado.save(buf, format="JPEG", quality=95)
                    st.download_button(
                        "💾 Descargar Resultado",
                        buf.getvalue(),
                        "resultado_alcachofa.jpg",
                        "image/jpeg"
                    )

                # Guardar en servidor si se pidió
                if save_local:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = os.path.join(SAVE_PATH, f"detec_{ts}.jpg")
                    img_resultado.save(fname, quality=95)
                    st.sidebar.success(f"Guardado: {fname}")

st.markdown("---")
st.caption("Sistema de identificación agrícola de precisión con YOLO v8.")

