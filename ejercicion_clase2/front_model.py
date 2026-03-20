import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import os

# ---------------------------
# CONFIGURACION DE LA PAGINA
# ---------------------------
st.set_page_config(page_title="Reconocimiento de rostros", page_icon="📷")

st.title("📷 Sistema de Reconocimiento Biométrico")
st.markdown("Sube una imagen o usa tu webcam para evaluar la red neuronal.")

# ---------------------------
# CARGA DEL MODELO (ruta relativa para evitar errores de carga)
# ---------------------------
@st.cache_resource
def cargar_modelo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(base_dir, 'modelo_practica.h5')
    return load_model(ruta_modelo)

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# ---------------------------
# HISTORIAL
# ---------------------------
if "historial" not in st.session_state:
    st.session_state.historial = []

# ---------------------------
# ENTRADA DE DATOS
# ---------------------------
opcion = st.radio("Selecciona el método de entrada:", ("Usar Webcam", "Subir Archivo"))

if opcion == "Usar Webcam":
    archivo_subido = st.camera_input("Tómate una foto")
else:
    archivo_subido = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# ---------------------------
# PROCESAMIENTO
# ---------------------------
if archivo_subido is not None:
    
    col1, col2 = st.columns(2)

    # Mostrar imagen original
    with col1:
        imagen_mostrar = Image.open(archivo_subido)
        st.image(imagen_mostrar, caption="Imagen original", use_container_width=True)

    if st.button("Ejecutar Clasificación"):

        with st.spinner("Procesando imagen..."):

            archivo_subido.seek(0)
            file_bytes = np.asarray(bytearray(archivo_subido.read()), dtype=np.uint8)

            img = cv2.imdecode(file_bytes, 1)

            if img is None:
                st.error("❌ No se pudo procesar la imagen")
                st.stop()

            # Redimensionamos estrictamente a 224x224 píxeles
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            
            pred = modelo.predict(img)

            prob_fondo = float(pred[0][0])
            prob_ivan = float(pred[0][1])

            # ---------------------------
            # LÓGICA DE DECISIÓN
            # ---------------------------
            threshold = 0.7

            if prob_ivan > prob_fondo:
                label = "IVAN"
                confidence = prob_ivan
            else:
                label = "FONDO"
                confidence = prob_fondo

            # Guardar en historial
            st.session_state.historial.append((label, confidence))

            # ---------------------------
            # MOSTRAR RESULTADO
            # ---------------------------
            with col2:
                st.subheader("🔍 Resultado")

                if confidence < threshold:
                    st.warning("⚠️ El modelo no está seguro de la predicción")
                elif label == "IVAN":
                    st.success(f"✅ Detectado: {label}")
                else:
                    st.warning(f"⚠️ Detectado: {label}")

                st.write(f"Confianza: {confidence*100:.2f}%")

                # Barras de probabilidad
                st.markdown("### Probabilidades")
                st.write(f"Ivan: {prob_ivan*100:.2f}%")
                st.progress(int(prob_ivan * 100))

                st.write(f"Fondo: {prob_fondo*100:.2f}%")
                st.progress(int(prob_fondo * 100))
                
# ---------------------------
# HISTORIAL VISUAL
# ---------------------------
st.markdown("---")
st.subheader("Historial de predicciones")

if st.session_state.historial:
    st.table(st.session_state.historial)
else:
    st.write("Sin predicciones aún")

# ---------------------------
# DETALLES TECNICOS
# ---------------------------
with st.expander("⚙️ Ver detalles técnicos"):
    # ---------------------------
    # MÉTRICAS DE CLASIFICACIÓN
    # ---------------------------
    st.markdown("## Métricas de clasificación")

    try:
        st.write("### Probabilidades (salida del modelo)")
        st.write(f"Fondo: {prob_fondo:.4f}")
        st.write(f"Ivan: {prob_ivan:.4f}")

        st.write("### Decisión del modelo")
        st.write(f"Clase predicha: {label}")
        st.write(f"Confianza: {confidence*100:.2f}%")

        st.write("### Parámetros de decisión")
        st.write(f"Threshold aplicado: {threshold}")

        st.write("### Vector crudo de salida")
        st.write(pred)

        st.write("### Forma de la entrada")
        st.write(f"Input shape: {img.shape}")

    except:
        st.info("Ejecuta una predicción para ver las métricas")