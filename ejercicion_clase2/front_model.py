import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

# 1. Configuración inicial de la página
st.set_page_config(page_title="Prueba de Reconocimiento Facial", page_icon="📷")
st.title("Sistema de Reconocimiento Biométrico")
st.write("Sube una imagen o usa tu webcam para evaluar la red neuronal VGG-16.")

# 2. Carga Inteligente del Modelo en Caché (Optimización de Memoria)
@st.cache_resource
def cargar_modelo():
    # Asegúrate de que el archivo .h5 se llame exactamente así y esté en la misma carpeta
    return load_model('modelo_practica.h5')

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo. Verifica que el archivo .h5 exista en esta ruta. Detalles: {e}")
    st.stop()

# 3. Interfaz interactiva: Selector de entrada
opcion = st.radio("Selecciona el método de entrada para la prueba:", ("Usar Webcam", "Subir Archivo"))

if opcion == "Usar Webcam":
    # Activa la cámara web nativa para la Prueba 1
    archivo_subido = st.camera_input("Tómate la foto en vivo para la prueba")
else:
    # Permite subir el archivo para las Pruebas 2 y 3
    archivo_subido = st.file_uploader("Sube la imagen para la prueba", type=["jpg", "jpeg", "png"])

# 4. Flujo de Ejecución y Predicción
if archivo_subido is not None:
    # Renderizamos la imagen en la interfaz web usando PIL (RGB) para visualización correcta
    imagen_mostrar = Image.open(archivo_subido)
    st.image(imagen_mostrar, caption="Imagen capturada para análisis topológico", use_container_width=True)
    
    if st.button("Ejecutar Prueba de Clasificación"):
        with st.spinner('Procesando extracción de características en red VGG-16...'):
            
            # A. PREPROCESAMIENTO MATEMÁTICO (Idéntico al backend con cv2.imread)
            # Asegurarnos de leer el archivo desde el inicio
            archivo_subido.seek(0)
            file_bytes = np.asarray(bytearray(archivo_subido.read()), dtype=np.uint8)
            
            # Decodificamos la imagen en formato BGR (Blue-Green-Red) tal como lo espera el modelo entrenado
            img = cv2.imdecode(file_bytes, 1) 
            
            # Redimensionamos estrictamente a 224x224 píxeles
            img = cv2.resize(img, (224, 224))
            
            # Expandimos dimensiones para que tenga la forma (1, 224, 224, 3)
            img = np.expand_dims(img, axis=0)
            
            # B. INFERENCIA DEL MODELO
            pred = modelo.predict(img)
            
            # Extraemos las dos probabilidades que arroja la capa Softmax
            prob_fondo = float(pred[0][0])
            prob_ivan = float(pred[0][1])
            
            # C. LÓGICA DE CLASIFICACIÓN BINARIA Y VISUALIZACIÓN
            st.markdown("---")
            if prob_fondo > prob_ivan:
                st.warning("⚠️ **RESULTADO:** El modelo lo clasifica como **FONDO**")
                st.info(f"Nivel de confianza - Fondo: {prob_fondo*100:.2f}% | Ivan: {prob_ivan*100:.2f}%")
            else:
                st.success("✅ **RESULTADO:** El modelo detecta a **IVAN**")
                st.info(f"Nivel de confianza - Ivan: {prob_ivan*100:.2f}% | Fondo: {prob_fondo*100:.2f}%")