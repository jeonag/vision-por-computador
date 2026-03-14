# Cargamos el modelo desde un archivo formato .h5
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = load_model('modelo_practica.h5')

# Leemos una imagen nunca vista por el modelo utilizando OpenCV
# img = cv2.imread('testing/testing_1.jpg')
# img = cv2.imread('testing/testing_2.jpg')
# img = cv2.imread('testing/famoso.jpeg')
# img = cv2.imread('testing/famoso_4.jpg')
img = cv2.imread('testing/famoso_5.jpg')

# Redimensionamos la imagen a 224x224 píxeles para que coincida con la dimensión de entrada del modelo VGG-16
img = cv2.resize(img, (224, 224))

# Expandimos las dimensiones de la imagen para que tenga la forma (1, 224, 224, 3)
# Esto es necesario porque el modelo espera un lote de imágenes como entrada
img = np.expand_dims(img, axis=0)

# Realizamos la predicción utilizando el modelo cargado
pred = model.predict(img)

# Imprimimos la predicción
print(pred)

# Comprobamos cuál de las dos probabilidades es mayor en la predicción
if pred[0][0] > pred[0][1]:
    # Si la probabilidad de que sea un gato es mayor, imprimimos 'Gato'
    plt.title('Fondo')
    print('Fondo')
else:
    # Si la probabilidad de que sea un perro es mayor, imprimimos 'Perro'
    plt.title('Ivan')
    print('Ivan')