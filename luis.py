import cv2
import dlib
import matplotlib.pyplot as plt

# Cargar los archivos de predicción
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('assets/pretrained_models/shape_predictor_68_face_landmarks.dat')

# Cargar la imagen
imagen = cv2.imread('assets/images/ale.jpg')

# Convertir la imagen a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectar los rostros en la imagen
caras = detector(gris)

# Iterar sobre los rostros detectados
for cara in caras:
    # Obtener los puntos de referencia faciales
    landmarks = predictor(gris, cara)

    # Iterar sobre los puntos de referencia faciales
    for punto in landmarks.parts():
        x, y = punto.x, punto.y

        # Dibujar un círculo en cada punto, grande y rojo
        cv2.circle(imagen, (x, y), 7, (0, 0, 255), -1)


# Mostrar la imagen con los puntos de referencia
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
