import cv2
import face_recognition

# Cargar las imágenes de las personas a aprender
imagen_persona1 = face_recognition.load_image_file('assets/images/Oceans-Eleven.jpg')
imagen_persona2 = face_recognition.load_image_file('assets/images/mateo.jpg')
imagen_persona3 = face_recognition.load_image_file('assets/images/mateo2.jpeg')
imagen_persona4 = face_recognition.load_image_file('assets/images/mateo3.jpeg')

# Encode the faces
codificacion_persona1 = face_recognition.face_encodings(imagen_persona1)[0]
codificacion_persona2 = face_recognition.face_encodings(imagen_persona2)[0]
codificacion_persona3 = face_recognition.face_encodings(imagen_persona3)[0]
codificacion_persona4 = face_recognition.face_encodings(imagen_persona4)[0]

# Create a list of known encodings and names
codificaciones_conocidas = [codificacion_persona1, codificacion_persona2, codificacion_persona3, codificacion_persona4]
nombres_conocidos = ['Persona 1', 'Mateo', 'Mateo', 'Mateo']


tolerancia = 0.6

# Inicializar la cámara
camara = cv2.VideoCapture('https://192.168.1.21:8080/video')
frame_width = 640
frame_height = 480
camara.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while True:
    # Leer un fotograma de la cámara
    ret, fotograma = camara.read()

    # Convertir la imagen a RGB para el procesamiento de face_recognition
    fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)

    # Detect the faces in the frame
    ubicaciones = face_recognition.face_locations(fotograma_rgb)
    codificaciones = face_recognition.face_encodings(fotograma_rgb, ubicaciones)

    # Iterate over the detected faces
    for codificacion, ubicacion in zip(codificaciones, ubicaciones):
        # Calculate the face distances to known encodings
        distancias = face_recognition.face_distance(codificaciones_conocidas, codificacion)
        min_distance = min(distancias)
        indice_coincidencia = distancias.argmin()

        # Assign the name if the distance is within tolerance
        if min_distance <= tolerancia:
            nombre = nombres_conocidos[indice_coincidencia]
        else:
            nombre = 'Desconocido'

        # Draw a rectangle and display the name on the frame
        y1, x2, y2, x1 = ubicacion
        cv2.rectangle(fotograma, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(fotograma, nombre, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el fotograma con los nombres reconocidos
    cv2.imshow('Reconocimiento facial', fotograma)

    # Detener el programa si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
camara.release()
cv2.destroyAllWindows()
