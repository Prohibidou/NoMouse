import cv2 # type: ignore
import mediapipe as mp # type: ignore
import pyautogui
import numpy as np
import time # Para calcular FPS (opcional)
import math # Para calcular la distancia

# --- Configuración Inicial ---
wCam, hCam = 640, 480 # Ancho y alto de la ventana de la cámara

# Área de la pantalla utilizable en la cámara (para evitar llegar a los bordes)
# Un valor MENOR en frameR_margin aumenta el área activa y la sensibilidad del mouse (más velocidad).
# Ajustado para mayor velocidad/sensibilidad
frameR_margin = 60 # Reducción del marco (pixeles desde cada borde) - Disminuido para más velocidad

# Factor de suavizado para el movimiento del cursor. Mayor valor = más suave pero más lento.
# Aumentado para más precisión (menos temblor)
smoothening = 10 # Factor de suavizado aumentado

pTime = 0 # Tiempo anterior para cálculo de FPS
plocX, plocY = 0, 0 # Ubicación anterior del cursor (para suavizado)
clocX, clocY = 0, 0 # Ubicación actual del cursor (para suavizado)

# Sensibilidad del clic (distancia entre pulgar e índice)
# Ajusta este valor. Un valor más bajo requiere que los dedos estén más juntos para hacer clic.
click_threshold = 40 # Distancia en pixeles, ajustar según necesidad

# Variable para controlar el estado del clic (evitar clics múltiples)
is_clicking = False # Inicialmente no estamos haciendo clic

# Inicializar Cámara
cap = cv2.VideoCapture(0) # 0 suele ser la cámara web integrada
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    # Intenta con otro índice si el 0 falla
    print("Intentando con índice 1...")
    cap = cv2.VideoCapture(1)
    time.sleep(1) # Espera un poco
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara con índice 1 tampoco.")
        print("Por favor, verifica si la cámara está conectada, no está en uso por otra aplicación y los drivers están instalados.")
        exit() # Sale si no se abre con 0 ni 1

cap.set(3, wCam) # Establecer ancho
cap.set(4, hCam) # Establecer alto

# Inicializar Mediapipe Hands
mpHands = mp.solutions.hands
# Parámetros: static_image_mode=False (para video), max_num_hands=1 (detectar solo una mano),
# min_detection_confidence=0.7 (confianza mínima para detectar), min_tracking_confidence=0.5 (confianza mínima para seguir)
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils # Utilidad para dibujar landmarks

# Obtener dimensiones de la pantalla
wScreen, hScreen = pyautogui.size()
# print(f"Resolución de pantalla: {wScreen}x{hScreen}")

# --- Bucle Principal ---
try:
    while True:
        # 1. Leer un fotograma de la cámara
        success, img = cap.read()
        if not success:
            print("Ignorando fotograma vacío de la cámara.")
            continue

        # No volteamos la imagen para que el movimiento del mouse sea directo
        # img = cv2.flip(img, 1) # Esta línea está comentada/eliminada

        # 2. Convertir la imagen a RGB (Mediapipe usa RGB)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Procesar la imagen para detectar manos
        results = hands.process(imgRGB)

        # Lista para almacenar coordenadas de landmarks importantes
        lmList = []

        # 4. Si se detectan manos (landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks: # Iterar sobre las manos detectadas (aunque configuramos para 1)

                # --- Control de Movimiento y Clic ---
                # Obtener coordenadas de landmarks específicos
                for id, lm in enumerate(handLms.landmark):
                    # lm.x, lm.y son coordenadas normalizadas (0 a 1). Convertirlas a píxeles.
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                    # Dibujar landmarks (opcional, para visualización)
                    # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # Dibuja todos

                if len(lmList) != 0:
                    # Obtener coordenadas de la punta del dedo índice (landmark 8) - Para movimiento y clic
                    x1, y1 = lmList[8][1], lmList[8][2]
                    # Obtener coordenadas de la punta del pulgar (landmark 4) - Para clic
                    x2, y2 = lmList[4][1], lmList[4][2]

                    # Dibujar círculos en las puntas del índice y pulgar (feedback visual)
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED) # Índice (morado)
                    cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)   # Pulgar (azul)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3) # Línea entre índice y pulgar


                    # 5. Mapear coordenadas de la cámara a la pantalla
                    # Definir un área activa en la cámara usando frameR_margin
                    active_x_start = frameR_margin
                    active_x_end = wCam - frameR_margin
                    active_y_start = frameR_margin
                    active_y_end = hCam - frameR_margin

                    # Dibujar el área activa en la ventana de la cámara
                    cv2.rectangle(img, (active_x_start, active_y_start), (active_x_end, active_y_end), (255, 0, 255), 2)

                    # Asegurarse que el dedo índice esté dentro del área activa para mapear
                    if active_x_start <= x1 <= active_x_end and active_y_start <= y1 <= active_y_end:

                        # Mapeo lineal de las coordenadas del índice dentro del área activa a la pantalla completa
                        # Invertimos el rango de salida para la coordenada X para movimiento directo
                        screenX = int(np.interp(x1, (active_x_start, active_x_end), (wScreen, 0))) # Rango de salida invertido
                        screenY = int(np.interp(y1, (active_y_start, active_y_end), (0, hScreen)))

                        # 6. Suavizar el movimiento
                        clocX = plocX + (screenX - plocX) / smoothening
                        clocY = plocY + (screenY - plocY) / smoothening

                        # 7. Mover el mouse (usando coordenadas suavizadas)
                        # PyAutoGUI puede fallar si intenta mover a (0,0) o extremos a veces, añadir un pequeño control
                        try:
                            # Movimiento directo (usamos las coordenadas suavizadas tal cual)
                            pyautogui.moveTo(clocX, clocY)
                        except pyautogui.FailSafeException:
                             print("FailSafe activado (cursor en esquina superior izquierda).")

                        plocX, plocY = clocX, clocY # Actualizar ubicación anterior


                        # --- Detección de Clic ---
                        # 8. Calcular distancia entre punta del índice (8) y punta del pulgar (4)
                        length = math.hypot(x2 - x1, y2 - y1) # Distancia Índice (8) - Pulgar (4)

                        # 9. Si la distancia es corta Y NO estamos ya haciendo clic, simular clic izquierdo
                        if length < click_threshold and not is_clicking:
                            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED) # Círculo verde al hacer clic
                            # Simular clic izquierdo
                            pyautogui.click()
                            print("¡Click!")
                            is_clicking = True # Establecer el estado a True porque acabamos de hacer clic
                            # Pequeña pausa para evitar clics múltiples muy rápidos (opcional con el estado)
                            # time.sleep(0.2) # Puedes ajustar o eliminar si el estado es suficiente

                        # 10. Si la distancia es larga Y estábamos haciendo clic, resetear el estado
                        if length > click_threshold:
                             is_clicking = False # Restablecer el estado para permitir un nuevo clic


        # --- Visualización (Opcional) ---
        # Calcular y mostrar FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (wCam - 150, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2) # Mostrar FPS arriba a la derecha

        # Añadir leyenda "NoMouse App" arriba a la izquierda
        cv2.putText(img, "NoMouse App", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Añadir leyenda "Presiona 'q' para salir" abajo
        cv2.putText(img, "Presiona 'q' para salir", (10, hCam - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)


        # Mostrar la imagen procesada
        cv2.imshow("NoMouse App", img) # Cambiamos el titulo de la ventana

        # --- Salir ---
        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- Limpieza ---
    print("\nCerrando aplicación...")
    cap.release() # Liberar la cámara
    cv2.destroyAllWindows() # Cerrar todas las ventanas de OpenCV
    hands.close() # Cerrar el objeto Hands de MediaPipe
