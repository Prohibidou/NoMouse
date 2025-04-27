import cv2 # type: ignore
import mediapipe as mp # type: ignore
import pyautogui
import numpy as np
import time # Para calcular FPS (opcional)

# --- Configuración Inicial ---
wCam, hCam = 640, 480 # Ancho y alto de la ventana de la cámara
frameR = 100 # Reducción del marco para el área de movimiento (evita llegar a los bordes)
smoothening = 7 # Factor de suavizado para el movimiento del cursor

pTime = 0 # Tiempo anterior para cálculo de FPS
plocX, plocY = 0, 0 # Ubicación anterior del cursor (para suavizado)
clocX, clocY = 0, 0 # Ubicación actual del cursor (para suavizado)

# Inicializar Cámara
cap = cv2.VideoCapture(0) # 0 suele ser la cámara web integrada
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
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
while True:
    # 1. Leer un fotograma de la cámara
    success, img = cap.read()
    if not success:
        print("Ignorando fotograma vacío de la cámara.")
        continue

    # Voltear la imagen horizontalmente para efecto espejo
    img = cv2.flip(img, 1)

    # 2. Convertir la imagen a RGB (Mediapipe usa RGB)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Procesar la imagen para detectar manos
    results = hands.process(imgRGB)

    # 4. Si se detectan manos (landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # Iterar sobre las manos detectadas (aunque configuramos para 1)

            # --- Control de Movimiento ---
            # Obtener coordenadas de landmarks específicos (ej. punta del índice y pulgar)
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                # lm.x, lm.y son coordenadas normalizadas (0 a 1). Convertirlas a píxeles.
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                # Dibujar landmarks (opcional, para visualización)
                # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # Dibuja todos

            if len(lmList) != 0:
                # Obtener coordenadas de la punta del dedo índice (landmark 8)
                x1, y1 = lmList[8][1], lmList[8][2]
                # Obtener coordenadas de la punta del dedo medio (landmark 12) - Otra opción para clic
                x2, y2 = lmList[12][1], lmList[12][2] # Usaremos dedo medio como referencia para clic

                # Dibujar un círculo en la punta del índice (feedback visual)
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

                # 5. Mapear coordenadas de la cámara a la pantalla
                # Interpolar las coordenadas de la cámara (dentro del frameR) a las coordenadas de la pantalla
                # screenX = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
                # screenY = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))

                # Mapeo más directo (ajustar frameR si es necesario)
                # Definir un área activa en la cámara
                active_x_start = frameR
                active_x_end = wCam - frameR
                active_y_start = frameR
                active_y_end = hCam - frameR

                # Asegurarse que el dedo esté dentro del área activa para mapear
                if active_x_start <= x1 <= active_x_end and active_y_start <= y1 <= active_y_end:

                    # Mapeo lineal
                    screenX = int(np.interp(x1, (active_x_start, active_x_end), (0, wScreen)))
                    screenY = int(np.interp(y1, (active_y_start, active_y_end), (0, hScreen)))

                    # 6. Suavizar el movimiento
                    clocX = plocX + (screenX - plocX) / smoothening
                    clocY = plocY + (screenY - plocY) / smoothening

                    # 7. Mover el mouse (usando coordenadas suavizadas)
                    # PyAutoGUI puede fallar si intenta mover a (0,0) o extremos a veces, añadir un pequeño control
                    try:
                        pyautogui.moveTo(wScreen - clocX, clocY) # Invertir X por el flip
                    except pyautogui.FailSafeException:
                         print("FailSafe activado (cursor en esquina superior izquierda).")
                    plocX, plocY = clocX, clocY # Actualizar ubicación anterior

                    # Dibujar el área activa (opcional)
                    cv2.rectangle(img, (active_x_start, active_y_start), (active_x_end, active_y_end), (0, 255, 0), 2)


                    # --- Detección de Clic ---
                    # 8. Calcular distancia entre punta del índice (8) y punta del medio (12)
                    # length = np.hypot(x2 - x1, y2 - y1)
                    # print(length) # Imprimir distancia para calibrar el umbral

                    # Obtener coordenadas de la punta del pulgar (landmark 4)
                    x_thumb, y_thumb = lmList[4][1], lmList[4][2]
                    length = np.hypot(x_thumb - x1, y_thumb - y1) # Distancia Índice-Pulgar

                    # Dibujar línea entre dedos y círculo si están cerca (feedback visual)
                    cv2.line(img, (x1, y1), (x_thumb, y_thumb), (255, 0, 0), 3)


                    # 9. Si la distancia es corta, simular clic izquierdo
                    click_threshold = 35 # ¡¡¡ESTE VALOR NECESITA AJUSTE!!! Depende de la distancia a la cámara, tamaño mano, etc.
                    if length < click_threshold:
                        cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED) # Círculo verde al hacer clic
                        # ¡Cuidado! Esto hará clic repetidamente mientras los dedos estén juntos.
                        # Se podría añadir lógica para hacer clic solo una vez por gesto.
                        pyautogui.click()
                        # Podrías añadir un pequeño 'sleep' para evitar clics demasiado rápidos,
                        # o mejor aún, detectar el *cambio* de estado (dedos separados -> juntos)
                        # time.sleep(0.1) # Pequeña pausa


    # --- Visualización (Opcional) ---
    # Calcular y mostrar FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Mostrar la imagen procesada
    cv2.imshow("Control de Mouse por Vision", img)

    # --- Salir ---
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Limpieza ---
cap.release()
cv2.destroyAllWindows()