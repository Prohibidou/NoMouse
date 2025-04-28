import cv2 # type: ignore
import mediapipe as mp # type: ignore
import pyautogui
import numpy as np
import time # Para calcular FPS y deltaTime
import math # Para calcular la distancia

# --- Configuración Inicial ---
wCam, hCam = 640, 480
wScreen, hScreen = pyautogui.size()

# --- AJUSTES REALIZADOS ---
# 1. Reducir el margen para AGRANDAR el recuadro rosa (área activa visual)
frameR_margin = 50 # Valor anterior: 100. Un valor menor hace el recuadro más grande.
# 2. Reducir el suavizado para AUMENTAR la velocidad/respuesta del cursor
smoothening = 5     # Valor anterior: 10. Un valor menor hace que el cursor reaccione más rápido.
# --- FIN DE AJUSTES ---

click_threshold = 40 # Sensibilidad de clic (ajustar según necesidad)
is_clicking = False

# --- Variables para Extrapolación y Re-anclaje ---
pTime = time.time() # Tiempo anterior
plocX, plocY = 0, 0 # Ubicación anterior del cursor (suavizada)
clocX, clocY = 0, 0 # Ubicación actual del cursor (suavizada)
velX, velY = 0, 0   # Velocidad actual estimada del cursor (pixels por frame)
decay_factor = 0.90 # Factor de reducción de velocidad por frame
extrapolation_threshold = 0.5 # Velocidad mínima para seguir extrapolando

# --- Variables de Estado para Re-anclaje ---
hand_was_detected_prev_frame = False # Para detectar la transición de no detectado a detectado
offset_calculated = False            # Flag para saber si ya calculamos el offset en esta aparición
offsetX, offsetY = 0, 0              # El offset a aplicar

# --- Inicialización ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara 0. Intentando con índice 1...")
    cap = cv2.VideoCapture(1)
    time.sleep(1)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara con índice 1 tampoco.")
        exit()

cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
# Aumentar un poco la confianza de detección puede ayudar a evitar detecciones falsas
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# --- Bucle Principal ---
try:
    while True:
        # 1. Leer fotograma y calcular tiempo delta
        success, img = cap.read()
        img = cv2.flip(img, 1) # Voltear horizontalmente para modo espejo
        cTime = time.time()
        deltaTime = cTime - pTime if pTime else 0.01 # Evitar división por cero al inicio
        pTime = cTime
        if not success:
            print("Ignorando fotograma vacío.")
            continue

        # 2. Procesar la imagen
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False # Optimización: Marcar como no escribible antes de procesar
        results = hands.process(imgRGB)
        imgRGB.flags.writeable = True # Volver a marcar como escribible

        hand_detected_this_frame = bool(results.multi_hand_landmarks) # True si se detectó mano

        # --- Área Activa Visual ---
        # Usamos el frameR_margin ajustado para dibujar el recuadro
        active_x_start = frameR_margin
        active_x_end = wCam - frameR_margin
        active_y_start = frameR_margin
        active_y_end = hCam - frameR_margin
        # Dibuja el rectángulo rosa (magenta) más grande ahora
        cv2.rectangle(img, (active_x_start, active_y_start), (active_x_end, active_y_end), (255, 0, 255), 2)

        # 3. Lógica Principal: Detección / Extrapolación / Re-anclaje
        if hand_detected_this_frame:
            handLms = results.multi_hand_landmarks[0] # Asumimos una sola mano
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) >= 9: # Necesitamos al menos índice (8) y pulgar (4)
                x1, y1 = lmList[8][1], lmList[8][2] # Índice
                x2, y2 = lmList[4][1], lmList[4][2] # Pulgar

                # --- Dibujo de Referencia ---
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED) # Índice
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED) # Pulgar (cambié color para diferenciar)

                # --- Calcular Posición Objetivo Cruda (sin offset aún) ---
                # Usa el área activa más grande (menor margen) para la interpolación
                screenX_raw = int(np.interp(x1, (active_x_start, active_x_end), (0, wScreen)))
                screenY_raw = int(np.interp(y1, (active_y_start, active_y_end), (0, hScreen)))

                # --- LÓGICA DE RE-ANCLAJE ---
                if not hand_was_detected_prev_frame:
                    # Calcular el offset entre dónde aparecería el cursor (raw)
                    # y dónde está realmente ahora (después de la extrapolación/última posición)
                    offsetX = screenX_raw - clocX
                    offsetY = screenY_raw - clocY
                    offset_calculated = True
                    print(f"[{time.strftime('%H:%M:%S')}] - Mano reaparecida. Offset calculado: ({offsetX}, {offsetY})")

                    # *** SOLUCIÓN: Resetear plocX/Y a la posición actual ANTES del primer suavizado ***
                    plocX, plocY = clocX, clocY
                    print(f"[{time.strftime('%H:%M:%S')}] - plocX/Y reseteado a clocX/Y: ({plocX:.2f}, {plocY:.2f})")

                # Aplicar el offset si fue calculado para esta aparición
                if offset_calculated:
                    screenX = screenX_raw - offsetX
                    screenY = screenY_raw - offsetY
                else:
                    # Fallback
                    screenX = screenX_raw
                    screenY = screenY_raw
                    if not hand_was_detected_prev_frame: # Doble check por si acaso
                        offsetX = screenX_raw - clocX
                        offsetY = screenY_raw - clocY
                        offset_calculated = True
                        plocX, plocY = clocX, clocY

                # --- Suavizado ---
                # Usa el smoothening ajustado (menor valor = más rápido)
                clocX = plocX + (screenX - plocX) / smoothening
                clocY = plocY + (screenY - plocY) / smoothening

                # --- Calcular Velocidad (para posible futura extrapolación) ---
                if deltaTime > 0.001:
                    velX = (clocX - plocX) # Pixels por frame (aprox)
                    velY = (clocY - plocY)
                else:
                    velX = 0
                    velY = 0

                # --- Mover el Mouse ---
                try:
                    move_x = max(0, min(wScreen - 1, int(clocX)))
                    move_y = max(0, min(hScreen - 1, int(clocY)))
                    pyautogui.moveTo(move_x, move_y)
                except pyautogui.FailSafeException:
                    print("FailSafe activado.")
                    velX, velY = 0, 0 # Detener si salta el failsafe

                # --- Detección de Clic ---
                length = math.hypot(x2 - x1, y2 - y1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0) if length < click_threshold else (255, 0, 0), 3) # Línea cambia color al click
                if length < click_threshold:
                    if not is_clicking:
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED) # Feedback visual clic
                        try:
                            pyautogui.click()
                            print("¡Click!")
                            is_clicking = True # Evitar clicks múltiples
                            velX, velY = 0, 0 # Detener extrapolación al hacer clic
                        except pyautogui.FailSafeException:
                            print("FailSafe activado durante el clic.")
                else:
                    is_clicking = False # Resetear estado de clic

                # Actualizar posición previa para el siguiente frame
                plocX, plocY = clocX, clocY

        else:
            # --- Mano NO Detectada (Extrapolación) ---
            offset_calculated = False # Reseteamos el flag de offset si la mano desaparece
            if abs(velX) > extrapolation_threshold or abs(velY) > extrapolation_threshold:
                # Aplicar decaimiento (fricción)
                velX *= decay_factor
                velY *= decay_factor

                # Actualizar posición extrapolada
                extrapolatedX = clocX + velX
                extrapolatedY = clocY + velY

                # Asegurarse de que el cursor permanezca dentro de la pantalla
                clocX = max(0, min(wScreen - 1, extrapolatedX))
                clocY = max(0, min(hScreen - 1, extrapolatedY))

                # Mover el mouse a la posición extrapolada
                try:
                    pyautogui.moveTo(int(clocX), int(clocY))
                    cv2.putText(img, "Extrapolando...", (wCam // 2 - 100, hCam - 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
                except pyautogui.FailSafeException:
                    print("FailSafe activado durante extrapolación.")
                    velX, velY = 0, 0 # Detener si salta el failsafe

                # Actualizar la posición 'previa' para el siguiente paso
                plocX, plocY = clocX, clocY

            else:
                # La velocidad es muy baja, detener completamente la extrapolación
                velX = 0
                velY = 0

        # --- Actualizar estado para el próximo frame ---
        hand_was_detected_prev_frame = hand_detected_this_frame

        # --- Visualización (FPS, etc.) ---
        fps = 1 / deltaTime if deltaTime > 0 else 0
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, "Presiona 'q' para salir", (20, hCam - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        # Mostrar la imagen
        cv2.imshow("NoMouse App", img)

        # --- Salir ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- Limpieza ---
    print("\nCerrando aplicación...")
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if 'hands' in locals() and hands:
        pass # No hay método close() explícito para Hands en MediaPipe