import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- Configuration Constants ---
# Smoothing factor for cursor movement (0.0 < SMOOTHING_FACTOR <= 1.0)
# Lower values mean more smoothing but more lag. Higher values are more responsive but jittery.
SMOOTHING_FACTOR = 0.2

# Sensitivity adjustment for eye movement mapping.
# You might need to tweak these if the cursor doesn't cover the whole screen
# or moves too erratically. These are scaling factors for the normalized gaze.
# For example, if your eye's normalized movement is only from 0.3 to 0.7,
# you might want to expand this range.
# For now, we'll use a direct mapping and you can experiment.
X_SENSITIVITY_SCALE = 1.0  # Multiplies the normalized X gaze
Y_SENSITIVITY_SCALE = 1.0  # Multiplies the normalized Y gaze

# Optional: Define a deadzone for gaze to prevent minor jitters when looking straight.
# Values are normalized (0.0 to 1.0).
# GAZE_DEADZONE_X = 0.05 # e.g., if abs(norm_x - 0.5) < GAZE_DEADZONE_X, consider it center
# GAZE_DEADZONE_Y = 0.05

# --- Initialization ---
def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Assuming one user
        refine_landmarks=True,  # This gives us iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles # For drawing styles if needed

    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    print(f"Screen dimensions: {screen_width}x{screen_height}")

    # Initialize OpenCV VideoCapture
    cap = cv2.VideoCapture(0) # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Variables for smoothed cursor position
    # Initialize to center of the screen
    smooth_cursor_x, smooth_cursor_y = screen_width / 2, screen_height / 2

    print("Starting eye gaze mouse control. Press 'q' to quit.")

    # --- Main Loop ---
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # This makes movements more intuitive.
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True # Not strictly necessary if not drawing on image_rgb

        # Convert back to BGR for OpenCV drawing
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        img_h, img_w, _ = image_bgr.shape # Get image dimensions for landmark scaling

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # --- Eye Landmark Indices (from MediaPipe documentation/diagrams) ---
                # These are approximate pupil centers from the refined landmarks set
                # For left eye (user's left, image right if not flipped)
                # If image is flipped, left eye is on the left side of the image.
                LEFT_PUPIL_CENTER_LM = 473 # Index for the center of the left iris
                # Left eye corners and vertical limits
                LEFT_EYE_OUTER_CORNER_LM = 33
                LEFT_EYE_INNER_CORNER_LM = 133
                LEFT_EYE_TOP_LM = 159
                LEFT_EYE_BOTTOM_LM = 145

                # For right eye (user's right, image left if not flipped)
                RIGHT_PUPIL_CENTER_LM = 468 # Index for the center of the right iris
                # Right eye corners and vertical limits
                RIGHT_EYE_OUTER_CORNER_LM = 263
                RIGHT_EYE_INNER_CORNER_LM = 362
                RIGHT_EYE_TOP_LM = 386
                RIGHT_EYE_BOTTOM_LM = 374

                # --- Extract Landmark Coordinates ---
                # Note: landmark coordinates are normalized (0.0-1.0) by image width/height.
                # We'll convert them to pixel coordinates for some calculations if needed,
                # but for normalization, relative positions are fine.

                try:
                    # Left Eye
                    lm_left_pupil = face_landmarks.landmark[LEFT_PUPIL_CENTER_LM]
                    lm_left_outer = face_landmarks.landmark[LEFT_EYE_OUTER_CORNER_LM]
                    lm_left_inner = face_landmarks.landmark[LEFT_EYE_INNER_CORNER_LM]
                    lm_left_top = face_landmarks.landmark[LEFT_EYE_TOP_LM]
                    lm_left_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM_LM]

                    # Right Eye
                    lm_right_pupil = face_landmarks.landmark[RIGHT_PUPIL_CENTER_LM]
                    lm_right_outer = face_landmarks.landmark[RIGHT_EYE_OUTER_CORNER_LM]
                    lm_right_inner = face_landmarks.landmark[RIGHT_EYE_INNER_CORNER_LM]
                    lm_right_top = face_landmarks.landmark[RIGHT_EYE_TOP_LM]
                    lm_right_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM_LM]

                    # --- Calculate Normalized Pupil Position for Each Eye ---
                    # Horizontal: (pupil_x - outer_x) / (inner_x - outer_x)
                    #   0 = looking far left (pupil at outer corner)
                    #   1 = looking far right (pupil at inner corner)
                    # Vertical: (pupil_y - top_y) / (bottom_y - top_y)
                    #   0 = looking far up (pupil at top)
                    #   1 = looking far down (pupil at bottom)

                    # Denominators for safety (prevent division by zero)
                    denom_x_left = (lm_left_inner.x - lm_left_outer.x)
                    denom_y_left = (lm_left_bottom.y - lm_left_top.y)
                    denom_x_right = (lm_right_inner.x - lm_right_outer.x)
                    denom_y_right = (lm_right_bottom.y - lm_right_top.y)

                    norm_pupil_x_left, norm_pupil_y_left = 0.5, 0.5 # Default to center
                    norm_pupil_x_right, norm_pupil_y_right = 0.5, 0.5

                    if abs(denom_x_left) > 1e-6: # Check for very small denominator
                        norm_pupil_x_left = (lm_left_pupil.x - lm_left_outer.x) / denom_x_left
                    if abs(denom_y_left) > 1e-6:
                        norm_pupil_y_left = (lm_left_pupil.y - lm_left_top.y) / denom_y_left
                    
                    if abs(denom_x_right) > 1e-6:
                        norm_pupil_x_right = (lm_right_pupil.x - lm_right_outer.x) / denom_x_right
                    if abs(denom_y_right) > 1e-6:
                        norm_pupil_y_right = (lm_right_pupil.y - lm_right_top.y) / denom_y_right

                    # Clamp values to [0, 1] range as pupil might go slightly outside landmarks
                    norm_pupil_x_left = np.clip(norm_pupil_x_left, 0.0, 1.0)
                    norm_pupil_y_left = np.clip(norm_pupil_y_left, 0.0, 1.0)
                    norm_pupil_x_right = np.clip(norm_pupil_x_right, 0.0, 1.0)
                    norm_pupil_y_right = np.clip(norm_pupil_y_right, 0.0, 1.0)
                    
                    # Average the normalized positions from both eyes
                    # (Could add logic here to use only one eye if the other is occluded/unreliable)
                    avg_norm_pupil_x = (norm_pupil_x_left + norm_pupil_x_right) / 2.0
                    avg_norm_pupil_y = (norm_pupil_y_left + norm_pupil_y_right) / 2.0
                    
                    # --- Map Normalized Gaze to Screen Coordinates ---
                    # This is a direct mapping. Calibration would make this much more accurate.
                    # The X_SENSITIVITY_SCALE and Y_SENSITIVITY_SCALE can be used to tune this.
                    # A common issue is that the normalized pupil movement (e.g., 0.2 to 0.8)
                    # doesn't span the full 0-1 range. Calibration would map this observed
                    # range to the full screen.
                    # For now, we assume 0-1 maps to full screen.
                    # Inverting X-axis because typically:
                    #   - looking left (pupil moves to image left, smaller X) should be screen left (0)
                    #   - looking right (pupil moves to image right, larger X) should be screen right (width)
                    # The current norm_pupil_x: 0 (outer corner) to 1 (inner corner).
                    # If outer corner is image left, then 0 is left, 1 is right. This is fine for screen X.
                    
                    # However, pyautogui screen coordinates are (0,0) at top-left.
                    # Our norm_pupil_x: 0 (user's left) to 1 (user's right). Maps to screen_width * norm_pupil_x.
                    # Our norm_pupil_y: 0 (user's up) to 1 (user's down). Maps to screen_height * norm_pupil_y.
                    # This seems correct.

                    target_cursor_x = avg_norm_pupil_x * screen_width * X_SENSITIVITY_SCALE
                    target_cursor_y = avg_norm_pupil_y * screen_height * Y_SENSITIVITY_SCALE
                    
                    # --- Apply Smoothing ---
                    smooth_cursor_x = (SMOOTHING_FACTOR * target_cursor_x) + \
                                      ((1.0 - SMOOTHING_FACTOR) * smooth_cursor_x)
                    smooth_cursor_y = (SMOOTHING_FACTOR * target_cursor_y) + \
                                      ((1.0 - SMOOTHING_FACTOR) * smooth_cursor_y)

                    # Ensure cursor stays within screen bounds
                    smooth_cursor_x = np.clip(smooth_cursor_x, 0, screen_width -1)
                    smooth_cursor_y = np.clip(smooth_cursor_y, 0, screen_height -1)

                    # --- Move Mouse ---
                    pyautogui.moveTo(smooth_cursor_x, smooth_cursor_y, duration=0) # duration=0 for direct jump

                    # --- Visualization (Optional) ---
                    # Draw landmarks on the image.
                    # mp_drawing.draw_landmarks(
                    #     image=image_bgr,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION, # Full mesh
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS, # Just contours
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
                    
                    # Draw circles on pupils and eye corners for debugging
                    for lm_idx in [LEFT_PUPIL_CENTER_LM, LEFT_EYE_INNER_CORNER_LM, LEFT_EYE_OUTER_CORNER_LM, LEFT_EYE_TOP_LM, LEFT_EYE_BOTTOM_LM,
                                   RIGHT_PUPIL_CENTER_LM, RIGHT_EYE_INNER_CORNER_LM, RIGHT_EYE_OUTER_CORNER_LM, RIGHT_EYE_TOP_LM, RIGHT_EYE_BOTTOM_LM]:
                        lm = face_landmarks.landmark[lm_idx]
                        cv2.circle(image_bgr, (int(lm.x * img_w), int(lm.y * img_h)), 3, (0, 0, 255), -1)

                    # Display normalized gaze values on screen (for debugging)
                    cv2.putText(image_bgr, f"Norm Gaze X: {avg_norm_pupil_x:.2f}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(image_bgr, f"Norm Gaze Y: {avg_norm_pupil_y:.2f}", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(image_bgr, f"Cursor: {int(smooth_cursor_x)}, {int(smooth_cursor_y)}", (20, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


                except Exception as e:
                    print(f"Error processing landmarks: {e}")
                    # You might want to continue or handle this more gracefully
                    pass
        
        # Display the resulting frame
        cv2.imshow('Eye Gaze Mouse Control - Press Q to Quit', image_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("Exiting program.")
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Disable PyAutoGUI fail-safe (moving mouse to a corner to stop)
    # Be careful with this if the script behaves erratically.
    # You can manually trigger the fail-safe by quickly moving the mouse to a corner if needed.
    pyautogui.FAILSAFE = False 
    main()
