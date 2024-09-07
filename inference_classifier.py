import warnings
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Suppress specific warnings related to deprecated protobuf methods
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open video capture
cap = cv2.VideoCapture(0)  # Try changing the index if necessary
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'I', 1: 'LOVE', 2: 'YOU'}

# List to keep track of all detected signs
detected_signs = []

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                min_x, min_y = min(x_), min(y_)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)

            # Calculate bounding box coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Make prediction
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
            except Exception as e:
                print(f"Error making prediction: {e}")
                predicted_character = "Unknown"

            # Check if the predicted character is different from the last one
            if not detected_signs or predicted_character != detected_signs[-1]:
                print(predicted_character)
                detected_signs.append(predicted_character)

            # Draw bounding box and predicted character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Break the loop if 'q' key is pressed
            break
        elif key == ord('s'):  # Stop detection and speak all detected signs if 's' key is pressed
            if detected_signs:
                for sign in detected_signs:
                    engine.say(sign)
                    engine.runAndWait()
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()
