import os
import cv2
import mediapipe as mp

# Directory to save collected images
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 300

cap = cv2.VideoCapture(0)  # Change index if necessary
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for user to press 'a' to start collecting data
    start_collection = False
    while not start_collection:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            continue

        cv2.putText(frame, 'Press "a" to start collecting photos', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('a'):
            start_collection = True

    # Collect 100 images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            continue

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1
        print(f'Photo taken: {counter}')  # Print photo count after each photo

    print(f'Collected {dataset_size} images for class {j}. Press Enter to proceed to the next class.')
    while True:
        key = cv2.waitKey(0)  # Wait indefinitely for a key press
        if key == 13:  # Enter key
            break

cap.release()
cv2.destroyAllWindows()
