import face_recognition
import cv2
import pickle
from datetime import datetime

# Load the face encodings and names
with open('face_encodings.pkl', 'rb') as f:
    data = pickle.load(f)

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.6)
        name = "Unknown"  # Default to Unknown

        # Check if a match was found
        face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = data["names"][best_match_index]

        # Draw a box around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the name below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Print the name, date, and time
        print(f"Name: {name}, Date: {datetime.now().strftime('%Y-%m-%d')}, Time: {datetime.now().strftime('%H:%M:%S')}")

    # Show the frame with detection
    cv2.imshow('Real-Time Face Recognition', frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
