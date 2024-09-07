import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Path where images are stored
image_paths = ['data\vijay.jpg', 'data\vijay_dev.jpg']

# Prepare data
labels = []
faces = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        faces.append(gray[y:y+h, x:x+w])
        label = os.path.basename(image_path).split('.')[0]  # Extract label from filename
        labels.append(label)

# Convert to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Train model
model = SVC(kernel='linear', probability=True)
model.fit(faces.reshape(len(faces), -1), labels)

# Save the model and label encoder
joblib.dump(model, 'face_recognition_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model training complete and saved!")
