import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load your trained model
model = tf.keras.models.load_model('model.h5')

# Load label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')

# Function to preprocess the face for your model
def preprocess_face(face):
    face = cv2.resize(face, (64, 64))  # Sesuaikan dengan input model kamu
    face = face / 255.0  # Normalisasi
    face = np.expand_dims(face, axis=0)
    return face

# Inisialisasi video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract the face
        face = gray[y:y+h, x:x+w]
        preprocessed_face = preprocess_face(face)

        # Predict the face using the model
        predictions = model.predict(preprocessed_face)
        name_index = np.argmax(predictions, axis=1)
        
        # Convert index to name
        name = label_encoder.inverse_transform(name_index)[0]

        # Annotate frame with the name
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the video
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
