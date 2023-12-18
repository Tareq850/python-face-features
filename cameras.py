import cv2
import keyboard
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tkinter as tk
import time

# Loading a pre-trained model for mood classification
model_path = 'C:/Users/UX/.spyder-py3/smile.keras'  # قم بتوفير مسار النموذج
emotion_model = load_model(model_path, compile=False)

# Download Haarcascades file for faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

expected_image_size = (224, 224)
# List of mood classifications
emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

# Open the webcam
cap = cv2.VideoCapture(0)
frame_delay = 0.8

# Create a tkinter window
root = tk.Tk()
root.title("Emotion Detection")

# Create a text element to display a psychological state
emotion_label = tk.Label(root, text="Emotion: ", font=("Helvetica", 16))
emotion_label.pack()

def update_emotion_label(emotion):
    emotion_label.config(text=f"Emotion: {emotion}")

# A function to update the psychological state in the user interface
def update_emotion_gui(emotion):
    root.after(100, update_emotion_label, emotion)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # Convert the frame to black and white (grayscale) to improve face recognition performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Select faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw boxes around faces and classify moods
    for (x, y, w, h) in faces:
        # Cut the face from the frame and change its size to (64, 64)
        face_roi = cv2.resize(gray[y:y+h, x:x+w], expected_image_size)

        # Duplicate the gray channels to make three channels
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        # Convert face to array
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0  # Reduce pixel values ​​to match the mood model

        # Prediction using a mood model
        emotion_probabilities = emotion_model.predict(face_roi)[0]
        emotion_label = emotion_labels[np.argmax(emotion_probabilities)]

        # Draw a square around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display mood rating on screen
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Update psychological status in the user interface
        update_emotion_gui(emotion_label)

    # Modified frame width
    cv2.imshow('Emotion Detection', frame)

    # Wait for a key press to stop the program (ESC)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Waiting to press a key to stop the camera (for example, the letter "q")
    if keyboard.is_pressed('q'):
        break

# Close the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()

# Trigger event cycle in UI
root.mainloop()
