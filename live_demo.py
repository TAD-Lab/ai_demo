"""
Live prediction of age and gender using pre-trained models.
Uses haar Cascades classifier to detect faces.
Uses pre-trained models for gender and age to predict them from live video feed.
"""
import copy
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from time import sleep

# Ensure image size matches models
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_DEPTH = 3

path = os.path.join('models', 'haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(path)

age_model = load_model(os.path.join('models', 'age_model_20epochs-v7.h5'))
gender_model = load_model(os.path.join('models', 'gender_model_20epochs-v7.h5'))

gender_labels = ['Male', 'Female']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)

        # Get image ready for prediction
        roi = roi_gray.astype('float') / 255.0  #Scale
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        roi_color = frame[y: y + h, x: x + w]
        roi_color = cv2.resize(roi_color, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)

        # Gender
        gender_predict = gender_model.predict(np.array(roi_color).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 3))
        conf = copy.copy(gender_predict)
        gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
        gender = gender_labels[gender_predict[0]]

        # Age
        age_predict = age_model.predict(np.array(roi_color).reshape(-1, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH))
        age = round(age_predict[0, 0])
        label_position=(x, y + h + 50)
        cv2.putText(
            frame,
            f"{str(gender)}  Age: {str(age)}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)

    cv2.imshow("AI Demonstration (Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
