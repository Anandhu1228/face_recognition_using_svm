import cv2
import pywt
import numpy as np
import joblib
import json
import sys
import os
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_last_best_model.pkl")
HAAR_CASCADES_PATH = os.path.join(BASE_DIR, "haarcascades")

model = joblib.load(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(os.path.join(HAAR_CASCADES_PATH, "haarcascade_frontalface_default.xml"))
eye_cascade = cv2.CascadeClassifier(os.path.join(HAAR_CASCADES_PATH, "haarcascade_eye.xml"))

def crop_and_find_face(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for x, y, w, h in faces:
        actual_face = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(actual_face)
        if len(eyes) >= 2:
            return actual_face
    return None

def w2d(img, mode="haar", level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def preprocess_and_predict(img):
    if img is None:
        raise ValueError("Could not read the image. Please check the file path.")

    cropped_face = crop_and_find_face(img)
    if cropped_face is None:
        return "No valid face found in the image."

    scaled_raw_img = cv2.resize(cropped_face, (32, 32))

    img_wavelet = w2d(cropped_face, "db1", 5)
    scaled_img_wavelet = cv2.resize(img_wavelet, (32, 32))

    combined_img = np.vstack((
        scaled_raw_img.reshape(32 * 32 * 3, 1),
        scaled_img_wavelet.reshape(32 * 32, 1)
    ))

    combined_img = combined_img.reshape(1, 4096).astype(float)
    prediction = model.predict(combined_img)[0]

    return prediction

if __name__ == "__main__":
    # Read base64 image from stdin
    base64_image = sys.stdin.read().strip()
    image_data = base64.b64decode(base64_image)

    # Decode the image using OpenCV
    image = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Predict the celebrity
    result = preprocess_and_predict(img)
    print(result)

