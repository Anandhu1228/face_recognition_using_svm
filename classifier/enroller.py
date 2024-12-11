import cv2
import pywt
import numpy as np
import os
import shutil
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_CASCADES_PATH = os.path.join(BASE_DIR, "haarcascades")
path_to_data = os.path.join(BASE_DIR, "camera_enroll_image_set")
path_to_cr_data = os.path.join(BASE_DIR, "camera_enroll_image_set", "cropped")
negative_data_path = os.path.join(BASE_DIR, "negative_image_set")
model_save_path = os.path.join(BASE_DIR, "face_last_best_model.pkl")  # Save model here

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
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray) / 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    return np.uint8(imArray_H)

if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

# Process positive images
cropped_image_dir = []
for img_file in os.scandir(path_to_data):
    if img_file.is_file() and img_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(img_file.path)
        actual_face = crop_and_find_face(img)
        if actual_face is not None:
            crop_file_name = f"face_{str(len(cropped_image_dir) + 1)}.jpg"
            crop_file_path = os.path.join(path_to_cr_data, crop_file_name)
            cv2.imwrite(crop_file_path, actual_face)
            cropped_image_dir.append(crop_file_path)

x = []
y = []

# Positive examples
for image in cropped_image_dir:
    img = cv2.imread(image)
    scaled_raw_img = cv2.resize(img, (32, 32))
    img_wavelet = w2d(img, 'db1', 5)
    scaled_img_wavelet = cv2.resize(img_wavelet, (32, 32))
    combined_img = np.vstack((scaled_raw_img.reshape(32*32*3, 1), scaled_img_wavelet.reshape(32*32, 1)))
    x.append(combined_img)
    y.append(1)  # Label as positive

# Negative examples
for img_file in os.scandir(negative_data_path):
    if img_file.is_file() and img_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(img_file.path)
        actual_face = crop_and_find_face(img)  # Only process valid faces
        if actual_face is not None:
            scaled_raw_img = cv2.resize(actual_face, (32, 32))
            img_wavelet = w2d(actual_face, 'db1', 5)
            scaled_img_wavelet = cv2.resize(img_wavelet, (32, 32))
            combined_img = np.vstack((scaled_raw_img.reshape(32*32*3, 1), scaled_img_wavelet.reshape(32*32, 1)))
            x.append(combined_img)
            y.append(0)  # Label as negative

x = np.array(x).reshape(len(x), 4096).astype(float)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
pipe_sv = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='linear', C=1, gamma="auto"))])
best_model = pipe_sv.fit(x_train, y_train)

joblib.dump(best_model, model_save_path)

shutil.rmtree(path_to_data)

