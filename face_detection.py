import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

img_rows = 64
img_cols = 64

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
print('CV2 haarcascades directory: ', cv2.data.haarcascades)

def face_detector(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return (0, 0, 0, 0), np.zeros((img_rows, img_cols), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

    try:
        roi_gray = cv2.resize(roi_gray, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
    except:
        return (x, y, w, h), np.zeros((img_rows, img_cols), np.uint8), img
    return (x, y, w, h), roi_gray, img

mood_classifier = load_model('./_mini_XCEPTION.102-0.66.hdf5')
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)

    preds = []

    if np.sum([face]) != 0.0:

        rescaled_roi = face.astype("float") / 255.0
        roi_array = img_to_array(rescaled_roi)
        expanded_roi_array = np.expand_dims(roi_array, axis=0)

        label_scores_array = mood_classifier.predict(expanded_roi_array)
        preds = label_scores_array[0]
        label = EMOTIONS[preds.argmax()]
        label_position = (rect[0], rect[1])
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('All', image)


    canvas = np.zeros((250, 300, 3), dtype="uint8")

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    cv2.imshow("Probabilities", canvas)


    if cv2.waitKey(1) == 13:  # 13 is for the Enter Key
        break

cap.release()
cv2.destroyAllWindows()

