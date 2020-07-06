import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_movie = cv2.VideoWriter('output1.avi', fourcc, 60, (1280, 720))

cap = cv2.VideoCapture(r'C:\Users\rahul\Desktop\Facial_Expression_Recognition\videos\facial_exp.mkv')
while True:
    _, fr = cap.read()
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        fc = gray_fr[y:y + h, x:x + w]

        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)

    output_movie.write(fr)
    #cv2.imshow('frame', fr)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()







