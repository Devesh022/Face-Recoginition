import face_recognition as fr
import cv2
import numpy as np
import pickle


face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")


labels = {"person_name: 1"}
with open('labels_pickle', 'rb') as f:  # reads file in write binary
    # loads label ids from the file in written in binary
    og_labels = pickle.load(f)

    labels = {v: k for k, v in og_labels.items()}  # reverse the key value pair


cap = cv2.VideoCapture(0)

while(True):
    # captures image frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=4)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        # (y co-ordinates_start,y co-ordinate_end)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]  # we can write colour image

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 40:  # and conf <= 85:
            # print(conf)   print(id_)    print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 0, 0)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)  # saves image of last frame

        color = (255, 0, 0)  # Blue  BGR(0-255)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y),
                      color, stroke)  # displays rectangle over face

        eyes = eye_cascade.detectMultiScale(roi_gray)  # detects eyes
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),
                          (0, 255, 0), 2)  # displays rectangle over eyes

        # smiles = smile_cascade.detectMultiScale(roi_gray)  # detects eyes
        # for(sx, sy, sw, sh) in smiles:
        #     cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh),(0, 0, 255), 2)  # displays rectangle over smile

# displays the resulting frame
    cv2.imshow('application', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
