import pickle
import os
from PIL import Image
import numpy as np
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.face.createLBPHFaceRecognizer()
current_id = 0
label_ids = {}
y_labels = []
x_train = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if(file.endswith("png") or file.endswith("jpg")):
            path = os.path.join(root, file)
            # label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()         (can also use this)
            label = os.path.basename(root).replace(" ", "_").lower()
            # print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
# y_labels.append(label)  (label id )
# x_train.append(path)  (verifys this image trun into numpy array,gray)
            pil_image = Image.open(path).convert(
                'L')  # converts image in to grayscale
            size = (550, 550)
            # resizes all the images in(550x550)
            final_image = pil_image.resize(size, Image.Image.ANTIALIAS)

            image_array = np.array(final_image, 'uint8')
            # print(image_array)
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)


with open('labels_pickle', 'wb') as f:  # opens file in write binary
    pickle.dump(label_ids, f)  # dumps label ids in the file in binary

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
