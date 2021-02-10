import cv2 as cv
import numpy as np
import os

# to get a list of all celebs names
# p = []
# for i in os.listdir(r'C:\Users\rajat\Desktop\kaggle\OpenCV\celebs\train'):
#     p.append(i)
# print(p)

people = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
DIR = r'C:\Users\rajat\Desktop\kaggle\OpenCV\celebs\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

# this function loops over every folder of the base folder
# inside the folder, loops over every image and grab the face of the image and add that to training list
#
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        # for looping over every image
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            # for detecting the face in an image
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                # basically cropping out the face
                features.append(faces_roi)
                labels.append(label)
create_train()

# print(f'Length of the features = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
print('Training done!!!!!!!!!!')