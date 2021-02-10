import numpy as np
import cv2 as cv

people = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load('features.npy', allow_pickel= True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\rajat\Desktop\kaggle\OpenCV\celebs\val\ben_afflek\httpcsvkmeuadecafjpg.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# detect the face in the image
face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

for (x,y,w,h) in face_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]),(20,20),cv.FONT_HERSHEY_DUPLEX,1.0,(0,255,0), thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h), (0,0,255), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)