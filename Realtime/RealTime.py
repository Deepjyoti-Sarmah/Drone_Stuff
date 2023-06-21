import tensorflow as tf
import pickle 
from keras_facenet import FaceNet
import cv2 as cv
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import LabelEncoder 


facenet = FaceNet()
faces_embeddings = np.load(r"/home/deepjyotisarmah/Documents/Drone_stuffs/MTCNN/faces_embeddings_1.npz")


Y = faces_embeddings['arr_1']

encoder =  LabelEncoder()

encoder.fit(Y)



haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

model = pickle.load(open("svm_model_160x160.pkl", 'rb'))



cap = cv.VideoCapture(0)



while cap.read():

    _,frame = cap.read()

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    

    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    

    for x,y,w,h in faces:

        img = rgb_img[y:y+h, x:x+w]

        img = cv.resize(img, (160,160))

        img = np.expand_dims(img, axis=0)

        ypred = facenet.embeddings(img)

        

        face_name = model.predict(ypred)

        final_name = encoder.inverse_transform(face_name)

        

        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 10)

        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv.LINE_AA)

    

    cv.imshow("Face Recognition :", frame)

    if cv.waitKey(1) & ord ('q') == 27:
        break

cap.release()
cv.destroyAllWindows
