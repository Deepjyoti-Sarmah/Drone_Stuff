import cv2 
import os
import numpy

image_path = r'/Users/arunabhadutta/Desktop/Project CSE/data'
os.chdir(image_path)
cam=cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('/Users/arunabhadutta/Desktop/Project CSE/haarcascade_frontalface_default.xml')
Id = int(input("enter your id: "))
sampleNum = 0
while(True):
    ret,img=cam.read()
    if ret:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray,1.3,5)

        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+y,w+h),(255,0,0),2)
            sampleNum += 1
            cv2.imwrite(str(sampleNum)+'.'+"jpg",gray[y:y+h,x:x+w])
            cv2.imshow('frame',img)
        if sampleNum == Id:
            break
cam.release()
cv2.destroyAllWindows()