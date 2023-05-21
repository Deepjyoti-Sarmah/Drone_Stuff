import cv2 
import os
# import numpy


image_path = '/home/tamaroy/Documents/Drone_Stuff/data_set/akshay'
detector = cv2.CascadeClassifier('/home/tamaroy/Documents/Drone_Stuff/haarcascade_frontalface_default.xml')
Id = int(input("enter your id: "))
sampleNum = 0

output_path = '/home/tamaroy/Documents/Drone_Stuff/images'
if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in os.listdir(image_path):
    img = cv2.imread(os.path.join(image_path, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    output_filename = os.path.join(output_path, f'{Id}_{sampleNum}.jpg')
    
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + y, w + h), (255, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sampleNum += 1
        cv2.imwrite(output_filename, gray[y:y + h, x:x + w])
        print(f'Saving image to {output_filename}')
        cv2.imshow('frame', img)
    if sampleNum == Id:
        break


# tama code
cv2.destroyAllWindows()