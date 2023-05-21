#### readind from camera ###############################################

# import cv2 
# import os
# # import numpy

# image_path = r'/home/deepjyotisarmah/Documents/Drone_stuffs/images'
# os.chdir(image_path)
# cam=cv2.VideoCapture(0)
# detector=cv2.CascadeClassifier('/home/deepjyotisarmah/Documents/Drone_stuffs/haarcascade_frontalface_default.xml')
# Id = int(input("enter your id: "))
# sampleNum = 0
# while(True):
#     ret,img=cam.read()
#     if ret:
#         gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         faces=detector.detectMultiScale(gray,1.3,5)

#         for(x,y,w,h) in faces:
#             cv2.rectangle(img,(x,y),(x+y,w+h),(255,0,0),2)
#             sampleNum += 1
#             cv2.imwrite(str(sampleNum)+'.'+"jpg",gray[y:y+h,x:x+w])
#             cv2.imshow('frame',img)
#         if sampleNum == Id:
#             break
# cam.release()
# cv2.destroyAllWindows()

##### reading from a folder ###################################################

import cv2
import os

# Path to the input images folder
input_folder = r'/home/deepjyotisarmah/Documents/Drone_stuffs/input_images'

# Path to the output images folder
output_folder = r'/home/deepjyotisarmah/Documents/Drone_stuffs/output_images'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the Haar cascade classifier
detector = cv2.CascadeClassifier('/home/deepjyotisarmah/Documents/Drone_stuffs/haarcascade_frontalface_default.xml')

# Iterate over the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = detector.detectMultiScale(gray, 1.3, 5)
        
        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Save the detected face to the output folder
            face_img = gray[y:y + h, x:x + w]
            output_path = os.path.join(output_folder, f'detected_{filename}')
            cv2.imwrite(output_path, face_img)
        
        # Display the image with detected faces
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(1) # Introduce a slight delay between images

# Clean up
cv2.destroyAllWindows()