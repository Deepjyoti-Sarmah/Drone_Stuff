import cv2
import os
import numpy as np
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Set the path to the input image folder
image_folder = '/home/deepjyotisarmah/Documents/Drone_stuffs/input_images'

# Load the Haar cascade classifier
detector = cv2.CascadeClassifier('/home/deepjyotisarmah/Documents/Drone_stuffs/haarcascade_frontalface_default.xml')

# List to store the HOG features and labels
hog_features = []
labels = []

# Iterate over the files in the input image folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Read the image
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector.detectMultiScale(gray, 1.3, 5)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Crop the face region
            face_img = gray[y:y+h, x:x+w]

            # Convert the face image to grayscale
            face_gray = color.rgb2gray(face_img)

            # Extract HOG features from the face image
            fd, hog_image = hog(face_gray, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(4, 4), block_norm='L2', visualize=True)

            # Append the HOG features and label to the lists
            hog_features.append(fd)
            labels.append(filename)

            # Save the detected face image to the output folder
            output_folder = '/path/to/output/images'  # Set the path to the output image folder
            output_path = os.path.join(output_folder, f'detected_{filename}')
            cv2.imwrite(output_path, face_img)

# Convert the lists to arrays
hog_features = np.array(hog_features)
labels = np.array(labels).reshape(len(labels), 1)

# Shuffle the data
data_frame = np.hstack((hog_features, labels))
np.random.shuffle(data_frame)

# Split the data into training and testing sets
percentage = 80  # Percentage of data to keep for training
partition = int(len(hog_features) * percentage / 100)
x_train, x_test = data_frame[:partition, :-1], data_frame[partition:, :-1]
y_train, y_test = data_frame[:partition, -1:].ravel(), data_frame[partition:, -1:].ravel()

# Initialize and train the SVM classifier
clf = svm.SVC()
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Evaluate the classifier's accuracy and print the results
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(report)