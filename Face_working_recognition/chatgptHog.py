import cv2
import os
import random
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# Set the path to the input image folder
image_folder = '/home/deepjyotisarmah/Documents/Drone_stuffs/lfw'

# Set the path to the Haar cascade XML file
haar_cascade_file = '/home/deepjyotisarmah/Documents/Drone_stuffs/haarcascade_frontalface_default.xml'

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(haar_cascade_file)

# List to store the HOG features and labels
hog_features = []
labels = []

# Iterate over the subfolders in the image folder
for subdir in os.listdir(image_folder):
    subfolder_path = os.path.join(image_folder, subdir)
    if not os.path.isdir(subfolder_path):
        continue

    # Process each image in the subfolder
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Read the image
            image_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Process each detected face
            for (x, y, w, h) in faces:
                # Crop the face region
                face_img = img[y:y+h, x:x+w]

                # Resize the face image to a fixed size (e.g., 80x80)
                resized_face = cv2.resize(face_img, (80, 80))

                # Convert the resized face image to grayscale
                gray_resized = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

                # Compute HOG features for the resized face image
                hog_feats = hog(gray_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

                # Append the HOG features to the list
                hog_features.append(hog_feats)
                labels.append(subdir)  # Use the subfolder name as the label

                # Break the loop if the maximum number of images is reached
                if len(hog_features) >= 3000:
                    break

        # Break the loop if the maximum number of images is reached
        if len(hog_features) >= 3000:
            break

# Convert the lists to arrays
hog_features = np.array(hog_features)
labels = np.array(labels)

# Combine the features and labels into a single array
data = np.hstack((hog_features, labels.reshape(-1, 1)))

# Shuffle the data
random.shuffle(data)

# Split the data into training and testing sets
train_ratio = 0.8  # Percentage of data to keep for training
train_size = int(train_ratio * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Separate the features and labels for training and testing
x_train = train_data[:, :-1]
y_train = train_data[:, -1]
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Initialize and train the SVM classifier
clf = svm.SVC()
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)