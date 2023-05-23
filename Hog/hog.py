import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

# Set the path to the input image folder
input_folder = '/home/deepjyotisarmah/Documents/Drone_stuffs/lfw'

# Load the Haar cascade classifier
detector = cv2.CascadeClassifier('/home/deepjyotisarmah/Documents/Drone_stuffs/haarcascade_frontalface_default.xml')

# List to store the HOG features and labels
hog_features = []
labels = []

# Iterate over subfolders and files in the input folder
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Read the image
            image_path = os.path.join(root, filename)
            img = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces = detector.detectMultiScale(gray, 1.3, 5)

            # Process each detected face
            for (x, y, w, h) in faces:
                # Crop the face region
                face_img = img[y:y+h, x:x+w]

                # Resize the face image to a fixed size (e.g., 80x80)
                resized_face = cv2.resize(face_img, (80, 80))

                # Compute HOG features for each color channel
                hog_feats = []
                for channel in range(resized_face.shape[2]):
                    hog_feats.append(hog(resized_face[:, :, channel], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4),block_norm='L2', visualize=False))

                # Concatenate the HOG features from all color channels
                hog_features.append(np.hstack(hog_feats))
                labels.append(os.path.basename(root))

# Convert the lists to arrays
hog_features = np.array(hog_features)
labels = np.array(labels)

# Reshape the labels array to match the shape of hog_features
labels = labels.reshape((-1, 1))

# Concatenate the arrays
data_frame = np.column_stack((hog_features, labels))

# Shuffle the data
np.random.shuffle(data_frame)

# Check if training samples are available
if len(data_frame) == 0:
    print("No training samples available.")
    exit()

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
