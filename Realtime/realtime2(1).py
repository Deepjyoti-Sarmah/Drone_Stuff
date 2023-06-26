import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load pre-trained models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # Load your custom pre-trained weights
# weights_path = 'new_dataset'
# resnet.load_state_dict(torch.load(weights_path))

# Load pre-trained face embeddings
# Assuming you have a file named 'face_embeddings.pt' containing embeddings
face_embeddings = torch.load('faces_embeddings_real.pt')

# Save the face embeddings with the key 'arr_0'
# np.savez('faces_embeddings_1.npz', arr_0=face_embeddings)


# Assuming you have a file named 'face_embeddings.npz' containing embeddings

# face_embeddings = np.load('faces_embeddings_1.npz')['arr_0']

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        # Iterate through each detected face
        for box in boxes:
            x, y, w, h = box
            face = frame[int(y):int(y+h), int(x):int(x+w)]

            # Resize face image to match FaceNet requirements
            face = cv2.resize(face, (160, 160))

            # Convert BGR image to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Preprocess the face image
            face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Calculate face embeddings using FaceNet
            embeddings = resnet(face).detach().cpu()

            # Compare face embeddings with pre-trained embeddings
            distances = (face_embeddings - embeddings).norm(dim=1).cpu()

            # Set a threshold for face recognition
            threshold = 1.0
            if distances.min() < threshold:
                # Face recognized
                recognized_label = 'Person'
            else:
                # Face not recognized
                recognized_label = 'Unknown'

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.putText(frame, recognized_label, (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()