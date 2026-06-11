import cv2
import os
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
MODEL_FILE = BASE_DIR / "face_model.xml"

os.makedirs(DATASET_DIR, exist_ok=True)

# Use DirectShow backend on Windows to avoid the common camera-start delay
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# reduce internal buffer and set a reasonable resolution for faster startup/detection
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# read a few frames to warm up camera (helps some webcams start instantly)
for _ in range(5):
    cap.read()

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Press SPACE to capture your face images. Press Q when done.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect on a smaller image for speed, then scale coordinates back up
    small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    faces_small = detector.detectMultiScale(small, 1.3, 5)

    faces = []
    face = None
    for (x, y, w, h) in faces_small:
        # scale coordinates to original frame size
        x1, y1, w1, h1 = int(x*2), int(y*2), int(w*2), int(h*2)
        faces.append((x1, y1, w1, h1))
        cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
        # take the first detected face region (grayscale)
        if face is None:
            face = gray[y1:y1+h1, x1:x1+w1]

    cv2.imshow("Collecting Faces", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):  # SPACE = capture
        if face is not None:
            # normalize face size for training and save
            face_resized = cv2.resize(face, (200, 200))
            cv2.imwrite(os.path.join(DATASET_DIR, f"user_{count}.jpg"), face_resized)
            count += 1
            print(f"Captured image {count}")
        else:
            print("No face detected - try again.")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Train LBPH model
print("Training model...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
images, labels = [], []

for filename in os.listdir(DATASET_DIR):
    img_path = os.path.join(DATASET_DIR, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    # ensure consistent size
    img = cv2.resize(img, (200, 200))
    images.append(img)
    labels.append(0)  # label 0 = you

if len(images) == 0:
    print("No training images found. Exiting.")
else:
    recognizer.train(images, np.array(labels))
    recognizer.save(str(MODEL_FILE))
    print(f"Training done. Model saved as {MODEL_FILE}")
