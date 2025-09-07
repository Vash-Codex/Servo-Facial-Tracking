import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
MODEL_FILE = "face_model.xml"

os.makedirs(DATASET_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Press SPACE to capture your face images. Press Q when done.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        face = gray[y:y+h, x:x+w]

    cv2.imshow("Collecting Faces", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):  # SPACE = capture
        if len(faces) > 0:
            cv2.imwrite(os.path.join(DATASET_DIR, f"user_{count}.jpg"), face)
            count += 1
            print(f"Captured image {count}")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Train LBPH model
print("Training model...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
images, labels = [], []

for i, filename in enumerate(os.listdir(DATASET_DIR)):
    img = cv2.imread(os.path.join(DATASET_DIR, filename), cv2.IMREAD_GRAYSCALE)
    images.append(img)
    labels.append(0)  # label 0 = you

recognizer.train(images, np.array(labels))
recognizer.save(MODEL_FILE)
print(f"Training done. Model saved as {MODEL_FILE}")
