import cv2
import serial
import time
import numpy as np

# ---------- Settings ----------
USE_ARDUINO = False    # toggle: True = control Arduino, False = face recognition only
COM_PORT   = 'COM7'   # <-- change for your Arduino
BAUD       = 9600
SERVO_MIN  = 45
SERVO_MAX  = 135
CENTER     = 90
DEADZONE_PCT = 0.04
SMOOTH_A     = 0.35
SEND_MIN_MS  = 15
ANGLE_STEP_MIN = 1
MIRROR_PREVIEW = True
MODEL_FILE = "face_model.xml"
# ------------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

arduino = None
if USE_ARDUINO:
    arduino = serial.Serial(COM_PORT, BAUD, timeout=0)
    time.sleep(2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found")

invert = False
paused = False
servo_angle = CENTER
last_sent_angle = None
last_send_time = 0

def send_angle(angle):
    global last_sent_angle, last_send_time
    if not USE_ARDUINO:
        return
    now = time.time() * 1000
    if last_sent_angle is None or abs(angle - last_sent_angle) >= ANGLE_STEP_MIN:
        if now - last_send_time >= SEND_MIN_MS:
            angle = int(np.clip(angle, SERVO_MIN, SERVO_MAX))
            try:
                arduino.write(f"{angle}\n".encode())
            except Exception:
                pass
            last_sent_angle = angle
            last_send_time = now

send_angle(servo_angle)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if MIRROR_PREVIEW:
        frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    h, w = frame.shape[:2]
    frame_center_x = w // 2
    deadzone_px = int(w * DEADZONE_PCT)

    tracked = False
    for (x, y, fw, fh) in faces:
        face_gray = gray[y:y+fh, x:x+fw]
        label, confidence = recognizer.predict(face_gray)

        if label == 0 and confidence < 70:  # 0 = your face
            cx = x + fw // 2

            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0,255,0), 2)
            cv2.putText(frame, f"YOU ({confidence:.1f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if invert:
                target = np.interp(cx, [0, w], [SERVO_MAX, SERVO_MIN])
            else:
                target = np.interp(cx, [0, w], [SERVO_MIN, SERVO_MAX])

            if abs(cx - frame_center_x) <= deadzone_px:
                target = servo_angle

            servo_angle = float(SMOOTH_A * target + (1 - SMOOTH_A) * servo_angle)

            if not paused:
                send_angle(servo_angle)

            cv2.putText(frame,
                        f"Angle:{int(servo_angle)} inv:{invert} pause:{paused} {'(Arduino)' if USE_ARDUINO else '(Sim)'}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            tracked = True
            break

    if not tracked:
        cv2.putText(frame, f"Angle:{int(servo_angle)} inv:{invert} pause:{paused} (no YOUR face)",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

    cv2.imshow("LBPH Face Tracker", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        servo_angle = CENTER
        if not paused: send_angle(servo_angle)
    elif key == ord('r'):
        servo_angle = 0
        if not paused: send_angle(servo_angle)
    elif key == ord('i'):
        invert = not invert
    elif key == ord('p'):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
if USE_ARDUINO and arduino:
    arduino.close()
