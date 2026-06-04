import cv2
import os
import time
import numpy as np
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ---------- Settings ----------
USE_ARDUINO = env_flag("FACE_TRACKER_USE_ARDUINO", False)
COM_PORT   = os.getenv("FACE_TRACKER_COM_PORT", "COM10").strip() or "COM10"
BAUD       = 9600
SERVO_MIN  = 45
SERVO_MAX  = 135
CENTER     = 90
DEADZONE_PCT = 0.04
SMOOTH_A     = 0.15      # Lower = smoother (was 0.35)
SEND_MIN_MS  = 20        # Slightly higher for stability
ANGLE_STEP_MIN = 0.5     # Lower threshold = more responsive (was 1)
MAX_VELOCITY = 2.0       # Max degrees per frame change (anti-jitter)
MIRROR_PREVIEW = True
MODEL_FILE = "face_model.xml"
MANUAL_SPEED = 1.5       # degrees per press/frame for A/D
MANUAL_DECAY = 0.85      # momentum decay (0.8 = strong glide, 0.95 = weak)
manual_velocity = 0.0    # stores momentum for manual control
AUTO_RETURN_TIME = 0.6   # seconds after last manual input
last_manual_time = 0


# Performance tuning
DETECT_SCALE = 0.5
DETECT_EVERY_N = 2
MIN_FACE_SIZE = (30, 30)

# Position averaging for stability
POSITION_HISTORY_SIZE = 5  # Average last N face positions
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / MODEL_FILE

# Create capture once and configure
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

recognizer = cv2.face.LBPHFaceRecognizer_create()
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
recognizer.read(str(MODEL_PATH))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

arduino = None
if USE_ARDUINO:
    if serial is None:
        print("[WARN] PySerial is not installed. Running without Arduino.")
        USE_ARDUINO = False
    else:
        try:
            arduino = serial.Serial(COM_PORT, BAUD, timeout=0)
            time.sleep(2)
            print(f"[OK] Connected to Arduino on {COM_PORT}.")
        except Exception as exc:
            print(f"[WARN] Could not connect to Arduino on {COM_PORT}: {exc}")
            print("[WARN] Running in face-recognition-only mode.")
            USE_ARDUINO = False
else:
    print("[INFO] Arduino access disabled. Running in face-recognition-only mode.")

if not cap.isOpened():
    raise RuntimeError("Camera not found")

invert = False
paused = False
servo_angle = float(CENTER)
last_sent_angle = None
last_send_time = 0
last_detected_angle = float(CENTER)

# Position history for smoothing
position_history = []

# tracking helpers
frame_counter = 0
last_faces = []
last_face_rect = None
face_lost_counter = 0  # Count frames without face detection
FACE_LOST_THRESHOLD = 10  # Hold position for this many frames before considering truly lost

def send_angle(angle, force=False):
    global last_sent_angle, last_send_time
    if not USE_ARDUINO or angle is None:
        return
    
    now = time.time() * 1000
    angle = float(np.clip(angle, SERVO_MIN, SERVO_MAX))
    
    # Check if we should send
    should_send = False
    if force:
        should_send = (now - last_send_time >= SEND_MIN_MS)
    else:
        if last_sent_angle is None or abs(angle - last_sent_angle) >= ANGLE_STEP_MIN:
            should_send = (now - last_send_time >= SEND_MIN_MS)
    
    if should_send:
        try:
            if arduino and arduino.is_open:
                arduino.write(f"{int(angle)}\n".encode())
                last_sent_angle = angle
                last_send_time = now
        except Exception:
            pass

def ease_smooth(current, target, factor=0.12):
    """Smooth easing toward target."""
    return current + (target - current) * factor

# Initialize servo position
send_angle(servo_angle, force=True)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if MIRROR_PREVIEW:
        frame = cv2.flip(frame, 1)

    # --- moved manual timing here so manual_active is defined before detection ---
    current_time = time.time()
    manual_active = (current_time - last_manual_time) < AUTO_RETURN_TIME
    # --------------------------------------------------------------------------

    h, w = frame.shape[:2]
    frame_center_x = w // 2
    deadzone_px = int(w * DEADZONE_PCT)

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray_full, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_LINEAR)

    faces = []
    # Run detection periodically
    if frame_counter % DETECT_EVERY_N == 0 or not last_faces:
        min_size_scaled = (max(1, int(MIN_FACE_SIZE[0] * DETECT_SCALE)), max(1, int(MIN_FACE_SIZE[1] * DETECT_SCALE)))
        detected = face_cascade.detectMultiScale(small_gray, scaleFactor=1.3, minNeighbors=5, minSize=min_size_scaled)
        if len(detected) > 0:
            inv_scale = 1.0 / DETECT_SCALE
            faces = [(int(x * inv_scale), int(y * inv_scale),
                      int(w_ * inv_scale), int(h * inv_scale)) for (x, y, w_, h) in detected]
            last_faces = faces
            last_face_rect = faces[0]
        else:
            faces = []
            last_faces = []
    else:
        faces = last_faces if last_faces else []

    tracked = False
    for (x, y, fw, fh) in faces:
        if fw <= 0 or fh <= 0 or x < 0 or y < 0 or x+fw > w or y+fh > h:
            continue

        face_gray = gray_full[y:y+fh, x:x+fw]
        try:
            label, confidence = recognizer.predict(face_gray)
        except Exception:
            continue

        if label == 0 and confidence < 70 and not manual_active:  # Recognized face
            cx = x + fw // 2

            # Add to position history for averaging
            position_history.append(cx)
            if len(position_history) > POSITION_HISTORY_SIZE:
                position_history.pop(0)
            
            # Use averaged position to reduce jitter
            avg_cx = int(np.mean(position_history))

            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0,255,0), 2)
            cv2.putText(frame, f"YOU ({confidence:.1f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Calculate target angle from averaged position
            if invert:
                target = np.interp(avg_cx, [0, w], [SERVO_MAX, SERVO_MIN])
            else:
                target = np.interp(avg_cx, [0, w], [SERVO_MIN, SERVO_MAX])

            # Apply deadzone
            if abs(avg_cx - frame_center_x) <= deadzone_px:
                target = servo_angle  # Stay put in deadzone

            # Apply velocity limiting to prevent sudden jumps
            angle_diff = target - servo_angle
            if abs(angle_diff) > MAX_VELOCITY:
                target = servo_angle + np.sign(angle_diff) * MAX_VELOCITY

            # Smooth the angle change
            servo_angle = ease_smooth(servo_angle, target, 0.10)

            if not paused:
                send_angle(servo_angle)
                last_detected_angle = servo_angle

            cv2.putText(frame,
                        f"Angle:{int(servo_angle)} inv:{invert} pause:{paused} {'(Arduino)' if USE_ARDUINO else '(Sim)'}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            tracked = True
            face_lost_counter = 0  # Reset lost counter
            last_face_rect = (x, y, fw, fh)
            break

    if not tracked:
        face_lost_counter += 1
        
        # Hold position firmly when face is lost
        if face_lost_counter <= FACE_LOST_THRESHOLD:
            status_text = f"Angle:{int(servo_angle)} inv:{invert} pause:{paused} (holding)"
            color = (0, 200, 200)
        else:
            status_text = f"Angle:{int(servo_angle)} inv:{invert} pause:{paused} (no face)"
            color = (0, 150, 200)
            # Clear position history when face is truly lost
            position_history.clear()
        
        cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Keep servo at last known position
        if not paused:
            send_angle(last_detected_angle, force=True)

    cv2.imshow("LBPH Face Tracker", frame)
    key = cv2.waitKey(1) & 0xFF

    # ---------------------- MANUAL CONTROL / GLOBAL KEYS ----------------------
    current_time = time.time()

    # Manual left/right input (A/D) affects momentum
    if key == ord('a'):      # Move Left
        manual_velocity -= MANUAL_SPEED
        last_manual_time = current_time

    elif key == ord('d'):    # Move Right
        manual_velocity += MANUAL_SPEED
        last_manual_time = current_time

    # Global keys (always active)
    if key == ord('q'):
        break
    elif key == ord('c'):
        servo_angle = float(CENTER)
        if not paused:
            send_angle(servo_angle)
        last_detected_angle = servo_angle
        position_history.clear()
    elif key == ord('r'):
        servo_angle = float(SERVO_MIN)
        if not paused:
            send_angle(servo_angle)
        last_detected_angle = servo_angle
        position_history.clear()
    elif key == ord('i'):
        invert = not invert
    elif key == ord('p'):
        paused = not paused

    # When manual is active, freeze face tracking and apply momentum
    manual_active = (current_time - last_manual_time) < AUTO_RETURN_TIME

    if manual_active:
        # Apply momentum
        servo_angle += manual_velocity
        manual_velocity *= MANUAL_DECAY

        # Clamp limits
        servo_angle = float(np.clip(servo_angle, SERVO_MIN, SERVO_MAX))

        if not paused:
            send_angle(servo_angle)

        last_detected_angle = servo_angle
        position_history.clear()

    # increment frame counter each loop
    frame_counter += 1

cap.release()
cv2.destroyAllWindows()
if USE_ARDUINO and arduino:
    try:
        arduino.close()
    except Exception:
        pass
