import cv2
import os
import sys
import time
import numpy as np
from pathlib import Path

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def available_serial_ports():
    if list_ports is None:
        return []
    try:
        return list(list_ports.comports())
    except Exception:
        return []


def detect_arduino_port(default="COM7"):
    ports = available_serial_ports()
    if not ports:
        return default

    preferred_terms = (
        "arduino", "ch340", "wch", "usb serial", "usb-serial",
        "cp210", "silicon labs", "ftdi", "usb modem",
    )

    for port in ports:
        details = f"{port.device} {port.description} {port.hwid}".lower()
        if "bluetooth" in details:
            continue
        if any(term in details for term in preferred_terms):
            return port.device

    for port in ports:
        details = f"{port.device} {port.description} {port.hwid}".lower()
        if "bluetooth" not in details:
            return port.device

    return ports[0].device or default


def main():
    # ---------- Settings ----------
    USE_ARDUINO    = env_flag("FACE_TRACKER_USE_ARDUINO", True)
    ENV_COM_PORT   = os.getenv("FACE_TRACKER_COM_PORT")
    COM_PORT       = (ENV_COM_PORT.strip() if ENV_COM_PORT else detect_arduino_port("COM7")) or "COM7"
    BAUD           = 9600
    SERVO_MIN      = 45
    SERVO_MAX      = 135
    CENTER         = 90
    DEADZONE_PCT   = 0.04   # ±4% of frame width around center
    SMOOTH_A       = 0.35   # 0..1 (higher = snappier)
    SEND_MIN_MS    = 15     # don't spam more often than this
    ANGLE_STEP_MIN = 1      # send only if |delta| >= this
    MIRROR_PREVIEW = True   # mirror cam view for natural feel
    # ------------------------------

    # Try to connect Arduino
    arduino = None
    if USE_ARDUINO:
        if serial is None:
            print("[WARN] PySerial is not installed. Running in camera-only mode.")
        else:
            try:
                arduino = serial.Serial(COM_PORT, BAUD, timeout=0)
                time.sleep(2)
                print(f"[OK] Connected to Arduino on {COM_PORT}.")
            except Exception as exc:
                print(f"[WARN] Arduino not found on {COM_PORT}: {exc}")
                if "access is denied" in str(exc).lower():
                    print("[WARN] Close Arduino IDE Serial Monitor/Plotter or any other app using the port, then restart.")
                ports = available_serial_ports()
                if ports:
                    visible = ", ".join(f"{p.device} ({p.description})" for p in ports)
                    print(f"[WARN] Visible serial ports: {visible}")
                else:
                    print("[WARN] No serial ports detected by PySerial.")
                print("[WARN] Running in camera-only mode.")
    else:
        print("[INFO] Arduino access disabled. Running in camera-only mode.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    # cv2.data.haarcascades is automatically resolved to _MEIPASS/cv2/data/
    # by PyInstaller when frozen, so no special handling needed here.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    invert         = False
    paused         = False
    servo_enabled  = arduino is not None   # toggle with 's'
    servo_angle    = float(CENTER)
    last_sent_angle = None
    last_send_time  = 0

    def send_angle(angle):
        nonlocal last_sent_angle, last_send_time
        if not servo_enabled or arduino is None:
            return
        now = time.time() * 1000
        if last_sent_angle is None or abs(angle - last_sent_angle) >= ANGLE_STEP_MIN:
            if now - last_send_time >= SEND_MIN_MS:
                clipped = int(np.clip(angle, SERVO_MIN, SERVO_MAX))
                try:
                    arduino.write(f"{clipped}\n".encode())
                except Exception:
                    pass
                last_sent_angle = clipped
                last_send_time  = now

    # Start centered
    send_angle(servo_angle)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)

            h, w           = frame.shape[:2]
            frame_center_x = w // 2
            deadzone_px    = int(w * DEADZONE_PCT)

            if len(faces) > 0:
                # Pick largest face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                (x, y, fw, fh) = faces[0]
                cx = x + fw // 2

                # Draw overlay
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                cv2.line(frame, (frame_center_x - deadzone_px, 0), (frame_center_x - deadzone_px, h), (255, 255, 0), 1)
                cv2.line(frame, (frame_center_x + deadzone_px, 0), (frame_center_x + deadzone_px, h), (255, 255, 0), 1)

                # Proportional mapping with optional inversion
                if invert:
                    target = np.interp(cx, [0, w], [SERVO_MAX, SERVO_MIN])
                else:
                    target = np.interp(cx, [0, w], [SERVO_MIN, SERVO_MAX])

                # Dead-zone: if inside, hold last angle
                if abs(cx - frame_center_x) <= deadzone_px:
                    target = servo_angle

                # Exponential smoothing toward target
                servo_angle = float(SMOOTH_A * target + (1 - SMOOTH_A) * servo_angle)

                if not paused:
                    send_angle(servo_angle)

                cv2.putText(frame, f"Angle:{int(servo_angle)} inv:{invert} pause:{paused} servo:{servo_enabled}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No face — hold position
                cv2.putText(frame, f"Angle:{int(servo_angle)} inv:{invert} pause:{paused} servo:{servo_enabled} (no face)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

            cv2.imshow("Fast Face Tracker", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                servo_angle = float(CENTER)
                if not paused:
                    send_angle(servo_angle)
            elif key == ord("r"):
                servo_angle = 0.0
                if not paused:
                    send_angle(servo_angle)
            elif key == ord("i"):
                invert = not invert
            elif key == ord("p"):
                paused = not paused
            elif key == ord("s"):
                servo_enabled = not servo_enabled
                print(f"[INFO] Servo {'enabled' if servo_enabled else 'disabled'}.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arduino:
            arduino.close()


if __name__ == "__main__":
    main()
