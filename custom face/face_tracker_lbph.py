import cv2
import os
import sys
import time
import numpy as np
import logging
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def available_serial_ports():
    if list_ports is None:
        return []
    try:
        return list(list_ports.comports())
    except Exception:
        return []


def detect_arduino_port(default="COM3"):
    ports = available_serial_ports()
    if not ports:
        return default

    preferred_terms = (
        "arduino",
        "ch340",
        "ch341",
        "wch",
        "usb serial",
        "usb-serial",
        "cp210",
        "cp2102",
        "cp2104",
        "silicon labs",
        "ftdi",
        "ft232",
        "usb modem",
        "cdc",
        "pro micro",
        "leonardo",
        "nano",
        "uno",
        "mega",
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

    return ports[0].device if ports else default


def get_cascade_path(filename: str = "haarcascade_frontalface_default.xml") -> str:
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)
        candidates.extend([
            meipass / filename,
            meipass / "cv2" / "data" / filename,
            meipass / "cv2" / "data" / "data" / filename,
            meipass / "data" / filename,
        ])
    here = Path(__file__).resolve().parent
    candidates.extend([
        here / filename,
        here.parent / filename,
        Path.cwd() / filename,
    ])
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        haar_dir = Path(cv2.data.haarcascades)
        candidates.extend([
            haar_dir / filename,
            haar_dir.parent / filename,
            haar_dir.parent / "data" / filename,
        ])
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return os.path.join(getattr(cv2.data, "haarcascades", ""), filename)


def _get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _get_model_path(model_filename: str) -> Path:
    user_model = _get_app_dir() / "custom face" / model_filename
    if user_model.exists():
        return user_model
    if getattr(sys, "frozen", False):
        try:
            bundled = Path(sys._MEIPASS) / "custom face" / model_filename
            if bundled.exists():
                return bundled
        except AttributeError:
            pass
    return user_model


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TrackerConfig:
    use_arduino: bool = False
    com_port: str = "COM3"
    baud_rate: int = 9600

    servo_min: int = 45
    servo_max: int = 135
    servo_center: int = 90

    deadzone_pct: float = 0.04
    smooth_factor: float = 0.10
    send_min_ms: int = 20
    angle_step_min: float = 0.5
    max_velocity: float = 2.0
    confidence_threshold: int = 70

    manual_speed: float = 1.5
    manual_decay: float = 0.85
    auto_return_time: float = 0.6

    detect_scale: float = 0.5
    detect_every_n: int = 2
    min_face_size: Tuple[int, int] = (30, 30)

    position_history_size: int = 5
    face_lost_threshold: int = 10

    camera_width: int = 1280
    camera_height: int = 720

    mirror_preview: bool = True
    model_file: str = "face_model.xml"

    def validate(self):
        errors = []
        if self.servo_min < 0 or self.servo_max > 180:
            errors.append("Servo angles must be between 0-180")
        if self.servo_center < self.servo_min or self.servo_center > self.servo_max:
            errors.append("Servo center must be between min and max")
        if not 0 <= self.deadzone_pct <= 0.5:
            errors.append("Deadzone percentage must be 0-0.5")
        if self.smooth_factor <= 0 or self.smooth_factor > 1:
            errors.append("Smooth factor must be 0-1")
        if self.confidence_threshold < 0 or self.confidence_threshold > 100:
            errors.append("Confidence threshold must be 0-100")
        if len(errors) > 0:
            raise ValueError("Invalid configuration: " + "; ".join(errors))

    @classmethod
    def from_env(cls):
        raw_port = os.getenv("FACE_TRACKER_COM_PORT", "").strip()
        port = raw_port if raw_port and raw_port.upper() != "AUTO" else detect_arduino_port("COM3")
        return cls(
            use_arduino=env_flag("FACE_TRACKER_USE_ARDUINO", False),
            com_port=port,
            confidence_threshold=int(os.getenv("FACE_TRACKER_CONFIDENCE", "70")),
        )


class ArduinoController:

    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_connected = False

    def connect(self) -> bool:
        if serial is None:
            logger.warning("PySerial not installed. Running without Arduino.")
            return False

        ports_to_try = []
        if self.port and self.port != "AUTO":
            ports_to_try.append(self.port)
        detected = detect_arduino_port(None)
        if detected and detected not in ports_to_try:
            ports_to_try.append(detected)
        for p in available_serial_ports():
            if p.device not in ports_to_try and "bluetooth" not in f"{p.description} {p.hwid}".lower():
                ports_to_try.append(p.device)

        for port_candidate in ports_to_try:
            try:
                self.serial = serial.Serial(port_candidate, self.baudrate, timeout=0)
                time.sleep(2)
                self.is_connected = True
                self.port = port_candidate
                logger.info(f"Connected to Arduino on {port_candidate}")
                return True
            except Exception as exc:
                logger.warning(f"Could not connect to Arduino on {port_candidate}: {exc}")

        self.is_connected = False
        return False

    def send_angle(self, angle: int) -> bool:
        if not self.is_connected or self.serial is None:
            return False

        try:
            if self.serial.is_open:
                self.serial.write(f"{int(angle)}\n".encode())
                return True
        except Exception as e:
            logger.warning(f"Failed to send angle: {e}")
            self.is_connected = False
        return False

    def close(self):
        if self.serial and self.serial.is_open:
            try:
                self.serial.close()
            except Exception:
                pass


class PositionSmoother:

    def __init__(self, history_size: int = 5):
        self.history = deque(maxlen=history_size)

    def add(self, position: float) -> float:
        self.history.append(position)
        return np.mean(self.history)

    def clear(self):
        self.history.clear()

    def is_valid(self) -> bool:
        return len(self.history) >= 2


class FaceTracker:

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.config.validate()

        self.cap = None
        self.face_cascade = None
        self.recognizer = None
        self.arduino = None

        self.position_smoother = PositionSmoother(config.position_history_size)

        self.servo_angle = float(config.servo_center)
        self.last_sent_angle = None
        self.last_send_time = 0
        self.last_detected_angle = float(config.servo_center)

        self.manual_velocity = 0.0
        self.last_manual_time = 0

        self.invert = False
        self.paused = False
        self.frame_counter = 0
        self.face_lost_counter = 0

        self.last_faces = []
        self.last_face_rect = None

        self.fps_times = deque(maxlen=30)
        self.current_fps = 0.0

        self._initialized = False

    def initialize(self) -> bool:
        try:
            cascade_path = get_cascade_path("haarcascade_frontalface_default.xml")
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load Haar Cascade classifier")
            logger.info("Loaded Haar Cascade classifier")

            if not self._init_camera():
                return False

            if not self._load_model():
                return False

            if self.config.use_arduino:
                self.arduino = ArduinoController(self.config.com_port, self.config.baud_rate)
                if self.arduino.connect():
                    self._send_angle(self.servo_angle, force=True)
                else:
                    logger.info("Running without Arduino")
            else:
                logger.info("Arduino access disabled")

            self._initialized = True
            logger.info("Tracker initialization successful")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.cleanup()
            return False

    def _init_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                raise RuntimeError("Camera not found or cannot be opened")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            for _ in range(5):
                ret, _ = self.cap.read()
                if not ret:
                    raise RuntimeError("Camera failed during warmup")

            logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def _load_model(self) -> bool:
        try:
            model_path = _get_model_path(self.config.model_file)

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(str(model_path))
            logger.info(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _send_angle(self, angle: float, force: bool = False):
        if not self.arduino or not self.arduino.is_connected:
            return

        now = time.time() * 1000
        angle = float(np.clip(angle, self.config.servo_min, self.config.servo_max))

        should_send = False
        if force:
            should_send = (now - self.last_send_time >= self.config.send_min_ms)
        else:
            if self.last_sent_angle is None or abs(angle - self.last_sent_angle) >= self.config.angle_step_min:
                should_send = (now - self.last_send_time >= self.config.send_min_ms)

        if should_send:
            if self.arduino.send_angle(angle):
                self.last_sent_angle = angle
                self.last_send_time = now

    def _ease_smooth(self, current: float, target: float, factor: Optional[float] = None) -> float:
        if factor is None:
            factor = self.config.smooth_factor
        return current + (target - current) * factor

    def _detect_faces(self, gray: np.ndarray) -> list:
        try:
            small_gray = cv2.resize(gray, (0, 0),
                                   fx=self.config.detect_scale,
                                   fy=self.config.detect_scale,
                                   interpolation=cv2.INTER_LINEAR)

            min_size_scaled = (
                max(1, int(self.config.min_face_size[0] * self.config.detect_scale)),
                max(1, int(self.config.min_face_size[1] * self.config.detect_scale))
            )

            detected = self.face_cascade.detectMultiScale(
                small_gray, scaleFactor=1.3, minNeighbors=5, minSize=min_size_scaled
            )

            if len(detected) == 0:
                return []

            inv_scale = 1.0 / self.config.detect_scale
            faces = [
                (int(x * inv_scale), int(y * inv_scale),
                 int(w * inv_scale), int(h * inv_scale))
                for (x, y, w, h) in detected
            ]
            return faces
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            return []

    def _is_valid_face_rect(self, x: int, y: int, fw: int, fh: int, frame_width: int, frame_height: int) -> bool:
        return fw > 0 and fh > 0 and x >= 0 and y >= 0 and x + fw <= frame_width and y + fh <= frame_height

    def _recognize_face(self, face_gray: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        try:
            label, confidence = self.recognizer.predict(face_gray)
            return label, confidence
        except Exception as e:
            logger.debug(f"Recognition error: {e}")
            return None, None

    def _update_fps(self):
        self.fps_times.append(time.time())
        if len(self.fps_times) >= 2:
            elapsed = self.fps_times[-1] - self.fps_times[0]
            if elapsed > 0:
                self.current_fps = len(self.fps_times) / elapsed

    def _calculate_target_angle(self, face_x: int, face_width: int, frame_center_x: int, frame_width: int) -> float:
        avg_cx = self.position_smoother.add(face_x)

        if self.invert:
            target = np.interp(avg_cx, [0, frame_width], [self.config.servo_max, self.config.servo_min])
        else:
            target = np.interp(avg_cx, [0, frame_width], [self.config.servo_min, self.config.servo_max])

        deadzone_px = int(frame_width * self.config.deadzone_pct)
        if abs(avg_cx - frame_center_x) <= deadzone_px:
            target = self.servo_angle

        angle_diff = target - self.servo_angle
        if abs(angle_diff) > self.config.max_velocity:
            target = self.servo_angle + np.sign(angle_diff) * self.config.max_velocity

        return target

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        try:
            if frame is None or frame.size == 0:
                logger.warning("Invalid frame received")
                return frame, False

            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                logger.warning("Invalid frame dimensions")
                return frame, False

            if self.config.mirror_preview:
                frame = cv2.flip(frame, 1)

            current_time = time.time()
            manual_active = (current_time - self.last_manual_time) < self.config.auto_return_time

            try:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.warning(f"Color conversion failed: {e}")
                return frame, False

            faces = []
            if self.frame_counter % self.config.detect_every_n == 0 or not self.last_faces:
                faces = self._detect_faces(gray_full)
                if len(faces) > 0:
                    self.last_faces = faces
                    self.last_face_rect = faces[0]
                else:
                    self.last_faces = []
            else:
                faces = self.last_faces

            tracked = False
            frame_center_x = w // 2

            for (x, y, fw, fh) in faces:
                if not self._is_valid_face_rect(x, y, fw, fh, w, h):
                    continue

                face_gray = gray_full[y:y+fh, x:x+fw]
                label, confidence = self._recognize_face(face_gray)

                if label is None or confidence is None:
                    continue

                if label == 0 and confidence < self.config.confidence_threshold and not manual_active:
                    cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                    cv2.putText(frame, f"YOU ({confidence:.1f})", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    face_center_x = x + fw // 2
                    target = self._calculate_target_angle(face_center_x, fw, frame_center_x, w)

                    self.servo_angle = self._ease_smooth(self.servo_angle, target)

                    if not self.paused:
                        self._send_angle(self.servo_angle)
                        self.last_detected_angle = self.servo_angle

                    status_text = f"Angle:{int(self.servo_angle)}° inv:{self.invert} pause:{self.paused}"
                    if self.arduino and self.arduino.is_connected:
                        status_text += " (Arduino)"
                    else:
                        status_text += " (Sim)"
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    tracked = True
                    self.face_lost_counter = 0
                    self.last_face_rect = (x, y, fw, fh)
                    break

            if not tracked:
                self.face_lost_counter += 1

                if self.face_lost_counter <= self.config.face_lost_threshold:
                    status_text = f"Angle:{int(self.servo_angle)}° inv:{self.invert} pause:{self.paused} (holding)"
                    color = (0, 200, 200)
                else:
                    status_text = f"Angle:{int(self.servo_angle)}° inv:{self.invert} pause:{self.paused} (no face)"
                    color = (0, 150, 200)
                    self.position_smoother.clear()

                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if not self.paused:
                    self._send_angle(self.last_detected_angle, force=True)

            self._update_fps()
            if self.current_fps > 0:
                cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            self.frame_counter += 1
            return frame, tracked

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame, False

    def handle_input(self, key: int):
        if key == 0xFF:
            return

        current_time = time.time()

        if key == ord('a'):
            self.manual_velocity -= self.config.manual_speed
            self.last_manual_time = current_time
        elif key == ord('d'):
            self.manual_velocity += self.config.manual_speed
            self.last_manual_time = current_time

        elif key == ord('q'):
            return 'quit'
        elif key == ord('c'):
            self.servo_angle = float(self.config.servo_center)
            if not self.paused:
                self._send_angle(self.servo_angle)
            self.last_detected_angle = self.servo_angle
            self.position_smoother.clear()
            logger.info("Servo reset to center")
        elif key == ord('r'):
            self.servo_angle = float(self.config.servo_min)
            if not self.paused:
                self._send_angle(self.servo_angle)
            self.last_detected_angle = self.servo_angle
            self.position_smoother.clear()
            logger.info("Servo reset to minimum")
        elif key == ord('i'):
            self.invert = not self.invert
            logger.info(f"Invert: {self.invert}")
        elif key == ord('p'):
            self.paused = not self.paused
            logger.info(f"Paused: {self.paused}")

        return None

    def update_manual_control(self):
        current_time = time.time()
        manual_active = (current_time - self.last_manual_time) < self.config.auto_return_time

        if manual_active:
            self.servo_angle += self.manual_velocity
            self.manual_velocity *= self.config.manual_decay
            self.servo_angle = float(np.clip(self.servo_angle, self.config.servo_min, self.config.servo_max))

            if not self.paused:
                self._send_angle(self.servo_angle)
            self.last_detected_angle = self.servo_angle
            self.position_smoother.clear()

    def run(self):
        if not self._initialized:
            logger.error("Tracker not initialized")
            return

        logger.info("Starting face tracking loop. Press Q to quit.")
        logger.info("Controls: A/D = manual, C = center, R = reset, I = invert, P = pause")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

                annotated_frame, tracked = self.process_frame(frame)

                cv2.imshow("LBPH Face Tracker", annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                result = self.handle_input(key)
                if result == 'quit':
                    break

                self.update_manual_control()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up...")

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        if self.arduino:
            self.arduino.close()

        logger.info("Cleanup complete")


def main():
    try:
        config = TrackerConfig.from_env()
        logger.info("Configuration loaded")

        tracker = FaceTracker(config)
        if tracker.initialize():
            tracker.run()
        else:
            logger.error("Failed to initialize tracker")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()