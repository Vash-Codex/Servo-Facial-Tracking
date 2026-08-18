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
except ImportError:
    serial = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def get_cascade_path(filename: str = "haarcascade_frontalface_default.xml") -> str:
    """Resolve absolute path to Haar cascade XML file, supporting PyInstaller bundles."""
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)
        candidates.extend([
            meipass / "cv2" / "data" / filename,
            meipass / "cv2" / "data" / "data" / filename,
            meipass / filename,
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
    """Return the application root directory.

    When frozen as a PyInstaller onefile EXE, ``__file__`` resolves into the
    temp ``_MEIPASS`` directory — NOT next to the EXE.  Use
    ``sys.executable`` so user data (model) is always found next to the EXE.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Dev mode: this file lives in "custom face/", parent is project root.
    return Path(__file__).resolve().parent.parent


def _get_model_path(model_filename: str) -> Path:
    """Resolve the LBPH model file path.

    Priority:
    1. User-trained model next to the EXE  (``<app_dir>/custom face/<file>``)
    2. Bundled model inside ``_MEIPASS``    (first-run before user trains)
    3. Falls back to the user path (will raise FileNotFoundError if missing)
    """
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
    return user_model  # Caller will raise FileNotFoundError if missing


def env_flag(name, default=False):
    """Parse environment variable as boolean."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TrackerConfig:
    """Configuration for face tracking."""
    # Arduino settings
    use_arduino: bool = False
    com_port: str = "COM3"
    baud_rate: int = 9600
    
    # Servo settings
    servo_min: int = 45
    servo_max: int = 135
    servo_center: int = 90
    
    # Tracking parameters
    deadzone_pct: float = 0.04
    smooth_factor: float = 0.10
    send_min_ms: int = 20
    angle_step_min: float = 0.5
    max_velocity: float = 2.0
    confidence_threshold: int = 70
    
    # Momentum/manual control
    manual_speed: float = 1.5
    manual_decay: float = 0.85
    auto_return_time: float = 0.6
    
    # Detection settings
    detect_scale: float = 0.5
    detect_every_n: int = 2
    min_face_size: Tuple[int, int] = (30, 30)
    
    # Smoothing
    position_history_size: int = 5
    face_lost_threshold: int = 10
    
    # Camera settings
    camera_width: int = 1280
    camera_height: int = 720
    
    # UI
    mirror_preview: bool = True
    model_file: str = "face_model.xml"
    
    def validate(self):
        """Validate configuration values."""
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
        """Load configuration from environment variables."""
        return cls(
            use_arduino=env_flag("FACE_TRACKER_USE_ARDUINO", False),
            com_port=os.getenv("FACE_TRACKER_COM_PORT", "COM3").strip() or "COM3",
            confidence_threshold=int(os.getenv("FACE_TRACKER_CONFIDENCE", "70")),
        )


class ArduinoController:
    """Manages Arduino serial communication."""
    
    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_connected = False
    
    def connect(self) -> bool:
        """Attempt to connect to Arduino."""
        if serial is None:
            logger.warning("PySerial not installed. Running without Arduino.")
            return False
        
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0)
            time.sleep(2)  # Wait for Arduino to initialize
            self.is_connected = True
            logger.info(f"Connected to Arduino on {self.port}")
            return True
        except Exception as exc:
            logger.warning(f"Could not connect to Arduino on {self.port}: {exc}")
            self.is_connected = False
            return False
    
    def send_angle(self, angle: int) -> bool:
        """Send angle to Arduino."""
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
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            try:
                self.serial.close()
            except Exception:
                pass


class PositionSmoother:
    """Smooths position measurements using history averaging."""
    
    def __init__(self, history_size: int = 5):
        self.history = deque(maxlen=history_size)
    
    def add(self, position: float) -> float:
        """Add position and return smoothed average."""
        self.history.append(position)
        return np.mean(self.history)
    
    def clear(self):
        """Clear history."""
        self.history.clear()
    
    def is_valid(self) -> bool:
        """Check if we have enough history for reliable averaging."""
        return len(self.history) >= 2


class FaceTracker:
    """Main face tracking controller."""
    
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.config.validate()
        
        # Initialize components
        self.cap = None
        self.face_cascade = None
        self.recognizer = None
        self.arduino = None
        
        self.position_smoother = PositionSmoother(config.position_history_size)
        
        # State variables
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
        
        # Performance tracking
        self.fps_times = deque(maxlen=30)
        self.current_fps = 0.0
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize camera, models, and connections."""
        try:
            # Load face cascade
            cascade_path = get_cascade_path("haarcascade_frontalface_default.xml")
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load Haar Cascade classifier")
            logger.info("Loaded Haar Cascade classifier")
            
            # Initialize camera
            if not self._init_camera():
                return False
            
            # Load LBPH model
            if not self._load_model():
                return False
            
            # Connect to Arduino if enabled
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
        """Initialize camera capture."""
        try:
            # Try DirectShow on Windows, fallback to default
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                raise RuntimeError("Camera not found or cannot be opened")
            
            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            
            # Reduce buffer size for lower latency
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass  # Not all cameras support this
            
            # Warm up camera
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
        """Load LBPH face recognizer model."""
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
        """Send angle to Arduino with throttling."""
        if not self.arduino or not self.arduino.is_connected:
            return
        
        now = time.time() * 1000
        angle = float(np.clip(angle, self.config.servo_min, self.config.servo_max))
        
        # Check if we should send
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
        """Smooth easing toward target."""
        if factor is None:
            factor = self.config.smooth_factor
        return current + (target - current) * factor
    
    def _detect_faces(self, gray: np.ndarray) -> list:
        """Detect faces in grayscale image."""
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
            
            # Scale coordinates back to original size
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
        """Validate face rectangle bounds."""
        return fw > 0 and fh > 0 and x >= 0 and y >= 0 and x + fw <= frame_width and y + fh <= frame_height
    
    def _recognize_face(self, face_gray: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Recognize face in ROI. Returns (label, confidence) or (None, None)."""
        try:
            label, confidence = self.recognizer.predict(face_gray)
            return label, confidence
        except Exception as e:
            logger.debug(f"Recognition error: {e}")
            return None, None
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_times.append(time.time())
        if len(self.fps_times) >= 2:
            elapsed = self.fps_times[-1] - self.fps_times[0]
            if elapsed > 0:
                self.current_fps = len(self.fps_times) / elapsed
    
    def _calculate_target_angle(self, face_x: int, face_width: int, frame_center_x: int, frame_width: int) -> float:
        """Calculate target servo angle from face position."""
        avg_cx = self.position_smoother.add(face_x)
        
        # Interpolate angle based on face position
        if self.invert:
            target = np.interp(avg_cx, [0, frame_width], [self.config.servo_max, self.config.servo_min])
        else:
            target = np.interp(avg_cx, [0, frame_width], [self.config.servo_min, self.config.servo_max])
        
        # Apply deadzone
        deadzone_px = int(frame_width * self.config.deadzone_pct)
        if abs(avg_cx - frame_center_x) <= deadzone_px:
            target = self.servo_angle
        
        # Apply velocity limiting to prevent sudden jumps
        angle_diff = target - self.servo_angle
        if abs(angle_diff) > self.config.max_velocity:
            target = self.servo_angle + np.sign(angle_diff) * self.config.max_velocity
        
        return target
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process single frame. Returns (annotated_frame, face_tracked)."""
        try:
            if frame is None or frame.size == 0:
                logger.warning("Invalid frame received")
                return frame, False
            
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                logger.warning("Invalid frame dimensions")
                return frame, False
            
            # Mirror if configured
            if self.config.mirror_preview:
                frame = cv2.flip(frame, 1)
            
            # Update timing
            current_time = time.time()
            manual_active = (current_time - self.last_manual_time) < self.config.auto_return_time
            
            # Convert to grayscale
            try:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.warning(f"Color conversion failed: {e}")
                return frame, False
            
            # Detect faces
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
            
            # Track face
            tracked = False
            frame_center_x = w // 2
            
            for (x, y, fw, fh) in faces:
                if not self._is_valid_face_rect(x, y, fw, fh, w, h):
                    continue
                
                # Extract and recognize face
                face_gray = gray_full[y:y+fh, x:x+fw]
                label, confidence = self._recognize_face(face_gray)
                
                if label is None or confidence is None:
                    continue
                
                # Check if recognized (label 0 = self, confidence below threshold = recognized)
                if label == 0 and confidence < self.config.confidence_threshold and not manual_active:
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                    cv2.putText(frame, f"YOU ({confidence:.1f})", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Calculate and apply target angle
                    face_center_x = x + fw // 2
                    target = self._calculate_target_angle(face_center_x, fw, frame_center_x, w)
                    
                    # Smooth and send angle
                    self.servo_angle = self._ease_smooth(self.servo_angle, target)
                    
                    if not self.paused:
                        self._send_angle(self.servo_angle)
                        self.last_detected_angle = self.servo_angle
                    
                    # Draw status
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
            
            # Handle face lost state
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
            
            # Draw FPS
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
        """Handle keyboard input."""
        if key == 0xFF:  # No key pressed
            return
        
        current_time = time.time()
        
        # Manual control (A/D)
        if key == ord('a'):
            self.manual_velocity -= self.config.manual_speed
            self.last_manual_time = current_time
        elif key == ord('d'):
            self.manual_velocity += self.config.manual_speed
            self.last_manual_time = current_time
        
        # Global controls
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
        """Update manual control momentum."""
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
        """Main tracking loop."""
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
                
                # Process frame
                annotated_frame, tracked = self.process_frame(frame)
                
                # Display
                cv2.imshow("LBPH Face Tracker", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                
                # Handle input
                result = self.handle_input(key)
                if result == 'quit':
                    break
                
                # Update manual control
                self.update_manual_control()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.arduino:
            self.arduino.close()
        
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = TrackerConfig.from_env()
        logger.info("Configuration loaded")
        
        # Create and run tracker
        tracker = FaceTracker(config)
        if tracker.initialize():
            tracker.run()
        else:
            logger.error("Failed to initialize tracker")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()