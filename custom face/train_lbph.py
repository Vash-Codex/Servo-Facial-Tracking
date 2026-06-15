import cv2
import os
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    dataset_dir: Path
    model_file: Path
    face_size: tuple = (200, 200)
    detect_scale: float = 0.5
    min_images: int = 20  # Minimum images required for training
    min_face_size: tuple = (50, 50)  # Minimum face size to capture
    
    def validate(self):
        """Validate configuration."""
        if self.face_size[0] <= 0 or self.face_size[1] <= 0:
            raise ValueError("Face size must be positive")
        if self.min_images < 1:
            raise ValueError("Need at least 1 image for training")
        if not 0 < self.detect_scale <= 1:
            raise ValueError("Detect scale must be between 0 and 1")


class FaceCapture:
    """Captures and manages face images from camera."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.cap = None
        self.detector = None
        self.count = 0
        self.duplicates_skipped = 0
        self.last_face = None
    
    def initialize(self) -> bool:
        """Initialize camera and face detector."""
        try:
            # Create dataset directory
            self.config.dataset_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Dataset directory: {self.config.dataset_dir}")
            
            # Initialize camera with DirectShow backend on Windows
            try:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            except Exception:
                logger.info("DirectShow not available, using default backend")
                self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                raise RuntimeError("Camera not found or cannot be opened")
            
            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass  # Not all cameras support this
            
            # Warm up camera
            for _ in range(5):
                ret, _ = self.cap.read()
                if not ret:
                    raise RuntimeError("Camera warmup failed")
            
            logger.info("Camera initialized successfully")
            
            # Load face detector
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise RuntimeError("Failed to load Haar Cascade")
            
            logger.info("Face detector loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.cleanup()
            return False
    
    def _detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0, 0), fx=self.config.detect_scale, fy=self.config.detect_scale)
            
            min_size_scaled = (
                max(1, int(self.config.min_face_size[0] * self.config.detect_scale)),
                max(1, int(self.config.min_face_size[1] * self.config.detect_scale))
            )
            
            detected = self.detector.detectMultiScale(small, 1.3, 5, minSize=min_size_scaled)
            
            if len(detected) == 0:
                return [], gray
            
            # Scale coordinates back to original frame
            inv_scale = 1.0 / self.config.detect_scale
            faces = [(int(x * inv_scale), int(y * inv_scale),
                     int(w * inv_scale), int(h * inv_scale)) for (x, y, w, h) in detected]
            
            return faces, gray
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            return [], None
    
    def _is_similar_face(self, face: np.ndarray, threshold: float = 0.9) -> bool:
        """Check if face is similar to last captured (simple deduplication)."""
        if self.last_face is None:
            return False
        
        try:
            # Ensure same size for comparison
            face_resized = cv2.resize(face, self.config.face_size)
            last_resized = cv2.resize(self.last_face, self.config.face_size)
            
            # Compute histogram correlation
            hist1 = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([last_resized], [0], None, [256], [0, 256])
            
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return correlation > threshold
        except Exception as e:
            logger.debug(f"Similarity check error: {e}")
            return False
    
    def capture_face(self, face: np.ndarray) -> bool:
        """Capture and save a face image."""
        try:
            # Check for duplicates
            if self._is_similar_face(face):
                self.duplicates_skipped += 1
                logger.debug(f"Skipped duplicate face (total skipped: {self.duplicates_skipped})")
                return False
            
            # Resize to consistent size
            face_resized = cv2.resize(face, self.config.face_size)
            
            # Save image
            filename = f"user_{self.count:04d}.jpg"
            filepath = self.config.dataset_dir / filename
            cv2.imwrite(str(filepath), face_resized)
            
            self.count += 1
            self.last_face = face.copy()
            logger.info(f"Captured image {self.count}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to capture face: {e}")
            return False
    
    def run(self):
        """Main capture loop."""
        logger.info("Starting face capture. Controls:")
        logger.info("  SPACE = capture image")
        logger.info("  Q = quit and train")
        logger.info(f"  Target: at least {self.config.min_images} images")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Detect faces
                faces, gray = self._detect_faces(frame)
                
                if faces and gray is not None:
                    # Draw all detected faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Show status
                    status = f"Detected {len(faces)} face(s) | Captured: {self.count}/{self.config.min_images}"
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show progress
                progress = f"Captured: {self.count}/{self.config.min_images}"
                cv2.putText(frame, progress, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                cv2.imshow("Collecting Faces", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(" "):  # SPACE = capture
                    if faces and gray is not None:
                        # Extract first face
                        x, y, w, h = faces[0]
                        if 0 <= x and 0 <= y and x+w < gray.shape[1] and y+h < gray.shape[0]:
                            face = gray[y:y+h, x:x+w]
                            self.capture_face(face)
                        else:
                            logger.warning("Face coordinates out of bounds")
                    else:
                        logger.info("No face detected - try again")
                
                elif key == ord("q"):
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Capture error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


class LBPHTrainer:
    """Trains LBPH face recognition model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.recognizer = None
    
    def load_training_data(self) -> tuple:
        """Load training images and labels."""
        try:
            images = []
            labels = []
            
            if not self.config.dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {self.config.dataset_dir}")
            
            # Load all images
            image_files = sorted(self.config.dataset_dir.glob("*.jpg"))
            
            if len(image_files) == 0:
                raise ValueError("No training images found")
            
            logger.info(f"Found {len(image_files)} images")
            
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        logger.warning(f"Failed to load: {img_path}")
                        continue
                    
                    # Ensure consistent size
                    img = cv2.resize(img, self.config.face_size)
                    images.append(img)
                    labels.append(0)  # Label 0 = recognized user
                
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
            
            if len(images) == 0:
                raise ValueError("No valid training images loaded")
            
            logger.info(f"Loaded {len(images)} images for training")
            return images, np.array(labels)
        
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return [], np.array([])
    
    def train(self) -> bool:
        """Train LBPH model."""
        try:
            # Load data
            images, labels = self.load_training_data()
            
            if len(images) < self.config.min_images:
                logger.error(f"Not enough images. Need {self.config.min_images}, got {len(images)}")
                return False
            
            # Create and train recognizer
            logger.info("Training LBPH model...")
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(images, labels)
            
            # Save model
            self.config.model_file.parent.mkdir(parents=True, exist_ok=True)
            self.recognizer.save(str(self.config.model_file))
            
            logger.info(f"Model trained and saved to {self.config.model_file}")
            logger.info(f"Training complete: {len(images)} images, 1 label (recognized user)")
            return True
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False


def main():
    """Main entry point."""
    try:
        # Setup paths
        base_dir = Path(__file__).resolve().parent
        dataset_dir = base_dir.parent / "dataset"
        model_file = base_dir / "face_model.xml"
        
        # Create config
        config = TrainingConfig(
            dataset_dir=dataset_dir,
            model_file=model_file
        )
        config.validate()
        
        # Capture phase
        logger.info("=== Face Capture Phase ===")
        capturer = FaceCapture(config)
        if capturer.initialize():
            capturer.run()
            logger.info(f"Captured {capturer.count} images (skipped {capturer.duplicates_skipped} duplicates)")
        else:
            logger.error("Failed to initialize capturer")
            return
        
        # Training phase
        logger.info("\n=== Training Phase ===")
        trainer = LBPHTrainer(config)
        if trainer.train():
            logger.info("✓ Training successful")
        else:
            logger.error("✗ Training failed")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
