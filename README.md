# Servo Facial Tracking

> [!CAUTION]
> **AI Usage Notice**: Parts of this project, including GUI design, and debugging were developed with the assistance of AI tools. Please review code and test hardware setups carefully before deployment.

A face recognition and camera-tracking system built with Python, OpenCV, and Arduino. It trains a custom LBPH (Local Binary Patterns Histograms) model on your face using a webcam, tracks your movement, and sends pan angles over serial to an Arduino-controlled servo motor.

Includes a Tkinter GUI dashboard (`face_tracker_gui.py`) to manage dataset capture, model training, serial settings, and tracker execution.

---

## Features

- **Tkinter Dashboard**: Manage camera capture, training, and tracking from a single interface.
- **Custom Face Recognition**: Collects face images locally to build a personal LBPH recognizer (`face_model.xml`).
- **Identity-Filtered Tracking**: Servo only follows your face (ignores unrecognized faces).
- **Smoothed Motor Movement**: Deadzone, acceleration limiting, and position averaging reduce servo jitter.
- **Manual & Basic Tracking Modes**: Supports manual keyboard overrides (`A`/`D` keys) and simple face-tracking without LBPH training (`face.py`).
- **Offline & Private**: All image processing and model files stay local.

---

## Hardware & Wiring

- **Microcontroller**: Arduino Uno or Nano
- **Servo Motor**: SG90, MG90S, or similar (180° range)
- **Connections**:
  - **Signal**: Pin **D9** (Orange/Yellow wire)
  - **Power**: **5V** (Red wire) — *use an external 5V supply with shared GND for larger servos*
  - **Ground**: **GND** (Brown/Black wire)

**Serial Protocol**: 9600 baud. The tracker maps horizontal face position across the camera frame to servo angles (default **45°–135°**) and sends integer angles ended with newline (`\n`).

---

## Project Structure

```
face tracker/
├── face_tracker_gui.py          # Main GUI dashboard
├── face.py                      # Basic Haar-cascade face tracker
├── requirements.txt             # Python dependencies
├── README.md
│
├── custom face/
│   ├── train_lbph.py            # Dataset collector & model trainer
│   ├── face_tracker_lbph.py     # LBPH face tracker & serial controller
│   └── face_model.xml           # Trained model (generated locally)
│
├── dataset/                     # Captured face crops (generated locally)
├── facearduino/
│   └── facearduino.ino          # Arduino C++ sketch (pin D9, 9600 baud)
└── vids/                        # Recorded demo videos (optional)
```

---

## Quick Setup

### 1. Installation

Requires Python 3.10+ and `opencv-contrib-python`.

```powershell
# Navigate to project directory
cd "C:\path\to\face tracker"

# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

> **Note**: `opencv-contrib-python` is required (not standard `opencv-python`) because it includes the OpenCV `face` module (`LBPHFaceRecognizer`).

### 2. Arduino Setup

1. Open `facearduino/facearduino.ino` in the Arduino IDE.
2. Connect your Arduino board via USB and wire the servo signal to pin **D9**.
3. Select your board and COM port, then click **Upload**.
4. Note your COM port (e.g., `COM10`).

---

## Usage

Run the GUI dashboard:

```powershell
python face_tracker_gui.py
```

### Workflow

1. **Train Model**:
   - Click **Start Training** (or run `python "custom face/train_lbph.py"`).
   - Press **Space** to capture face samples (~20–50 samples recommended under various lighting/angles).
   - Press **Q** to finish and generate `face_model.xml`.
2. **Configure Hardware**:
   - Enter your Arduino COM port (e.g., `COM10`) and toggle **Arduino ON**.
3. **Start Tracker**:
   - Click **Start Tracker** (or run `python "custom face/face_tracker_lbph.py"`).
   - The camera will detect your face, verify identity, and pan the servo motor.

---

## Keyboard Controls

### Training Mode (`train_lbph.py`)
- `Space`: Capture current face crop to `dataset/`
- `Q`: Stop capture, train model, save `face_model.xml`

### LBPH Tracker (`face_tracker_lbph.py`)
- `Q`: Quit tracker
- `C`: Center servo (90°)
- `R`: Move servo to minimum angle (45°)
- `I`: Invert left/right mapping
- `P`: Pause/resume servo serial output
- `A` / `D`: Manual nudge left / right

### Basic Tracker (`face.py`)
- `Q`: Quit tracker
- `C`: Center servo (90°)
- `R`: Move servo to 0°
- `I`: Invert mapping
- `P`: Pause output
- `S`: Toggle servo enable

---

## Configuration Reference

Settings can be adjusted in script headers or passed via environment variables:

| Setting / Variable | Description | Default |
|--------------------|-------------|---------|
| `FACE_TRACKER_USE_ARDUINO` | Enable or disable serial output | `0` / `1` |
| `FACE_TRACKER_COM_PORT` | Target serial port | `COM10` |
| `SERVO_MIN` / `SERVO_MAX` | Min/max tracker angle range | `45` / `135` |
| `CENTER` | Home angle | `90` |
| `DEADZONE_PCT` | Deadzone fraction of frame width | `0.04` |

---

## Troubleshooting

- **`AttributeError: module 'cv2' has no attribute 'face'`**
  - Uninstall `opencv-python` and install `opencv-contrib-python`.
- **`Model file not found`**
  - Run the trainer (`train_lbph.py`) and capture samples before launching the LBPH tracker.
- **Serial / COM Port Error**
  - Close the Arduino IDE Serial Monitor before starting the tracker, and double-check the COM port in Windows Device Manager (**Ports (COM & LPT)**).
- **Servo Stuttering or Board Resetting**
  - Servos draw high peak currents. Use an external 5V power supply and connect its ground to Arduino GND.

---

## License & Links

- **License**: [MIT License](LICENSE)
- **Repository**: [vash-codex/Servo-Facial-Tracking](https://github.com/vash-codex/Servo-Facial-Tracking)

