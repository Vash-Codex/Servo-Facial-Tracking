
```markdown
# Servo Facial Tracking Dashboard

A Python-powered face recognition and physical tracking system. This project utilizes computer vision to capture and train a custom Local Binary Patterns Histograms (LBPH) biometric model, track your face in real-time using an OpenCV Haar Cascade, and smoothly steer a hardware Arduino servo to follow your movements. 

The entire workflow is managed through a modern, themed Tkinter GUI desktop controller dashboard.

---

## 🚀 Features

- **Tkinter Parent Dashboard:** A clean, modern desktop graphical interface with multiple selectable color themes (Midnight, Daylight, Mint, Rose) to configure settings and launch sub-modules easily.
- **Biometric Training Suite:** Capture personal face samples locally with visual feedback and train a private LBPH recognition model (`face_model.xml`).
- **Real-Time Video Processing:** High-performance face detection using OpenCV Haar Cascades combined with custom LBPH verification.
- **Hardware Integration:** Translates coordinates to mapping angles ($45^\circ - 135^\circ$) and streams commands over serial (`pySerial`) to an Arduino board.
- **Privacy-First Architecture:** All training image datasets, biometric model XMLs, and video recordings remain strictly on your local machine.

---

## 🛠️ System Architecture & Workflow


```
[ Webcam Input ]
│
▼
[ Grayscale / Haar Cascade Detection ] ──► [ Local LBPH Recognition Check ]
│
▼
[ Coordinate Smoothing & Angle Mapping ]
│
▼ (Via Configuration GUI over Serial)
[ Arduino Nano/Uno / Pin D9 ] ──► [ Servo Movement (45°-135°) ]
```

1. **Capture:** Run the trainer through the GUI dashboard to crop and save localized face samples.
2. **Train:** The pipeline processes the images, assigns labels, and compiles your local weights file.
3. **Track:** The tracker reads your video stream, ensures authentication matching, smooths target jitter, and calculates dynamic pan/tilt vectors.
4. **Control:** Hardware flags (`FACE_TRACKER_USE_ARDUINO`) and chosen serial communication lines (e.g., `COM10`) are cleanly passed from the master dashboard script.

---

## 📁 Repository Structure

```text
├── custom face/
│   ├── face_tracker_lbph.py   # Core facial recognition tracker
│   └── train_lbph.py          # Captures dataset frames and outputs trained weights
├── facearduino/
│   └── facearduino.ino        # Arduino C++ sketch managing servo hardware lines
├── face_tracker_gui.py        # Central master Tkinter application dashboard
├── face.py                    # Independent basic webcam tracker snippet
├── requirements.txt           # Specified library dependencies
└── README.md                  # System instruction sheet

```
*Note: The project automatically generates the private local directories dataset/ and custom face/face_model.xml during runtime configuration.*
## ⚙️ Installation & Setup Guide
### 1. Project Initialization
Clone this repository or extract the downloaded source files. Keep the file directory layout intact so the GUI relative paths find their corresponding sub-modules:
```bash
git clone [https://github.com/Vash-Codex/Servo-Facial-Tracking.git](https://github.com/Vash-Codex/Servo-Facial-Tracking.git)
cd Servo-Facial-Tracking

```
### 2. Configure Python Virtual Environment
It is highly recommended to isolate the project dependencies using a python virtual environment:
```bash
# Create environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source .venv/bin/activate

# Upgrade pip and install required components
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```
### 3. Setup the Hardware Link (Arduino)
 1. Launch your **Arduino IDE**.
 2. Open the file facearduino/facearduino.ino.
 3. Connect your micro-controller to your machine.
 4. Wiring layout:
   * Connect the Servo **Signal Wire** to Digital Pin **9**.
   * Connect the Servo **Power (VCC)** and **Ground (GND)** lines to the appropriate board rails.
 5. Upload the code file. Take note of the active device port assigned by the IDE (e.g., COM10 or /dev/ttyUSB0).
## 🎮 Execution & Runtime Operation
Run the main application using the virtual environment interpreter:
```bash
python face_tracker_gui.py

```
### 🔹 Interface Steps
 1. Select your application color layout styling (Midnight, Daylight, Mint, Rose).
 2. Enter your Arduino's identified **COM Port** name in the input box and toggle **Arduino Access** if hardware integration is ready.
 3. Launch the **Trainer Script**. Face the camera and use your runtime controls to build your data footprint.
 4. Close the training script to auto-generate your local model, then click **Run LBPH Tracker**.
### ⌨️ Interactive Command Keys
| Mode | Key Binding | Action Command |
|---|---|---|
| **Training Pipeline** | Spacebar | Captures a bounding-box face sample frame into dataset/ |
|  | Q | Halts sampling, executes compiler, and saves model file |
| **Tracking Module** | Q | Gracefully closes tracking module window feed |
|  | C | Immediately centers tracking target angle vectors |
|  | R | Shifts servo axis boundary back to the minimum angle configuration |
|  | I | Inverts hardware orientation tracking steering loops |
|  | P | Pauses real-time dynamic hardware communication packets |
| **Manual Override** | A / D | Manually nudges the servo angle configuration using momentum vectors |
## 🔒 Privacy & Local Security Notice
Biometric records, visual images compiled in your dataset/ folder, generated training weights, and tracking demo video recordings remain completely offline. Ensure you do not accidentally commit or push your generated .xml or image directories to public platforms.
## 📄 License
This project is open-source and available under the MIT License.
```
https://vash-codex.github.io/Servo-Facial-Tracking/#setup
```
