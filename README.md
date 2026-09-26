# Servo Facial Tracking

A Python and Arduino-based system that detects and tracks a face using a webcam and moves a servo according to the person's horizontal movement.

It uses **OpenCV** for face detection and a custom **LBPH model** for face recognition. A **Tkinter GUI** is included for dataset collection, training, serial settings, and tracking.

## Features

* Face detection and tracking
* Custom LBPH face recognition
* Servo follows the trained face
* Smooth servo movement
* Tkinter GUI
* Manual keyboard controls
* Offline operation
* Tracking without face recognition

## Hardware

* Arduino Uno/Nano
* SG90/MG90S servo
* Webcam
* Computer running Python

### Servo Wiring

```text
Signal → D9
VCC    → 5V
GND    → GND
```

For larger servos, use an external 5V supply and connect its GND to Arduino GND.

The Arduino communicates at **9600 baud**.

## Project Structure

```text
Servo-Facial-Tracking/
├── face_tracker_gui.py
├── face.py
├── requirements.txt
├── custom face/
│   ├── train_lbph.py
│   ├── face_tracker_lbph.py
│   └── face_model.xml
├── dataset/
├── facearduino/
│   └── facearduino.ino
└── vids/
```

`face_model.xml` and `dataset/` are generated during training.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Make sure `opencv-contrib-python` is installed because it provides the LBPH face recognizer.

Upload:

```text
facearduino/facearduino.ino
```

to the Arduino and connect the servo to **D9**.

Start the GUI:

```bash
python face_tracker_gui.py
```

## Training

Run the training program and press **Space** to capture face samples.

Around **20–50 samples** from different angles and lighting conditions usually work well.

Press **Q** when finished. The program will create `face_model.xml`.

## Controls

| Key   | Action                |
| ----- | --------------------- |
| Q     | Quit                  |
| C     | Center servo          |
| R     | Move to minimum angle |
| I     | Invert direction      |
| P     | Pause/resume          |
| A / D | Move left/right       |

Default servo range: **45°–135°**
Center position: **90°**

## Troubleshooting

**`cv2.face` is missing**

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

**Model not found**

Run the training program first.

**Arduino won't connect**

Check the COM port and make sure the Arduino Serial Monitor is closed.

**Servo jitters or Arduino resets**

Use a separate 5V power supply for the servo and connect the grounds together.

## License

MIT License

## Repository

[Vash-Codex/Servo-Facial-Tracking](https://github.com/Vash-Codex/Servo-Facial-Tracking)
