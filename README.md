Servo Facial Tracking

A Python and Arduino-based system that detects and tracks a face using a webcam and moves a servo according to the person's horizontal movement.

The project uses OpenCV for face detection and a custom LBPH model to recognize the target face. A Tkinter GUI is included to make dataset collection, training, serial settings, and tracking easier to manage.

Features
Face detection and tracking using OpenCV
Custom LBPH face recognition
Servo follows only the trained/recognized face
Smooth servo movement with deadzone and position averaging
Tkinter GUI for controlling the project
Manual controls using keyboard
Works completely offline
Supports basic tracking without face recognition
Hardware
Arduino Uno or Nano
SG90 / MG90S servo
Webcam
Computer running Python
Servo wiring
Signal → D9
VCC → 5V
GND → GND

For larger servos, use an external 5V supply and connect its GND to the Arduino GND.

The Arduino communicates with the Python program at 9600 baud.

Project Structure
face tracker/
├── face_tracker_gui.py
├── face.py
├── requirements.txt
├── README.md
│
├── custom face/
│   ├── train_lbph.py
│   ├── face_tracker_lbph.py
│   └── face_model.xml
│
├── dataset/
├── facearduino/
│   └── facearduino.ino
│
└── vids/

face_model.xml and the dataset folder are generated locally while training.

Setup

Create a virtual environment and install the requirements:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Make sure opencv-contrib-python is installed, since the normal OpenCV package does not include the LBPH face recognizer.

Upload facearduino/facearduino.ino to the Arduino and connect the servo to D9.

Then start the dashboard:

python face_tracker_gui.py
Training

Start the training program and press Space to capture face samples. Around 20–50 samples from different angles and lighting conditions usually work well.

Press Q when finished. The program will train the LBPH model and create face_model.xml.

Controls

Tracker:

Q — Quit
C — Center servo
R — Move to minimum angle
I — Invert direction
P — Pause/resume servo
A / D — Manually move left/right

The default servo range is 45°–135°, with 90° as the center position.

Troubleshooting

If cv2.face is missing, remove opencv-python and install opencv-contrib-python.

If the model cannot be found, run the training program first.

If the Arduino cannot connect, check the COM port and make sure the Arduino Serial Monitor is closed.

If the servo jitters or the Arduino resets, use a separate 5V power supply for the servo and connect the grounds together.

License

MIT License

Repository: vash-codex/Servo-Facial-Tracking
