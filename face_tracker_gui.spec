# -*- mode: python ; coding: utf-8 -*-
#
# face_tracker_gui.spec
# PyInstaller build spec for Face Tracker Control Center.
#
# Produces ONE dist/FaceTrackerGUI.exe  (single self-contained file).
#
# The GUI re-launches this same EXE with --mode <name> when the user clicks
# a tracker/trainer button, so only ONE file needs to be distributed.
#
# Build with:
#   pyinstaller face_tracker_gui.spec
#
# Distribute: just share dist/FaceTrackerGUI.exe — no other files needed
# (except placing dataset/ and custom face/ next to the EXE for user data).

import sys
from pathlib import Path
import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(SPECPATH)

# OpenCV ships its Haar cascade XMLs inside the package data directory.
CV2_HAAR_DIR = Path(cv2.data.haarcascades)

# ---------------------------------------------------------------------------
# Bundled data
#   cv2/data      – Haar cascade XML files (read by cv2.data.haarcascades)
#   facearduino   – Arduino sketch (opened by the "Arduino" shortcut button)
#   custom face   – face_model.xml bundled so LBPH tracker works first-run
#                   before the user trains their own model
# ---------------------------------------------------------------------------
CASCADE_LOCAL  = (str(HERE / "haarcascade_frontalface_default.xml"), ".")
CASCADE_DATA_1 = (str(CV2_HAAR_DIR), "cv2/data")
CASCADE_DATA_2 = (str(CV2_HAAR_DIR), "cv2/data/data")
CASCADE_DATA_3 = (str(CV2_HAAR_DIR), ".")
ARDUINO_DATA    = (str(HERE / "facearduino"),   "facearduino")
CUSTOM_DIR_DATA = (str(HERE / "custom face"),   "custom face")

# Bundle dataset dir only if it exists (it may be empty / absent on a clean checkout)
_dataset_src = HERE / "dataset"
DATASET_DATA = (str(_dataset_src), "dataset") if _dataset_src.exists() else None

datas = [CASCADE_LOCAL, CASCADE_DATA_1, CASCADE_DATA_2, CASCADE_DATA_3, ARDUINO_DATA, CUSTOM_DIR_DATA]
if DATASET_DATA:
    datas.append(DATASET_DATA)

# ---------------------------------------------------------------------------
# Hidden imports
#   The three sub-scripts are imported at runtime via --mode dispatch.
#   PyInstaller won't see those imports during static analysis, so we list
#   them explicitly.  The "custom face" directory is added to pathex so
#   train_lbph and face_tracker_lbph are importable as top-level modules.
# ---------------------------------------------------------------------------
HIDDEN = [
    # Sub-scripts (dispatched at runtime)
    "face",
    "train_lbph",
    "face_tracker_lbph",
    # PySerial
    "serial",
    "serial.tools",
    "serial.tools.list_ports",
    "serial.tools.list_ports_common",
    "serial.tools.list_ports_windows",
    # OpenCV
    "cv2",
]

EXCLUDES = [
    "matplotlib", "scipy", "pandas", "IPython",
    "PIL", "Pillow", "PyQt5", "PyQt6", "PySide2", "PySide6",
]

# ---------------------------------------------------------------------------
# Single Analysis — all scripts in one build
# ---------------------------------------------------------------------------
a = Analysis(
    [str(HERE / "face_tracker_gui.py")],
    pathex=[
        str(HERE),                    # face.py lives here
        str(HERE / "custom face"),    # train_lbph.py and face_tracker_lbph.py live here
    ],
    binaries=[],
    datas=datas,
    hiddenimports=HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# Single onefile EXE
#   console=False – no CMD window.  Sub-mode subprocesses (--mode train etc.)
#                   restore their own stdout from the pipe file descriptor
#                   inside _dispatch_submode() so the GUI log panel still
#                   captures all output correctly.
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FaceTrackerGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,   # no CMD window — sub-mode stdout is restored from fd
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
