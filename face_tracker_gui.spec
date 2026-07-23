# -*- mode: python ; coding: utf-8 -*-
#
# face_tracker_gui.spec
# PyInstaller build spec for Face Tracker Control Center.
#
# Produces ONE dist/FaceTrackerGUI/ folder containing four executables:
#   FaceTrackerGUI.exe      – main launcher (Tkinter GUI, no console)
#   TrainFace.exe           – face capture + LBPH training
#   FaceTrackerLBPH.exe     – LBPH recognition + servo tracking
#   FaceTrackerBasic.exe    – basic Haar cascade tracker
#
# The GUI launches the sub-exes as subprocesses when frozen, so each
# script gets its own bundled Python interpreter – no system Python needed.
#
# Build with:
#   pyinstaller face_tracker_gui.spec
#
# Distribute: zip the entire dist/FaceTrackerGUI/ folder.

import sys
from pathlib import Path
import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(SPECPATH)

# OpenCV ships its Haar cascade XMLs inside the package data directory.
CV2_DATA_DIR = Path(cv2.data.haarcascades).parent

# ---------------------------------------------------------------------------
# Shared data: Haar cascades + Arduino sketch
# (each Analysis that needs cascades gets its own copy via its datas list)
# ---------------------------------------------------------------------------
CASCADE_DATA  = (str(CV2_DATA_DIR), "cv2/data")
ARDUINO_DATA  = (str(HERE / "facearduino"), "facearduino")
CUSTOM_DIR_DATA = (str(HERE / "custom face"), "custom face")

HIDDEN = [
    "serial",
    "serial.tools",
    "serial.tools.list_ports",
    "serial.tools.list_ports_common",
    "serial.tools.list_ports_windows",
    "cv2",
]

EXCLUDES = [
    "matplotlib", "scipy", "pandas", "IPython",
    "PIL", "Pillow", "PyQt5", "PyQt6", "PySide2", "PySide6",
]

# ---------------------------------------------------------------------------
# 1. Main GUI  (FaceTrackerGUI.exe – windowed, no console)
# ---------------------------------------------------------------------------
a_gui = Analysis(
    [str(HERE / "face_tracker_gui.py")],
    pathex=[str(HERE)],
    binaries=[],
    datas=[CASCADE_DATA, CUSTOM_DIR_DATA, ARDUINO_DATA],
    hiddenimports=HIDDEN,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=1,
)
pyz_gui = PYZ(a_gui.pure)
exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name="FaceTrackerGUI",
    debug=False,
    strip=False,
    upx=True,
    console=False,   # GUI – hide console
    disable_windowed_traceback=False,
)

# ---------------------------------------------------------------------------
# 2. Train Face  (TrainFace.exe – console so training output is visible)
# ---------------------------------------------------------------------------
a_train = Analysis(
    [str(HERE / "custom face" / "train_lbph.py")],
    pathex=[str(HERE), str(HERE / "custom face")],
    binaries=[],
    datas=[CASCADE_DATA, CUSTOM_DIR_DATA],
    hiddenimports=HIDDEN,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=1,
)
pyz_train = PYZ(a_train.pure)
exe_train = EXE(
    pyz_train,
    a_train.scripts,
    [],
    exclude_binaries=True,
    name="TrainFace",
    debug=False,
    strip=False,
    upx=True,
    console=True,   # show training output
)

# ---------------------------------------------------------------------------
# 3. LBPH Tracker  (FaceTrackerLBPH.exe – console for live output)
# ---------------------------------------------------------------------------
a_lbph = Analysis(
    [str(HERE / "custom face" / "face_tracker_lbph.py")],
    pathex=[str(HERE), str(HERE / "custom face")],
    binaries=[],
    datas=[CASCADE_DATA, CUSTOM_DIR_DATA],
    hiddenimports=HIDDEN,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=1,
)
pyz_lbph = PYZ(a_lbph.pure)
exe_lbph = EXE(
    pyz_lbph,
    a_lbph.scripts,
    [],
    exclude_binaries=True,
    name="FaceTrackerLBPH",
    debug=False,
    strip=False,
    upx=True,
    console=True,
)

# ---------------------------------------------------------------------------
# 4. Basic Tracker  (FaceTrackerBasic.exe – console for live output)
# ---------------------------------------------------------------------------
a_basic = Analysis(
    [str(HERE / "face.py")],
    pathex=[str(HERE)],
    binaries=[],
    datas=[CASCADE_DATA],
    hiddenimports=HIDDEN,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=1,
)
pyz_basic = PYZ(a_basic.pure)
exe_basic = EXE(
    pyz_basic,
    a_basic.scripts,
    [],
    exclude_binaries=True,
    name="FaceTrackerBasic",
    debug=False,
    strip=False,
    upx=True,
    console=True,
)

# ---------------------------------------------------------------------------
# Collect all four exes into a single dist/FaceTrackerGUI/ folder.
# PyInstaller deduplicates shared DLLs automatically.
# ---------------------------------------------------------------------------
coll = COLLECT(
    # GUI
    exe_gui,
    a_gui.binaries,
    a_gui.datas,
    # Trainer
    exe_train,
    a_train.binaries,
    a_train.datas,
    # LBPH tracker
    exe_lbph,
    a_lbph.binaries,
    a_lbph.datas,
    # Basic tracker
    exe_basic,
    a_basic.binaries,
    a_basic.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FaceTrackerGUI",   # output: dist/FaceTrackerGUI/
)
