# -*- mode: python ; coding: utf-8 -*-
#
# face_tracker_gui.spec
# PyInstaller build spec for Face Tracker Control Center.
#
# Build with:
#   pyinstaller face_tracker_gui.spec
#
# Output:  dist/FaceTrackerGUI/FaceTrackerGUI.exe  (+ supporting files)
# To distribute: zip the entire dist/FaceTrackerGUI/ folder.

import sys
from pathlib import Path
import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(SPECPATH)

# OpenCV ships its Haar cascade XMLs inside the package data directory.
# We must include them explicitly so face detection works in the frozen app.
CV2_DATA_DIR = Path(cv2.data.haarcascades).parent

# ---------------------------------------------------------------------------
# Data files to bundle
# (source_path, destination_folder_inside_the_app)
# ---------------------------------------------------------------------------
added_datas = [
    # OpenCV Haar cascade XML files
    (str(CV2_DATA_DIR), "cv2/data"),

    # Custom face scripts (train_lbph.py, face_tracker_lbph.py)
    # Note: face_model.xml is gitignored/user-generated and NOT bundled.
    #       The app falls back gracefully when the model is missing.
    (str(HERE / "custom face"), "custom face"),

    # Arduino sketch (used by the "open file" shortcut button)
    (str(HERE / "facearduino"), "facearduino"),

    # face.py basic tracker (launched as a subprocess from the GUI)
    (str(HERE / "face.py"), "."),
    # Note: dataset/ is gitignored (user-captured images) and is NOT bundled.
    #       The workflow creates an empty placeholder folder before building.
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(HERE / "face_tracker_gui.py")],
    pathex=[str(HERE)],
    binaries=[],
    datas=added_datas,
    hiddenimports=[
        # PySerial sub-modules that PyInstaller may not auto-detect
        "serial",
        "serial.tools",
        "serial.tools.list_ports",
        "serial.tools.list_ports_common",
        "serial.tools.list_ports_windows",
        # OpenCV headless fallbacks
        "cv2",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy scientific libs we don't use
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "PIL",
        "Pillow",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir mode: dlls stay alongside the exe
    name="FaceTrackerGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                # compress with UPX if available
    console=False,           # no black console window (Tkinter GUI)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FaceTrackerGUI",   # output folder: dist/FaceTrackerGUI/
)
