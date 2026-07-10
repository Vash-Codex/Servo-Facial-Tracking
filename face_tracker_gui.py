import importlib.util
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None


# When frozen by PyInstaller the real "app root" is next to the .exe,
# not inside the _MEIPASS extraction temp dir.
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent
CUSTOM_DIR = APP_DIR / "custom face"
DATASET_DIR = APP_DIR / "dataset"
TRAIN_SCRIPT = CUSTOM_DIR / "train_lbph.py"
LBPH_TRACKER_SCRIPT = CUSTOM_DIR / "face_tracker_lbph.py"
BASIC_TRACKER_SCRIPT = APP_DIR / "face.py"
MODEL_FILE = CUSTOM_DIR / "face_model.xml"
ARDUINO_SKETCH = APP_DIR / "facearduino" / "facearduino.ino"
DEMO_VIDEO = APP_DIR / "vids" / "Facial Tracking .mp4"


def available_serial_ports():
    if list_ports is None:
        return []
    try:
        return list(list_ports.comports())
    except Exception:
        return []


def detect_arduino_port(default="COM10"):
    ports = available_serial_ports()
    if not ports:
        return default

    preferred_terms = (
        "arduino",
        "ch340",
        "wch",
        "usb serial",
        "usb-serial",
        "cp210",
        "silicon labs",
        "ftdi",
        "usb modem",
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

    return ports[0].device or default


def format_serial_ports():
    ports = available_serial_ports()
    if not ports:
        return "none detected"
    return ", ".join(f"{port.device} ({port.description})" for port in ports)


THEMES = {
    "dark": {
        "bg": "#0f1314",
        "panel": "#151a1c",
        "panel_alt": "#1d2427",
        "field": "#111719",
        "border": "#2e3a3f",
        "text": "#f2f7f5",
        "muted": "#a2afaa",
        "green": "#00d48a",
        "green_hover": "#18e49d",
        "pink": "#ff4da6",
        "pink_hover": "#ff6ab5",
        "danger": "#ff5f6d",
        "danger_hover": "#ff7581",
        "shadow": "#0a0d0e",
    },
    "light": {
        "bg": "#f5f8f7",
        "panel": "#ffffff",
        "panel_alt": "#edf4f1",
        "field": "#f8fbfa",
        "border": "#d7e3df",
        "text": "#13201b",
        "muted": "#65736e",
        "green": "#009b68",
        "green_hover": "#00ad76",
        "pink": "#d93687",
        "pink_hover": "#ec4a99",
        "danger": "#d94a5f",
        "danger_hover": "#e85c70",
        "shadow": "#dfe8e4",
    },
}


class FaceTrackerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Tracker Control Center")
        self.geometry("1120x720")
        self.minsize(940, 620)

        self.theme_name = "dark"
        self.processes = {}
        self.output_queue = queue.Queue()
        self.log_lines = []
        self.stat_labels = {}
        self.process_labels = {}
        self.log_box = None
        self.arduino_enabled = tk.BooleanVar(value=False)
        self.com_port = tk.StringVar(value=detect_arduino_port("COM10"))

        self.configure(bg=self.colors["bg"])
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.after(150, self.drain_output_queue)
        self.after(1500, self.refresh_status_loop)

    @property
    def colors(self):
        return THEMES[self.theme_name]

    def build_ui(self):
        for child in self.winfo_children():
            child.destroy()

        c = self.colors
        self.configure(bg=c["bg"])
        self.grid_columnconfigure(0, minsize=280)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = tk.Frame(self, bg=c["panel"], padx=24, pady=24)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(8, weight=1)

        main = tk.Frame(self, bg=c["bg"], padx=28, pady=24)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(4, weight=1)

        self.build_sidebar(sidebar)
        self.build_main(main)
        self.refresh_status()
        self.restore_log()

    def build_sidebar(self, parent):
        c = self.colors
        self.label(parent, "Face", 30, "bold", c["green"]).grid(row=0, column=0, sticky="w")
        self.label(parent, "Tracker", 30, "bold", c["pink"]).grid(row=1, column=0, sticky="w")
        self.label(
            parent,
            "Control training, recognition, Arduino files, and project assets from one place.",
            10,
            "normal",
            c["muted"],
            wraplength=220,
            justify="left",
        ).grid(row=2, column=0, sticky="ew", pady=(10, 26))

        mode_text = "Switch to Light" if self.theme_name == "dark" else "Switch to Dark"
        self.button(parent, mode_text, self.toggle_theme, "secondary").grid(row=3, column=0, sticky="ew")

        self.stat_card(parent, "Dataset Images", "dataset", 4)
        self.stat_card(parent, "LBPH Model", "model", 5)
        self.stat_card(parent, "Arduino", "arduino", 6)
        self.stat_card(parent, "Running Task", "running", 7)

        footer = tk.Frame(parent, bg=c["panel"])
        footer.grid(row=9, column=0, sticky="ew", pady=(18, 0))
        footer.grid_columnconfigure(0, weight=1)
        self.button(footer, "Open Project Folder", lambda: self.open_path(APP_DIR), "secondary").grid(
            row=0, column=0, sticky="ew", pady=(0, 10)
        )
        self.button(footer, "Check Setup", self.check_setup, "outline").grid(row=1, column=0, sticky="ew")

    def build_main(self, parent):
        c = self.colors
        header = tk.Frame(parent, bg=c["bg"])
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        self.label(header, "Control Center", 24, "bold", c["text"]).grid(row=0, column=0, sticky="w")
        self.label(
            header,
            "Start the trainer, run tracking, or open the project pieces you use most.",
            10,
            "normal",
            c["muted"],
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.process_labels["status_pill"] = self.pill(header, "Idle")
        self.process_labels["status_pill"].grid(row=0, column=1, rowspan=2, sticky="e")

        action_grid = tk.Frame(parent, bg=c["bg"])
        action_grid.grid(row=1, column=0, sticky="ew", pady=(24, 18))
        for index in range(3):
            action_grid.grid_columnconfigure(index, weight=1, uniform="cards")

        self.action_card(
            action_grid,
            "Train Face Model",
            "Capture new face samples and rebuild the LBPH model.",
            "green",
            "Start Training",
            lambda: self.launch_script("Training", TRAIN_SCRIPT),
            0,
        )
        self.action_card(
            action_grid,
            "Run LBPH Tracker",
            "Use your trained model for recognition and servo tracking.",
            "pink",
            "Start Tracker",
            lambda: self.launch_script("LBPH Tracker", LBPH_TRACKER_SCRIPT),
            1,
        )
        self.action_card(
            action_grid,
            "Run Basic Tracker",
            "Launch the older face-only tracker for quick camera testing.",
            "green",
            "Start Basic",
            lambda: self.launch_script("Basic Tracker", BASIC_TRACKER_SCRIPT),
            2,
        )

        self.build_hardware_panel(parent, 2)

        tools = tk.Frame(parent, bg=c["panel"], padx=18, pady=16, highlightbackground=c["border"], highlightthickness=1)
        tools.grid(row=3, column=0, sticky="ew", pady=(0, 18))
        for index in range(5):
            tools.grid_columnconfigure(index, weight=1, uniform="tools")
        self.label(tools, "Project Shortcuts", 12, "bold", c["text"]).grid(row=0, column=0, columnspan=5, sticky="w")
        self.button(tools, "Dataset", lambda: self.open_path(DATASET_DIR), "secondary").grid(
            row=1, column=0, sticky="ew", padx=(0, 8), pady=(14, 0)
        )
        self.button(tools, "Custom Face", lambda: self.open_path(CUSTOM_DIR), "secondary").grid(
            row=1, column=1, sticky="ew", padx=8, pady=(14, 0)
        )
        self.button(tools, "Arduino", lambda: self.open_path(ARDUINO_SKETCH), "secondary").grid(
            row=1, column=2, sticky="ew", padx=8, pady=(14, 0)
        )
        self.button(tools, "Demo Video", lambda: self.open_path(DEMO_VIDEO), "secondary").grid(
            row=1, column=3, sticky="ew", padx=8, pady=(14, 0)
        )
        self.button(tools, "Stop All", self.stop_all_processes, "danger").grid(
            row=1, column=4, sticky="ew", padx=(8, 0), pady=(14, 0)
        )

        log_panel = tk.Frame(parent, bg=c["panel"], padx=18, pady=16, highlightbackground=c["border"], highlightthickness=1)
        log_panel.grid(row=4, column=0, sticky="nsew")
        log_panel.grid_columnconfigure(0, weight=1)
        log_panel.grid_rowconfigure(1, weight=1)

        log_header = tk.Frame(log_panel, bg=c["panel"])
        log_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        log_header.grid_columnconfigure(0, weight=1)
        self.label(log_header, "Activity Log", 12, "bold", c["text"]).grid(row=0, column=0, sticky="w")
        self.button(log_header, "Clear", self.clear_log, "outline").grid(row=0, column=1, sticky="e")

        self.log_box = tk.Text(
            log_panel,
            bg=c["field"],
            fg=c["text"],
            insertbackground=c["green"],
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            wrap="word",
            font=("Consolas", 10),
            height=10,
        )
        self.log_box.grid(row=1, column=0, sticky="nsew")
        self.log_box.configure(state="disabled")

    def build_hardware_panel(self, parent, row):
        c = self.colors
        panel = tk.Frame(parent, bg=c["panel"], padx=18, pady=16, highlightbackground=c["border"], highlightthickness=1)
        panel.grid(row=row, column=0, sticky="ew", pady=(0, 18))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_columnconfigure(1, minsize=170)
        panel.grid_columnconfigure(2, minsize=210)

        self.label(panel, "Arduino Access", 12, "bold", c["text"]).grid(row=0, column=0, sticky="w")
        self.label(
            panel,
            "These settings are used when you start a tracker from this dashboard.",
            10,
            "normal",
            c["muted"],
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        toggle_text = "Arduino ON" if self.arduino_enabled.get() else "Arduino OFF"
        toggle_kind = "green" if self.arduino_enabled.get() else "outline"
        self.button(panel, toggle_text, self.toggle_arduino_access, toggle_kind).grid(
            row=0, column=1, rowspan=2, sticky="ew", padx=(18, 12)
        )

        port_box = tk.Frame(panel, bg=c["panel"])
        port_box.grid(row=0, column=2, rowspan=2, sticky="ew")
        port_box.grid_columnconfigure(0, weight=1)
        self.label(port_box, "COM Port", 9, "bold", c["muted"]).grid(row=0, column=0, sticky="w")
        entry_row = tk.Frame(port_box, bg=c["panel"])
        entry_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        entry_row.grid_columnconfigure(0, weight=1)
        entry = tk.Entry(
            entry_row,
            textvariable=self.com_port,
            bg=c["field"],
            fg=c["text"],
            insertbackground=c["green"],
            relief="flat",
            bd=0,
            font=("Segoe UI", 11, "bold"),
            width=12,
        )
        entry.grid(row=0, column=0, sticky="ew", ipady=9)
        entry.bind("<Return>", lambda _event: self.apply_com_port())
        self.button(entry_row, "Use", self.apply_com_port, "pink").grid(row=0, column=1, sticky="e", padx=(8, 0))

    def action_card(self, parent, title, body, accent_key, button_text, command, column):
        c = self.colors
        card = tk.Frame(parent, bg=c["panel"], padx=18, pady=16, highlightbackground=c["border"], highlightthickness=1)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0 if column == 2 else 8))
        card.grid_columnconfigure(0, weight=1)
        tk.Frame(card, bg=c[accent_key], height=4).grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.label(card, title, 14, "bold", c["text"]).grid(row=1, column=0, sticky="w")
        self.label(card, body, 10, "normal", c["muted"], wraplength=220, justify="left").grid(
            row=2, column=0, sticky="ew", pady=(8, 18)
        )
        self.button(card, button_text, command, accent_key).grid(row=3, column=0, sticky="ew")

    def stat_card(self, parent, title, key, row):
        c = self.colors
        card = tk.Frame(parent, bg=c["panel_alt"], padx=14, pady=12, highlightbackground=c["border"], highlightthickness=1)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)
        self.label(card, title, 9, "normal", c["muted"]).grid(row=0, column=0, sticky="w")
        value = self.label(card, "...", 16, "bold", c["text"])
        value.grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.stat_labels[key] = value

    def label(self, parent, text, size, weight, fg, **kwargs):
        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=fg,
            font=("Segoe UI", size, weight),
            **kwargs,
        )

    def button(self, parent, text, command, kind):
        c = self.colors
        styles = {
            "green": (c["green"], "#041411", c["green_hover"]),
            "pink": (c["pink"], "#1c0712", c["pink_hover"]),
            "secondary": (c["panel_alt"], c["text"], c["border"]),
            "outline": (c["panel"], c["muted"], c["panel_alt"]),
            "danger": (c["danger"], "#18070a", c["danger_hover"]),
        }
        bg, fg, hover = styles[kind]
        widget = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=10,
        )
        widget.bind("<Enter>", lambda _event: widget.configure(bg=hover))
        widget.bind("<Leave>", lambda _event: widget.configure(bg=bg))
        return widget

    def pill(self, parent, text):
        c = self.colors
        return tk.Label(
            parent,
            text=text,
            bg=c["panel_alt"],
            fg=c["green"],
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=8,
        )

    def toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.build_ui()
        self.append_log("GUI", f"Theme changed to {self.theme_name}.")

    def toggle_arduino_access(self):
        self.arduino_enabled.set(not self.arduino_enabled.get())
        state = "enabled" if self.arduino_enabled.get() else "disabled"
        self.build_ui()
        self.append_log("Arduino", f"Access {state}. Restart any running tracker to apply this.")

    def current_com_port(self):
        return self.com_port.get().strip().upper() or detect_arduino_port("COM10")

    def normalized_com_port(self):
        port = self.current_com_port()
        self.com_port.set(port)
        return port

    def apply_com_port(self):
        port = self.normalized_com_port()
        active = [name for name, proc in self.processes.items() if proc.poll() is None]
        note = " Restart running trackers to apply it." if active else ""
        self.append_log("Arduino", f"COM port set to {port}.{note}")
        self.refresh_status()

    def launch_script(self, name, script_path):
        if not script_path.exists():
            self.append_log("GUI", f"Missing script: {script_path}")
            messagebox.showerror("Missing file", f"Could not find:\n{script_path}")
            return

        existing = self.processes.get(name)
        if existing and existing.poll() is None:
            self.append_log("GUI", f"{name} is already running.")
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["FACE_TRACKER_USE_ARDUINO"] = "1" if self.arduino_enabled.get() else "0"
        env["FACE_TRACKER_COM_PORT"] = self.normalized_com_port()
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0

        try:
            process = subprocess.Popen(
                [sys.executable, "-u", str(script_path)],
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except Exception as exc:
            self.append_log("GUI", f"Could not start {name}: {exc}")
            messagebox.showerror("Launch failed", str(exc))
            return

        self.processes[name] = process
        self.append_log("GUI", f"Started {name}. Use the OpenCV window controls inside that task.")
        if script_path in (LBPH_TRACKER_SCRIPT, BASIC_TRACKER_SCRIPT):
            state = "on" if self.arduino_enabled.get() else "off"
            self.append_log("Arduino", f"Access {state}; port {self.normalized_com_port()}.")
        threading.Thread(target=self.read_process_output, args=(name, process), daemon=True).start()
        self.refresh_status()

    def read_process_output(self, name, process):
        if process.stdout:
            for line in process.stdout:
                clean_line = line.rstrip()
                if clean_line:
                    self.output_queue.put(("line", name, clean_line))
        return_code = process.wait()
        self.output_queue.put(("exit", name, return_code))

    def drain_output_queue(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item[0] == "line":
                    _, name, line = item
                    self.append_log(name, line)
                elif item[0] == "exit":
                    _, name, return_code = item
                    self.processes.pop(name, None)
                    self.append_log("GUI", f"{name} finished with exit code {return_code}.")
                    self.refresh_status()
        except queue.Empty:
            pass
        self.after(150, self.drain_output_queue)

    def stop_all_processes(self):
        running = [(name, proc) for name, proc in self.processes.items() if proc.poll() is None]
        if not running:
            self.append_log("GUI", "No running tasks to stop.")
            return
        for name, process in running:
            try:
                process.terminate()
                self.append_log("GUI", f"Stop requested for {name}.")
            except Exception as exc:
                self.append_log("GUI", f"Could not stop {name}: {exc}")
        self.refresh_status()

    def check_setup(self):
        self.append_log("Setup", f"Python: {sys.version.split()[0]}")
        cv2_spec = importlib.util.find_spec("cv2")
        serial_spec = importlib.util.find_spec("serial")

        if cv2_spec is None:
            self.append_log("Setup", "OpenCV is not installed.")
        else:
            try:
                import cv2

                has_face = hasattr(cv2, "face")
                self.append_log("Setup", f"OpenCV: {cv2.__version__}")
                self.append_log("Setup", f"LBPH support: {'available' if has_face else 'missing cv2.face'}")
            except Exception as exc:
                self.append_log("Setup", f"OpenCV import failed: {exc}")

        self.append_log("Setup", f"PySerial: {'available' if serial_spec else 'missing'}")
        self.append_log("Setup", f"Serial ports: {format_serial_ports()}")
        state = "enabled" if self.arduino_enabled.get() else "disabled"
        self.append_log("Setup", f"Arduino setting: {state} on {self.normalized_com_port()}")
        self.refresh_status()

    def open_path(self, path):
        path = Path(path)
        if not path.exists():
            self.append_log("GUI", f"Path not found: {path}")
            messagebox.showwarning("Not found", f"Could not find:\n{path}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            self.append_log("GUI", f"Opened {path.name}.")
        except Exception as exc:
            self.append_log("GUI", f"Could not open {path}: {exc}")
            messagebox.showerror("Open failed", str(exc))

    def refresh_status_loop(self):
        self.refresh_status()
        self.after(1500, self.refresh_status_loop)

    def refresh_status(self):
        dataset_count = len(list(DATASET_DIR.glob("*.jpg"))) if DATASET_DIR.exists() else 0
        if "dataset" in self.stat_labels:
            self.stat_labels["dataset"].configure(text=f"{dataset_count} JPGs")

        if MODEL_FILE.exists():
            updated = time.strftime("%d %b, %I:%M %p", time.localtime(MODEL_FILE.stat().st_mtime))
            model_text = f"Ready\n{updated}"
        else:
            model_text = "Missing"
        if "model" in self.stat_labels:
            self.stat_labels["model"].configure(text=model_text)

        arduino_text = f"On\n{self.current_com_port()}" if self.arduino_enabled.get() else f"Off\n{self.current_com_port()}"
        if "arduino" in self.stat_labels:
            self.stat_labels["arduino"].configure(text=arduino_text)

        active = [name for name, proc in self.processes.items() if proc.poll() is None]
        running_text = ", ".join(active) if active else "Idle"
        if "running" in self.stat_labels:
            self.stat_labels["running"].configure(text=running_text)
        if "status_pill" in self.process_labels:
            self.process_labels["status_pill"].configure(text=running_text)

    def append_log(self, source, message):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {source}: {message}"
        self.log_lines.append(line)
        self.log_lines = self.log_lines[-300:]
        if self.log_box is not None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", line + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

    def restore_log(self):
        if self.log_box is None:
            return
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        if self.log_lines:
            self.log_box.insert("end", "\n".join(self.log_lines) + "\n")
        else:
            self.log_box.insert("end", "Ready. Start training or launch a tracker when you are set.\n")
        self.log_box.configure(state="disabled")

    def clear_log(self):
        self.log_lines.clear()
        self.restore_log()

    def on_close(self):
        running = [proc for proc in self.processes.values() if proc.poll() is None]
        if running:
            should_close = messagebox.askyesno(
                "Close Face Tracker",
                "A tracker or trainer is still running. Stop it and close the dashboard?",
            )
            if not should_close:
                return
            self.stop_all_processes()
        self.destroy()


if __name__ == "__main__":
    FaceTrackerGUI().mainloop()
