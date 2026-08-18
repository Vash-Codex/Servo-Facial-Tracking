import cv2
import os
import random
import time
import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


# ---------- UI layout ----------
WINDOW_NAME = "biobot"
CANVAS_W, CANVAS_H = 1280, 720
TAB_H = 44
HUD_H = 48
EYE_PANEL_W = int(CANVAS_W * 0.62)
CAM_PANEL_W = CANVAS_W - EYE_PANEL_W
MAIN_H = CANVAS_H - TAB_H - HUD_H

EYE_STYLES = ("realistic", "robotic", "cartoon", "minimal")
STYLE_LABELS = ("Realistic", "Robotic", "Cartoon", "Minimal")

STYLE_THEMES = {
    "realistic": {"bg": (30, 28, 38), "tab_active": (90, 140, 200), "tab_idle": (50, 48, 62), "hud": (180, 190, 210)},
    "robotic":   {"bg": (12, 12, 18), "tab_active": (0, 220, 255),   "tab_idle": (30, 30, 40),  "hud": (0, 200, 220)},
    "cartoon":   {"bg": (235, 225, 250), "tab_active": (255, 120, 180), "tab_idle": (210, 200, 230), "hud": (80, 60, 100)},
    "minimal":   {"bg": (42, 42, 42), "tab_active": (200, 200, 200), "tab_idle": (60, 60, 60),  "hud": (170, 170, 170)},
}


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def available_serial_ports():
    if list_ports is None:
        return []
    try:
        return list(list_ports.comports())
    except Exception:
        return []


def detect_arduino_port(default="COM3"):
    ports = available_serial_ports()
    if not ports:
        return default

    preferred_terms = (
        "arduino", "ch340", "wch", "usb serial", "usb-serial",
        "cp210", "silicon labs", "ftdi", "usb modem",
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


def lerp(current, target, alpha):
    return float(alpha * target + (1.0 - alpha) * current)


def draw_tabs(canvas, style_idx):
    tab_w = CANVAS_W // len(EYE_STYLES)
    for i, label in enumerate(STYLE_LABELS):
        x1 = i * tab_w
        x2 = x1 + tab_w
        theme = STYLE_THEMES[EYE_STYLES[i]]
        color = theme["tab_active"] if i == style_idx else theme["tab_idle"]
        cv2.rectangle(canvas, (x1, 0), (x2, TAB_H), color, -1)
        cv2.line(canvas, (x2 - 1, 0), (x2 - 1, TAB_H), (80, 80, 80), 1)
        text_color = (20, 20, 20) if EYE_STYLES[i] == "cartoon" and i != style_idx else (240, 240, 240)
        if EYE_STYLES[i] == "cartoon" and i == style_idx:
            text_color = (255, 255, 255)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        tx = x1 + (tab_w - text_size[0]) // 2
        ty = (TAB_H + text_size[1]) // 2
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(i + 1), (x1 + 8, TAB_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)


def tab_index_at(x, y):
    if y < 0 or y >= TAB_H:
        return None
    tab_w = CANVAS_W // len(EYE_STYLES)
    idx = x // tab_w
    if 0 <= idx < len(EYE_STYLES):
        return idx
    return None


def draw_style_background(panel, style_name):
    bg = STYLE_THEMES[style_name]["bg"]
    panel[:] = bg

    if style_name == "robotic":
        step = 32
        h, w = panel.shape[:2]
        grid_color = (22, 28, 32)
        for x in range(0, w, step):
            cv2.line(panel, (x, 0), (x, h), grid_color, 1)
        for y in range(0, h, step):
            cv2.line(panel, (0, y), (w, y), grid_color, 1)
    elif style_name == "minimal":
        cv2.line(panel, (0, panel.shape[0] // 2), (panel.shape[1], panel.shape[0] // 2), (55, 55, 55), 1)
        cv2.line(panel, (panel.shape[1] // 2, 0), (panel.shape[1] // 2, panel.shape[0]), (55, 55, 55), 1)


_REALISTIC_IRIS_CACHE = {}


def _make_realistic_iris_texture(radius):
    if radius in _REALISTIC_IRIS_CACHE:
        return _REALISTIC_IRIS_CACHE[radius]

    size = radius * 2 + 4
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = radius + 1
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)

    iris = np.zeros((size, size, 3), dtype=np.float32)
    mask = dist <= radius

    # Hazel-green base with radial color variation
    base_h = 0.52 + 0.08 * np.sin(angle * 3.0)
    base_s = 0.45 + 0.12 * np.cos(angle * 5.0 + dist * 0.08)
    r_ch = 55 + 35 * base_h + 20 * np.sin(dist * 0.22)
    g_ch = 95 + 40 * base_s + 18 * np.cos(angle * 18)
    b_ch = 115 + 30 * (1.0 - base_h) + 12 * np.sin(angle * 11)

    iris[:, :, 0] = np.clip(r_ch, 0, 255)
    iris[:, :, 1] = np.clip(g_ch, 0, 255)
    iris[:, :, 2] = np.clip(b_ch, 0, 255)

    # Radial collagen fibers
    fibers = 0.78 + 0.22 * np.sin(angle * 28.0 + dist * 0.55)
    fibers *= 0.92 + 0.08 * np.sin(angle * 7.0 - dist * 0.9)
    for c in range(3):
        iris[:, :, c] *= fibers

    # Dark limbal ring and lighter inner pool
    limbal = np.clip((dist - radius * 0.78) / (radius * 0.22), 0, 1)
    inner = np.clip((radius * 0.35 - dist) / (radius * 0.35), 0, 1) * 0.08
    for c in range(3):
        iris[:, :, c] *= (1.0 - limbal * 0.55)
        iris[:, :, c] += inner * 35

    # Outer edge falloff
    edge = np.clip((radius - dist) / 3.0, 0, 1)
    iris *= edge[:, :, None]

    iris = np.clip(iris, 0, 255).astype(np.uint8)
    iris[~mask] = 0
    _REALISTIC_IRIS_CACHE[radius] = iris
    return iris


def _paste_circle(dst, src, center_x, center_y):
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]
    x1 = center_x - sw // 2
    y1 = center_y - sh // 2
    x2 = x1 + sw
    y2 = y1 + sh

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = sw - max(0, x2 - dw)
    sy2 = sh - max(0, y2 - dh)
    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    if dx2 <= dx1 or dy2 <= dy1:
        return

    patch = src[sy1:sy2, sx1:sx2]
    alpha = (patch[:, :, 0] > 0) | (patch[:, :, 1] > 0) | (patch[:, :, 2] > 0)
    roi = dst[dy1:dy2, dx1:dx2]
    roi[alpha] = patch[alpha]


def _draw_realistic_sclera(panel, cx, cy, rx, ry):
    sclera = np.zeros_like(panel)
    cv2.ellipse(sclera, (cx, cy), (rx, ry), 0, 0, 360, (248, 244, 238), -1, cv2.LINE_AA)

    # Soft top shadow from upper lid
    shadow = np.ones((ry * 2 + 2, rx * 2 + 2), dtype=np.float32)
    for i, alpha in enumerate(np.linspace(0.22, 0.0, max(3, ry // 3))):
        cv2.ellipse(
            shadow,
            (rx, ry),
            (rx - 2, max(2, ry - i * 2)),
            0, 180, 360,
            float(1.0 - alpha),
            -1,
            cv2.LINE_AA,
        )
    sx1, sy1 = cx - rx, cy - ry
    sx2, sy2 = sx1 + shadow.shape[1], sy1 + shadow.shape[0]
    h, w = panel.shape[:2]
    dx1, dy1 = max(0, sx1), max(0, sy1)
    dx2, dy2 = min(w, sx2), min(h, sy2)
    if dx2 > dx1 and dy2 > dy1:
        sh_sy1 = max(0, -sy1)
        sh_sy2 = shadow.shape[0] - max(0, sy2 - h)
        sh_sx1 = max(0, -sx1)
        sh_sx2 = shadow.shape[1] - max(0, sx2 - w)
        sh = shadow[sh_sy1:sh_sy2, sh_sx1:sh_sx2]
        roi = sclera[dy1:dy2, dx1:dx2].astype(np.float32)
        sclera[dy1:dy2, dx1:dx2] = np.clip(roi * sh[:, :, None], 0, 255).astype(np.uint8)

    # Pink inner/outer corners
    cv2.ellipse(sclera, (cx - int(rx * 0.82), cy + int(ry * 0.08)), (int(rx * 0.12), int(ry * 0.18)), 0, 0, 360, (200, 210, 245), -1, cv2.LINE_AA)
    cv2.ellipse(sclera, (cx + int(rx * 0.82), cy + int(ry * 0.08)), (int(rx * 0.12), int(ry * 0.18)), 0, 0, 360, (200, 210, 245), -1, cv2.LINE_AA)

    mask = np.zeros(panel.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1, cv2.LINE_AA)
    panel[mask > 0] = sclera[mask > 0]


def _draw_realistic_skin(panel, cx, cy, rx, ry):
    skin_rx, skin_ry = int(rx * 1.18), int(ry * 1.55)
    cv2.ellipse(panel, (cx, cy + int(ry * 0.08)), (skin_rx, skin_ry), 0, 0, 360, (175, 155, 135), -1, cv2.LINE_AA)
    cv2.ellipse(panel, (cx, cy + int(ry * 0.08)), (skin_rx, skin_ry), 0, 0, 360, (130, 110, 95), 1, cv2.LINE_AA)
    # Subtle crease above eye
    cv2.ellipse(panel, (cx, cy - int(ry * 0.95)), (int(rx * 0.95), int(ry * 0.35)), 0, 190, 350, (120, 100, 88), 1, cv2.LINE_AA)


def _draw_realistic_lids(panel, cx, cy, rx, ry, blink_amount):
    lid_skin = (168, 148, 128)
    lid_shadow = (95, 78, 68)
    lid_line = (85, 68, 58)

    # Upper lid fold
    cv2.ellipse(panel, (cx, cy - int(ry * 0.55)), (int(rx * 1.02), int(ry * 0.62)), 0, 190, 350, lid_shadow, 2, cv2.LINE_AA)
    cv2.ellipse(panel, (cx, cy - int(ry * 0.35)), (int(rx * 1.05), int(ry * 0.78)), 0, 200, 340, lid_line, 1, cv2.LINE_AA)

    # Lower lid
    cv2.ellipse(panel, (cx, cy + int(ry * 0.72)), (int(rx * 0.92), int(ry * 0.28)), 0, 10, 170, lid_line, 1, cv2.LINE_AA)
    cv2.ellipse(panel, (cx, cy + int(ry * 0.55)), (int(rx * 0.75), int(ry * 0.16)), 0, 0, 180, (210, 195, 180), -1, cv2.LINE_AA)

    if blink_amount <= 0.01:
        return

    cover = int(ry * 2 * blink_amount)
    bg = STYLE_THEMES["realistic"]["bg"]

    # Curved upper lid closing
    top_mask = np.zeros(panel.shape[:2], dtype=np.uint8)
    cv2.ellipse(top_mask, (cx, cy - ry + cover // 2), (rx + 8, max(4, cover // 2 + 6)), 0, 0, 180, 255, -1, cv2.LINE_AA)
    cv2.rectangle(top_mask, (cx - rx - 12, 0), (cx + rx + 12, cy - ry + cover), 255, -1)
    panel[top_mask > 0] = bg

    lid_cover_y = cy - ry + cover
    cv2.ellipse(panel, (cx, lid_cover_y), (rx + 4, max(3, cover // 4)), 0, 0, 180, lid_skin, -1, cv2.LINE_AA)
    cv2.ellipse(panel, (cx, lid_cover_y + 1), (int(rx * 0.98), max(2, cover // 5)), 0, 0, 180, lid_line, 1, cv2.LINE_AA)

    # Lower lid rise for full blink
    if blink_amount > 0.55:
        bottom_mask = np.zeros(panel.shape[:2], dtype=np.uint8)
        rise = int(cover * 0.65)
        cv2.ellipse(bottom_mask, (cx, cy + ry - rise // 2), (rx + 8, max(4, rise // 2 + 6)), 0, 180, 360, 255, -1, cv2.LINE_AA)
        cv2.rectangle(bottom_mask, (cx - rx - 12, cy + ry - rise), (cx + rx + 12, panel.shape[0]), 255, -1)
        panel[bottom_mask > 0] = bg
        cv2.ellipse(panel, (cx, cy + ry - rise), (rx + 4, max(3, rise // 4)), 0, 180, 360, lid_skin, -1, cv2.LINE_AA)


def draw_eye_realistic(panel, pupil_x, pupil_y, blink_amount):
    h, w = panel.shape[:2]
    cx, cy = w // 2, h // 2
    rx, ry = int(min(w, h) * 0.34), int(min(w, h) * 0.21)

    _draw_realistic_skin(panel, cx, cy, rx, ry)
    _draw_realistic_sclera(panel, cx, cy, rx, ry)

    iris_r = int(min(rx, ry) * 0.78)
    px = int(cx + pupil_x * iris_r * 0.34)
    py = int(cy + pupil_y * iris_r * 0.28)

    iris_tex = _make_realistic_iris_texture(iris_r)
    _paste_circle(panel, iris_tex, px, py)

    # Limbal ring outline
    cv2.circle(panel, (px, py), iris_r, (25, 30, 35), 2, cv2.LINE_AA)
    cv2.circle(panel, (px, py), iris_r - 1, (40, 55, 65), 1, cv2.LINE_AA)

    pupil_r = max(7, int(iris_r * 0.34))
    cv2.circle(panel, (px, py), pupil_r, (8, 8, 10), -1, cv2.LINE_AA)
    cv2.circle(panel, (px, py), max(4, pupil_r - 2), (18, 18, 22), 1, cv2.LINE_AA)

    # Primary and secondary catchlights
    cv2.circle(panel, (px - pupil_r // 2, py - pupil_r // 2), max(4, pupil_r // 3), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(panel, (px + pupil_r // 2, py - pupil_r // 3), max(2, pupil_r // 5), (220, 235, 255), -1, cv2.LINE_AA)
    cv2.ellipse(panel, (px - iris_r // 3, py - iris_r // 2), (max(3, iris_r // 6), max(2, iris_r // 10)), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)

    # Upper lid shadow across sclera
    cv2.ellipse(panel, (cx, cy - int(ry * 0.15)), (int(rx * 0.98), int(ry * 0.92)), 0, 200, 340, (210, 205, 198), 3, cv2.LINE_AA)

    _draw_realistic_lids(panel, cx, cy, rx, ry, blink_amount)


def draw_eye_robotic(panel, pupil_x, pupil_y, blink_amount):
    h, w = panel.shape[:2]
    cx, cy = w // 2, h // 2
    outer = int(min(w, h) * 0.36)

    for i, (scale, color, thickness) in enumerate(
        ((1.0, (0, 180, 220), 2), (0.78, (0, 120, 160), 1), (0.56, (0, 90, 120), 1))
    ):
        r = int(outer * scale)
        cv2.circle(panel, (cx, cy), r, color, thickness, cv2.LINE_AA)

    px = int(cx + pupil_x * outer * 0.28)
    py = int(cy + pupil_y * outer * 0.24)
    cv2.circle(panel, (px, py), max(8, outer // 6), (0, 255, 255), -1, cv2.LINE_AA)
    cv2.line(panel, (px - 12, py), (px + 12, py), (0, 80, 100), 1)
    cv2.line(panel, (px, py - 12), (px, py + 12), (0, 80, 100), 1)

    cv2.ellipse(panel, (cx, cy), (outer + 8, outer // 3), 0, 200, 340, (0, 60, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "TRACK", (cx - outer, cy - outer - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 220), 1, cv2.LINE_AA)

    if blink_amount > 0.01:
        shutter = int(outer * 2 * blink_amount)
        cv2.rectangle(panel, (cx - outer - 10, cy - outer), (cx + outer + 10, cy - outer + shutter), (12, 12, 18), -1)
        cv2.rectangle(panel, (cx - outer - 10, cy + outer - shutter), (cx + outer + 10, cy + outer), (12, 12, 18), -1)


def draw_eye_cartoon(panel, pupil_x, pupil_y, blink_amount):
    h, w = panel.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * 0.32)

    cv2.circle(panel, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy), r, (30, 30, 30), 4, cv2.LINE_AA)

    px = int(cx + pupil_x * r * 0.35)
    py = int(cy + pupil_y * r * 0.30)
    pr = max(10, r // 3)
    cv2.circle(panel, (px, py), pr, (20, 20, 25), -1, cv2.LINE_AA)
    cv2.circle(panel, (px - pr // 3, py - pr // 3), max(4, pr // 5), (255, 255, 255), -1, cv2.LINE_AA)

    if blink_amount > 0.01:
        lid_h = int(r * 2 * blink_amount)
        bg = STYLE_THEMES["cartoon"]["bg"]
        cv2.rectangle(panel, (cx - r - 8, cy - r - 6), (cx + r + 8, cy - r + lid_h), bg, -1)
        cv2.rectangle(panel, (cx - r - 8, cy + r - lid_h), (cx + r + 8, cy + r + 6), bg, -1)
        cv2.line(panel, (cx - r, cy - r + lid_h), (cx + r, cy - r + lid_h), (30, 30, 30), 4, cv2.LINE_AA)


def draw_eye_minimal(panel, pupil_x, pupil_y, blink_amount):
    h, w = panel.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * 0.28)

    cv2.circle(panel, (cx, cy), r, (120, 120, 120), 1, cv2.LINE_AA)
    px = int(cx + pupil_x * r * 0.42)
    py = int(cy + pupil_y * r * 0.38)
    cv2.circle(panel, (px, py), max(5, r // 8), (220, 220, 220), -1, cv2.LINE_AA)

    if blink_amount > 0.01:
        lid_h = int(r * 2 * blink_amount)
        bg = STYLE_THEMES["minimal"]["bg"]
        cv2.rectangle(panel, (cx - r - 4, cy - r), (cx + r + 4, cy - r + lid_h), bg, -1)
        cv2.rectangle(panel, (cx - r - 4, cy + r - lid_h), (cx + r + 4, cy + r), bg, -1)
        cv2.line(panel, (cx - r, cy - r + lid_h), (cx + r, cy - r + lid_h), (120, 120, 120), 1)


EYE_DRAWERS = {
    "realistic": draw_eye_realistic,
    "robotic": draw_eye_robotic,
    "cartoon": draw_eye_cartoon,
    "minimal": draw_eye_minimal,
}


def render_eye_panel(style_name, pupil_x, pupil_y, tilt_deg, blink_amount):
    size = max(EYE_PANEL_W, MAIN_H)
    work = np.zeros((size, size, 3), dtype=np.uint8)
    draw_style_background(work, style_name)
    EYE_DRAWERS[style_name](work, pupil_x, pupil_y, blink_amount)

    if abs(tilt_deg) > 0.1:
        matrix = cv2.getRotationMatrix2D((size // 2, size // 2), tilt_deg, 1.0)
        work = cv2.warpAffine(work, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=STYLE_THEMES[style_name]["bg"])

    panel = np.zeros((MAIN_H, EYE_PANEL_W, 3), dtype=np.uint8)
    panel[:] = STYLE_THEMES[style_name]["bg"]
    y_off = (MAIN_H - size) // 2
    x_off = (EYE_PANEL_W - size) // 2
    y1, y2 = max(0, y_off), min(MAIN_H, y_off + size)
    x1, x2 = max(0, x_off), min(EYE_PANEL_W, x_off + size)
    sy1, sy2 = max(0, -y_off), max(0, -y_off) + (y2 - y1)
    sx1, sx2 = max(0, -x_off), max(0, -x_off) + (x2 - x1)
    panel[y1:y2, x1:x2] = work[sy1:sy2, sx1:sx2]
    return panel


def draw_hud(canvas, y0, style_name, servo_angle, face_detected, servo_enabled, paused, invert):
    hud = canvas[y0:y0 + HUD_H]
    hud[:] = (25, 25, 30)
    color = STYLE_THEMES[style_name]["hud"]
    face_text = "FACE LOCK" if face_detected else "SCANNING"
    servo_text = "SERVO ON" if servo_enabled else "SERVO OFF"
    pause_text = "PAUSED" if paused else "LIVE"

    cv2.putText(hud, "biobot", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(
        hud,
        f"{STYLE_LABELS[EYE_STYLES.index(style_name)]}  |  {face_text}  |  angle {int(servo_angle)}  |  {servo_text}  |  {pause_text}  |  inv:{int(invert)}",
        (160, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        hud,
        "1-4 style  q quit  c center  p pause  s servo  i invert",
        (16, HUD_H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (130, 130, 140),
        1,
        cv2.LINE_AA,
    )


def fit_camera_panel(frame, panel_w, panel_h):
    h, w = frame.shape[:2]
    scale = min(panel_w / w, panel_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (18, 18, 22)
    x_off = (panel_w - new_w) // 2
    y_off = (panel_h - new_h) // 2
    panel[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    cv2.rectangle(panel, (x_off, y_off), (x_off + new_w, y_off + new_h), (70, 70, 80), 1)
    cv2.putText(panel, "CAM", (x_off + 8, y_off + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 190), 1, cv2.LINE_AA)
    return panel


class BlinkController:
    def __init__(self):
        self.amount = 0.0
        self.phase = "idle"
        self.phase_start = time.time()
        self.next_blink = time.time() + random.uniform(3.0, 5.0)

    def update(self):
        now = time.time()
        if self.phase == "idle":
            self.amount = 0.0
            if now >= self.next_blink:
                self.phase = "closing"
                self.phase_start = now
        elif self.phase == "closing":
            t = (now - self.phase_start) / 0.08
            self.amount = min(1.0, t)
            if t >= 1.0:
                self.phase = "closed"
                self.phase_start = now
        elif self.phase == "closed":
            self.amount = 1.0
            if now - self.phase_start >= 0.05:
                self.phase = "opening"
                self.phase_start = now
        elif self.phase == "opening":
            t = (now - self.phase_start) / 0.10
            self.amount = max(0.0, 1.0 - t)
            if t >= 1.0:
                self.phase = "idle"
                self.next_blink = now + random.uniform(3.0, 5.0)


def main():
    # ---------- Settings ----------
    USE_ARDUINO    = env_flag("FACE_TRACKER_USE_ARDUINO", True)
    ENV_COM_PORT   = os.getenv("FACE_TRACKER_COM_PORT")
    COM_PORT       = (ENV_COM_PORT.strip() if ENV_COM_PORT else detect_arduino_port("COM3")) or "COM3"
    BAUD           = 9600
    SERVO_MIN      = 45
    SERVO_MAX      = 135
    CENTER         = 90
    DEADZONE_PCT   = 0.04
    SMOOTH_A       = 0.35
    PUPIL_SMOOTH   = 0.18
    TILT_SMOOTH    = 0.12
    MAX_TILT_DEG   = 8.0
    SEND_MIN_MS    = 15
    ANGLE_STEP_MIN = 1
    MIRROR_PREVIEW = True
    # ------------------------------

    arduino = None
    if USE_ARDUINO:
        if serial is None:
            print("[WARN] PySerial is not installed. Running in camera-only mode.")
        else:
            try:
                arduino = serial.Serial(COM_PORT, BAUD, timeout=0)
                time.sleep(2)
                print(f"[OK] Connected to Arduino on {COM_PORT}.")
            except Exception as exc:
                print(f"[WARN] Arduino not found on {COM_PORT}: {exc}")
                if "access is denied" in str(exc).lower():
                    print("[WARN] Close Arduino IDE Serial Monitor/Plotter or any other app using the port, then restart.")
                ports = available_serial_ports()
                if ports:
                    visible = ", ".join(f"{p.device} ({p.description})" for p in ports)
                    print(f"[WARN] Visible serial ports: {visible}")
                else:
                    print("[WARN] No serial ports detected by PySerial.")
                print("[WARN] Running in camera-only mode.")
    else:
        print("[INFO] Arduino access disabled. Running in camera-only mode.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    invert         = False
    paused         = False
    servo_enabled  = arduino is not None
    servo_angle    = float(CENTER)
    last_sent_angle = None
    last_send_time  = 0
    style_idx      = 0
    pupil_x        = 0.0
    pupil_y        = 0.0
    tilt_deg       = 0.0
    face_detected  = False
    blink          = BlinkController()

    ui_state = {"style_idx": style_idx}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = tab_index_at(x, y)
            if idx is not None:
                ui_state["style_idx"] = idx

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    def send_angle(angle):
        nonlocal last_sent_angle, last_send_time
        if not servo_enabled or arduino is None:
            return
        now = time.time() * 1000
        if last_sent_angle is None or abs(angle - last_sent_angle) >= ANGLE_STEP_MIN:
            if now - last_send_time >= SEND_MIN_MS:
                clipped = int(np.clip(angle, SERVO_MIN, SERVO_MAX))
                try:
                    arduino.write(f"{clipped}\n".encode())
                except Exception:
                    pass
                last_sent_angle = clipped
                last_send_time  = now

    send_angle(servo_angle)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)

            h, w           = frame.shape[:2]
            frame_center_x = w // 2
            frame_center_y = h // 2
            deadzone_px    = int(w * DEADZONE_PCT)

            target_px, target_py, target_tilt = 0.0, 0.0, 0.0
            face_detected = False

            if len(faces) > 0:
                face_detected = True
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                (x, y, fw, fh) = faces[0]
                cx = x + fw // 2
                cy = y + fh // 2

                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                cv2.line(frame, (frame_center_x - deadzone_px, 0), (frame_center_x - deadzone_px, h), (255, 255, 0), 1)
                cv2.line(frame, (frame_center_x + deadzone_px, 0), (frame_center_x + deadzone_px, h), (255, 255, 0), 1)
                cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1)

                norm_x = np.clip((cx - frame_center_x) / max(1, w * 0.45), -1.0, 1.0)
                norm_y = np.clip((cy - frame_center_y) / max(1, h * 0.45), -1.0, 1.0)
                target_px = float(norm_x)
                target_py = float(norm_y)
                target_tilt = float(norm_x * MAX_TILT_DEG)

                if invert:
                    target = np.interp(cx, [0, w], [SERVO_MAX, SERVO_MIN])
                else:
                    target = np.interp(cx, [0, w], [SERVO_MIN, SERVO_MAX])

                if abs(cx - frame_center_x) <= deadzone_px:
                    target = servo_angle

                servo_angle = lerp(servo_angle, target, SMOOTH_A)
                if not paused:
                    send_angle(servo_angle)

            pupil_x = lerp(pupil_x, target_px, PUPIL_SMOOTH)
            pupil_y = lerp(pupil_y, target_py, PUPIL_SMOOTH)
            tilt_deg = lerp(tilt_deg, target_tilt, TILT_SMOOTH)
            blink.update()

            style_idx = ui_state["style_idx"]
            style_name = EYE_STYLES[style_idx]

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            draw_tabs(canvas, style_idx)

            eye_panel = render_eye_panel(style_name, pupil_x, pupil_y, tilt_deg, blink.amount)
            cam_panel = fit_camera_panel(frame, CAM_PANEL_W, MAIN_H)

            main_y = TAB_H
            canvas[main_y:main_y + MAIN_H, 0:EYE_PANEL_W] = eye_panel
            canvas[main_y:main_y + MAIN_H, EYE_PANEL_W:CANVAS_W] = cam_panel
            cv2.line(canvas, (EYE_PANEL_W, main_y), (EYE_PANEL_W, main_y + MAIN_H), (60, 60, 70), 2)

            draw_hud(canvas, TAB_H + MAIN_H, style_name, servo_angle, face_detected, servo_enabled, paused, invert)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                servo_angle = float(CENTER)
                pupil_x = pupil_y = target_px = target_py = 0.0
                tilt_deg = 0.0
                if not paused:
                    send_angle(servo_angle)
            elif key == ord("r"):
                servo_angle = float(SERVO_MIN)
                if not paused:
                    send_angle(servo_angle)
            elif key == ord("i"):
                invert = not invert
            elif key == ord("p"):
                paused = not paused
            elif key == ord("s"):
                servo_enabled = not servo_enabled
                print(f"[INFO] Servo {'enabled' if servo_enabled else 'disabled'}.")
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                ui_state["style_idx"] = key - ord("1")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arduino:
            arduino.close()


if __name__ == "__main__":
    main()
