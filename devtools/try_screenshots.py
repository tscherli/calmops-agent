"""Developer tool: the README screenshots, from the real window and a made-up conversation.

Ad-hoc, not a pytest file, meant to be run by hand from the project root.

The window is the real `ChatApp`; only the conversation is invented, inserted
straight into the chat. No agent runs, nothing calls an API, and the check-in
log is a fresh file in the temp directory, so no real data can end up in a
picture. The approval card's text comes from the graph's own `action_request`,
so it reads exactly as the app would show it.

Windows only: the window is captured through user32/gdi32 with ctypes, and the
PNG is written with zlib, so nothing needs installing.

    .venv\\Scripts\\python.exe devtools\\try_screenshots.py            # show scene 1 to look at
    .venv\\Scripts\\python.exe devtools\\try_screenshots.py --write    # refresh docs/images/*.png

Each scene's window flashes up for about a second while it is captured.
"""
from __future__ import annotations

import argparse
import ctypes
import struct
import sys
import tempfile
import tkinter as tk
import zlib
from ctypes import wintypes
from pathlib import Path

# The project root, one level up: this file lives in devtools/, and the
# window it shows is imported from the root.
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from agent.graph import action_request
from app import ChatApp

OUT_DIR = PROJECT / "docs" / "images"

CHECK_IN = [
    ("msg", "You", "Back-to-back meetings all morning, and my head is buzzing. I'm wired, I just want to calm down."),
    ("tool", {"tool": "record_choices", "exit_code": 0, "detail": "energy wired, goal calm down now"}),
    ("tool", {"tool": "find_measures", "exit_code": 0, "detail": "3 measures"}),
    ("msg", "Agent", "That sounds like a lot in one morning. Two from your own list that fit a wired, "
                     "calm-down-now moment: box breathing for four rounds (about two minutes), or a "
                     "short walk without your phone. If you'd like the why behind short breaks, here is "
                     "[a short read on micro-breaks](https://example.org/micro-breaks). Which one feels doable?"),
]
PICKED = [
    ("msg", "You", "the breathing one"),
    ("msg", "Agent", "Good choice. Shall I check in the current situation into the circumstances log?"),
    ("msg", "You", "yes"),
]
LOGGED = [
    ("msg", "You", 'approved - Log: "back-to-back meetings all morning" (measure: box_breathing)'),
    ("tool", {"tool": "log_checkin", "exit_code": 0, "detail": "checkin row 3f2a9c1e"}),
    ("msg", "Agent", "Saved. Take your two minutes -- and tell me afterwards whether it helped."),
]
CARD = action_request({
    "name": "log_checkin",
    "id": "demo-1",
    "args": {"measure_id": "box_breathing", "circumstances": "back-to-back meetings all morning"},
})

# scene -> (conversation so far, energy and goal answered?, approval card, measure waiting for a rating)
SCENES = {
    "1-check-in": (CHECK_IN, False, None, ""),
    "2-approval-card": (CHECK_IN + PICKED, True, CARD, ""),
    "3-rating": (CHECK_IN + PICKED + LOGGED, True, None, "box_breathing"),
}


def build(root: tk.Tk, scene: str) -> ChatApp:
    """The window in the state the scene describes, set the way the app itself sets it."""
    conversation, answered, card, rating = SCENES[scene]
    log = Path(tempfile.mkdtemp()) / "checkin_log.csv"
    app = ChatApp(root, skip_consent=True, agent_graph=object(), log_path=log)
    for kind, *rest in conversation:
        if kind == "msg":
            app._append_message(*rest)
        else:
            app._append_tool_event(rest[0])
    if answered:
        app.state.update(energy="wired", goal="calm down now")
    app._set_pending_ratings([rating] if rating else [])
    app._refresh_controls()
    if card:
        app._show_request(card)
    return app


def capture(root: tk.Tk) -> tuple[int, int, bytes]:
    """The window's inside as top-down BGRA pixels, without frame or title bar.

    The title bar is left out on purpose: Windows keeps the focus with the
    terminal that started this script, so the title bar would be drawn greyed
    out in some scenes and not in others.
    """
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    hwnd = user32.GetAncestor(root.winfo_id(), 2)  # GA_ROOT: the frame with the title bar
    rect, client = wintypes.RECT(), wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    user32.GetClientRect(hwnd, ctypes.byref(client))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    full_w, full_h = rect.right - rect.left, rect.bottom - rect.top
    left, top = origin.x - rect.left, origin.y - rect.top
    width, height = client.right, client.bottom

    window_dc = user32.GetWindowDC(hwnd)
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, full_w, full_h)
    gdi32.SelectObject(memory_dc, bitmap)
    user32.PrintWindow(hwnd, memory_dc, 2)  # PW_RENDERFULLCONTENT

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]

    # A negative height asks for the rows top-down, the order PNG wants.
    info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), full_w, -full_h, 1, 32, 0, 0, 0, 0, 0, 0)
    buffer = ctypes.create_string_buffer(full_w * full_h * 4)
    gdi32.GetDIBits(memory_dc, bitmap, 0, full_h, buffer, ctypes.byref(info), 0)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(memory_dc)
    user32.ReleaseDC(hwnd, window_dc)

    raw = buffer.raw
    cropped = b"".join(
        raw[((top + y) * full_w + left) * 4:((top + y) * full_w + left + width) * 4] for y in range(height)
    )
    return width, height, cropped


def save_png(path: Path, width: int, height: int, bgra: bytes) -> None:
    """A plain RGB PNG: each row is a filter byte (0, none) and the pixels."""
    rows = b"".join(
        b"\x00" + bytes(
            value
            for x in range(width)
            for value in (bgra[(y * width + x) * 4 + 2], bgra[(y * width + x) * 4 + 1], bgra[(y * width + x) * 4])
        )
        for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"save every scene to {OUT_DIR}")
    args = parser.parse_args()
    if sys.platform != "win32":
        print("Windows only: the capture uses user32/gdi32.")
        return 1

    # Real pixels on a scaled display; must happen before the first window.
    ctypes.windll.shcore.SetProcessDpiAwareness(2)

    if not args.write:
        root = tk.Tk()
        build(root, "1-check-in")
        root.mainloop()
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # One window for every scene, rebuilt in place; each scene takes the
    # window's natural size, as the app does when it opens.
    root = tk.Tk()
    root.attributes("-topmost", True)
    scenes = list(SCENES)

    def shoot(index: int = 0) -> None:
        for child in root.winfo_children():
            child.destroy()
        build(root, scenes[index])
        root.update()
        root.after(1200, lambda: save(index))

    def save(index: int) -> None:
        root.update()
        path = OUT_DIR / f"calmops-{scenes[index]}.png"
        save_png(path, *capture(root))
        print(f"saved {path.relative_to(PROJECT)}")
        if index + 1 < len(scenes):
            shoot(index + 1)
        else:
            root.destroy()

    root.after(0, shoot)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
