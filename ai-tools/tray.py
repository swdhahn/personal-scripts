#!/usr/bin/env python3
"""System-tray front end: the Windows/macOS stand-in for the waybar module.

Runs as its own process (`ai_assistant.py --tray`), exactly like waybar's
custom/ai module does on Linux: it polls the daemon for AI Mode status, paints
a coloured icon, and turns clicks into daemon commands or popup launches.

  left click (Windows)   toggle AI Mode
  menu                   chat, capture, session manager, sync, daemon start/quit
  hotkeys (Windows)      hotkey_chat / hotkey_capture / hotkey_ai_toggle in .config

The daemon is started automatically when it is not running
(tray_start_daemon=0 turns that off), so one shortcut brings up everything.
"""
import os
import sys
import threading
import subprocess

import platform_support

# Kanagawa Paper ink accents, matching the overlay bar's level-* borders.
PHASE_COLORS = {
    "on": "#8ea49e",
    "starting": "#b6927b",
    "error": "#c4746e",
    "off": "#727169",
    "down": "#54546D",
}
PHASE_LABEL = {
    "on": "AI Mode: ON",
    "starting": "AI Mode: starting...",
    "error": "AI Mode: ERROR",
    "off": "AI Mode: OFF",
    "down": "Daemon not running",
}
BG = "#1F1F28"


def _icon_image(phase, size=64):
    from PIL import Image, ImageDraw, ImageFont

    color = PHASE_COLORS.get(phase, PHASE_COLORS["down"])
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((2, 2, size - 3, size - 3), radius=size // 5,
                           fill=BG, outline=color, width=max(3, size // 12))
    font = None
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc"):
        try:
            font = ImageFont.truetype(name, int(size * 0.42))
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((size / 2, size / 2), "AI", fill=color, font=font, anchor="mm")
    return img


class TrayApp:
    def __init__(self, config, script_path, request, describe):
        self.config = config
        self.script_path = script_path
        self.request = request
        self.describe = describe
        self.phase = "down"
        self.status = {}
        self.icon = None
        self._images = {}
        self._wake = threading.Event()
        try:
            self.poll_seconds = max(1.0, float(config.get("tray_poll_seconds", 3)))
        except (TypeError, ValueError):
            self.poll_seconds = 3.0

    # ------------------------------------------------------------------ run
    def run(self):
        import pystray
        from pystray import Menu, MenuItem as Item

        menu = Menu(
            Item(lambda _: PHASE_LABEL.get(self.phase, self.phase), None, enabled=False),
            Menu.SEPARATOR,
            Item("AI Mode", self.on_toggle, default=True,
                 checked=lambda _: self.phase in ("on", "starting"),
                 enabled=lambda _: self.phase != "down"),
            Item(self._label("Chat", "hotkey_chat"), self.on_chat),
            Item(self._label("Capture screen", "hotkey_capture"), self.on_capture,
                 enabled=lambda _: self.phase == "on"),
            Item("Session manager", self.on_session, enabled=lambda _: self.phase != "down"),
            Item("Sync notes now", self.on_sync, enabled=lambda _: bool(self.status.get("session"))),
            Menu.SEPARATOR,
            Item("Start daemon", self.on_start_daemon, visible=lambda _: self.phase == "down"),
            Item("Stop daemon", self.on_stop_daemon, visible=lambda _: self.phase != "down"),
            Item("Quit tray", self.on_quit),
        )
        self.icon = pystray.Icon("ai-assistant", self._image("down"), "AI Assistant", menu)
        self.icon.run(setup=self._setup)

    def _setup(self, icon):
        icon.visible = True
        threading.Thread(target=self._poll_loop, daemon=True).start()
        if platform_support.IS_WINDOWS:
            threading.Thread(target=self._hotkey_loop, daemon=True).start()

    def _label(self, text, hotkey_key):
        hotkey = (self.config.get(hotkey_key) or "").strip()
        if hotkey and platform_support.IS_WINDOWS:
            return f"{text}\t{hotkey.title()}"
        return text

    def _image(self, phase):
        if phase not in self._images:
            self._images[phase] = _icon_image(phase)
        return self._images[phase]

    # --------------------------------------------------------------- status
    def _poll_loop(self):
        autostart = str(self.config.get("tray_start_daemon", "1")).strip() not in ("0", "false", "no")
        first = True
        while True:
            self.refresh()
            if first and autostart and self.phase == "down":
                self.on_start_daemon()
            first = False
            self._wake.wait(self.poll_seconds)
            self._wake.clear()

    def refresh(self):
        status = self.request({"command": "ai_mode_status"}, 3.0)
        if not status or status.get("error") == "unauthorized":
            self.status, self.phase = {}, "down"
            tooltip = "AI Assistant: daemon not running"
        else:
            self.status, self.phase = status, status.get("phase", "off")
            tooltip = "AI Assistant\n" + "\n".join(l for l in self.describe(status) if l)
        if self.icon is None:
            return
        self.icon.icon = self._image(self.phase)
        self.icon.title = tooltip[:127]           # Windows caps tooltips at 128 chars
        self.icon.update_menu()

    def _nudge(self):
        self._wake.set()

    # -------------------------------------------------------------- actions
    def _spawn(self, flag, log_name=None):
        argv = [platform_support.gui_python(), self.script_path, flag]
        out = subprocess.DEVNULL
        if log_name:
            log_dir = os.path.join(platform_support.data_dir(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            out = open(os.path.join(log_dir, log_name), "ab", buffering=0)
        try:
            subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                             cwd=os.path.dirname(self.script_path),
                             **platform_support.detached_popen_kwargs())
        except OSError as e:
            platform_support.notify(f"Could not launch {flag}: {e}")
        finally:
            if out is not subprocess.DEVNULL:
                out.close()

    def _background(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def on_toggle(self, *_):
        def work():
            self.request({"command": "ai_mode_toggle"}, 15.0)
            self._nudge()
        self._background(work)

    def on_chat(self, *_):
        self._spawn("--chat")

    def on_capture(self, *_):
        self._spawn("--capture")

    def on_session(self, *_):
        self._spawn("--gui")

    def on_sync(self, *_):
        self._background(lambda: self.request({"command": "sync_queue"}, 10.0))

    def on_start_daemon(self, *_):
        self._spawn("--daemon", log_name="daemon.log")
        # Give it a moment to bind before the next poll paints it.
        threading.Timer(2.0, self._nudge).start()

    def on_stop_daemon(self, *_):
        def work():
            self.request({"command": "shutdown"}, 30.0)
            self._nudge()
        self._background(work)

    def on_quit(self, *_):
        self.icon.stop()

    # -------------------------------------------------------------- hotkeys
    def _hotkey_loop(self):
        """RegisterHotKey + a message pump on this thread (Windows only)."""
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                       wintypes.UINT, wintypes.UINT]
        MOD_NOREPEAT, WM_HOTKEY = 0x4000, 0x0312

        bindings = {}
        for hotkey_id, (key, action) in enumerate((
            ("hotkey_chat", self.on_chat),
            # Spawned even when AI Mode is off: the capture client explains why
            # it refuses, so that message lives in one place.
            ("hotkey_capture", self.on_capture),
            ("hotkey_ai_toggle", self.on_toggle),
        ), start=1):
            spec = self.config.get(key, "")
            parsed = platform_support.parse_hotkey(spec)
            if not parsed:
                if spec:
                    print(f"[AI Tray] Ignoring unparseable {key}={spec!r}", file=sys.stderr)
                continue
            mods, vk = parsed
            if user32.RegisterHotKey(None, hotkey_id, mods | MOD_NOREPEAT, vk):
                bindings[hotkey_id] = action
            else:
                print(f"[AI Tray] {key}={spec} is taken by another app.", file=sys.stderr)

        if not bindings:
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam in bindings:
                bindings[msg.wParam]()


def run(config, script_path, request, describe):
    try:
        import pystray  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:
        platform_support.notify(f"The tray needs pystray and Pillow ({e}). "
                                "Delete ai-tools/.venv and relaunch to reinstall.")
        sys.exit(1)
    TrayApp(config, script_path, request, describe).run()
