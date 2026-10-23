#!/usr/bin/env python3
"""Per-OS plumbing for the AI assistant.

Everything that differs between Linux (Sway/waybar), Windows and macOS lives
here so ai_assistant.py can stay one code path: data directories, the daemon
IPC transport, server process groups, audio playback, notifications, the
clipboard and command-line splitting. Stdlib only (ctypes on Windows), so it
imports before the bootstrapped venv exists.

The Linux branches reproduce the original behaviour exactly; Windows and macOS
are additive. A new platform means a new branch in each helper, nothing more.
"""
import os
import re
import sys
import json
import time
import shlex
import shutil
import socket
import secrets
import tempfile
import subprocess

IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

OS_NAME = "Windows" if IS_WINDOWS else ("macOS" if IS_MAC else "Linux")

# Where status lives on this platform -- used in user-facing hints.
STATUS_SURFACE = "waybar" if IS_LINUX else "the tray icon"

DEFAULT_TCP_PORT = 9393


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def data_dir():
    """Per-user data root (sessions, memory db, TTS models, logs)."""
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, "ai_assistant")
    if IS_MAC:
        return os.path.expanduser("~/Library/Application Support/ai_assistant")
    return os.path.expanduser("~/.local/share/ai_assistant")


def venv_python(venv_dir, windowless=False):
    if IS_WINDOWS:
        return os.path.join(venv_dir, "Scripts", "pythonw.exe" if windowless else "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def launched_windowless():
    """True when started via pythonw (tray autostart, shortcuts)."""
    return IS_WINDOWS and os.path.basename(sys.executable).lower().startswith("pythonw")


def gui_python():
    """Interpreter for spawning GUI children without flashing a console."""
    if IS_WINDOWS:
        candidate = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def is_local_path(target):
    """Local file path (vs. a Nextcloud-relative one) for the ODT target."""
    if target.startswith("/") or target.startswith("~") or target.startswith("file://"):
        return True
    return bool(re.match(r"^[A-Za-z]:[\\/]", target)) or target.startswith("\\\\")


def local_path_from_target(target):
    path = os.path.expanduser(target.replace("file://", ""))
    if IS_WINDOWS and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]                       # file:///C:/x -> C:/x
    return path


# ---------------------------------------------------------------------------
# GTK on Windows
# ---------------------------------------------------------------------------
_DLL_HANDLES = []


def prepare_gtk_dll_path():
    """Let PyGObject find a gvsbuild / standalone GTK runtime on Windows.

    Python 3.8+ no longer resolves extension DLL dependencies through PATH, so
    the GTK bin directory is registered explicitly. AI_ASSISTANT_GTK_BIN wins;
    otherwise the usual gvsbuild install locations are tried.
    """
    if not IS_WINDOWS:
        return
    candidates = [
        os.environ.get("AI_ASSISTANT_GTK_BIN"),
        r"C:\gtk\bin",
        r"C:\gtk-build\gtk\x64\release\bin",
    ]
    for directory in candidates:
        if not directory or not os.path.isdir(directory):
            continue
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(directory))
        typelibs = os.path.join(os.path.dirname(directory), "lib", "girepository-1.0")
        if os.path.isdir(typelibs):
            os.environ.setdefault("GI_TYPELIB_PATH", typelibs)
        return


def gtk_install_hint():
    if IS_WINDOWS:
        return (
            "GTK 3 + PyGObject were not found. Install a GTK runtime (gvsbuild release, e.g. "
            "to C:\\gtk), then `pip install` its bundled PyGObject and pycairo wheels into the "
            "Python you launch this with. Set AI_ASSISTANT_GTK_BIN if GTK lives elsewhere. "
            "Then delete ai-tools\\.venv and run again. See README.md."
        )
    if IS_MAC:
        return "GTK 3 + PyGObject were not found. Run: brew install gtk+3 pygobject3"
    return "GTK 3 + PyGObject were not found. Install gtk3 and python-gobject."


# ---------------------------------------------------------------------------
# command lines from the config
# ---------------------------------------------------------------------------
def split_command(raw):
    """argv from a config command string.

    POSIX shlex would eat the backslashes in C:\\models\\x.gguf, so Windows
    splits in non-POSIX mode and strips the quotes that mode leaves behind.
    """
    if not IS_WINDOWS:
        return shlex.split(raw)
    out = []
    for tok in shlex.split(raw, posix=False):
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            tok = tok[1:-1]
        out.append(tok)
    return out


# ---------------------------------------------------------------------------
# IPC: AF_UNIX on POSIX, loopback TCP + token on Windows
# ---------------------------------------------------------------------------
def default_ipc_address():
    if IS_WINDOWS:
        return f"tcp://127.0.0.1:{DEFAULT_TCP_PORT}"
    return "/tmp/ai_assistant.sock"


def _parse(address):
    """('unix', path) or ('tcp', (host, port))."""
    address = (address or "").strip()
    if address.startswith("tcp://"):
        host, _, port = address[len("tcp://"):].rpartition(":")
        try:
            return "tcp", (host or "127.0.0.1", int(port))
        except ValueError:
            return "tcp", ("127.0.0.1", DEFAULT_TCP_PORT)
    if not hasattr(socket, "AF_UNIX"):
        # A shared .config may still carry the Linux socket path.
        return "tcp", ("127.0.0.1", DEFAULT_TCP_PORT)
    return "unix", address


def _token_path():
    return os.path.join(data_dir(), "ipc_token")


def _read_token():
    try:
        with open(_token_path(), "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def ipc_maybe_running(address):
    """Cheap pre-check: a unix socket must exist; TCP has to be tried."""
    kind, target = _parse(address)
    return os.path.exists(target) if kind == "unix" else True


def ipc_listen(address, backlog=5):
    """Bind the daemon's listening socket.

    TCP is loopback-only and every request must carry a per-user random token
    (stored under LOCALAPPDATA), matching the owner-only access a unix socket
    in /tmp gets from its file mode.
    """
    kind, target = _parse(address)
    if kind == "unix":
        if os.path.exists(target):
            os.remove(target)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(target)
    else:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if IS_WINDOWS and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        server.bind(target)
        os.makedirs(data_dir(), exist_ok=True)
        with open(_token_path(), "w", encoding="utf-8") as f:
            f.write(secrets.token_hex(24))
    server.listen(backlog)
    return server


def ipc_authorized(address, request):
    kind, _ = _parse(address)
    if kind == "unix":
        return True
    expected = _read_token()
    return bool(expected) and secrets.compare_digest(str(request.get("token", "")), expected)


def ipc_request(address, payload, timeout=15.0):
    """Send one JSON command and read the reply to EOF.

    Raises ConnectionRefusedError/FileNotFoundError when nothing listens.
    """
    kind, target = _parse(address)
    if kind == "unix":
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    else:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        payload = dict(payload, token=_read_token())
    try:
        client.settimeout(timeout)
        client.connect(target)
        client.sendall(json.dumps(payload).encode("utf-8"))
        # Read to EOF; the daemon closes the connection after replying, and a
        # single recv() would truncate larger payloads such as memory_stats.
        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return json.loads(b"".join(chunks).decode("utf-8"))
    finally:
        client.close()


def daemon_not_running_hint():
    if IS_LINUX:
        return "Daemon not running! Start it or check dinitctl."
    return "Daemon not running! Start it from the tray icon (or run ai_assistant.py --daemon)."


# ---------------------------------------------------------------------------
# child process groups (llama-server)
# ---------------------------------------------------------------------------
def detached_popen_kwargs():
    """Popen kwargs that put the child in its own group, with no console window."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def no_window_kwargs():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WINDOWS else {}


def _win_kernel32():
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    return k32


def win_process_info(pid):
    """(alive, image_path) for a Windows PID, without signalling it.

    os.kill(pid, 0) is not a probe on Windows: signal 0 is CTRL_C_EVENT.
    """
    import ctypes
    from ctypes import wintypes
    k32 = _win_kernel32()
    handle = k32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False, ""
    try:
        code = wintypes.DWORD()
        alive = bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        image = buf.value if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)) else ""
        return alive, image
    finally:
        k32.CloseHandle(handle)


def win_kill_tree(pid, label="server", grace=8.0, expect_images=None, exit_timeout=120.0):
    """Terminate a Windows server process and wait until it has really exited.

    A windowless console server has no polite shutdown channel (no WM_CLOSE,
    no console for CTRL_BREAK), and taskkill misreports some of them as "no
    running instance", so this calls TerminateProcess directly. Termination is
    not instant: a llama-server killed mid-load can sit in the GPU driver for
    a minute while VRAM is torn down, still holding its port, so we wait on
    the process handle rather than assume it is gone.

    expect_images (lowercase exe stems) guards PIDs read back from the state
    file: Windows recycles PIDs quickly, so an unrelated process is left alone.
    """
    import ctypes
    from ctypes import wintypes

    alive, image = win_process_info(pid)
    if not alive:
        return
    if expect_images:
        stem = os.path.splitext(os.path.basename(image))[0].lower()
        if stem not in expect_images:
            print(f"[AI Mode] PID {pid} is now {image or 'unknown'}; not killing {label}.",
                  file=sys.stderr)
            return

    k32 = _win_kernel32()
    k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.WaitForSingleObject.restype = wintypes.DWORD
    k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    # PROCESS_TERMINATE | SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
    handle = k32.OpenProcess(0x0001 | 0x00100000 | 0x1000, False, int(pid))
    if not handle:
        print(f"[AI Mode] Cannot open {label} (PID {pid}): error {ctypes.get_last_error()}",
              file=sys.stderr)
        return
    try:
        k32.TerminateProcess(handle, 1)
        if k32.WaitForSingleObject(handle, int(exit_timeout * 1000)) == 0:   # WAIT_OBJECT_0
            print(f"[AI Mode] {label} terminated.", file=sys.stderr)
        else:
            print(f"[AI Mode] {label} (PID {pid}) still exiting after {exit_timeout:.0f}s.",
                  file=sys.stderr)
    finally:
        k32.CloseHandle(handle)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def play_wav(data):
    """Play an in-memory WAV, blocking until done (non-Linux; Linux uses pw-play)."""
    if IS_WINDOWS:
        import winsound
        winsound.PlaySound(data, winsound.SND_MEMORY)
        return
    player = None
    if IS_MAC:
        player = ["afplay"]
    else:
        for cand in (["pw-play"], ["paplay"], ["aplay", "-q"]):
            if shutil.which(cand[0]):
                player = cand
                break
    if not player:
        print("[AI Assistant] No audio player found for TTS.", file=sys.stderr)
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        subprocess.run(player + [path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# notifications & clipboard
# ---------------------------------------------------------------------------
def notify(message, title="AI Assistant"):
    """Desktop notification, degrading to stderr where nothing is available."""
    if IS_WINDOWS:
        print(f"[{title}] {message}", file=sys.stderr)
        try:
            import ctypes
            # MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST. A modal
            # box, because launched from the tray there is no console to read.
            ctypes.WinDLL("user32").MessageBoxW(None, message, title, 0x40 | 0x10000 | 0x40000)
        except (OSError, AttributeError):
            pass
        return
    if IS_MAC and shutil.which("osascript"):
        try:
            subprocess.run(
                ["osascript", "-e", "on run argv", "-e",
                 "display notification (item 1 of argv) with title (item 2 of argv)",
                 "-e", "end run", message, title],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    if shutil.which("notify-send"):
        try:
            subprocess.run(["notify-send", title, message],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    print(f"[{title}] {message}", file=sys.stderr)


def copy_image_to_clipboard(png_path):
    """Best-effort: put a captured PNG on the clipboard (non-Wayland paths)."""
    try:
        if IS_WINDOWS:
            _win_copy_image(png_path)
        elif IS_MAC:
            subprocess.run(
                ["osascript", "-e",
                 f'set the clipboard to (read (POSIX file "{png_path}") as «class PNGf»)'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif shutil.which("xclip"):
            subprocess.run(["xclip", "-selection", "clipboard", "-t", "image/png", "-i", png_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[AI Assistant] Clipboard copy failed: {e}", file=sys.stderr)


def _win_copy_image(png_path):
    import io
    import ctypes
    from ctypes import wintypes
    from PIL import Image

    buf = io.BytesIO()
    Image.open(png_path).convert("RGB").save(buf, "BMP")
    dib = buf.getvalue()[14:]                  # CF_DIB is a BMP minus its file header

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    u32 = ctypes.WinDLL("user32", use_last_error=True)
    k32.GlobalAlloc.restype = wintypes.HGLOBAL
    k32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    k32.GlobalLock.restype = wintypes.LPVOID
    k32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    k32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    k32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    u32.OpenClipboard.argtypes = [wintypes.HWND]
    u32.SetClipboardData.restype = wintypes.HANDLE
    u32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]

    handle = k32.GlobalAlloc(0x0002, len(dib))  # GMEM_MOVEABLE
    if not handle:
        raise OSError("GlobalAlloc failed")
    ptr = k32.GlobalLock(handle)
    ctypes.memmove(ptr, dib, len(dib))
    k32.GlobalUnlock(handle)

    if not u32.OpenClipboard(None):
        k32.GlobalFree(handle)
        raise OSError("clipboard is busy")
    try:
        u32.EmptyClipboard()
        if not u32.SetClipboardData(8, handle):   # CF_DIB; the clipboard now owns it
            k32.GlobalFree(handle)
    finally:
        u32.CloseClipboard()


# ---------------------------------------------------------------------------
# global hotkeys (Windows; Sway binds keys itself)
# ---------------------------------------------------------------------------
_MODS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8, "super": 0x8}


def parse_hotkey(spec):
    """'ctrl+alt+a' -> (modifiers, virtual-key) for RegisterHotKey, or None."""
    parts = [p.strip().lower() for p in (spec or "").split("+") if p.strip()]
    if not parts:
        return None
    mods = 0
    for p in parts[:-1]:
        if p not in _MODS:
            return None
        mods |= _MODS[p]
    key = parts[-1]
    if len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    elif re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", key):
        vk = 0x6F + int(key[1:])
    else:
        return None
    return mods, vk
