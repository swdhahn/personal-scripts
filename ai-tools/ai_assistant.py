#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import urllib.request
import venv

# ==============================================================================
# 1. SELF-BOOTSTRAPPING ENVIRONMENT & DEPENDENCY CHECK
# ==============================================================================
def bootstrap_env():
    """Ensures system dependencies, Python packages, and Kokoro TTS models exist."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, ".venv")
    venv_python = os.path.join(venv_dir, "bin", "python")

    if os.path.abspath(sys.prefix) != os.path.abspath(venv_dir):
        if not os.path.exists(venv_python):
            missing_bins = [cmd for cmd in ("grim", "slurp", "wl-copy", "espeak-ng") if not shutil.which(cmd)]
            if missing_bins and sys.stdin.isatty():
                print(f"[AI Assistant] Missing system packages: {', '.join(missing_bins)}")
                print("[AI Assistant] Installing required system packages via pacman...")
                subprocess.run([
                    "sudo", "pacman", "-S", "--needed",
                    "grim", "slurp", "wl-clipboard", "gtk3", "gtk-layer-shell", "python-gobject", "espeak-ng"
                ])

            print("[AI Assistant] Creating isolated .venv (with system GTK bindings)...")
            builder = venv.EnvBuilder(system_site_packages=True, with_pip=True)
            builder.create(venv_dir)

        # The waybar module re-runs every few seconds; skip the dependency probe
        # for it and go straight to the exec so the bar stays responsive.
        fast_path = any(a in ("--waybar", "--ai-toggle", "--ai-on", "--ai-off") for a in sys.argv[1:])

        # Ensure all required pip packages (including kokoro-onnx) are present
        check_cmd = [venv_python, "-c", "import requests, odf, kokoro_onnx, soundfile, numpy, pygments"]
        if not fast_path and subprocess.call(check_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            print("[AI Assistant] Installing Python packages (requests, odfpy, kokoro-onnx, soundfile, numpy, pygments)...")
            subprocess.check_call([
                venv_python, "-m", "pip", "install", "--upgrade",
                "pip", "requests", "odfpy", "kokoro-onnx", "soundfile", "numpy", "pygments"
            ])

        os.execv(venv_python, [venv_python, os.path.abspath(__file__)] + sys.argv[1:])

bootstrap_env()

# ==============================================================================
# 2. IMPORTS
# ==============================================================================
import re
import io
import gc
import json
import time
import queue
import shlex
import socket
import base64
import signal
import argparse
import threading
import tempfile
from datetime import datetime
import requests
import soundfile as sf
from kokoro_onnx import Kokoro

from odf.opendocument import load, OpenDocumentText
from odf.style import Style, TextProperties, ParagraphProperties
from odf.text import P, H, Span, List as OdfList, ListItem as OdfListItem

import memory_db

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GtkLayerShell', '0.1')
from gi.repository import Gtk, Gdk, GLib, Pango, GtkLayerShell

# ==============================================================================
# 3. CONFIGURATION PARSER (9x9x Port Standard)
# ==============================================================================
def load_config():
    """Reads ../.config first, falling back to local .config."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.abspath(os.path.join(script_dir, "..", ".config"))
    if not os.path.exists(cfg_path):
        cfg_path = os.path.abspath(os.path.join(script_dir, ".config"))

    config = {
        "default_project_path": os.path.expanduser("~/Programming"),
        "llama_url": "http://127.0.0.1:9090/v1/chat/completions",
        "llama_model": "Abiray/Huihui-Qwen3.6-35B-A3B-abliterated-GGUF:Q4_K_M",
        "llama_small_url": "http://127.0.0.1:9191/v1/chat/completions",
        "llama_small_model": "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0",
        "tts_url": "http://127.0.0.1:9292/v1/audio/speech",
        "tts_voice": "am_michael",
        "tts_speed": "0.95",
        "tts_model_dir": os.path.expanduser("~/.local/share/ai_assistant/tts"),
        "nextcloud_url": "https://your-nextcloud.domain.com",
        "nextcloud_user": "your_username",
        "nextcloud_token": "your_app_password_token",
        "session_history_dir": os.path.expanduser("~/.local/share/ai_assistant/sessions"),
        "socket_path": "/tmp/ai_assistant.sock",

        # --- AI Mode: llama.cpp server lifecycle ---
        "llama_server_cmd": "",
        "llama_small_server_cmd": "",
        "llama_health_url": "http://127.0.0.1:9090/health",
        "llama_small_health_url": "http://127.0.0.1:9191/health",
        "ai_mode_state_file": os.path.expanduser("~/.local/share/ai_assistant/ai_mode.json"),
        "ai_mode_start_timeout": "180",
        "llama_context": "",
        "waybar_signal": "8",

        # --- Chat memory ---
        "memory_db_path": os.path.expanduser("~/.local/share/ai_assistant/memory.db"),
        "memory_top_k": "6",
        "memory_recency_half_life_days": "30",
        "memory_min_importance": "0.25",
        "memory_compress_threshold": "2048",
        "memory_summarize_after": "20",
        "chat_system_prompt": (
            "You are a concise, technically fluent desktop assistant running locally on the "
            "user's Linux machine. You have a persistent memory of past conversations: relevant "
            "entries are supplied to you automatically, and you can call the memory_search tool "
            "to look up anything else you have been told before. Never invent a memory -- if you "
            "do not know something, search for it or say you do not know. Format with Markdown."
        ),
    }

    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("--") or line.startswith("//"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    # Only expand a leading ~. Command strings keep embedded paths
                    # intact and are expanded per-token by config_argv().
                    config[k] = os.path.expanduser(v) if v.startswith("~") else v
    return config


def config_argv(config, key):
    """Split a shell command from the config into argv, expanding ~ per token."""
    raw = (config.get(key) or "").strip()
    if not raw:
        return []
    try:
        return [os.path.expanduser(tok) for tok in shlex.split(raw)]
    except ValueError as e:
        print(f"[AI Assistant] Malformed command in config key {key!r}: {e}", file=sys.stderr)
        return []


def context_size(config):
    """Context window, from `llama_context` or the -c flag of the server command."""
    raw = (config.get("llama_context") or "").strip()
    if raw.isdigit():
        return int(raw)
    argv = config_argv(config, "llama_server_cmd")
    for flag in ("-c", "--ctx-size"):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 < len(argv) and argv[index + 1].isdigit():
                return int(argv[index + 1])
    return 0


def config_num(config, key, default, cast=float):
    try:
        return cast(config.get(key, default))
    except (TypeError, ValueError):
        return cast(default)

# ==============================================================================
# 3.5 KANAGAWA PAPER (INK) THEME
# ==============================================================================
# Palette taken from thesimonho/kanagawa-paper.nvim --
#   lua/kanagawa-paper/colors.lua  (M.palette)
#   lua/kanagawa-paper/themes/ink.lua  (semantic mapping)
KANAGAWA = {
    "bg":       "#1F1F28",  # sumiInk3      ink bg
    "bg_dim":   "#181820",  # sumiInk1
    "bg_m3":    "#16161D",  # sumiInk0
    "surface":  "#2A2A37",  # sumiInk4      ui.float.bg / pmenu.bg
    "raised":   "#363646",  # sumiInk5      pmenu.bg_sel
    "border":   "#625e5a",  # dragonBlack6  ui.border
    "fg":       "#DCD7BA",  # fujiWhite
    "fg_float": "#C8C093",  # oldWhite      ui.float.fg
    "fg_dim":   "#9e9b93",  # dragonGray2
    "nontext":  "#54546D",  # sumiInk6
    "accent":   "#c4b28a",  # dragonYellow  ink primary accent
    "info":     "#658594",  # dragonBlue    accent1
    "warn":     "#b6927b",  # dragonOrange  accent2
    "ok":       "#8ea49e",  # dragonAqua    ui.special
    "error":    "#c4746e",  # dragonRed     accent5
    "violet":   "#8992a7",  # dragonViolet  ui.win_separator
    "pink":     "#a292a3",  # dragonPink
    "gray":     "#a6a69c",  # dragonGray    syn.parameter
    "comment":  "#727169",  # fujiGray      syn.comment
    "docstr":   "#717e67",  # dragonGreen3  syn.docstring
    "string":   "#8a9a7b",  # dragonGreen2  syn.string
    "func":     "#859fac",  # dragonBlue2   syn.fun
}

# Syntax colours taken from the same theme's `syn` table (themes/ink.lua), so
# fenced code in chat matches what the editor shows.
SYNTAX = {
    "comment":  KANAGAWA["comment"],
    "docstring": KANAGAWA["docstr"],
    "keyword":  KANAGAWA["pink"],
    "statement": KANAGAWA["violet"],
    "string":   KANAGAWA["string"],
    "number":   KANAGAWA["pink"],
    "operator": KANAGAWA["error"],
    "function": KANAGAWA["func"],
    "type":     KANAGAWA["ok"],
    "constant": KANAGAWA["warn"],
    "member":   KANAGAWA["error"],
    "parameter": KANAGAWA["gray"],
    "punct":    KANAGAWA["fg_dim"],
    "name":     KANAGAWA["accent"],
    "error":    KANAGAWA["error"],
}

GTK_CSS = """
* {{
    font-family: 'JetBrainsMono Nerd Font', 'JetBrains Mono', monospace;
    font-size: 13px;
}}

window {{ background-color: {bg}; color: {fg}; }}
label  {{ color: {fg}; }}

.heading   {{ color: {accent};  font-weight: bold; }}
.dim       {{ color: {fg_dim};  }}
.status-ok {{ color: {ok};      }}
.status-warn {{ color: {warn};  }}
.status-err  {{ color: {error}; }}

/* ---- daemon overlay bar (gtk-layer-shell) ---- */
window.status-bar {{
    background-color: rgba(31, 31, 40, 0.94);
    border: 2px solid {accent};
    border-radius: 8px;
}}
window.status-bar label {{
    color: {fg};
    font-size: 14px;
    font-weight: bold;
    padding: 10px 24px;
}}
window.status-bar.level-ok   {{ border-color: {ok};    }}
window.status-bar.level-warn {{ border-color: {warn};  }}
window.status-bar.level-err  {{ border-color: {error}; }}

/* ---- inputs ---- */
entry {{
    background-color: {surface};
    color: {fg};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 6px 8px;
    caret-color: {accent};
}}
/* Adwaita paints focus as an inset box-shadow, which would otherwise show
   through as its stock blue -- clear it and keep the border as the indicator. */
entry:focus {{
    border-color: {accent};
    box-shadow: none;
    outline: none;
}}
entry selection {{ background-color: {raised}; color: {fg}; }}

textview {{ background-color: transparent; color: {fg}; caret-color: {accent}; }}
textview text {{
    background-color: {surface};
    color: {fg};
    caret-color: {accent};
    -gtk-secondary-caret-color: {accent};
}}
textview text selection {{ background-color: {raised}; color: {fg}; }}
scrolledwindow:focus-within {{ border-color: {accent}; box-shadow: none; }}
*:focus {{ outline-color: {accent}; }}

/* ---- buttons ----
   GTK packs a GtkLabel inside every button, and the global `label` rule above
   would win over a rule that only targets the button, so each state names the
   child label explicitly. */
button {{
    background-image: none;
    background-color: {surface};
    color: {fg};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 6px 12px;
}}
button label {{ color: {fg}; }}
button:hover, button:hover label  {{ color: {accent}; }}
button:hover  {{ background-color: {raised}; }}
button:active {{ background-color: {raised}; }}
button:disabled {{
    background-color: {bg_dim};
    border-color: {surface};
}}
button:disabled, button:disabled label {{ color: {nontext}; }}
button.accent-btn {{
    background-color: {raised};
    border-color: {accent};
    font-weight: bold;
}}
button.accent-btn, button.accent-btn label {{ color: {accent}; }}
button.accent-btn:hover {{ background-color: {accent}; }}
button.accent-btn:hover, button.accent-btn:hover label {{ color: {bg}; }}
button.accent-btn:disabled {{ background-color: {bg_dim}; border-color: {border}; }}
button.accent-btn:disabled, button.accent-btn:disabled label {{ color: {nontext}; }}
button.danger-btn {{ border-color: {error}; }}
button.danger-btn, button.danger-btn label {{ color: {error}; }}
button.danger-btn:hover {{ background-color: {error}; }}
button.danger-btn:hover, button.danger-btn:hover label {{ color: {bg}; }}

/* ---- containers ----
   The viewport and the undershoot/overshoot nodes must be painted too: left
   transparent they render as black bands the moment a scrollbar appears. */
scrolledwindow, viewport, overlay, scrolledwindow viewport {{
    background-color: {bg};
}}
scrolledwindow {{
    border: 1px solid {surface};
    border-radius: 4px;
}}
scrolledwindow undershoot.top, scrolledwindow undershoot.bottom,
scrolledwindow undershoot.left, scrolledwindow undershoot.right,
scrolledwindow overshoot.top, scrolledwindow overshoot.bottom,
scrolledwindow overshoot.left, scrolledwindow overshoot.right,
scrolledwindow junction {{
    background-image: none;
    background-color: transparent;
    border: none;
}}
scrollbar {{ background-color: {bg_dim}; border: none; }}
scrollbar slider {{
    background-color: {raised};
    border-radius: 6px;
    min-width: 8px;
    min-height: 8px;
}}
scrollbar slider:hover {{ background-color: {nontext}; }}

expander title  {{ color: {fg_dim}; }}
expander title:hover {{ color: {accent}; }}
separator {{ background-color: {surface}; min-height: 1px; min-width: 1px; }}

/* ---- chat ---- */
.chat-log, .chat-surface {{ background-color: {bg}; }}
.chat-log {{ padding: 10px; }}
.chat-user {{
    background-color: {raised};
    border-radius: 6px;
    padding: 8px 12px;
}}
.chat-assistant {{
    background-color: {surface};
    border-left: 2px solid {info};
    border-radius: 6px;
    padding: 8px 12px;
}}
.chat-user label     {{ color: {fg};       }}
.chat-assistant label {{ color: {fg_float}; }}

/* ---- sidebar ---- */
.sidebar {{
    background-color: {bg_dim};
    border-right: 1px solid {surface};
}}
.sidebar scrolledwindow {{ border: none; background-color: {bg_dim}; }}
.sidebar viewport, .sidebar list {{ background-color: {bg_dim}; }}
.sidebar-title {{ color: {fg_dim}; font-size: 11px; font-weight: bold; }}

list, list row {{ background-color: transparent; }}
list row {{ padding: 5px 7px; border-radius: 4px; }}
list row:hover {{ background-color: {surface}; }}
list row:selected {{ background-color: {raised}; }}
list row:selected label {{ color: {accent}; }}
.row-title {{ color: {fg}; }}
.row-sub {{ color: {nontext}; font-size: 10px; }}

button.icon-btn {{
    background-image: none;
    background-color: transparent;
    border: none;
    box-shadow: none;
    padding: 4px 9px;
    margin: 0;
    border-radius: 6px;
}}
button.icon-btn label {{ color: {fg_dim}; font-size: 15px; }}
button.icon-btn:hover {{ background-color: {surface}; }}
button.icon-btn:hover label {{ color: {accent}; }}

entry.search {{ background-color: {surface}; border-radius: 6px; }}

/* ---- composer: one bordered bar containing the send button + text view ---- */
.input-bar {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 2px 4px;
}}
.input-bar:focus-within {{ border-color: {accent}; }}
.input-bar scrolledwindow,
.input-bar viewport,
.input-bar textview,
.input-bar textview text {{
    background-color: transparent;
    border: none;
}}
button.send-btn {{
    background-image: none;
    background-color: transparent;
    border: none;
    box-shadow: none;
    padding: 6px 10px;
    margin: 2px;
    border-radius: 7px;
}}
button.send-btn label {{ color: {accent}; font-size: 17px; }}
button.send-btn:hover {{ background-color: {raised}; }}
button.send-btn:disabled {{ background-color: transparent; }}
button.send-btn:disabled label {{ color: {nontext}; }}

/* ---- jump-to-bottom pill, overlaid on the transcript ---- */
button.scroll-btn {{
    background-color: {raised};
    border: 1px solid {accent};
    border-radius: 13px;
    padding: 3px 11px;
    margin: 10px;
}}
button.scroll-btn label {{ color: {accent}; font-size: 13px; }}
button.scroll-btn:hover {{ background-color: {accent}; }}
button.scroll-btn:hover label {{ color: {bg}; }}

/* ---- fenced code ---- */
.code-block {{
    background-color: {bg_m3};
    border: 1px solid {surface};
    border-radius: 6px;
    padding: 7px 9px;
}}
.code-block scrolledwindow, .code-block viewport {{
    background-color: transparent;
    border: none;
}}
.code-lang {{ color: {nontext}; font-size: 10px; }}
.code-block scrollbar, .md-table scrollbar {{ background-color: transparent; }}
.code-block scrollbar slider, .md-table scrollbar slider {{
    min-height: 5px;
    min-width: 24px;
    background-color: {surface};
}}
.code-block scrollbar slider:hover, .md-table scrollbar slider:hover {{
    background-color: {nontext};
}}
.md-body {{ color: {fg_float}; }}


/* ---- token usage / speed ---- */
.chat-stats {{ color: {nontext}; font-size: 10px; }}

/* ---- reasoning dropdown ---- */
.think-box {{
    background-color: {bg_m3};
    border: 1px solid {surface};
    border-radius: 6px;
    padding: 4px 8px;
}}
.think-box > label {{ color: {violet}; font-size: 11px; }}
.think-box:hover > label {{ color: {accent}; }}
.think-box scrolledwindow {{ border: none; background-color: transparent; }}
.think-surface {{ background-color: transparent; }}
.think-body {{ color: {fg_dim}; font-size: 11px; }}
.think-box scrollbar slider {{ min-width: 6px; background-color: {surface}; }}

/* ---- compact sidebar controls ---- */
.sidebar button, button.thin-btn {{ padding: 3px 9px; min-height: 0; }}
.sidebar entry, entry.thin-entry {{
    padding: 2px 6px;
    min-height: 0;
    font-size: 12px;
}}
.sidebar list row {{ padding: 3px 6px; }}
.sidebar .sidebar-title {{ font-size: 10px; }}
.sidebar separator {{ margin: 2px 0; }}

/* ---- context menu ---- */
menu {{ background-color: {surface}; border: 1px solid {border}; padding: 2px; }}
menu menuitem {{ padding: 4px 10px; border-radius: 4px; }}
menu menuitem label {{ color: {fg}; }}
menu menuitem:hover {{ background-color: {raised}; }}
menu menuitem:hover label {{ color: {accent}; }}

/* ---- dialogs ---- */
.themed-dialog {{ background-color: {bg}; color: {fg}; }}
.themed-dialog label {{ color: {fg}; }}
.themed-dialog button {{ padding: 5px 14px; }}
spinbutton, spinbutton entry {{
    background-color: {surface};
    color: {fg};
    border: 1px solid {border};
    border-radius: 4px;
}}
spinbutton button {{ background-color: {surface}; border: none; }}
spinbutton button:hover {{ background-color: {raised}; }}
combobox button {{ padding: 3px 8px; }}
combobox window.popup, combobox window.popup list {{ background-color: {surface}; }}
.md-table {{ padding: 4px 2px; }}
.md-th {{ color: {accent}; }}
.md-td {{ color: {fg_float}; }}
.md-rule {{ background-color: {border}; min-height: 1px; }}

.chat-meta   {{ color: {nontext}; font-size: 11px; }}
.chat-error  {{ color: {error}; }}
.memory-chip {{ color: {ok}; font-size: 11px; }}
""".format(**KANAGAWA)

_THEME_INSTALLED = False


def install_theme():
    """Install the Kanagawa CSS once per process (daemon bar and popups share it)."""
    global _THEME_INSTALLED
    if _THEME_INSTALLED:
        return
    screen = Gdk.Screen.get_default()
    if screen is None:
        # Started without a display (e.g. by a service manager that has no
        # WAYLAND_DISPLAY). Skip rather than die inside the CSS provider.
        print("[AI Assistant] No display available; skipping theme install.", file=sys.stderr)
        return
    provider = Gtk.CssProvider()
    provider.load_from_data(GTK_CSS.encode("utf-8"))
    Gtk.StyleContext.add_provider_for_screen(
        screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    _THEME_INSTALLED = True


def style(widget, *classes):
    """Attach CSS classes to a widget and return it (keeps builders terse)."""
    ctx = widget.get_style_context()
    for name in classes:
        ctx.add_class(name)
    return widget


# ==============================================================================
# 3.6 MARKDOWN -> PANGO MARKUP (with Pygments-coloured code fences)
# ==============================================================================
from pygments import lex
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.token import Token
from pygments.util import ClassNotFound

# Pygments token roots mapped onto the theme's `syn` colours. Lookup walks up
# the token hierarchy, so Token.Name.Function.Magic resolves via Name.Function.
_TOKEN_COLORS = [
    (Token.Comment.Preproc,      SYNTAX["operator"]),
    (Token.Comment,              SYNTAX["comment"]),
    (Token.String.Doc,           SYNTAX["docstring"]),
    (Token.String.Escape,        SYNTAX["constant"]),
    (Token.String.Interpol,      SYNTAX["constant"]),
    (Token.String,               SYNTAX["string"]),
    (Token.Number,               SYNTAX["number"]),
    (Token.Keyword.Type,         SYNTAX["type"]),
    (Token.Keyword,              SYNTAX["keyword"]),
    (Token.Operator.Word,        SYNTAX["keyword"]),
    (Token.Operator,             SYNTAX["operator"]),
    (Token.Punctuation,          SYNTAX["punct"]),
    (Token.Name.Function,        SYNTAX["function"]),
    (Token.Name.Class,           SYNTAX["type"]),
    (Token.Name.Decorator,       SYNTAX["constant"]),
    (Token.Name.Builtin.Pseudo,  SYNTAX["constant"]),
    (Token.Name.Builtin,         SYNTAX["type"]),
    (Token.Name.Attribute,       SYNTAX["member"]),
    (Token.Name.Tag,             SYNTAX["keyword"]),
    (Token.Name.Constant,        SYNTAX["constant"]),
    (Token.Name.Variable,        SYNTAX["parameter"]),
    (Token.Name.Namespace,       SYNTAX["type"]),
    (Token.Name,                 SYNTAX["name"]),
    (Token.Generic.Error,        SYNTAX["error"]),
    (Token.Error,                SYNTAX["error"]),
]


def _token_color(token):
    while token is not None:
        for candidate, color in _TOKEN_COLORS:
            if token is candidate:
                return color
        token = token.parent
    return None


def esc(text):
    return GLib.markup_escape_text(text or "")


def highlight_code(code, language=""):
    """Return Pango markup for a code block, coloured with Pygments."""
    code = code.rstrip("\n")
    lexer = None
    if language:
        try:
            lexer = get_lexer_by_name(language.strip().lower())
        except ClassNotFound:
            lexer = None
    if lexer is None:
        try:
            lexer = guess_lexer(code)
        except (ClassNotFound, ValueError):
            lexer = None
    if lexer is None:
        return esc(code)

    out = []
    try:
        for token, value in lex(code, lexer):
            if not value:
                continue
            color = _token_color(token)
            out.append(f'<span foreground="{color}">{esc(value)}</span>' if color else esc(value))
    except Exception:
        return esc(code)
    return "".join(out).rstrip("\n")


_INLINE_RE = re.compile(
    r"(?P<code>`+[^`\n]+`+)"
    r"|(?P<bold>\*\*[^*\n]+\*\*|__[^_\n]+__)"
    r"|(?P<italic>\*[^*\n]+\*|(?<![A-Za-z0-9_])_[^_\n]+_(?![A-Za-z0-9_]))"
    r"|(?P<strike>~~[^~\n]+~~)"
    r"|(?P<link>\[[^\]\n]+\]\([^)\n]+\))"
)


def render_inline(text):
    """Bold/italic/inline-code/strikethrough/links -> Pango markup."""
    out, pos = [], 0
    for m in _INLINE_RE.finditer(text):
        out.append(esc(text[pos:m.start()]))
        kind = m.lastgroup
        raw = m.group()
        if kind == "code":
            inner = raw.strip("`")
            out.append(
                f'<tt><span background="{KANAGAWA["raised"]}" '
                f'foreground="{KANAGAWA["accent"]}"> {esc(inner)} </span></tt>'
            )
        elif kind == "bold":
            out.append(f"<b>{esc(raw[2:-2])}</b>")
        elif kind == "italic":
            out.append(f"<i>{esc(raw[1:-1])}</i>")
        elif kind == "strike":
            out.append(f"<s>{esc(raw[2:-2])}</s>")
        elif kind == "link":
            label, _, href = raw[1:-1].partition("](")
            out.append(
                f'<span foreground="{KANAGAWA["info"]}" underline="single">{esc(label)}</span>'
                f'<span foreground="{KANAGAWA["fg_dim"]}" size="small"> ({esc(href)})</span>'
            )
        pos = m.end()
    out.append(esc(text[pos:]))
    return "".join(out)


_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


_HEADING_SIZES = ("xx-large", "x-large", "large", "medium", "medium", "medium")
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)\s*([A-Za-z0-9_+-]*)\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")


def render_markdown_blocks(text):
    """Split a reply into renderable blocks.

    Returns a list of ("text", pango_markup) and ("code", language, code) pairs
    so the caller can give fenced code its own styled widget.
    """
    blocks, para, lines = [], [], (text or "").splitlines()
    code, lang, in_code = [], "", False
    index = [0]

    def flush_para():
        if para:
            blocks.append(("text", "\n".join(para)))
            para.clear()

    def peek(offset):
        pos = index[0] + offset
        return lines[pos] if 0 <= pos < len(lines) else None

    def consume(count):
        index[0] += count

    while index[0] < len(lines):
        line = lines[index[0]]
        index[0] += 1
        fence = _FENCE_RE.match(line)
        if fence:
            if in_code:
                blocks.append(("code", lang, "\n".join(code)))
                code, lang, in_code = [], "", False
            else:
                flush_para()
                lang, in_code = fence.group(1), True
            continue

        if in_code:
            code.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            para.append("")
            continue

        # A pipe table: header row, a |:---|---| separator, then body rows.
        # peek(0) is the next unread line: index already advanced past `line`.
        if stripped.startswith("|") and _TABLE_SEP_RE.match(peek(0) or ""):
            header = _split_row(stripped)
            consume(1)
            rows = []
            while True:
                nxt = peek(0)
                if nxt is None or not nxt.strip().startswith("|"):
                    break
                rows.append(_split_row(nxt))
                consume(1)
            flush_para()
            blocks.append(("table", header, rows))
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped.lstrip("#").strip()
            if heading:
                size = _HEADING_SIZES[min(level, len(_HEADING_SIZES)) - 1]
                para.append(
                    f'<span size="{size}" weight="bold" foreground="{KANAGAWA["accent"]}">'
                    f"{render_inline(heading)}</span>"
                )
                continue

        if set(stripped) <= {"-", "*", "_"} and len(stripped) >= 3:
            para.append(f'<span foreground="{KANAGAWA["nontext"]}">{"─" * 24}</span>')
            continue

        if stripped.startswith(">"):
            quote = stripped.lstrip(">").strip()
            para.append(
                f'<span foreground="{KANAGAWA["info"]}">┃ </span>'
                f'<span foreground="{KANAGAWA["fg_dim"]}"><i>{render_inline(quote)}</i></span>'
            )
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            indent = " " * (len(bullet.group(1)) // 2 * 2)
            para.append(
                f'{indent}<span foreground="{KANAGAWA["accent"]}">•</span> '
                f"{render_inline(bullet.group(2))}"
            )
            continue

        ordered = _ORDERED_RE.match(line)
        if ordered:
            indent = " " * (len(ordered.group(1)) // 2 * 2)
            para.append(
                f'{indent}<span foreground="{KANAGAWA["accent"]}">{ordered.group(2)}.</span> '
                f"{render_inline(ordered.group(3))}"
            )
            continue

        para.append(render_inline(line))

    if in_code:                       # reply ended mid-fence (still streaming)
        blocks.append(("code", lang, "\n".join(code)))
    flush_para()
    return [b for b in blocks if b[0] != "text" or b[1].strip()]


# ==============================================================================
# 4. MARKDOWN TO ODT PARSER
# ==============================================================================
def setup_odt_styles(doc):
    """Injects bold, italic, and code text/paragraph styles into the ODT document."""
    styles = {}
    
    bold = Style(name="AIBold", family="text")
    bold.addElement(TextProperties(fontweight="bold", fontweightasian="bold", fontweightcomplex="bold"))
    doc.automaticstyles.addElement(bold)
    styles["bold"] = bold

    italic = Style(name="AIItalic", family="text")
    italic.addElement(TextProperties(fontstyle="italic", fontstyleasian="italic", fontstylecomplex="italic"))
    doc.automaticstyles.addElement(italic)
    styles["italic"] = italic

    code = Style(name="AICodeInline", family="text")
    code.addElement(TextProperties(fontfamily="monospace", backgroundcolor="#f0f0f4"))
    doc.automaticstyles.addElement(code)
    styles["code"] = code

    code_block = Style(name="AICodeBlock", family="paragraph")
    code_block.addElement(ParagraphProperties(backgroundcolor="#f6f8fa", margintop="4pt", marginbottom="4pt", marginleft="10pt"))
    code_block.addElement(TextProperties(fontfamily="monospace", fontsize="10pt"))
    doc.automaticstyles.addElement(code_block)
    styles["code_block"] = code_block

    return styles

def parse_inline_markdown(text, styles):
    """Splits text into plain text and styled Span elements (**bold**, *italic*, `code`)."""
    pattern = re.compile(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)')
    parts = pattern.split(text)
    elements = []
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**') and len(part) >= 4:
            elements.append(Span(stylename=styles["bold"], text=part[2:-2]))
        elif part.startswith('*') and part.endswith('*') and len(part) >= 2:
            elements.append(Span(stylename=styles["italic"], text=part[1:-1]))
        elif part.startswith('`') and part.endswith('`') and len(part) >= 2:
            elements.append(Span(stylename=styles["code"], text=part[1:-1]))
        else:
            elements.append(part)
    return elements

def append_markdown_content(doc, md_text, styles):
    """Parses markdown lines into native ODT headings, lists, code blocks, and styled paragraphs."""
    in_code_block = False
    current_list = None

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            in_code_block = not in_code_block
            current_list = None
            continue

        if in_code_block:
            p = P(stylename=styles["code_block"])
            p.addText(raw_line)
            doc.text.addElement(p)
            continue

        if not line.strip():
            current_list = None
            continue

        if line.startswith("#"):
            current_list = None
            level = len(line) - len(line.lstrip('#'))
            header_text = line.lstrip('#').strip()
            h = H(outlinelevel=min(level, 4))
            for item in parse_inline_markdown(header_text, styles):
                if isinstance(item, str):
                    h.addText(item)
                else:
                    h.addElement(item)
            doc.text.addElement(h)
            continue

        if line.lstrip().startswith(("- ", "* ")):
            indent_content = line.lstrip()[2:]
            if current_list is None:
                current_list = OdfList()
                doc.text.addElement(current_list)

            item = OdfListItem()
            p = P()
            for part in parse_inline_markdown(indent_content, styles):
                if isinstance(part, str):
                    p.addText(part)
                else:
                    p.addElement(part)
            item.addElement(p)
            current_list.addElement(item)
            continue

        current_list = None
        p = P()
        for part in parse_inline_markdown(line, styles):
            if isinstance(part, str):
                p.addText(part)
            else:
                p.addElement(part)
        doc.text.addElement(p)

def extract_json(text):
    """Pull the first balanced JSON object out of a model reply.

    Local models wrap JSON in <think> blocks, prose, or fenced code; this walks
    the string so a brace inside a string literal does not end the object early.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    start = text.find("{")
    if start < 0:
        return None

    depth, in_str, esc = 0, False, False
    cuts = []
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                cuts.append(i + 1)
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            cuts.append(i + 1)
            if depth == 0 and ch == "}":
                try:
                    return json.loads(text[start:i + 1])
                except ValueError:
                    break
    return repair_json(text[start:], [c - start for c in cuts])


def repair_json(fragment, cuts):
    """Recover an object that generation cut off mid-way.

    Local models routinely burn their token budget on a <think> block and stop
    partway through the JSON. Walk the candidate cut points newest-first, close
    whatever brackets are still open at each, and keep the first that parses --
    a cut landing after a key rather than a value simply fails and we back up.
    """
    for cut in list(reversed(cuts))[:60]:
        candidate = fragment[:cut].rstrip().rstrip(",")
        stack, in_str, esc = [], False, False
        for ch in candidate:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        if in_str:
            continue
        try:
            return json.loads(candidate + "".join(reversed(stack)))
        except ValueError:
            continue
    return None


# ==============================================================================
# 5. DAEMON: STATUS BAR & SERVER
# ==============================================================================
class BottomStatusBar(Gtk.Window):
    def __init__(self):
        super().__init__(type=Gtk.WindowType.TOPLEVEL)
        GtkLayerShell.init_for_window(self)
        GtkLayerShell.set_layer(self, GtkLayerShell.Layer.OVERLAY)
        GtkLayerShell.set_anchor(self, GtkLayerShell.Edge.BOTTOM, True)
        GtkLayerShell.set_margin(self, GtkLayerShell.Edge.BOTTOM, 16)
        GtkLayerShell.set_keyboard_interactivity(self, False)

        self.set_decorated(False)
        self.set_app_paintable(True)

        install_theme()
        style(self, "status-bar")

        self.label = Gtk.Label(label="AI Assistant Ready")
        self.add(self.label)
        self.hide_timer = None

    def show_message(self, text, timeout=0, level=None):
        """level: None | 'ok' | 'warn' | 'err' -- tints the border."""
        GLib.idle_add(self._update_ui, text, timeout, level)

    def _update_ui(self, text, timeout, level=None):
        if self.hide_timer:
            GLib.source_remove(self.hide_timer)
            self.hide_timer = None

        ctx = self.get_style_context()
        for name in ("level-ok", "level-warn", "level-err"):
            ctx.remove_class(name)
        if level in ("ok", "warn", "err"):
            ctx.add_class(f"level-{level}")

        self.label.set_text(text)
        self.show_all()
        if timeout > 0:
            self.hide_timer = GLib.timeout_add_seconds(timeout, self._hide_bar)
        return False

    def _hide_bar(self):
        self.hide()
        self.hide_timer = None
        return False


class AssistantDaemon:
    def __init__(self, config):
        self.config = config
        self.sock_path = self.config["socket_path"]
        self.history_dir = self.config["session_history_dir"]
        os.makedirs(self.history_dir, exist_ok=True)
        self.bar = BottomStatusBar()
        self.session = None
        self.kokoro_engine = None

        # --- AI Mode state (master switch for every AI subsystem) ---
        self.ai_enabled = False
        self.ai_phase = "off"        # off | starting | on | error
        self.ai_error = ""
        self.ai_external = []        # ports already served by someone else's process
        self._health = {}            # name -> bool, refreshed on a short TTL
        self._health_checked = 0.0
        self.servers = {}            # name -> Popen we own
        self.ai_lock = threading.RLock()
        self.state_file = self.config["ai_mode_state_file"]
        self.log_dir = os.path.join(os.path.dirname(self.state_file), "logs")

        # One shared sqlite handle guarded by a lock: worker threads and the
        # per-connection socket threads must not each open their own.
        self.mem = memory_db.open_db(self.config["memory_db_path"], check_same_thread=False)
        self.mem_lock = threading.Lock()
        self.mem_count = 0
        self._refresh_mem_count()

        # FIFO Processing Queue for screenshots
        self.analysis_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
        self.worker_thread.start()

        # Dedicated FIFO Queue for Small LLM + Kokoro TTS announcements
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

        # Background distillation of chats/notes into long-term memory
        self.memory_queue = queue.Queue()
        self.memory_thread = threading.Thread(target=self._memory_worker, daemon=True)
        self.memory_thread.start()

        self._reap_previous_servers()

    def _ensure_kokoro_loaded(self):
        """Auto-downloads Kokoro ONNX files on first use and initializes the local CPU TTS engine."""
        if self.kokoro_engine is not None:
            return self.kokoro_engine

        tts_dir = self.config.get("tts_model_dir", os.path.expanduser("~/.local/share/ai_assistant/tts"))
        os.makedirs(tts_dir, exist_ok=True)

        model_path = os.path.join(tts_dir, "kokoro-v1.0.int8.onnx")
        voices_path = os.path.join(tts_dir, "voices-v1.0.bin")

        base_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
        
        # Safe download helper using requests instead of urllib
        def download_file(url, dest_path):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        if not os.path.exists(model_path):
            self.bar.show_message("Downloading Kokoro TTS model (~88MB)...", timeout=0)
            download_file(f"{base_url}/kokoro-v1.0.int8.onnx", model_path)

        if not os.path.exists(voices_path):
            self.bar.show_message("Downloading Kokoro voice profiles...", timeout=0)
            download_file(f"{base_url}/voices-v1.0.bin", voices_path)

        self.kokoro_engine = Kokoro(model_path, voices_path)
        return self.kokoro_engine

    def _queue_worker(self):
        """Worker loop that pulls screenshots from the queue sequentially."""
        while True:
            image_path = self.analysis_queue.get()
            try:
                if self.session and self.require_ai():
                    self.process_image(image_path)
            except Exception as e:
                self.bar.show_message(f"Worker Error: {str(e)}", timeout=5)
                self.announce_event(f"Encountered a background worker error: {str(e)}")
            finally:
                self.analysis_queue.task_done()

    def _speech_worker(self):
        """Worker loop that processes operational status updates into speech sequentially."""
        while True:
            raw_event = self.speech_queue.get()
            try:
                spoken_script = self._formulate_status_sentence(raw_event)
                self._play_kokoro_tts(spoken_script)
            except Exception as e:
                print(f"Narrator pipeline error: {e}", file=sys.stderr)
            finally:
                self.speech_queue.task_done()

    def announce_event(self, event_description):
        """Queues a raw operational event to be converted into spoken sentences.

        Silently dropped while AI Mode is off -- the narrator needs the small
        model on :9191 and the Kokoro engine, both of which are torn down.
        """
        if not self.require_ai():
            return
        self.speech_queue.put(event_description)

    def _extract_note_title(self, md_text):
        """Extracts the primary heading or topic title from the AI's Markdown response."""
        for line in md_text.splitlines():
            clean = line.strip()
            if not clean or clean.startswith("```"):
                continue
            if clean.startswith("#"):
                title = clean.lstrip("#").strip()
                title = re.sub(r'(\*\*|\*|`)', '', title)
                if title:
                    return title[:65]
            cleaned_line = re.sub(r'(\*\*|\*|`)', '', clean)
            if len(cleaned_line) > 3:
                return cleaned_line[:55]
        return "the captured screen region"

    def _formulate_status_sentence(self, raw_event):
        """Uses the small LLM on port 9191 to turn raw telemetry into natural spoken sentences."""
        system_prompt = (
            "You are a calm, articulate, and slightly distinguished desktop AI voice assistant "
            "with a cadence similar to Morgan Freeman. Convert the following raw system status update "
            "into 1 or 2 natural, clear spoken sentences telling the user exactly what action you just completed "
            "and any next steps required (such as needing to resync because of a document lock). "
            "Keep all titles, numbers, and statuses accurate. Do NOT use markdown, asterisks, quotes, or emojis."
        )
        small_url = self.config.get("llama_small_url", "http://127.0.0.1:9191/v1/chat/completions")
        small_model = self.config.get("llama_small_model", "")

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Status update to announce: {raw_event}"}
            ],
            "temperature": 0.4,
            "max_tokens": 100
        }
        if small_model:
            payload["model"] = small_model

        try:
            resp = requests.post(small_url, json=payload, timeout=15)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip <think>...</think> blocks if using a Qwen3 thinking model
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            text = re.sub(r'[*`"#]', '', text).strip(' "\'')
            return text if text else raw_event
        except Exception as e:
            print(f"Small LLM ({small_url}) fallback ({e}) -> speaking raw event directly.", file=sys.stderr)
            return raw_event

    def _resolve_voice_path(self, voice_cfg):
        """Returns an absolute path to a custom .npy voice vector, or None for built-in voices.

        A value ending in .npy is always treated as a file reference: if it cannot be
        found we raise instead of silently falling through to the built-in voice list
        (which produced the misleading "Voice ./x.npy not found in available voices").
        """
        tts_dir = self.config.get("tts_model_dir", os.path.expanduser("~/.local/share/ai_assistant/tts"))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw = os.path.expanduser(voice_cfg.strip())
        looks_like_file = raw.endswith(".npy")

        if os.path.isabs(raw):
            candidates = [raw]
        else:
            stem = raw[:-4] if looks_like_file else raw
            name = f"{stem}.npy"
            candidates = [
                os.path.join(tts_dir, os.path.basename(name)),
                os.path.join(script_dir, name),
                os.path.join(script_dir, "..", name),
                os.path.abspath(name),
            ]

        for c in candidates:
            c = os.path.abspath(c)
            if os.path.isfile(c):
                return c

        if looks_like_file:
            raise FileNotFoundError(
                f"Custom voice {voice_cfg!r} not found. Searched: "
                + ", ".join(os.path.abspath(c) for c in candidates)
                + f". Place the .npy in {tts_dir} and set tts_voice to its name."
            )
        return None  # treat as a built-in Kokoro voice name

    def _play_kokoro_tts(self, text):
        """Synthesizes speech via external tts_url (port 9292) or built-in local Kokoro ONNX engine."""
        import numpy as np  # Required to load custom voice vectors
        
        voice_cfg = self.config.get("tts_voice", "am_michael")
        speed = float(self.config.get("tts_speed", "0.95"))
        tts_url = self.config.get("tts_url", "http://127.0.0.1:9292/v1/audio/speech")

        # Resolve voice: either a built-in Kokoro voice name or a custom .npy style vector
        voice_path = self._resolve_voice_path(voice_cfg)
        is_custom_file = voice_path is not None
        voice_param = np.load(voice_path) if is_custom_file else voice_cfg

        # --- PITCH & TONE ADJUSTMENT FOR CUSTOM VECTORS ---
        if is_custom_file:
            if voice_param.ndim == 3:
                voice_param = voice_param.squeeze(1)
            
            # Subtly shift/normalize the vector weights to deepen the pitch 
            # (multiplying by a factor or dampening high-frequency style indices)
            voice_param = voice_param * 0.9

        # 1. Try external Kokoro server (only works with built-in string voices)
        if not is_custom_file:
            try:
                r = requests.post(
                    tts_url,
                    json={"model": "kokoro", "input": text, "voice": voice_param, "speed": speed, "response_format": "wav"},
                    stream=True,
                    timeout=2
                )
                if r.status_code == 200:
                    p = subprocess.Popen(["pw-play", "-"], stdin=subprocess.PIPE)
                    for chunk in r.iter_content(chunk_size=4096):
                        if chunk:
                            p.stdin.write(chunk)
                    p.stdin.close()
                    p.wait()
                    return
            except requests.exceptions.RequestException:
                pass  # Fall back to built-in kokoro-onnx engine

        # 2. Built-in Local CPU Kokoro ONNX Engine
        try:
            kokoro = self._ensure_kokoro_loaded()
            
            # Custom cloned vectors must match the built-in layout: [510, 1, 256].
            # kokoro-onnx indexes voice[len(tokens)] and feeds the result straight to
            # the ONNX "style" input, which expects shape [1, 256] -- so do NOT squeeze
            # the middle axis here.
            if is_custom_file:
                voice_param = np.asarray(voice_param, dtype=np.float32)
                if voice_param.ndim == 2 and voice_param.shape[-1] == 256:
                    voice_param = voice_param[:, None, :]  # [510, 256] -> [510, 1, 256]
                if voice_param.ndim != 3 or voice_param.shape[1:] != (1, 256):
                    raise ValueError(
                        f"Custom voice {voice_cfg} has unusable shape {voice_param.shape}; "
                        "expected [510, 1, 256]"
                    )

            samples, sample_rate = kokoro.create(text, voice=voice_param, speed=speed, lang="en-us")
            wav_buf = io.BytesIO()
            sf.write(wav_buf, samples, sample_rate, format="WAV")
            wav_buf.seek(0)

            p = subprocess.Popen(["pw-play", "-"], stdin=subprocess.PIPE)
            p.communicate(input=wav_buf.read())
        except Exception as e:
            print(f"Kokoro TTS playback failed: {e}", file=sys.stderr)

    # ======================================================================
    # AI MODE -- master switch for every AI subsystem
    # ======================================================================
    @staticmethod
    def _drain(q):
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except queue.Empty:
            pass

    def _ping_waybar(self):
        """Nudge waybar so the module repaints immediately instead of on its poll."""
        sig = self.config.get("waybar_signal", "8")
        try:
            subprocess.run(["pkill", f"-RTMIN+{sig}", "waybar"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass

    def _write_ai_state(self):
        payload = {
            "enabled": self.ai_enabled,
            "phase": self.ai_phase,
            "error": self.ai_error,
            "external": self.ai_external,
            "pids": {n: p.pid for n, p in self.servers.items() if p and p.poll() is None},
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self.state_file)
        except OSError as e:
            print(f"Could not write AI mode state: {e}", file=sys.stderr)
        self._ping_waybar()

    def _kill_pid(self, pid, label="server", grace=8.0):
        """SIGTERM the process group, escalating to SIGKILL after `grace`."""
        if not pid:
            return
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return

        deadline = time.time() + grace
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError, OSError):
                return
            time.sleep(0.2)
        try:
            os.killpg(pgid, signal.SIGKILL)
            print(f"[AI Mode] {label} ignored SIGTERM; killed.", file=sys.stderr)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _reap_previous_servers(self):
        """Clear servers left behind by a crashed/restarted daemon.

        We cannot adopt a Popen we did not spawn, and leaving a 35B model
        resident while the bar reads "off" defeats the whole switch.
        """
        try:
            with open(self.state_file) as f:
                prev = json.load(f)
        except (OSError, ValueError):
            prev = {}
        for name, pid in (prev.get("pids") or {}).items():
            self._kill_pid(pid, f"orphaned {name}")
        self.ai_enabled, self.ai_phase, self.ai_error = False, "off", ""
        self._write_ai_state()

    def _health_ok(self, url, timeout=1.5):
        try:
            return requests.get(url, timeout=timeout).status_code == 200
        except requests.exceptions.RequestException:
            return False

    SERVER_SPECS = (
        ("big", "llama_server_cmd", "llama_health_url"),
        ("small", "llama_small_server_cmd", "llama_small_health_url"),
    )

    def ai_mode_on(self):
        with self.ai_lock:
            if self.ai_phase in ("starting", "on"):
                return self.ai_mode_status()
            missing = [k for _, k, _ in self.SERVER_SPECS if not config_argv(self.config, k)]
            if missing:
                self.ai_enabled = False
                self.ai_phase = "error"
                self.ai_error = "Not configured: " + ", ".join(missing)
                self._write_ai_state()
                self.bar.show_message(f"AI Mode: {self.ai_error}", timeout=6, level="err")
                return self.ai_mode_status()

            self.ai_phase = "starting"
            self.ai_error = ""
            self.ai_external = []
            self._write_ai_state()

        self.bar.show_message("AI Mode: starting llama.cpp servers...", timeout=0, level="warn")
        threading.Thread(target=self._start_servers, daemon=True).start()
        return self.ai_mode_status()

    def _start_servers(self):
        os.makedirs(self.log_dir, exist_ok=True)
        launched = []

        for name, cmd_key, health_key in self.SERVER_SPECS:
            health_url = self.config[health_key]
            if self._health_ok(health_url):
                # Someone already serves this port; adopt it rather than fight
                # for the bind, but remember we do not own it.
                self.ai_external.append(name)
                continue
            argv = config_argv(self.config, cmd_key)
            try:
                logfile = open(os.path.join(self.log_dir, f"llama-{name}.log"), "ab", buffering=0)
                proc = subprocess.Popen(
                    argv, stdout=logfile, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
            except (OSError, ValueError) as e:
                with self.ai_lock:
                    self.ai_phase, self.ai_error = "error", f"{name}: {e}"
                    self._write_ai_state()
                self.bar.show_message(f"AI Mode failed ({name}): {e}", timeout=8, level="err")
                return
            with self.ai_lock:
                self.servers[name] = proc
            launched.append((name, health_key, proc))

        self._write_ai_state()

        timeout = config_num(self.config, "ai_mode_start_timeout", 180, float)
        deadline = time.time() + timeout
        pending = list(launched)
        while pending and time.time() < deadline:
            still = []
            for name, health_key, proc in pending:
                if proc.poll() is not None:
                    with self.ai_lock:
                        self.ai_phase = "error"
                        self.ai_error = (f"{name} exited with code {proc.returncode} "
                                         f"(see {self.log_dir}/llama-{name}.log)")
                        self._write_ai_state()
                    self.bar.show_message(f"AI Mode: {self.ai_error}", timeout=10, level="err")
                    return
                if not self._health_ok(self.config[health_key]):
                    still.append((name, health_key, proc))
            pending = still
            if pending:
                remaining = int(deadline - time.time())
                self.bar.show_message(
                    f"AI Mode: loading {', '.join(n for n, _, _ in pending)} ({remaining}s left)...",
                    timeout=0, level="warn",
                )
                time.sleep(2.0)

        if pending:
            with self.ai_lock:
                self.ai_phase = "error"
                self.ai_error = "Timed out waiting for: " + ", ".join(n for n, _, _ in pending)
                self._write_ai_state()
            self.bar.show_message(f"AI Mode: {self.ai_error}", timeout=10, level="err")
            return

        with self.ai_lock:
            self.ai_enabled, self.ai_phase, self.ai_error = True, "on", ""
            self._write_ai_state()

        note = ""
        if self.ai_external:
            note = " (" + ", ".join(self.ai_external) + " was already running)"
        self.bar.show_message(f"AI Mode ON{note}", timeout=4, level="ok")
        self.announce_event("AI mode is now active. Both local language model servers are up.")

    def ai_mode_off(self):
        with self.ai_lock:
            was_on = self.ai_phase != "off"
            procs = list(self.servers.items())
            self.servers.clear()

            # Release every in-process AI resource up front -- freeing RAM/VRAM
            # is the point of the switch (e.g. before launching a game).
            self.kokoro_engine = None
            self._drain(self.analysis_queue)
            self._drain(self.speech_queue)
            self._drain(self.memory_queue)
            gc.collect()

            self.ai_enabled, self.ai_phase, self.ai_error = False, "off", ""
            stale = list(self.ai_external)
            self.ai_external = []
            self._write_ai_state()

        GLib.idle_add(self.bar.hide)
        if was_on and stale:
            print(f"[AI Mode] Left externally-started server(s) running: {', '.join(stale)}",
                  file=sys.stderr)

        # A 35B model can take several seconds to exit; reap off-thread so the
        # waybar click gets an immediate reply instead of a timeout.
        if procs:
            threading.Thread(target=self._reap_servers, args=(procs,), daemon=True).start()
        return self.ai_mode_status()

    def _reap_servers(self, procs):
        for name, proc in procs:
            if proc and proc.poll() is None:
                self._kill_pid(proc.pid, name)
        self._write_ai_state()

    def _refresh_health(self, max_age=4.0):
        """Live /health for both servers, cached so the waybar poll stays cheap."""
        now = time.time()
        if now - self._health_checked < max_age and self._health:
            return self._health
        self._health = {
            "big": self._health_ok(self.config["llama_health_url"], timeout=1.0),
            "small": self._health_ok(self.config["llama_small_health_url"], timeout=1.0),
        }
        self._health_checked = now
        return self._health

    def ai_mode_status(self):
        health = self._refresh_health() if self.ai_phase == "on" else {}
        with self.ai_lock:
            phase, error = self.ai_phase, self.ai_error
            # A server can die after a successful start -- an adopted one we do
            # not supervise, or a crash. Report that instead of claiming "on",
            # which previously let 9191-backed features fail silently.
            down = [name for name, ok in health.items() if not ok]
            if phase == "on" and down:
                phase = "error"
                error = "not responding: " + ", ".join(down)
            return {
                "enabled": self.ai_enabled and not down,
                "phase": phase,
                "error": error,
                "external": list(self.ai_external),
                "health": health,
                "servers": {n: (p.poll() is None) for n, p in self.servers.items()},
                "memories": self.mem_count,
                "queued_jobs": self.analysis_queue.qsize(),
                "session": self.session is not None,
            }

    def require_ai(self):
        """True when model-backed work may proceed."""
        return self.ai_enabled and self.ai_phase == "on"

    # ======================================================================
    # MEMORY
    # ======================================================================
    def _refresh_mem_count(self):
        try:
            with self.mem_lock:
                row = self.mem.execute(
                    "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL").fetchone()
            self.mem_count = row[0] if row else 0
        except Exception:
            self.mem_count = 0

    def remember(self, content, kind="fact", importance=0.5, tags=None,
                 conversation_id=None, source="chat"):
        try:
            with self.mem_lock:
                memory_db.add_memory(
                    self.mem, content, kind=kind, importance=importance, tags=tags,
                    conversation_id=conversation_id, source=source,
                )
            self.mem_count += 1
        except Exception as e:
            print(f"Memory write failed: {e}", file=sys.stderr)

    def _memory_worker(self):
        """Distills finished exchanges into tagged, scored long-term memories."""
        while True:
            job = self.memory_queue.get()
            try:
                if self.require_ai():
                    self._distill(job)
            except Exception as e:
                print(f"Memory distillation error: {e}", file=sys.stderr)
            finally:
                self.memory_queue.task_done()

    DISTILL_SYSTEM = (
        "You extract durable long-term memories from a conversation between a user and their "
        "local assistant. Reply with ONE JSON object and nothing else:\n"
        '{"summary": "<2-3 sentence recap>", "title": "<short thread title>", '
        '"tags": ["topic", ...], "importance": 0.0-1.0, '
        '"memories": [{"kind": "fact|preference|project|event", "content": "<one self-contained '
        'sentence>", "importance": 0.0-1.0, "tags": ["topic", ...]}]}\n'
        "Record only things worth recalling weeks later: decisions, preferences, configuration, "
        "names, ports, paths, ongoing projects, commitments. Each memory must stand alone without "
        "the conversation. Skip pleasantries and anything transient. Prefer 0 memories over "
        "speculation. Use at most 6."
    )

    def _llm_json(self, system_prompt, user_content, max_tokens=2048):
        resp = requests.post(
            self.config["llama_url"],
            json={
                "model": self.config.get("llama_model", ""),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=180,
        )
        resp.raise_for_status()
        return extract_json(resp.json()["choices"][0]["message"]["content"])

    def _distill(self, job):
        conversation_id = job.get("conversation_id")
        if not conversation_id:
            return

        with self.mem_lock:
            convo = memory_db.get_conversation(self.mem, conversation_id)
            turns = memory_db.get_messages(self.mem, conversation_id, limit=16)
        if not convo or not turns:
            return

        transcript = "\n\n".join(
            f"{t['role'].upper()}: {t['content'][:4000]}" for t in turns if t["role"] != "system"
        )
        if not transcript.strip():
            return

        data = self._llm_json(self.DISTILL_SYSTEM, transcript)
        if not isinstance(data, dict):
            # Never lose the exchange: keep a raw summary row instead.
            self.remember(
                transcript[:1500], kind="summary", importance=0.3,
                tags=["unparsed"], conversation_id=conversation_id,
            )
            self._refresh_mem_count()
            return

        base_tags = [t for t in (data.get("tags") or []) if isinstance(t, str)]
        items = []
        for raw in (data.get("memories") or [])[:6]:
            if isinstance(raw, str):
                raw = {"content": raw}
            if not isinstance(raw, dict) or not str(raw.get("content", "")).strip():
                continue
            tags = [t for t in (raw.get("tags") or []) if isinstance(t, str)] or base_tags
            items.append({
                "content": str(raw["content"]).strip(),
                "kind": raw.get("kind", "fact"),
                "importance": raw.get("importance", data.get("importance", 0.5)),
                "tags": tags,
            })

        summary = (data.get("summary") or "").strip()
        with self.mem_lock:
            if items:
                memory_db.add_memories_bulk(self.mem, items, conversation_id, source="chat")
            if summary:
                memory_db.set_summary(self.mem, conversation_id, summary)
            title = (data.get("title") or "").strip()
            if title and not convo.get("title"):
                memory_db.rename_conversation(self.mem, conversation_id, title)
        self._refresh_mem_count()

    def save_session_history(self):
        if not self.session:
            return
        hist_file = os.path.join(self.history_dir, f"{self.session['id']}.json")
        with open(hist_file, "w") as f:
            json.dump(self.session, f, indent=2)

    def get_unsynced_entries(self):
        if not self.session:
            return []
        return [e for e in self.session["entries"] if not e.get("synced", False)]

    def handle_client(self, conn):
        try:
            data = conn.recv(65536).decode("utf-8")
            if not data:
                return
            req = json.loads(data)
            cmd = req.get("command")

            if cmd == "status":
                unsynced = len(self.get_unsynced_entries())
                pending_jobs = self.analysis_queue.qsize()
                conn.sendall(json.dumps({
                    "active": self.session is not None,
                    "session": self.session,
                    "unsynced_count": unsynced,
                    "queued_jobs": pending_jobs,
                    "ai": self.ai_mode_status(),
                }).encode("utf-8"))

            elif cmd in ("ai_mode_on", "ai_mode_off", "ai_mode_toggle", "ai_mode_status"):
                if cmd == "ai_mode_on":
                    result = self.ai_mode_on()
                elif cmd == "ai_mode_off":
                    result = self.ai_mode_off()
                elif cmd == "ai_mode_toggle":
                    result = self.ai_mode_off() if self.ai_phase != "off" else self.ai_mode_on()
                else:
                    result = self.ai_mode_status()
                conn.sendall(json.dumps(result).encode("utf-8"))

            elif cmd == "analyze_memory":
                # Fire-and-forget from the chat popup once an exchange completes.
                conversation_id = req.get("conversation_id")
                if conversation_id and self.require_ai():
                    self.memory_queue.put({"conversation_id": conversation_id})
                conn.sendall(json.dumps({"status": "queued"}).encode("utf-8"))

            elif cmd == "remember":
                self.remember(
                    req.get("content", ""),
                    kind=req.get("kind", "fact"),
                    importance=req.get("importance", 0.5),
                    tags=req.get("tags"),
                    conversation_id=req.get("conversation_id"),
                    source=req.get("source", "manual"),
                )
                conn.sendall(json.dumps({"status": "ok"}).encode("utf-8"))

            elif cmd == "memory_stats":
                with self.mem_lock:
                    payload = memory_db.stats(self.mem)
                conn.sendall(json.dumps(payload).encode("utf-8"))

            elif cmd == "memory_compact":
                with self.mem_lock:
                    payload = memory_db.compact(self.mem)
                self._refresh_mem_count()
                self.bar.show_message(
                    f"Memory compacted: {payload['merged']} merged, "
                    f"{payload['remaining']} kept", timeout=4, level="ok")
                conn.sendall(json.dumps(payload).encode("utf-8"))

            elif cmd == "new_session":
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.session = {
                    "id": session_id,
                    "doc_path": req["doc_path"],
                    "prompt": req["prompt"],
                    "created_at": datetime.now().isoformat(),
                    "entries": []
                }
                self.save_session_history()
                doc_name = os.path.basename(req["doc_path"])
                self.bar.show_message(f"Session Active: {req['doc_path']}", timeout=4)
                self.announce_event(f"Started a new note-taking session targeting the document {doc_name}.")
                conn.sendall(json.dumps({"status": "ok", "session": self.session}).encode("utf-8"))

            elif cmd == "update_session":
                if self.session:
                    self.session["doc_path"] = req["doc_path"]
                    self.session["prompt"] = req["prompt"]
                    self.save_session_history()
                    doc_name = os.path.basename(req["doc_path"])
                    self.bar.show_message(f"Updated Session: {req['doc_path']}", timeout=3)
                    self.announce_event(f"Updated the session configuration for {doc_name}.")
                conn.sendall(json.dumps({"status": "ok"}).encode("utf-8"))

            elif cmd == "end_session":
                unsynced = len(self.get_unsynced_entries())
                self.session = None
                self.bar.show_message("Session Ended", timeout=3)
                if unsynced > 0:
                    self.announce_event(f"Ended the session. Note that {unsynced} items were still unsynced.")
                else:
                    self.announce_event("Ended the current session. All notes are safely synced.")
                conn.sendall(json.dumps({"status": "ok"}).encode("utf-8"))

            elif cmd == "sync_queue":
                if not self.session:
                    conn.sendall(json.dumps({"status": "error", "message": "No active session"}).encode("utf-8"))
                else:
                    threading.Thread(target=self.flush_sync_queue, daemon=True).start()
                    conn.sendall(json.dumps({"status": "syncing"}).encode("utf-8"))

            elif cmd == "analyze":
                if not self.require_ai():
                    conn.sendall(json.dumps({"error": "ai_off"}).encode("utf-8"))
                elif not self.session:
                    conn.sendall(json.dumps({"error": "no_session"}).encode("utf-8"))
                else:
                    self.analysis_queue.put(req["image_path"])
                    pending = self.analysis_queue.qsize()
                    if pending > 1:
                        self.bar.show_message(f"Screenshot queued ({pending} in queue)...", timeout=3)
                    conn.sendall(json.dumps({"status": "queued", "position": pending}).encode("utf-8"))

            elif cmd == "announce":
                msg = req.get("message", "")
                if msg:
                    self.announce_event(msg)
                conn.sendall(json.dumps({"status": "ok"}).encode("utf-8"))

        except Exception as e:
            print(f"Socket error: {e}", file=sys.stderr)
        finally:
            conn.close()

    def flush_sync_queue(self):
        unsynced = self.get_unsynced_entries()
        if not unsynced:
            self.bar.show_message("Sync queue is already empty!", timeout=3)
            self.announce_event("The sync queue is already empty. There are no pending notes to push.")
            return

        count = len(unsynced)
        self.bar.show_message(f"Pushing {count} queued note(s) to Nextcloud...", timeout=0)
        try:
            self.append_to_odt(self.session["doc_path"], unsynced)
            for e in unsynced:
                e["synced"] = True
            self.save_session_history()
            self.bar.show_message(f"Synced {count} note(s) to ODT!", timeout=4)
            self.announce_event(
                f"Successfully synced {count} pending {'note' if count == 1 else 'notes'} to the Nextcloud document."
            )
        except Exception as e:
            self.bar.show_message(f"Sync failed: {str(e)}", timeout=5)
            if "Locked" in str(e) or "423" in str(e):
                self.announce_event(
                    f"Could not sync the {count} queued notes because the document is still open and locked in Nextcloud."
                )
            else:
                self.announce_event(f"Failed to sync the queue due to an error: {str(e)}")

    def process_image(self, image_path):
        remaining = self.analysis_queue.qsize()
        status_msg = "Analyzing screenshot with 35B model..."
        if remaining > 0:
            status_msg += f" ({remaining} waiting in queue)"
        self.bar.show_message(status_msg, timeout=0)

        try:
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            messages = [{"role": "system", "content": self.session["prompt"]}]
            for entry in self.session["entries"][-5:]:
                messages.append({"role": "assistant", "content": entry["response"]})

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Instruction: {self.session['prompt']}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]
            })

            resp = requests.post(
                self.config["llama_url"],
                json={
                    "model": self.config.get("llama_model", ""),
                    "messages": messages,
                    "temperature": 0.3
                },
                timeout=180
            )
            resp.raise_for_status()
            output_text = resp.json()["choices"][0]["message"]["content"]

            note_title = self._extract_note_title(output_text)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = {
                "timestamp": timestamp,
                "image_path": image_path,
                "title": note_title,
                "response": output_text,
                "synced": False
            }
            self.session["entries"].append(new_entry)
            self.save_session_history()

            # Mirror the note into long-term memory so chat can recall captures.
            self.remember(
                f"Screen capture note -- {note_title}: {output_text[:1200]}",
                kind="event", importance=0.45,
                tags=["screenshot", "notes"],
                source="screenshot",
            )

            self.bar.show_message(f"Writing '{note_title}' to {self.session['doc_path']}...", timeout=0)
            
            unsynced = self.get_unsynced_entries()
            remaining_after = self.analysis_queue.qsize()

            try:
                self.append_to_odt(self.session["doc_path"], unsynced)
                for e in unsynced:
                    e["synced"] = True
                self.save_session_history()
                
                if remaining_after > 0:
                    self.bar.show_message(f"Saved '{note_title}'! {remaining_after} more in queue...", timeout=2)
                    self.announce_event(
                        f"Analyzed screenshot of {note_title} and saved it to the notes document. "
                        f"Moving on to the next screenshot, with {remaining_after} remaining in the queue."
                    )
                else:
                    self.bar.show_message("Done! All notes saved to ODT.", timeout=4)
                    self.announce_event(
                        f"Analyzed screenshot of {note_title}. Finished analyzing all screenshots in the queue, "
                        f"and your notes document is completely up to date."
                    )
            except Exception as e:
                self.bar.show_message(f"Queue saved locally: {str(e)}", timeout=5)
                locked_reason = (
                    "there is a lock on the notes document"
                    if "Locked" in str(e) or "423" in str(e)
                    else f"of a sync error: {str(e)}"
                )
                if remaining_after > 0:
                    self.announce_event(
                        f"Analyzed screenshot of {note_title}, and queued it locally because {locked_reason}. "
                        f"{remaining_after} {'screenshot remains' if remaining_after == 1 else 'screenshots remain'} in the analysis queue."
                    )
                else:
                    self.announce_event(
                        f"Analyzed screenshot of {note_title}. Finished analyzing all screenshots, "
                        f"you just have to resync when ready because {locked_reason}."
                    )

        except Exception as e:
            self.bar.show_message(f"Error: {str(e)}", timeout=6)
            self.announce_event(f"Failed to analyze the screenshot due to an error: {str(e)}")

    def append_to_odt(self, doc_target, entries):
        nc_url = self.config["nextcloud_url"].rstrip("/")
        nc_user = self.config["nextcloud_user"]
        nc_token = self.config["nextcloud_token"]

        is_remote = not (
            doc_target.startswith("/") or doc_target.startswith("~") or doc_target.startswith("file://")
        )
        if not is_remote:
            local_odt = os.path.expanduser(doc_target.replace("file://", ""))
            doc = load(local_odt) if os.path.exists(local_odt) else OpenDocumentText()
        else:
            local_odt = tempfile.NamedTemporaryFile(suffix=".odt", delete=False).name
            clean_target = doc_target.lstrip("/")
            webdav_url = f"{nc_url}/remote.php/dav/files/{nc_user}/{clean_target}"

            r = requests.get(webdav_url, auth=(nc_user, nc_token))
            if r.status_code == 200:
                with open(local_odt, "wb") as f:
                    f.write(r.content)
                doc = load(local_odt)
            elif r.status_code == 404:
                doc = OpenDocumentText()
            else:
                r.raise_for_status()

        styles = setup_odt_styles(doc)

        for entry in entries:
            doc.text.addElement(H(outlinelevel=2, text=f"Session Note ({entry['timestamp']})"))
            append_markdown_content(doc, entry["response"], styles)

        doc.save(local_odt)

        if is_remote:
            with open(local_odt, "rb") as f:
                put_resp = requests.put(webdav_url, data=f, auth=(nc_user, nc_token))
                if put_resp.status_code == 423:
                    os.remove(local_odt)
                    raise Exception("Document is open in Nextcloud (Locked).")
                put_resp.raise_for_status()
            os.remove(local_odt)

    def start_server(self):
        if os.path.exists(self.sock_path):
            os.remove(self.sock_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.sock_path)
        server.listen(5)
        while True:
            conn, _ = server.accept()
            threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()

# ==============================================================================
# 6. CLIENT: POPUPS, CHAT & MANUAL SYNC
# ==============================================================================
def notify(message, title="AI Assistant"):
    """Desktop notification, degrading to stderr where notify-send is absent."""
    if shutil.which("notify-send"):
        try:
            subprocess.run(["notify-send", title, message],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    print(f"[{title}] {message}", file=sys.stderr)


def send_cmd(sock_path, payload, quiet=False, timeout=15.0):
    """Round-trip one command to the daemon.

    quiet=True returns None instead of exiting, for callers that must never
    fail loudly (the waybar module polls this every few seconds).
    """
    if not os.path.exists(sock_path):
        if quiet:
            return None
        notify("Daemon not running! Start it or check dinitctl.")
        sys.exit(1)

    client = None
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect(sock_path)
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
    except (OSError, ValueError) as e:
        if quiet:
            return None
        notify(f"Daemon error: {e}")
        sys.exit(1)
    finally:
        if client is not None:
            client.close()


def clamp_title(raw, max_words=7, max_chars=52):
    """Tidy a model-written title.

    The 0.5B model regularly ignores the word limit and answers with a whole
    sentence, so keep the first line, cap the word count, and trim on a word
    boundary rather than slicing mid-word.
    """
    text = (raw or "").strip().splitlines()
    text = text[0] if text else ""
    text = re.sub(r"^\s*(?:title\s*[:\-]\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*`#\"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,;:-'\"")

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text.strip(" .,;:-")


def fmt_count(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def format_stats(stats, live=False):
    """One-line token usage / speed summary for an assistant turn."""
    completion = stats.get("done_completion", 0) + stats.get("cur_completion", 0)
    elapsed_ms = stats.get("done_ms", 0.0) + stats.get("cur_ms", 0.0)
    parts = []

    if live:
        rate = stats.get("cur_tps") or 0.0
        parts.append(f"{fmt_count(completion)} tok")
        if rate:
            parts.append(f"{rate:.1f} tok/s")
        return "  ·  ".join(parts) if parts else "…"

    parts.append(f"{fmt_count(completion)} tok")
    if elapsed_ms > 0:
        parts.append(f"{completion / (elapsed_ms / 1000.0):.1f} tok/s")
        parts.append(f"{elapsed_ms / 1000.0:.1f}s")

    prompt = stats.get("prompt", 0)
    if prompt:
        prompt_bit = f"prompt {fmt_count(prompt)} tok"
        if stats.get("prompt_tps"):
            prompt_bit += f" @ {stats['prompt_tps']:.0f} tok/s"
        if stats.get("cached"):
            prompt_bit += f", {fmt_count(stats['cached'])} cached"
        parts.append(prompt_bit)

    total = stats.get("total", 0)
    if total:
        parts.append(f"{fmt_count(total)} total")
    return "  ·  ".join(parts)


def strip_think(text):
    """Remove <think> blocks, including one still being streamed."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


class SessionPopup(Gtk.Window):
    def __init__(self, sock_path, current_session=None, unsynced_count=0):
        super().__init__(title="AI Assistant Session")
        install_theme()
        self.sock_path = sock_path
        self.set_wmclass("ai-assistant-popup", "ai-assistant-popup")
        self.set_default_size(540, 390)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_border_width(16)
        self.started_ok = False

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(vbox)

        if current_session:
            queue_str = f" | {unsynced_count} pending in queue" if unsynced_count > 0 else " | Queue empty"
            status_text = f"Active Session: {current_session['id']} ({len(current_session['entries'])} total{queue_str})"
        else:
            status_text = "No Active Session — Create One Below"
            
        header = style(Gtk.Label(label=status_text, xalign=0), "heading")
        vbox.pack_start(header, False, False, 0)

        vbox.pack_start(
            style(Gtk.Label(label="Nextcloud ODT Path (e.g. Documents/notes.odt) or Local Path:",
                            xalign=0), "dim"),
            False, False, 0
        )
        self.doc_entry = Gtk.Entry()
        self.doc_entry.set_text(current_session["doc_path"] if current_session else "Notes/session_notes.odt")
        vbox.pack_start(self.doc_entry, False, False, 0)

        vbox.pack_start(
            style(Gtk.Label(label="Analysis / Generation Prompt:", xalign=0), "dim"),
            False, False, 0)
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        self.prompt_view = Gtk.TextView()
        self.prompt_view.set_wrap_mode(Gtk.WrapMode.WORD)
        buf = self.prompt_view.get_buffer()
        default_prompt = (
            current_session["prompt"] if current_session
            else "Analyze this screenshot, summarize key findings, and format cleanly as study/project notes."
        )
        buf.set_text(default_prompt)
        scrolled.add(self.prompt_view)
        vbox.pack_start(scrolled, True, True, 0)

        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        vbox.pack_start(btn_box, False, False, 0)

        start_btn = style(
            Gtk.Button(label="Start New Session" if not current_session else "Update Session"),
            "accent-btn")
        start_btn.connect("clicked", self.on_save_session, bool(current_session))
        btn_box.pack_start(start_btn, True, True, 0)

        if current_session:
            sync_btn = Gtk.Button(label=f"Sync Queue ({unsynced_count})")
            sync_btn.connect("clicked", self.on_sync_queue)
            if unsynced_count == 0:
                sync_btn.set_sensitive(False)
            btn_box.pack_start(sync_btn, False, False, 0)

            end_btn = style(Gtk.Button(label="End Session"), "danger-btn")
            end_btn.connect("clicked", self.on_end_session)
            btn_box.pack_start(end_btn, False, False, 0)

        cancel_btn = Gtk.Button(label="Close")
        cancel_btn.connect("clicked", lambda _: Gtk.main_quit())
        btn_box.pack_start(cancel_btn, False, False, 0)

    def get_prompt_text(self):
        buf = self.prompt_view.get_buffer()
        start, end = buf.get_bounds()
        return buf.get_text(start, end, True).strip()

    def on_sync_queue(self, _):
        send_cmd(self.sock_path, {"command": "sync_queue"})
        Gtk.main_quit()

    def on_save_session(self, _, is_update):
        doc_path = self.doc_entry.get_text().strip()
        prompt = self.get_prompt_text()
        if not doc_path or not prompt:
            return
        cmd = "update_session" if is_update else "new_session"
        send_cmd(self.sock_path, {"command": cmd, "doc_path": doc_path, "prompt": prompt})
        self.started_ok = True
        Gtk.main_quit()

    def on_end_session(self, _):
        send_cmd(self.sock_path, {"command": "end_session"})
        Gtk.main_quit()


MEMORY_TOOLS = [{
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Search your long-term memory of previous conversations and screen-capture notes. "
            "Call this whenever the user refers to something from the past, or you need a fact, "
            "preference, path, port or decision you were told before."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for."},
                "limit": {"type": "integer", "description": "How many memories to return (1-12)."},
            },
            "required": ["query"],
        },
    },
}]

MAX_TOOL_ROUNDS = 3

# Nerd Font glyphs (JetBrainsMono Nerd Font is the configured UI font).
ICON_SEND = ""        # paper plane
ICON_MENU = "☰"        # hamburger
ICON_DOWN = "↓"
ICON_NEW = ""         # plus

SHORTCUTS = [
    ("Enter", "send"),
    ("Shift+Enter", "newline"),
    ("Ctrl+N", "new chat"),
    ("Ctrl+B", "toggle sidebar"),
    ("Ctrl+F", "search conversations"),
    ("Ctrl+Shift+F", "search memories"),
    ("Ctrl+K", "compact memories"),
    ("Ctrl+End", "jump to bottom"),
    ("Ctrl+L", "focus message box"),
    ("Esc / Ctrl+W", "close"),
]


class MemoryDialog(Gtk.Dialog):
    """Edit or delete a single memory row."""

    RESPONSE_SAVE = 1
    RESPONSE_DELETE = 2

    def __init__(self, parent, item):
        super().__init__(title="Memory", transient_for=parent, modal=True)
        install_theme()
        self.set_default_size(560, 400)
        self.item = item
        style(self, "themed-dialog")
        for pane in (self.get_content_area(), self.get_action_area()):
            if pane is not None:
                style(pane, "themed-dialog")

        self.add_button("Delete", self.RESPONSE_DELETE)
        self.add_button("Cancel", Gtk.ResponseType.CANCEL)
        self.add_button("Save", self.RESPONSE_SAVE)
        for response, css in ((self.RESPONSE_DELETE, "danger-btn"),
                              (self.RESPONSE_SAVE, "accent-btn")):
            button = self.get_widget_for_response(response)
            if button is not None:
                style(button, css)

        content = self.get_content_area()
        content.set_spacing(7)
        content.set_border_width(12)

        meta = style(Gtk.Label(xalign=0.0), "chat-meta")
        meta.set_text("#%s  -  %s  -  seen %s time(s)  -  created %s" % (
            item["id"], item.get("source", "chat"), item.get("access_count", 0),
            (item.get("created_at") or "")[:10]))
        content.pack_start(meta, False, False, 0)

        content.pack_start(style(Gtk.Label(label="Content", xalign=0.0), "dim"), False, False, 0)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_vexpand(True)
        self.content_view = Gtk.TextView()
        self.content_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        for setter in ("set_left_margin", "set_right_margin",
                       "set_top_margin", "set_bottom_margin"):
            getattr(self.content_view, setter)(6)
        self.content_view.get_buffer().set_text(item["content"])
        scroll.add(self.content_view)
        content.pack_start(scroll, True, True, 0)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.pack_start(style(Gtk.Label(label="Kind", xalign=0.0), "dim"), False, False, 0)
        self.kind_combo = Gtk.ComboBoxText()
        for kind in memory_db.KINDS:
            self.kind_combo.append_text(kind)
        try:
            self.kind_combo.set_active(list(memory_db.KINDS).index(item["kind"]))
        except ValueError:
            self.kind_combo.set_active(0)
        row.pack_start(self.kind_combo, False, False, 0)

        row.pack_start(style(Gtk.Label(label="Importance", xalign=0.0), "dim"), False, False, 0)
        self.importance = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
        self.importance.set_value(float(item.get("importance", 0.5)))
        row.pack_start(self.importance, False, False, 0)
        content.pack_start(row, False, False, 0)

        content.pack_start(style(Gtk.Label(label="Tags (comma separated)", xalign=0.0), "dim"),
                           False, False, 0)
        self.tags_entry = Gtk.Entry()
        self.tags_entry.set_text(", ".join(item.get("tags") or []))
        content.pack_start(self.tags_entry, False, False, 0)

        self.show_all()

    def values(self):
        buf = self.content_view.get_buffer()
        start, end = buf.get_bounds()
        return {
            "content": buf.get_text(start, end, True).strip(),
            "kind": self.kind_combo.get_active_text() or self.item["kind"],
            "importance": self.importance.get_value(),
            "tags": [t.strip() for t in self.tags_entry.get_text().split(",") if t.strip()],
        }


class ChatPopup(Gtk.Window):
    """Chat window backed by the SQLite memory the model can also query itself."""

    def __init__(self, config, sock_path, conversation_id=None):
        super().__init__(title="AI Assistant Chat")
        install_theme()
        self.config = config
        self.sock_path = sock_path
        self.set_wmclass("ai-assistant-chat", "ai-assistant-chat")
        self.set_default_size(900, 680)
        self.set_position(Gtk.WindowPosition.CENTER)

        self.top_k = int(config_num(config, "memory_top_k", 6, float))
        self.min_importance = config_num(config, "memory_min_importance", 0.25, float)
        self.half_life = config_num(config, "memory_recency_half_life_days", 30, float)
        self.compress_threshold = int(config_num(config, "memory_compress_threshold", 2048, float))

        # The streaming worker runs off the main thread, so the connection is
        # shared under a lock rather than opened per-thread.
        self.db = memory_db.open_db(config["memory_db_path"], check_same_thread=False)
        self.db_lock = threading.Lock()
        self._db(memory_db.purge_empty_conversations)

        # A thread is only persisted once it actually holds a message, so
        # opening the window and closing it again leaves no empty rows behind.
        self.conversation_id = conversation_id
        self.busy = False
        self._stream_raw = ""
        self._stream_reasoning = ""
        self._stream_frame = None
        self._stream_label = None
        self._stats_label = None
        self._stats = {}
        self.session_tokens = 0
        self.context_size = context_size(config)
        self._stick_to_bottom = True
        self._anchor = None          # held scroll position while detached
        self._conv_rows = []
        self._mem_rows = []

        status = send_cmd(sock_path, {"command": "ai_mode_status"}, quiet=True) or {}
        self.ai_ready = status.get("phase") == "on"

        self._build_ui()
        if self.ai_ready:
            threading.Thread(target=self._probe_context, daemon=True).start()
        self.connect("key-press-event", self._on_window_key)
        self.connect("destroy", self._on_destroy)
        if self.conversation_id:
            self._load_history()
        self._refresh_sidebar()
        if not self.ai_ready:
            self._add_notice(
                "AI Mode is off -- the model servers are not running. "
                "Toggle AI Mode in waybar (or $mod+Shift+a) to start chatting.")
        GLib.idle_add(self._focus_input)

    def _db(self, fn, *args, **kwargs):
        """Run a memory_db call against the shared connection under the lock."""
        with self.db_lock:
            return fn(self.db, *args, **kwargs)

    # ---------------- UI ----------------
    def _build_ui(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(outer)

        # --- header ---
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_border_width(8)
        burger = style(Gtk.Button(label=ICON_MENU), "icon-btn")
        burger.set_tooltip_text("History & memory search  (Ctrl+B)\n\n"
                                + "\n".join(f"{k}  -  {v}" for k, v in SHORTCUTS))
        burger.connect("clicked", self.on_toggle_sidebar)
        header.pack_start(burger, False, False, 0)

        self.title_label = style(Gtk.Label(xalign=0.0), "heading")
        self.title_label.set_ellipsize(Pango.EllipsizeMode.END)
        header.pack_start(self.title_label, True, True, 0)

        self.usage_label = style(Gtk.Label(xalign=1.0), "chat-stats")
        self.usage_label.set_tooltip_text(
            "Prompt size against the server's context window, and the total "
            "tokens billed to this window since it opened.")
        header.pack_start(self.usage_label, False, False, 0)

        self.state_label = style(Gtk.Label(xalign=1.0),
                                 "status-ok" if self.ai_ready else "status-err")
        self.state_label.set_text("AI Mode: ON" if self.ai_ready else "AI Mode: OFF")
        header.pack_start(self.state_label, False, False, 0)
        outer.pack_start(header, False, False, 0)

        # --- body: sidebar | transcript ---
        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        outer.pack_start(body, True, True, 0)

        self.sidebar_revealer = Gtk.Revealer()
        self.sidebar_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_RIGHT)
        self.sidebar_revealer.set_transition_duration(140)
        self.sidebar_revealer.add(self._build_sidebar())
        body.pack_start(self.sidebar_revealer, False, False, 0)

        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        right.set_border_width(10)
        body.pack_start(right, True, True, 0)

        # transcript, with the jump-to-bottom pill floating over it
        overlay = Gtk.Overlay()
        self.scroller = Gtk.ScrolledWindow()
        self.scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.scroller.set_vexpand(True)
        self.log_box = style(Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12), "chat-log")
        # Inset via CSS padding, NOT set_border_width(): GTK3 does not paint a
        # container's background across its border-width, so the unpainted
        # viewport showed through as a black band around the transcript.
        # Detach focus-follows-scrolling: clicking or selecting inside the
        # transcript must never reposition it.
        self.log_box.set_focus_vadjustment(Gtk.Adjustment())
        viewport = style(Gtk.Viewport(), "chat-surface")
        viewport.set_shadow_type(Gtk.ShadowType.NONE)
        viewport.add(self.log_box)
        self.scroller.add(viewport)
        overlay.add(self.scroller)

        self.scroll_btn = style(Gtk.Button(label=ICON_DOWN), "scroll-btn")
        self.scroll_btn.set_tooltip_text("Jump to latest  (Ctrl+End)")
        self.scroll_btn.set_halign(Gtk.Align.END)
        self.scroll_btn.set_valign(Gtk.Align.END)
        self.scroll_btn.set_no_show_all(True)
        self.scroll_btn.connect("clicked", lambda _: self.scroll_to_bottom())
        overlay.add_overlay(self.scroll_btn)
        right.pack_start(overlay, True, True, 0)

        adj = self.scroller.get_vadjustment()
        adj.connect("value-changed", self._on_scroll)
        adj.connect("changed", self._on_scroll_extent)

        self.memory_expander = Gtk.Expander(label="Recalled memories")
        self.memory_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.memory_box.set_border_width(6)
        self.memory_expander.add(self.memory_box)
        right.pack_start(self.memory_expander, False, False, 0)

        # --- composer: send button and text view share one bordered bar ---
        bar = style(Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0), "input-bar")
        self.send_btn = style(Gtk.Button(label=ICON_SEND), "send-btn")
        self.send_btn.set_tooltip_text("Send  (Enter)")
        self.send_btn.set_valign(Gtk.Align.END)
        self.send_btn.connect("clicked", self.on_send)
        self.send_btn.set_sensitive(self.ai_ready)
        bar.pack_start(self.send_btn, False, False, 0)

        input_scroll = Gtk.ScrolledWindow()
        input_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        input_scroll.set_min_content_height(34)
        input_scroll.set_max_content_height(150)
        input_scroll.set_propagate_natural_height(True)
        self.input_view = Gtk.TextView()
        self.input_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        for setter in ("set_left_margin", "set_right_margin",
                       "set_top_margin", "set_bottom_margin"):
            getattr(self.input_view, setter)(7)
        self.input_view.connect("key-press-event", self._on_input_key)
        input_scroll.add(self.input_view)
        bar.pack_start(input_scroll, True, True, 0)
        right.pack_start(bar, False, False, 0)

        self._refresh_title()

    def _build_sidebar(self):
        box = style(Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5), "sidebar")
        box.set_size_request(268, -1)
        box.set_border_width(7)

        new_btn = style(Gtk.Button(label=f"{ICON_NEW}   New Chat"), "thin-btn")
        new_btn.set_tooltip_text("Start a new conversation  (Ctrl+N)")
        new_btn.connect("clicked", self.on_new_chat)
        box.pack_start(new_btn, False, False, 0)

        # --- conversations (top) ---
        self.conv_search = style(Gtk.SearchEntry(), "search", "thin-entry")
        self.conv_search.set_placeholder_text("Search conversations...")
        self.conv_search.set_tooltip_text(
            "Semantic-ish search across conversation titles, summaries and messages "
            "(Ctrl+F).\nThe 0.5B model expands your query with related terms first.")
        self.conv_search.connect("search-changed", self._on_conv_search_changed)
        box.pack_start(self.conv_search, False, False, 0)

        self.conv_header = style(Gtk.Label(xalign=0.0), "sidebar-title")
        self.conv_header.set_text("CONVERSATIONS")
        box.pack_start(self.conv_header, False, False, 0)

        conv_scroll = Gtk.ScrolledWindow()
        conv_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        conv_scroll.set_vexpand(True)
        self.conv_list = Gtk.ListBox()
        self.conv_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.conv_list.connect("row-activated", self._on_conv_activated)
        self.conv_list.connect("button-press-event", self._on_conv_click)
        conv_scroll.add(self.conv_list)
        box.pack_start(conv_scroll, True, True, 0)

        box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 2)

        # --- memories (bottom), search bar under the list ---
        self.mem_header = style(Gtk.Label(xalign=0.0), "sidebar-title")
        self.mem_header.set_text("MEMORIES")
        box.pack_start(self.mem_header, False, False, 0)

        mem_scroll = Gtk.ScrolledWindow()
        mem_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        mem_scroll.set_size_request(-1, 180)
        self.mem_list = Gtk.ListBox()
        self.mem_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.mem_list.connect("row-activated", self._on_mem_activated)
        mem_scroll.add(self.mem_list)
        box.pack_start(mem_scroll, False, False, 0)

        self.mem_search = style(Gtk.SearchEntry(), "search", "thin-entry")
        self.mem_search.set_placeholder_text("Search memories...")
        self.mem_search.set_tooltip_text(
            "Semantic-ish search over the memory database (Ctrl+Shift+F).\n"
            "Click a result to edit or delete it.")
        self.mem_search.connect("search-changed", self._on_mem_search_changed)
        box.pack_start(self.mem_search, False, False, 0)

        compact_btn = style(Gtk.Button(label="Compact memories"), "thin-btn")
        compact_btn.set_tooltip_text("Merge duplicate memories  (Ctrl+K)")
        compact_btn.connect("clicked", self.on_compact)
        box.pack_start(compact_btn, False, False, 0)
        return box

    def _focus_input(self):
        self.input_view.grab_focus()
        return False

    def _probe_context(self):
        """Ask the server for its real context window.

        The command line in .config can disagree with the running server (the
        daemon reads config at start-up, and llama.cpp may clamp -c), so /props
        is the authoritative source for the ctx denominator.
        """
        try:
            base = self.config["llama_url"].split("/v1/")[0]
            resp = requests.get(base + "/props", timeout=5)
            resp.raise_for_status()
            n_ctx = (resp.json().get("default_generation_settings") or {}).get("n_ctx")
            if n_ctx and int(n_ctx) != self.context_size:
                self.context_size = int(n_ctx)
                GLib.idle_add(self._refresh_title)
        except Exception as e:
            print(f"Context probe failed ({e}); using the configured value.", file=sys.stderr)

    def _refresh_title(self):
        if self.conversation_id:
            convo = self._db(memory_db.get_conversation, self.conversation_id) or {}
            name = convo.get("title") or f"Conversation #{self.conversation_id}"
        else:
            name = "New conversation"
        model = self.config.get("llama_model", "local model").split("/")[-1]
        self.title_label.set_text(f"{name}   -   {model}")

        bits = []
        prompt = self._stats.get("prompt", 0)
        if prompt:
            bits.append(f"ctx {fmt_count(prompt)}/{fmt_count(self.context_size)}"
                        if self.context_size else f"ctx {fmt_count(prompt)}")
        if self.session_tokens:
            bits.append(f"session {fmt_count(self.session_tokens)} tok")
        if getattr(self, "usage_label", None) is not None:
            self.usage_label.set_text("   ".join(bits))

    def _plain_label(self, text, selectable=True):
        label = Gtk.Label(label=text, xalign=0.0)
        label.set_line_wrap(True)
        label.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)
        # A selectable label takes focus, and replacing the text of a focused
        # child makes the ScrolledWindow scroll to reveal it -- which would drag
        # the view back down on every streamed delta. The final rendered labels
        # are selectable; the volatile streaming one is not.
        label.set_selectable(selectable)
        return label

    def _code_widget(self, language, code):
        box = style(Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2), "code-block")
        if language:
            box.pack_start(style(Gtk.Label(label=language, xalign=0.0), "code-lang"),
                           False, False, 0)
        label = Gtk.Label(xalign=0.0)
        label.set_selectable(True)
        label.set_markup(f'<tt>{highlight_code(code, language)}</tt>')
        # Long lines scroll sideways rather than wrapping, which keeps code readable.
        # Overlay scrollbars would be painted over the last line, so give the
        # bar its own space instead.
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
        scroll.set_propagate_natural_height(True)
        scroll.set_overlay_scrolling(False)
        scroll.add(label)
        box.pack_start(scroll, False, False, 0)
        return box

    def _table_widget(self, header, rows):
        grid = style(Gtk.Grid(), "md-table")
        grid.set_column_spacing(16)
        grid.set_row_spacing(3)
        for col, cell in enumerate(header):
            label = style(Gtk.Label(xalign=0.0), "md-th")
            label.set_markup(f"<b>{render_inline(cell)}</b>")
            label.set_line_wrap(True)
            grid.attach(label, col, 0, 1, 1)
        grid.attach(style(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), "md-rule"),
                    0, 1, max(1, len(header)), 1)
        for r, row in enumerate(rows, start=2):
            for col, cell in enumerate(row[:max(1, len(header))]):
                label = style(Gtk.Label(xalign=0.0), "md-td")
                label.set_markup(render_inline(cell))
                label.set_line_wrap(True)
                label.set_selectable(True)
                grid.attach(label, col, r, 1, 1)

        # Wide tables scroll sideways rather than squeezing the transcript.
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
        scroll.set_propagate_natural_height(True)
        scroll.set_overlay_scrolling(False)
        scroll.add(grid)
        return scroll

    def _thinking_widget(self, text):
        """Collapsed dropdown holding the model's reasoning for this turn."""
        expander = style(Gtk.Expander(), "think-box")
        expander.set_label(f"Thinking  ({fmt_count(len(text))} characters)")
        expander.set_expanded(False)

        body = style(self._plain_label(text), "think-body")
        inner = style(Gtk.Viewport(), "think-surface")
        inner.set_shadow_type(Gtk.ShadowType.NONE)
        inner.add(body)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_max_content_height(280)
        scroll.set_propagate_natural_height(True)
        scroll.set_overlay_scrolling(False)
        scroll.add(inner)
        expander.add(scroll)
        return expander

    def _fill_markdown(self, frame, text):
        for child in frame.get_children():
            frame.remove(child)
        for block in render_markdown_blocks(text):
            if block[0] == "code":
                frame.pack_start(self._code_widget(block[1], block[2]), False, False, 0)
            elif block[0] == "table":
                frame.pack_start(self._table_widget(block[1], block[2]), False, False, 0)
            else:
                label = style(self._plain_label(""), "md-body")
                try:
                    label.set_markup(block[1])
                except Exception:
                    label.set_text(block[1])
                frame.pack_start(label, False, False, 0)
        frame.show_all()

    def _add_bubble(self, role, text, markdown=True):
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        meta = style(Gtk.Label(xalign=0.0), "chat-meta")
        meta.set_text("you" if role == "user" else "assistant")

        frame = style(Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6),
                      "chat-user" if role == "user" else "chat-assistant")
        row.pack_start(meta, False, False, 0)
        row.pack_start(frame, False, False, 0)
        self.log_box.pack_start(row, False, False, 0)

        if markdown:
            self._fill_markdown(frame, text)
        else:
            frame.pack_start(self._plain_label(text, selectable=False), False, False, 0)
        row.show_all()
        self._maybe_autoscroll()
        return frame

    def _add_notice(self, text, css="chat-error"):
        label = style(Gtk.Label(label=text, xalign=0.0), css)
        label.set_line_wrap(True)
        self.log_box.pack_start(label, False, False, 0)
        label.show_all()
        self._maybe_autoscroll()

    def _load_history(self):
        for turn in self._db(memory_db.get_messages, self.conversation_id):
            if turn["role"] in ("user", "assistant") and turn["content"].strip():
                self._add_bubble(turn["role"], turn["content"])

    # ---------------- scrolling ----------------
    def _at_bottom(self, adj=None):
        adj = adj or self.scroller.get_vadjustment()
        return adj.get_value() + adj.get_page_size() >= adj.get_upper() - 40

    def _on_scroll(self, adj):
        # Scrolling up detaches the view so a streaming reply stops yanking it
        # back down; the pill appears until the user returns to the bottom.
        self._stick_to_bottom = self._at_bottom(adj)
        # The anchor tracks the user's own scrolling, so they always win; it is
        # only used to undo drift caused by the content growing underneath.
        self._anchor = None if self._stick_to_bottom else adj.get_value()
        self.scroll_btn.set_visible(not self._stick_to_bottom)

    def _on_scroll_extent(self, adj):
        """The content changed size: either follow it, or hold position."""
        if self._stick_to_bottom:
            adj.set_value(max(0, adj.get_upper() - adj.get_page_size()))
            return
        if self._anchor is not None and abs(adj.get_value() - self._anchor) > 1:
            adj.set_value(self._anchor)
        self.scroll_btn.set_visible(True)

    def _maybe_autoscroll(self):
        if self._stick_to_bottom:
            GLib.idle_add(self._autoscroll)

    def _autoscroll(self):
        """Follow new output, but only while the view is still attached.

        Re-checked here rather than at queue time: the user may have scrolled
        away between a delta arriving and this idle callback running, and
        honouring the stale request would yank them back down mid-read.
        """
        if self._stick_to_bottom:
            adj = self.scroller.get_vadjustment()
            adj.set_value(max(0, adj.get_upper() - adj.get_page_size()))
        return False

    def scroll_to_bottom(self):
        """Explicit jump (button or Ctrl+End); re-attaches the view."""
        self._stick_to_bottom = True
        self._anchor = None
        adj = self.scroller.get_vadjustment()
        adj.set_value(max(0, adj.get_upper() - adj.get_page_size()))
        self.scroll_btn.set_visible(False)
        return False

    # ---------------- keys ----------------
    def _on_input_key(self, _widget, event):
        if event.keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            if event.state & (Gdk.ModifierType.SHIFT_MASK | Gdk.ModifierType.CONTROL_MASK):
                return False
            self.on_send(None)
            return True
        return False

    def _on_window_key(self, _widget, event):
        ctrl = event.state & Gdk.ModifierType.CONTROL_MASK
        key = event.keyval

        if ctrl:
            if key in (Gdk.KEY_n, Gdk.KEY_N):
                self.on_new_chat(None)
            elif key in (Gdk.KEY_b, Gdk.KEY_B):
                self.on_toggle_sidebar(None)
            elif key in (Gdk.KEY_f, Gdk.KEY_F):
                self._open_search(memories=bool(event.state & Gdk.ModifierType.SHIFT_MASK))
            elif key in (Gdk.KEY_k, Gdk.KEY_K):
                self.on_compact(None)
            elif key in (Gdk.KEY_l, Gdk.KEY_L):
                self._focus_input()
            elif key in (Gdk.KEY_w, Gdk.KEY_W):
                self.destroy()
            elif key == Gdk.KEY_End:
                self.scroll_to_bottom()
            else:
                return False
            return True

        if key == Gdk.KEY_Escape:
            # Unwind the UI one step at a time before closing the window.
            if self.conv_search.get_text() or self.mem_search.get_text():
                self.conv_search.set_text("")
                self.mem_search.set_text("")
                self._focus_input()
            elif self.sidebar_revealer.get_reveal_child():
                self.sidebar_revealer.set_reveal_child(False)
                self._focus_input()
            else:
                self.destroy()
            return True
        return False

    # ---------------- sidebar actions ----------------
    def on_toggle_sidebar(self, _):
        shown = not self.sidebar_revealer.get_reveal_child()
        self.sidebar_revealer.set_reveal_child(shown)
        if shown:
            self._refresh_sidebar()
        else:
            self._focus_input()

    def _open_search(self, memories=False):
        self.sidebar_revealer.set_reveal_child(True)
        (self.mem_search if memories else self.conv_search).grab_focus()

    # ---- list helpers -------------------------------------------------
    @staticmethod
    def _clear_list(listbox, rows):
        for child in listbox.get_children():
            listbox.remove(child)
        rows.clear()

    def _append_row(self, listbox, rows, title, subtitle, payload):
        row = Gtk.ListBoxRow()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        name = style(Gtk.Label(xalign=0.0), "row-title")
        name.set_ellipsize(Pango.EllipsizeMode.END)
        name.set_text(title)
        box.pack_start(name, False, False, 0)
        if subtitle:
            sub = style(Gtk.Label(xalign=0.0), "row-sub")
            sub.set_ellipsize(Pango.EllipsizeMode.END)
            sub.set_text(subtitle)
            box.pack_start(sub, False, False, 0)
        row.add(box)
        listbox.add(row)
        rows.append(payload)
        return row

    # ---- conversations -------------------------------------------------
    def _refresh_sidebar(self):
        self._refresh_conversations()
        self._refresh_memories()

    def _refresh_conversations(self):
        if self.conv_search.get_text().strip():
            return
        self._clear_list(self.conv_list, self._conv_rows)
        self.conv_header.set_text("CONVERSATIONS")
        for convo in self._db(memory_db.list_conversations, limit=200):
            when = (convo.get("updated_at") or "")[:16].replace("T", " ")
            self._append_row(self.conv_list, self._conv_rows,
                             convo.get("title") or f"Conversation #{convo['id']}",
                             when, convo["id"])
        self.conv_list.show_all()

    def _on_conv_search_changed(self, entry):
        query = entry.get_text().strip()
        if not query:
            self._refresh_conversations()
            return
        if len(query) < 2:
            return
        threading.Thread(target=self._conv_search_worker, args=(query,), daemon=True).start()

    def _conv_search_worker(self, query):
        expanded = self._expand_query(query) if self.ai_ready else query
        try:
            rows = self._db(memory_db.search_conversations, expanded, k=40,
                            half_life_days=self.half_life)
        except Exception as e:
            print(f"Conversation search failed: {e}", file=sys.stderr)
            rows = []
        GLib.idle_add(self._show_conv_results, query, rows)

    def _show_conv_results(self, query, rows):
        if self.conv_search.get_text().strip() != query:
            return False                      # a newer keystroke already won
        self._clear_list(self.conv_list, self._conv_rows)
        self.conv_header.set_text(f"CONVERSATIONS ({len(rows)})")
        for convo in rows:
            subtitle = convo.get("snippet") or (convo.get("updated_at") or "")[:16]
            self._append_row(self.conv_list, self._conv_rows,
                             convo.get("title") or f"Conversation #{convo['id']}",
                             "%.2f  %s" % (convo["score"], subtitle), convo["id"])
        if not rows:
            self._append_row(self.conv_list, self._conv_rows, "No matching conversations", "", None)
        self.conv_list.show_all()
        return False

    def _on_conv_activated(self, _listbox, row):
        index = row.get_index()
        if 0 <= index < len(self._conv_rows) and self._conv_rows[index] is not None:
            self._open_conversation(self._conv_rows[index])

    def _on_conv_click(self, listbox, event):
        """Right click offers to delete the conversation under the pointer."""
        if event.button != 3:
            return False
        row = listbox.get_row_at_y(int(event.y))
        if row is None:
            return False
        index = row.get_index()
        if not (0 <= index < len(self._conv_rows)) or self._conv_rows[index] is None:
            return False

        listbox.select_row(row)
        conversation_id = self._conv_rows[index]
        convo = self._db(memory_db.get_conversation, conversation_id) or {}
        title = convo.get("title") or f"Conversation #{conversation_id}"

        menu = Gtk.Menu()
        item = Gtk.MenuItem(label=f'Delete "{title[:34]}"')
        item.connect("activate", lambda _w: self._delete_conversation(conversation_id, title))
        menu.append(item)
        menu.show_all()
        menu.attach_to_widget(self, None)
        menu.popup_at_pointer(event)
        return True

    def _confirm(self, heading, detail):
        dialog = Gtk.MessageDialog(
            transient_for=self, modal=True,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text=heading,
        )
        dialog.format_secondary_text(detail)
        style(dialog, "themed-dialog")
        for pane in (dialog.get_content_area(), dialog.get_action_area()):
            if pane is not None:
                style(pane, "themed-dialog")
        answer = dialog.run()
        dialog.destroy()
        return answer == Gtk.ResponseType.OK

    def _delete_conversation(self, conversation_id, title):
        count = (self._db(memory_db.get_conversation, conversation_id) or {}).get("message_count", 0)
        if not self._confirm("Delete this conversation?",
                             f"{title}\n{count} message(s) will be removed permanently. "
                             "Memories already distilled from it are kept."):
            return
        self._db(memory_db.purge_conversation, conversation_id)
        if self.conversation_id == conversation_id:
            self.conversation_id = None
            for child in self.log_box.get_children():
                self.log_box.remove(child)
            self._refresh_title()
        self._refresh_conversations()

    # ---- memories -------------------------------------------------
    def _refresh_memories(self):
        if self.mem_search.get_text().strip():
            return
        self._clear_list(self.mem_list, self._mem_rows)
        try:
            recent = self._db(memory_db.search, "", k=25, touch=False,
                              half_life_days=self.half_life)
        except Exception:
            recent = []
        self.mem_header.set_text(f"MEMORIES ({len(recent)})")
        for item in recent:
            self._append_row(self.mem_list, self._mem_rows, item["content"][:120],
                             "%s  %s" % (item["kind"], ", ".join(item["tags"])), item)
        self.mem_list.show_all()

    def _on_mem_search_changed(self, entry):
        query = entry.get_text().strip()
        if not query:
            self._refresh_memories()
            return
        if len(query) < 2:
            return
        threading.Thread(target=self._search_worker, args=(query,), daemon=True).start()

    def _search_worker(self, query):
        expanded = self._expand_query(query) if self.ai_ready else query
        try:
            rows = self._db(memory_db.search, expanded, k=40,
                            half_life_days=self.half_life, touch=False)
        except Exception as e:
            print(f"Memory search failed: {e}", file=sys.stderr)
            rows = []
        GLib.idle_add(self._show_search_results, query, rows)

    def _expand_query(self, query):
        """Widen a search with related terms from the 0.5B model.

        Without an embedding endpoint this is as close to semantic search as we
        get: the extra terms feed straight into the same FTS ranking.
        """
        try:
            payload = {
                "messages": [
                    {"role": "system", "content":
                        "Expand the search query. Reply with a comma-separated list of 8 to 12 "
                        "single words that are synonyms of, or strongly associated with, the "
                        "query. Include the query's own words. No numbering, no explanation, "
                        "no markdown, no sentences."},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.3,
                "max_tokens": 90,
            }
            model = self.config.get("llama_small_model", "")
            if model:
                payload["model"] = model
            resp = requests.post(self.config["llama_small_url"], json=payload, timeout=20)
            resp.raise_for_status()
            text = strip_think(resp.json()["choices"][0]["message"]["content"])
            terms = [t.strip(" .\"'`*-") for t in re.split(r"[,\n]", text)]
            terms = [t for t in terms if t and len(t) < 40][:12]
            if terms:
                print(f"[chat] query expansion: {query!r} + {terms}", file=sys.stderr)
                return query + " " + " ".join(terms)
        except Exception as e:
            print(f"Query expansion unavailable ({e}); using literal search.", file=sys.stderr)
        return query

    def _show_search_results(self, query, rows):
        if self.mem_search.get_text().strip() != query:
            return False                      # a newer keystroke already won
        self._clear_list(self.mem_list, self._mem_rows)
        self.mem_header.set_text(f"MEMORIES ({len(rows)})")
        for item in rows:
            tags = ", ".join(item["tags"]) if item["tags"] else item["kind"]
            self._append_row(self.mem_list, self._mem_rows, item["content"][:120],
                             "%.2f  %s" % (item["score"], tags), item)
        if not rows:
            self._append_row(self.mem_list, self._mem_rows, "No matching memories", "", None)
        self.mem_list.show_all()
        return False

    def _on_mem_activated(self, _listbox, row):
        index = row.get_index()
        if not (0 <= index < len(self._mem_rows)) or self._mem_rows[index] is None:
            return
        self._edit_memory(self._mem_rows[index])

    def _edit_memory(self, item):
        """Open one memory for editing or deletion."""
        fresh = self._db(memory_db.get_memory, item["id"])
        if fresh is None:
            self._add_notice("That memory no longer exists.")
            self._refresh_memories()
            return

        dialog = MemoryDialog(self, fresh)
        response = dialog.run()
        values = dialog.values()
        dialog.destroy()

        if response == MemoryDialog.RESPONSE_SAVE:
            self._db(memory_db.update_memory, fresh["id"],
                     content=values["content"], kind=values["kind"],
                     importance=values["importance"], tags=values["tags"])
            self._add_notice("Memory updated.", css="memory-chip")
        elif response == MemoryDialog.RESPONSE_DELETE:
            if self._confirm("Delete this memory?",
                             fresh["content"][:200] + "\n\nThis cannot be undone."):
                self._db(memory_db.delete_memory, fresh["id"])
                self._add_notice("Memory deleted.", css="memory-chip")
            else:
                return
        else:
            return

        if self.mem_search.get_text().strip():
            self._on_mem_search_changed(self.mem_search)
        else:
            self._refresh_memories()

    def _open_conversation(self, conversation_id):
        if self.busy:
            return
        self._discard_if_empty()
        self.conversation_id = conversation_id
        for child in self.log_box.get_children():
            self.log_box.remove(child)
        self._stick_to_bottom = True
        self._load_history()
        self._refresh_title()
        self._focus_input()

    # ---------------- chat actions ----------------
    def _discard_if_empty(self):
        """Drop the current thread if nothing was ever said in it."""
        if not self.conversation_id:
            return
        convo = self._db(memory_db.get_conversation, self.conversation_id)
        if convo and not convo.get("message_count"):
            self._db(memory_db.purge_conversation, self.conversation_id)
        self.conversation_id = None

    def _ensure_conversation(self):
        if not self.conversation_id:
            self.conversation_id = self._db(memory_db.start_conversation)
        return self.conversation_id

    def _input_text(self):
        buf = self.input_view.get_buffer()
        start, end = buf.get_bounds()
        return buf.get_text(start, end, True).strip()

    def on_send(self, _):
        if self.busy or not self.ai_ready:
            return
        text = self._input_text()
        if not text:
            return
        self.input_view.get_buffer().set_text("")
        self.busy = True
        self.send_btn.set_sensitive(False)

        first_turn = not self.conversation_id
        self._ensure_conversation()
        self._add_bubble("user", text)
        self._db(memory_db.add_message, self.conversation_id, "user", text,
                 compress_threshold=self.compress_threshold)

        self._stream_raw = ""
        self._stream_reasoning = ""
        self._stats = {}
        # Plain text while streaming (cheap, no relayout churn); the markdown
        # and syntax highlighting are rendered once the reply is complete.
        self._stream_frame = self._add_bubble("assistant", "...", markdown=False)
        self._stream_label = self._stream_frame.get_children()[0]
        self._stats_label = style(Gtk.Label(xalign=1.0), "chat-stats")
        self._stats_label.set_text("…")
        row = self._stream_frame.get_parent()
        row.pack_start(self._stats_label, False, False, 0)
        row.show_all()
        self._stick_to_bottom = True
        if first_turn:
            self._refresh_title()
        threading.Thread(target=self._chat_worker, args=(text,), daemon=True).start()

    def on_new_chat(self, _):
        if self.busy:
            return
        self._discard_if_empty()
        for child in self.log_box.get_children():
            self.log_box.remove(child)
        for child in self.memory_box.get_children():
            self.memory_box.remove(child)
        self.memory_expander.set_label("Recalled memories")
        self._stick_to_bottom = True
        self._refresh_title()
        self._refresh_sidebar()
        self._focus_input()

    def on_compact(self, _):
        result = send_cmd(self.sock_path, {"command": "memory_compact"}, quiet=True, timeout=180.0)
        if result:
            self._add_notice(
                f"Memory compacted: {result['merged']} merged, {result['remaining']} kept.",
                css="memory-chip")
        else:
            self._add_notice("Could not reach the daemon to compact memory.")

    def _on_destroy(self, _widget):
        self._discard_if_empty()
        Gtk.main_quit()

    # ---------------- model plumbing ----------------
    def _build_messages(self, memories):
        convo = self._db(memory_db.get_conversation, self.conversation_id) or {}

        # Qwen3's chat template raises "System message must be at the beginning",
        # so the prompt, the thread summary and the recalled memories are merged
        # into a single leading system turn rather than sent as separate ones.
        parts = [self.config["chat_system_prompt"]]
        if convo.get("summary"):
            parts.append("Summary of this conversation so far:\n" + convo["summary"])
        if memories:
            parts.append("Relevant memories from earlier conversations (most relevant first):\n"
                         + memory_db.format_for_prompt(memories))
        messages = [{"role": "system", "content": "\n\n".join(parts)}]

        for turn in self._db(memory_db.get_messages, self.conversation_id, limit=12):
            if turn["role"] in ("user", "assistant") and turn["content"].strip():
                messages.append({"role": turn["role"], "content": turn["content"]})
        return messages

    def _stream_round(self, messages):
        """One streamed completion; returns (content, [tool_calls])."""
        resp = requests.post(
            self.config["llama_url"],
            json={
                "model": self.config.get("llama_model", ""),
                "messages": messages,
                "temperature": 0.6,
                "stream": True,
                "tools": MEMORY_TOOLS,
                # llama.cpp puts exact usage in the final chunk and live
                # tok/s in almost every chunk when these are set.
                "stream_options": {"include_usage": True},
                "timings_per_token": True,
            },
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()

        parts, tool_acc = [], {}
        reasoning_chars, reasoning_mark = 0, 0
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            timings = chunk.get("timings")
            if timings:
                self._stats["cur_completion"] = timings.get("predicted_n") or 0
                self._stats["cur_ms"] = timings.get("predicted_ms") or 0.0
                self._stats["cur_tps"] = timings.get("predicted_per_second") or 0.0
                if timings.get("prompt_n"):
                    self._stats["prompt"] = timings["prompt_n"]
                    self._stats["prompt_tps"] = timings.get("prompt_per_second") or 0.0
            usage = chunk.get("usage")
            if usage:
                self._stats["prompt"] = usage.get("prompt_tokens", self._stats.get("prompt", 0))
                self._stats["total"] = (self._stats.get("total", 0)
                                        + (usage.get("total_tokens") or 0))
                details = usage.get("prompt_tokens_details") or {}
                if details.get("cached_tokens"):
                    self._stats["cached"] = details["cached_tokens"]

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            piece = delta.get("content")
            if piece:
                parts.append(piece)
                GLib.idle_add(self._append_delta, piece)

            # llama.cpp streams this model's chain-of-thought in its own field
            # rather than inside <think> tags, so without this the bubble would
            # sit on a motionless placeholder through a long reasoning phase.
            reasoning = delta.get("reasoning_content")
            if reasoning:
                self._stream_reasoning += reasoning
                reasoning_chars += len(reasoning)
                if reasoning_chars - reasoning_mark >= 200:
                    reasoning_mark = reasoning_chars
                    GLib.idle_add(self._note_thinking, reasoning_chars)

            # tool_calls arrive as fragments keyed by index
            for call in delta.get("tool_calls") or []:
                slot = tool_acc.setdefault(call.get("index", 0),
                                           {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

        self._stats["done_completion"] = (self._stats.get("done_completion", 0)
                                          + self._stats.pop("cur_completion", 0))
        self._stats["done_ms"] = self._stats.get("done_ms", 0.0) + self._stats.pop("cur_ms", 0.0)
        self._stats.pop("cur_tps", None)
        return "".join(parts), [tool_acc[k] for k in sorted(tool_acc)]

    def _run_tool(self, call):
        if call.get("name") != "memory_search":
            return "Unknown tool."
        try:
            args = json.loads(call.get("arguments") or "{}")
        except ValueError:
            args = {}
        query = str(args.get("query", "")).strip()
        try:
            limit = max(1, min(12, int(args.get("limit") or self.top_k)))
        except (TypeError, ValueError):
            limit = self.top_k

        rows = self._db(memory_db.search, query, k=limit, half_life_days=self.half_life)
        print(f"[chat] memory_search({query!r}, limit={limit}) -> {len(rows)} hit(s)",
              file=sys.stderr)
        GLib.idle_add(self._note_tool_use, query, rows)
        return memory_db.format_for_prompt(rows) or "No matching memories."

    def _chat_worker(self, user_text):
        try:
            memories = self._db(
                memory_db.search, user_text, k=self.top_k,
                min_importance=self.min_importance, half_life_days=self.half_life,
            )
        except Exception as e:
            print(f"Memory retrieval failed: {e}", file=sys.stderr)
            memories = []
        GLib.idle_add(self._show_recalled, memories)

        messages = self._build_messages(memories)
        try:
            for _ in range(MAX_TOOL_ROUNDS + 1):
                content, tool_calls = self._stream_round(messages)
                if not tool_calls:
                    break
                messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {"id": c["id"] or f"call_{i}", "type": "function",
                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for i, c in enumerate(tool_calls)
                    ],
                })
                for i, call in enumerate(tool_calls):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"] or f"call_{i}",
                        "content": self._run_tool(call),
                    })
        except requests.exceptions.RequestException as e:
            GLib.idle_add(self._finish_turn, "", f"Request failed: {e}")
            return
        except Exception as e:
            GLib.idle_add(self._finish_turn, "", f"Chat error: {e}")
            return

        GLib.idle_add(self._finish_turn, strip_think(self._stream_raw), None)

    def _title_worker(self, conversation_id, user_text, answer):
        """Name the thread with the small model; the 35B is busy and overkill."""
        try:
            payload = {
                "messages": [
                    {"role": "system", "content":
                        "Give the conversation a title of three to six words. Reply with the "
                        "title only: no quotes, no trailing punctuation, no markdown, no preamble."},
                    {"role": "user", "content":
                        f"User: {user_text[:600]}\n\nAssistant: {answer[:600]}"},
                ],
                "temperature": 0.3,
                "max_tokens": 32,
            }
            model = self.config.get("llama_small_model", "")
            if model:
                payload["model"] = model
            resp = requests.post(self.config["llama_small_url"], json=payload, timeout=25)
            resp.raise_for_status()
            raw = strip_think(resp.json()["choices"][0]["message"]["content"])
            title = clamp_title(raw)
            if title:
                self._db(memory_db.rename_conversation, conversation_id, title)
                GLib.idle_add(self._title_ready)
        except Exception as e:
            print(f"Title generation failed: {e}", file=sys.stderr)

    def _title_ready(self):
        self._refresh_title()
        if self.sidebar_revealer.get_reveal_child():
            self._refresh_sidebar()
        return False

    # ---------------- main-thread UI callbacks ----------------
    def _append_delta(self, piece):
        self._stream_raw += piece
        if self._stream_label is not None:
            # Re-render from the raw buffer so a <think> block that is still
            # open never flashes into view. Reasoning models can sit inside one
            # for a long time, so say so rather than showing a frozen "...".
            visible = strip_think(self._stream_raw)
            self._stream_label.set_text(visible or "thinking\u2026")
            if self._stats_label is not None:
                self._stats_label.set_text(format_stats(self._stats, live=True))
            self._maybe_autoscroll()
        return False

    def _note_thinking(self, chars):
        """Show reasoning progress while no answer text has arrived yet."""
        if self._stream_label is not None and not strip_think(self._stream_raw):
            self._stream_label.set_text("thinking\u2026  (%s characters)" % fmt_count(chars))
            if self._stats_label is not None:
                self._stats_label.set_text(format_stats(self._stats, live=True))
            self._maybe_autoscroll()
        return False

    def _show_recalled(self, memories):
        for child in self.memory_box.get_children():
            self.memory_box.remove(child)
        self.memory_expander.set_label(f"Recalled memories ({len(memories)})")
        for item in memories:
            tags = (" [%s]" % ", ".join(item["tags"])) if item["tags"] else ""
            label = style(Gtk.Label(xalign=0.0), "memory-chip")
            label.set_line_wrap(True)
            label.set_text("%.2f  %s%s  %s" % (
                item["score"], item["kind"], tags, item["content"][:180]))
            self.memory_box.pack_start(label, False, False, 0)
        self.memory_box.show_all()
        return False

    def _note_tool_use(self, query, rows):
        label = style(Gtk.Label(xalign=0.0), "memory-chip")
        label.set_line_wrap(True)
        label.set_text(f"[tool] memory_search({query!r}) -> {len(rows)} hit(s)")
        self.memory_box.pack_start(label, False, False, 0)
        self.memory_expander.set_expanded(True)
        self.memory_box.show_all()
        return False

    def _finish_turn(self, answer, error):
        self.busy = False
        self.send_btn.set_sensitive(self.ai_ready)

        if error:
            if self._stream_frame is not None:
                self._fill_markdown(self._stream_frame, answer or "(no response)")
            if self._stats_label is not None:
                self._stats_label.set_text(format_stats(self._stats))
            self._add_notice(error)
            self._stream_frame = self._stream_label = self._stats_label = None
            self._focus_input()
            return False

        answer = answer or "(no response)"
        if self._stream_frame is not None:
            self._fill_markdown(self._stream_frame, answer)
            if self._stream_reasoning.strip():
                thinking = self._thinking_widget(self._stream_reasoning.strip())
                self._stream_frame.pack_start(thinking, False, False, 0)
                self._stream_frame.reorder_child(thinking, 0)
                thinking.show_all()
        if self._stats_label is not None:
            self._stats_label.set_text(format_stats(self._stats))
        self.session_tokens += self._stats.get("total", 0)
        self._refresh_title()
        self._stream_frame = self._stream_label = self._stats_label = None

        self._db(memory_db.add_message, self.conversation_id, "assistant", answer,
                 compress_threshold=self.compress_threshold)

        convo = self._db(memory_db.get_conversation, self.conversation_id) or {}
        if not convo.get("title"):
            turns = self._db(memory_db.get_messages, self.conversation_id, limit=2)
            first_user = next((t["content"] for t in turns if t["role"] == "user"), "")
            threading.Thread(target=self._title_worker,
                             args=(self.conversation_id, first_user, answer),
                             daemon=True).start()

        self._refresh_title()
        if self.sidebar_revealer.get_reveal_child():
            self._refresh_sidebar()
        # Hand distillation to the daemon so the UI stays responsive.
        send_cmd(self.sock_path,
                 {"command": "analyze_memory", "conversation_id": self.conversation_id},
                 quiet=True)
        self._maybe_autoscroll()
        self._focus_input()
        return False


def take_screenshot_and_analyze(sock_path):
    time.sleep(0.3)
    region = subprocess.run(["slurp"], capture_output=True, text=True).stdout.strip()
    if not region:
        sys.exit(0)

    img_dir = os.path.expanduser("~/.local/share/ai_assistant/images")
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"cap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    subprocess.run(["grim", "-g", region, img_path], check=True)
    subprocess.run(f"wl-copy -t image/png < {img_path}", shell=True)
    send_cmd(sock_path, {"command": "analyze", "image_path": img_path})

# ==============================================================================
# 7. WAYBAR MODULE
# ==============================================================================
WAYBAR_TEXT = {
    "off": "[ AI: OFF ]",
    "starting": "[ AI: ... ]",
    "on": "[ AI: ON ]",
    "error": "[ AI: ERR ]",
}


def render_waybar(sock_path):
    """Print one JSON line for waybar's custom/ai module. Never fails loudly."""
    status = send_cmd(sock_path, {"command": "ai_mode_status"}, quiet=True, timeout=3.0)

    if status is None:
        payload = {
            "text": "[ AI: --- ]",
            "class": "off",
            "tooltip": "AI assistant daemon is not running (dinitctl start ai-assistant)",
        }
    else:
        phase = status.get("phase", "off")
        health = status.get("health") or {}
        lines = []
        if phase == "on":
            lines.append("Servers up on :9090 and :9191")
        elif phase == "error" and health:
            lines.append("35B :9090 %s   0.5B :9191 %s" % (
                "up" if health.get("big") else "DOWN",
                "up" if health.get("small") else "DOWN"))
            lines.append(status.get("error") or "")
        elif phase == "starting":
            lines.append("Loading models...")
        elif phase == "error":
            lines.append(status.get("error") or "Unknown error")
        else:
            lines.append("All AI subsystems stopped")

        if status.get("external"):
            lines.append("Externally started: " + ", ".join(status["external"]))
        lines.append(f"{status.get('memories', 0)} memories stored")
        if status.get("queued_jobs"):
            lines.append(f"{status['queued_jobs']} screenshot(s) queued")
        if status.get("session"):
            lines.append("Note session active")
        lines.append("Left click: toggle   Right click: chat")

        payload = {
            "text": WAYBAR_TEXT.get(phase, WAYBAR_TEXT["off"]),
            "class": phase,
            "tooltip": "\n".join(lines),
        }

    print(json.dumps(payload), flush=True)


# ==============================================================================
# 8. CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sway AI Screenshot, Chat & Nextcloud ODT Assistant")
    parser.add_argument("--daemon", action="store_true", help="Run the background daemon and bottom overlay bar")
    parser.add_argument("--gui", action="store_true", help="Open the centered session manager popup")
    parser.add_argument("--chat", action="store_true", help="Open the chat window backed by long-term memory")
    parser.add_argument("--capture", action="store_true", help="Capture screenshot (opens popup first if no session)")
    parser.add_argument("--sync", action="store_true", help="Manually push queued notes to Nextcloud")
    parser.add_argument("--say", type=str, help="Send a custom status event for the narrator to announce")
    parser.add_argument("--waybar", action="store_true", help="Print the waybar custom/ai module JSON")
    parser.add_argument("--ai-on", action="store_true", help="Enter AI Mode (start both llama.cpp servers)")
    parser.add_argument("--ai-off", action="store_true", help="Leave AI Mode and free all AI resources")
    parser.add_argument("--ai-toggle", action="store_true", help="Toggle AI Mode")
    parser.add_argument("--memory-stats", action="store_true", help="Print memory database statistics")
    parser.add_argument("--memory-compact", action="store_true", help="Merge duplicate memories and vacuum")
    args = parser.parse_args()

    config = load_config()
    sock_path = config["socket_path"]

    if args.daemon:
        if not os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
            print(
                "[AI Assistant] No WAYLAND_DISPLAY or DISPLAY is set, so the overlay bar "
                "cannot be created. When starting from a service manager, give the service "
                "the compositor's environment (e.g. an env-file containing "
                "WAYLAND_DISPLAY=wayland-1), or launch it from your sway config instead.",
                file=sys.stderr,
            )
            sys.exit(1)
        install_theme()
        daemon = AssistantDaemon(config)
        threading.Thread(target=daemon.start_server, daemon=True).start()
        Gtk.main()
        return

    if args.waybar:
        render_waybar(sock_path)
        return

    if args.ai_on or args.ai_off or args.ai_toggle:
        cmd = "ai_mode_on" if args.ai_on else ("ai_mode_off" if args.ai_off else "ai_mode_toggle")
        result = send_cmd(sock_path, {"command": cmd})
        print(json.dumps(result))
        return

    if args.memory_stats:
        print(json.dumps(send_cmd(sock_path, {"command": "memory_stats"}), indent=2))
        return

    if args.memory_compact:
        print(json.dumps(send_cmd(sock_path, {"command": "memory_compact"}, timeout=300.0), indent=2))
        return

    if args.sync:
        send_cmd(sock_path, {"command": "sync_queue"})
        return

    if args.say:
        send_cmd(sock_path, {"command": "announce", "message": args.say})
        return

    if args.chat:
        # ChatPopup wires its own destroy handler (it discards an empty thread
        # before quitting the loop).
        win = ChatPopup(config, sock_path)
        win.show_all()
        Gtk.main()
        return

    status = send_cmd(sock_path, {"command": "status"})
    active_session = status.get("session")
    unsynced_count = status.get("unsynced_count", 0)
    ai_phase = (status.get("ai") or {}).get("phase", "off")

    want_capture = args.capture or not args.gui

    # Capture needs the vision model; refuse instead of hanging on a dead port.
    if want_capture and ai_phase != "on":
        notify(f"AI Mode is {ai_phase.upper()} -- enable it in waybar before capturing.")
        return

    if args.gui or not active_session:
        win = SessionPopup(sock_path, current_session=active_session, unsynced_count=unsynced_count)
        win.connect("destroy", Gtk.main_quit)
        win.show_all()
        Gtk.main()
        if want_capture and not active_session and win.started_ok:
            take_screenshot_and_analyze(sock_path)
    elif want_capture:
        take_screenshot_and_analyze(sock_path)


if __name__ == "__main__":
    main()
