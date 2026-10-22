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

        # Ensure all required pip packages (including kokoro-onnx) are present
        check_cmd = [venv_python, "-c", "import requests, odf, kokoro_onnx, soundfile"]
        if subprocess.call(check_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            print("[AI Assistant] Installing Python packages (requests, odfpy, kokoro-onnx, soundfile)...")
            subprocess.check_call([
                venv_python, "-m", "pip", "install", "--upgrade",
                "pip", "requests", "odfpy", "kokoro-onnx", "soundfile"
            ])

        os.execv(venv_python, [venv_python, os.path.abspath(__file__)] + sys.argv[1:])

bootstrap_env()

# ==============================================================================
# 2. IMPORTS
# ==============================================================================
import re
import io
import json
import time
import queue
import socket
import base64
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

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GtkLayerShell', '0.1')
from gi.repository import Gtk, Gdk, GLib, GtkLayerShell

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
    }

    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("--") or line.startswith("//"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    config[k.strip()] = os.path.expanduser(v.strip())
    return config

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

        css = b"""
        window {
            background-color: rgba(30, 30, 46, 0.94);
            border: 2px solid #89b4fa;
            border-radius: 8px;
        }
        label {
            color: #cdd6f4;
            font-family: monospace;
            font-size: 14px;
            font-weight: bold;
            padding: 10px 24px;
        }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.label = Gtk.Label(label="AI Assistant Ready")
        self.add(self.label)
        self.hide_timer = None

    def show_message(self, text, timeout=0):
        GLib.idle_add(self._update_ui, text, timeout)

    def _update_ui(self, text, timeout):
        if self.hide_timer:
            GLib.source_remove(self.hide_timer)
            self.hide_timer = None
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

        # FIFO Processing Queue for screenshots
        self.analysis_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
        self.worker_thread.start()

        # Dedicated FIFO Queue for Small LLM + Kokoro TTS announcements
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

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
                if self.session:
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
        """Queues a raw operational event to be converted into spoken sentences."""
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

    def _play_kokoro_tts(self, text):
        """Synthesizes speech via external tts_url (port 9292) or built-in local Kokoro ONNX engine."""
        voice = self.config.get("tts_voice", "am_michael")
        speed = float(self.config.get("tts_speed", "0.95"))
        tts_url = self.config.get("tts_url", "http://127.0.0.1:9292/v1/audio/speech")

        # 1. Try external Kokoro server on port 9292 first if running
        try:
            r = requests.post(
                tts_url,
                json={"model": "kokoro", "input": text, "voice": voice, "speed": speed, "response_format": "wav"},
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

        # 2. Built-in Local CPU Kokoro ONNX Engine (Zero VRAM, No external server required)
        try:
            kokoro = self._ensure_kokoro_loaded()
            samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
            wav_buf = io.BytesIO()
            sf.write(wav_buf, samples, sample_rate, format="WAV")
            wav_buf.seek(0)

            p = subprocess.Popen(["pw-play", "-"], stdin=subprocess.PIPE)
            p.communicate(input=wav_buf.read())
        except Exception as e:
            print(f"Kokoro TTS playback failed: {e}", file=sys.stderr)

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
                    "queued_jobs": pending_jobs
                }).encode("utf-8"))

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
                if not self.session:
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
# 6. CLIENT: POPUP GUI & MANUAL SYNC BUTTON
# ==============================================================================
def send_cmd(sock_path, payload):
    if not os.path.exists(sock_path):
        subprocess.run(["notify-send", "AI Assistant", "Daemon not running! Start it or check dinitctl."])
        sys.exit(1)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    client.sendall(json.dumps(payload).encode("utf-8"))
    resp = client.recv(65536).decode("utf-8")
    client.close()
    return json.loads(resp)


class SessionPopup(Gtk.Window):
    def __init__(self, sock_path, current_session=None, unsynced_count=0):
        super().__init__(title="AI Assistant Session")
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
            
        header = Gtk.Label(label=status_text, xalign=0)
        vbox.pack_start(header, False, False, 0)

        vbox.pack_start(
            Gtk.Label(label="Nextcloud ODT Path (e.g. Documents/notes.odt) or Local Path:", xalign=0),
            False, False, 0
        )
        self.doc_entry = Gtk.Entry()
        self.doc_entry.set_text(current_session["doc_path"] if current_session else "Notes/session_notes.odt")
        vbox.pack_start(self.doc_entry, False, False, 0)

        vbox.pack_start(Gtk.Label(label="Analysis / Generation Prompt:", xalign=0), False, False, 0)
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

        start_btn = Gtk.Button(label="Start New Session" if not current_session else "Update Session")
        start_btn.connect("clicked", self.on_save_session, bool(current_session))
        btn_box.pack_start(start_btn, True, True, 0)

        if current_session:
            sync_btn = Gtk.Button(label=f"Sync Queue ({unsynced_count})")
            sync_btn.connect("clicked", self.on_sync_queue)
            if unsynced_count == 0:
                sync_btn.set_sensitive(False)
            btn_box.pack_start(sync_btn, False, False, 0)

            end_btn = Gtk.Button(label="End Session")
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
# 7. CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sway AI Screenshot & Nextcloud ODT Assistant")
    parser.add_argument("--daemon", action="store_true", help="Run the background daemon and bottom overlay bar")
    parser.add_argument("--gui", action="store_true", help="Open the centered session manager popup")
    parser.add_argument("--capture", action="store_true", help="Capture screenshot (opens popup first if no session)")
    parser.add_argument("--sync", action="store_true", help="Manually push queued notes to Nextcloud")
    parser.add_argument("--say", type=str, help="Send a custom status event for the narrator to announce")
    args = parser.parse_args()

    config = load_config()
    sock_path = config["socket_path"]

    if args.daemon:
        daemon = AssistantDaemon(config)
        threading.Thread(target=daemon.start_server, daemon=True).start()
        Gtk.main()
    elif args.sync:
        send_cmd(sock_path, {"command": "sync_queue"})
    elif args.say:
        send_cmd(sock_path, {"command": "announce", "message": args.say})
    else:
        status = send_cmd(sock_path, {"command": "status"})
        active_session = status.get("session")
        unsynced_count = status.get("unsynced_count", 0)

        want_capture = args.capture or not (args.gui or args.daemon)

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
