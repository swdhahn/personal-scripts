# My Personal Scripts Repo

I created this to save time with my workflow and to sync my workflow across all my computers.

## .config file
There is an example `.config` file named `.config-example`.
Rename any config file to `.config` if you want to use it. There are a few config options supported:
 - `default_project_path`: set this equal to the root directory of your projects. This allows for fast fuzzy finding for all your different projects.
 - `llama_url`: the local endpoint for your `llama.cpp` server (defaults to `http://127.0.0.1:9090/v1/chat/completions`).
 - `nextcloud_url`: the base URL of your Nextcloud instance.
 - `nextcloud_user`: your Nextcloud username for WebDAV access.
 - `nextcloud_token`: a dedicated Nextcloud App Password (generated under *Settings -> Security -> Devices & sessions*).
 - `session_history_dir`: local directory where active session histories and captured screenshots are logged (defaults to `~/.local/share/ai_assistant/sessions`).
 - `socket_path`: UNIX socket path used for communication between the background daemon and the popup client (defaults to `/tmp/ai_assistant.sock`).

**AI Mode** (starting and stopping the local model servers):
 - `llama_server_cmd` / `llama_small_server_cmd`: the full `llama-server` command lines to run, one per line. These are what AI Mode starts; tokens are `~`-expanded individually, so model paths may use `~`.
 - `llama_health_url` / `llama_small_health_url`: `/health` endpoints polled until each server reports ready.
 - `ai_mode_state_file`: where the current AI Mode state and server PIDs are recorded (defaults to `~/.local/share/ai_assistant/ai_mode.json`).
 - `ai_mode_start_timeout`: seconds to wait for the models to load before reporting an error (defaults to `180`).
 - `waybar_signal`: the waybar real-time signal to raise after a toggle so the module repaints instantly (defaults to `8`).

**Chat memory**:
 - `memory_db_path`: SQLite database of conversations and distilled memories (defaults to `~/.local/share/ai_assistant/memory.db`).
 - `memory_top_k`: how many remembered items are injected into each chat turn (defaults to `6`).
 - `memory_recency_half_life_days`: how fast a memory's relevance decays (defaults to `30`).
 - `memory_min_importance`: memories scored below this are not auto-injected (defaults to `0.25`).
 - `memory_compress_threshold`: chat turns larger than this many bytes are stored zlib-compressed (defaults to `2048`).
 - `llama_context`: optional override for the context-window figure in the chat header. Normally left blank — the window is read from the server's `/props`, falling back to the `-c` flag of `llama_server_cmd`.
 - `chat_system_prompt`: the system prompt used by the chat window.

### tmux-dev.sh

This opens a fuzzy finder starting from the `~/Programming` folder, this will be moved to a `.config` file soon.
It then opens a tmux session with nvim on the left and 2 panels on the right. The bottom right panel I use for running cmake and other build commands while the top right is typically used for everything else, from git to man pages.

### ai/ai_assistant.py

A self-bootstrapping, single-file Wayland AI screenshot and note-taking assistant designed for Sway, Artix Linux (`dinit`), a local `llama.cpp` vision server, and Nextcloud.

On its first run on a new machine, the script automatically creates an isolated `.venv` (inheriting system GTK3 and `gtk-layer-shell` bindings), installs its required Python dependencies (`requests`, `odfpy`, `pygments`), and checks for required Wayland utilities (`grim`, `slurp`, `wl-clipboard`).

#### Features & Workflow
* **Background Daemon (`--daemon`)**: Runs continuously as a user service and renders a non-intrusive `gtk-layer-shell` status overlay bar at the bottom of the screen to display live progress without interfering with Waybar.
* **Kanagawa Paper theming**: Every GTK surface (overlay bar, session manager, chat window) uses the `kanagawa-paper` ink palette, matching the nvim colourscheme and waybar.
* **Centered Session Manager (`--gui`)**: Opens a floating, centered GTK popup window to start a new session, set the target `.odt` file path (remote Nextcloud path or local file), and customize the analysis/generation prompt.
* **Screenshot & Note Capture (`--capture`)**: Uses `slurp` and `grim` to capture a screen region (copying it to the clipboard as well). If no session is active, it prompts to create one first. The image and rolling session context are sent to the `llama.cpp` server on port `9090`, and the generated output is appended directly to the target `.odt` document on Nextcloud via WebDAV. Each note is also written into the memory database, so the chat window can recall what was captured.
* **AI Mode (`--ai-toggle`)**: A master switch, surfaced as a waybar module. Turning it on launches both `llama-server` processes from the config and waits for their `/health` endpoints. Turning it off kills both process groups, unloads the Kokoro TTS model from RAM, drains every work queue and silences the narrator — so no AI subsystem holds memory or VRAM while you are doing something else (gaming, for instance). Capture and chat refuse politely while it is off instead of hanging on a dead port.
* **Chat with long-term memory (`--chat`, `$mod+a`)**: A GTK chat window that streams replies from the 35B model. Every exchange is stored in SQLite, then distilled in the background into tagged, importance-scored memories. Each turn automatically injects the top-ranked relevant memories, and the model can additionally call a `memory_search` tool to look things up for itself. The window shows exactly which memories were recalled, so you can tell recall from hallucination.
  * **Sidebar** (hamburger, `Ctrl+B`): conversation history at the top with its own search, and the memory list below with its search bar at the bottom. Both searches are semantic-ish — the 0.5B model expands the query with related terms, which then feed the same FTS ranking (there is no embedding endpoint on these servers). Conversation search matches titles, rolling summaries and message bodies.
  * **Right-click a conversation** to delete it (asks first). **Click a memory** to open it for editing — content, kind, importance and tags — or to delete it; edits are re-indexed for search immediately.
  * **Markdown is rendered**, not shown raw: headings, lists, quotes, links, inline code, and fenced code blocks syntax-highlighted with Pygments using the theme's own `syn` colours. Long code lines scroll sideways rather than wrapping.
  * **Titles come from the 0.5B model** on port 9191 — fast, and it keeps the 35B free.
  * **Token usage and speed**: each reply carries a line like `549 tok · 20.2 tok/s · 27.2s · prompt 433 tok @ 343 tok/s · 982 total`, updating live while the answer streams. The header shows `ctx <prompt>/<window>` and a running session total. Numbers come from llama.cpp's own `usage` and `timings` (requested with `stream_options.include_usage` and `timings_per_token`), and the context window is read from the server's `/props` rather than the config, since the two can disagree.
  * **Thinking dropdown**: when the model reasons before answering, a collapsed `Thinking (N characters)` expander sits above the reply. It holds the raw chain-of-thought for that turn and is kept in memory only — reopening a conversation later will not show it.
  * **Threads are only saved once something is said.** Opening the window, or hitting New Chat and closing, leaves no empty rows; pre-existing empty rows are purged at startup.
  * The transcript stays where you put it: scrolling up during a stream detaches the view and a jump-to-latest pill appears until you return to the bottom.

| key | action | | key | action |
|---|---|---|---|---|
| `Enter` | send | | `Ctrl+K` | compact memories |
| `Shift+Enter` | newline | | `Ctrl+End` | jump to latest |
| `Ctrl+N` | new chat | | `Ctrl+L` | focus message box |
| `Ctrl+B` | toggle sidebar | | `Esc` / `Ctrl+W` | close |
| `Ctrl+F` | search conversations | | `Ctrl+Shift+F` | search memories |

#### Usage
```bash
# Start the background daemon and bottom overlay bar
python ai/ai_assistant.py --daemon

# Capture a screenshot and analyze (prompts to create a session if none is active)
python ai/ai_assistant.py --capture

# Open the centered Session Manager GUI to start, update, or end a session
python ai-tools/ai_assistant.py --gui

# Open the chat window (backed by the memory database)
python ai-tools/ai_assistant.py --chat

# AI Mode: start/stop both llama.cpp servers and free all AI resources
python ai-tools/ai_assistant.py --ai-toggle
python ai-tools/ai_assistant.py --ai-on
python ai-tools/ai_assistant.py --ai-off

# Inspect or compact the memory database
python ai-tools/ai_assistant.py --memory-stats
python ai-tools/ai_assistant.py --memory-compact

# Waybar module (JSON on stdout)
python ai-tools/ai_assistant.py --waybar
```

#### Memory database

`ai-tools/memory_db.py` is a standalone, stdlib-only SQLite layer. Chats are stored in
`conversations` / `messages` (bodies zlib-compressed past a threshold), then distilled into
`memories` — one atomic fact per row with a `kind`, tags, an importance score, and a
`superseded_by` chain for deduplication. Retrieval ranks FTS5 `bm25` relevance against
importance, recency decay, and how often a memory has been used before:

```
score = relevance x (0.5 + 0.5 x importance) x exp(-age_days / half_life) x (1 + 0.1 x ln(1 + access_count))
```

`--memory-compact` merges duplicate and near-duplicate rows (exact match, or an 85% token
overlap across at least four tokens) onto the newest survivor and vacuums the database.

Conversations are created lazily, so an unused chat window never leaves a row behind;
`purge_empty_conversations()` also runs when the chat window opens. `search_conversations()`
ranks whole threads (best message match, title/summary overlap, recency), and
`update_memory()` / `delete_memory()` keep the FTS mirror and tag table in sync.

#### Waybar module

```json
"custom/ai": {
    "exec": "python ~/Programming/scripts/ai-tools/ai_assistant.py --waybar",
    "return-type": "json",
    "interval": 5,
    "signal": 8,
    "on-click": "python ~/Programming/scripts/ai-tools/ai_assistant.py --ai-toggle",
    "on-click-right": "python ~/Programming/scripts/ai-tools/ai_assistant.py --chat"
}


#### Dependencies (Artix Linux)

The script automatically sets up its own isolated Python virtual environment (`.venv`) and installs `requests`, `odfpy`, `kokoro-onnx` and `pygments`. However, you need the underlying Wayland utilities, GTK3 libraries, and GObject introspection packages installed on your system.

Install them via `pacman`:

```bash
sudo pacman -S --needed \
    grim \
    slurp \
    wl-clipboard \
    gtk3 \
    gtk-layer-shell \
    python-gobject \
    python-pip
