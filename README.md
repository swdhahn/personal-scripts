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

### tmux-dev.sh

This opens a fuzzy finder starting from the `~/Programming` folder, this will be moved to a `.config` file soon.
It then opens a tmux session with nvim on the left and 2 panels on the right. The bottom right panel I use for running cmake and other build commands while the top right is typically used for everything else, from git to man pages.

### ai/ai_assistant.py

A self-bootstrapping, single-file Wayland AI screenshot and note-taking assistant designed for Sway, Artix Linux (`dinit`), a local `llama.cpp` vision server, and Nextcloud.

On its first run on a new machine, the script automatically creates an isolated `.venv` (inheriting system GTK3 and `gtk-layer-shell` bindings), installs its required Python dependencies (`requests`, `odfpy`), and checks for required Wayland utilities (`grim`, `slurp`, `wl-clipboard`).

#### Features & Workflow
* **Background Daemon (`--daemon`)**: Runs continuously as a user service and renders a non-intrusive `gtk-layer-shell` status overlay bar at the bottom of the screen to display live progress without interfering with Waybar.
* **Centered Session Manager (`--gui`)**: Opens a floating, centered GTK popup window to start a new session, set the target `.odt` file path (remote Nextcloud path or local file), and customize the analysis/generation prompt.
* **Screenshot & Note Capture (`--capture`)**: Uses `slurp` and `grim` to capture a screen region (copying it to the clipboard as well). If no session is active, it prompts to create one first. The image and rolling session context are sent to the `llama.cpp` server on port `9090`, and the generated output is appended directly to the target `.odt` document on Nextcloud via WebDAV.

#### Usage
```bash
# Start the background daemon and bottom overlay bar
python ai/ai_assistant.py --daemon

# Capture a screenshot and analyze (prompts to create a session if none is active)
python ai/ai_assistant.py --capture

# Open the centered Session Manager GUI to start, update, or end a session
python ai/ai_assistant.py --gui


#### Dependencies (Artix Linux)

The script automatically sets up its own isolated Python virtual environment (`.venv`) and installs `requests` and `odfpy`. However, you need the underlying Wayland utilities, GTK3 libraries, and GObject introspection packages installed on your system.

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
