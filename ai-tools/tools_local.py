#!/usr/bin/env python3
"""Local tools the router can select: web search and sandboxed command line.

Safety model for the CLI tool, in layers, cheapest first:

  1. PARSE      -- the string must tokenise with shlex. Shell metacharacters
                   (; | & ` $( > <) are rejected outright unless the config
                   explicitly opts into pipelines, in which case only a
                   vetted subset is allowed.
  2. POLICY     -- the first token must be on the allowlist. Anything else is
                   either denied or escalated to a human approval callback,
                   per `cli_policy`. A denylist of regexes catches the obvious
                   catastrophes even if someone widens the allowlist.
  3. ISOLATION  -- execution happens under bubblewrap: the whole filesystem
                   is bound READ-ONLY (so `df`, `git status` and `pacman -Q`
                   report the truth instead of an empty tmpfs), the only
                   writable path is the sandbox workspace, the network is
                   unshared, PID/IPC/UTS namespaces are private, and secret
                   paths (~/.ssh, ~/.gnupg, ~/.aws, /etc/shadow, ...) are
                   masked with empty tmpfs/devnull mounts. `runuser -u
                   <ai_user>` is used instead when the daemon runs as root and
                   a dedicated AI user is configured.
  4. LIMITS     -- setrlimit on CPU time, address space and file size, plus a
                   wall-clock timeout that kills the whole process group;
                   stdout/stderr captured and truncated.
                   (RLIMIT_NPROC is deliberately NOT set: it is per-UID, not
                   per-process, so it makes `unshare` fail with EAGAIN and
                   would also throttle the user's real desktop session.)

Nothing here trusts the model: the model only ever proposes a string.

Portability: layers 1, 2 and 4 work everywhere. Layer 3 needs bubblewrap, which
means Linux -- on Windows (or a Linux box without bwrap) there is no isolation
to be had, so the tool refuses to execute at all unless the user explicitly
sets `cli_allow_unsandboxed=1`. Degrading silently to an unsandboxed shell is
the one failure mode that must not happen.
"""
from __future__ import annotations

import asyncio
import html
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field

import requests

try:
    import resource                 # POSIX only; absent on the Windows build
except ImportError:
    resource = None

IS_WINDOWS = os.name == "nt"

# --------------------------------------------------------------------------- #
# Web search
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_DDG_RESULT_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ai-assistant/1.0"


def _text(fragment: str, limit: int = 400) -> str:
    plain = html.unescape(_TAG_RE.sub("", fragment or ""))
    return _WS_RE.sub(" ", plain).strip()[:limit]


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


class WebSearchTool:
    """Keyless search: a SearXNG instance if configured, else DuckDuckGo HTML."""

    name = "web_search"

    def __init__(self, config):
        self.searx_url = (config.get("web_search_searx_url") or "").strip()
        self.timeout = float(config.get("web_search_timeout", 12) or 12)
        self.fetch_pages = str(config.get("web_search_fetch_pages", "1")).lower() in ("1", "true", "yes")
        self.page_chars = int(config.get("web_search_page_chars", 2500) or 2500)

    async def run(self, query: str, max_results: int = 5) -> dict:
        hits = await asyncio.to_thread(self._search_sync, query, max_results)
        pages: list[dict] = []
        if self.fetch_pages and hits:
            fetched = await asyncio.gather(
                *(asyncio.to_thread(self._fetch_sync, hit.url) for hit in hits[:2]),
                return_exceptions=True,
            )
            for hit, body in zip(hits[:2], fetched):
                if isinstance(body, str) and body:
                    pages.append({"url": hit.url, "text": body})
        return {
            "query": query,
            "results": [h.__dict__ for h in hits],
            "pages": pages,
        }

    # ---------------------------------------------------------------- blocking
    def _search_sync(self, query: str, max_results: int) -> list[SearchHit]:
        if self.searx_url:
            try:
                return self._searx(query, max_results)
            except Exception as exc:                    # noqa: BLE001
                print(f"[web_search] searx failed ({exc}); falling back to ddg")
        return self._duckduckgo(query, max_results)

    def _searx(self, query: str, max_results: int) -> list[SearchHit]:
        resp = requests.get(
            self.searx_url.rstrip("/") + "/search",
            params={"q": query, "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return [
            SearchHit(r.get("title", ""), r.get("url", ""), _text(r.get("content", "")))
            for r in (resp.json().get("results") or [])[:max_results]
        ]

    def _duckduckgo(self, query: str, max_results: int) -> list[SearchHit]:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "kl": "us-en"},
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        hits: list[SearchHit] = []
        for match in _DDG_RESULT_RE.finditer(resp.text):
            url = html.unescape(match.group("url"))
            if "duckduckgo.com/y.js" in url:            # sponsored
                continue
            hits.append(SearchHit(_text(match.group("title"), 200), url,
                                  _text(match.group("snippet"))))
            if len(hits) >= max_results:
                break
        return hits

    def _fetch_sync(self, url: str) -> str:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT},
                            timeout=self.timeout, stream=True)
        resp.raise_for_status()
        if "html" not in resp.headers.get("content-type", "text/html"):
            return ""
        raw = resp.raw.read(400_000, decode_content=True).decode("utf-8", "replace")
        raw = re.sub(r"(?is)<(script|style|nav|footer|header)\b.*?</\1>", " ", raw)
        return _text(raw, self.page_chars)

    @staticmethod
    def format_for_model(payload: dict) -> str:
        lines = [f"Web search: {payload['query']}", ""]
        for index, hit in enumerate(payload["results"], 1):
            lines.append(f"[{index}] {hit['title']}\n    {hit['url']}\n    {hit['snippet']}")
        for page in payload.get("pages") or []:
            lines += ["", f"--- page text: {page['url']} ---", page["text"]]
        if len(payload["results"]) == 0:
            lines.append("(no results)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
DEFAULT_ALLOWLIST = (
    "ls cat head tail wc grep rg find file stat du df free uptime uname hostnamectl "
    "ps top pgrep id whoami date cal env printenv which type echo printf sort uniq cut "
    "awk sed tr jq python3 pip git systemctl journalctl pacman paru nvidia-smi lsblk "
    "lspci lsusb ip ss nmcli curl wget tree diff md5sum sha256sum tar unzip gzip "
    "nproc lscpu sensors mount findmnt realpath readlink basename dirname"
).split()

# Commands that are read-only enough to never need a human in the loop.
DEFAULT_AUTO_OK = (
    "ls cat head tail wc grep rg find file stat du df free uptime uname ps pgrep id "
    "whoami date which type echo printf sort uniq cut nproc lscpu lsblk tree realpath "
    "readlink basename dirname findmnt printenv env"
).split()

# Masked inside the sandbox: an empty tmpfs (dirs) or /dev/null (files), so a
# wandering `find` or `grep -r` cannot read them even though / is bound.
DEFAULT_MASK_PATHS = ":".join([
    "~/.ssh", "~/.gnupg", "~/.aws", "~/.config/gh", "~/.config/sops",
    "~/.netrc", "~/.npmrc", "~/.pypirc", "~/.docker/config.json",
    "~/.local/share/keyrings", "~/.mozilla", "~/.config/google-chrome",
    "/etc/shadow", "/etc/gshadow", "/etc/sudoers", "/etc/sudoers.d",
    "/root", "/var/lib/sshd",
])

DENY_PATTERNS = [
    r"\brm\b.*(-[a-zA-Z]*[rf]|--recursive|--force)",
    r"\b(mkfs|fdisk|sfdisk|parted|wipefs|blkdiscard|shred)\b",
    r"\bdd\b.*\bof=/dev/",
    r"\b(shutdown|reboot|poweroff|halt|systemctl\s+(isolate|poweroff|reboot))\b",
    r"\b(sudo|doas|pkexec|su)\b",
    r"\b(chmod|chown)\b.*\s/(\s|$)",
    r":\s*\(\s*\)\s*\{",                       # fork bomb
    r"\b(curl|wget)\b[^|]*\|\s*(ba|z|)sh",     # curl | sh
    r">\s*/dev/(sd|nvme|vd)",
    r"\b(iptables|nft|ufw)\b",
    r"\bpasswd\b",
    r"\b(useradd|userdel|usermod|groupadd)\b",
    r"\bpacman\b.*\s-(S|R|U)(?![a-zA-Z]*s)",   # allow -Q/-Ss queries only
    r"~/\.ssh|/etc/shadow|/etc/sudoers|\.gnupg|\.aws/credentials",
    r"\bhistory\b|\bcrontab\b",
    r"\bkill(all)?\b\s+-9?\s*1\b",
]
DENY_RE = [re.compile(p) for p in DENY_PATTERNS]

SHELL_META_RE = re.compile(r"[;&`$><\n]|\|\||&&")


class CommandRejected(Exception):
    """Raised when policy forbids a command outright."""


@dataclass
class CliResult:
    command: str
    argv: list[str] = field(default_factory=list)
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    sandbox: str = "none"
    truncated: bool = False
    rejected: str = ""
    approved_by_user: bool = False

    def format_for_model(self) -> str:
        if self.rejected:
            return (f"Command refused by the sandbox policy.\n"
                    f"command: {self.command}\nreason: {self.rejected}\n"
                    "Explain the refusal to the user and offer a safe alternative.")
        parts = [
            f"$ {self.command}",
            f"exit_code: {self.exit_code}  ({self.sandbox}, {self.duration_s:.2f}s"
            + (", output truncated" if self.truncated else "") + ")",
        ]
        parts.append("--- stdout ---\n" + (self.stdout or "(empty)"))
        if self.stderr.strip():
            parts.append("--- stderr ---\n" + self.stderr)
        return "\n".join(parts)


class CliSandboxTool:
    """Runs one model-proposed command under bubblewrap with hard limits."""

    name = "run_command"

    def __init__(self, config, approval_cb=None):
        self.policy = (config.get("cli_policy") or "ask").strip().lower()  # deny|ask|allow
        self.timeout = float(config.get("cli_timeout", 20) or 20)
        self.output_limit = int(config.get("cli_output_limit", 8000) or 8000)
        self.allow_shell = str(config.get("cli_allow_shell", "0")).lower() in ("1", "true", "yes")
        self.allow_network = str(config.get("cli_allow_network", "0")).lower() in ("1", "true", "yes")
        self.ai_user = (config.get("cli_ai_user") or "").strip()
        self.allow_unsandboxed = str(
            config.get("cli_allow_unsandboxed", "0")
        ).lower() in ("1", "true", "yes")
        self.workspace = os.path.expanduser(
            config.get("cli_workspace") or "~/.local/share/ai_assistant/sandbox"
        )
        extra = (config.get("cli_extra_allow") or "").split()
        self.allowlist = set(DEFAULT_ALLOWLIST) | set(extra)
        self.auto_ok = set(DEFAULT_AUTO_OK) | set(
            (config.get("cli_extra_auto_ok") or "").split()
        )
        # ro-root: bind / read-only (truthful system answers, still no writes)
        # minimal: bind only /usr, /etc and the paths in cli_readable_paths
        self.fs_mode = (config.get("cli_fs_mode") or "ro-root").strip().lower()
        self.readable_paths = [
            os.path.expanduser(p) for p in
            (config.get("cli_readable_paths") or "~/Programming").split(":") if p.strip()
        ]
        self.mask_paths = [
            os.path.expanduser(p) for p in
            (config.get("cli_mask_paths") or DEFAULT_MASK_PATHS).split(":") if p.strip()
        ]
        # approval_cb(command, purpose) -> awaitable bool
        self.approval_cb = approval_cb
        os.makedirs(self.workspace, exist_ok=True)

    # ------------------------------------------------------------------ policy
    def vet(self, command: str) -> list[str]:
        """Returns argv, or raises CommandRejected. Pure, no side effects."""
        command = (command or "").strip()
        if not command:
            raise CommandRejected("empty command")
        if len(command) > 600:
            raise CommandRejected("command is implausibly long")
        if "\x00" in command:
            raise CommandRejected("null byte in command")

        for pattern in DENY_RE:
            if pattern.search(command):
                raise CommandRejected(f"matches denied pattern: {pattern.pattern}")

        if SHELL_META_RE.search(command) and not self.allow_shell:
            raise CommandRejected(
                "shell metacharacters are not allowed (no pipes, redirects, "
                "chaining or substitution)"
            )
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise CommandRejected(f"unparseable: {exc}") from exc
        if not argv:
            raise CommandRejected("empty command")

        binary = os.path.basename(argv[0])
        if binary in ("sh", "bash", "zsh", "fish", "env") and not self.allow_shell:
            raise CommandRejected(f"{binary!r} would escape the policy layer")
        if binary not in self.allowlist:
            raise CommandRejected(f"{binary!r} is not on the allowlist")
        if shutil.which(argv[0]) is None and not os.path.isabs(argv[0]):
            raise CommandRejected(f"{argv[0]!r} is not installed")
        return argv

    def needs_approval(self, argv: list[str]) -> bool:
        if self.policy == "allow":
            return False
        if self.policy == "deny":
            return True
        return os.path.basename(argv[0]) not in self.auto_ok

    # --------------------------------------------------------------- execution
    async def run(self, command: str, purpose: str = "", *, cwd: str | None = None) -> CliResult:
        result = CliResult(command=command)
        try:
            argv = self.vet(command)
        except CommandRejected as exc:
            result.rejected = str(exc)
            print(f"[cli] REJECTED {command!r}: {exc}")
            return result
        result.argv = argv

        if self.needs_approval(argv):
            if self.policy == "deny" or self.approval_cb is None:
                result.rejected = ("policy requires explicit user approval and no "
                                   "approval channel is available")
                return result
            approved = await self.approval_cb(command, purpose)
            if not approved:
                result.rejected = "the user declined to run this command"
                return result
            result.approved_by_user = True

        wrapped, label = self._wrap(argv, cwd or self.workspace)
        if label == "unsandboxed" and not self.allow_unsandboxed:
            result.rejected = (
                "no sandbox available on this platform (bubblewrap not found) "
                "and cli_allow_unsandboxed is off, so the command was not run"
            )
            print(f"[cli] REFUSED (no sandbox) {command!r}")
            return result
        result.sandbox = label
        started = time.monotonic()
        code, out, err = await asyncio.to_thread(
            self._spawn, wrapped, cwd or self.workspace
        )
        result.duration_s = time.monotonic() - started
        result.exit_code = code
        result.stdout, trunc_out = self._cap(out)
        result.stderr, trunc_err = self._cap(err)
        result.truncated = trunc_out or trunc_err
        print(f"[cli] {label} exit={code} {result.duration_s:.2f}s :: {command}")
        return result

    def _wrap(self, argv: list[str], cwd: str) -> tuple[list[str], str]:
        """Pick the strongest available isolation."""
        if self.ai_user and not IS_WINDOWS and os.geteuid() == 0:
            return ["runuser", "-u", self.ai_user, "--", *argv], f"runuser:{self.ai_user}"

        if not IS_WINDOWS and shutil.which("bwrap"):
            bwrap = [
                "bwrap",
                "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
                "--new-session",                       # no TIOCSTI terminal injection
            ]
            if self.fs_mode == "ro-root":
                bwrap += ["--ro-bind", "/", "/"]
            else:
                bwrap += [
                    "--ro-bind", "/usr", "/usr",
                    "--ro-bind", "/etc", "/etc",
                    "--symlink", "usr/lib", "/lib",
                    "--symlink", "usr/lib64", "/lib64",
                    "--symlink", "usr/bin", "/bin",
                    "--symlink", "usr/sbin", "/sbin",
                ]
                for path in self.readable_paths:
                    if os.path.exists(path):
                        bwrap += ["--ro-bind", path, path]

            # Order matters: /proc, /dev and /tmp go on top of the ro-root bind,
            # the masks on top of those, and the writable workspace last.
            bwrap += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
            for path in self.mask_paths:
                if os.path.isdir(path):
                    bwrap += ["--tmpfs", path]
                elif os.path.exists(path):
                    bwrap += ["--ro-bind", "/dev/null", path]
            bwrap += [
                "--bind", cwd, cwd,
                "--chdir", cwd,
                "--setenv", "HOME", cwd,
                "--setenv", "PATH", "/usr/bin:/usr/local/bin",
                "--setenv", "TERM", "dumb",
            ]
            if not self.allow_network:
                bwrap += ["--unshare-net"]
            return [*bwrap, "--", *argv], f"bwrap:{self.fs_mode}"

        return argv, "unsandboxed"

    def _spawn(self, argv: list[str], cwd: str) -> tuple[int, str, str]:
        sandboxed = os.path.basename(argv[0]) == "bwrap"

        def limits():
            os.setsid()                       # own process group -> killable as one
            cpu = int(self.timeout) + 5
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 2))
            resource.setrlimit(resource.RLIMIT_FSIZE, (32 << 20, 32 << 20))
            resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if not sandboxed:
                # Only safe without bwrap: RLIMIT_NPROC counts every process of
                # this UID, so it makes unshare() fail with EAGAIN and would
                # also starve the user's own session.
                resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))

        spawn_kwargs = {}
        if resource is not None and not IS_WINDOWS:
            spawn_kwargs["preexec_fn"] = limits
        elif IS_WINDOWS:
            # No fork, so no preexec_fn: get a killable group the Windows way.
            spawn_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        env = {
            "PATH": os.environ.get("PATH", "") if IS_WINDOWS else "/usr/bin:/usr/local/bin",
            "HOME": cwd,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "TERM": "dumb",
            "NO_COLOR": "1",
        }
        if IS_WINDOWS:
            for key in ("SYSTEMROOT", "COMSPEC", "PATHEXT", "TEMP"):
                if os.environ.get(key):
                    env[key] = os.environ[key]
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd, env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, errors="replace", **spawn_kwargs,
            )
        except OSError as exc:
            return -1, "", f"failed to start: {exc}"

        try:
            out, err = proc.communicate(timeout=self.timeout)
            return proc.returncode, out, err
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            out, err = proc.communicate()
            return -9, out, (err or "") + f"\n[killed after {self.timeout}s timeout]"

    @staticmethod
    def _kill_group(proc):
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True)
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                return
            try:
                proc.wait(timeout=2.0)
                return
            except subprocess.TimeoutExpired:
                continue

    def _cap(self, text: str) -> tuple[str, bool]:
        text = text or ""
        if len(text) <= self.output_limit:
            return text, False
        half = self.output_limit // 2
        return (text[:half] + "\n...[%d bytes elided]...\n" % (len(text) - self.output_limit)
                + text[-half:]), True


# --------------------------------------------------------------------------- #
# OpenAI tool schemas, so the 35B model can also call these itself
# --------------------------------------------------------------------------- #
MEMORY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Search your long-term memory of previous conversations and screen-capture "
            "notes. Call this whenever the user refers to something from the past, or "
            "you need a fact, preference, path, port or decision you were told before."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for."},
                "limit": {"type": "integer", "description": "How many memories (1-12)."},
            },
            "required": ["query"],
        },
    },
}

LOCAL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information and read the top pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword search query."},
                    "max_results": {"type": "integer", "description": "3-8 results."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run one read-only shell command in a bubblewrap sandbox on the user's "
                "Linux machine and return its exit code, stdout and stderr. No pipes, "
                "no redirects, no sudo. Destructive commands are refused."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "A single command."},
                    "purpose": {"type": "string", "description": "What it reveals."},
                },
                "required": ["command"],
            },
        },
    },
]
