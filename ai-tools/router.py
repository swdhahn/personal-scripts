#!/usr/bin/env python3
"""The "Jev-like" decision engine: the 0.5B model as a zero-shot router.

Every prompt passes through here before a single token of prose is generated.
Stage 1 is one grammar-constrained JSON object answering *what the system must
do*; stage 2 (only when a tool was selected) is a second constrained object
carrying that tool's parameters.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

from constrained import ConstrainedDecoder


# --------------------------------------------------------------------------- #
# 1. The decision vocabulary -- these Enums ARE the grammar
# --------------------------------------------------------------------------- #
class Route(str, Enum):
    """Which brain answers."""
    SMALL = "SMALL"      # chit-chat, one-liners, reformatting, trivia
    LARGE = "LARGE"      # reasoning, code, long-form, anything non-trivial


class YesNo(str, Enum):
    """For binary sub-questions, which a 0.5B answers far better than n-way."""
    NO = "NO"
    YES = "YES"


class MemoryAction(str, Enum):
    """Whether SQLite has to be consulted or written."""
    NONE = "NONE"        # self-contained prompt
    RECALL = "RECALL"    # "what did we talk about yesterday?", "my API key"
    STORE = "STORE"      # "remember that...", "from now on..."
    BOTH = "BOTH"

    @classmethod
    def combine(cls, past: bool, store: bool) -> "MemoryAction":
        if past and store:
            return cls.BOTH
        if past:
            return cls.RECALL
        if store:
            return cls.STORE
        return cls.NONE


class ToolAction(str, Enum):
    """Which local tool, if any, has to run first."""
    NONE = "NONE"
    TOOL_WEB = "TOOL_WEB"    # live/current information, docs, prices, news
    TOOL_CLI = "TOOL_CLI"    # inspect or act on this machine via the sandbox

    @classmethod
    def combine(cls, web: bool, cli: bool) -> "ToolAction":
        # When both fire, the local answer wins: it is concrete, free, offline
        # and bounded, where a web search is none of those.
        if cli:
            return cls.TOOL_CLI
        if web:
            return cls.TOOL_WEB
        return cls.NONE


class Urgency(str, Enum):
    """Drives the live-commentary cadence."""
    FAST = "FAST"        # answer expected immediately, stay quiet
    SLOW = "SLOW"        # long job, narrate progress


# The field spec handed to the grammar generator. Key order is the decode
# order, so it is also the model's reasoning order: tool -> memory -> route.
# `default` is what a malformed/missing field falls back to. They are chosen so
# that a confused router degrades toward "do nothing risky, ask the good model".
DECISION_FIELDS = {
    "tool": {"enum": ToolAction, "default": ToolAction.NONE.value},
    "memory": {"enum": MemoryAction, "default": MemoryAction.NONE.value},
    "route": {"enum": Route, "default": Route.LARGE.value},
    "urgency": {"enum": Urgency, "default": Urgency.FAST.value},
}

WEB_PARAM_FIELDS = {
    "query": {"type": "string"},
    "max_results": {"type": "integer", "default": 5},
}

CLI_PARAM_FIELDS = {
    "command": {"type": "string"},
    "purpose": {"type": "string"},
    "writes_files": {"type": "boolean"},
}

# A 0.5B instruct model cannot do a four-field joint classification from a
# prose rubric: it anchors hard on whichever option is listed first and
# ignores the discriminative content. Measured on Qwen2.5-0.5B-Instruct, the
# joint-JSON router returned TOOL_CLI + RECALL for literally every prompt,
# "hello there" included.
#
# What does work on a model this size: one decision per call, one token of
# output, NONE listed first, and few-shot examples carrying the actual signal.
# Each call is a single constrained token, so three of them are still cheaper
# than one 40-token JSON object.
# Tool selection is two binary questions, not one three-way choice, for the
# same reason memory is (below). Measured on Qwen2.5-0.5B-Instruct, a 24-entry
# three-way list scored 13/20 held-out and was dominated by example ordering:
# broadening TOOL_WEB coverage silently destroyed NONE precision. Split into
# balanced yes/no questions each classifier stays stable as examples are added.
WEB_SYSTEM = """Does answering this message require looking something up on the
internet?

YES - it asks about the current state of the outside world: who currently holds
      a position, what something costs now, release dates and schedules, news,
      what people are currently saying, or any fact that may have changed since
      you were trained.
NO  - you can answer it from knowledge, reasoning or the conversation itself:
      explanations, writing or debugging code, maths, definitions, opinions,
      small talk, or anything about this computer."""

# Interleaved deliberately: a flat block of YES followed by a block of NO makes
# a model this size answer with whichever label it saw last.
WEB_EXAMPLES = [
    ("hello there", "NO"),
    ("what is the latest llama.cpp release version?", "YES"),
    ("explain how constrained decoding works", "NO"),
    ("who is the mayor of Berlin", "YES"),
    ("write me a python script that parses a GBNF grammar", "NO"),
    ("what is the current price of an RX 9070 XT", "YES"),
    ("my asyncio task never cancels, what am I doing wrong", "NO"),
    ("when does the next Debian stable come out", "YES"),
    ("compare ROCm and CUDA for inference", "NO"),
    ("what do people think of the new Framework laptop", "YES"),
    ("what is the difference between a mutex and a semaphore", "NO"),
    ("any news about the ROCm 7 release", "YES"),
    ("how much disk space is left on this machine?", "NO"),
    ("look up the Kokoro TTS licence terms", "YES"),
    ("what did we talk about yesterday?", "NO"),
]

CLI_SYSTEM = """Does answering this message require reading THIS computer's own
state by running a command on it?

YES - it asks about this machine: its files, folders, disk usage, memory, load,
      processes, services, installed packages, git repositories, kernel,
      hardware or configuration.
NO  - it is about anything else: general knowledge, code, explanations, the
      outside world, the conversation, or small talk."""

CLI_EXAMPLES = [
    ("hello there", "NO"),
    ("how much disk space is left on this machine?", "YES"),
    ("explain how constrained decoding works", "NO"),
    ("what kernel am I on", "YES"),
    ("write me a python script that parses a GBNF grammar", "NO"),
    ("which git branch is this repo on", "YES"),
    ("what is the latest llama.cpp release version?", "NO"),
    ("is pipewire installed", "YES"),
    ("my asyncio task never cancels, what am I doing wrong", "NO"),
    ("how much RAM is free", "YES"),
    ("what did we talk about yesterday?", "NO"),
    ("is the ai-assistant service running?", "YES"),
    ("what is the difference between a thread and a process", "NO"),
    ("show me the biggest folders in my home directory", "YES"),
    ("remind me which editor I said I use", "NO"),
]

# Memory is asked as two binary questions rather than one four-way choice.
# "remember that X" collapses into RECALL on a 0.5B when STORE and RECALL are
# options in the same list -- the word "remember" dominates. Split apart, each
# question is a yes/no the model gets right.
PAST_SYSTEM = """Does answering this message require knowing what was said in an
EARLIER conversation -- a fact, preference, name, port, path or decision the
user stated before?

YES - it refers back to something previously said or agreed.
NO  - it is self-contained and can be answered without any history.

A question about the present state of the machine, of the world, or of
mathematics is NO, even when it begins with "what" or "which". Only a
reference to the shared past is YES."""

PAST_EXAMPLES = [
    ("hello there", "NO"),
    ("how much disk space is left?", "NO"),
    ("write me a python script", "NO"),
    ("what is 17 times 4", "NO"),
    ("remember that I prefer tabs over spaces", "NO"),
    ("from now on always use British spelling", "NO"),
    # The failure class this classifier falls into without them: a question
    # about present or external state, which merely *looks* like a lookup.
    ("what version of python is installed", "NO"),
    ("how many CPU cores does this box have", "NO"),
    ("what is the capital of France", "NO"),
    ("what time is it", "NO"),
    ("which window manager is running", "NO"),
    ("what is the newest ROCm release", "NO"),
    ("what did we talk about yesterday?", "YES"),
    ("what was that port number I gave you", "YES"),
    ("which project was I working on last week", "YES"),
    ("what did I say my nextcloud url was", "YES"),
]

STORE_SYSTEM = """Is the user TELLING the assistant something to remember for
future conversations -- a durable preference, a standing instruction, or a
lasting fact about themselves or their setup?

YES - it states something that should be kept for later.
NO  - it is a question, a request for work, or small talk."""

STORE_EXAMPLES = [
    ("hello there", "NO"),
    ("what version of python is installed", "NO"),
    ("explain how constrained decoding works", "NO"),
    ("what did we talk about yesterday?", "NO"),
    ("how much disk space is left?", "NO"),
    ("write me a python script", "NO"),
    ("what was that port number I gave you", "NO"),
    ("remember that I prefer tabs over spaces", "YES"),
    ("from now on always use British spelling", "YES"),
    ("my llama server lives on port 9090", "YES"),
    ("I use Artix with dinit, not systemd", "YES"),
]

ROUTE_SYSTEM = """Decide which model should answer this message.

SMALL - trivial: any greeting or farewell however casual, thanks, a bare
        acknowledgement ("ok", "yep", "sure"), yes/no, one-line facts, simple
        arithmetic, simple rewording.
LARGE - everything else: code, debugging, explanations, reasoning, planning,
        long answers, or anything where a wrong answer matters."""

ROUTE_EXAMPLES = [
    ("hello there", "SMALL"),
    ("thanks, that worked", "SMALL"),
    ("good morning", "SMALL"),
    ("what is 2 + 2", "SMALL"),
    ("ok", "SMALL"),
    ("hey", "SMALL"),
    ("ta", "SMALL"),
    ("sure", "SMALL"),
    ("night", "SMALL"),
    ("how many metres in a kilometre", "SMALL"),
    ("write me a python script that parses a GBNF grammar", "LARGE"),
    ("explain how constrained decoding works", "LARGE"),
    ("why is my bwrap sandbox failing with EAGAIN", "LARGE"),
    ("what did we talk about yesterday?", "LARGE"),
    ("summarise this error and suggest a fix", "LARGE"),
]

WEB_PARAM_SYSTEM = """Extract web-search parameters from the user's request.
`query` is a terse search-engine query: keywords only, no question words, no
quotes, no site: operators unless the user named a site. `max_results` is 3-8.
Reply with JSON only."""

WEB_PARAM_EXAMPLES = [
    ("what is the latest llama.cpp release version?",
     '{"query": "llama.cpp latest release version", "max_results": 5}'),
    ("what do people think of the new Framework laptop",
     '{"query": "Framework laptop review", "max_results": 6}'),
]

CLI_PARAM_SYSTEM = """Extract a single shell command that answers the user's
question about this Linux (Arch/Artix) machine.

Rules:
  * ONE command. No pipes, no redirects, no &&, no ;, no sudo, no backticks.
  * The command must only READ. Never install, update, remove, start, stop or
    write anything -- the question is always "what is the state of this
    machine", never "change this machine".
  * This is Arch/Artix Linux: use `pacman -Q` to query packages, never
    apt/apt-get/dnf/yum. Use `dinitctl` or `systemctl status`, not `service`.
  * Read-only only: ls, cat, head, tail, grep, find, df, du, free, ps, uname,
    git, systemctl status, pacman -Q, lsblk, nproc, uptime.
  * `purpose` is a short description of what the command reveals.
  * `writes_files` is true only if the command creates or modifies files.
Reply with JSON only."""

CLI_PARAM_EXAMPLES = [
    ("how much disk space is left?",
     '{"command": "df -h", "purpose": "free space per filesystem", "writes_files": false}'),
    ("is pipewire installed?",
     '{"command": "pacman -Q pipewire", "purpose": "whether the package is installed", "writes_files": false}'),
    ("what kernel am I on?",
     '{"command": "uname -r", "purpose": "running kernel version", "writes_files": false}'),
    ("what is using the most space in my home directory?",
     '{"command": "du -h --max-depth=1 /home/swdhahn", "purpose": "folder sizes in home", "writes_files": false}'),
    ("which branch is this repo on?",
     '{"command": "git -C . branch --show-current", "purpose": "current git branch", "writes_files": false}'),
]


# --------------------------------------------------------------------------- #
# 2. The decision object
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    route: Route = Route.LARGE
    memory: MemoryAction = MemoryAction.NONE
    tool: ToolAction = ToolAction.NONE
    urgency: Urgency = Urgency.FAST
    params: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    degraded: bool = False          # router failed; safe defaults in use

    @property
    def needs_recall(self) -> bool:
        return self.memory in (MemoryAction.RECALL, MemoryAction.BOTH)

    @property
    def needs_store(self) -> bool:
        return self.memory in (MemoryAction.STORE, MemoryAction.BOTH)

    @property
    def uses_tool(self) -> bool:
        return self.tool is not ToolAction.NONE

    def summary(self) -> str:
        bits = [f"route={self.route.value}", f"mem={self.memory.value}",
                f"tool={self.tool.value}", f"{self.latency_ms:.0f}ms"]
        if self.degraded:
            bits.append("DEGRADED")
        return " ".join(bits)


# --------------------------------------------------------------------------- #
# 3. The router
# --------------------------------------------------------------------------- #
class Router:
    def __init__(self, small_client, *, debug: bool = False):
        self.decoder = ConstrainedDecoder(small_client, temperature=0.0, debug=debug)
        self.debug = debug

    async def decide(self, prompt: str, *, recent: str = "") -> Decision:
        """Stage 1 + stage 2. Never raises: a failure degrades to LARGE/NONE."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        user = prompt if not recent else f"Recent context:\n{recent}\n\nRequest:\n{prompt}"

        try:
            # Three independent single-token decodes. They are gathered rather
            # than awaited in turn, so the wall-clock cost is roughly that of
            # the slowest one (the server needs --parallel 2 or more to
            # actually overlap them; it degrades to serial, not to wrong).
            route, past, store = await asyncio.gather(
                self.decoder.choose(Route, ROUTE_SYSTEM, user,
                                    examples=ROUTE_EXAMPLES,
                                    default=Route.LARGE),
                self.decoder.choose(YesNo, PAST_SYSTEM, user,
                                    examples=PAST_EXAMPLES, default=YesNo.NO),
                self.decoder.choose(YesNo, STORE_SYSTEM, user,
                                    examples=STORE_EXAMPLES, default=YesNo.NO),
            )
            # A prompt the route classifier calls trivial ("yep", "yo", "what is
            # 9 divided by 3") cannot need a web search or a shell command. The
            # tool classifiers are exactly the ones that over-fire on inputs
            # that short, and route is the most reliable of the five, so gating
            # on it both removes that error class and saves two round trips on
            # the prompts that most need to feel instant.
            tool = ToolAction.NONE
            if route is Route.LARGE:
                web, cli = await asyncio.gather(
                    self.decoder.choose(YesNo, WEB_SYSTEM, user,
                                        examples=WEB_EXAMPLES, default=YesNo.NO),
                    self.decoder.choose(YesNo, CLI_SYSTEM, user,
                                        examples=CLI_EXAMPLES, default=YesNo.NO),
                )
                tool = ToolAction.combine(web is YesNo.YES, cli is YesNo.YES)
            decision = Decision(
                route=route, tool=tool,
                memory=MemoryAction.combine(past is YesNo.YES,
                                            store is YesNo.YES),
            )
        except Exception as exc:                        # noqa: BLE001
            print(f"[router] classification failed ({exc}); defaulting to LARGE")
            decision = Decision(degraded=True)
            decision.latency_ms = (loop.time() - started) * 1000
            return decision

        if decision.uses_tool:
            try:
                decision.params = await self.extract_params(decision.tool, prompt)
            except Exception as exc:                    # noqa: BLE001
                print(f"[router] param extraction failed ({exc}); dropping tool")
                decision.tool = ToolAction.NONE
                decision.degraded = True

        # A tool result always wants a competent summariser, and anything the
        # small model cannot verify should not be answered by the small model.
        if decision.uses_tool:
            decision.route = Route.LARGE
        # Urgency is derived, not asked: the 0.5B has no idea how long the 35B
        # will take, but we do -- it is exactly "did we route to the big model".
        decision.urgency = (Urgency.SLOW if decision.route is Route.LARGE
                            else Urgency.FAST)

        decision.latency_ms = (loop.time() - started) * 1000
        if self.debug:
            print(f"[router] {decision.summary()} params={decision.params}")
        return decision

    async def decide_joint(self, prompt: str, *, recent: str = "") -> Decision:
        """One-call variant: all four fields as a single constrained JSON object.

        Kept for when the router model is upgraded -- it is one round trip
        instead of three, but it needs a model that can actually hold four
        rubrics in its head at once. Qwen2.5-0.5B-Instruct cannot.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        user = prompt if not recent else f"Recent context:\n{recent}\n\nRequest:\n{prompt}"
        system = "\n\n".join([
            "You are the decision router for a local desktop assistant. "
            "Classify the request; do not answer it. Reply with JSON only.",
            WEB_SYSTEM, CLI_SYSTEM, PAST_SYSTEM, STORE_SYSTEM, ROUTE_SYSTEM,
            "urgency: FAST if a short answer is expected, SLOW for a long job.",
        ])
        raw = await self.decoder.extract(DECISION_FIELDS, system, user, max_tokens=64)
        decision = Decision(
            route=Route(raw["route"]), memory=MemoryAction(raw["memory"]),
            tool=ToolAction(raw["tool"]), urgency=Urgency(raw["urgency"]),
            degraded=bool(self.decoder.last_missing),
        )
        if decision.uses_tool:
            decision.params = await self.extract_params(decision.tool, prompt)
        decision.latency_ms = (loop.time() - started) * 1000
        return decision

    async def extract_params(self, tool: ToolAction, prompt: str) -> dict:
        """Stage 2: grammar-constrained parameter extraction."""
        if tool is ToolAction.TOOL_WEB:
            params = await self.decoder.extract(
                WEB_PARAM_FIELDS, WEB_PARAM_SYSTEM, prompt,
                examples=WEB_PARAM_EXAMPLES, max_tokens=96,
            )
            params["max_results"] = max(3, min(8, params.get("max_results") or 5))
            if not params["query"].strip():
                params["query"] = prompt[:120]
            return params

        if tool is ToolAction.TOOL_CLI:
            params = await self.decoder.extract(
                CLI_PARAM_FIELDS, CLI_PARAM_SYSTEM, prompt,
                examples=CLI_PARAM_EXAMPLES, max_tokens=128,
            )
            if not params["command"].strip():
                raise ValueError("router produced an empty command")
            return params

        return {}

    # --------------------------------------------------------------- utilities
    async def classify_field(self, spec, system: str, prompt: str, default=None):
        """Single-token enum decision, for ad-hoc questions elsewhere in the app.

        e.g. `await router.classify_field(Route, "Pick the model.", text)`
        """
        return await self.decoder.choose(spec, system, prompt, default=default)
