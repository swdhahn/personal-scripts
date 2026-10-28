#!/usr/bin/env python3
"""Live commentary: the 0.5B model narrates while the 35B model thinks.

The pattern is one `asyncio.Task` that runs *beside* the main generation task,
not inside it:

    commentary = LiveCommentator(small, speak)
    async with commentary.session("Checking disk usage") as live:
        answer = await big_model_task()        # minutes
        # `live` keeps talking the whole time; cancelled on exit

Design constraints that matter in practice:

  * The 0.5B server runs with `--parallel 1`, so a commentary generation and a
    router call would queue behind each other. Commentary therefore only runs
    while the big model holds the floor, and yields immediately on cancel.
  * Never speak over yourself: the next line is only generated once the
    previous one has been handed to the TTS queue *and* the minimum gap has
    elapsed. `speak` is awaited, so a slow Kokoro queue naturally throttles
    the narrator instead of piling up a backlog.
  * Never repeat: the last few lines are passed back as negative context and
    deduplicated on a normalised form.
  * The first line is a canned, instant one per stage -- the user hears
    something within milliseconds rather than after a model round trip.
"""
from __future__ import annotations

import asyncio
import contextlib
import random
import re

COMMENTARY_SYSTEM = """You are the voice of a desktop AI assistant, speaking
aloud while a slower, larger model works on the user's request.

Produce ONE short spoken sentence, under fourteen words, in first person
present continuous, describing what is happening right now. Be specific to the
task. Never invent a result, a finding, or a conclusion. Never ask a question.
No markdown, no quotes, no emoji, no numbers you were not given.

Good: I am still reading through the search results.
Good: I am working out how the pieces fit together.
Bad: I found three relevant articles about this.
Bad: Let me know if you want more detail."""

# Spoken instantly, before the first model round trip finishes.
OPENERS = {
    "routing": ["One moment.", "Let me think about that."],
    "memory": ["Let me check what we talked about before.",
               "Pulling up my notes on this."],
    "web": ["I am looking that up on the web right now.",
            "Searching the web for that now."],
    "cli": ["Running that on your machine now.",
            "Let me take a look at the system."],
    "generating": ["Alright, working on it.", "Thinking this through."],
    "summarising": ["Reading through what came back.",
                    "Going over the results now."],
    "compacting": ["Our conversation is getting long, so I am condensing it.",
                   "Give me a moment to summarise what we have covered."],
}

STAGE_HINTS = {
    "routing": "deciding how to handle the request",
    "memory": "searching the local SQLite memory of past conversations",
    "web": "waiting on web search results and reading the pages",
    "cli": "running a sandboxed shell command and reading its output",
    "generating": "the large 35B model is composing the answer",
    "summarising": "the large model is summarising the tool output",
    "compacting": ("condensing the earlier part of this long conversation into "
                   "a summary so it still fits in the context window"),
}

_NORM_RE = re.compile(r"[^a-z ]+")
# "The large model is summarising the results for X" vs "The large model is
# summarising X" are different strings and the same sentence out loud, so
# near-duplicates are caught by word overlap rather than exact equality.
_DUP_OVERLAP = 0.6
_SOLICIT_RE = re.compile(
    r"let me know|if you('d| would)? (want|like)|would you like|anything else|"
    r"feel free|shall i|do you want|hope (this|that) helps"
)


def _norm(text: str) -> str:
    return _NORM_RE.sub("", (text or "").lower()).strip()


def _too_similar(text: str, previous: list[str]) -> bool:
    words = set(_norm(text).split())
    if not words:
        return True
    for earlier in previous:
        other = set(_norm(earlier).split())
        if not other:
            continue
        overlap = len(words & other) / len(words | other)
        if overlap >= _DUP_OVERLAP:
            return True
    return False


class LiveCommentator:
    """Owns the narration task. One instance per chat window is plenty."""

    def __init__(
        self,
        small_client,
        speak,
        *,
        first_delay: float = 1.2,
        interval: float = 9.0,
        jitter: float = 3.0,
        max_lines: int = 6,
        enabled: bool = True,
    ):
        self.small = small_client
        self.speak = speak                 # async def speak(text: str) -> None
        self.first_delay = first_delay
        self.interval = interval
        self.jitter = jitter
        self.max_lines = max_lines
        self.enabled = enabled

        self._stage = "generating"
        self._quiet = False
        self._task_desc = ""
        self._spoken: list[str] = []
        self._stage_event = asyncio.Event()

    # ------------------------------------------------------------------ public
    def set_stage(self, stage: str, *, speak_opener: bool = True):
        """Called by the orchestrator as the pipeline moves on.

        Cheap and non-blocking: it only flips state and wakes the narrator, so
        it is safe to call from inside a tight streaming loop.
        """
        if stage == self._stage:
            return
        self._stage = stage
        if speak_opener:
            self._stage_event.set()

    def go_quiet(self):
        """Stop narrating but keep the task alive (e.g. first token arrived)."""
        self._quiet = True
        self._stage_event.set()

    @contextlib.asynccontextmanager
    async def session(self, task_desc: str, *, stage: str = "generating"):
        """Narrate for the duration of the block, then stop cleanly."""
        self._task_desc = task_desc
        self._stage = stage
        self._quiet = False
        self._spoken.clear()
        self._stage_event = asyncio.Event()

        if not self.enabled:
            yield self
            return

        task = asyncio.create_task(self._loop(), name="live-commentary")
        try:
            yield self
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # -------------------------------------------------------------- the engine
    async def _loop(self):
        """The whole point: this runs concurrently with the 35B generation."""
        try:
            await self._say(self._opener())
            spoken = 1
            while spoken < self.max_lines:
                delay = max(2.0, self.interval + random.uniform(-self.jitter, self.jitter))
                # Wake early if the stage changed, so the narration tracks the
                # pipeline instead of lagging a whole interval behind it.
                stage_before = self._stage
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stage_event.wait(), timeout=delay)
                self._stage_event.clear()
                if self._quiet:
                    return

                if self._stage != stage_before:
                    line = self._opener()
                else:
                    line = await self._generate()
                if line:
                    await self._say(line)
                    spoken += 1
        except asyncio.CancelledError:
            raise                               # the big model finished; go quiet
        except Exception as exc:                # noqa: BLE001 - never kill the turn
            print(f"[commentary] stopped: {exc}")

    def _opener(self) -> str:
        return random.choice(OPENERS.get(self._stage, OPENERS["generating"]))

    async def _generate(self) -> str:
        hint = STAGE_HINTS.get(self._stage, STAGE_HINTS["generating"])
        avoid = "; ".join(self._spoken[-3:])
        user = (
            f"The user asked: {self._task_desc[:300]}\n"
            f"What is happening right now: {hint}\n"
            + (f"Do not repeat these lines: {avoid}\n" if avoid else "")
            + "Your one spoken sentence:"
        )
        try:
            raw = await asyncio.wait_for(
                self.small.complete(
                    [
                        {"role": "system", "content": COMMENTARY_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    timeout=12.0,
                    temperature=0.85,
                    max_tokens=36,
                    cache_prompt=True,
                ),
                timeout=14.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:   # noqa: BLE001
            if isinstance(exc, asyncio.CancelledError):
                raise
            print(f"[commentary] generation skipped: {exc}")
            return ""
        return self._clean(raw)

    def _clean(self, raw: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
        text = re.sub(r"[*`#_\"]", "", text).strip().strip("-").strip()
        text = text.split("\n")[0].strip()
        # One sentence only; a 0.5B model will happily keep going.
        parts = re.split(r"(?<=[.!?])\s+", text)
        text = parts[0].strip() if parts else text
        if not text or len(text.split()) > 20:
            return ""
        # The 0.5B drifts into narrating the architecture in third person
        # ("The large model is summarising...") which breaks the illusion that
        # the assistant is the one talking. Ask again rather than say it.
        lowered = text.lower()
        if re.match(r"^(the|a)\s+(large|small|big|main|language)?\s*model\b", lowered):
            return ""
        if not re.search(r"\b(i|i'm|im|my|me)\b", lowered):
            return ""
        # Progress updates must not solicit a reply: the user is mid-wait and
        # the answer has not arrived yet, so an offer or a question is noise.
        if "?" in text or _SOLICIT_RE.search(lowered):
            return ""
        if _too_similar(text, self._spoken):
            return ""
        return text

    async def _say(self, line: str):
        if not line:
            return
        self._spoken.append(line)
        # Awaited: a backed-up Kokoro queue throttles the narrator for free.
        await self.speak(line)
