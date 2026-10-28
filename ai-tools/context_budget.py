#!/usr/bin/env python3
"""Context accounting and compression.

The prompt is assembled from blocks of very different value per token:

    system prompt     small, constant, never droppable
    rolling summary   small, high value, replaceable
    recalled memories ranked facts; dropping them loses real information
    history turns     the oldest fold into the summary almost losslessly
    tool output       biggest block, most redundant, safe to trim middle-out
    the user's prompt tiny, never droppable

Folding history is tried BEFORE dropping memories. The first version of this
ladder had it the other way round and a 40-turn thread threw away all six
recalled memories -- the facts the assistant was told -- when summarising 36
stale turns would have freed four times as much on its own.

So rather than one blunt "truncate the front" rule, compression walks a ladder
from cheapest loss to dearest, re-measuring after each rung, and stops as soon
as the prompt fits. Measurement is exact: llama-server tokenises the fully
templated prompt (/apply-template + /tokenize, about 4ms), so the decision is
never made on a len()//4 guess that the chat template invalidates.

Folding history into the summary is persisted with a watermark
(conversations.summary_through_id), so a long thread pays the summarisation
cost once rather than on every turn.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import requests

ELISION = "\n...[%d tokens of tool output elided]...\n"


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
class TokenCounter:
    """Exact token counts from the model that will actually serve the request."""

    def __init__(self, chat_url: str, timeout: float = 20.0):
        self.base = chat_url.split("/v1/")[0]
        self.timeout = timeout
        self._n_ctx: int | None = None
        self._cache: dict[str, int] = {}

    # ------------------------------------------------------------- blocking
    def _n_ctx_sync(self) -> int:
        resp = requests.get(self.base + "/props", timeout=self.timeout)
        resp.raise_for_status()
        settings = resp.json().get("default_generation_settings") or {}
        return int(settings.get("n_ctx") or 0)

    def _count_text_sync(self, text: str) -> int:
        if not text:
            return 0
        resp = requests.post(self.base + "/tokenize",
                             json={"content": text}, timeout=self.timeout)
        resp.raise_for_status()
        return len(resp.json().get("tokens") or [])

    def _count_messages_sync(self, messages, tools=None) -> int:
        body: dict[str, Any] = {"messages": messages}
        if tools:
            body["tools"] = tools
        resp = requests.post(self.base + "/apply-template",
                             json=body, timeout=self.timeout)
        resp.raise_for_status()
        return self._count_text_sync(resp.json().get("prompt") or "")

    # ---------------------------------------------------------------- async
    async def n_ctx(self) -> int:
        if self._n_ctx is None:
            try:
                self._n_ctx = await asyncio.to_thread(self._n_ctx_sync)
            except Exception as exc:                    # noqa: BLE001
                print(f"[budget] /props unavailable ({exc}); assuming 4096")
                self._n_ctx = 4096
        return self._n_ctx

    async def count_text(self, text: str) -> int:
        if not text:
            return 0
        key = text if len(text) < 512 else f"{len(text)}:{hash(text)}"
        if key not in self._cache:
            self._cache[key] = await asyncio.to_thread(self._count_text_sync, text)
        return self._cache[key]

    async def count_messages(self, messages, tools=None) -> int:
        """Tokens of the fully templated prompt, scaffolding included."""
        return await asyncio.to_thread(self._count_messages_sync, messages, tools)


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
@dataclass
class Budget:
    n_ctx: int
    reserve_completion: int = 2048     # room the reply needs to exist in
    safety: int = 256                  # template drift, tool-call wrappers
    compress_at: float = 0.9           # of `limit`, start compressing early

    @property
    def limit(self) -> int:
        return max(512, self.n_ctx - self.reserve_completion - self.safety)

    @property
    def trigger(self) -> int:
        return int(self.limit * self.compress_at)

    def describe(self, used: int) -> str:
        return (f"{used}/{self.limit} tok "
                f"(n_ctx {self.n_ctx} - {self.reserve_completion} reply "
                f"- {self.safety} safety)")


# --------------------------------------------------------------------------- #
# The structured prompt
# --------------------------------------------------------------------------- #
@dataclass
class PromptPlan:
    """The prompt as separable blocks, so compression can target the cheapest.

    `render()` is the single place that turns blocks into the message array, so
    measuring and sending can never drift apart.
    """
    system_prompt: str = ""
    extra_system: list[str] = field(default_factory=list)
    summary: str = ""
    memories: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)      # {role, content, id}
    tool_blocks: list[str] = field(default_factory=list)
    prompt: str = ""
    tools: list[dict] | None = None

    def system_message(self, format_memories) -> dict:
        parts = [self.system_prompt, *self.extra_system]
        if self.summary:
            parts.append("Summary of the earlier part of this conversation:\n"
                         + self.summary)
        if self.memories:
            parts.append(
                "Relevant memories from earlier conversations (most relevant first):\n"
                + format_memories(self.memories))
        return {"role": "system", "content": "\n\n".join(p for p in parts if p)}

    def render(self, format_memories) -> list[dict]:
        messages = [self.system_message(format_memories)]
        for turn in self.history:
            if turn.get("role") in ("user", "assistant") and (turn.get("content") or "").strip():
                messages.append({"role": turn["role"], "content": turn["content"]})
        if not messages[-1:] or messages[-1].get("content") != self.prompt:
            messages.append({"role": "user", "content": self.prompt})
        for block in self.tool_blocks:
            messages.append({
                "role": "user",
                "content": ("Tool output for the request above. Base your answer on "
                            "it, cite URLs when present, and say so if it does not "
                            "actually answer the question.\n\n" + block),
            })
        return messages


@dataclass
class FitResult:
    messages: list[dict]
    tokens: int
    limit: int
    steps: list[str] = field(default_factory=list)
    summary_written: str = ""          # new rolling summary, if one was made
    summary_through_id: int = 0
    overflowed: bool = False           # could not be made to fit
    breakdown: dict = field(default_factory=dict)   # per-block token counts

    @property
    def compressed(self) -> bool:
        return bool(self.steps)


# --------------------------------------------------------------------------- #
# Compression
# --------------------------------------------------------------------------- #
def truncate_middle(text: str, keep_chars: int) -> str:
    """Keep the head and tail of a block, elide the middle.

    Tool output is front-loaded (headers, the first rows) and back-loaded (the
    totals, the error) with the least useful material in between, so cutting
    the middle preserves far more meaning than cutting the tail.
    """
    if keep_chars <= 0 or len(text) <= keep_chars:
        return text
    half = max(200, keep_chars // 2)
    cut = len(text) - (half * 2)
    return text[:half] + ELISION % max(1, cut // 4) + text[-half:]


SUMMARY_SYSTEM = """You are compacting the earlier part of a conversation so it
can be dropped from the context window without losing what matters.

Write a dense factual summary covering:
  * what the user is trying to achieve, and any constraints they stated
  * decisions taken and the reasons for them
  * concrete details that would be expensive to rediscover: file paths,
    commands, port numbers, version numbers, names, error messages
  * what has already been done, and what is still outstanding

Rules: prose or terse bullets, no preamble, no markdown headings, no invented
detail. If an earlier summary is supplied, merge it in rather than restating
it. Be complete over being short, but never pad."""


class ContextCompressor:
    def __init__(self, counter: TokenCounter, budget: Budget, *,
                 summariser=None, format_memories=None,
                 keep_recent_turns: int = 4, min_memories: int = 2,
                 tool_share: float = 0.5, debug: bool = False):
        self.counter = counter
        self.budget = budget
        # async (turns: list[dict], prev_summary: str) -> str. Takes the turns
        # rather than a joined string so the implementation can batch them to
        # fit its own context window -- folding a 16k-token backlog into a
        # model with an 8k window is otherwise an instant HTTP 400.
        self.summariser = summariser
        self.format_memories = format_memories or (lambda rows: "")
        self.keep_recent_turns = keep_recent_turns
        self.min_memories = min_memories
        self.tool_share = tool_share          # max fraction of limit for tools
        self.debug = debug

    async def fit(self, plan: PromptPlan) -> FitResult:
        """Compress `plan` until it fits, cheapest loss first."""
        limit = self.budget.limit
        result = FitResult(messages=[], tokens=0, limit=limit)

        async def measure() -> int:
            messages = plan.render(self.format_memories)
            return await self.counter.count_messages(messages, plan.tools)

        used = await measure()
        if used <= self.budget.trigger:
            # Still goes through _finish, so the per-block breakdown is
            # produced on the common path and not only when compressing.
            return await self._finish(plan, result, used)

        if self.debug:
            print(f"[budget] {used} tok exceeds trigger {self.budget.trigger}; compressing")

        # --- rung 1: cap tool output -------------------------------------
        if plan.tool_blocks:
            allowance = int(limit * self.tool_share)
            tool_tokens = sum(
                [await self.counter.count_text(b) for b in plan.tool_blocks]
            )
            if tool_tokens > allowance:
                # ~4 chars per token is a safe over-estimate to aim at; the
                # re-measure below is what actually decides.
                per_block = max(1, allowance // len(plan.tool_blocks))
                plan.tool_blocks = [
                    truncate_middle(b, per_block * 4) for b in plan.tool_blocks
                ]
                used = await measure()
                result.steps.append(
                    f"trimmed tool output {tool_tokens}->{allowance} tok")
                if used <= limit:
                    return await self._finish(plan, result, used)

        # --- rung 2: fold old history into the rolling summary -----------
        if self.summariser and len(plan.history) > self.keep_recent_turns:
            foldable = plan.history[:-self.keep_recent_turns]
            kept = plan.history[-self.keep_recent_turns:]
            try:
                merged = await self.summariser(foldable, plan.summary)
            except Exception as exc:                    # noqa: BLE001
                print(f"[budget] summarisation failed ({exc}); truncating instead")
                merged = ""
            if merged:
                plan.summary = merged
                plan.history = kept
                result.summary_written = merged
                result.summary_through_id = max(
                    (t.get("id") or 0) for t in foldable) if foldable else 0
                used = await measure()
                result.steps.append(
                    f"folded {len(foldable)} turns into the summary")
                if used <= limit:
                    return await self._finish(plan, result, used)

        # --- rung 3: drop the weakest memories ---------------------------
        dropped = 0
        while len(plan.memories) > self.min_memories and used > limit:
            plan.memories.pop()                # search() returns best-first
            used = await measure()
            dropped += 1
        if dropped:
            result.steps.append(f"dropped the {dropped} weakest recalled memor"
                                + ("y" if dropped == 1 else "ies"))
        if used <= limit:
            return await self._finish(plan, result, used)

        # --- rung 4: shed remaining history, oldest first ----------------
        shed = 0
        while len(plan.history) > 1 and used > limit:
            plan.history.pop(0)
            used = await measure()
            shed += 1
        if shed:
            result.steps.append(f"dropped the {shed} oldest remaining turn(s)")
        if used <= limit:
            return await self._finish(plan, result, used)

        # --- rung 5: last resort -----------------------------------------
        # Keep only what the request cannot be served without. Anything still
        # over after this is a prompt that physically cannot fit, and saying so
        # beats silently sending something the server will truncate.
        plan.memories, plan.summary, plan.history = [], "", []
        used = await measure()
        result.steps.append("dropped summary, memories and all history")
        if used > limit and plan.tool_blocks:
            room = max(0, limit - await self.counter.count_text(plan.prompt) - 512)
            plan.tool_blocks = [truncate_middle(b, room * 4 // max(1, len(plan.tool_blocks)))
                                for b in plan.tool_blocks]
            used = await measure()
            result.steps.append("hard-trimmed tool output")

        result.overflowed = used > limit
        if result.overflowed:
            print(f"[budget] still {used} tok over a {limit} tok limit")
        return await self._finish(plan, result, used)

    async def breakdown(self, plan: PromptPlan) -> dict:
        """Per-block token counts for the final prompt, for display.

        Counted against the same tokeniser as everything else, so the parts
        add up to the measured total apart from the chat template's own
        scaffolding, which is reported as the remainder.
        """
        counter = self.counter
        parts = {
            "system prompt": await counter.count_text(plan.system_prompt),
            "tool instructions": sum(
                [await counter.count_text(x) for x in plan.extra_system]),
            "conversation summary": await counter.count_text(plan.summary),
            "recalled memories": await counter.count_text(
                self.format_memories(plan.memories) if plan.memories else ""),
            "history turns": sum(
                [await counter.count_text(t.get("content") or "")
                 for t in plan.history]),
            "tool output": sum(
                [await counter.count_text(b) for b in plan.tool_blocks]),
            "your prompt": await counter.count_text(plan.prompt),
            "tool schemas": 0,
        }
        if plan.tools:
            bare = await counter.count_messages(
                [{"role": "system", "content": ""}, {"role": "user", "content": ""}])
            withtools = await counter.count_messages(
                [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
                plan.tools)
            parts["tool schemas"] = max(0, withtools - bare)
        parts["counts"] = {
            "memories": len(plan.memories),
            "history_turns": len(plan.history),
            "tool_blocks": len(plan.tool_blocks),
        }
        return parts

    async def _finish(self, plan, result: FitResult, used: int) -> FitResult:
        result.messages = plan.render(self.format_memories)
        result.tokens = used
        try:
            result.breakdown = await self.breakdown(plan)
        except Exception as exc:                        # noqa: BLE001
            print(f"[budget] breakdown unavailable ({exc})")
        if self.debug and result.steps:
            print(f"[budget] compressed to {used}/{result.limit}: "
                  + "; ".join(result.steps))
        return result
