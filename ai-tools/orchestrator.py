#!/usr/bin/env python3
"""The request lifecycle.

    prompt
      |
      +-- [0.5B, grammar-constrained]  Router.decide()        ~40ms
      |        route / memory / tool / urgency  (+ tool params)
      |
      +-- SQLite recall            (if memory in RECALL/BOTH)
      +-- Tool execution           (if tool in TOOL_WEB/TOOL_CLI)
      |        web search  |  bwrap-sandboxed command
      |
      +-- Text generation          SMALL -> 0.5B   |   LARGE -> 35B
      |        ... while the LiveCommentator narrates on the 0.5B model
      |
      +-- SQLite store             (if memory in STORE/BOTH)
      +-- done

Everything is an `asyncio` task, and the whole thing is exposed as an async
generator of events so a UI (GTK here) can render it incrementally without
knowing anything about the pipeline.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import memory_db
from commentary import LiveCommentator
from context_budget import (Budget, ContextCompressor, PromptPlan,
                            SUMMARY_SYSTEM, TokenCounter, truncate_middle)
from llm_client import LlamaClient, clients_from_config, strip_think
from router import Decision, Route, Router, ToolAction
from tools_local import (LOCAL_TOOL_SCHEMAS, MEMORY_TOOL_SCHEMA,
                         CliSandboxTool, WebSearchTool)

MAX_TOOL_ROUNDS = 3


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    kind: str                     # decision|recall|tool_start|tool_result|delta
                                  # |reasoning|usage|notice|done|error
    payload: Any = None

    def __repr__(self):
        text = repr(self.payload)
        return f"Event({self.kind}, {text[:80]})"


# --------------------------------------------------------------------------- #
# Async wrapper around the (blocking, lock-guarded) sqlite handle
# --------------------------------------------------------------------------- #
class MemoryGateway:
    """Serialises every sqlite call onto a worker thread under one lock."""

    def __init__(self, conn, lock, *, top_k=6, half_life=30.0, min_importance=0.25,
                 compress_threshold=2048, min_coverage=0.5, min_score=0.05):
        self.conn = conn
        self.lock = lock
        self.top_k = top_k
        self.half_life = half_life
        self.min_importance = min_importance
        self.compress_threshold = compress_threshold
        self.min_coverage = min_coverage
        self.min_score = min_score

    def _call(self, fn, *args, **kwargs):
        with self.lock:
            return fn(self.conn, *args, **kwargs)

    async def call(self, fn, *args, **kwargs):
        return await asyncio.to_thread(self._call, fn, *args, **kwargs)

    async def recall(self, query: str, k: int | None = None):
        return await self.call(
            memory_db.search, query, k=k or self.top_k,
            min_importance=self.min_importance, half_life_days=self.half_life,
            min_coverage=self.min_coverage, min_score=self.min_score,
        )

    async def recall_broad(self, query: str, k: int = 12):
        """Used when the router said RECALL: drop the importance floor so an
        explicit "what did we say about X" can reach low-salience notes."""
        return await self.call(
            memory_db.search, query, k=k, min_importance=0.0,
            half_life_days=self.half_life,
            # The router has said the user is asking about the past, so a
            # looser gate and the recent-and-important fallback are wanted.
            min_coverage=max(0.0, self.min_coverage - 0.2),
            min_score=self.min_score, allow_fallback=True,
        )

    async def store(self, content: str, *, kind="preference", importance=0.7,
                    tags=None, conversation_id=None):
        return await self.call(
            memory_db.add_memory, content, kind=kind, importance=importance,
            tags=tags or [], conversation_id=conversation_id, source="router",
        )

    async def add_message(self, conversation_id, role, content):
        return await self.call(
            memory_db.add_message, conversation_id, role, content,
            compress_threshold=self.compress_threshold,
        )

    async def history(self, conversation_id, limit=12, after_id=0):
        return await self.call(memory_db.get_messages, conversation_id,
                               limit=limit, after_id=after_id)

    async def save_summary(self, conversation_id, summary, through_id):
        return await self.call(memory_db.set_summary, conversation_id, summary,
                               through_id=through_id)

    async def conversation(self, conversation_id):
        return await self.call(memory_db.get_conversation, conversation_id)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class Orchestrator:
    def __init__(
        self,
        config,
        memory: MemoryGateway,
        *,
        speak: Callable[[str], Any] | None = None,
        approval_cb=None,
        debug: bool = False,
    ):
        self.config = config
        self.memory = memory
        self.debug = debug
        self.big, self.small = clients_from_config(config)
        self.router = Router(self.small, debug=debug)
        self.web = WebSearchTool(config)
        self.cli = CliSandboxTool(config, approval_cb=approval_cb)
        self.speak = speak or self._null_speak
        # No voice sink means nobody can hear the commentary, so don't spend
        # 0.5B generations writing lines into the void.
        narrate = speak is not None and str(
            config.get("commentary_enabled", "1")).lower() in ("1", "true", "yes")
        self.commentator = LiveCommentator(
            self.small,
            self.speak,
            interval=float(config.get("commentary_interval", 9) or 9),
            max_lines=int(config.get("commentary_max_lines", 6) or 6),
            enabled=narrate,
        )
        self.system_prompt = config.get("chat_system_prompt", "")

        # --- context budget -------------------------------------------------
        self.counter = TokenCounter(self.big.url)
        self.budget_cfg = {
            "reserve_completion": int(float(config.get("context_reserve_completion", 2048) or 2048)),
            "safety": int(float(config.get("context_safety_margin", 256) or 256)),
            "compress_at": float(config.get("context_compress_at", 0.9) or 0.9),
        }
        self.history_turns = int(float(config.get("context_history_turns", 12) or 12))
        self.keep_recent_turns = int(float(config.get("context_keep_recent_turns", 4) or 4))
        self.tool_share = float(config.get("context_tool_share", 0.5) or 0.5)
        self.min_memories = int(float(config.get("context_min_memories", 2) or 2))
        self.summariser_model = (config.get("context_summary_model") or "large").lower()

    @staticmethod
    async def _null_speak(_text):
        return None

    # ===================================================================== run
    async def handle(self, prompt: str, conversation_id=None) -> AsyncIterator[Event]:
        """The whole lifecycle, as a stream of events."""
        started = time.monotonic()
        try:
            # ---------------------------------------------------- 1. ROUTE
            recent = await self._recent_text(conversation_id)
            decision = await self.router.decide(prompt, recent=recent)
            yield Event("decision", decision)

            # ---------------------------------------------------- 2. MEMORY
            memories = []
            if decision.needs_recall:
                memories = await self.memory.recall_broad(prompt)
            else:
                memories = await self.memory.recall(prompt)
            yield Event("recall", memories)

            plan = await self._build_plan(conversation_id, memories, decision, prompt)

            # ---------------------------------------------------- 3. TOOLS
            tool_block = None
            if decision.uses_tool:
                async with self.commentator.session(
                    prompt, stage="web" if decision.tool is ToolAction.TOOL_WEB else "cli"
                ):
                    async for event in self._run_router_tool(decision):
                        if event.kind == "tool_result":
                            tool_block = event.payload["text"]
                        yield event
            if tool_block:
                plan.tool_blocks.append(tool_block)

            # ------------------------------------------ 3.5 CONTEXT BUDGET
            client = self.big if decision.route is Route.LARGE else self.small
            plan.tools = self._tool_schemas(client)
            # Compaction can cost several 35B generations, so narrate it for the
            # same reason generation is narrated: it must not happen in silence.
            needs_compaction = await self._over_trigger(plan)
            if needs_compaction and decision.route is Route.LARGE:
                async with self.commentator.session(prompt, stage="compacting"):
                    fit, budget = await self._fit(plan, conversation_id)
            else:
                fit, budget = await self._fit(plan, conversation_id)
            messages = fit.messages
            # Everything the prompt did not use is available to generate into.
            headroom = max(256, budget.n_ctx - fit.tokens - budget.safety)
            yield Event("context", {
                "tokens": fit.tokens, "limit": fit.limit, "n_ctx": budget.n_ctx,
                "steps": fit.steps, "overflowed": fit.overflowed,
                "reserve": budget.reserve_completion, "safety": budget.safety,
                "trigger": budget.trigger, "breakdown": fit.breakdown,
                "model": "35B" if client is self.big else "0.5B",
            })
            if fit.compressed:
                yield Event("notice", "Context compressed: " + "; ".join(fit.steps))
            if fit.overflowed:
                yield Event("notice",
                            f"Prompt is {fit.tokens} tok against a {fit.limit} tok "
                            "budget even after compression; the server may truncate it.")

            # ---------------------------------------------- 4. GENERATION
            stage = "summarising" if tool_block else "generating"
            narrate = decision.route is Route.LARGE

            if narrate:
                # THE live-commentary pattern: generation and narration are two
                # sibling tasks on the same loop. The narrator is cancelled the
                # moment the generator's context manager exits.
                async with self.commentator.session(prompt, stage=stage):
                    answer, quiet, truncated = "", False, None
                    async for event in self._generate(client, messages, plan.tools, headroom):
                        if event.kind == "answer":
                            answer = event.payload
                        elif event.kind == "truncated":
                            truncated = event.payload
                        else:
                            yield event
                        # First visible token means the wait is over.
                        if event.kind == "delta" and not quiet:
                            quiet = True
                            self.commentator.go_quiet()
            else:
                answer, truncated = "", None
                async for event in self._generate(client, messages, plan.tools, headroom):
                    if event.kind == "answer":
                        answer = event.payload
                    elif event.kind == "truncated":
                        truncated = event.payload
                    else:
                        yield event

            if truncated and not answer.strip():
                # A reasoning model spent the whole budget thinking and emitted
                # no answer at all. Rather than show the user an empty bubble,
                # ask again with thinking switched off so the tokens go into
                # the reply instead.
                yield Event("notice",
                            "The model used all "
                            f"{truncated['reasoning_chars']} characters of its budget "
                            "thinking without answering; retrying without the "
                            "reasoning step.")
                answer = await client.complete_text(
                    messages, timeout=600.0, temperature=0.6,
                    max_tokens=min(headroom, 1536),
                )
                for piece in (answer,):
                    if piece:
                        yield Event("delta", piece)
            elif truncated:
                yield Event("notice",
                            "The reply hit the context limit and may be cut short. "
                            "Start a new chat, or raise -c on the server, for more room.")

            # ---------------------------------------------------- 5. STORE
            if decision.needs_store:
                memory_id = await self.memory.store(
                    prompt, kind="preference", importance=0.75,
                    conversation_id=conversation_id,
                )
                yield Event("notice", f"Stored as memory #{memory_id}.")

            yield Event("done", {
                "answer": answer,
                "decision": decision,
                "elapsed_s": time.monotonic() - started,
            })
        except asyncio.CancelledError:
            raise
        except Exception as exc:                        # noqa: BLE001
            yield Event("error", f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------- generation
    async def _generate(self, client: LlamaClient, messages, tools=None,
                        max_tokens=0) -> AsyncIterator[Event]:
        """Streamed generation with the 35B's own tool-calling loop on top."""
        parts: list[str] = []
        reasoning_chars = 0
        for round_index in range(MAX_TOOL_ROUNDS + 1):
            chunk_text: list[str] = []
            tool_calls = []
            finish_reason = ""
            stream_params = {"temperature": 0.6, "tools": tools}
            if max_tokens:
                # Without an explicit cap the server generates until the KV
                # cache is physically full and then truncates mid-token
                # ("n_tokens = 8191, truncated = 1" in the server log), which
                # is what cuts a reply off mid-sentence. Capping it makes the
                # server stop cleanly and report finish_reason="length".
                stream_params["max_tokens"] = max_tokens
            async for kind, value in client.stream(messages, **stream_params):
                if kind == "finish_reason":
                    finish_reason = value
                    continue
                if kind == "content":
                    chunk_text.append(value)
                    yield Event("delta", value)
                elif kind == "reasoning":
                    reasoning_chars += len(value)
                    yield Event("reasoning", value)
                elif kind == "usage":
                    yield Event("usage", value)
                elif kind == "tool_calls":
                    tool_calls = value

            parts.append("".join(chunk_text))
            if not tool_calls or round_index == MAX_TOOL_ROUNDS:
                if finish_reason == "length":
                    yield Event("truncated", {
                        "had_text": bool("".join(parts).strip()),
                        "reasoning_chars": reasoning_chars,
                        "max_tokens": max_tokens,
                    })
                break

            messages.append({
                "role": "assistant",
                "content": "".join(chunk_text) or None,
                "tool_calls": [
                    {"id": call["id"] or f"call_{i}", "type": "function",
                     "function": {"name": call["name"], "arguments": call["arguments"]}}
                    for i, call in enumerate(tool_calls)
                ],
            })
            for index, call in enumerate(tool_calls):
                yield Event("tool_start", {"name": call["name"],
                                           "arguments": call["arguments"]})
                text = await self._dispatch_tool_call(call)
                yield Event("tool_result", {"name": call["name"], "text": text})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"] or f"call_{index}",
                    "content": text,
                })

        yield Event("answer", strip_think("".join(parts)).strip())

    def _tool_schemas(self, client: LlamaClient):
        if client is self.small:
            return None                 # a 0.5B model calling tools is a liability
        return [MEMORY_TOOL_SCHEMA] + LOCAL_TOOL_SCHEMAS

    # ------------------------------------------------------------------ tools
    async def _run_router_tool(self, decision: Decision) -> AsyncIterator[Event]:
        """Executes the tool the *router* picked, before generation starts."""
        params = decision.params
        if decision.tool is ToolAction.TOOL_WEB:
            query = params.get("query", "")
            yield Event("tool_start", {"name": "web_search", "arguments": params})
            payload = await self.web.run(query, params.get("max_results", 5))
            yield Event("tool_result", {
                "name": "web_search",
                "text": self.web.format_for_model(payload),
                "data": payload,
            })
            return

        if decision.tool is ToolAction.TOOL_CLI:
            command = params.get("command", "")
            yield Event("tool_start", {"name": "run_command", "arguments": params})
            result = await self.cli.run(command, params.get("purpose", ""))
            yield Event("tool_result", {
                "name": "run_command",
                "text": result.format_for_model(),
                "data": result,
            })

    async def _dispatch_tool_call(self, call) -> str:
        """Tool calls the 35B model makes on its own mid-generation."""
        name = call.get("name") or ""
        try:
            args = json.loads(call.get("arguments") or "{}")
        except ValueError:
            args = {}
        try:
            if name == "web_search":
                payload = await self.web.run(
                    str(args.get("query", "")), int(args.get("max_results") or 5)
                )
                return self.web.format_for_model(payload)
            if name == "run_command":
                result = await self.cli.run(
                    str(args.get("command", "")), str(args.get("purpose", ""))
                )
                return result.format_for_model()
            if name == "memory_search":
                rows = await self.memory.recall_broad(
                    str(args.get("query", "")),
                    k=max(1, min(12, int(args.get("limit") or self.memory.top_k))),
                )
                return memory_db.format_for_prompt(rows) or "No matching memories."
        except Exception as exc:                        # noqa: BLE001
            return f"Tool {name} failed: {type(exc).__name__}: {exc}"
        return f"Unknown tool {name!r}."

    # ----------------------------------------------------------------- prompts
    async def _recent_text(self, conversation_id, turns=2) -> str:
        if not conversation_id:
            return ""
        rows = await self.memory.history(conversation_id, limit=turns * 2)
        return "\n".join(
            f"{row['role']}: {row['content'][:200]}" for row in rows[-turns * 2:]
        )

    async def _build_plan(self, conversation_id, memories, decision,
                          prompt) -> PromptPlan:
        extra = []
        if decision.uses_tool:
            extra.append(
                "A local tool has been run for this request and its raw output is "
                "supplied below the conversation. Never claim you cannot access the "
                "web or the machine when tool output is present."
            )
        convo = await self.memory.conversation(conversation_id) if conversation_id else None
        history = []
        if conversation_id:
            history = await self.memory.history(
                conversation_id, limit=self.history_turns,
                after_id=(convo or {}).get("summary_through_id") or 0,
            )
        return PromptPlan(
            system_prompt=self.system_prompt,
            extra_system=extra,
            summary=(convo or {}).get("summary") or "",
            memories=list(memories),
            history=[dict(row) for row in history],
            prompt=prompt,
        )

    SUMMARY_MAX_TOKENS = 900

    async def _summarise(self, turns: list[dict], previous: str) -> str:
        """Fold old turns into a rolling summary, in context-sized batches.

        The backlog being folded is by definition larger than what fits, so it
        is summarised in chunks that each fit the summarising model's own
        window, merging into the running summary as it goes. Folding the whole
        backlog in one call returns HTTP 400 from llama.cpp the moment the
        transcript exceeds n_ctx -- which is precisely when compaction runs.

        Defaults to the 35B: a bad summary silently loses the conversation, and
        this runs once per compaction rather than once per turn, so fidelity is
        worth more than latency. Set context_summary_model=small to trade the
        other way.
        """
        client = self.small if self.summariser_model == "small" else self.big
        counter = (TokenCounter(client.url) if client is not self.big
                   else self.counter)
        n_ctx = await counter.n_ctx()
        # Room for the system prompt, the previous summary and the reply.
        per_call = max(512, n_ctx - self.SUMMARY_MAX_TOKENS - 1024)

        summary = previous
        batch: list[str] = []
        batch_tokens = 0
        pending = 0

        async def flush():
            nonlocal summary, batch, batch_tokens, pending
            if not batch:
                return
            transcript = "\n\n".join(batch)
            user = transcript if not summary else (
                f"Earlier summary to merge in:\n{summary}\n\n"
                f"Transcript to fold in:\n{transcript}")
            text = await client.complete_text(
                [{"role": "system", "content": SUMMARY_SYSTEM},
                 {"role": "user", "content": user}],
                timeout=300.0, temperature=0.2, max_tokens=self.SUMMARY_MAX_TOKENS,
            )
            text = (text or "").strip()
            if not text:
                # Never silently keep the old summary while reporting a fold:
                # the caller must be free to fall through to shedding turns.
                raise RuntimeError("summariser returned no text")
            summary = text
            batch, batch_tokens = [], 0
            pending += 1

        for turn in turns:
            line = f"{turn.get('role')}: {turn.get('content') or ''}"
            size = await counter.count_text(line)
            if size > per_call:                  # one colossal turn: trim it
                line = truncate_middle(line, per_call * 4)
                size = per_call
            if batch and batch_tokens + size > per_call:
                await flush()
            batch.append(line)
            batch_tokens += size
        await flush()

        if self.debug and pending > 1:
            print(f"[budget] folded {len(turns)} turns in {pending} summary passes")
        return summary

    async def _budget(self) -> Budget:
        return Budget(n_ctx=await self.counter.n_ctx(), **self.budget_cfg)

    async def _over_trigger(self, plan: PromptPlan) -> bool:
        """Cheap pre-check (one 4ms measurement) so narration can start first."""
        budget = await self._budget()
        used = await self.counter.count_messages(
            plan.render(memory_db.format_for_prompt), plan.tools)
        return used > budget.trigger

    async def _fit(self, plan: PromptPlan, conversation_id):
        """Measure the prompt and compress it if it will not fit."""
        budget = await self._budget()
        compressor = ContextCompressor(
            self.counter, budget,
            summariser=self._summarise,
            format_memories=memory_db.format_for_prompt,
            keep_recent_turns=self.keep_recent_turns,
            min_memories=self.min_memories,
            tool_share=self.tool_share,
            debug=self.debug,
        )
        result = await compressor.fit(plan)
        # Persist the folded summary so the next turn starts already compacted.
        if result.summary_written and conversation_id:
            await self.memory.save_summary(conversation_id, result.summary_written,
                                           result.summary_through_id)
        return result, budget

