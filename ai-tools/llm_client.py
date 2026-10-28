#!/usr/bin/env python3
"""Async llama.cpp (OpenAI-compatible) client shared by the router, the tools
and the live commentator.

Only `requests` is used for I/O -- blocking calls are pushed onto worker
threads so the asyncio orchestrator never stalls. That keeps the bootstrap
dependency list in ai_assistant.py unchanged (no aiohttp/httpx).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
from dataclasses import dataclass, field

import requests

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


@dataclass
class Usage:
    prompt: int = 0
    completion: int = 0
    tps: float = 0.0


@dataclass
class LlamaClient:
    """One llama-server endpoint (`/v1/chat/completions`)."""

    url: str
    model: str = ""
    name: str = "llm"
    default_timeout: float = 300.0
    # Populated by the last streamed round; the GTK layer reads it for the
    # tok/s readout.
    last_usage: Usage = field(default_factory=Usage)

    # ---------------------------------------------------------------- blocking
    def _post(self, payload: dict, timeout: float, stream: bool = False,
              session: "requests.Session | None" = None):
        if self.model and "model" not in payload:
            payload["model"] = self.model
        poster = session.post if session is not None else requests.post
        resp = poster(self.url, json=payload, stream=stream, timeout=timeout)
        resp.raise_for_status()
        return resp

    def complete_sync(self, messages, *, timeout=None, **params) -> str:
        payload = {"messages": messages, **params}
        resp = self._post(payload, timeout or 60.0)
        return resp.json()["choices"][0]["message"]["content"] or ""

    # ------------------------------------------------------------------- async
    async def complete(self, messages, *, timeout=None, **params) -> str:
        """Non-streamed completion. Extra kwargs go straight to llama-server,
        so `grammar=...`, `response_format=...`, `temperature=...` all work."""
        return await asyncio.to_thread(
            self.complete_sync, messages, timeout=timeout, **params
        )

    # Thinking models (the 35B here is one) will spend an entire max_tokens
    # budget inside reasoning_content and return content="" with
    # finish_reason="length". For internal, non-user-facing generations -- a
    # summary, a title -- the thoughts are worthless and the text is the whole
    # point, so ask the chat template to skip them.
    NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

    async def complete_text(self, messages, *, timeout=None, **params) -> str:
        """A completion that is guaranteed to be prose, not chain-of-thought.

        Tries with thinking disabled; if the template ignored the hint and the
        model still answered with nothing but reasoning, retries once with a
        much larger token budget so the text has room to appear after it.
        """
        try:
            text = await self.complete(messages, timeout=timeout,
                                       **{**params, **self.NO_THINK})
        except requests.exceptions.HTTPError:
            text = ""                       # template rejected the kwarg
        if strip_think(text):
            return strip_think(text)

        retry = dict(params)
        retry["max_tokens"] = max(int(retry.get("max_tokens") or 512) * 4, 2048)
        text = await self.complete(messages, timeout=timeout, **retry)
        return strip_think(text)

    async def stream(self, messages, *, timeout=None, **params):
        """Async generator of (kind, value) events.

        kind is one of: "content", "reasoning", "tool_calls", "usage",
        "finish_reason".
        Cancelling the generator closes the HTTP response, which makes
        llama-server drop the slot instead of generating into the void.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def emit(item):
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker():
            try:
                for item in self._stream_sync(messages, stop, timeout, params):
                    if stop.is_set():
                        break
                    emit(item)
            except BaseException as exc:          # noqa: BLE001 - relayed below
                emit(("__error__", exc))
            finally:
                emit(("__done__", None))

        thread = threading.Thread(target=worker, name=f"{self.name}-stream", daemon=True)
        thread.start()
        try:
            while True:
                kind, value = await queue.get()
                if kind == "__done__":
                    return
                if kind == "__error__":
                    raise value
                yield kind, value
        finally:
            stop.set()

    def _stream_sync(self, messages, stop: threading.Event, timeout, params):
        payload = {
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "timings_per_token": True,
            **params,
        }
        # A dedicated Session per stream, because Response.close() only
        # *releases* the connection back to the pool -- the socket stays open
        # and llama.cpp keeps generating into it. Session.close() tears the
        # pool down, the server sees the disconnect and logs
        # "srv stop: cancel task". Without this, pressing Stop freed the UI
        # while the model carried on burning the slot for minutes.
        session = requests.Session()
        resp = self._post(payload, timeout or self.default_timeout,
                          stream=True, session=session)
        # llama.cpp labels the SSE stream text/event-stream with no charset, and
        # requests then falls back to ISO-8859-1 for text/* -- which renders
        # every smart quote and dash as mojibake ("didn-a-EUR-(tm)-t").
        resp.encoding = "utf-8"
        tool_acc: dict[int, dict] = {}
        usage = Usage()
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if stop.is_set():
                    break
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue

                timings = chunk.get("timings") or {}
                if timings:
                    usage.completion = timings.get("predicted_n") or usage.completion
                    usage.tps = timings.get("predicted_per_second") or usage.tps
                    usage.prompt = timings.get("prompt_n") or usage.prompt
                    yield "usage", timings
                api_usage = chunk.get("usage") or {}
                if api_usage:
                    usage.prompt = api_usage.get("prompt_tokens") or usage.prompt
                    usage.completion = api_usage.get("completion_tokens") or usage.completion

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason"):
                    # "length" means the server stopped because it ran out of
                    # room, not because the model was done. The caller needs to
                    # know: for a reasoning model that can mean a reply that is
                    # cut mid-sentence, or no reply at all.
                    yield "finish_reason", choices[0]["finish_reason"]
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    yield "content", delta["content"]
                if delta.get("reasoning_content"):
                    yield "reasoning", delta["reasoning_content"]
                for call in delta.get("tool_calls") or []:
                    slot = tool_acc.setdefault(
                        call.get("index", 0), {"id": "", "name": "", "arguments": ""}
                    )
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
        finally:
            aborted = stop.is_set()
            resp.close()
            session.close()
            self.last_usage = usage
            if aborted:
                print(f"[{self.name}] generation aborted by the client",
                      file=sys.stderr)

        if tool_acc:
            yield "tool_calls", [tool_acc[k] for k in sorted(tool_acc)]


def clients_from_config(config) -> tuple[LlamaClient, LlamaClient]:
    """(big, small) clients built from the ai_assistant .config dict."""
    big = LlamaClient(
        url=config.get("llama_url", "http://127.0.0.1:9090/v1/chat/completions"),
        model=config.get("llama_model", ""),
        name="35B",
    )
    small = LlamaClient(
        url=config.get("llama_small_url", "http://127.0.0.1:9191/v1/chat/completions"),
        model=config.get("llama_small_model", ""),
        name="0.5B",
        default_timeout=30.0,
    )
    return big, small
