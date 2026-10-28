#!/usr/bin/env python3
"""One asyncio event loop living on a background thread, for GTK3.

GTK owns the main thread, so the orchestrator gets its own loop:

    bridge = AsyncBridge()                  # starts the loop thread
    handle = bridge.spawn(coro)             # schedule work
    handle.cancel()                         # stop the turn (Esc, new prompt)

Rules of the road:
  * Anything touching GTK goes through `GLib.idle_add` (that is the only
    thread-safe GTK3 entry point).
  * Anything touching asyncio from the GTK thread goes through `spawn()` /
    `call()`, which use `run_coroutine_threadsafe`.
"""
from __future__ import annotations

import asyncio
import threading


class AsyncBridge:
    def __init__(self, name: str = "orchestrator-loop"):
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()
        self._ready.wait(5.0)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self._ready.set)
        self.loop.run_forever()

    # ------------------------------------------------------------------ public
    def spawn(self, coro) -> "asyncio.Future":
        """Fire-and-forget from the GTK thread. Returns a cancellable future."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def call(self, coro, timeout=None):
        """Block the calling thread until the coroutine finishes. Never call
        this from the GTK main thread during a long operation."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def shutdown(self):
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=3.0)


def gtk_callback(fn, *args):
    """`GLib.idle_add` that never re-arms, imported lazily so the async modules
    stay importable without GTK (for the CLI harness and tests)."""
    from gi.repository import GLib

    def once():
        fn(*args)
        return False

    GLib.idle_add(once)
