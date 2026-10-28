#!/usr/bin/env python3
"""Headless harness for the router/tool/commentary pipeline -- no GTK, no daemon.

    ./orchestrate_cli.py --grammar                  # print generated GBNF, offline
    ./orchestrate_cli.py --vet 'rm -rf /'           # exercise the CLI policy, offline
    ./orchestrate_cli.py --route "what did we say about kokoro yesterday?"
    ./orchestrate_cli.py "how much disk space is left?"
    ./orchestrate_cli.py --speak "latest llama.cpp release notes"   # with TTS

Needs AI Mode on (both llama-servers up) for anything that hits a model.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import memory_db                                    # noqa: E402
from constrained import enum_grammar, json_object_grammar   # noqa: E402
from orchestrator import MemoryGateway, Orchestrator        # noqa: E402
from router import (CLI_PARAM_FIELDS, DECISION_FIELDS, Route,  # noqa: E402
                    WEB_PARAM_FIELDS)
from tools_local import CliSandboxTool, CommandRejected      # noqa: E402


def load_cfg():
    import ai_assistant
    return ai_assistant.load_config()


def show_grammar():
    print("=== enum grammar (Route) ===")
    print(enum_grammar(Route))
    print("\n=== router decision grammar ===")
    print(json_object_grammar(DECISION_FIELDS))
    print("=== web params grammar ===")
    print(json_object_grammar(WEB_PARAM_FIELDS))
    print("=== cli params grammar ===")
    print(json_object_grammar(CLI_PARAM_FIELDS))


def vet(config, command):
    tool = CliSandboxTool(config)
    try:
        argv = tool.vet(command)
    except CommandRejected as exc:
        print(f"REJECTED: {exc}")
        return 1
    wrapped, label = tool._wrap(argv, tool.workspace)
    print(f"ALLOWED  argv={argv}")
    print(f"approval_required={tool.needs_approval(argv)}  sandbox={label}")
    print("wrapped:", " ".join(wrapped))
    return 0


async def run(config, prompt, *, route_only=False, speak=False):
    conn = memory_db.open_db(config["memory_db_path"], check_same_thread=False)
    gateway = MemoryGateway(conn, threading.Lock())

    async def say(text):
        print(f"\n  \033[36m[spoken] {text}\033[0m")
        if speak:
            import ai_assistant
            await asyncio.to_thread(
                ai_assistant.send_cmd, config["socket_path"],
                {"command": "speak", "text": text, "wait": True}, True, 60.0,
            )

    async def approve(command, purpose):
        print(f"\n  [approval] {command}   ({purpose})")
        return (await asyncio.to_thread(input, "  run it? [y/N] ")).strip().lower() == "y"

    orch = Orchestrator(config, gateway, speak=say, approval_cb=approve, debug=True)

    if route_only:
        decision = await orch.router.decide(prompt)
        print(json.dumps({
            "route": decision.route.value, "memory": decision.memory.value,
            "tool": decision.tool.value, "urgency": decision.urgency.value,
            "params": decision.params, "latency_ms": round(decision.latency_ms),
            "degraded": decision.degraded,
        }, indent=2))
        return

    async for event in orch.handle(prompt):
        if event.kind == "delta":
            sys.stdout.write(event.payload)
            sys.stdout.flush()
        elif event.kind == "decision":
            print(f"\033[33m[router] {event.payload.summary()} "
                  f"params={event.payload.params}\033[0m")
        elif event.kind == "recall":
            print(f"\033[33m[memory] {len(event.payload)} recalled\033[0m")
        elif event.kind in ("tool_start", "tool_result"):
            text = event.payload.get("text") or event.payload.get("arguments")
            print(f"\033[33m[{event.kind}] {event.payload['name']}: "
                  f"{str(text)[:300]}\033[0m")
        elif event.kind == "context":
            info = event.payload
            line = (f"\033[33m[context] {info['tokens']}/{info['limit']} tok "
                    f"(n_ctx {info['n_ctx']})")
            if info["steps"]:
                line += "  compressed: " + "; ".join(info["steps"])
            if info["overflowed"]:
                line += "  OVERFLOW"
            print(line + "\033[0m")
        elif event.kind == "notice":
            print(f"\033[33m[notice] {event.payload}\033[0m")
        elif event.kind == "error":
            print(f"\033[31m[error] {event.payload}\033[0m")
        elif event.kind == "done":
            print(f"\n\033[32m[done in {event.payload['elapsed_s']:.1f}s]\033[0m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*")
    ap.add_argument("--grammar", action="store_true", help="print generated GBNF and exit")
    ap.add_argument("--vet", metavar="CMD", help="run a command through the CLI policy only")
    ap.add_argument("--route", action="store_true", help="router decision only")
    ap.add_argument("--speak", action="store_true", help="send commentary to Kokoro")
    args = ap.parse_args()

    if args.grammar:
        show_grammar()
        return
    config = load_cfg()
    if args.vet:
        sys.exit(vet(config, args.vet))
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        ap.error("need a prompt")
    asyncio.run(run(config, prompt, route_only=args.route, speak=args.speak))


if __name__ == "__main__":
    main()
