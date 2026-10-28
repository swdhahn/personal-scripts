#!/usr/bin/env python3
"""Constrained decoding for the 0.5B model.

The 0.5B model already runs inside llama-server, which supports GBNF grammars
natively -- so instead of pulling in `outlines`/`instructor` (both of which
would load a *second* copy of the weights in-process through transformers) the
grammar is generated from Python `Enum`/field specs and sent with the request.
llama.cpp then masks the logits itself, which means:

  * zero extra dependencies, zero extra VRAM,
  * decoding is literally incapable of emitting an invalid token,
  * an enum decision costs a handful of tokens (single-digit milliseconds).

Fallback ladder, in case a given llama.cpp build rejects `grammar` on the
OpenAI-compatible route:

  1. `grammar=<gbnf>`                      (llama.cpp native, preferred)
  2. `response_format={"type":"json_schema",...}`
  3. unconstrained + tolerant parse        (last resort, may fail -> default)
"""
from __future__ import annotations

import asyncio
import json
import re
from enum import Enum
from typing import Any, Sequence

import requests

# --------------------------------------------------------------------------- #
# GBNF building blocks
# --------------------------------------------------------------------------- #
_GBNF_PRELUDE = r"""
ws      ::= [ \t\n]*
string  ::= "\"" char* "\""
char    ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
integer ::= "-"? ([0-9] | [1-9] [0-9]{0,9})
boolean ::= "true" | "false"
"""


def _lit(value: str) -> str:
    return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')


def enum_values(spec) -> list[str]:
    """Accepts an Enum class or a plain sequence of strings."""
    if isinstance(spec, type) and issubclass(spec, Enum):
        return [str(member.value) for member in spec]
    return [str(v) for v in spec]


def enum_grammar(spec) -> str:
    """Grammar for a bare, unquoted enum token: the cheapest possible decision.

    >>> enum_grammar(["SMALL", "LARGE"])
    'root ::= "SMALL" | "LARGE"'
    """
    values = enum_values(spec)
    if not values:
        raise ValueError("enum_grammar needs at least one value")
    return "root ::= " + " | ".join(_lit(v) for v in values)


def _field_rule(name: str, spec: dict[str, Any]) -> tuple[str, str]:
    """Returns (inline_rule_reference, extra_rule_text)."""
    if "enum" in spec:
        rule = "fld-%s" % name
        body = " | ".join(_lit('"%s"' % v) for v in enum_values(spec["enum"]))
        return rule, "%s ::= %s\n" % (rule, body)
    kind = spec.get("type", "string")
    if kind == "integer":
        return "integer", ""
    if kind == "number":
        return "integer", ""
    if kind == "boolean":
        return "boolean", ""
    if kind == "array":
        item, extra = _field_rule(name + "-item", spec.get("items", {"type": "string"}))
        rule = "fld-%s" % name
        text = '%s ::= "[" ws (%s (ws "," ws %s)*)? ws "]"\n' % (rule, item, item)
        return rule, extra + text
    return "string", ""


def json_object_grammar(fields: dict[str, dict[str, Any]]) -> str:
    """Grammar for a *fixed-key-order, no-extra-keys* JSON object.

    Fixing the key order is what makes the grammar tiny and makes a 0.5B model
    reliable: it cannot omit a field, reorder them, hallucinate a key, or keep
    rambling after the closing brace.

        json_object_grammar({
            "query":      {"type": "string"},
            "freshness":  {"enum": ["ANY", "RECENT"]},
            "max_results":{"type": "integer"},
        })
    """
    if not fields:
        raise ValueError("json_object_grammar needs at least one field")
    extras, parts = [], []
    for index, (name, spec) in enumerate(fields.items()):
        ref, extra = _field_rule(name, spec)
        if extra:
            extras.append(extra)
        sep = ' ws "," ws ' if index else " ws "
        parts.append('%s%s ws ":" ws %s' % (sep, _lit('"%s"' % name), ref))
    root = 'root ::= "{" %s ws "}"' % "".join(parts)
    return root + "\n" + "".join(extras) + _GBNF_PRELUDE


def json_schema_of(fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The same field spec expressed as JSON Schema, for the fallback path."""
    props: dict[str, Any] = {}
    for name, spec in fields.items():
        if "enum" in spec:
            props[name] = {"type": "string", "enum": enum_values(spec["enum"])}
        elif spec.get("type") == "array":
            props[name] = {"type": "array", "items": {"type": "string"}}
        else:
            props[name] = {"type": spec.get("type", "string")}
    return {
        "type": "object",
        "properties": props,
        "required": list(fields),
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class ConstrainedDecoder:
    """Runs strictly-typed decisions on a small llama-server endpoint."""

    def __init__(self, client, *, temperature: float = 0.0, debug: bool = False):
        self.client = client
        self.temperature = temperature
        self.debug = debug
        self._grammar_supported: bool | None = None
        # Keys the model failed to supply on the last extract(); callers use
        # this to decide whether to trust the result or fall back safely.
        self.last_missing: set[str] = set()

    # ------------------------------------------------------------- primitives
    async def choose(
        self,
        spec,
        system: str,
        user: str,
        *,
        examples: Sequence[tuple[str, str]] = (),
        default=None,
        max_tokens: int = 12,
    ):
        """Decode exactly one enum member. Returns the Enum member (or str).

        `examples` are injected as real user/assistant turn pairs, not as text
        inside the system prompt. On a 0.5B this is the difference between the
        few-shot being followed and being ignored: measured on
        Qwen2.5-0.5B-Instruct, moving the same examples out of the system
        prompt and into turn pairs took the memory classifier from 7/15 to
        13/15, including cases quoted verbatim in the prompt that the in-prompt
        version still got wrong.
        """
        values = enum_values(spec)
        prompt = (
            f"{system}\n\nAnswer with exactly one of: " + ", ".join(values) + "."
        )
        messages = [{"role": "system", "content": prompt}]
        for sample_user, sample_label in examples:
            messages.append({"role": "user", "content": sample_user})
            messages.append({"role": "assistant", "content": str(sample_label)})
        messages.append({"role": "user", "content": user})

        raw = await self._decode(
            messages=messages,
            grammar=enum_grammar(values),
            schema=None,
            max_tokens=max_tokens,
        )
        picked = self._match_enum(raw, values)
        if picked is None:
            picked = enum_values([default])[0] if default is not None else values[0]
        if isinstance(spec, type) and issubclass(spec, Enum):
            return spec(picked)
        return picked

    async def extract(
        self,
        fields: dict[str, dict[str, Any]],
        system: str,
        user: str,
        *,
        examples: Sequence[tuple[str, str]] = (),
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        """Decode a flat JSON object matching `fields` exactly.

        `examples` are (request, json_answer) pairs, injected as turn pairs for
        the same reason as in choose(): a 0.5B ignores examples that sit in the
        system prompt. With the Arch-vs-Debian example in the system prompt the
        router kept proposing `sudo apt-get install pipewire`; as a turn pair it
        proposes `pacman -Q pipewire`.
        """
        messages = [{"role": "system", "content": system}]
        for sample_user, sample_json in examples:
            messages.append({"role": "user", "content": sample_user})
            messages.append({"role": "assistant", "content": sample_json})
        messages.append({"role": "user", "content": user})

        raw = await self._decode(
            messages=messages,
            grammar=json_object_grammar(fields),
            schema=json_schema_of(fields),
            max_tokens=max_tokens,
        )
        data = _loose_json(raw)
        if data is None:
            self.last_missing = set(fields)
            return _coerce({}, fields)
        values, missing = _coerce_strict(data, fields)
        self.last_missing = missing
        return values

    # ----------------------------------------------------------------- plumbing
    async def _decode(self, *, messages, grammar, schema, max_tokens) -> str:
        params = {"temperature": self.temperature, "max_tokens": max_tokens,
                  "cache_prompt": True}

        if self._grammar_supported is not False and grammar:
            try:
                out = await self.client.complete(
                    messages, timeout=30.0, grammar=grammar, **params
                )
                self._grammar_supported = True
                return self._log(out)
            except requests.exceptions.HTTPError as exc:
                status = getattr(exc.response, "status_code", 0)
                if status and 400 <= status < 500:
                    self._grammar_supported = False     # build lacks passthrough
                else:
                    raise
            except asyncio.CancelledError:
                raise

        if schema:
            try:
                out = await self.client.complete(
                    messages,
                    timeout=30.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "decision", "strict": True,
                                        "schema": schema},
                    },
                    **params,
                )
                return self._log(out)
            except requests.exceptions.HTTPError:
                pass

        return self._log(await self.client.complete(messages, timeout=30.0, **params))

    def _log(self, raw: str) -> str:
        if self.debug:
            print(f"[constrained] -> {raw!r}")
        return raw or ""

    @staticmethod
    def _match_enum(raw: str, values: Sequence[str]) -> str | None:
        text = (raw or "").strip().strip('"').strip()
        for value in values:
            if text == value:
                return value
        upper = text.upper()
        for value in values:                      # tolerate the unconstrained path
            if value.upper() in upper:
                return value
        return None


# --------------------------------------------------------------------------- #
# Tolerant parsing / coercion (only ever needed on the unconstrained fallback)
# --------------------------------------------------------------------------- #
def _loose_json(raw: str) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            return None
    return None


def _coerce(data: dict, fields: dict[str, dict[str, Any]]) -> dict:
    return _coerce_strict(data, fields)[0]


def _coerce_strict(data: dict, fields: dict[str, dict[str, Any]]):
    """Returns (values, missing_keys).

    A field falls back to its declared `default` -- never silently to the first
    enum member, which for a routing decision would mean quietly picking the
    weakest model on every malformed reply.
    """
    out: dict[str, Any] = {}
    missing: set[str] = set()
    for name, spec in fields.items():
        value = data.get(name)
        if name not in data or value is None:
            missing.add(name)
        if "enum" in spec:
            allowed = enum_values(spec["enum"])
            fallback = str(spec.get("default") or allowed[0])
            text = str(value or "").strip().upper()
            match = next((a for a in allowed if a.upper() == text), None)
            if match is None:
                missing.add(name)
            out[name] = match or fallback
        elif spec.get("type") == "integer":
            try:
                out[name] = int(value)
            except (TypeError, ValueError):
                out[name] = int(spec.get("default", 0))
        elif spec.get("type") == "boolean":
            out[name] = bool(value) if isinstance(value, bool) else str(value).lower() == "true"
        elif spec.get("type") == "array":
            out[name] = [str(v) for v in value] if isinstance(value, list) else []
        else:
            out[name] = "" if value is None else str(value)
    return out, missing
