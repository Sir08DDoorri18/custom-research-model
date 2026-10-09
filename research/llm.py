"""Calls to the free models (Groq, Mistral, OpenRouter, NVIDIA NIM).

All four speak the OpenAI chat API, so one client covers them. A role in models.yaml lists
models in order; ask() moves to the next one when a call fails, and every call is traced
with its reasoning so it can be inspected later.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from . import config, trace
from .config import ModelRef, Provider


class LLMError(RuntimeError):
    pass


class NoModelAvailable(LLMError):
    pass


@dataclass
class Reply:
    text: str
    reasoning: str
    model: ModelRef


_clients: dict[str, OpenAI] = {}
_last_call: dict[str, float] = {}
_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_benched: dict[str, tuple[float, str]] = {}   # model -> (until, why): skipped instead of retried

# How long a model sits out after each kind of failure. Without this, every call to a dead
# model waits for its timeout again (one NIM model hung for over an hour across a single eval).
BENCH = {"rate": 120, "timeout": 600, "gone": 1800, "error": 60, "network": 30}


DEFAULT_TIMEOUT = 150  # seconds; reasoning models (DeepSeek on NIM) took ~80 s per judging batch in evals


def _client(p: Provider) -> OpenAI:
    if p.name not in _clients:
        timeout = httpx.Timeout(DEFAULT_TIMEOUT, connect=10)
        _clients[p.name] = OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=timeout, max_retries=0)
    return _clients[p.name]


def _within(seconds: float, fn, *args, **kwargs):
    """fn(...) with a hard wall-clock limit. The HTTP timeout only limits the gap between bytes,
    so a server that trickles a byte now and then can hold a call for many minutes (one held
    a researcher for 15 min). Past the limit the call is abandoned in its thread."""
    box: dict = {}

    def run():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:  # handed to the caller below
            box["error"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"no complete reply within {seconds:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _bench(ref: ModelRef, kind: str, why: str) -> None:
    _benched[str(ref)] = (time.time() + BENCH[kind], why)


def benched(ref: ModelRef) -> str | None:
    """Why the model is sitting out, or None if it may be called."""
    entry = _benched.get(str(ref))
    if entry and entry[0] > time.time():
        return entry[1]
    return None


def _wait_turn(p: Provider) -> None:
    """Space requests to one provider at least min_interval apart (free tiers count requests)."""
    with _locks[p.name]:
        wait = _last_call.get(p.name, 0) + p.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[p.name] = time.time()


def usable(ref: ModelRef) -> bool:
    """Has a key and isn't sitting out after a recent failure."""
    if ref.provider == "claude":
        return bool(shutil.which("claude")) and not benched(ref)
    p = config.providers().get(ref.provider)
    return bool(p and p.api_key) and not benched(ref)


def call(ref: ModelRef, messages: list[dict], label: str, timeout: float | None = None) -> Reply:
    """One request to one model. `timeout` (seconds) overrides the default 150 s limit."""
    if ref.provider == "claude":
        return _call_claude(ref, messages, label, timeout)
    p = config.providers().get(ref.provider)
    if p is None:
        raise LLMError(f"{ref}: unknown provider '{ref.provider}' (see providers in models.yaml)")
    if not p.api_key:
        raise LLMError(f"{ref}: {p.key_env} is not set in .env")
    if why := benched(ref):
        raise LLMError(f"{ref}: 잠시 제외됨 ({why})")
    with trace.span(label, "llm", input=_show(messages)) as s:
        s.model = str(ref)
        extra = p.extra_body
        for attempt in range(3):
            _wait_turn(p)
            client = _client(p)
            if timeout:
                client = client.with_options(timeout=httpx.Timeout(timeout, connect=10))
            try:
                resp = _within((timeout or DEFAULT_TIMEOUT) + 10, client.chat.completions.create,
                               model=ref.model, messages=messages, extra_body=extra)
                break
            except APIStatusError as e:
                if e.status_code == 400 and extra:     # provider rejected our extra options: retry plain
                    extra = None
                    continue
                if e.status_code == 429 and attempt == 0:  # one short wait, then let the next model answer
                    time.sleep(min(float(e.response.headers.get("retry-after") or 3), 5))
                    continue
                kind = ("rate" if e.status_code == 429 else "gone" if e.status_code in (401, 403, 404)
                        else "error")
                _bench(ref, kind, f"HTTP {e.status_code}")
                raise LLMError(f"{ref}: HTTP {e.status_code} {_short(e)}") from e
            except (APITimeoutError, TimeoutError) as e:  # the model hung: sit out long
                _bench(ref, "timeout", type(e).__name__)
                raise LLMError(f"{ref}: {type(e).__name__}") from e
            except APIConnectionError as e:
                # Usually our own network blinking, which hits every model at once: retry after
                # a short wait, and if it persists sit out only briefly instead of 10 minutes.
                if attempt == 0:
                    time.sleep(2)
                    continue
                _bench(ref, "network", type(e).__name__)
                raise LLMError(f"{ref}: {type(e).__name__}") from e
        else:
            _bench(ref, "error", "retries exhausted")
            raise LLMError(f"{ref}: failed after retries")
        if not resp.choices:
            raise LLMError(f"{ref}: empty response")
        text, reasoning = _split_reasoning(resp.choices[0].message)
        s.output, s.reasoning = text, reasoning
        return Reply(text, reasoning, ref)


def ask(role: str, messages: list[dict], label: str, skip: set[ModelRef] = frozenset()) -> Reply:
    """Try the role's models in order; raise NoModelAvailable if none of them answers."""
    return ask_any(config.role(role), messages, label, f"role '{role}'", skip)


def ask_any(refs: list[ModelRef], messages: list[dict], label: str, what: str = "models",
            skip: set[ModelRef] = frozenset(), timeout: float | None = None) -> Reply:
    """Try the models in order; raise NoModelAvailable if none of them answers."""
    errors = []
    for ref in refs:
        if ref in skip:
            continue
        if not usable(ref):
            errors.append(f"{ref}: " + (benched(ref) or "no API key"))
            continue
        try:
            return call(ref, messages, label, timeout=timeout)
        except LLMError as e:
            errors.append(str(e))
    raise NoModelAvailable(f"no model for {what}: " + "; ".join(errors))


def role_available(role: str) -> bool:
    return any(usable(r) for r in config.role(role))


def _call_claude(ref: ModelRef, messages: list[dict], label: str, timeout: float | None) -> Reply:
    """One plain-text exchange with Claude ("claude/haiku") through the local claude CLI, with
    no tools. Used only when a parallel researcher is set to Claude; uses the subscription."""
    if why := benched(ref):
        raise LLMError(f"{ref}: 잠시 제외됨 ({why})")
    prompt = _show(messages)

    async def once() -> tuple[str, str]:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, ThinkingBlock, query
        os.makedirs(config.SANDBOX, exist_ok=True)
        options = ClaudeAgentOptions(model=ref.model, tools=[], allowed_tools=[], permission_mode="dontAsk",
                                     setting_sources=[], strict_mcp_config=True, cwd=config.SANDBOX, max_turns=1)
        text, thinking = "", []
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                thinking += [b.thinking for b in msg.content if isinstance(b, ThinkingBlock) and b.thinking.strip()]
            elif isinstance(msg, ResultMessage):
                if msg.is_error:
                    raise LLMError(f"{ref}: {msg.result or msg.subtype}")
                text = msg.result or ""
        return text, "\n".join(thinking)

    with trace.span(label, "llm", input=prompt) as s:
        s.model = str(ref)
        try:
            text, reasoning = asyncio.run(asyncio.wait_for(once(), timeout or 150))
        except LLMError:
            _bench(ref, "error", "Claude error")
            raise
        except Exception as e:  # CLI missing, logged out, timeout
            _bench(ref, "timeout" if isinstance(e, TimeoutError) else "error", type(e).__name__)
            raise LLMError(f"{ref}: {type(e).__name__}: {str(e)[:120]}") from e
        s.output, s.reasoning = text.strip(), reasoning.strip()
        return Reply(s.output, s.reasoning, ref)


def _split_reasoning(msg) -> tuple[str, str]:
    extra = getattr(msg, "model_extra", None) or {}
    reasoning = getattr(msg, "reasoning", None) or extra.get("reasoning") or extra.get("reasoning_content") or ""
    text = msg.content or ""
    if isinstance(text, list):  # some providers return typed chunks (e.g. Mistral thinking models)
        chunks = [c for c in text if isinstance(c, dict)]
        reasoning = reasoning or "\n".join(str(c.get("thinking", "")) for c in chunks if c.get("type") == "thinking")
        text = "".join(str(c.get("text", "")) for c in chunks if c.get("type") == "text")
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if m:
        reasoning = reasoning or m.group(1).strip()
        text = text.replace(m.group(0), "")
    return text.strip(), str(reasoning).strip()


def _show(messages: list[dict]) -> str:
    def part(p) -> str:  # images are logged as a marker, not as megabytes of base64
        return p.get("text", "") if p.get("type") == "text" else f"[{p.get('type', 'part')}]"

    return "\n\n".join(f"[{m['role']}]\n" + (m["content"] if isinstance(m["content"], str)
                                             else "\n".join(part(p) for p in m["content"]))
                       for m in messages)


def _short(e: APIStatusError) -> str:
    try:
        body = e.response.json()
        err = body.get("error", body)
        return str(err.get("message", err) if isinstance(err, dict) else err)[:200]
    except Exception:
        return e.response.text[:200]


def extract_json(text: str):
    """Pull the first JSON object/array out of a model reply (handles ```json fences)."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = ([fence.group(1)] if fence else []) + [text]
    for cand in candidates:
        # Whichever bracket comes first is the outer one: an object holding a list is an object.
        pairs = sorted((("[", "]"), ("{", "}")), key=lambda p: (cand.find(p[0]) < 0, cand.find(p[0])))
        for opener, closer in pairs:
            start, end = cand.find(opener), cand.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(cand[start:end + 1])
                except json.JSONDecodeError:
                    continue
    raise LLMError(f"no JSON found in reply: {text[:300]}")
