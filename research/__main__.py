"""python -m research serve | ask "question" | doctor | clean | eval"""
import argparse
import asyncio
import os
import subprocess
import sys

from . import config


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m research")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="open the chat UI on localhost")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--headless", action="store_true", help="don't open a browser tab")

    ask = sub.add_parser("ask", help="one question in the terminal (shows every step)")
    ask.add_argument("question")
    ask.add_argument("--preset", default=None, choices=list(config.presets()))
    ask.add_argument("--file", action="append", default=[], help="attach a file (repeatable)")
    ask.add_argument("--mode", default="basic", choices=["basic", "parallel"],
                     help="parallel = several researchers search at once and their findings are compared")

    doc = sub.add_parser("doctor", help="check API keys, models and search APIs")
    doc.add_argument("--claude", action="store_true", help="also send one tiny message to Claude (uses subscription)")

    cln = sub.add_parser("clean", help="trim caches and old logs (limits: storage in models.yaml)")
    cln.add_argument("--all", action="store_true",
                     help="remove every cached file, page, log and saved conversation (not the models)")

    ev = sub.add_parser("eval", help="score each judge model on eval/claims.jsonl")
    ev.add_argument("--file", default=str(config.ROOT / "eval" / "claims.jsonl"))

    a = ap.parse_args()
    if a.cmd == "serve":
        app = config.ROOT / "research" / "app.py"
        args = [sys.executable, "-m", "research.serve", str(app), "--port", str(a.port)] + (["--headless"] if a.headless else [])
        sys.exit(subprocess.call(args, cwd=config.ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}))
    if a.cmd == "ask":
        asyncio.run(_ask(a.question, a.preset or config.default_preset(), a.file, a.mode))
    elif a.cmd == "doctor":
        from .doctor import run
        run(check_claude=a.claude)
    elif a.cmd == "clean":
        from . import cleanup
        print("before:", cleanup.usage())
        print(cleanup.prune(0, 0, 0, 0) if a.all else cleanup.prune())
        print("after: ", cleanup.usage())
    elif a.cmd == "eval":
        from .evaluate import run
        run(a.file)


class ConsoleListener:
    """Prints steps as they happen. In parallel mode researchers run at once, so their lines
    interleave; each is tagged with its researcher, e.g. "[papers]"."""

    def on_start(self, span):
        if span.kind == "tool":
            print(f"{'  ' * span.depth}{_lane(span)}▶ {span.name}", flush=True)

    def on_end(self, span):
        pad = "  " * span.depth + _lane(span)
        if span.kind == "llm":
            print(f"{pad}· {span.name}{f' [{span.model}]' if span.model else ''} ({span.seconds:.1f}s)", flush=True)
            text = span.reasoning or span.output
            if text:
                print(pad + "  " + text[:300].replace("\n", " ") + ("…" if len(text) > 300 else ""), flush=True)
        else:
            print(f"{pad}✓ {span.name} ({span.seconds:.1f}s)", flush=True)


def _lane(span) -> str:
    return f"[{span.lane.removeprefix('r:')}] " if span.lane.startswith("r:") else ""


async def _ask(question: str, preset: str, attachments: list[str], mode: str = "basic") -> None:
    from . import judge, parallel
    from .agents import make_agent
    from .awake import awake
    from .engine import Session

    session = Session(ConsoleListener())
    session.mode = mode
    images: list[str] = []
    if attachments:
        summary, images = session.add_files([(path, None) for path in attachments])
        question = "[첨부 파일]\n" + summary + "\n\n" + question
    if mode == "parallel":
        question = parallel.turn_text(question)
    agent = make_agent(preset, session)
    with awake():  # don't let the PC sleep in the middle of a long question
        await agent.start()
        try:
            answer = await agent.turn(question, images)
        finally:
            await agent.close()
        report = await asyncio.to_thread(judge.check, session, answer)
    print("\n" + "=" * 60 + "\n" + answer + "\n\n" + judge.render(report, session, answer))
    print(f"\n(trace: {session.tracer.path})")


if __name__ == "__main__":
    main()
