"""Chat UI on localhost (Chainlit). Start it with:  python -m research serve

Every tool call, model call and thinking block shows up as a collapsible step while it runs.
The answer is followed by the sentence-by-sentence verification and the source list.
"""
import asyncio
import logging
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # chainlit loads this file by path

import chainlit as cl
from chainlit.utils import utc_now

# Keep the server console to what matters: silence per-request INFO lines and known
# harmless warnings (a Chainlit-internal un-awaited profile call, torch's 3.14 notice,
# the missing chainlit_ko.md lookup that falls back to chainlit.md as intended).
for noisy in ("httpx", "mcp", "claude_agent_sdk"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("chainlit").addFilter(
    lambda r: not any(s in r.getMessage() for s in ("Translated markdown file", "Missing custom logo")))
warnings.filterwarnings("ignore", message="coroutine 'chat_profiles' was never awaited")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)

from research import cleanup, config, judge, llm, prompts
from research.agents import make_agent
from research.engine import Session
from research.verify import NLI


class UIListener:
    """Tracer callbacks arrive from worker threads; hand them to the UI loop in order."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop, self.queue = loop, queue

    def on_start(self, span):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, ("start", span))

    def on_end(self, span):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, ("end", span))


async def render_steps(queue: asyncio.Queue, ui: dict) -> None:
    """ui["parent"]: id of the current message's run step; top-level spans are nested under it
    so they show up above the answer instead of after it."""
    steps: dict[str, cl.Step] = {}
    while True:
        op, span = await queue.get()
        try:
            if op == "start":
                step = cl.Step(name=span.name, type="llm" if span.kind == "llm" else "tool", id=span.id,
                               parent_id=span.parent_id if span.parent_id in steps else ui.get("parent"),
                               show_input="text")
                step.input = span.input[:20000]
                step.start = utc_now()
                steps[span.id] = step
                await step.send()
            else:
                step = steps.get(span.id)
                if step:
                    if span.model and not span.name.startswith("Claude"):  # Claude steps already say which
                        step.name = f"{span.name} · {span.model.split('/')[-1]}"
                    step.output = _step_output(span)
                    step.end = utc_now()
                    await step.update()
        except Exception as e:  # a display problem must never break the conversation
            print(f"[ui] step render failed: {e}", flush=True)
        finally:
            queue.task_done()


def _step_output(span) -> str:
    parts = []
    if span.reasoning:
        parts.append("**사고 과정**\n\n" + _quote(span.reasoning))
    if span.output:
        parts.append(("**결과**\n\n" if span.reasoning else "") + span.output[:20000])
    return "\n\n".join(parts) or "(내용 없음)"


def _quote(text: str) -> str:
    return "\n".join("> " + line for line in text[:20000].splitlines())


@cl.on_app_startup
async def startup():
    """Trim caches once per server start (limits: `storage` in models.yaml)."""
    print("[cleanup]", await asyncio.to_thread(cleanup.prune), flush=True)


@cl.set_chat_profiles
async def chat_profiles():
    default = config.default_preset()
    return [cl.ChatProfile(name=name, display_name=p.get("label", name),
                           markdown_description="Claude 구독" if p["backend"] == "claude" else "무료 API만",
                           default=name == default)
            for name, p in config.presets().items()]


@cl.on_chat_start
async def start():
    preset = cl.user_session.get("chat_profile") or config.default_preset()
    queue: asyncio.Queue = asyncio.Queue()
    session = Session(UIListener(asyncio.get_running_loop(), queue))
    ui: dict = {"parent": None}
    renderer = asyncio.create_task(render_steps(queue, ui))
    cl.user_session.set("ui", ui)
    # Load the local NLI model now so the first answer's check doesn't wait for it.
    cl.user_session.set("nli_preload", asyncio.create_task(asyncio.to_thread(NLI.get)))
    cl.user_session.set("session", session)
    cl.user_session.set("queue", queue)
    cl.user_session.set("renderer", renderer)
    try:
        # Started on the first message, not here: every opened or reloaded tab starts a chat,
        # and each start would launch a Claude process nobody may use.
        agent = make_agent(preset, session)
    except Exception as e:
        await cl.Message(content=f"! {preset} 시작 실패: {e}").send()
        return
    cl.user_session.set("agent", agent)

    # Say nothing when everything is ready; only report what's missing.
    missing = [name for role, name in (("rcs", "읽기"), ("first", "1차 채점")) if not llm.role_available(role)]
    if missing:
        await cl.Message(content=f"! {', '.join(missing)} 모델 없음 · `.env` 키 확인").send()


@cl.on_message
async def on_message(message: cl.Message):
    run = cl.context.current_step  # Chainlit wraps each message in a "run" step
    parent = run.id if run else message.id
    uploads = [(el.path, el.name) for el in message.elements or [] if getattr(el, "path", None)]
    await turn(message.content, parent, uploads)


async def turn(text: str, parent_id: str | None = None, uploads: list[tuple[str, str]] = ()) -> None:
    agent, session, queue, ui = (cl.user_session.get(k) for k in ("agent", "session", "queue", "ui"))
    ui["parent"] = parent_id
    if agent is None:
        await cl.Message(content="에이전트가 시작되지 않았어요. 새 채팅을 열거나 다른 프리셋을 골라주세요.").send()
        return
    images: list[str] = []
    if uploads:
        summary, images = await asyncio.to_thread(session.add_files, list(uploads))
        text = "[첨부 파일]\n" + summary + "\n\n" + (text or "첨부한 파일을 요약해줘.")
    try:
        await agent.start()
        answer = await agent.turn(text, images)
    except Exception as e:
        await queue.join()
        await cl.Message(content=f"! 오류: {e}").send()
        return
    # Nothing to verify when the answer cites no evidence (textbook answers, small talk).
    report = await asyncio.to_thread(judge.check, session, answer) if judge.CITE.search(answer or "") else None
    await queue.join()  # all steps on screen before the answer

    content = _fix_bold(answer) or "(빈 답변)"
    actions = [cl.Action(name="deeper", payload={}, label="더 깊게"),
               cl.Action(name="counter", payload={}, label="반론 찾기")]
    if report and (report.claims or report.uncited):
        content += "\n\n---\n\n" + judge.render(report, session, answer)
        problems = [f"- {c.sentence} ({c.reason})" for c in report.problems()]
        problems += [f"- {u} (출처 없음)" for u in report.uncited]
        if problems:
            actions.insert(0, cl.Action(name="fix", payload={"problems": "\n".join(problems)}, label="걸린 문장 고치기"))
    await cl.Message(content=content, actions=actions).send()


def _fix_bold(text: str) -> str:
    """Markdown can't close **bold** right before a Korean particle (**28%**를), so the
    asterisks would show literally. Drop the markers in that case."""
    return re.sub(r"\*\*([^*\n]+?)\*\*(?=[가-힣])", r"\1", text or "")


async def _follow_up(text: str, shown: str) -> None:
    msg = cl.Message(content=shown, type="user_message")
    await msg.send()
    await turn(text, msg.id)


@cl.action_callback("deeper")
async def deeper(action: cl.Action):
    await _follow_up(prompts.DEEPER, "더 깊게 조사해줘")


@cl.action_callback("counter")
async def counter(action: cl.Action):
    await _follow_up(prompts.COUNTER, "반론을 찾아줘")


@cl.action_callback("fix")
async def fix(action: cl.Action):
    await _follow_up(prompts.FIX.format(problems=action.payload.get("problems", "")), "걸린 문장을 고쳐줘")


@cl.on_stop
async def stop():
    agent = cl.user_session.get("agent")
    client = getattr(agent, "client", None)
    if client:
        await client.interrupt()


@cl.on_chat_end
async def end():
    agent, renderer = cl.user_session.get("agent"), cl.user_session.get("renderer")
    if agent:
        await agent.close()
    if renderer:
        renderer.cancel()
