"""Chat UI on localhost (Chainlit). Start it with:  python -m research serve

Basic mode: every tool call, model call and thinking block shows up as a collapsible step
while it runs. Parallel mode: one panel instead, with a column per researcher
(public/elements/ResearchPanel.jsx, data from research/board.py). The mode is picked in the
menu bar (public/rail.js), which talks to the /research/prefs endpoint below.
A cited answer opens the answer window on the right (research/answer_view.py).
Conversations are saved on this PC (research/history.py) and listed under 지난 대화.
"""
import asyncio
import logging
import re
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # chainlit loads this file by path

import chainlit as cl
from chainlit.server import app as server
from chainlit.utils import utc_now
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

# Keep the server console to what matters: silence per-request INFO lines and known
# harmless warnings (a Chainlit-internal un-awaited profile call, torch's 3.14 notice,
# the missing chainlit_ko.md lookup that falls back to chainlit.md as intended).
for noisy in ("httpx", "httpx2", "primp", "mcp", "claude_agent_sdk"):  # httpx2/primp: web search requests
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("chainlit").addFilter(
    lambda r: not any(s in r.getMessage() for s in ("Translated markdown file", "Missing custom logo")))
warnings.filterwarnings("ignore", message="coroutine 'chat_profiles' was never awaited")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)

from research import answer_view, checkpoint, cleanup, config, history, judge, llm, parallel, prefs, prompts
from research.agents import make_agent
from research.awake import awake
from research.board import Board
from research.engine import Session
from research.verify import NLI

PANEL_INTERVAL = 0.8  # seconds between panel refreshes while researchers work
IDLE_LIMIT = 360      # seconds without any sign of work before a turn counts as stalled
ANSWER_WINDOW = "답변창"  # element name = public/elements/답변창.jsx; Chainlit links this word in messages
PROGRESS_WINDOW = "중간자료"  # side text with what a cut-off question had found
RUNNING: set[str] = set()  # conversations with a question being worked on in this server process


# ---------- saved conversations and the menu bar's endpoints ----------

@cl.data_layer
def data_layer():
    return history.DataLayer()


@cl.header_auth_callback
async def local_user(headers) -> cl.User:
    """One user, logged in automatically: saved conversations need a user to belong to."""
    return cl.User(identifier="local", metadata={"provider": "local"})


async def _get_prefs(request: Request) -> Response:
    return JSONResponse(prefs.load())


async def _set_prefs(request: Request) -> Response:
    try:
        return JSONResponse(prefs.save(mode=(await request.json()).get("mode")))
    except (ValueError, AttributeError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def _history_file(request: Request) -> Response:
    try:
        path = history.file_path(request.path_params["key"])
    except ValueError:
        return Response(status_code=404)
    return FileResponse(path) if path.is_file() else Response(status_code=404)


async def _public_file(request: Request) -> Response:
    """public/ (styles, menu bar, panels) as Chainlit serves it, but with Cache-Control: browsers
    kept old copies after an update. no-cache = check the ETag each time (unchanged: a 304)."""
    base = (config.ROOT / "public").resolve()
    path = (base / request.path_params["filename"]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        return Response(status_code=404)
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


# Ahead of Chainlit's catch-all route, which would otherwise answer every GET with the page.
server.router.routes[0:0] = [Route("/research/prefs", _get_prefs, methods=["GET"]),
                             Route("/research/prefs", _set_prefs, methods=["POST"]),
                             Route(history.FILES_URL + "{key:path}", _history_file, methods=["GET"]),
                             Route("/public/{filename:path}", _public_file, methods=["GET"])]


# ---------- live steps ----------

class UIListener:
    """Tracer callbacks arrive from worker threads; hand them to the UI loop in order."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, ui: dict):
        self.loop, self.queue, self.ui = loop, queue, ui

    def on_start(self, span):
        self.ui["active"] = time.monotonic()
        self.loop.call_soon_threadsafe(self.queue.put_nowait, ("start", span))

    def on_end(self, span):
        self.ui["active"] = time.monotonic()
        self.loop.call_soon_threadsafe(self.queue.put_nowait, ("end", span))


async def render_steps(queue: asyncio.Queue, ui: dict) -> None:
    """ui["parent"]: id of the current message's run step; top-level spans are nested under it
    so they show up above the answer instead of after it. ui["board"]: set during a parallel
    turn, which shows everything in its panel instead of as steps."""
    steps: dict[str, cl.Step] = {}
    while True:
        op, span = await queue.get()
        try:
            if board := ui.get("board"):
                board.on_start(span) if op == "start" else board.on_end(span)
            elif op == "start":
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


# ---------- chat lifecycle ----------

@cl.on_app_startup
async def startup():
    """Trim caches and old conversations once per server start (limits: `storage` in models.yaml)."""
    print("[cleanup]", await asyncio.to_thread(cleanup.prune), flush=True)


@cl.set_chat_profiles
async def chat_profiles():
    default = config.default_preset()
    return [cl.ChatProfile(name=name, display_name=p.get("label", name),
                           markdown_description="Claude 구독" if p["backend"] == "claude" else "무료 API만",
                           default=name == default)
            for name, p in config.presets().items()]


async def _setup(preset: str) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ui: dict = {"parent": None, "active": time.monotonic()}
    session = Session(UIListener(asyncio.get_running_loop(), queue, ui))
    for key, value in (("ui", ui), ("session", session), ("queue", queue), ("transcript", []),
                       ("renderer", asyncio.create_task(render_steps(queue, ui))),
                       # Load the local NLI model now so the first answer's check doesn't wait for it.
                       ("nli_preload", asyncio.create_task(asyncio.to_thread(NLI.get)))):
        cl.user_session.set(key, value)
    try:
        # Started on the first message, not here: every opened or reloaded tab starts a chat,
        # and each start would launch a Claude process nobody may use.
        cl.user_session.set("agent", make_agent(preset, session))
    except Exception as e:
        await cl.Message(content=f"! {preset} 시작 실패: {e}").send()


@cl.on_chat_start
async def start():
    await _setup(cl.user_session.get("chat_profile") or config.default_preset())
    # Say nothing when everything is ready; only report what's missing.
    missing = [name for role, name in (("rcs", "읽기"), ("first", "1차 채점")) if not llm.role_available(role)]
    if missing:
        await cl.Message(content=f"! {', '.join(missing)} 모델 없음 · `.env` 키 확인").send()


@cl.on_chat_resume
async def resume(thread: dict):
    """A saved conversation reopened: the chat model starts fresh, so its earlier questions and
    answers go along with the next question. (Answer windows come back with their messages.)"""
    # The preset the conversation was started with: its tag (set when the thread was created).
    # metadata["chat_profile"] is overwritten with the header's default when the page reconnects.
    presets = config.presets()
    preset = next((t for t in thread.get("tags") or [] if t in presets), None)
    await _setup(preset or (thread.get("metadata") or {}).get("chat_profile") or config.default_preset())
    earlier = history.previous_turns(thread)
    if earlier:
        cl.user_session.set("transcript", [earlier])
        cl.user_session.set("carry", True)
    data = checkpoint.load(thread.get("id"))
    if not data:
        return
    cl.user_session.get("session").restore(data)  # E-ids cited earlier stay valid for checking
    if data.get("status") == "running":
        if thread.get("id") in RUNNING:  # only the page was reloaded; the question is still being worked on
            await cl.Message(content="이 대화의 마지막 질문은 아직 진행 중이에요. 끝나면 이 대화에 답이 저장돼요. "
                                     "잠시 뒤 `지난 대화`에서 다시 열어보세요.").send()
        else:
            # Sent a moment later: the page is still loading the saved messages, and a message sent
            # now gets replaced by its saved copy, which has no buttons.
            asyncio.create_task(_offer_resume(data, "이 대화의 마지막 질문이 답을 내기 전에 끊겼어요.", delay=1.5))


@cl.on_message
async def on_message(message: cl.Message):
    run = cl.context.current_step  # Chainlit wraps each message in a "run" step
    parent = run.id if run else message.id
    uploads = [(el.path, el.name) for el in message.elements or [] if getattr(el, "path", None)]
    await turn(message.content, parent, uploads)


class Stalled(Exception):
    pass


async def turn(text: str, parent_id: str | None = None, uploads: list[tuple[str, str]] = (),
               mode: str | None = None) -> None:
    agent, session, queue, ui = (cl.user_session.get(k) for k in ("agent", "session", "queue", "ui"))
    ui["parent"] = parent_id
    if agent is None:
        await cl.Message(content="에이전트가 시작되지 않았어요. 새 채팅을 열거나 다른 프리셋을 골라주세요.").send()
        return
    # A cut-off question is continued by the 이어서 하기 button or a message like "이어서 해줘";
    # anything else is a new question.
    pending = cl.user_session.get("resume_work")
    resuming = bool(pending) and _wants_continue(text)
    cl.user_session.set("resume_work", None)
    question = pending["question"] if resuming else text
    session.mode = (pending["mode"] if resuming else None) or mode or prefs.mode()
    session.stopped.clear()
    session.progress = list(pending.get("progress") or []) if resuming else []
    thread_id = cl.context.session.thread_id
    session.on_progress = lambda: checkpoint.save(thread_id, session, question, session.mode, "running")
    session.on_progress()
    RUNNING.add(thread_id)
    panel = _Panel(agent.label, ui) if session.mode == "parallel" else None
    try:
        with awake():  # a long question must not be cut off by the PC going to sleep
            if panel:
                await panel.open()
            try:
                images: list[str] = []
                if resuming:
                    text = checkpoint.continuation(pending, "" if _wants_continue(text, strict=True) else text)
                if uploads:
                    summary, images = await asyncio.to_thread(session.add_files, list(uploads))
                    text = "[첨부 파일]\n" + summary + "\n\n" + (text or "첨부한 파일을 요약해줘.")
                if cl.user_session.get("carry"):  # the chat model lost its memory (reopened or reset)
                    text = (f"[앞선 대화 · 참고용]\n{_transcript()}\n\n[이번 질문]\n{text}")
                    cl.user_session.set("carry", False)
                if panel:
                    text = parallel.turn_text(text)
                try:
                    answer = await _run_agent(agent, text, images, ui)
                except Stalled:
                    await queue.join()
                    await _offer_resume(_progress(session, question),
                                        f"{IDLE_LIMIT // 60}분 넘게 진행이 없어 멈췄어요 (절전, 연결 끊김 등).")
                    return
                except Exception as e:
                    await queue.join()
                    await _offer_resume(_progress(session, question), f"오류로 멈췄어요: {e}")
                    return
                # Nothing to verify when the answer cites no evidence (textbook answers, small talk).
                report = await asyncio.to_thread(judge.check, session, answer) if judge.CITE.search(answer or "") else None
                await queue.join()  # all steps on screen before the answer
            finally:
                if panel:
                    await queue.join()
                    await panel.close()
        checkpoint.save(thread_id, session, question, session.mode, "done")
    finally:
        session.on_progress = None
        RUNNING.discard(thread_id)

    _remember(question, answer)
    tag = "[병렬]" if session.mode == "parallel" else "[기본]"
    content = _fix_markdown(answer) or "(빈 답변)"
    actions = [cl.Action(name="deeper", payload={}, label="더 깊게"),
               cl.Action(name="counter", payload={}, label="반론 찾기")]
    if session.mode == "basic" and question and not uploads:
        actions.append(cl.Action(name="parallel", payload={"question": question[:4000]}, label="병렬 조사로 다시"))
    view = None
    if report and (report.claims or report.uncited):
        # Verdicts per sentence and the sources go to the answer window; the chat keeps a summary.
        view = answer_view.build(question or text, answer, report, session, session.mode)
        bits = [f"검증 {report.summary()}"] + ([f"출처 없는 수치 {len(report.uncited)}"] if report.uncited else [])
        content += f"\n\n---\n\n_{tag} " + " · ".join(bits) + f" — 문장별 판정과 출처: {ANSWER_WINDOW}_"
        problems = [f"- {c.sentence} ({c.reason})" for c in report.problems()]
        problems += [f"- {u} (출처 없음)" for u in report.uncited]
        if problems:
            actions.insert(0, cl.Action(name="fix", payload={"problems": "\n".join(problems)}, label="걸린 문장 고치기"))
    else:
        content += f"\n\n_{tag}_"
    # The window is an element of the message, so it is saved with the conversation; the word
    # ANSWER_WINDOW in the text becomes the link that reopens it (buttons are not saved).
    elements = [cl.CustomElement(name=ANSWER_WINDOW, props=view, display="side")] if view else []
    await cl.Message(content=content, actions=actions, elements=elements).send()
    if view:
        await _open_view(view)


async def _run_agent(agent, text: str, images: list[str], ui: dict) -> str:
    """The chat model's turn, watched: if nothing at all happens for IDLE_LIMIT seconds (the PC
    slept, the Claude connection dropped), give up instead of waiting forever."""
    ui["active"] = time.monotonic()
    await agent.start()
    task = asyncio.create_task(agent.turn(text, images))
    while True:
        done, _ = await asyncio.wait({task}, timeout=15)
        if done:
            return task.result()
        if time.monotonic() - ui["active"] > IDLE_LIMIT:
            task.cancel()
            await _reset(agent)
            raise Stalled


async def _reset(agent) -> None:
    """Drop a stuck chat model; the next message starts it again with the conversation so far."""
    for step in (getattr(getattr(agent, "client", None), "interrupt", None), agent.close):
        if step:
            try:
                await asyncio.wait_for(step(), 10)
            except Exception:  # already broken: that is why we are here
                pass
    if hasattr(agent, "client"):
        agent.client = None
    cl.user_session.set("carry", bool(cl.user_session.get("transcript")))


CONTINUE_WORDS = ("이어서", "계속", "마저", "continue")


def _wants_continue(text: str, strict: bool = False) -> bool:
    """A short request to continue ("이어서 해줘"). strict: nothing but that request."""
    t = (text or "").strip()
    if not any(w in t for w in CONTINUE_WORDS):
        return False
    return len(t) <= 15 if strict else len(t) <= 60


def _progress(session: Session, question: str) -> dict:
    return {"question": question, "mode": session.mode, "status": "running",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), **session.snapshot()}


async def _offer_resume(data: dict, why: str, delay: float = 0) -> None:
    """Show what a cut-off question had found (side window) and offer to continue it."""
    cl.user_session.set("resume_work", data)
    if delay:
        await asyncio.sleep(delay)
    docs, evs = len(data.get("docs") or []), len(data.get("evidence") or [])
    found = checkpoint.found_something(data)
    content = (f"! {why}\n\n**질문** {data.get('question', '')[:300]}\n\n"
               + (f"그때까지 찾은 것: 문서 {docs}개, 근거 {evs}개 — {PROGRESS_WINDOW}\n\n"
                  "`이어서 하기`를 누르거나 \"이어서 해줘\"라고 보내면, 찾은 자료를 이어받아 모자란 부분만 더 찾고 답을 마저 써요. "
                  if found else "찾아둔 자료는 없어요. `이어서 하기`를 누르면 같은 질문을 다시 시작해요. ")
               + "다른 질문을 보내면 새로 시작해요.")
    elements = [cl.Text(name=PROGRESS_WINDOW, content=checkpoint.summary(data), display="side")] if found else []
    await cl.Message(content=content, elements=elements,
                     actions=[cl.Action(name="resume", payload={}, label="이어서 하기")]).send()


def _remember(question: str, answer: str) -> None:
    transcript = cl.user_session.get("transcript")
    transcript.append(f"질문: {question.strip()}\n\n답변: {(answer or '').strip()}")
    del transcript[:-6]  # the last few turns are enough context


def _transcript(limit: int = 6000) -> str:
    return "\n\n".join(cl.user_session.get("transcript") or [])[-limit:]


async def _open_view(view: dict) -> None:
    """Show an answer in the window on the right (replacing whatever it showed)."""
    try:
        element = cl.CustomElement(name=ANSWER_WINDOW, props=view, display="side")
        await cl.ElementSidebar.set_elements([element])
        await cl.ElementSidebar.set_title("답변")
    except Exception as e:  # the answer is already in the chat; never fail the turn over the window
        print(f"[ui] answer window failed: {e}", flush=True)


def _fix_markdown(text: str) -> str:
    """Two Markdown traps in Korean answers:
    - **bold** can't close right before a particle (**28%**를): the asterisks would show. Drop them.
    - a range like 100~283 uses a single ~, and two of them in a line strike the text between
      through. Escape single tildes (~~deliberate strikethrough~~ is left alone)."""
    text = re.sub(r"\*\*([^*\n]+?)\*\*(?=[가-힣])", r"\1", text or "")
    return re.sub(r"(?<![~\\])~(?!~)", r"\\~", text)


class _Panel:
    """The parallel-mode panel: one message holding the ResearchPanel element, refreshed while
    the turn runs. Every span of the turn goes to the board instead of becoming a step."""

    def __init__(self, brain: str, ui: dict):
        self.board, self.ui = Board(brain), ui
        self.element = cl.CustomElement(name="ResearchPanel", props=self.board.props(), display="inline")
        self.ticker: asyncio.Task | None = None
        self.pushed = 0.0

    async def open(self) -> None:
        self.ui["board"] = self.board
        await cl.Message(content="", elements=[self.element]).send()
        self.ticker = asyncio.create_task(self._tick())

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(PANEL_INTERVAL)
            # A slow model sends no events for a while; refresh anyway so the clocks keep moving.
            if self.board.dirty or asyncio.get_running_loop().time() - self.pushed > 3:
                await self._push()

    async def _push(self, save: bool = False) -> None:
        self.board.dirty = False
        self.pushed = asyncio.get_running_loop().time()
        try:
            self.element.props = self.board.props()
            if save:  # the final state is stored with the conversation
                await self.element.update()
            else:  # live refresh: over the socket only, nothing written
                await cl.context.emitter.send_element(self.element.to_dict())
        except Exception as e:  # a display problem must never break the conversation
            print(f"[ui] panel update failed: {e}", flush=True)

    async def close(self) -> None:
        if self.ticker:
            self.ticker.cancel()
        self.board.finish()
        await self._push(save=True)
        self.ui["board"] = None


# ---------- buttons ----------

async def _follow_up(text: str, shown: str, mode: str | None = None) -> None:
    """A button press as if the user had typed `shown`. Its steps go in a run step, like a typed
    message's: nested under the user message itself they were drawn above older messages."""
    await cl.Message(content=shown, type="user_message").send()
    # Not `async with`: inside the step's context the answer message would be nested in it too.
    run = cl.Step(name=shown, type="run")
    run.start = utc_now()
    await run.send()
    try:
        await turn(text, run.id, mode=mode)
    finally:
        run.end = utc_now()
        await run.update()


@cl.action_callback("deeper")
async def deeper(action: cl.Action):
    await _follow_up(prompts.DEEPER, "더 깊게 조사해줘")


@cl.action_callback("counter")
async def counter(action: cl.Action):
    await _follow_up(prompts.COUNTER, "반론을 찾아줘")


@cl.action_callback("resume")
async def resume_work(action: cl.Action):
    if cl.user_session.get("resume_work"):
        await _follow_up("이어서 해줘", "이어서 하기")
    else:
        await cl.Message(content="이어서 할 질문이 없어요 (이미 이어서 했거나 새 질문을 보냈어요).").send()


@cl.action_callback("parallel")
async def parallel_again(action: cl.Action):
    question = action.payload.get("question", "")
    await _follow_up(f"같은 질문을 병렬 조사로 다시 해줘: {question}", "병렬 조사로 다시 해줘", mode="parallel")


@cl.action_callback("fix")
async def fix(action: cl.Action):
    # Fixing rewrites from the evidence already collected: no new research round, even in parallel mode.
    await _follow_up(prompts.FIX.format(problems=action.payload.get("problems", "")), "걸린 문장을 고쳐줘", mode="basic")


@cl.on_stop
async def stop():
    agent, session = cl.user_session.get("agent"), cl.user_session.get("session")
    if session:
        session.stopped.set()  # parallel researchers stop at their next step
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
