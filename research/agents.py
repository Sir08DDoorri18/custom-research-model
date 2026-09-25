"""The chat model ("brain") that decides when to search and writes the answer.

- Claude presets: Claude Agent SDK. It runs the local `claude` CLI in the background (your
  Claude subscription login), with our research functions as its only tools.
- no-claude preset: PydanticAI with the free models from the `brain` role in models.yaml.

Both expose the same interface: `await agent.turn(text) -> answer`. Thinking and tool calls
are reported through the session's tracer as they happen.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Annotated

from . import config, llm, prompts
from .engine import Session

SANDBOX = os.path.join(tempfile.gettempdir(), "research_agent_sandbox")  # no CLAUDE.md, no project files

DOCS = {
    "search_papers": "Search scholarly papers (OpenAlex, Semantic Scholar, arXiv). Returns document ids D#. Use English keywords.",
    "search_web": "Web search for news, official documentation and institutional pages. Returns document ids D#.",
    "citation_graph": "Papers that a paper cites (direction='references') or that cite it (direction='citations'). Returns D#.",
    "read_sources": "Read documents and extract evidence (E#) relevant to the question: relevance, summary, verbatim quote. "
                    "Only E# ids may be cited in answers.",
}


def make_agent(preset: str, session: Session):
    p = config.presets()[preset]
    if p["backend"] == "claude":
        return ClaudeAgent(session, p["model"])
    return FreeAgent(session)


class ClaudeAgent:
    def __init__(self, session: Session, model: str):
        self.session, self.model = session, model
        self.label = f"Claude {model}"
        self.client = None
        self.usage: dict = {}

    async def start(self) -> None:
        """Launch the Claude process; does nothing if it is already running."""
        if self.client:
            return
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server
        server = create_sdk_mcp_server("research", tools=self._tools())
        os.makedirs(SANDBOX, exist_ok=True)
        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=prompts.SYSTEM.format(language=config.language()),
            mcp_servers={"research": server},
            strict_mcp_config=True,                       # don't load the account's claude.ai connectors (Gmail, Drive...)
            tools=[],                                     # no built-in tools (files, shell, web)
            allowed_tools=[f"mcp__research__{n}" for n in DOCS],
            permission_mode="dontAsk",                    # anything not allowed above is refused
            setting_sources=[],                           # ignore user/project settings and CLAUDE.md
            cwd=SANDBOX,
            thinking={"type": "adaptive", "display": "summarized"},
            max_turns=40,
        )
        client = ClaudeSDKClient(options=options)
        await client.connect()
        self.client = client  # only once connected, so a failed start is retried next message

    async def turn(self, text: str, images: list[str] = ()) -> str:
        """images: pictures the user attached; Claude looks at them itself, next to the
        vision model's description that the tools already registered."""
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ThinkingBlock, ToolUseBlock
        tr = self.session.tracer
        final, last_texts = "", []
        if images:
            from .files import image_b64
            blocks = [{"type": "text", "text": text}]
            for path in images:
                data, mime = image_b64(path)
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})

            async def message():
                yield {"type": "user", "message": {"role": "user", "content": blocks}, "parent_tool_use_id": None}

            await self.client.query(message())
        else:
            await self.client.query(text)
        async for msg in self.client.receive_response():
            if isinstance(msg, AssistantMessage):
                texts = [b.text for b in msg.content if isinstance(b, TextBlock) and b.text.strip()]
                for b in msg.content:
                    if isinstance(b, ThinkingBlock) and b.thinking.strip():
                        tr.note(f"{self.label} 생각", b.thinking, model=msg.model)
                if any(isinstance(b, ToolUseBlock) for b in msg.content):
                    if texts:  # text written before a tool call is a working note, not the answer
                        tr.note(f"{self.label} 메모", "\n".join(texts), model=msg.model)
                    last_texts = []
                else:
                    last_texts += texts
            elif isinstance(msg, ResultMessage):
                self.usage = {"turns": msg.num_turns, "usage": msg.usage, "cost_usd": msg.total_cost_usd}
                if msg.is_error:
                    raise RuntimeError(f"Claude 오류: {msg.result or msg.errors or msg.subtype}")
                final = msg.result or "\n".join(last_texts)
        return final

    async def close(self) -> None:
        if self.client:
            await self.client.disconnect()
            self.client = None

    def _tools(self):
        from claude_agent_sdk import tool
        s = self.session

        async def run(fn, *args):
            try:
                out = await asyncio.to_thread(fn, *args)
                return {"content": [{"type": "text", "text": out}]}
            except Exception as e:  # report to the model instead of crashing the turn
                return {"content": [{"type": "text", "text": f"도구 오류: {type(e).__name__}: {e}"}], "is_error": True}

        @tool("search_papers", DOCS["search_papers"], {"query": Annotated[str, "English keywords"]})
        async def search_papers(args):
            return await run(s.search_papers, args["query"])

        @tool("search_web", DOCS["search_web"], {"query": str})
        async def search_web(args):
            return await run(s.search_web, args["query"])

        @tool("citation_graph", DOCS["citation_graph"], {
            "type": "object",
            "properties": {"doc_id": {"type": "string"},
                           "direction": {"type": "string", "enum": ["references", "citations"]}},
            "required": ["doc_id", "direction"]})
        async def citation_graph(args):
            return await run(s.citation_graph, args["doc_id"], args.get("direction", "references"))

        @tool("read_sources", DOCS["read_sources"], {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "what you want to learn from these documents"},
                           "doc_ids": {"type": "array", "items": {"type": "string"}, "description": "e.g. [\"D1\", \"D4\"]"},
                           "focus": {"type": "string", "description": "English keywords for locating passages"}},
            "required": ["question", "doc_ids"]})
        async def read_sources(args):
            return await run(s.read_sources, args["question"], list(args["doc_ids"]), args.get("focus", ""))

        return [search_papers, search_web, citation_graph, read_sources]


class FreeAgent:
    def __init__(self, session: Session):
        from pydantic_ai import Agent
        from pydantic_ai.models.fallback import FallbackModel
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        self.session = session
        providers = config.providers()
        refs = [r for r in config.role("brain") if llm.usable(r)]
        if not refs:
            raise RuntimeError("no-claude 프리셋에 쓸 모델이 없어요. .env에 API 키를 넣어주세요 (models.yaml의 brain 역할).")
        models = [OpenAIChatModel(r.model, provider=OpenAIProvider(base_url=providers[r.provider].base_url,
                                                                    api_key=providers[r.provider].api_key))
                  for r in refs]
        self.label = f"무료 모델 ({refs[0]})"
        s = session

        def search_papers(query: str) -> str:
            return s.search_papers(query)

        def search_web(query: str) -> str:
            return s.search_web(query)

        def citation_graph(doc_id: str, direction: str = "references") -> str:
            return s.citation_graph(doc_id, direction)

        def read_sources(question: str, doc_ids: list[str], focus: str = "") -> str:
            return s.read_sources(question, doc_ids, focus)

        for fn in (search_papers, search_web, citation_graph, read_sources):
            fn.__doc__ = DOCS[fn.__name__]
        self.agent = Agent(FallbackModel(*models) if len(models) > 1 else models[0],
                           instructions=prompts.SYSTEM.format(language=config.language()),
                           tools=[search_papers, search_web, citation_graph, read_sources])
        self.history = []

    async def start(self) -> None:
        pass

    async def turn(self, text: str, images: list[str] = ()) -> str:
        """images are not sent: the free brain models can't see, so they rely on the
        vision model's descriptions registered when the files were added."""
        from pydantic_ai import AgentRunResultEvent, PartEndEvent, ThinkingPart, UsageLimits
        tr = self.session.tracer
        final = ""
        async with self.agent.run_stream_events(text, message_history=self.history,
                                                usage_limits=UsageLimits(request_limit=30)) as run:
            async for ev in run:
                if isinstance(ev, PartEndEvent) and isinstance(ev.part, ThinkingPart) and ev.part.content.strip():
                    tr.note(f"{self.label} 생각", ev.part.content)
                elif isinstance(ev, AgentRunResultEvent):
                    final = str(ev.result.output)
                    self.history = ev.result.all_messages()
        return final

    async def close(self) -> None:
        pass
