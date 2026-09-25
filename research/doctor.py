"""`python -m research doctor`: is every configured model and API still reachable?"""
from __future__ import annotations

import shutil
import subprocess

from openai import OpenAI

from . import config, llm, sources
from .trace import pmap


def run(check_claude: bool = False) -> None:
    print("== providers ==")
    listed: dict[str, set[str] | None] = {}
    for name, p in config.providers().items():
        if not p.api_key:
            print(f"  -  {name:<11} {p.key_env} 없음 (.env)")
            listed[name] = None
            continue
        try:
            ids = {m.id for m in OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=20).models.list()}
            listed[name] = ids
            print(f"  ok {name:<11} 키 정상, 모델 {len(ids)}개")
        except Exception as e:
            listed[name] = set()  # empty = can't list; the per-model test calls below still run
            if any(code in str(e) for code in ("401", "403")):
                print(f"  !  {name:<11} 모델 목록 조회 권한 없음 (제한된 키). 아래에서 모델별로 직접 확인해요")
            else:
                print(f"  x  {name:<11} {type(e).__name__}: {str(e)[:120]}")

    print("\n== roles (위에서부터 순서대로 사용) ==")
    jury, spare = config.jury()
    groups = {r: config.role(r) for r in ("brain", "rcs", "first")} | {"jury": jury, "jury_fallback": spare}
    def status(ref) -> str:
        ids = listed.get(ref.provider)
        if ids is None:
            return "-  (키 없음)"
        unlisted = bool(ids) and ref.model not in ids
        try:
            reply = llm.call(ref, [{"role": "user", "content": "Reply with the single word OK."}], "doctor")
        except llm.LLMError as e:
            return "x  " + ("목록에 없음 (이름이 바뀌었거나 종료됨)" if unlisted else str(e)[:140])
        note = f"ok 응답 \"{reply.text[:20]}\"" + (" · 사고과정 있음" if reply.reasoning else "")
        # An old alias can still answer while missing from the list; it may stop working any day.
        return note + (" · !  목록에 없는 옛 이름, 곧 없어질 수 있음" if unlisted else "")

    # Test every distinct model once, in parallel (calls to one provider are still spaced out).
    unique = list(dict.fromkeys(r for refs in groups.values() for r in refs))
    tested = dict(zip(unique, pmap(status, unique, workers=8)))
    for role, refs in groups.items():
        print(f"  [{role}]")
        for ref in refs:
            mark, _, note = tested[ref].partition(" ")
            print(f"    {mark} {ref}  {note}")

    print("\n== search APIs ==")
    for name, fn in (("OpenAlex", sources.openalex_search), ("Semantic Scholar", sources.s2_search),
                     ("arXiv", sources.arxiv_search), ("Web (DuckDuckGo)", sources.web_search)):
        try:
            n = len(fn("graphene thermal conductivity", 3))
            print(f"  {'ok' if n else '! '} {name}: {n}건")
        except Exception as e:
            if "429" in str(e) and name == "Semantic Scholar":
                print(f"  !  {name}: 익명 공용 한도 초과(429). 무료 키를 받아 .env의 S2_API_KEY에 넣으면 안정적이에요")
            else:
                print(f"  x  {name}: {type(e).__name__}: {str(e)[:100]}")

    print("\n== local NLI ==")
    try:
        import transformers  # noqa: F401
        print(f"  ok 설치됨 ({config.nli_model()}, 첫 사용 때 모델을 내려받아요)")
    except ImportError:
        print("  -  미설치 — pip install -r requirements-nli.txt (없으면 이 단계는 건너뛰어요)")

    print("\n== Claude ==")
    exe = shutil.which("claude")
    if not exe:
        print("  -  claude CLI 없음 — Claude 프리셋은 못 쓰고 no-claude만 가능")
        return
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"  ok claude CLI {version}")
    if check_claude:
        out = subprocess.run([exe, "-p", "Reply with the single word OK.", "--model", "haiku"],
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
        ok = out.returncode == 0 and out.stdout.strip()
        print(f"  {'ok' if ok else 'x '} 로그인 확인: {(out.stdout or out.stderr).strip()[:120]}")
