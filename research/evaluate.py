"""`python -m research eval`: how good is each judge, and do they fail together?

Every judge (local NLI + each configured model with a key) labels every item in
eval/claims.jsonl on its own. Reported per judge: accuracy on "supported vs not"
(the decision that matters) and on the three labels. Then the pairwise correlation of
their mistakes, and the panel's effective number of independent votes
n_eff = k / (1 + (k - 1) * mean correlation). A panel of 3 with n_eff near 1 is one vote
bought three times.
"""
from __future__ import annotations

import json
import math
from itertools import combinations

from . import config, llm, prompts
from .trace import pmap
from .verify import NLI


def run(path: str) -> None:
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    print(f"{len(items)}개 문장으로 채점자 평가\n")
    predictions: dict[str, list[str | None]] = {}

    nli = NLI.get()
    if nli:
        verdicts = nli.verdicts([(it["premise"], it["claim"]) for it in items])
        predictions["local NLI"] = [_nli_label(v[0]) for v in verdicts]
    else:
        print("로컬 NLI 없음 (requirements-nli.txt)")

    jury, spare = config.jury()
    refs = list(dict.fromkeys(config.role("first") + jury + spare))
    usable = [r for r in refs if llm.usable(r)]
    for r in refs:
        if r not in usable:
            print(f"건너뜀 {r} ({llm.benched(r) or '키 없음'})")
    for ref, preds in zip(usable, pmap(lambda r: _judge_all(r, items), usable, workers=4)):
        predictions[str(ref)] = preds

    if not predictions:
        print("평가할 채점자가 없어요. .env에 키를 넣거나 NLI를 설치하세요.")
        return

    gold = [it["label"] for it in items]
    print(f"\n{'judge':<48} {'binary':>7} {'3-class':>8} {'failed':>7}")
    for name, preds in predictions.items():
        answered = [(p, g) for p, g in zip(preds, gold) if p]
        if not answered:
            print(f"{name:<48} {'-':>7} {'-':>8} {len(preds):>7}")
            continue
        binary = sum((p == "supported") == (g == "supported") for p, g in answered) / len(answered)
        three = sum(p == g for p, g in answered) / len(answered)
        print(f"{name:<48} {binary:>7.0%} {three:>8.0%} {len(preds) - len(answered):>7}")

    errors = {n: [None if p is None else (p == "supported") != (g == "supported") for p, g in zip(ps, gold)]
              for n, ps in predictions.items()}
    names = [n for n in errors if any(e is not None for e in errors[n])]
    if len(names) >= 2:
        print("\n오류 상관 (phi, 1 = 항상 같이 틀림)")
        rhos = []
        for a, b in combinations(names, 2):
            rho = _phi(errors[a], errors[b])
            if rho is not None:
                rhos.append(rho)
                print(f"  {a.split('/')[-1]:<30} × {b.split('/')[-1]:<30} {rho:+.2f}")
        if rhos:
            k, mean = len(names), sum(rhos) / len(rhos)
            print(f"\n채점자 {k}명, 평균 오류 상관 {mean:+.2f} → 유효 독립 표 수 n_eff ≈ {k / (1 + (k - 1) * max(mean, 0)):.1f}")


def _nli_label(v: str) -> str:
    return {"supported": "supported", "unsupported": "unsupported"}.get(v, "partial")


def _judge_all(ref, items: list[dict], size: int = 6) -> list[str | None]:
    preds: list[str | None] = [None] * len(items)
    for start in range(0, len(items), size):
        batch = items[start:start + size]
        body = "\n\n".join(f"[{k}] Sentence: {it['claim']}\nPassages:\n{it['premise']}" for k, it in enumerate(batch, 1))
        try:
            reply = llm.call(ref, [{"role": "user", "content": prompts.JUDGE.format(items=body)}], f"eval {ref}")
            rows = llm.extract_json(reply.text)
        except llm.LLMError as e:
            print(f"  {ref}: {str(e)[:120]}")
            continue
        for row in rows if isinstance(rows, list) else []:
            try:
                i = start + int(row["i"]) - 1
                v = str(row["verdict"]).lower()
            except (KeyError, ValueError, TypeError):
                continue
            if start <= i < start + len(batch) and v in ("supported", "partial", "unsupported"):
                preds[i] = v
    return preds


def _phi(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n11 = sum(x and y for x, y in pairs)
    n10 = sum(x and not y for x, y in pairs)
    n01 = sum(y and not x for x, y in pairs)
    n00 = len(pairs) - n11 - n10 - n01
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / den if den else None
