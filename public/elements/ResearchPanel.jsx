// Parallel research panel. Props come from research/board.py and are replaced about once a
// second while the researchers work. Styles: .rp-* in public/terminal.css.
import { useState } from "react";

const MARK = { run: "▸", ok: "✓", err: "x" };

function Head({ status, title, model, seconds }) {
  return (
    <div className="rp-head">
      <span className={`rp-mark rp-${status}`}>{MARK[status] || "·"}</span>
      <span className="rp-title">{title}</span>
      {model ? <span className="rp-model">{model}</span> : null}
      <span className="rp-secs">{seconds != null ? `${seconds}s` : ""}</span>
    </div>
  );
}

function Block({ label, text }) {
  if (!text) return null;
  return (
    <div className="rp-block">
      <div className="rp-block-label">{label}</div>
      <pre className="rp-pre">{text}</pre>
    </div>
  );
}

function Step({ step, open, toggle }) {
  const has = step.input || step.reasoning || step.output;
  return (
    <div className="rp-step" style={{ paddingLeft: `${step.depth * 0.9}rem` }}>
      <div className={`rp-step-line ${has ? "rp-click" : ""}`} onClick={() => has && toggle(step.id)}>
        <span className={`rp-mark rp-${step.status}`}>{MARK[step.status]}</span>
        <span className="rp-step-name">{step.name}</span>
        {step.model ? <span className="rp-model">{step.model}</span> : null}
        <span className="rp-secs">{step.seconds}s</span>
      </div>
      {open && has ? (
        <div className="rp-io">
          <Block label="IN" text={step.input} />
          <Block label="사고" text={step.reasoning} />
          <Block label="OUT" text={step.output} />
        </div>
      ) : null}
    </div>
  );
}

function Steps({ steps, opened, toggle }) {
  return (
    <div className="rp-steps">
      {steps.map((s) => <Step key={s.id} step={s} open={!!opened[s.id]} toggle={toggle} />)}
    </div>
  );
}

function Researcher({ r, opened, toggle }) {
  return (
    <div className={`rp-card rp-card-${r.status}`}>
      <Head status={r.status} title={r.label} model={r.model} seconds={r.seconds} />
      {r.brief ? <div className="rp-brief">지시: {r.brief}</div> : null}
      <Steps steps={r.steps} opened={opened} toggle={toggle} />
      {r.status !== "run" ? (
        <div className="rp-result">
          {r.error ? <div className="rp-err">실패: {r.error}</div> : null}
          {r.conclusion ? <div className="rp-conclusion">{r.conclusion}</div> : null}
          {r.claims.map((c, i) => (
            <div key={i} className="rp-claim">
              - {c.claim} <span className="rp-ev">[{c.evidence.join(", ")}]</span>{" "}
              <span className={`rp-conf rp-conf-${c.confidence}`}>{c.confidence}</span>
            </div>
          ))}
          {r.gaps ? <div className="rp-gaps">부족: {r.gaps}</div> : null}
        </div>
      ) : null}
    </div>
  );
}

function Compare({ c, opened, toggle }) {
  return (
    <div className={`rp-card rp-card-${c.status}`}>
      <Head status={c.status} title="비교" seconds={c.seconds} />
      <Steps steps={c.steps} opened={opened} toggle={toggle} />
      {c.rows.map((row, i) => (
        <div key={i} className="rp-row">
          <span className={`rp-tag rp-tag-${row.status}`}>[{row.label}]</span>
          <span className="rp-statement">{row.statement}</span>
          <div className="rp-row-meta">
            지지 {row.support.join(", ") || "-"}
            {row.oppose.length ? ` / 반대 ${row.oppose.join(", ")}` : ""}
            {` · 독립 출처 ${row.sources}`}
            {row.evidence.length ? ` · ${row.evidence.join(" ")}` : ""}
            {row.note ? ` · ${row.note}` : ""}
          </div>
        </div>
      ))}
      {c.note ? <div className="rp-gaps">{c.note}</div> : null}
    </div>
  );
}

export default function ResearchPanel() {
  const [opened, setOpened] = useState({});
  const toggle = (id) => setOpened((o) => ({ ...o, [id]: !o[id] }));
  const p = props;
  const at = p.phases.findIndex((x) => x.key === p.phase);
  const done = p.phase === "done";

  return (
    <div className="rp">
      <div className="rp-phases">
        {p.phases.map((x, i) => (
          <span key={x.key} className={`rp-phase ${done || i < at ? "rp-past" : i === at ? "rp-now" : ""}`}>
            {x.label}
          </span>
        ))}
        <span className="rp-secs">{done ? "완료" : "진행 중"} · {p.seconds}s</span>
      </div>

      <div className={`rp-card rp-card-${p.brain.status}`}>
        <Head status={p.brain.status} title={`두뇌 · ${p.brain.label}`} />
        {p.brain.hidden ? <div className="rp-gaps">앞선 단계 {p.brain.hidden}개 생략</div> : null}
        <Steps steps={p.brain.steps} opened={opened} toggle={toggle} />
      </div>

      {p.researchers.length ? (
        <div className="rp-grid">
          {p.researchers.map((r) => <Researcher key={r.key} r={r} opened={opened} toggle={toggle} />)}
        </div>
      ) : null}

      {p.compare ? <Compare c={p.compare} opened={opened} toggle={toggle} /> : null}

      {p.verify ? (
        <div className={`rp-card rp-card-${p.verify.status}`}>
          <Head status={p.verify.status} title={`답변 검증${p.verify.summary ? " · " + p.verify.summary : ""}`}
                seconds={p.verify.seconds} />
          <Steps steps={p.verify.steps} opened={opened} toggle={toggle} />
        </div>
      ) : null}
    </div>
  );
}
