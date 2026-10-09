// Answer window, opened on the right after a cited answer (all modes). Props come from
// research/answer_view.py. Each checked sentence is underlined by its verdict; clicking it
// shows why, and the cited evidence. Styles: .av-* in public/terminal.css.
import { useState } from "react";

const BADGE = { supported: "[ok]", partial: "[~]", unsupported: "[x]", unchecked: "[?]" };
const NAMES = { supported: "근거 있음", partial: "일부만", unsupported: "근거 없음", unchecked: "미확인" };
const VERDICT = { supported: "근거 있음", partial: "일부만", unsupported: "근거 없음" };
const INLINE = /(\*\*[^*\n]+?\*\*|`[^`\n]+`|\[E\d+(?:\s*[,，/]\s*E?\d+)*\])/g;

function Inline({ text, onCite }) {
  return text.split(INLINE).map((part, i) => {
    if (!part) return null;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <b key={i}>{part.slice(2, -2)}</b>;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) return <code key={i}>{part.slice(1, -1)}</code>;
    if (/^\[E\d/.test(part)) {
      const ids = part.slice(1, -1).split(/\s*[,，/]\s*/).map((x) => "E" + x.replace(/^E/, ""));
      return (
        <span key={i} className="av-cites">
          {ids.map((id) => (
            <button key={id} className="av-cite" onClick={(e) => { e.stopPropagation(); onCite(id); }}>{id}</button>
          ))}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

function Detail({ c, sources, onCite }) {
  const byId = Object.fromEntries(sources.map((s) => [s.id, s]));
  return (
    <div className={`av-detail av-detail-${c.final}`}>
      <div className="av-detail-head">{BADGE[c.final]} {c.label} — {c.reason || "-"}</div>
      <div className="av-checks">
        {c.nli ? <span>로컬 NLI: {VERDICT[c.nli] || c.nli}</span> : null}
        {c.first ? <span>1차 {c.first[2].split("/").pop()}: {VERDICT[c.first[0]] || c.first[0]}</span> : null}
        {c.votes.map((v, i) => <span key={i}>심사 {v[2].split("/").pop()}: {VERDICT[v[0]] || v[0]}</span>)}
      </div>
      {c.bad.length ? <div className="av-bad">없는 근거 번호: {c.bad.join(", ")}</div> : null}
      {c.ids.map((id) => {
        const s = byId[id];
        if (!s) return null;
        return (
          <div key={id} className="av-quote">
            <button className="av-cite" onClick={() => onCite(id)}>{id}</button> <span className="av-muted">{s.title}{s.page ? ` p.${s.page}` : ""}</span>
            {s.quote ? <div className="av-quote-text">"{s.quote}"</div> : <div className="av-quote-text">{s.summary}</div>}
          </div>
        );
      })}
    </div>
  );
}

function Line({ line, idx, claims, sources, open, setOpen, onCite }) {
  if (line.kind === "blank") return <div className="av-gap" />;
  if (line.kind === "code" || line.kind === "table") return <pre className="av-pre">{line.segs[0].t}</pre>;
  if (line.kind.startsWith("h")) return <div className={`av-${line.kind}`}><Inline text={line.segs[0].t} onCite={onCite} /></div>;

  const here = open && open.line === idx ? claims[open.claim] : null;
  const body = line.segs.map((seg, i) => {
    const c = seg.c != null ? claims[seg.c] : null;
    const cls = c ? `av-s av-${c.final}${open && open.claim === String(seg.c) ? " av-on" : ""}` : seg.u ? "av-s av-uncited" : "";
    const title = c ? `${NAMES[c.final]}: ${c.reason}` : seg.u ? "출처 없는 수치" : undefined;
    const click = c ? () => setOpen(open && open.claim === String(seg.c) ? null : { claim: String(seg.c), line: idx }) : undefined;
    return (
      <span key={i}>
        <span className={cls} title={title} onClick={click}><Inline text={seg.t} onCite={onCite} /></span>
        {seg.u ? <span className="av-tag-u">[?]</span> : null}{" "}
      </span>
    );
  });
  const marker = line.kind === "li" ? "•" : line.kind === "ol" ? `${line.n}.` : null;
  return (
    <div className={`av-line av-${line.kind}`}>
      {marker ? <span className="av-marker">{marker}</span> : null}
      <div className="av-line-body">
        {body}
        {here ? <Detail c={here} sources={sources} onCite={onCite} /> : null}
      </div>
    </div>
  );
}

function Source({ s, flash }) {
  return (
    <div id={`av-src-${s.id}`} className={`av-src ${flash === s.id ? "av-flash" : ""}`}>
      <div className="av-src-head">
        <span className="av-src-id">[{s.id}]</span>
        {s.badge ? <span className={`av-pill av-pill-${s.status}`}>{s.badge}</span> : null}
        <span className="av-pill">{s.kind}</span>
        {s.retracted ? <span className="av-pill av-pill-bad">철회된 논문</span> : null}
      </div>
      <div className="av-src-title">
        {s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a> : s.title}
        {s.page ? ` · p.${s.page}` : ""}{s.doi ? ` · doi:${s.doi}` : ""}
      </div>
      {s.summary ? <div className="av-muted">{s.summary}</div> : null}
      {s.quote ? <div className="av-quote-text">"{s.quote}"</div> : null}
    </div>
  );
}

export default function AnswerView() {
  const p = props;
  const [tab, setTab] = useState("answer");
  const [open, setOpen] = useState(null);
  const [flash, setFlash] = useState(null);
  const pool = tab === "answer" ? p.sources : p.all;

  const onCite = (id) => {
    const inAnswer = p.sources.some((s) => s.id === id);
    if (!inAnswer) setTab("all");
    setFlash(id);
    setTimeout(() => {
      const el = document.getElementById(`av-src-${id}`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 50);
    setTimeout(() => setFlash(null), 1600);
  };

  return (
    <div className="av">
      <div className="av-q">{p.question}</div>
      <div className="av-summary">
        {Object.keys(NAMES).map((k) => (p.counts[k] ? <span key={k} className={`av-count av-c-${k}`}>{BADGE[k]} {NAMES[k]} {p.counts[k]}</span> : null))}
        {p.uncited ? <span className="av-count av-c-uncited">[?] 출처 없는 수치 {p.uncited}</span> : null}
        {p.mode === "parallel" ? <span className="av-pill">병렬 조사</span> : null}
      </div>
      {p.checkers.length ? <div className="av-muted av-small">검증: {p.checkers.join(" · ")} · 밑줄 친 문장을 누르면 근거가 보여요</div> : null}

      <div className="av-tabs">
        <button className={tab === "answer" ? "av-tab-on" : ""} onClick={() => setTab("answer")}>답변 · 출처 {p.sources.length}</button>
        <button className={tab === "all" ? "av-tab-on" : ""} onClick={() => setTab("all")}>대화 전체 근거 {p.total}</button>
      </div>

      {tab === "answer" ? (
        <div className="av-body">
          {p.lines.map((line, i) => (
            <Line key={i} idx={i} line={line} claims={p.claims} sources={p.sources} open={open} setOpen={setOpen} onCite={onCite} />
          ))}
        </div>
      ) : null}

      <div className="av-sources">
        {tab === "answer" ? <div className="av-h3">출처</div> : null}
        {tab === "all" && p.total > p.all.length ? <div className="av-muted av-small">최근 {p.all.length}개만 표시</div> : null}
        {pool.map((s) => <Source key={s.id} s={s} flash={flash} />)}
      </div>
    </div>
  );
}
