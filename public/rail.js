// Menu bar: research mode on/off plus 지난 대화 / 새 대화, Instagram-style.
// Laptop (>= 900px wide): a narrow rail on the left. Phone or narrow window: a bar at the bottom.
// The mode is saved on the server (/research/prefs, research/prefs.py) and used by the next
// question. Styles: .rail-* in public/terminal.css.
(function () {
  const ICONS = {
    basic: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
    parallel: '<rect x="3.5" y="3.5" width="7" height="7" rx="1.6"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.6"/>' +
              '<rect x="3.5" y="13.5" width="7" height="7" rx="1.6"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.6"/>',
    history: '<path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1"/><path d="M3.5 4.5v4h4"/><path d="M12 7.5V12l3 2"/>',
    new: '<rect x="3.5" y="3.5" width="17" height="17" rx="4.5"/><path d="M12 8v8M8 12h8"/>',
  };
  const ITEMS = [
    { key: "basic", label: "기본", group: "mode", tip: "기본 조사: 한 번 검색하고 바로 답해요. 빠르고 무료 한도를 덜 써요." },
    { key: "parallel", label: "병렬", group: "mode", tip: "병렬 조사: 논문 · 공식 자료 · 뉴스 · 반론 조사원이 동시에 찾고 비교해요. 2~5분 걸려요." },
    { key: "history", label: "지난 대화", group: "nav", tip: "저장된 대화 목록 (30일 보관)" },
    { key: "new", label: "새 대화", group: "nav", tip: "새 대화 시작" },
  ];

  let mode = "basic";

  function icon(key) {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      ICONS[key] + "</svg>";
  }

  function build() {
    const nav = document.createElement("nav");
    nav.id = "rail";
    nav.setAttribute("aria-label", "메뉴");
    nav.innerHTML =
      '<div class="rail-brand" title="research">R</div>' +
      ITEMS.map((it, i) =>
        (i && ITEMS[i - 1].group !== it.group ? '<div class="rail-sep"></div>' : "") +
        `<button type="button" class="rail-item" data-key="${it.key}" title="${it.tip}"` +
        (it.group === "mode" ? ' aria-pressed="false"' : "") + `>${icon(it.key)}<span>${it.label}</span></button>`
      ).join("");
    nav.addEventListener("click", (e) => {
      const btn = e.target.closest(".rail-item");
      if (btn) act(btn.dataset.key);
    });
    document.body.appendChild(nav);
    document.documentElement.classList.add("has-rail");
  }

  function paint() {
    document.querySelectorAll("#rail .rail-item").forEach((b) => {
      const k = b.dataset.key;
      if (k === "basic" || k === "parallel") {
        const on = k === mode;
        b.classList.toggle("rail-on", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      }
    });
    document.documentElement.dataset.mode = mode;
  }

  async function act(key) {
    if (key === "basic" || key === "parallel") {
      const before = mode;
      mode = key;
      paint();
      try {
        const r = await fetch("/research/prefs", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode: key }),
        });
        if (!r.ok) throw new Error(String(r.status));
      } catch (err) {
        mode = before; // server didn't take it: show what is really in effect
        paint();
      }
    } else if (key === "history") {
      const trigger = document.getElementById("sidebar-trigger-button");
      if (trigger) trigger.click();
    } else if (key === "new") {
      const button = document.getElementById("new-chat-button");
      if (button) button.click();
    }
  }

  async function load() {
    try {
      const r = await fetch("/research/prefs");
      if (r.ok) mode = (await r.json()).mode || "basic";
    } catch (err) { /* keep the default */ }
    paint();
  }

  function init() {
    if (document.getElementById("rail")) return;
    build();
    paint();
    load();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
