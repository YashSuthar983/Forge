(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  // How many of the solver's own report fields get promoted to summary tiles.
  const TILE_COUNT = 4;

  const state = {
    caps: null,
    checkClasses: [],
    templates: [],
    groups: [],
    current: null, // { id, title, path, info, origin, note }
    source: "mps", // mps | eq
    baseline: "",
    solToken: null,
    sol: null, // parsed /api/solution payload
    solView: null,
    solShown: 0,
  };

  const mpsEl = $("mps_text");
  const eqEl = $("model_text");
  const btnSolve = $("btn-solve");
  const result = $("result");

  $("solve-kbd").textContent = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
    ? "⌘ ↵"
    : "Ctrl ↵";

  // ---------- formatting ----------

  const fmtInt = (n) => (n == null ? "-" : Number(n).toLocaleString());
  const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

  function fmtBytes(b) {
    if (b == null) return "";
    if (b < 1024) return `${b} B`;
    if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / 1024 ** 2).toFixed(1)} MB`;
  }

  function fmtObjective(v) {
    const n = Number(v);
    const a = Math.abs(n);
    return a >= 1e15 || (a > 0 && a < 1e-4)
      ? n.toExponential(8)
      : n.toLocaleString(undefined, { maximumFractionDigits: 6 });
  }

  function dims(info) {
    if (!info || info.rows == null) return "";
    return `${fmtInt(info.rows)} × ${fmtInt(info.cols)}` + (info.nnz != null ? ` · ${fmtInt(info.nnz)} nnz` : "");
  }

  function metaLine(meta) {
    if (!meta) return "";
    return Object.entries(meta)
      .filter(([k]) => k !== "generator")
      .map(([k, v]) => `${k} ${v}`)
      .join(" · ");
  }

  // Short ids (cpu, gpu) read best upper-cased; longer ones (vulkan) capitalised.
  const backendName = (id) => (id.length <= 4 ? id.toUpperCase() : cap(id));
  const kindBadge = (kind) => el("span", `kind kind-${kind}`, kind);
  const shellQuote = (a) => (/^[\w./:=,+-]+$/.test(a) ? a : `'${a.replace(/'/g, `'\\''`)}'`);

  async function api(url, opts) {
    const r = await fetch(url, opts);
    let j = null;
    try { j = await r.json(); } catch { /* non-JSON error page */ }
    if (!r.ok) {
      const d = j && j.detail;
      throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `HTTP ${r.status}`);
    }
    return j;
  }

  function setAlert(text, title) {
    const a = $("alert");
    a.hidden = !text;
    a.textContent = text || "";
    a.title = title || "";
  }

  // ---------- capabilities → controls ----------

  function fillSelect(select, items, { placeholder } = {}) {
    select.innerHTML = "";
    if (placeholder) select.append(new Option(placeholder, ""));
    for (const it of items) {
      const o = new Option(it.label || it.id, it.id);
      if (it.about) o.title = it.about;
      if (it.disabled) o.disabled = true;
      select.append(o);
    }
  }

  function renderControls(health) {
    const caps = health.capabilities || {};
    fillSelect($("engine"), caps.engines || [], { placeholder: "Match model" });

    const backends = caps.backends || [];
    fillSelect(
      $("backend"),
      backends.map((b) => ({
        id: b.id,
        label: backendName(b.id) + (b.available ? "" : " (unavailable)"),
        about: b.reason,
        disabled: !b.available,
      }))
    );
    if (caps.default_backend) $("backend").value = caps.default_backend;

    const off = backends.filter((b) => !b.available);
    const note = $("hw-note");
    note.innerHTML = "";
    note.hidden = !off.length;
    for (const b of off) {
      const p = el("p");
      p.append(el("strong", null, `${backendName(b.id)}: `), document.createTextNode(b.reason || "unavailable"));
      note.append(p);
    }

    fillSelect($("method"), caps.methods || [], { placeholder: "Solver default" });
    $("method-field").hidden = !caps.methods?.length;
    $("threads-field").hidden = !caps.has_threads;

    const tl = $("time_limit");
    tl.value = health.time_limit?.default ?? "";
    if (health.time_limit?.max) tl.max = health.time_limit.max;
  }

  function updateEngineHint() {
    const opt = $("engine").options[0];
    const auto = state.current?.info?.engine;
    if (opt) opt.textContent = auto ? `Match model (${auto})` : "Match model";
  }

  // ---------- sidebar ----------

  function showSidePanel(name) {
    for (const b of document.querySelectorAll(".seg-btn")) b.classList.toggle("active", b.dataset.panel === name);
    for (const p of ["models", "write"]) $(`panel-${p}`).hidden = p !== name;
  }

  function renderLibrary() {
    const nav = $("library");
    nav.innerHTML = "";
    const q = $("filter").value.trim().toLowerCase();
    let shown = 0;
    for (const g of state.groups) {
      const models = g.models.filter(
        (m) => !q || [m.title, m.file, m.kind, g.title, g.id].join(" ").toLowerCase().includes(q)
      );
      if (!models.length) continue;
      shown += models.length;
      const d = el("details", "group");
      const hasActive = models.some((m) => m.id === state.current?.id);
      d.open = !!q || hasActive || g.models.length <= 6;
      const s = el("summary");
      s.append(el("span", null, g.title), el("span", "count", String(models.length)));
      s.title = g.id;
      d.append(s);
      if (g.about) d.append(el("p", "group-about", g.about));
      for (const m of models) d.append(libraryItem(m));
      nav.append(d);
    }
    if (!shown) nav.append(el("p", "empty", q ? "No models match." : "No models found in the configured folders."));
  }

  function libraryItem(m) {
    const b = el("button", "item" + (m.id === state.current?.id ? " active" : ""));
    b.type = "button";
    b.title = m.id;
    b.append(el("span", "item-title", m.title), kindBadge(m.kind));
    const sub = m.meta ? metaLine(m.meta) : [dims(m), fmtBytes(m.bytes)].filter(Boolean).join(" · ");
    b.append(el("span", "item-sub", sub));
    b.addEventListener("click", () => openModel(m.id));
    return b;
  }

  function pickCard({ title, sub, kind, active, onClick }) {
    const b = el("button", "pick" + (active ? " active" : ""));
    b.type = "button";
    b.append(el("span", "pick-title", title));
    b.append(kind ? kindBadge(kind) : el("span"));
    if (sub) b.append(el("span", "pick-sub", sub));
    b.addEventListener("click", onClick);
    return b;
  }

  function renderTemplates() {
    const box = $("templates");
    box.innerHTML = "";
    $("seg-write").hidden = !state.templates.length;
    for (const t of state.templates) {
      box.append(pickCard({
        title: t.title,
        sub: t.about,
        kind: t.kind,
        active: state.current?.id === `template:${t.id}`,
        onClick: () => openTemplate(t),
      }));
    }
  }

  function markActive() {
    renderLibrary();
    renderTemplates();
  }

  // ---------- current model ----------

  function setCurrent(cur) {
    state.current = cur;
    state.solToken = null;
    state.sol = null;
    $("tab-sol").hidden = true;
    const info = cur.info || {};

    $("model-title").textContent = cur.title;
    const kind = $("model-kind");
    kind.hidden = !info.kind;
    kind.className = `kind kind-${info.kind}`;
    kind.textContent = info.kind || "";
    $("model-path").textContent = cur.path || "";
    renderStats(info);

    const note = $("run-note");
    note.hidden = !cur.note;
    note.textContent = cur.note || "";

    $("engine").value = "";
    updateEngineHint();
    btnSolve.disabled = false;
    resetResult();
    markActive();
  }

  function renderStats(info) {
    const stats = $("model-stats");
    stats.innerHTML = "";
    const stat = (label, value) => {
      const d = el("div");
      d.append(el("dt", null, label), el("dd", null, value));
      stats.append(d);
    };
    if (info.rows != null) stat("rows", fmtInt(info.rows));
    if (info.cols != null) stat("columns", fmtInt(info.cols));
    if (info.nnz != null) stat("nonzeros", fmtInt(info.nnz));
    if (info.integers) stat("integer", fmtInt(info.integers));
    if (info.bytes != null) stat("file", fmtBytes(info.bytes));
  }

  function setSource(text, { readonly = false, truncated = false } = {}) {
    mpsEl.value = text || "";
    mpsEl.readOnly = readonly;
    state.baseline = mpsEl.value;
    const lines = (text || "").split("\n").length;
    $("editor-meta").textContent = text
      ? truncated
        ? `First ${fmtBytes(new Blob([text]).size)} shown · read-only`
        : `${fmtInt(lines)} lines` + (readonly ? " · read-only" : " · editable")
      : "";
    updateDirty();
  }

  const isDirty = () => state.source === "mps" && !mpsEl.readOnly && mpsEl.value !== state.baseline;
  const updateDirty = () => { $("dirty").hidden = !isDirty(); };

  function showTab(tab) {
    state.source = tab;
    $("tab-mps").classList.toggle("active", tab === "mps");
    $("tab-eq").classList.toggle("active", tab === "eq");
    $("tab-sol").classList.toggle("active", tab === "sol");
    $("sol-pane").hidden = tab !== "sol";
    mpsEl.hidden = tab !== "mps";
    $("eq-pane").hidden = tab !== "eq";
    updateDirty();
  }

  function modelToCurrent(p, origin, extra = {}) {
    const libEntry = state.groups.flatMap((g) => g.models).find((m) => m.id === p.id);
    const notes = [];
    if (p.truncated) notes.push(`Large model: the editor shows the first ${fmtBytes(p.text.length)}. Solve always uses the full file.`);
    if (libEntry?.meta?.generator) notes.push(`Synthetic instance from ${libEntry.meta.generator} (${metaLine(libEntry.meta)}).`);
    if (extra.note) notes.push(extra.note);
    return {
      id: p.id,
      title: extra.title || p.file.split(".")[0],
      path: p.id.startsWith("session:") ? p.file : p.id,
      info: p.info,
      origin,
      note: notes.join(" "),
    };
  }

  async function openModel(id) {
    try {
      loadPayload(await api("/api/model?id=" + encodeURIComponent(id)), "repo");
    } catch (e) {
      showError("Could not open model", e);
    }
  }

  function loadPayload(p, origin, extra) {
    $("tab-eq").hidden = true;
    $("tab-mps").textContent = "Source";
    setCurrent(modelToCurrent(p, origin, extra));
    setSource(p.text, { readonly: p.truncated, truncated: p.truncated });
    showTab("mps");
  }

  async function openTemplate(t) {
    eqEl.value = t.text;
    $("tab-eq").hidden = false;
    $("tab-mps").textContent = "MPS preview";
    setCurrent({ id: `template:${t.id}`, title: t.title, path: t.about, info: { kind: t.kind }, origin: "write" });
    showTab("eq");
    await syncEquations();
  }

  let syncTimer = null;
  async function syncEquations() {
    const text = eqEl.value.trim();
    if (!text) return false;
    const meta = $("editor-meta");
    try {
      const fd = new FormData();
      fd.append("model_text", text);
      const j = await api("/api/lp-to-mps", { method: "POST", body: fd });
      setSource(j.mps, { readonly: true });
      if (state.current) {
        state.current.info = { ...j.info };
        const k = $("model-kind");
        k.hidden = false;
        k.className = `kind kind-${j.info.kind}`;
        k.textContent = j.info.kind;
        renderStats(j.info);
        updateEngineHint();
      }
      meta.textContent = `${j.meta.n_vars} variables · ${j.meta.n_constraints} constraints`;
      meta.style.color = "";
      return true;
    } catch (e) {
      meta.textContent = e.message;
      meta.style.color = "var(--bad)";
      return false;
    }
  }

  // ---------- result ----------

  function resetResult() {
    result.className = "card result idle";
    $("result-empty").hidden = false;
    $("result-body").hidden = true;
  }

  function tone(status, timedOut) {
    if (timedOut) return "warn";
    if (status === "Optimal") return "ok";
    if (status === "Infeasible" || status === "Unbounded") return "info";
    if (status && /Feasible|Interrupted|Limit/i.test(status)) return "warn";
    return "bad";
  }

  function showRTab(name) {
    for (const b of document.querySelectorAll(".rtab")) b.classList.toggle("active", b.dataset.rtab === name);
    for (const p of ["checks", "report", "command", "log"]) $(`rpanel-${p}`).hidden = p !== name;
  }

  function clearSummary(cls, statusText, human) {
    result.className = `card result ${cls}`;
    $("result-empty").hidden = true;
    $("result-body").hidden = false;
    $("status").textContent = statusText;
    $("proof").textContent = "";
    $("wall").textContent = "";
    $("objective-wrap").hidden = true;
    $("human").textContent = human || "";
    $("notes").innerHTML = "";
    $("tiles").innerHTML = "";
    $("verify-line").hidden = true;
    $("rtabs").hidden = true;
    for (const p of ["checks", "report", "command", "log"]) $(`rpanel-${p}`).hidden = true;
  }

  const showRunning = () => clearSummary("run", "Solving…", "Running sor_solve…");
  const showError = (title, e) => clearSummary("bad", title, e.message || String(e));

  function renderTiles(rows) {
    const box = $("tiles");
    box.innerHTML = "";
    for (const [k, v] of rows.slice(0, TILE_COUNT)) {
      const t = el("dl", "tile");
      const [main] = v.split(/\s+/);
      const dd = el("dd", null, main);
      dd.title = v;
      t.append(el("dt", null, cap(k)), dd);
      box.append(t);
    }
  }

  function renderSolve(j) {
    const t = tone(j.status, j.timed_out);
    clearSummary(t, j.timed_out ? "Timed out" : j.status || "Error", j.human && j.human !== j.status ? j.human : "");
    $("proof").textContent = j.proof_level && j.proof_level !== "None" ? j.proof_level : "";
    $("proof").title = "Proof level reported by the solver's certification gate";
    $("wall").textContent = j.wall_s != null ? `${Number(j.wall_s).toFixed(3)} s` : "";

    // An Infeasible / Unbounded verdict has no meaningful objective value.
    const hasObj = j.objective != null && t !== "info";
    $("objective-wrap").hidden = !hasObj;
    $("objective").textContent = hasObj ? fmtObjective(j.objective) : "";

    const notes = $("notes");
    for (const n of [
      ...(j.header || []).filter(([k]) => k === k.toUpperCase()).map(([k, v]) => `${k}: ${v}`),
      ...(j.warnings || []),
      ...(j.timed_out ? [j.stderr] : []),
    ]) notes.append(el("li", null, n));

    const report = j.report || [];
    renderTiles(report);

    const dl = $("report");
    dl.innerHTML = "";
    for (const [k, v] of [...report, ...(j.header || []).filter(([k]) => k !== k.toUpperCase())]) {
      dl.append(el("dt", null, cap(k)), el("dd", null, v));
    }

    $("cmd").textContent = (j.cmd || []).map(shellQuote).join(" ");
    $("log").textContent = [j.stdout?.trimEnd(), j.stderr?.trim() ? "--- stderr ---\n" + j.stderr.trimEnd() : ""]
      .filter(Boolean)
      .join("\n\n");

    $("rtabs").hidden = false;
    state.solToken = j.check_ready ? j.sol_token : null;
    $("rtab-checks").hidden = !state.solToken;
    $("tab-sol").hidden = true;
    state.sol = null;
    if (state.source === "sol") showTab(state.current?.origin === "write" ? "eq" : "mps");
    showRTab(state.solToken ? "checks" : t === "bad" || $("verbose").checked ? "log" : "report");
    if (state.solToken) loadSolution();
    renderCheckIntro(j);
  }

  function setVerifyLine(cls, strong, text, go) {
    const v = $("verify-line");
    v.hidden = false;
    v.className = "verify-line " + (cls || "");
    v.innerHTML = "";
    if (strong) v.append(el("strong", null, strong));
    v.append(el("span", null, text));
    if (go) v.append(el("span", "go", go));
  }

  function renderCheckIntro(j) {
    const box = $("check");
    box.innerHTML = "";
    if (!state.solToken) return;

    if (j.check_authoritative) {
      if ($("autocheck").checked) return runCheck();
      setVerifyLine("", null, "Independent check not run.", "Run →");
      $("verify-line").onclick = runCheck;
      const b = el("button", "btn btn-small", "Run sor_check");
      b.type = "button";
      b.onclick = runCheck;
      box.append(b);
      return;
    }
    const kind = j.model_info?.kind || "this";
    const scope = state.checkClasses.join(" / ");
    setVerifyLine("info", null, `sor_check certifies ${scope} models only. Not run for ${kind}.`, "Details →");
    $("verify-line").onclick = () => showRTab("checks");
    const v = el("div", "verdict info");
    v.append(el("span", null,
      `The independent checker is authoritative for ${scope} models in this build. ` +
      `Its verdict on a ${kind} answer isn't reliable yet, so it doesn't run automatically.`));
    box.append(v);
    const b = el("button", "btn btn-small", "Run sor_check anyway");
    b.type = "button";
    b.onclick = runCheck;
    box.append(b);
  }

  async function runCheck() {
    if (!state.solToken) return;
    const box = $("check");
    box.innerHTML = "";
    box.append(el("div", "verdict", "Checking…"));
    setVerifyLine("", null, "Running the independent checker…");
    try {
      const fd = new FormData();
      fd.append("sol_token", state.solToken);
      const j = await api("/api/check", { method: "POST", body: fd });
      const passed = j.checks.filter((c) => c.ok).length;
      const cls = !j.authoritative ? "info" : j.passed ? "ok" : "bad";
      const word = j.verdict || (j.passed ? "PASSED" : "FAILED");

      setVerifyLine(cls, word,
        j.authoritative
          ? `${passed}/${j.checks.length} checks passed · independent sor_check`
          : `Not authoritative for ${j.model_kind}`,
        "Details →");
      $("verify-line").onclick = () => showRTab("checks");

      box.innerHTML = "";
      if (j.authoritative) {
        box.append(el("p", "panel-intro", j.passed
          ? "sor_check re-derived every residual from the raw model file. It is a separate binary that shares no search code with the solver."
          : "The solver's claim did not pass the independent checker."));
      } else {
        const v = el("div", "verdict info");
        v.append(el("strong", null, word));
        v.append(el("span", null, `Not authoritative for ${j.model_kind} models. Shown for transparency only.`));
        box.append(v);
      }
      if (j.authoritative && !j.passed && result.classList.contains("ok")) result.classList.replace("ok", "bad");

      if (j.checks?.length) {
        const ul = el("ul", "checks" + (j.authoritative ? "" : " muted"));
        for (const c of j.checks) {
          const li = el("li", c.ok ? "pass" : "fail");
          li.append(el("span", "mark", c.ok ? "✓" : "✗"), el("span", null, cap(c.name)), el("span", "res", `${c.residual} / ${c.tol}`));
          li.title = `residual ${c.residual}, tolerance ${c.tol}`;
          ul.append(li);
        }
        box.append(ul);
      }
      if (j.warnings?.length) {
        const n = el("ul", "notes");
        for (const w of j.warnings) n.append(el("li", null, w));
        box.append(n);
      }
      if (j.stdout) $("log").textContent += "\n\n=== sor_check ===\n" + j.stdout.trimEnd();
    } catch (e) {
      box.innerHTML = "";
      box.append(el("div", "verdict bad", e.message));
      setVerifyLine("bad", null, `Checker failed: ${e.message}`);
    }
  }

  // ---------- solution (.sol) ----------

  const SOL_PAGE = 200;
  const fmtVal = (v) => {
    if (v === 0) return "0";
    const a = Math.abs(v);
    return a >= 1e12 || a < 1e-6 ? v.toExponential(6) : v.toLocaleString(undefined, { maximumFractionDigits: 10 });
  };

  // Name each vector from the model: x -> columns, y -> rows. Anything else is indexed.
  function solViews(s) {
    const names = { x: s.col_names, y: s.row_names };
    const label = { x: "Variables (x)", y: "Duals (y)" };
    const views = Object.entries(s.vectors).map(([k, vals]) => ({
      id: k,
      label: label[k] || k,
      nameHead: k === "x" ? "Variable" : k === "y" ? "Constraint" : "Index",
      valueHead: k === "x" ? "Value" : k === "y" ? "Dual" : "Value",
      rows: vals.map((v, i) => [(names[k] && names[k][i]) || `${k}[${i}]`, v, i]),
    })).filter((v) => v.rows.length);
    views.push({ id: "raw", label: "Raw file", raw: true });
    return views;
  }

  async function loadSolution() {
    const box = $("sol-table");
    try {
      const s = await api("/api/solution?sol_token=" + encodeURIComponent(state.solToken));
      state.sol = { ...s, views: solViews(s) };
      $("tab-sol").hidden = false;
      showTab("sol");
      state.solView = state.sol.views[0].id;
      const sc = $("sol-scalars");
      sc.innerHTML = "";
      for (const [k, v] of s.scalars) {
        const t = el("dl", "tile");
        t.append(el("dt", null, cap(k.replace(/_/g, " "))), el("dd", null, v));
        sc.append(t);
      }
      const seg = $("sol-views");
      seg.innerHTML = "";
      for (const v of state.sol.views) {
        const b = el("button", "seg-btn", v.label + (v.rows ? ` · ${fmtInt(v.rows.length)}` : ""));
        b.type = "button";
        b.dataset.view = v.id;
        b.addEventListener("click", () => { state.solView = v.id; state.solShown = SOL_PAGE; renderSol(); });
        seg.append(b);
      }
      state.solShown = SOL_PAGE;
      renderSol();
    } catch (e) {
      $("tab-sol").hidden = true;
      showError("Could not load solution", e);
    }
  }

  function renderSol() {
    const s = state.sol;
    if (!s) return;
    for (const b of $("sol-views").children) b.classList.toggle("active", b.dataset.view === state.solView);
    const view = s.views.find((v) => v.id === state.solView);
    const box = $("sol-table");
    const more = $("sol-more");
    const q = $("sol-q").value.trim().toLowerCase();
    box.innerHTML = "";
    more.hidden = true;
    for (const id of ["sol-q", "sol-nz"]) $(id).disabled = !!view.raw;

    if (view.raw) {
      const pre = el("pre", "log", s.text);
      box.append(pre);
      $("sol-count").textContent = `${fmtBytes(s.bytes)}${s.truncated ? " · first part shown" : ""} · exactly what sor_solve wrote`;
      return;
    }
    const nz = $("sol-nz").checked;
    const rows = view.rows.filter(([n, v]) =>
      (!nz || v !== 0) && (!q || n.toLowerCase().includes(q) || String(v).includes(q)));
    const nonzero = view.rows.filter((r) => r[1] !== 0).length;
    $("sol-count").textContent =
      `${fmtInt(rows.length)} of ${fmtInt(view.rows.length)} shown · ${fmtInt(nonzero)} non-zero`;

    if (!rows.length) { box.append(el("p", "empty", "Nothing matches.")); return; }
    const table = el("table");
    const head = el("tr");
    head.append(el("th", null, "#"), el("th", null, view.nameHead), el("th", "num", view.valueHead));
    table.append(head);
    for (const [n, v, i] of rows.slice(0, state.solShown)) {
      const tr = el("tr", v === 0 ? "zero" : "");
      const val = el("td", "num", fmtVal(v));
      val.title = String(v);
      tr.append(el("td", "idx", String(i + 1)), el("td", null, n), val);
      table.append(tr);
    }
    box.append(table);
    if (rows.length > state.solShown) {
      more.hidden = false;
      more.textContent = `Show more (${fmtInt(rows.length - state.solShown)} left)`;
    }
  }

  function clearSolFilters() {
    $("sol-q").value = "";
    $("sol-nz").checked = false;
    state.solShown = SOL_PAGE;
    renderSol();
  }

  // ---------- solve ----------

  async function solve() {
    if (btnSolve.disabled || !state.current) return;
    btnSolve.disabled = true;
    btnSolve.classList.add("busy");
    $("solve-label").textContent = "Solving";
    showRunning();

    try {
      const fd = new FormData();
      if (state.current.origin === "write") {
        if (!(await syncEquations())) throw new Error($("editor-meta").textContent);
        fd.append("model_text", eqEl.value);
      } else if (isDirty()) {
        fd.append("mps_text", mpsEl.value);
      } else {
        fd.append("model", state.current.id);
      }
      fd.append("engine", $("engine").value);
      fd.append("backend", $("backend").value);
      if ($("method").value) fd.append("method", $("method").value);
      if ($("threads").value) fd.append("threads", $("threads").value);
      fd.append("time_limit", $("time_limit").value || "0");
      if ($("verbose").checked) fd.append("verbose", "true");

      renderSolve(await api("/api/solve", { method: "POST", body: fd }));
    } catch (e) {
      showError("Failed", e);
    } finally {
      btnSolve.disabled = false;
      btnSolve.classList.remove("busy");
      $("solve-label").textContent = "Solve";
    }
  }

  // ---------- events ----------

  for (const b of document.querySelectorAll(".seg-btn")) b.addEventListener("click", () => showSidePanel(b.dataset.panel));
  for (const b of document.querySelectorAll(".rtab")) b.addEventListener("click", () => showRTab(b.dataset.rtab));

  $("filter").addEventListener("input", renderLibrary);
  $("engine").addEventListener("change", updateEngineHint);
  $("tab-mps").addEventListener("click", () => showTab("mps"));
  $("tab-eq").addEventListener("click", () => showTab("eq"));
  $("tab-sol").addEventListener("click", () => showTab("sol"));
  mpsEl.addEventListener("input", updateDirty);
  eqEl.addEventListener("input", () => {
    clearTimeout(syncTimer);
    syncTimer = setTimeout(syncEquations, 400);
  });

  $("file").addEventListener("change", async () => {
    const f = $("file").files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    try {
      const p = await api("/api/upload", { method: "POST", body: fd });
      loadPayload(p, "upload", { title: f.name.split(".")[0], note: `Uploaded ${f.name}.` });
    } catch (e) {
      showError("Upload failed", e);
    } finally {
      $("file").value = "";
    }
  });

  $("sol-q").addEventListener("input", () => { state.solShown = SOL_PAGE; renderSol(); });
  $("sol-nz").addEventListener("change", () => { state.solShown = SOL_PAGE; renderSol(); });
  $("sol-clear").addEventListener("click", clearSolFilters);
  $("sol-more").addEventListener("click", () => { state.solShown += SOL_PAGE; renderSol(); });
  $("sol-copy").addEventListener("click", async () => {
    if (!state.sol) return;
    try {
      await navigator.clipboard.writeText(state.sol.text);
      $("sol-copy").textContent = "Copied";
      setTimeout(() => ($("sol-copy").textContent = "Copy"), 1200);
    } catch { /* clipboard blocked */ }
  });
  $("sol-dl").addEventListener("click", () => {
    if (!state.sol) return;
    const a = el("a");
    a.href = URL.createObjectURL(new Blob([state.sol.text], { type: "text/plain" }));
    a.download = `${(state.current?.title || "model").replace(/\W+/g, "_")}.sol`;
    a.click();
    URL.revokeObjectURL(a.href);
  });

  $("copy-cmd").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("cmd").textContent);
      $("copy-cmd").textContent = "Copied";
      setTimeout(() => ($("copy-cmd").textContent = "Copy"), 1200);
    } catch { /* clipboard blocked */ }
  });

  document.addEventListener("click", (e) => {
    const more = $("more");
    if (more.open && !more.contains(e.target)) more.open = false;
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      solve();
    }
  });

  btnSolve.addEventListener("click", solve);

  // ---------- boot ----------

  async function boot() {
    let health;
    try {
      health = await api("/api/health");
    } catch (e) {
      setAlert("Server offline - run web/run.sh");
      showError("Server offline", e);
      return;
    }
    const missing = Object.entries(health.bins || {}).filter(([, ok]) => !ok).map(([n]) => n);
    if (missing.length) setAlert(`Missing ${missing.join(", ")}`, health.bin_dir);

    state.caps = health.capabilities || {};
    state.checkClasses = health.check_classes || [];
    state.templates = health.templates || [];
    renderControls(health);
    renderTemplates();

    try {
      const { groups } = await api("/api/models");
      state.groups = groups || [];
      renderLibrary();
      const all = state.groups.flatMap((g) => g.models);
      const first = all.find((m) => m.meta) || all[0];
      if (first) await openModel(first.id);
    } catch (e) {
      $("library").innerHTML = "";
      $("library").append(el("p", "empty", e.message));
    }
  }

  boot();
})();
