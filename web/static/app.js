(() => {
  const $ = (id) => document.getElementById(id);

  const PLAIN = {
    Optimal: "Proved best answer for this model.",
    Feasible: "Good feasible answer — not yet proved globally best.",
    Interrupted: "Stopped on the time limit. Best answer found so far.",
    Infeasible: "No feasible answer exists for this model.",
    Unbounded: "Objective can improve without bound.",
    NumericalFailure: "Numerics failed — try another engine or tighten the model.",
    Unsupported: "This model needs a capability we have not shipped yet.",
    Timeout: "Stopped — raise the time limit in Advanced.",
    error: "Something went wrong. See the log in Advanced.",
  };

  const ORDER = ["blend", "schedule", "dispatch", "sparse"];
  const FALLBACK_TEMPLATES = [
    {
      id: "toy_lp",
      label: "Start here · tiny LP",
      blurb: "Two variables, two inequalities.",
      text: `Maximize
  3 x + 4 y
Subject To
  wood:   x + 2 y <= 14
  metal:  3 x + y <= 18
Bounds
  x >= 0
  y >= 0
End
`,
    },
  ];

  let selected = "blend";
  let mode = "preset";
  let solToken = null;
  let presets = {};
  let templates = FALLBACK_TEMPLATES;
  let activeTmpl = null;
  let customName = null;

  const answer = $("answer");
  const btnSolve = $("btn-solve");
  const btnCheck = $("btn-check");
  const btnWrite = $("btn-write");
  const writer = $("writer");
  const logEl = $("log");

  function currentLabel() {
    if (mode === "write") {
      const t = templates.find((x) => x.id === activeTmpl);
      return t ? t.label : "Your equations";
    }
    if (mode === "file" && customName) return customName;
    return presets[selected]?.label || selected;
  }

  function updateSolveHint() {
    $("solve-hint").textContent = currentLabel();
  }

  function renderTemplates() {
    const box = $("templates");
    if (!box) return;
    box.innerHTML = "";
    for (const t of templates) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "tmpl" + (t.id === activeTmpl ? " active" : "");
      b.title = t.blurb || "";
      b.textContent = t.label;
      b.addEventListener("click", () => applyTemplate(t.id));
      box.appendChild(b);
    }
  }

  function applyTemplate(id) {
    const t = templates.find((x) => x.id === id) || templates[0];
    if (!t) return;
    activeTmpl = t.id;
    $("model_text").value = t.text;
    mode = "write";
    writer.hidden = false;
    btnWrite.classList.add("active");
    customName = null;
    $("file").value = "";
    $("file").closest(".file-hit").classList.remove("has-file");
    $("engine").value = /Binary|General/i.test(t.text) ? "milp" : "simplex";
    $("time_limit").value = "30";
    renderTemplates();
    renderProblems();
    updateSolveHint();
    resetAnswer();
  }

  function setWriteMode(on) {
    if (on) {
      mode = "write";
      customName = null;
      $("file").value = "";
      $("file").closest(".file-hit").classList.remove("has-file");
      writer.hidden = false;
      btnWrite.classList.add("active");
      if (!$("model_text").value.trim() && templates[0]) applyTemplate(templates[0].id);
      else {
        renderTemplates();
        updateSolveHint();
        resetAnswer();
      }
    } else {
      writer.hidden = true;
      btnWrite.classList.remove("active");
      activeTmpl = null;
    }
    renderProblems();
  }

  function setAnswer(modeCls, verdict, plain, objective, bits) {
    answer.className = "answer " + modeCls;
    $("verdict").textContent = verdict;
    $("plain").textContent = plain || "";
    if (objective === null || objective === undefined || Number.isNaN(Number(objective))) {
      $("obj").textContent = "";
    } else {
      const n = Number(objective);
      $("obj").textContent =
        Math.abs(n) >= 1e6 || (Math.abs(n) > 0 && Math.abs(n) < 1e-3)
          ? n.toExponential(6)
          : n.toLocaleString(undefined, { maximumFractionDigits: 6 });
    }

    const meta = $("meta");
    if (bits) {
      meta.hidden = false;
      $("meta-time").textContent = bits.time || "";
      $("meta-proof").textContent = bits.proof || "";
      $("meta-check").textContent = bits.check || "";
    } else {
      meta.hidden = true;
    }
  }

  function showLog(text, force) {
    const want = force || $("verbose").checked;
    if (!want || !text) {
      logEl.hidden = true;
      logEl.textContent = "";
      return;
    }
    logEl.hidden = false;
    logEl.textContent = text;
  }

  function renderProblems() {
    const box = $("problems");
    box.innerHTML = "";
    const ids = ORDER.filter((id) => presets[id]).concat(
      Object.keys(presets).filter((id) => !ORDER.includes(id))
    );
    for (const id of ids) {
      const p = presets[id];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "problem" + (mode === "preset" && id === selected ? " active" : "");
      btn.innerHTML =
        `<span class="kind">${escapeHtml(p.kind)}</span>` +
        `<span class="name">${escapeHtml(shortName(p.label))}</span>` +
        `<span class="blurb">${escapeHtml(p.blurb)}</span>`;
      btn.addEventListener("click", () => {
        selected = id;
        mode = "preset";
        customName = null;
        activeTmpl = null;
        writer.hidden = true;
        btnWrite.classList.remove("active");
        $("file").value = "";
        $("file").closest(".file-hit").classList.remove("has-file");
        $("engine").value = p.engine || "simplex";
        $("backend").value = p.backend || "cpu";
        if (p.kind === "milp") $("time_limit").value = "30";
        else $("time_limit").value = "15";
        renderProblems();
        updateSolveHint();
        resetAnswer();
      });
      box.appendChild(btn);
    }
  }

  function shortName(label) {
    return String(label)
      .replace(/\s+LP$/i, "")
      .replace(/\s+MILP$/i, "")
      .replace(/\s+QP$/i, "")
      .replace(/\s*\(first-order\)/i, "")
      .trim();
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function resetAnswer() {
    solToken = null;
    btnCheck.hidden = true;
    setAnswer("idle", "Waiting", "Pick a problem — or write equations — then Solve.", null, null);
    showLog("");
  }

  async function loadHealth() {
    const el = $("health");
    try {
      const r = await fetch("/api/health");
      const j = await r.json();
      presets = j.presets || {};
      if (Array.isArray(j.learn_templates) && j.learn_templates.length) {
        templates = j.learn_templates;
      }
      renderProblems();
      renderTemplates();
      if (presets[selected]) {
        $("engine").value = presets[selected].engine;
        $("backend").value = presets[selected].backend || "cpu";
      }
      updateSolveHint();
      const missing = Object.entries(j.bins || {})
        .filter(([, ok]) => !ok)
        .map(([n]) => n);
      if (!j.ok) {
        el.className = "statusline bad";
        el.textContent = "Solver binaries missing: " + missing.join(", ");
      } else {
        el.className = "statusline";
        el.textContent = "Ready · demos, write-your-own, or .mps";
      }
    } catch {
      el.className = "statusline bad";
      el.textContent = "Server offline — run web/run.sh";
    }
  }

  async function solve() {
    solToken = null;
    btnCheck.hidden = true;
    btnSolve.disabled = true;
    $("solve-label").textContent = "Solving…";
    setAnswer("run", "Working", "Running the from-scratch engine…", null, null);
    showLog("");

    const fd = new FormData();
    const file = $("file").files[0];
    const text = $("model_text").value.trim();

    if (mode === "write" || (!file && text && !writer.hidden)) {
      if (!text) {
        setAnswer("bad", "Failed", "Write some equations first (use a starter above).", null, null);
        btnSolve.disabled = false;
        $("solve-label").textContent = "Solve";
        return;
      }
      fd.append("model_text", text);
    } else if (file) {
      fd.append("file", file);
    } else {
      fd.append("preset", selected);
    }

    fd.append("engine", $("engine").value);
    fd.append("backend", $("backend").value);
    fd.append("method", $("method").value);
    fd.append("time_limit", $("time_limit").value || "30");
    if ($("verbose").checked) fd.append("verbose", "true");

    try {
      const r = await fetch("/api/solve", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) {
        const detail = typeof j.detail === "string" ? j.detail : "Request failed";
        setAnswer("bad", "Failed", detail, null, null);
        showLog(JSON.stringify(j, null, 2), true);
        return;
      }

      if (j.engine) $("engine").value = j.engine;

      const status = j.timed_out ? "Timeout" : j.status || "error";
      const modeCls =
        status === "Optimal" ? "ok" :
        status === "Feasible" || status === "Interrupted" || status === "Timeout" ? "warn" :
        "bad";

      const verdict =
        status === "Optimal" ? "Optimal" :
        status === "Feasible" ? "Feasible" :
        status === "Interrupted" || status === "Timeout" ? "Time limit" :
        status;

      setAnswer(modeCls, verdict, PLAIN[status] || j.human || "", j.objective, {
        time: j.wall_s != null ? `${Number(j.wall_s).toFixed(3)} s` : "",
        proof: j.proof_level ? `proof ${j.proof_level}` : "",
        check: "",
      });

      const parts = [];
      if (j.cmd) parts.push("$ " + j.cmd.join(" "));
      if (j.stdout) parts.push(j.stdout.trimEnd());
      if (j.stderr) parts.push("--- stderr ---\n" + j.stderr.trimEnd());
      showLog(parts.join("\n\n"));

      if (j.check_ready && j.sol_token) {
        solToken = j.sol_token;
        btnCheck.hidden = false;
        if ($("autocheck").checked) await verify(true);
      }
    } catch (e) {
      setAnswer("bad", "Failed", String(e), null, null);
      showLog(String(e), true);
    } finally {
      btnSolve.disabled = false;
      $("solve-label").textContent = "Solve";
    }
  }

  async function verify(silent) {
    if (!solToken) return;
    btnCheck.disabled = true;
    if (!silent) $("meta-check").textContent = "checking…";

    const fd = new FormData();
    fd.append("sol_token", solToken);
    try {
      const r = await fetch("/api/check", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) {
        $("meta-check").textContent = "check error";
        return;
      }
      $("meta-check").textContent = j.passed ? "✓ independently verified" : "✗ check failed";
      if (!j.passed && answer.classList.contains("ok")) {
        answer.classList.remove("ok");
        answer.classList.add("bad");
        $("plain").textContent = "Solver claim did not pass the independent checker.";
      }
      if ($("verbose").checked && j.stdout) {
        showLog((logEl.textContent ? logEl.textContent + "\n\n" : "") + "=== verify ===\n" + j.stdout, true);
      }
    } catch (e) {
      $("meta-check").textContent = String(e);
    } finally {
      btnCheck.disabled = false;
    }
  }

  btnWrite.addEventListener("click", () => {
    if (mode === "write" && !writer.hidden) {
      // already open — reload first starter if empty
      if (!$("model_text").value.trim() && templates[0]) applyTemplate(templates[0].id);
      return;
    }
    setWriteMode(true);
    if (templates[0]) applyTemplate(templates[0].id);
  });

  $("model_text").addEventListener("input", () => {
    if (mode !== "write") {
      mode = "write";
      btnWrite.classList.add("active");
      writer.hidden = false;
    }
    activeTmpl = null;
    renderTemplates();
    updateSolveHint();
  });

  $("file").addEventListener("change", () => {
    const f = $("file").files[0];
    const hit = $("file").closest(".file-hit");
    if (f) {
      mode = "file";
      customName = f.name;
      activeTmpl = null;
      writer.hidden = true;
      btnWrite.classList.remove("active");
      hit.classList.add("has-file");
      hit.querySelector("span").textContent = "Using " + f.name;
      if (f.name.toLowerCase().endsWith(".qps")) $("engine").value = "qp";
      else if ($("engine").value === "qp") $("engine").value = "simplex";
      renderProblems();
      updateSolveHint();
      resetAnswer();
    }
  });

  $("verbose").addEventListener("change", () => {
    if (!$("verbose").checked) showLog("");
  });

  btnSolve.addEventListener("click", solve);
  btnCheck.addEventListener("click", () => verify(false));
  loadHealth();
})();
