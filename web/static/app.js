(() => {
  const $ = (id) => document.getElementById(id);

  const PLAIN = {
    Optimal: "Proved best answer for this model.",
    Feasible: "Good feasible answer - not yet proved globally best.",
    Interrupted: "Stopped on the time limit. Best answer found so far.",
    Infeasible: "No feasible answer exists for this model.",
    Unbounded: "Objective can improve without bound.",
    NumericalFailure: "Numerics failed - try another engine or tighten the model.",
    Unsupported: "This model needs a capability we have not shipped yet.",
    Timeout: "Stopped - raise the time limit in Advanced.",
    error: "Something went wrong. See the log.",
  };

  const ORDER = ["blend", "schedule", "dispatch", "sparse"];

  let selected = "blend";
  let mode = "preset"; // preset | write | file
  let sourceTab = "mps"; // mps | eq
  let solToken = null;
  let presets = {};
  let templates = [];
  let activeTmpl = null;
  let customName = null;
  let mpsBaseline = "";
  let mpsDirty = false;

  const answer = $("answer");
  const btnSolve = $("btn-solve");
  const btnCheck = $("btn-check");
  const btnWrite = $("btn-write");
  const logEl = $("log");
  const mpsEl = $("mps_text");
  const eqEl = $("model_text");

  function currentLabel() {
    if (mode === "file" && customName) return customName;
    if (mode === "write") {
      const t = templates.find((x) => x.id === activeTmpl);
      return t ? t.label : "Your equations";
    }
    return presets[selected]?.label || selected || "Model";
  }

  function updateChrome() {
    $("solve-hint").textContent = currentLabel();
    $("model-title").textContent = currentLabel();
    $("dirty").hidden = !mpsDirty;
  }

  function setDirty(on) {
    mpsDirty = on;
    updateChrome();
  }

  function showTab(tab) {
    sourceTab = tab;
    $("tab-mps").classList.toggle("active", tab === "mps");
    $("tab-eq").classList.toggle("active", tab === "eq");
    $("panel-mps").hidden = tab !== "mps";
    $("panel-eq").hidden = tab !== "eq";
  }

  function setMpsText(text, { clean = true } = {}) {
    mpsEl.value = text || "";
    mpsBaseline = clean ? mpsEl.value : mpsBaseline;
    if (clean) setDirty(false);
    const lines = (text || "").split("\n").length;
    const bytes = new Blob([text || ""]).size;
    $("mps-meta").textContent = text
      ? `${lines} lines · ${(bytes / 1024).toFixed(1)} KB · editable`
      : "Select a model to inspect";
  }

  async function loadPreset(id) {
    selected = id;
    mode = "preset";
    customName = null;
    activeTmpl = null;
    btnWrite.classList.remove("active");
    $("file").value = "";
    const hit = $("file").closest(".file-hit");
    hit.classList.remove("has-file");
    hit.querySelector("span").innerHTML = 'Open <code>.mps</code> / <code>.qps</code>';

    const p = presets[id];
    if (p) {
      $("engine").value = p.engine || "simplex";
      $("backend").value = p.backend || "cpu";
      $("time_limit").value = p.kind === "milp" ? "30" : "15";
    }

    $("mps-meta").textContent = "Loading...";
    try {
      const r = await fetch("/api/model/" + encodeURIComponent(id));
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "load failed");
      setMpsText(j.text, { clean: true });
      if (j.truncated) $("mps-meta").textContent += " · truncated";
      showTab("mps");
    } catch (e) {
      setMpsText("", { clean: true });
      $("mps-meta").textContent = String(e);
    }

    renderProblems();
    renderTemplates();
    updateChrome();
    resetAnswer();
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

  async function applyTemplate(id) {
    const t = templates.find((x) => x.id === id) || templates[0];
    if (!t) return;
    mode = "write";
    activeTmpl = t.id;
    customName = null;
    btnWrite.classList.add("active");
    eqEl.value = t.text;
    $("engine").value = /Binary|General/i.test(t.text) ? "milp" : "simplex";
    $("time_limit").value = "30";
    showTab("eq");
    renderTemplates();
    renderProblems();
    updateChrome();
    resetAnswer();
    await syncEqToMps();
  }

  async function syncEqToMps() {
    const text = eqEl.value.trim();
    if (!text) return;
    try {
      const fd = new FormData();
      fd.append("model_text", text);
      const r = await fetch("/api/lp-to-mps", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : "parse failed");
      setMpsText(j.mps, { clean: true });
      if (j.meta?.suggested_engine === "milp" && $("engine").value === "simplex") {
        $("engine").value = "milp";
      }
    } catch (e) {
      $("mps-meta").textContent = String(e);
    }
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
      btn.addEventListener("click", () => loadPreset(id));
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
    } else meta.hidden = true;
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

  function resetAnswer() {
    solToken = null;
    btnCheck.hidden = true;
    setAnswer("idle", "Waiting", "Inspect or edit the model, then Optimize.", null, null);
    showLog("");
  }

  async function loadHealth() {
    const el = $("health");
    try {
      const r = await fetch("/api/health");
      const j = await r.json();
      presets = j.presets || {};
      templates = Array.isArray(j.learn_templates) ? j.learn_templates : [];
      renderProblems();
      renderTemplates();
      const missing = Object.entries(j.bins || {})
        .filter(([, ok]) => !ok)
        .map(([n]) => n);
      if (!j.ok) {
        el.className = "topbar-right bad";
        el.textContent = "Solver binaries missing: " + missing.join(", ");
      } else {
        el.className = "topbar-right";
        el.textContent = "Ready · inspect MPS · edit · optimize";
      }
      if (presets.blend) await loadPreset("blend");
      else if (Object.keys(presets)[0]) await loadPreset(Object.keys(presets)[0]);
    } catch {
      el.className = "topbar-right bad";
      el.textContent = "Server offline - run web/run.sh";
    }
  }

  async function solve() {
    solToken = null;
    btnCheck.hidden = true;
    btnSolve.disabled = true;
    $("solve-label").textContent = "Optimizing...";
    setAnswer("run", "Working", "Running the from-scratch engine...", null, null);
    showLog("");

    // If on equations tab, refresh MPS first so edits flow through.
    if (sourceTab === "eq" && eqEl.value.trim()) {
      await syncEqToMps();
    }

    const fd = new FormData();
    const mps = mpsEl.value.trim();
    // Equations tab: send model_text so Maximize/Minimize display flip is known.
    // Otherwise prefer the MPS buffer (inspect / edit / upload).
    if (sourceTab === "eq" && eqEl.value.trim()) {
      fd.append("model_text", eqEl.value.trim());
    } else if (mps) {
      fd.append("mps_text", mps);
    }
    if (mode === "preset" && selected) {
      fd.append("preset", selected);
    }
    const file = $("file").files[0];
    if (!fd.has("mps_text") && !fd.has("model_text") && file) {
      fd.append("file", file);
    }

    if (!fd.has("mps_text") && !fd.has("model_text") && !fd.has("preset") && !fd.has("file")) {
      setAnswer("bad", "Failed", "Model buffer is empty - pick a model or paste MPS.", null, null);
      btnSolve.disabled = false;
      $("solve-label").textContent = "Optimize";
      return;
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
      // Always keep log available; visible when verbose is on
      if (parts.length) {
        logEl.textContent = parts.join("\n\n");
        logEl.hidden = !$("verbose").checked;
      }

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
      $("solve-label").textContent = "Optimize";
    }
  }

  async function verify(silent) {
    if (!solToken) return;
    btnCheck.disabled = true;
    if (!silent) $("meta-check").textContent = "checking...";
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

  $("tab-mps").addEventListener("click", () => showTab("mps"));
  $("tab-eq").addEventListener("click", () => {
    mode = "write";
    btnWrite.classList.add("active");
    showTab("eq");
    renderProblems();
    updateChrome();
  });

  btnWrite.addEventListener("click", () => {
    mode = "write";
    btnWrite.classList.add("active");
    showTab("eq");
    if (!eqEl.value.trim() && templates[0]) applyTemplate(templates[0].id);
    else {
      renderProblems();
      updateChrome();
    }
  });

  $("btn-to-mps").addEventListener("click", async () => {
    await syncEqToMps();
    showTab("mps");
  });

  mpsEl.addEventListener("input", () => {
    setDirty(mpsEl.value !== mpsBaseline);
    mode = mode === "write" ? "write" : "file";
    updateChrome();
  });

  eqEl.addEventListener("input", () => {
    mode = "write";
    activeTmpl = null;
    btnWrite.classList.add("active");
    renderTemplates();
    updateChrome();
  });

  $("file").addEventListener("change", () => {
    const f = $("file").files[0];
    const hit = $("file").closest(".file-hit");
    if (!f) return;
    mode = "file";
    customName = f.name;
    activeTmpl = null;
    btnWrite.classList.remove("active");
    hit.classList.add("has-file");
    hit.querySelector("span").textContent = "Using " + f.name;
    if (f.name.toLowerCase().endsWith(".qps")) $("engine").value = "qp";
    const reader = new FileReader();
    reader.onload = () => {
      setMpsText(String(reader.result || ""), { clean: true });
      showTab("mps");
      renderProblems();
      updateChrome();
      resetAnswer();
    };
    reader.readAsText(f);
  });

  $("verbose").addEventListener("change", () => {
    if ($("verbose").checked && logEl.textContent.trim()) logEl.hidden = false;
    else if (!$("verbose").checked) logEl.hidden = true;
  });

  btnSolve.addEventListener("click", solve);
  btnCheck.addEventListener("click", () => verify(false));
  loadHealth();
})();
