const state = {
  activeJob: null,
  runs: [],
  selectedRunId: null,
  selectedArtifactPath: null,
  selectedArtifactName: null,
  pollTimer: null,
};

const pageMode = document.body.dataset.page || "mission";

const els = {
  repoRoot: document.getElementById("repo-root"),
  runsDir: document.getElementById("runs-dir"),
  heroStatus: document.getElementById("hero-status"),
  launchForm: document.getElementById("launch-form"),
  launchRun: document.getElementById("launch-run"),
  mode: document.getElementById("mode"),
  model: document.getElementById("model"),
  registryPreset: document.getElementById("registry-preset"),
  registryPath: document.getElementById("registry-path"),
  logOutput: document.getElementById("log-output"),
  jobPill: document.getElementById("job-pill"),
  jobCommand: document.getElementById("job-command"),
  stopRun: document.getElementById("stop-run"),
  runsList: document.getElementById("runs-list"),
  detailEmpty: document.getElementById("detail-empty"),
  detailContent: document.getElementById("detail-content"),
  detailTitle: document.getElementById("detail-title"),
  detailSubtitle: document.getElementById("detail-subtitle"),
  detailHeadline: document.getElementById("detail-headline"),
  detailJson: document.getElementById("detail-json"),
  artifactLinks: document.getElementById("artifact-links"),
  detailDashboardLink: document.getElementById("detail-dashboard-link"),
  artifactPreviewFrame: document.getElementById("artifact-preview-frame"),
  artifactPreviewText: document.getElementById("artifact-preview-text"),
  artifactPreviewEmpty: document.getElementById("artifact-preview-empty"),
  previewLabel: document.getElementById("preview-label"),
  formStatus: document.getElementById("form-status"),
  stressFields: document.getElementById("stress-fields"),
  controlFields: document.getElementById("control-fields"),
};

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function formatDate(epochSeconds) {
  if (!epochSeconds) return "Unknown time";
  return new Date(epochSeconds * 1000).toLocaleString();
}

function modeFields() {
  const mode = els.mode.value;
  els.stressFields.classList.toggle("hidden", mode !== "stress");
  els.controlFields.classList.toggle("hidden", mode !== "control");
}

function renderModels(models) {
  const current = els.model.value;
  els.model.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.key;
    option.textContent = `${model.key}  ${model.hf_id}`;
    els.model.appendChild(option);
  }
  if (current && [...els.model.options].some((opt) => opt.value === current)) {
    els.model.value = current;
  }
}

function renderRegistryPresets(presets, currentPath) {
  els.registryPreset.innerHTML = "";
  for (const preset of presets) {
    const option = document.createElement("option");
    option.value = preset.path;
    option.textContent = `${preset.label}  (${preset.path})`;
    els.registryPreset.appendChild(option);
  }
  if (currentPath && [...els.registryPreset.options].some((opt) => opt.value === currentPath)) {
    els.registryPreset.value = currentPath;
  }
}

function renderJob(job) {
  state.activeJob = job;
  const status = job?.status || "idle";
  if (els.heroStatus) {
    els.heroStatus.textContent = status.toUpperCase();
  }
  if (els.jobPill) {
    els.jobPill.textContent = status.toUpperCase();
  }
  if (els.jobCommand) {
    els.jobCommand.textContent = job?.command ? job.command.join(" ") : "No active command.";
  }
  if (els.logOutput) {
    els.logOutput.textContent = job?.logs?.length ? job.logs.join("\n") : "Waiting for your first run.";
  }
  if (els.stopRun) {
    els.stopRun.disabled = !(status === "running" || status === "stopping");
  }
}

function setFormStatus(message, tone = "info") {
  if (!els.formStatus) return;
  els.formStatus.textContent = message;
  els.formStatus.classList.remove("hidden", "is-info", "is-success", "is-error");
  els.formStatus.classList.add(`is-${tone}`);
}

function clearFormStatus() {
  if (!els.formStatus) return;
  els.formStatus.textContent = "";
  els.formStatus.classList.add("hidden");
  els.formStatus.classList.remove("is-info", "is-success", "is-error");
}

function setLaunching(isLaunching) {
  if (!els.launchRun) return;
  els.launchRun.disabled = isLaunching;
  els.launchRun.textContent = isLaunching ? "Launching..." : "Launch Run";
}

function buildHeadlineCards(run) {
  if (!els.detailHeadline) return;
  els.detailHeadline.innerHTML = "";
  const headline = run.headline || {};
  const entries = Object.entries(headline).slice(0, 6);
  if (!entries.length) return;
  for (const [key, value] of entries) {
    const card = document.createElement("div");
    card.className = "headline-card";
    const label = document.createElement("span");
    label.textContent = key.replaceAll("_", " ");
    const strong = document.createElement("strong");
    strong.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
    card.append(label, strong);
    els.detailHeadline.appendChild(card);
  }
}

function resetArtifactPreview() {
  if (!els.previewLabel || !els.artifactPreviewFrame || !els.artifactPreviewText || !els.artifactPreviewEmpty) return;
  els.previewLabel.textContent = "Nothing selected";
  els.artifactPreviewFrame.classList.add("hidden");
  els.artifactPreviewText.classList.add("hidden");
  els.artifactPreviewEmpty.classList.remove("hidden");
  els.artifactPreviewFrame.removeAttribute("src");
  els.artifactPreviewText.textContent = "";
}

async function previewArtifact(name, relPath) {
  if (!els.previewLabel || !els.artifactPreviewFrame || !els.artifactPreviewText || !els.artifactPreviewEmpty) return;
  state.selectedArtifactName = name;
  state.selectedArtifactPath = relPath;
  els.previewLabel.textContent = name;
  const url = `/api/artifact?path=${encodeURIComponent(relPath)}`;
  const lower = name.toLowerCase();

  if (lower.endsWith(".html")) {
    els.artifactPreviewEmpty.classList.add("hidden");
    els.artifactPreviewText.classList.add("hidden");
    els.artifactPreviewFrame.classList.remove("hidden");
    els.artifactPreviewFrame.src = url;
    return;
  }

  try {
    const res = await fetch(url);
    const text = await res.text();
    els.artifactPreviewEmpty.classList.add("hidden");
    els.artifactPreviewFrame.classList.add("hidden");
    els.artifactPreviewFrame.removeAttribute("src");
    els.artifactPreviewText.classList.remove("hidden");
    els.artifactPreviewText.textContent = text;
  } catch (error) {
    els.artifactPreviewEmpty.classList.add("hidden");
    els.artifactPreviewFrame.classList.add("hidden");
    els.artifactPreviewText.classList.remove("hidden");
    els.artifactPreviewText.textContent = `Preview failed: ${error.message}`;
  }
}

function renderArtifacts(run) {
  if (!els.artifactLinks || !els.detailDashboardLink) return;
  els.artifactLinks.innerHTML = "";
  els.detailDashboardLink.classList.add("hidden");
  els.detailDashboardLink.removeAttribute("href");

  const artifacts = Object.entries(run.artifacts || {});
  const selected = state.selectedArtifactPath
    ? artifacts.find(([, relPath]) => relPath === state.selectedArtifactPath)
    : null;

  for (const [name, relPath] of artifacts) {
    const link = document.createElement("a");
    link.href = `/api/artifact?path=${encodeURIComponent(relPath)}`;
    if (relPath === state.selectedArtifactPath) {
      link.classList.add("active");
    }
    link.addEventListener("click", async (event) => {
      event.preventDefault();
      await previewArtifact(name, relPath);
      renderArtifacts(run);
    });
    link.textContent = name;
    els.artifactLinks.appendChild(link);

    if (name === "dashboard.html" || name === "dashboard_path") {
      els.detailDashboardLink.href = `/api/artifact?path=${encodeURIComponent(relPath)}`;
      els.detailDashboardLink.classList.remove("hidden");
    }
  }

  const preferred = selected
    || artifacts.find(([name]) => name === "summary.json")
    || artifacts.find(([name]) => name === "results.json")
    || artifacts.find(([name]) => name.endsWith(".html"))
    || artifacts[0];

  if (preferred) {
    previewArtifact(preferred[0], preferred[1]);
  } else {
    resetArtifactPreview();
  }
}

function renderRunDetail(run) {
  if (!els.detailEmpty || !els.detailContent || !els.detailTitle || !els.detailSubtitle || !els.detailJson) return;
  const changedRun = state.selectedRunId !== run.id;
  state.selectedRunId = run.id;
  if (changedRun) {
    state.selectedArtifactPath = null;
    state.selectedArtifactName = null;
  }
  if (pageMode === "archive") {
    const url = new URL(window.location.href);
    url.searchParams.set("run", run.id);
    window.history.replaceState({}, "", url);
  }
  els.detailEmpty.classList.add("hidden");
  els.detailContent.classList.remove("hidden");
  els.detailTitle.textContent = run.id;
  els.detailSubtitle.textContent = `${run.mode.toUpperCase()}  •  ${formatDate(run.updated_at)}`;
  buildHeadlineCards(run);
  renderArtifacts(run);
  els.detailJson.textContent = JSON.stringify(run.summary || run.results || run.config || {}, null, 2);
  renderRuns(state.runs);
}

function renderRuns(runs) {
  state.runs = runs;
  if (!els.runsList) return;
  els.runsList.innerHTML = "";

  if (!runs.length) {
    const empty = document.createElement("p");
    empty.textContent = "No runs yet. Launch one from the left panel.";
    els.runsList.appendChild(empty);
    return;
  }

  for (const run of runs) {
    const card = document.createElement(pageMode === "archive" ? "button" : "a");
    if (pageMode === "archive") {
      card.type = "button";
      card.addEventListener("click", async () => {
        const detail = await fetchJSON(`/api/run/${run.id}`);
        renderRunDetail(detail.run);
      });
    } else {
      card.href = `/runs.html?run=${encodeURIComponent(run.id)}`;
    }
    card.className = "run-card";
    if (run.id === state.selectedRunId) card.classList.add("active");

    const header = document.createElement("header");
    const title = document.createElement("h3");
    title.textContent = run.id;
    const mode = document.createElement("span");
    mode.className = "run-mode";
    mode.textContent = run.mode;
    header.append(title, mode);

    const desc = document.createElement("p");
    const headline = run.headline || {};
    const summary = Object.entries(headline)
      .slice(0, 2)
      .map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`)
      .join("  •  ");
    desc.textContent = summary || "No headline metrics yet.";

    const meta = document.createElement("div");
    meta.className = "meta-row";
    const updated = document.createElement("span");
    updated.className = "meta-chip";
    updated.textContent = formatDate(run.updated_at);
    meta.append(updated);

    card.append(header, desc, meta);
    els.runsList.appendChild(card);
  }
}

async function loadConfig() {
  const registry = encodeURIComponent(els.registryPath?.value || "models.json");
  const data = await fetchJSON(`/api/config?registry=${registry}`);
  if (els.repoRoot) {
    els.repoRoot.textContent = data.repo_root;
  }
  if (els.runsDir) {
    els.runsDir.textContent = data.runs_dir;
  }
  if (els.registryPreset) {
    renderRegistryPresets(data.registry_presets || [], els.registryPath.value || "models.json");
  }
  if (els.model) {
    renderModels(data.models.models || []);
    if (data.models.default_model) {
      els.model.value = data.models.default_model;
    }
  }
  renderJob(data.active_job);
}

async function refreshStatus() {
  const [{ job }, { runs }] = await Promise.all([
    fetchJSON("/api/status"),
    fetchJSON("/api/runs"),
  ]);
  renderJob(job);
  renderRuns(runs);
  if (pageMode === "archive" && state.selectedRunId && runs.some((run) => run.id === state.selectedRunId)) {
    const detail = await fetchJSON(`/api/run/${state.selectedRunId}`);
    renderRunDetail(detail.run);
  }
}

function collectPayload() {
  const mode = els.mode.value;
  const form = new FormData(els.launchForm);
  const payload = {
    mode,
    registry_path: form.get("registry_path"),
    model: form.get("model"),
    custom_model_key: form.get("custom_model_key"),
    backend: form.get("backend"),
    max_tokens: Number(form.get("max_tokens")),
    seed: Number(form.get("seed")),
    prompt: form.get("prompt"),
    nnsight_device: form.get("nnsight_device"),
    nnsight_remote: form.get("nnsight_remote") === "on",
    shadow: form.get("shadow") === "on",
  };

  if (mode === "stress") {
    payload.layer = Number(form.get("layer"));
    payload.intervention_type = form.get("stress_intervention_type");
    payload.magnitude = Number(form.get("magnitude"));
    payload.start = Number(form.get("start"));
    payload.duration = Number(form.get("duration"));
  }

  if (mode === "control") {
    payload.measure_layer = Number(form.get("measure_layer"));
    payload.act_layer = Number(form.get("act_layer"));
    payload.intervention_type = form.get("control_intervention_type");
  }

  return payload;
}

if (els.mode) {
  els.mode.addEventListener("change", modeFields);
}
if (els.registryPreset) {
  els.registryPreset.addEventListener("change", async () => {
    els.registryPath.value = els.registryPreset.value;
    await loadConfig();
  });
}
if (els.registryPath) {
  els.registryPath.addEventListener("change", loadConfig);
}

if (els.launchForm) {
  els.launchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setLaunching(true);
    setFormStatus("Sending run to the local launcher…", "info");
    try {
      const payload = collectPayload();
      const data = await fetchJSON("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderJob(data.job);
      setFormStatus("Run launched. Logs will stream below.", "success");
      await refreshStatus();
    } catch (error) {
      setFormStatus(error.message, "error");
    } finally {
      setLaunching(false);
    }
  });
}

if (els.stopRun) {
  els.stopRun.addEventListener("click", async () => {
    try {
      const data = await fetchJSON("/api/stop", { method: "POST" });
      renderJob(data.job);
      setFormStatus("Stop signal sent to the active run.", "info");
    } catch (error) {
      setFormStatus(error.message, "error");
    }
  });
}

async function boot() {
  state.selectedRunId = new URLSearchParams(window.location.search).get("run");
  if (els.mode) {
    modeFields();
  }
  clearFormStatus();
  await loadConfig();
  await refreshStatus();
  if (pageMode === "archive" && !state.selectedRunId && state.runs.length) {
    const detail = await fetchJSON(`/api/run/${state.runs[0].id}`);
    renderRunDetail(detail.run);
  } else if (pageMode === "archive" && state.selectedRunId) {
    try {
      const detail = await fetchJSON(`/api/run/${state.selectedRunId}`);
      renderRunDetail(detail.run);
    } catch (error) {
      state.selectedRunId = null;
      if (els.detailEmpty) {
        els.detailEmpty.textContent = `That run could not be loaded: ${error.message}`;
      }
    }
  }
  state.pollTimer = window.setInterval(refreshStatus, 2500);
}

boot().catch((error) => {
  els.logOutput.textContent = `Dashboard boot failed: ${error.message}`;
});
