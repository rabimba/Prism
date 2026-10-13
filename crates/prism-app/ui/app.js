const invoke = window.__TAURI__?.core?.invoke;

const ICONS = {
  search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><defs><linearGradient id="brandGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#ff6b4a"/><stop offset="52%" stop-color="#a855f7"/><stop offset="100%" stop-color="#4f8cff"/></linearGradient></defs><circle cx="11" cy="11" r="7"/><path d="M20 20 L16.2 16.2"/></svg>`,
  code: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 7 L4 12 L9 17 M15 7 L20 12 L15 17"/></svg>`,
  doc: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3 H14 L19 8 V21 H6 Z M14 3 V8 H19 M9 12 H16 M9 15 H16 M9 18 H13"/></svg>`,
  pdf: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3 H14 L19 8 V21 H6 Z M14 3 V8 H19 M9 13 H16 M9 16 H14"/></svg>`,
  audio: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12 H5 M7 8 V16 M10 5 V19 M13 9 V15 M16 6 V18 M19 10 V14 M21 12 H22"/></svg>`,
  video: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6 H15 V18 H4 Z M15 10 L20 7 V17 L15 14"/></svg>`,
  image: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5 H20 V19 H4 Z M4 16 L9 11 L13 15 L16 12 L20 16"/></svg>`,
  chevron: `<svg class="chevron" viewBox="0 0 16 16" width="12" height="12"><path d="M6 3.5 L10.5 8 L6 12.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  play: `<svg viewBox="0 0 20 20"><path d="M5 3 L17 10 L5 17 Z"/></svg>`,
  found: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="6.5"/><path d="M15 15 L20.5 20.5 M10 7 V13 M7 10 H13"/></svg>`,
  mic: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="11" rx="3"/><path d="M5 10a7 7 0 0 0 14 0M12 17v4M8 21h8"/></svg>`,
  contain: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><rect x="8" y="8" width="8" height="8" rx="1.5"/></svg>`,
  grid: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>`,
  list: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>`,
  sidebar: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M15 3v18"/></svg>`,
  gear: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`,
};
ICONS.text = ICONS.doc;
ICONS.document = ICONS.pdf;
ICONS.spreadsheet = ICONS.doc;
ICONS.presentation = ICONS.doc;

const GROUP_TITLES = {
  text: "Documents & Notes",
  document: "PDFs & Office Docs",
  code: "Source Code",
  image: "Images & Screenshots",
  video: "Videos",
  audio: "Audio & Recordings",
  spreadsheet: "Spreadsheets",
  presentation: "Presentations",
};

const KIND_FILTERS = [
  { key: null, label: "All" },
  { key: "document", label: "PDFs & Office" },
  { key: "text", label: "Notes & Docs" },
  { key: "image", label: "Images" },
  { key: "video", label: "Videos" },
  { key: "audio", label: "Audio" },
  { key: "code", label: "Code" },
];

const WEAKER_ID = "__weaker_toggle__";
const TYPING_PAUSE_MS = 110;

const state = {
  activeDir: "ALL_OS",
  stats: null,
  pendingEmbeddings: 0,
  bgTimer: null,
  bgRunning: false,
  presets: [],
  demoProbes: [],
  kindCounts: {},
  query: "",
  kindFilter: null,
  mode: "hybrid",          // hybrid | semantic | lexical
  viewMode: "list",        // list (Spotlight panel split view) | grid (SearchWindow visual grid)
  showPreview: true,
  resultMode: "search",    // search | containment
  strong: [],
  weaker: [],
  groups: [],
  showWeaker: false,
  selectedKey: null,
  generation: 0,
  finalGeneration: -1,
  timer: null,
  elapsedMs: 0,
  phase: "browse",         // browse | words | meaning | containment
};

const previewCache = new Map();

// DOM references
const $dropZone = document.getElementById("dropZone");
const $searchIcon = document.getElementById("searchIcon");
const $queryInput = document.getElementById("queryInput");
const $shortcutHint = document.getElementById("shortcutHint");
const $clearQueryBtn = document.getElementById("clearQueryBtn");
const $voiceDrawerBtn = document.getElementById("voiceDrawerBtn");
const $containDrawerBtn = document.getElementById("containDrawerBtn");
const $viewModeBtn = document.getElementById("viewModeBtn");
const $togglePreviewBtn = document.getElementById("togglePreviewBtn");
const $foldersModalBtn = document.getElementById("foldersModalBtn");
const $scopePills = document.getElementById("scopePills");
const $kindChips = document.getElementById("kindChips");
const $modePills = document.getElementById("modePills");
const $voiceDrawer = document.getElementById("voiceDrawer");
const $containDrawer = document.getElementById("containDrawer");
const $demoProbesBar = document.getElementById("demoProbesBar");
const $statusBanner = document.getElementById("statusBanner");
const $workspaceBody = document.getElementById("workspaceBody");
const $resultsList = document.getElementById("resultsList");
const $previewPane = document.getElementById("previewPane");
const $previewContent = document.getElementById("previewContent");
const $footerFolderBtn = document.getElementById("footerFolderBtn");
const $footerFolderText = document.getElementById("footerFolderText");
const $footerSummary = document.getElementById("footerSummary");
const $statusDot = document.getElementById("statusDot");
const $foldersModal = document.getElementById("foldersModal");
const $folderInput = document.getElementById("folderInput");
const $folderPresetsGrid = document.getElementById("folderPresetsGrid");
const $estimateBox = document.getElementById("estimateBox");

// Inject toolbar icons
$searchIcon.innerHTML = ICONS.search;
document.getElementById("micIconSlot").innerHTML = ICONS.mic;
document.getElementById("containIconSlot").innerHTML = ICONS.contain;
document.getElementById("viewModeIconSlot").innerHTML = ICONS.grid;
document.getElementById("previewIconSlot").innerHTML = ICONS.sidebar;
document.getElementById("gearIconSlot").innerHTML = ICONS.gear;

function hitKey(hit) {
  return `${hit.relative_path}#${hit.chunk_id ?? hit.location_summary ?? 0}`;
}

function clock(seconds) {
  const total = Math.max(0, Math.trunc(Number(seconds) || 0));
  const two = (n) => String(n).padStart(2, "0");
  return total >= 3600
    ? `${Math.trunc(total / 3600)}:${two(Math.trunc(total / 60) % 60)}:${two(total % 60)}`
    : `${Math.trunc(total / 60)}:${two(total % 60)}`;
}

function relativeTime(modifiedUnix) {
  const ts = Number(modifiedUnix) || 0;
  if (ts <= 0) return "";
  const nowSecs = Math.floor(Date.now() / 1000);
  const diff = Math.max(0, nowSecs - ts);
  if (diff < 90) return "Just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 172800) return "Yesterday";
  if (diff < 7 * 86400) return `${Math.floor(diff / 86400)}d ago`;
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function formatBytes(bytes) {
  const b = Number(bytes) || 0;
  if (b <= 0) return "";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(b < 10240 ? 1 : 0)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

function prettyPath(p) {
  const str = String(p || "");
  return str.replace(/^\/home\/[^/]+/, "~").replace(/^\/Users\/[^/]+/, "~");
}

function parentFolderDisplay(p) {
  const pretty = prettyPath(p);
  const idx = pretty.lastIndexOf("/");
  return idx > 0 ? pretty.slice(0, idx) : pretty;
}

function placeOf(hit) {
  if (state.resultMode === "containment") {
    return `${Math.round((hit.confidence || 0) * 100)}% · ${hit.location_summary || "Match"}`;
  }
  const orig = hit.origin || {};
  if (orig.page != null) return `p. ${orig.page}`;
  if (orig.start_seconds != null && orig.end_seconds != null) {
    const prefix = hit.modality === "audio" && hit.kind === "video" ? "♪ " : "";
    return `${prefix}${clock(orig.start_seconds)}–${clock(orig.end_seconds)}`;
  }
  if (orig.line_start != null && orig.line_end != null) {
    return orig.line_start === orig.line_end
      ? `L${orig.line_start}`
      : `L${orig.line_start}–${orig.line_end}`;
  }
  if (orig.line_start != null) {
    return `L${orig.line_start}`;
  }
  return null;
}

function reasonOf(hit) {
  if (state.resultMode === "containment") {
    const parts = [
      `${hit.verdict?.toUpperCase() || "MATCH"} (${((hit.confidence || 0) * 100).toFixed(1)}% confidence)`,
      `sem=${(hit.semantic_similarity || 0).toFixed(3)}`,
    ];
    if (hit.perceptual_hash_similarity != null) parts.push(`aHash=${hit.perceptual_hash_similarity.toFixed(3)}`);
    if (hit.lexical_containment != null && hit.lexical_containment > 0) parts.push(`lex=${hit.lexical_containment.toFixed(3)}`);
    if (hit.temporal_alignment != null) parts.push(`temp=${hit.temporal_alignment.toFixed(3)}`);
    return parts.join(" · ");
  }

  // Ongoing browse mode: clean human recency & size summary without fake scores
  if (state.phase === "browse" || (!state.query.trim() && !hit.raw_cosine)) {
    const rel = relativeTime(hit.modified_unix);
    const sz = formatBytes(hit.size_bytes);
    const folder = parentFolderDisplay(hit.path);
    return [rel ? `Modified ${rel}` : "Indexed file", sz, folder].filter(Boolean).join(" · ");
  }

  const orig = hit.origin || {};
  const details = [];
  if (hit.raw_cosine && hit.raw_cosine > 0.05) {
    details.push(`${Math.round(hit.raw_cosine * 100)}% semantic match`);
  }
  if (hit.semantic_z && Math.abs(hit.semantic_z) >= 0.1) {
    details.push(`z=${hit.semantic_z >= 0 ? "+" : ""}${hit.semantic_z.toFixed(2)}`);
  }
  if (hit.lexical_coverage && hit.lexical_coverage > 0) {
    details.push("Exact words");
  }
  const suffix = details.length ? ` (${details.join(" · ")})` : "";

  if (hit.modality === "image") {
    if (orig.start_seconds != null) {
      return `Matches video frame at ${clock(orig.start_seconds)}${suffix}`;
    }
    return `Matches visual appearance${suffix}`;
  }
  if (hit.modality === "audio") {
    const at = orig.start_seconds != null ? ` at ${clock(orig.start_seconds)}` : "";
    return `Matches what's heard${at} via 768-D Conformer + Semantic ASR${suffix}`;
  }
  if (orig.page != null) {
    return `Matches page ${orig.page}${suffix}`;
  }
  if (orig.line_start != null) {
    return `Matches lines ${orig.line_start}–${orig.line_end ?? orig.line_start}${suffix}`;
  }
  return `Matches content${suffix}`;
}

function escaped(text) {
  return String(text || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

function highlighted(text, query) {
  const words = [...new Set(String(query || "").toLowerCase().split(/[^\p{L}\p{N}_]+/u).filter((w) => w.length > 1))];
  let html = escaped(text);
  if (!words.length) return html;
  const pattern = new RegExp(`(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "giu");
  return html.replace(pattern, "<mark>$1</mark>");
}

function getPreviewDataUrl(path, kind, loc, pixels) {
  const key = `${path}|${kind}|${loc ?? 0}|${pixels}`;
  if (!previewCache.has(key)) {
    previewCache.set(
      key,
      invoke("render_preview", { path, kind, loc: loc ?? null, pixels }).catch(() => null)
    );
  }
  return previewCache.get(key);
}

// Render 1-Click Scope Pills in Header (macOS Finder / Spotlight style)
function renderScopePills() {
  if (!$scopePills) return;
  $scopePills.replaceChildren();
  const shortLabels = {
    "🌐 Whole Operating System (All User Folders)": "🌐 All Computer",
    "Documents": "📄 Documents",
    "Downloads": "⬇️ Downloads",
    "Pictures": "🖼️ Pictures",
    "Videos": "🎬 Videos",
    "Music & Audio": "🎵 Audio",
    "Desktop": "💻 Desktop",
    "Live Demo Corpus": "🧪 Demo Corpus",
  };

  for (const preset of state.presets) {
    if (preset.label === "Current Workspace") continue;
    const pill = document.createElement("button");
    pill.className = "scope-pill" + (preset.path === state.activeDir ? " active" : "");
    pill.textContent = shortLabels[preset.label] || preset.label;
    pill.title = `${preset.label} (${preset.indexed_files} indexed files)`;
    pill.onclick = () => switchActiveFolder(preset.path);
    $scopePills.appendChild(pill);
  }
}

// Render KindFilter chips
function renderKindChips() {
  $kindChips.replaceChildren();
  const totalCount = Object.values(state.kindCounts).reduce((a, b) => a + b, 0);

  for (const item of KIND_FILTERS) {
    const count = item.key === null ? totalCount : (state.kindCounts[item.key] || 0);
    if (item.key !== null && count === 0 && state.kindFilter !== item.key) continue;

    const btn = document.createElement("button");
    btn.className = "kind-chip" + (state.kindFilter === item.key ? " selected" : "");
    btn.innerHTML = `<span>${item.label}</span><span class="chip-count">${count}</span>`;
    btn.onclick = () => {
      state.kindFilter = state.kindFilter === item.key ? null : item.key;
      renderKindChips();
      triggerSearchNow();
    };
    $kindChips.appendChild(btn);
  }
}

function renderDemoProbes() {
  $demoProbesBar.replaceChildren();
  if (!state.demoProbes.length) return;
  const label = document.createElement("span");
  label.style.fontSize = "11px";
  label.style.color = "var(--text-secondary)";
  label.textContent = "Sample Clips:";
  $demoProbesBar.appendChild(label);

  for (const probe of state.demoProbes) {
    const chip = document.createElement("button");
    chip.className = "probe-chip";
    chip.title = probe.description;
    chip.textContent = `${probe.label} (${probe.kind})`;
    chip.onclick = () => {
      document.getElementById("containFileInput").value = probe.path;
      runFileContainment(probe.path);
    };
    $demoProbesBar.appendChild(chip);
  }
}

function renderFolderPresets() {
  $folderPresetsGrid.replaceChildren();
  for (const preset of state.presets) {
    const card = document.createElement("button");
    card.className = "preset-card" + (preset.path === state.activeDir ? " active" : "");
    card.innerHTML = `
      <div class="p-label">${escaped(preset.label)}</div>
      <div class="p-path">${escaped(prettyPath(preset.path))}</div>
      <div class="p-meta">${preset.indexed_files} indexed files (${preset.total_chunks} chunks)</div>
    `;
    card.onclick = async () => {
      await switchActiveFolder(preset.path);
    };
    $folderPresetsGrid.appendChild(card);
  }
}

async function switchActiveFolder(newDir) {
  state.activeDir = newDir;
  $folderInput.value = newDir;
  try {
    state.stats = await invoke("get_status", { dir: newDir });
    state.kindCounts = await invoke("get_kind_counts", { dir: newDir });
  } catch (_) {}
  renderScopePills();
  renderKindChips();
  renderFolderPresets();
  updateFooterStatus();
  await triggerSearchNow();
}

function updateFooterStatus() {
  if (!state.stats) {
    $footerFolderText.textContent = state.activeDir === "ALL_OS" ? "🌐 All Computer" : prettyPath(state.activeDir);
    return;
  }
  const scopeLabel =
    state.activeDir === "ALL_OS"
      ? "All Computer (Documents, Downloads, Pictures, Videos, Music, Desktop)"
      : prettyPath(state.activeDir);
  if (state.pendingEmbeddings > 0) {
    $statusDot.classList.add("busy");
    $footerFolderText.textContent = `Watching ${state.stats.indexed_files.toLocaleString()} files · indexing 768-D vectors in background · ${scopeLabel}`;
  } else {
    $statusDot.classList.remove("busy");
    $footerFolderText.textContent = `Live · ${state.stats.indexed_files.toLocaleString()} files indexed · ${scopeLabel}`;
  }
}

function scheduleBackgroundEmbedStep(delayMs = 2000) {
  clearTimeout(state.bgTimer);
  state.bgTimer = setTimeout(async () => {
    if (state.timer || state.bgRunning) {
      scheduleBackgroundEmbedStep(2200);
      return;
    }
    state.bgRunning = true;
    try {
      const prevFiles = state.stats?.indexed_files || 0;
      const prog = await invoke("embed_os_step", {
        dir: state.activeDir,
        batchSize: state.pendingEmbeddings > 0 ? 3 : 0,
      });
      state.pendingEmbeddings = prog.remaining_pending || 0;
      if (prog.stats) {
        state.stats = prog.stats;
      }
      updateFooterStatus();

      // If new OS files were created/modified on disk while in browse mode, refresh live!
      if (prog.stats && prog.stats.indexed_files !== prevFiles && state.phase === "browse") {
        await browseCurrentFolder(state.generation);
      }
    } catch (_) {
    } finally {
      state.bgRunning = false;
      scheduleBackgroundEmbedStep(state.pendingEmbeddings > 0 ? 1500 : 4000);
    }
  }, delayMs);
}

// Instant lexical + 110ms hybrid search pipeline
function onQueryChanged() {
  const q = $queryInput.value;
  state.query = q;
  $clearQueryBtn.classList.toggle("hidden", q.length === 0);
  if ($shortcutHint) {
    $shortcutHint.classList.toggle("hidden", q.length > 0);
  }

  const isCode = q.trim().toLowerCase().startsWith("code:");
  $searchIcon.innerHTML = isCode ? ICONS.code : ICONS.search;

  const gen = ++state.generation;
  clearTimeout(state.timer);

  if (!q.trim()) {
    $statusBanner.classList.add("hidden");
    browseCurrentFolder(gen);
    return;
  }

  // Step 1: Immediate FTS5 keyword search (<5ms)
  if (state.mode !== "semantic") {
    invoke("search_folder", {
      dir: state.activeDir,
      query: q,
      mode: "lexical",
      kind: state.kindFilter,
      limit: 36,
    })
      .then((resp) => {
        if (gen === state.generation && state.finalGeneration !== gen) {
          applySearchResponse(resp, "words");
        }
      })
      .catch(() => {});
  }

  // Step 2: After 110ms pause, full EmbeddingGemma 2 768-D Hybrid/Semantic search
  state.timer = setTimeout(() => {
    invoke("search_folder", {
      dir: state.activeDir,
      query: q,
      mode: state.mode,
      kind: state.kindFilter,
      limit: 36,
    })
      .then((resp) => {
        if (gen !== state.generation) return;
        state.finalGeneration = gen;
        applySearchResponse(resp, "meaning");
      })
      .catch((err) => {
        $footerSummary.textContent = `Search error: ${err}`;
      });
  }, TYPING_PAUSE_MS);
}

async function triggerSearchNow() {
  clearTimeout(state.timer);
  state.query = $queryInput.value;
  $clearQueryBtn.classList.toggle("hidden", state.query.length === 0);
  if ($shortcutHint) {
    $shortcutHint.classList.toggle("hidden", state.query.length > 0);
  }
  const gen = ++state.generation;
  if (!state.query.trim()) {
    await browseCurrentFolder(gen);
  } else {
    const resp = await invoke("search_folder", {
      dir: state.activeDir,
      query: state.query,
      mode: state.mode,
      kind: state.kindFilter,
      limit: 36,
    });
    state.finalGeneration = gen;
    applySearchResponse(resp, "meaning");
  }
}

async function browseCurrentFolder(gen) {
  try {
    const resp = await invoke("browse_folder", {
      dir: state.activeDir,
      kind: state.kindFilter,
      limit: 60,
    });
    if (gen !== state.generation) return;
    applySearchResponse(resp, "browse");
  } catch (err) {
    $footerSummary.textContent = `Status: ${err}`;
  }
}

function applySearchResponse(resp, phase) {
  state.resultMode = "search";
  state.phase = phase;
  state.elapsedMs = resp.elapsed_ms || 0;

  if (resp.decoded_audio_transcript) {
    $statusBanner.className = "status-banner";
    const kws = (resp.distilled_audio_keywords || [])
      .slice(0, 6)
      .map((k) => `${escaped(k.term)} (z=${k.z_score.toFixed(2)})`)
      .join(" · ");
    $statusBanner.innerHTML = `
      <div><strong>🎙️ 768-D Semantic Audio Decode:</strong> “${escaped(resp.decoded_audio_transcript)}”</div>
      <div style="color: var(--text-secondary); font-size: 11.5px;">Distilled Corpus Keywords: ${kws || "direct 768-D vector match"}</div>
    `;
  } else if (phase !== "words") {
    $statusBanner.classList.add("hidden");
  }

  const hits = resp.hits || [];
  const strongHits = [];
  const weakerHits = [];
  for (let i = 0; i < hits.length; i++) {
    const h = hits[i];
    if (phase === "browse" || i === 0 || h.score >= 1.0 || h.semantic_z >= 0.8 || h.lexical_coverage >= 0.4) {
      strongHits.push(h);
    } else {
      weakerHits.push(h);
    }
  }

  state.strong = strongHits;
  state.weaker = weakerHits;
  buildGroups(strongHits);

  const ids = allSelectableKeys();
  if (!ids.includes(state.selectedKey)) {
    state.selectedKey = ids[0] || null;
  }

  renderWorkspace();
}

function applyContainmentResponse(resp) {
  state.resultMode = "containment";
  state.phase = "containment";
  state.elapsedMs = resp.elapsed_ms || 0;

  $statusBanner.className = "status-banner " + (resp.is_contained ? "contained" : "not-contained");
  $statusBanner.innerHTML = `
    <div>
      <strong>${resp.is_contained ? "✅ CONTAINED IN INDEX" : "⚠️ NOT CONTAINED IN INDEX"}</strong>
      — Probe: <b>${escaped(resp.query_description)}</b> (${escaped(resp.query_modality)}) · Best Confidence: <b>${(resp.best_confidence * 100).toFixed(1)}%</b> (${resp.elapsed_ms.toFixed(1)} ms)
    </div>
  `;

  const hits = (resp.hits || []).map((h, idx) => ({
    ...h,
    chunk_id: idx,
  }));

  const strongHits = hits.filter((h) => h.confidence >= 0.65);
  const weakerHits = hits.filter((h) => h.confidence < 0.65);
  state.strong = strongHits.length ? strongHits : hits.slice(0, 3);
  state.weaker = strongHits.length ? weakerHits : hits.slice(3);

  buildGroups(state.strong);
  const ids = allSelectableKeys();
  state.selectedKey = ids[0] || null;
  renderWorkspace();
}

function buildGroups(hits) {
  const map = new Map();
  for (const hit of hits) {
    const title = GROUP_TITLES[hit.kind] || "Files";
    if (!map.has(title)) map.set(title, []);
    map.get(title).push(hit);
  }
  state.groups = [...map].map(([title, rows]) => ({ title, rows }));
}

function allSelectableKeys() {
  return [
    ...state.groups.flatMap((g) => g.rows.map(hitKey)),
    ...(state.weaker.length ? [WEAKER_ID] : []),
    ...(state.showWeaker ? state.weaker.map(hitKey) : []),
  ];
}

function selectedHit() {
  return [...state.strong, ...state.weaker].find((h) => hitKey(h) === state.selectedKey) || null;
}

function renderWorkspace() {
  $workspaceBody.className =
    "workspace-body" +
    (state.showPreview ? " with-preview" : "") +
    (state.viewMode === "grid" ? " grid-mode" : " list-mode");
  $previewPane.classList.toggle("hidden", !state.showPreview);

  renderResultsList();
  if (state.showPreview) {
    renderPreviewPane();
  }
  renderFooterSummary();
}

function renderResultsList() {
  $resultsList.replaceChildren();
  $resultsList.className =
    "results-list " + (state.viewMode === "grid" ? "grid-layout" : "list-layout");

  if (!state.strong.length && !state.weaker.length) {
    const empty = document.createElement("div");
    empty.className = "empty-message";
    empty.innerHTML = `
      <strong>No matching files found</strong>
      <span>Try describing what you'd see or hear in it, words you remember from it, or switch scopes above.</span>
    `;
    $resultsList.appendChild(empty);
    return;
  }

  if (state.viewMode === "grid") {
    const allRows = [...state.strong, ...(state.showWeaker ? state.weaker : [])];
    for (const hit of allRows) {
      $resultsList.appendChild(createGridCell(hit));
    }
    return;
  }

  for (const group of state.groups) {
    const header = document.createElement("div");
    header.className = "group-title";
    header.textContent = group.title;
    $resultsList.appendChild(header);

    for (const hit of group.rows) {
      $resultsList.appendChild(createListRow(hit));
    }
  }

  if (state.weaker.length) {
    const fold = document.createElement("div");
    fold.className =
      "weaker-row" +
      (state.showWeaker ? " open" : "") +
      (state.selectedKey === WEAKER_ID ? " selected" : "");
    const count = state.weaker.length;
    fold.innerHTML = `${ICONS.chevron}<span>${
      state.showWeaker ? "Weaker matches" : `${count} weaker ${count === 1 ? "match" : "matches"}`
    }</span>`;
    fold.onclick = () => {
      state.showWeaker = !state.showWeaker;
      state.selectedKey = WEAKER_ID;
      renderWorkspace();
    };
    $resultsList.appendChild(fold);

    if (state.showWeaker) {
      for (const hit of state.weaker) {
        $resultsList.appendChild(createListRow(hit));
      }
    }
  }

  $resultsList.querySelector(".selected")?.scrollIntoView({ block: "nearest" });
}

function createThumbnailNode(hit, sidePx) {
  const orig = hit.origin || {};
  const canRender = hit.kind === "image" || hit.kind === "video";
  if (!canRender) {
    const iconWrap = document.createElement("div");
    iconWrap.className = "thumb icon";
    iconWrap.innerHTML = ICONS[hit.kind] || ICONS.doc;
    return iconWrap;
  }

  const img = document.createElement("img");
  img.className = "thumb";
  img.alt = "";
  const loc = hit.kind === "video" ? (orig.start_seconds ?? 0.5) : null;
  getPreviewDataUrl(hit.path, hit.kind, loc, sidePx).then((dataUrl) => {
    if (dataUrl) {
      img.src = dataUrl;
    } else {
      const fallback = document.createElement("div");
      fallback.className = "thumb icon";
      fallback.innerHTML = ICONS[hit.kind] || ICONS.doc;
      img.replaceWith(fallback);
    }
  });
  return img;
}

function createListRow(hit) {
  const key = hitKey(hit);
  const row = document.createElement("div");
  row.className = "row" + (state.selectedKey === key ? " selected" : "");
  row.setAttribute("role", "option");

  row.appendChild(createThumbnailNode(hit, 96));

  const textCol = document.createElement("div");
  textCol.className = "row-text";

  const titleRow = document.createElement("div");
  titleRow.className = "row-title";

  const nameSpan = document.createElement("span");
  nameSpan.className = "name";
  nameSpan.textContent = hit.file_name || hit.relative_path;
  titleRow.appendChild(nameSpan);

  const place = placeOf(hit);
  if (place) {
    const badge = document.createElement("span");
    const isExact = state.resultMode === "containment" && (hit.confidence || 0) >= 0.88;
    badge.className = "place-badge" + (isExact ? " exact" : "");
    badge.textContent = place;
    titleRow.appendChild(badge);
  }

  if (state.resultMode === "search" && state.phase !== "browse" && hit.raw_cosine && hit.raw_cosine >= 0.45) {
    const semBadge = document.createElement("span");
    semBadge.className = "place-badge semantic";
    semBadge.textContent = `${Math.round(hit.raw_cosine * 100)}%`;
    semBadge.title = `768-D EmbeddingGemma 2 similarity (z=${(hit.semantic_z || 0).toFixed(2)})`;
    titleRow.appendChild(semBadge);
  }

  const relTime = relativeTime(hit.modified_unix);
  if (relTime) {
    const timeSpan = document.createElement("span");
    timeSpan.className = "time-badge";
    timeSpan.textContent = relTime;
    titleRow.appendChild(timeSpan);
  }

  const subRow = document.createElement("div");
  subRow.className = "row-sub";
  const folder = parentFolderDisplay(hit.path);
  if (hit.snippet && !hit.snippet.startsWith(hit.file_name)) {
    subRow.innerHTML = `<span class="folder-hint">${escaped(folder)} · </span>${highlighted(hit.snippet, state.query)}`;
  } else {
    const sz = formatBytes(hit.size_bytes);
    subRow.innerHTML = `<span class="folder-hint">${escaped(folder)}${sz ? ` · ${escaped(sz)}` : ""}</span>`;
  }

  textCol.appendChild(titleRow);
  textCol.appendChild(subRow);
  row.appendChild(textCol);

  row.onclick = () => {
    state.selectedKey = key;
    renderWorkspace();
  };
  row.ondblclick = () => openHit(hit);
  return row;
}

function createGridCell(hit) {
  const key = hitKey(hit);
  const cell = document.createElement("div");
  cell.className = "grid-cell" + (state.selectedKey === key ? " selected" : "");

  const frame = document.createElement("div");
  frame.className = "grid-card-frame";

  const orig = hit.origin || {};
  if (hit.kind === "image" || hit.kind === "video") {
    const img = document.createElement("img");
    const loc = hit.kind === "video" ? (orig.start_seconds ?? 0.5) : null;
    getPreviewDataUrl(hit.path, hit.kind, loc, 360).then((url) => {
      if (url) img.src = url;
    });
    frame.appendChild(img);
  } else {
    const icon = document.createElement("div");
    icon.className = "grid-icon";
    icon.innerHTML = ICONS[hit.kind] || ICONS.doc;
    frame.appendChild(icon);
  }

  const kindPill = document.createElement("div");
  kindPill.className = "grid-kind-pill";
  kindPill.innerHTML = ICONS[hit.kind] || ICONS.doc;
  frame.appendChild(kindPill);

  const place = placeOf(hit) || relativeTime(hit.modified_unix);
  if (place) {
    const pBadge = document.createElement("div");
    pBadge.className = "grid-place-pill";
    pBadge.textContent = place;
    frame.appendChild(pBadge);
  }

  const meta = document.createElement("div");
  meta.className = "grid-meta";
  meta.innerHTML = `
    <div class="title">${escaped(hit.file_name || hit.relative_path)}</div>
    <div class="folder">${escaped(parentFolderDisplay(hit.path))}</div>
  `;

  cell.appendChild(frame);
  cell.appendChild(meta);

  cell.onclick = () => {
    state.selectedKey = key;
    renderWorkspace();
  };
  cell.ondblclick = () => openHit(hit);
  return cell;
}

function renderPreviewPane() {
  $previewContent.replaceChildren();
  const hit = selectedHit();
  if (!hit) {
    const empty = document.createElement("div");
    empty.className = "empty-message";
    empty.innerHTML = `
      <strong>Select a file to inspect</strong>
      <span>Preview images, play matched video &amp; audio moments, inspect highlighted passages, or verify containment.</span>
    `;
    $previewContent.appendChild(empty);
    return;
  }

  const pictureBox = document.createElement("div");
  pictureBox.className = "picture-box";
  const orig = hit.origin || {};

  if (hit.kind === "image") {
    const img = document.createElement("img");
    getPreviewDataUrl(hit.path, "image", null, 960).then((url) => {
      if (url) img.src = url;
    });
    pictureBox.appendChild(img);
  } else if (hit.kind === "video") {
    const at = orig.start_seconds ?? 0;
    const overlay = document.createElement("div");
    overlay.className = "video-moment-overlay";
    const img = document.createElement("img");
    getPreviewDataUrl(hit.path, "video", at > 0 ? at : 0.5, 960).then((url) => {
      if (url) img.src = url;
    });
    const playCircle = document.createElement("div");
    playCircle.className = "play-circle";
    playCircle.innerHTML = ICONS.play;
    const badge = document.createElement("span");
    badge.className = "moment-at-badge";
    badge.textContent = `▶ Play from ${clock(at)}`;
    overlay.append(img, playCircle, badge);

    overlay.onclick = async () => {
      badge.textContent = "Loading video…";
      try {
        const dataUrl = await invoke("load_media_data_url", { path: hit.path });
        const video = document.createElement("video");
        video.src = `${dataUrl}#t=${at}`;
        video.controls = true;
        video.autoplay = true;
        pictureBox.replaceChildren(video);
      } catch (err) {
        alert(`Could not load inline video: ${err}`);
      }
    };
    pictureBox.appendChild(overlay);
  } else if (hit.kind === "audio") {
    const at = orig.start_seconds ?? 0;
    const card = document.createElement("div");
    card.className = "audio-preview-card";
    const wave = document.createElement("div");
    wave.className = "audio-wave-icon";
    wave.innerHTML = ICONS.audio;
    const audioEl = document.createElement("audio");
    audioEl.controls = true;
    invoke("load_media_data_url", { path: hit.path })
      .then((dataUrl) => {
        audioEl.src = `${dataUrl}#t=${at}`;
      })
      .catch(() => {});

    const decodeBtn = document.createElement("button");
    decodeBtn.className = "btn accent";
    decodeBtn.textContent = "🎙️ Inspect 768-D Semantic-Space ASR Timeline";
    decodeBtn.onclick = () => inspectAudioAsr(hit.path);

    card.append(wave, audioEl, decodeBtn);
    pictureBox.appendChild(card);
  } else {
    const passage = document.createElement("div");
    passage.className = "passage" + (hit.kind === "code" ? " code" : "");
    passage.innerHTML = highlighted(hit.snippet || "", state.query);
    invoke("read_file_passage", {
      path: hit.path,
      startLine: orig.line_start ?? null,
      endLine: orig.line_end ?? null,
    })
      .then((fullText) => {
        if (fullText) {
          passage.innerHTML = highlighted(fullText, state.query);
        }
      })
      .catch(() => {});
    pictureBox.appendChild(passage);
  }

  const facts = document.createElement("div");
  facts.className = "facts";

  const titleEl = document.createElement("div");
  titleEl.className = "title";
  titleEl.textContent = hit.file_name || hit.relative_path;

  const pathEl = document.createElement("div");
  pathEl.className = "line";
  const sz = formatBytes(hit.size_bytes);
  const rel = relativeTime(hit.modified_unix);
  pathEl.textContent = [prettyPath(hit.path), sz, rel ? `Modified ${rel}` : ""].filter(Boolean).join(" · ");

  const reasonEl = document.createElement("div");
  reasonEl.className = "reason";
  reasonEl.innerHTML = `${ICONS.found}<span>${escaped(reasonOf(hit))}</span>`;

  facts.append(titleEl, pathEl, reasonEl);

  if (hit.snippet && !hit.snippet.startsWith(hit.file_name)) {
    const excerptBox = document.createElement("div");
    excerptBox.className = "reason-text";
    excerptBox.innerHTML = highlighted(hit.snippet, state.query);
    facts.appendChild(excerptBox);
  }

  const btnRow = document.createElement("div");
  btnRow.className = "preview-buttons";

  const openBtn = document.createElement("button");
  openBtn.className = "btn primary";
  openBtn.textContent = "Open File";
  openBtn.onclick = () => openHit(hit);

  const revealBtn = document.createElement("button");
  revealBtn.className = "btn secondary";
  revealBtn.textContent = "Show in Files";
  revealBtn.onclick = () => revealHit(hit);

  const containThisBtn = document.createElement("button");
  containThisBtn.className = "btn accent";
  containThisBtn.textContent = "Verify Part-of-File";
  containThisBtn.onclick = () => runFileContainment(hit.path);

  btnRow.append(openBtn, revealBtn, containThisBtn);
  facts.appendChild(btnRow);

  $previewContent.append(pictureBox, facts);
}

function renderFooterSummary() {
  const strongCount = state.strong.length;
  const weakerCount = state.weaker.length;
  if (state.phase === "words") {
    $footerSummary.textContent = `Matching words… (${state.elapsedMs.toFixed(1)} ms)`;
  } else if (state.phase === "browse") {
    $footerSummary.textContent = `Recent ongoing activity · ${strongCount} files`;
  } else if (state.phase === "containment") {
    $footerSummary.textContent = `Part-of-File Verification · ${strongCount} strong · ${weakerCount} weaker (${state.elapsedMs.toFixed(1)} ms)`;
  } else {
    $footerSummary.textContent = `${strongCount} ${strongCount === 1 ? "result" : "results"}${
      weakerCount ? ` · ${weakerCount} weaker` : ""
    } (${state.elapsedMs.toFixed(1)} ms)`;
  }
}

function openHit(hit) {
  if (!hit) return;
  invoke("open_path", { path: hit.path }).catch((err) => alert(err));
}

function revealHit(hit) {
  if (!hit) return;
  invoke("reveal_path", { path: hit.path }).catch((err) => alert(err));
}

function moveSelection(delta) {
  const keys = allSelectableKeys();
  if (!keys.length) return;
  const idx = keys.indexOf(state.selectedKey);
  const nextIdx = Math.max(0, Math.min(keys.length - 1, (idx < 0 ? 0 : idx) + delta));
  state.selectedKey = keys[nextIdx];
  renderWorkspace();
}

// Voice / ASR & Part-of-File Containment actions
async function runFileContainment(filePath) {
  if (!filePath || !filePath.trim()) return;
  $footerSummary.textContent = `Verifying containment for ${filePath}…`;
  try {
    const resp = await invoke("verify_containment_file", {
      dir: state.activeDir,
      filePath: filePath.trim(),
      limit: 16,
    });
    applyContainmentResponse(resp);
  } catch (err) {
    alert(`Containment error: ${err}`);
  }
}

async function runTextContainment(text) {
  if (!text || !text.trim()) return;
  $footerSummary.textContent = "Verifying text passage containment…";
  try {
    const resp = await invoke("verify_containment_text", {
      dir: state.activeDir,
      text: text.trim(),
      limit: 16,
    });
    applyContainmentResponse(resp);
  } catch (err) {
    alert(`Text containment error: ${err}`);
  }
}

async function inspectAudioAsr(audioPath) {
  if (!audioPath || !audioPath.trim()) return;
  $footerSummary.textContent = "Running EmbeddingGemma 2 768-D Semantic-Space ASR…";
  try {
    const res = await invoke("decode_asr_file", {
      audioPath: audioPath.trim(),
      dir: state.activeDir,
    });
    $statusBanner.className = "status-banner";
    const kws = (res.top_keywords || [])
      .slice(0, 8)
      .map((k) => `${escaped(k.term)} (z=${k.z_score.toFixed(2)}, cos=${k.cosine.toFixed(3)})`)
      .join(" · ");
    const winPills = (res.windows || [])
      .map((w) => {
        const topTerms = (w.keywords || [])
          .slice(0, 3)
          .map((k) => `${escaped(k.term)}(${k.z_score.toFixed(1)})`)
          .join(", ");
        return `<span class="asr-win-pill">[${w.start_seconds.toFixed(1)}s–${w.end_seconds.toFixed(1)}s] ${topTerms}</span>`;
      })
      .join("");
    $statusBanner.innerHTML = `
      <div><strong>🎙️ EmbeddingGemma 2 Semantic-Space ASR (${res.duration_seconds.toFixed(2)}s):</strong> “${escaped(res.semantic_transcript)}”</div>
      <div style="font-size: 11.5px; color: var(--text-secondary);">Top 768-D Keywords: ${kws}</div>
      <div class="asr-timeline">${winPills}</div>
    `;
    $footerSummary.textContent = `Decoded ${res.windows?.length || 0} acoustic sub-windows`;
  } catch (err) {
    alert(`Semantic ASR error: ${err}`);
  }
}

// Event Listeners
$queryInput.addEventListener("input", onQueryChanged);

$clearQueryBtn.addEventListener("click", () => {
  $queryInput.value = "";
  $queryInput.focus();
  onQueryChanged();
});

$modePills.querySelectorAll(".mode-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    $modePills.querySelectorAll(".mode-pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    state.mode = pill.dataset.mode;
    triggerSearchNow();
  });
});

$voiceDrawerBtn.addEventListener("click", () => {
  const hidden = $voiceDrawer.classList.toggle("hidden");
  $voiceDrawerBtn.classList.toggle("active", !hidden);
  if (!hidden) {
    $containDrawer.classList.add("hidden");
    $containDrawerBtn.classList.remove("active");
  }
});

document.getElementById("closeVoiceDrawerBtn").addEventListener("click", () => {
  $voiceDrawer.classList.add("hidden");
  $voiceDrawerBtn.classList.remove("active");
});

$containDrawerBtn.addEventListener("click", () => {
  const hidden = $containDrawer.classList.toggle("hidden");
  $containDrawerBtn.classList.toggle("active", !hidden);
  if (!hidden) {
    $voiceDrawer.classList.add("hidden");
    $voiceDrawerBtn.classList.remove("active");
  }
});

document.getElementById("closeContainDrawerBtn").addEventListener("click", () => {
  $containDrawer.classList.add("hidden");
  $containDrawerBtn.classList.remove("active");
});

$viewModeBtn.addEventListener("click", () => {
  state.viewMode = state.viewMode === "list" ? "grid" : "list";
  document.getElementById("viewModeIconSlot").innerHTML =
    state.viewMode === "list" ? ICONS.grid : ICONS.list;
  renderWorkspace();
});

$togglePreviewBtn.addEventListener("click", () => {
  state.showPreview = !state.showPreview;
  $togglePreviewBtn.classList.toggle("active", state.showPreview);
  renderWorkspace();
});

$foldersModalBtn.addEventListener("click", () => {
  $folderInput.value = state.activeDir;
  $foldersModal.classList.remove("hidden");
});

$footerFolderBtn.addEventListener("click", () => {
  $folderInput.value = state.activeDir;
  $foldersModal.classList.remove("hidden");
});

document.getElementById("closeModalBtn").addEventListener("click", () => {
  $foldersModal.classList.add("hidden");
});

document.getElementById("estimateFolderBtn").addEventListener("click", async () => {
  const dir = $folderInput.value.trim();
  if (!dir) return;
  try {
    const est = await invoke("estimate_folder", { dir });
    const kindsHtml = Object.entries(est.by_kind || {})
      .map(([k, v]) => `<span class="place-badge">${escaped(k)}: ${v}</span>`)
      .join(" ");
    $estimateBox.classList.remove("hidden");
    $estimateBox.innerHTML = `
      <div><strong>Folder Cost Estimate:</strong> ${escaped(est.root)}</div>
      <div>Files: <b>${est.total_files}</b> · Size: <b>${(est.total_bytes / (1024 * 1024)).toFixed(2)} MB</b> · Est. Chunks: <b>${est.estimated_chunks}</b> · Est. Time: <b>${est.estimated_seconds.toFixed(1)}s</b></div>
      <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:4px;">${kindsHtml}</div>
    `;
  } catch (err) {
    alert(`Estimate failed: ${err}`);
  }
});

async function runFolderIndexing(force) {
  const dir = $folderInput.value.trim();
  if (!dir) return;
  const btn = document.getElementById(force ? "forceIndexBtn" : "indexFolderBtn");
  const prevText = btn.textContent;
  btn.textContent = "Indexing…";
  $statusDot.classList.add("busy");
  try {
    const stats = await invoke("index_folder", { dir, force });
    await switchActiveFolder(dir);
    $estimateBox.classList.remove("hidden");
    $estimateBox.innerHTML = `
      <div><strong>✅ Indexing Complete:</strong> Indexed <b>${stats.indexed_files}</b> files (<b>${stats.total_chunks}</b> chunks: ${stats.text_chunks} text, ${stats.image_chunks} image, ${stats.audio_chunks} audio) in <b>${Math.round(stats.elapsed_ms)} ms</b></div>
    `;
  } catch (err) {
    alert(`Indexing error: ${err}`);
  } finally {
    btn.textContent = prevText;
    $statusDot.classList.remove("busy");
  }
}

document.getElementById("indexFolderBtn").addEventListener("click", () => runFolderIndexing(false));
document.getElementById("forceIndexBtn").addEventListener("click", () => runFolderIndexing(true));

// Voice Drawer controls
document.getElementById("recordMicBtn").addEventListener("click", async () => {
  const btn = document.getElementById("recordMicBtn");
  btn.textContent = "🎙️ Listening (4s)…";
  try {
    const resp = await invoke("record_mic_search", {
      dir: state.activeDir,
      seconds: 4.0,
      hint: state.query || null,
      limit: 24,
    });
    applySearchResponse(resp, "meaning");
  } catch (err) {
    alert(`Microphone recording error: ${err}`);
  } finally {
    btn.textContent = "🔴 Record 4s Voice Query";
  }
});

document.getElementById("tryDemoAudioBtn").addEventListener("click", async () => {
  const demoAudio = "/tmp/prism_live_demo/probes/probe_audio_excerpt.wav";
  document.getElementById("audioPathInput").value = demoAudio;
  try {
    const resp = await invoke("search_audio_path", {
      dir: state.activeDir,
      audioPath: demoAudio,
      hint: null,
      limit: 24,
    });
    applySearchResponse(resp, "meaning");
    await inspectAudioAsr(demoAudio);
  } catch (err) {
    alert(`Audio search error: ${err}`);
  }
});

document.getElementById("runAudioSearchBtn").addEventListener("click", async () => {
  const path = document.getElementById("audioPathInput").value.trim();
  if (!path) return;
  try {
    const resp = await invoke("search_audio_path", {
      dir: state.activeDir,
      audioPath: path,
      hint: state.query || null,
      limit: 24,
    });
    applySearchResponse(resp, "meaning");
  } catch (err) {
    alert(`Audio search error: ${err}`);
  }
});

document.getElementById("runAsrOnlyBtn").addEventListener("click", () => {
  const path =
    document.getElementById("audioPathInput").value.trim() ||
    "/tmp/prism_live_demo/probes/probe_audio_excerpt.wav";
  inspectAudioAsr(path);
});

// Containment Drawer controls
document.getElementById("runContainFileBtn").addEventListener("click", () => {
  runFileContainment(document.getElementById("containFileInput").value);
});

document.getElementById("runMultimodalSearchBtn").addEventListener("click", async () => {
  const filePath = document.getElementById("containFileInput").value.trim();
  if (!filePath) return;
  try {
    const resp = await invoke("search_multimodal_file", {
      dir: state.activeDir,
      filePath,
      limit: 24,
    });
    applySearchResponse(resp, "meaning");
  } catch (err) {
    alert(`Multimodal file search error: ${err}`);
  }
});

document.getElementById("runContainTextBtn").addEventListener("click", () => {
  runTextContainment(document.getElementById("containTextInput").value);
});

// Global Keyboard Navigation (macOS Spotlight / DigUp style)
document.addEventListener("keydown", (event) => {
  if (event.isComposing) return;
  const mod = event.metaKey || event.ctrlKey;

  if (mod && event.key.toLowerCase() === "k") {
    event.preventDefault();
    $queryInput.focus();
    $queryInput.select();
    return;
  }

  if (event.key === "/" && document.activeElement !== $queryInput && document.activeElement?.tagName !== "INPUT") {
    event.preventDefault();
    $queryInput.focus();
    return;
  }

  if (event.key === "Escape") {
    if (!$foldersModal.classList.contains("hidden")) {
      $foldersModal.classList.add("hidden");
      return;
    }
    if (!$voiceDrawer.classList.contains("hidden") || !$containDrawer.classList.contains("hidden")) {
      $voiceDrawer.classList.add("hidden");
      $containDrawer.classList.add("hidden");
      $voiceDrawerBtn.classList.remove("active");
      $containDrawerBtn.classList.remove("active");
      return;
    }
    if ($queryInput.value) {
      $queryInput.value = "";
      onQueryChanged();
    }
    return;
  }

  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    moveSelection(event.key === "ArrowDown" ? 1 : -1);
    return;
  }

  if (event.key === " " && document.activeElement !== $queryInput && document.activeElement?.tagName !== "INPUT") {
    event.preventDefault();
    const overlay = $previewContent.querySelector(".video-moment-overlay");
    const media = $previewContent.querySelector("video, audio");
    if (overlay) {
      overlay.click();
    } else if (media) {
      if (media.paused) media.play();
      else media.pause();
    } else {
      $togglePreviewBtn.click();
    }
    return;
  }

  if (event.key === "Enter" && (document.activeElement === $queryInput || document.activeElement?.tagName !== "INPUT")) {
    event.preventDefault();
    const hit = selectedHit();
    if (mod && hit) revealHit(hit);
    else if (state.selectedKey === WEAKER_ID) {
      state.showWeaker = !state.showWeaker;
      renderWorkspace();
    } else if (hit) {
      openHit(hit);
    }
  }
});

// Initialize on load
async function initApp() {
  if (!invoke) return;
  try {
    const init = await invoke("get_initial_state");
    state.activeDir = init.active_dir;
    state.stats = init.stats;
    state.pendingEmbeddings = init.pending_embeddings || 0;
    state.presets = init.presets || [];
    state.demoProbes = init.demo_probes || [];
    state.kindCounts = init.kind_counts || {};

    $folderInput.value = state.activeDir;
    renderScopePills();
    renderKindChips();
    renderDemoProbes();
    renderFolderPresets();
    updateFooterStatus();
    await browseCurrentFolder(state.generation);
    $queryInput.focus();

    const testMode = await invoke("get_ui_test_mode");
    if (testMode) {
      await runAutomatedUiTest(testMode);
    } else {
      scheduleBackgroundEmbedStep(1500);
    }
  } catch (err) {
    $footerSummary.textContent = `Initialization error: ${err}`;
  }
}

async function runAutomatedUiTest(mode) {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  await sleep(600);

  const snapshotDom = (stepName) => {
    const groups = [...$resultsList.querySelectorAll(".group-title")].map((el) => el.textContent);
    const rows = [...$resultsList.querySelectorAll(".row .name, .grid-cell .title")].map((el) => el.textContent);
    const selectedTitle = $previewContent.querySelector(".facts .title")?.textContent || null;
    const reason = $previewContent.querySelector(".facts .reason span")?.textContent || null;
    const banner = !$statusBanner.classList.contains("hidden") ? $statusBanner.textContent.trim().replace(/\s+/g, " ") : null;
    const footer = $footerSummary.textContent;
    return { stepName, query: $queryInput.value, viewMode: state.viewMode, groups, rows: rows.slice(0, 8), totalRows: rows.length, selectedTitle, reason, banner, footer };
  };

  const steps = [];
  steps.push(snapshotDom("1_initial_ongoing_os_browse"));

  // Step 2: Real OS-Wide File Search (searching user's real OS files in ~/Downloads & ~/Documents)
  $queryInput.value = "PhD Proposal";
  await triggerSearchNow();
  await sleep(400);
  steps.push(snapshotDom("2_os_wide_real_files_search"));

  // Step 3: Non-matching query qualification gate test (must return 0 false-positive hits!)
  $queryInput.value = "zzzznotfound99999xyz";
  await triggerSearchNow();
  await sleep(300);
  steps.push(snapshotDom("3_qualification_gate_zero_false_positives"));

  // Step 4: Hybrid Text Search (Multimodal 768-D + Lexical)
  $queryInput.value = "quantum error correction logical qubits";
  await triggerSearchNow();
  await sleep(400);
  steps.push(snapshotDom("4_hybrid_text_search"));

  // Step 5: Spoken Audio Search + 768-D Semantic-Space ASR Drawer
  $voiceDrawerBtn.click();
  const demoAudio = "/tmp/prism_live_demo/probes/probe_audio_excerpt.wav";
  document.getElementById("audioPathInput").value = demoAudio;
  const audioResp = await invoke("search_audio_path", {
    dir: state.activeDir,
    audioPath: demoAudio,
    hint: null,
    limit: 24,
  });
  applySearchResponse(audioResp, "meaning");
  await inspectAudioAsr(demoAudio);
  await sleep(400);
  steps.push(snapshotDom("5_voice_search_and_semantic_asr"));

  // Step 6: Part-of-File Containment Verification (Video clip inside lecture_demo.mp4)
  $containDrawerBtn.click();
  const demoVideoProbe = "/tmp/prism_live_demo/probes/probe_video_clip.mp4";
  document.getElementById("containFileInput").value = demoVideoProbe;
  await runFileContainment(demoVideoProbe);
  await sleep(500);
  steps.push(snapshotDom("6_part_of_file_video_containment"));

  if (mode === "stay") {
    return;
  }

  await invoke("report_ui_test", {
    report: {
      ok: true,
      active_dir: state.activeDir,
      indexed_files: state.stats?.indexed_files || 0,
      total_chunks: state.stats?.total_chunks || 0,
      kind_counts: state.kindCounts,
      steps,
    },
  });
}

initApp();
