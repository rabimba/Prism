const invoke = window.__TAURI__?.core?.invoke;

const folderInput = document.getElementById("folderInput");
const indexBtn = document.getElementById("indexBtn");
const queryInput = document.getElementById("queryInput");
const modeSelect = document.getElementById("modeSelect");
const searchBtn = document.getElementById("searchBtn");
const voiceBtn = document.getElementById("voiceBtn");
const containBtn = document.getElementById("containBtn");
const containModal = document.getElementById("containModal");
const containFileInput = document.getElementById("containFileInput");
const runContainFileBtn = document.getElementById("runContainFileBtn");
const containTextInput = document.getElementById("containTextInput");
const runContainTextBtn = document.getElementById("runContainTextBtn");
const asrBanner = document.getElementById("asrBanner");
const resultsContainer = document.getElementById("resultsContainer");

containBtn.addEventListener("click", () => {
  containModal.classList.toggle("hidden");
});

indexBtn.addEventListener("click", async () => {
  if (!invoke) return;
  indexBtn.textContent = "Indexing...";
  try {
    const stats = await invoke("index_folder", {
      dir: folderInput.value,
      force: false,
    });
    asrBanner.classList.remove("hidden");
    asrBanner.textContent = `Indexed ${stats.indexed_files} files (${stats.total_chunks} chunks) in ${Math.round(stats.elapsed_ms)} ms`;
  } catch (err) {
    alert(`Index error: ${err}`);
  } finally {
    indexBtn.textContent = "Index Folder";
  }
});

searchBtn.addEventListener("click", async () => {
  if (!invoke) return;
  const resp = await invoke("search_folder", {
    dir: folderInput.value,
    query: queryInput.value,
    mode: modeSelect.value,
    limit: 24,
  });
  renderSearchResults(resp);
});

runContainFileBtn.addEventListener("click", async () => {
  if (!invoke) return;
  const resp = await invoke("verify_containment_file", {
    dir: folderInput.value,
    filePath: containFileInput.value,
    limit: 12,
  });
  renderContainmentResults(resp);
});

runContainTextBtn.addEventListener("click", async () => {
  if (!invoke) return;
  const resp = await invoke("verify_containment_text", {
    dir: folderInput.value,
    text: containTextInput.value,
    limit: 12,
  });
  renderContainmentResults(resp);
});

function renderSearchResults(resp) {
  if (resp.decoded_audio_transcript) {
    asrBanner.classList.remove("hidden");
    const kws = (resp.distilled_audio_keywords || [])
      .slice(0, 5)
      .map((k) => `${k.term} (z=${k.z_score.toFixed(2)})`)
      .join(", ");
    asrBanner.textContent = `768-D Semantic Audio Decode: "${resp.decoded_audio_transcript}" · Keywords: ${kws}`;
  } else {
    asrBanner.classList.add("hidden");
  }

  resultsContainer.innerHTML = "";
  for (const hit of resp.hits) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-header">
        <strong>${hit.relative_path}</strong>
        <span class="badge">${hit.kind}</span>
      </div>
      <div class="metrics">score=${hit.score.toFixed(2)} · z=${hit.semantic_z.toFixed(2)} · cos=${hit.raw_cosine.toFixed(3)} · lex=${hit.lexical_coverage.toFixed(2)}</div>
      <div class="snippet">${hit.snippet}</div>
    `;
    resultsContainer.appendChild(card);
  }
}

function renderContainmentResults(resp) {
  asrBanner.classList.remove("hidden");
  asrBanner.textContent = `Part-of-File Verdict for "${resp.query_description}": ${resp.is_contained ? "CONTAINED" : "NOT FOUND"} (Best confidence: ${(resp.best_confidence * 100).toFixed(1)}%)`;

  resultsContainer.innerHTML = "";
  for (const hit of resp.hits) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-header">
        <strong>${hit.relative_path}</strong>
        <span class="badge exact">${hit.verdict} (${(hit.confidence * 100).toFixed(1)}%)</span>
      </div>
      <div class="metrics">Location: ${hit.location_summary} · sem=${hit.semantic_similarity.toFixed(3)}</div>
      <div class="snippet">${hit.snippet}</div>
    `;
    resultsContainer.appendChild(card);
  }
}
