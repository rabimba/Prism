#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use image::ImageFormat;
use parking_lot::Mutex;
use prism_asr::{
    decode_wav_bytes_16k_mono, record_microphone_16k_mono, SemanticAsrEngine, SemanticAsrResult,
};
use prism_core::{
    estimate_directory, ContainmentEngine, ContainmentResponse, EstimateSummary, IndexStats,
    IndexStore, Indexer, SearchFilter, SearchHit, SearchKind, SearchMode, SearchResponse, Searcher,
    ALL_OS_SCOPE,
};
use prism_llama::{LlamaEmbedder, LlamaModels};
use serde::Serialize;
use tauri::State;

#[derive(Clone)]
struct AppState {
    runtime: Arc<Mutex<Option<(Arc<LlamaEmbedder>, Arc<SemanticAsrEngine>)>>>,
}

impl AppState {
    fn get_or_init(&self) -> Result<(Arc<LlamaEmbedder>, Arc<SemanticAsrEngine>), String> {
        let mut guard = self.runtime.lock();
        if let Some(ref pair) = *guard {
            return Ok(pair.clone());
        }
        let models = LlamaModels::load_default(true).map_err(|e| e.to_string())?;
        let embedder = Arc::new(LlamaEmbedder::new(models, None).map_err(|e| e.to_string())?);
        let asr = Arc::new(SemanticAsrEngine::new(&embedder).map_err(|e| e.to_string())?);
        *guard = Some((embedder.clone(), asr.clone()));
        Ok((embedder, asr))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FolderPreset {
    pub label: String,
    pub path: String,
    pub indexed_files: usize,
    pub total_chunks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DemoProbe {
    pub label: String,
    pub kind: String,
    pub path: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct InitialAppState {
    pub active_dir: String,
    pub stats: IndexStats,
    pub pending_embeddings: usize,
    pub presets: Vec<FolderPreset>,
    pub demo_probes: Vec<DemoProbe>,
    pub kind_counts: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackgroundEmbedProgress {
    pub embedded_now: usize,
    pub remaining_pending: usize,
    pub stats: IndexStats,
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn compute_scoped_stats_and_kinds(dir: &str) -> Result<(IndexStats, HashMap<String, usize>, usize), String> {
    let (store, scope) = IndexStore::open_target(dir).map_err(|e| e.to_string())?;
    let mut stats = store
        .stats_scoped(scope.as_deref(), 0.0)
        .map_err(|e| e.to_string())?;
    if IndexStore::is_global_dir(dir) {
        stats.root = "Whole Operating System (Documents, Downloads, Pictures, Videos, Music, Desktop)".to_string();
    }
    let kind_counts = store
        .kind_counts_scoped(scope.as_deref())
        .map_err(|e| e.to_string())?;
    let (_, pending) = store.pending_unembedded_files(1).unwrap_or((Vec::new(), 0));
    Ok((stats, kind_counts, pending))
}

#[tauri::command]
async fn get_initial_state() -> Result<InitialAppState, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let demo_corpus = PathBuf::from("/tmp/prism_live_demo/corpus");
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        // Check CLI argument if user explicitly passed a folder path to prism-app
        let cli_dir = std::env::args()
            .skip(1)
            .find(|a| !a.starts_with('-') && (a == ALL_OS_SCOPE || Path::new(a).is_dir()))
            .map(|a| a.to_string());

        // Sync OS-wide directories into the Global OS Index in <250ms so the entire OS is searchable immediately
        let os_roots = IndexStore::default_os_roots();
        let _ = Indexer::sync_os_roots_fast(&os_roots);

        let active_dir = cli_dir.unwrap_or_else(|| ALL_OS_SCOPE.to_string());
        let (stats, kind_counts, pending_embeddings) = compute_scoped_stats_and_kinds(&active_dir)?;

        let mut presets = Vec::new();
        if let Ok((global_stats, _, _)) = compute_scoped_stats_and_kinds(ALL_OS_SCOPE) {
            presets.push(FolderPreset {
                label: "🌐 Whole Operating System (All User Folders)".to_string(),
                path: ALL_OS_SCOPE.to_string(),
                indexed_files: global_stats.indexed_files,
                total_chunks: global_stats.total_chunks,
            });
        }

        let mut candidate_dirs: Vec<(String, PathBuf)> = Vec::new();
        if let Some(doc) = dirs::document_dir() {
            if doc.is_dir() {
                candidate_dirs.push(("Documents".to_string(), doc));
            }
        }
        if let Some(dl) = dirs::download_dir() {
            if dl.is_dir() {
                candidate_dirs.push(("Downloads".to_string(), dl));
            }
        }
        if let Some(pic) = dirs::picture_dir() {
            if pic.is_dir() {
                candidate_dirs.push(("Pictures".to_string(), pic));
            }
        }
        if let Some(vid) = dirs::video_dir() {
            if vid.is_dir() {
                candidate_dirs.push(("Videos".to_string(), vid));
            }
        }
        if let Some(aud) = dirs::audio_dir() {
            if aud.is_dir() {
                candidate_dirs.push(("Music & Audio".to_string(), aud));
            }
        }
        if let Some(desk) = dirs::desktop_dir() {
            if desk.is_dir() {
                candidate_dirs.push(("Desktop".to_string(), desk));
            }
        }
        if demo_corpus.is_dir() {
            candidate_dirs.push(("Live Demo Corpus".to_string(), demo_corpus));
        }
        candidate_dirs.push(("Current Workspace".to_string(), cwd));

        let mut seen = std::collections::HashSet::new();
        for (label, path) in candidate_dirs {
            let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
            let p_str = canon.display().to_string();
            if !seen.insert(p_str.clone()) {
                continue;
            }
            let (indexed_files, total_chunks) = compute_scoped_stats_and_kinds(&p_str)
                .map(|(st, _, _)| (st.indexed_files, st.total_chunks))
                .unwrap_or((0, 0));
            presets.push(FolderPreset {
                label,
                path: p_str,
                indexed_files,
                total_chunks,
            });
        }

        let mut demo_probes = Vec::new();
        let probes_dir = PathBuf::from("/tmp/prism_live_demo/probes");
        if probes_dir.is_dir() {
            let items = [
                (
                    "Text Passage",
                    "text",
                    "probe_snippet.txt",
                    "Excerpt from architecture_guide.md (Lines 1–8)",
                ),
                (
                    "Red Square Crop",
                    "image",
                    "probe_red_crop.png",
                    "200×200 crop matching red_square.png & lecture_demo.mp4 (0:03–0:06)",
                ),
                (
                    "3.5s Audio Excerpt",
                    "audio",
                    "probe_audio_excerpt.wav",
                    "Clipped from meeting_recording.wav (12.0s–15.5s) & lecture_demo.mp4",
                ),
                (
                    "2.0s Video Clip",
                    "video",
                    "probe_video_clip.mp4",
                    "Clipped from lecture_demo.mp4 (3.5s–5.5s)",
                ),
            ];
            for (label, kind, fname, desc) in items {
                let p = probes_dir.join(fname);
                if p.exists() {
                    demo_probes.push(DemoProbe {
                        label: label.to_string(),
                        kind: kind.to_string(),
                        path: p.display().to_string(),
                        description: desc.to_string(),
                    });
                }
            }
        }

        Ok(InitialAppState {
            active_dir,
            stats,
            pending_embeddings,
            presets,
            demo_probes,
            kind_counts,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn get_status(dir: String) -> Result<IndexStats, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let (stats, _, _) = compute_scoped_stats_and_kinds(&dir)?;
        Ok(stats)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn get_kind_counts(dir: String) -> Result<HashMap<String, usize>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let (_, counts, _) = compute_scoped_stats_and_kinds(&dir)?;
        Ok(counts)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn estimate_folder(dir: String) -> Result<EstimateSummary, String> {
    tauri::async_runtime::spawn_blocking(move || {
        estimate_directory(Path::new(&dir)).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn index_folder(
    state: State<'_, AppState>,
    dir: String,
    force: bool,
) -> Result<IndexStats, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (embedder, asr) = app_state.get_or_init()?;
        let stats = Indexer::index_directory(Path::new(&dir), &embedder, Some(&asr), force, None)
            .map_err(|e| e.to_string())?;
        if !IndexStore::is_global_dir(&dir) {
            if let Ok(mut global) = IndexStore::open_global() {
                let _ = global.import_folder_store(Path::new(&dir));
            }
        }
        Ok(stats)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn embed_os_step(
    state: State<'_, AppState>,
    dir: String,
    batch_size: Option<usize>,
) -> Result<BackgroundEmbedProgress, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (embedder, asr) = app_state.get_or_init()?;
        let (embedded_now, remaining_pending) =
            Indexer::embed_pending_global_batch(&embedder, Some(&asr), batch_size.unwrap_or(2))
                .map_err(|e| e.to_string())?;
        let (stats, _, _) = compute_scoped_stats_and_kinds(&dir)?;
        Ok(BackgroundEmbedProgress {
            embedded_now,
            remaining_pending,
            stats,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn browse_folder(
    dir: String,
    kind: Option<String>,
    limit: Option<usize>,
) -> Result<SearchResponse, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let start = Instant::now();
        let (store, scope_prefix) = IndexStore::open_target(&dir).map_err(|e| e.to_string())?;
        let mut rows = store.load_all_chunks().map_err(|e| e.to_string())?;
        if let Some(ref prefix) = scope_prefix {
            rows.retain(|r| r.relative_path.starts_with(prefix));
        }

        // Prioritize demo corpus & embedded multimodal files, then recent files across categories
        rows.sort_by(|a, b| {
            let a_demo = a.relative_path.starts_with("/tmp/prism_live_demo");
            let b_demo = b.relative_path.starts_with("/tmp/prism_live_demo");
            b_demo
                .cmp(&a_demo)
                .then_with(|| b.vector_slot.is_some().cmp(&a.vector_slot.is_some()))
                .then_with(|| b.modified_unix.cmp(&a.modified_unix))
        });

        let kind_filter = kind.as_deref().and_then(SearchKind::from_str);
        let max_hits = limit.unwrap_or(60);

        let mut seen_files = std::collections::HashSet::new();
        let mut hits = Vec::new();
        for row in rows {
            if let Some(kf) = kind_filter {
                if row.kind != kf {
                    continue;
                }
            }
            if !seen_files.insert(row.relative_path.clone()) {
                continue;
            }
            let abs_path = store.root().join(&row.relative_path);
            hits.push(SearchHit {
                file_id: row.file_id,
                chunk_id: row.chunk_id,
                path: abs_path.display().to_string(),
                relative_path: row.relative_path,
                file_name: row.file_name,
                kind: row.kind,
                modality: row.modality,
                score: 3.0,
                semantic_z: 2.2,
                raw_cosine: 1.0,
                lexical_coverage: 1.0,
                snippet: row.text_content.chars().take(260).collect(),
                origin: row.origin,
                modified_unix: row.modified_unix,
                size_bytes: row.size_bytes,
            });
            if hits.len() >= max_hits {
                break;
            }
        }

        Ok(SearchResponse {
            query: String::new(),
            mode: SearchMode::Hybrid,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            decoded_audio_transcript: None,
            distilled_audio_keywords: Vec::new(),
            hits,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn search_folder(
    state: State<'_, AppState>,
    dir: String,
    query: String,
    mode: Option<String>,
    kind: Option<String>,
    limit: Option<usize>,
) -> Result<SearchResponse, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let searcher = Searcher::open(Path::new(&dir)).map_err(|e| e.to_string())?;
        let search_mode = match mode.as_deref().unwrap_or("hybrid") {
            "semantic" => SearchMode::Semantic,
            "lexical" => SearchMode::Lexical,
            _ => SearchMode::Hybrid,
        };

        let (effective_query, forced_kind) = if let Some(stripped) = query.strip_prefix("code:") {
            (stripped.trim().to_string(), Some(SearchKind::Code))
        } else {
            (query, None)
        };

        let mut kinds = Vec::new();
        if let Some(fk) = forced_kind {
            kinds.push(fk);
        } else if let Some(k) = kind {
            if !k.is_empty() && k != "all" {
                if let Some(sk) = SearchKind::from_str(&k) {
                    kinds.push(sk);
                }
            }
        }

        let filter = SearchFilter {
            kinds,
            path_prefix: None,
            limit: limit.unwrap_or(36),
        };

        if search_mode == SearchMode::Lexical {
            return searcher
                .search_text(None, &effective_query, SearchMode::Lexical, &filter)
                .map_err(|e| e.to_string());
        }

        let (embedder, _) = app_state.get_or_init()?;
        searcher
            .search_text(Some(&embedder), &effective_query, search_mode, &filter)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn search_multimodal_file(
    state: State<'_, AppState>,
    dir: String,
    file_path: String,
    limit: Option<usize>,
) -> Result<SearchResponse, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (embedder, asr) = app_state.get_or_init()?;
        let searcher = Searcher::open(Path::new(&dir)).map_err(|e| e.to_string())?;
        let filter = SearchFilter {
            kinds: Vec::new(),
            path_prefix: None,
            limit: limit.unwrap_or(36),
        };
        searcher
            .search_by_file(&embedder, &asr, Path::new(&file_path), &filter)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn search_audio_path(
    state: State<'_, AppState>,
    dir: String,
    audio_path: String,
    hint: Option<String>,
    limit: Option<usize>,
) -> Result<SearchResponse, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (embedder, asr) = app_state.get_or_init()?;
        let searcher = Searcher::open(Path::new(&dir)).map_err(|e| e.to_string())?;
        let filter = SearchFilter {
            kinds: Vec::new(),
            path_prefix: None,
            limit: limit.unwrap_or(36),
        };
        searcher
            .search_audio_file(
                &embedder,
                &asr,
                Path::new(&audio_path),
                false,
                hint.as_deref(),
                &filter,
            )
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn search_voice_wav(
    state: State<'_, AppState>,
    dir: String,
    wav_bytes: Vec<u8>,
    hint: Option<String>,
    limit: Option<usize>,
) -> Result<SearchResponse, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (embedder, asr) = app_state.get_or_init()?;
        let samples = decode_wav_bytes_16k_mono(&wav_bytes).map_err(|e| e.to_string())?;
        let searcher = Searcher::open(Path::new(&dir)).map_err(|e| e.to_string())?;
        let filter = SearchFilter {
            kinds: Vec::new(),
            path_prefix: None,
            limit: limit.unwrap_or(36),
        };
        searcher
            .search_audio_samples(&embedder, &asr, &samples, hint.as_deref(), &filter)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn record_mic_search(
    state: State<'_, AppState>,
    dir: String,
    seconds: Option<f64>,
    hint: Option<String>,
    limit: Option<usize>,
) -> Result<SearchResponse, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let dur = seconds.unwrap_or(4.0).clamp(1.0, 15.0);
        let samples = record_microphone_16k_mono(dur).map_err(|e| e.to_string())?;
        let (embedder, asr) = app_state.get_or_init()?;
        let searcher = Searcher::open(Path::new(&dir)).map_err(|e| e.to_string())?;
        let filter = SearchFilter {
            kinds: Vec::new(),
            path_prefix: None,
            limit: limit.unwrap_or(36),
        };
        searcher
            .search_audio_samples(&embedder, &asr, &samples, hint.as_deref(), &filter)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn verify_containment_file(
    state: State<'_, AppState>,
    dir: String,
    file_path: String,
    limit: Option<usize>,
) -> Result<ContainmentResponse, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (embedder, asr) = app_state.get_or_init()?;
        ContainmentEngine::verify_file(
            Path::new(&dir),
            &embedder,
            &asr,
            Path::new(&file_path),
            limit.unwrap_or(16),
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn verify_containment_text(
    state: State<'_, AppState>,
    dir: String,
    text: String,
    limit: Option<usize>,
) -> Result<ContainmentResponse, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (embedder, _) = app_state.get_or_init()?;
        ContainmentEngine::verify_text(Path::new(&dir), &embedder, &text, limit.unwrap_or(16))
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn decode_asr_file(
    state: State<'_, AppState>,
    audio_path: String,
    dir: Option<String>,
) -> Result<SemanticAsrResult, String> {
    let app_state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (embedder, asr) = app_state.get_or_init()?;
        let mut res = asr
            .decode_file(&embedder, Path::new(&audio_path), false)
            .map_err(|e| e.to_string())?;
        if let Some(d) = dir {
            if let Ok(searcher) = Searcher::open(Path::new(&d)) {
                if let Ok(chunks) = searcher.store().load_all_chunks() {
                    let texts: Vec<String> = chunks
                        .into_iter()
                        .take(16)
                        .map(|c| format!("{} {}", c.file_name, c.text_content))
                        .collect();
                    let _ = asr.distill_corpus_keywords(&embedder, &mut res, &texts, 8);
                }
            }
        }
        Ok(res)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn render_preview(
    path: String,
    kind: String,
    loc: Option<f64>,
    pixels: Option<u32>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let p = Path::new(&path);
        if !p.exists() {
            return Err(format!("File not found: {}", path));
        }
        let max_side = pixels.unwrap_or(360).clamp(48, 1280);

        match kind.as_str() {
            "image" => {
                let img = image::open(p).map_err(|e| e.to_string())?;
                let thumb = img.thumbnail(max_side, max_side);
                let mut buf = Cursor::new(Vec::new());
                thumb
                    .write_to(&mut buf, ImageFormat::Jpeg)
                    .map_err(|e| e.to_string())?;
                Ok(format!(
                    "data:image/jpeg;base64,{}",
                    base64_encode(&buf.into_inner())
                ))
            }
            "video" => {
                let ts = loc.unwrap_or(0.5).max(0.0);
                let out = Command::new("ffmpeg")
                    .args([
                        "-nostdin",
                        "-ss",
                        &format!("{:.2}", ts),
                        "-i",
                        &path,
                        "-frames:v",
                        "1",
                        "-vf",
                        &format!("scale='min({},iw)':-2", max_side),
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "mjpeg",
                        "-q:v",
                        "5",
                        "-",
                    ])
                    .output()
                    .map_err(|e| e.to_string())?;
                if out.status.success() && !out.stdout.is_empty() {
                    Ok(format!(
                        "data:image/jpeg;base64,{}",
                        base64_encode(&out.stdout)
                    ))
                } else {
                    Err("Failed to extract video frame".to_string())
                }
            }
            _ => Err("No image preview for this kind".to_string()),
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn load_media_data_url(path: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let p = Path::new(&path);
        let bytes = fs::read(p).map_err(|e| e.to_string())?;
        if bytes.len() > 30 * 1024 * 1024 {
            return Err("Media file too large for inline data URL; use Open button".to_string());
        }
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let mime = match ext.as_str() {
            "mp4" | "m4v" => "video/mp4",
            "webm" => "video/webm",
            "mov" => "video/quicktime",
            "wav" => "audio/wav",
            "mp3" => "audio/mpeg",
            "m4a" | "aac" => "audio/mp4",
            "ogg" | "opus" => "audio/ogg",
            "flac" => "audio/flac",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            _ => "application/octet-stream",
        };
        Ok(format!("data:{};base64,{}", mime, base64_encode(&bytes)))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn read_file_passage(
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Ok(String::new());
        }
        if let (Some(s), Some(e)) = (start_line, end_line) {
            let from = s.saturating_sub(4).min(lines.len().saturating_sub(1));
            let to = (e + 12).min(lines.len());
            Ok(lines[from..to].join("\n"))
        } else {
            Ok(lines.into_iter().take(80).collect::<Vec<_>>().join("\n"))
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(not(target_os = "macos"))]
    let cmd = "xdg-open";

    Command::new(cmd)
        .arg(&path)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn reveal_path(path: String) -> Result<(), String> {
    let p = Path::new(&path);
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .args(["-R", &path])
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let parent = p.parent().unwrap_or(p);
        Command::new("xdg-open")
            .arg(parent)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn get_ui_test_mode() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--ui-test" {
            return Some(
                args.get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| "all".to_string()),
            );
        }
    }
    None
}

#[tauri::command]
fn report_ui_test(app: tauri::AppHandle, report: serde_json::Value) {
    println!(
        "PRISM_GUI_E2E_REPORT: {}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );
    app.exit(0);
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            runtime: Arc::new(Mutex::new(None)),
        })
        .invoke_handler(tauri::generate_handler![
            get_initial_state,
            get_status,
            get_kind_counts,
            estimate_folder,
            index_folder,
            embed_os_step,
            browse_folder,
            search_folder,
            search_multimodal_file,
            search_audio_path,
            search_voice_wav,
            record_mic_search,
            verify_containment_file,
            verify_containment_text,
            decode_asr_file,
            render_preview,
            load_media_data_url,
            read_file_passage,
            open_path,
            reveal_path,
            get_ui_test_mode,
            report_ui_test,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Prism desktop application");
}

