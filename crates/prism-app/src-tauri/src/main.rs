#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use prism_asr::{decode_wav_bytes_16k_mono, SemanticAsrEngine, SemanticAsrResult};
use prism_core::{
    estimate_directory, ContainmentEngine, ContainmentResponse, EstimateSummary, IndexStats,
    IndexStore, Indexer, SearchFilter, SearchKind, SearchMode, SearchResponse, Searcher,
};
use prism_llama::{LlamaEmbedder, LlamaModels};
use tauri::State;

struct AppState {
    runtime: Mutex<Option<(Arc<LlamaEmbedder>, Arc<SemanticAsrEngine>)>>,
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

#[tauri::command]
fn get_status(dir: String) -> Result<IndexStats, String> {
    let store = IndexStore::open(Path::new(&dir)).map_err(|e| e.to_string())?;
    store.stats(0.0).map_err(|e| e.to_string())
}

#[tauri::command]
fn estimate_folder(dir: String) -> Result<EstimateSummary, String> {
    estimate_directory(Path::new(&dir)).map_err(|e| e.to_string())
}

#[tauri::command]
fn index_folder(
    state: State<'_, AppState>,
    dir: String,
    force: bool,
) -> Result<IndexStats, String> {
    let (embedder, asr) = state.get_or_init()?;
    Indexer::index_directory(Path::new(&dir), &embedder, Some(&asr), force, None)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn search_folder(
    state: State<'_, AppState>,
    dir: String,
    query: String,
    mode: Option<String>,
    kind: Option<String>,
    limit: Option<usize>,
) -> Result<SearchResponse, String> {
    let (embedder, _) = state.get_or_init()?;
    let searcher = Searcher::open(Path::new(&dir)).map_err(|e| e.to_string())?;
    let search_mode = match mode.as_deref().unwrap_or("hybrid") {
        "semantic" => SearchMode::Semantic,
        "lexical" => SearchMode::Lexical,
        _ => SearchMode::Hybrid,
    };
    let mut kinds = Vec::new();
    if let Some(k) = kind {
        if let Some(sk) = SearchKind::from_str(&k) {
            kinds.push(sk);
        }
    }
    let filter = SearchFilter {
        kinds,
        path_prefix: None,
        limit: limit.unwrap_or(24),
    };
    searcher
        .search_text(Some(&embedder), &query, search_mode, &filter)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn search_audio_path(
    state: State<'_, AppState>,
    dir: String,
    audio_path: String,
    hint: Option<String>,
    limit: Option<usize>,
) -> Result<SearchResponse, String> {
    let (embedder, asr) = state.get_or_init()?;
    let searcher = Searcher::open(Path::new(&dir)).map_err(|e| e.to_string())?;
    let filter = SearchFilter {
        kinds: Vec::new(),
        path_prefix: None,
        limit: limit.unwrap_or(24),
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
}

#[tauri::command]
fn search_voice_wav(
    state: State<'_, AppState>,
    dir: String,
    wav_bytes: Vec<u8>,
    hint: Option<String>,
    limit: Option<usize>,
) -> Result<SearchResponse, String> {
    let (embedder, asr) = state.get_or_init()?;
    let samples = decode_wav_bytes_16k_mono(&wav_bytes).map_err(|e| e.to_string())?;
    let searcher = Searcher::open(Path::new(&dir)).map_err(|e| e.to_string())?;
    let filter = SearchFilter {
        kinds: Vec::new(),
        path_prefix: None,
        limit: limit.unwrap_or(24),
    };
    searcher
        .search_audio_samples(&embedder, &asr, &samples, hint.as_deref(), &filter)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn verify_containment_file(
    state: State<'_, AppState>,
    dir: String,
    file_path: String,
    limit: Option<usize>,
) -> Result<ContainmentResponse, String> {
    let (embedder, asr) = state.get_or_init()?;
    ContainmentEngine::verify_file(
        Path::new(&dir),
        &embedder,
        &asr,
        Path::new(&file_path),
        limit.unwrap_or(10),
    )
    .map_err(|e| e.to_string())
}

#[tauri::command]
fn verify_containment_text(
    state: State<'_, AppState>,
    dir: String,
    text: String,
    limit: Option<usize>,
) -> Result<ContainmentResponse, String> {
    let (embedder, _) = state.get_or_init()?;
    ContainmentEngine::verify_text(Path::new(&dir), &embedder, &text, limit.unwrap_or(10))
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn decode_asr_file(
    state: State<'_, AppState>,
    audio_path: String,
) -> Result<SemanticAsrResult, String> {
    let (embedder, asr) = state.get_or_init()?;
    asr.decode_file(&embedder, Path::new(&audio_path), false)
        .map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            runtime: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            estimate_folder,
            index_folder,
            search_folder,
            search_audio_path,
            search_voice_wav,
            verify_containment_file,
            verify_containment_text,
            decode_asr_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Prism desktop application");
}
