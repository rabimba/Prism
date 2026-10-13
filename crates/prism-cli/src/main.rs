use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use prism_asr::{record_microphone_16k_mono, SemanticAsrEngine};
use prism_core::{
    estimate_directory, ContainmentEngine, IndexStore, Indexer, SearchFilter, SearchKind,
    SearchMode, Searcher,
};
use prism_llama::{ImageSizing, LlamaEmbedder, LlamaModels, ModelFiles};
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "prism",
    version,
    about = "Prism — Cross-Platform Multimodal Search, Semantic-Space ASR & Part-of-File Verification Engine powered by EmbeddingGemma 2"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Crawl a folder and estimate indexing cost and file breakdown
    Estimate {
        #[arg(default_value = ".")]
        dir: PathBuf,
        #[arg(long)]
        json: bool,
    },

    /// Index a folder using EmbeddingGemma 2 (Text + Vision + Audio + Semantic ASR)
    Index {
        #[arg(default_value = ".")]
        dir: PathBuf,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        json: bool,
    },

    /// Search an indexed folder by text, spoken audio, microphone, or multimodal file
    Search {
        /// Optional text query
        query: Option<String>,
        /// Target indexed directory
        #[arg(short = 'd', long, default_value = ".")]
        dir: PathBuf,
        /// Search using a spoken audio file (WAV, MP3, M4A, FLAC, OGG)
        #[arg(long)]
        audio: Option<PathBuf>,
        /// Record spoken query from microphone for N seconds
        #[arg(long)]
        mic: Option<f64>,
        /// Search using any multimodal file (Image, Video, Audio, PDF, Text)
        #[arg(short = 'i', long)]
        input: Option<PathBuf>,
        /// Search mode: hybrid, semantic, or lexical
        #[arg(short = 'm', long, default_value = "hybrid")]
        mode: String,
        /// Filter by file kind (text, code, document, image, audio, video, spreadsheet, presentation)
        #[arg(short = 'k', long)]
        kind: Vec<String>,
        /// Max results to return
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
        /// Enable optional verbatim ASR fallback if available
        #[arg(long)]
        verbatim: bool,
        #[arg(long)]
        json: bool,
    },

    /// Check whether an Image, Video, Audio clip, Document, or Text snippet is contained inside ("part of") the indexed files
    Contain {
        /// Path to an Image, Video, Audio, PDF, Office doc, or Code/Text file (or omit if using --text)
        file: Option<PathBuf>,
        /// Raw text passage to check for containment
        #[arg(long)]
        text: Option<String>,
        /// Target indexed directory
        #[arg(short = 'd', long, default_value = ".")]
        dir: PathBuf,
        /// Max candidate parent files to return
        #[arg(short = 'n', long, default_value_t = 8)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },

    /// Run EmbeddingGemma 2 Semantic-Space ASR on an audio file or microphone recording
    Asr {
        /// Audio file path (or omit if using --mic)
        audio: Option<PathBuf>,
        /// Record from microphone for N seconds
        #[arg(long)]
        mic: Option<f64>,
        /// Optional indexed directory to ground Semantic Pseudo-Relevance Keyword Distillation
        #[arg(short = 'd', long)]
        dir: Option<PathBuf>,
        #[arg(long)]
        verbatim: bool,
        #[arg(long)]
        json: bool,
    },

    /// Show index statistics for a folder
    Status {
        #[arg(default_value = ".")]
        dir: PathBuf,
        #[arg(long)]
        json: bool,
    },

    /// Benchmark EmbeddingGemma 2 text, vision, and audio throughput on this machine
    Speed {
        #[arg(long)]
        json: bool,
    },

    /// Download and verify EmbeddingGemma 2 GGUF models
    DownloadModel {
        #[arg(long)]
        dir: Option<PathBuf>,
    },

    /// Run a persistent JSON-RPC worker over stdio keeping models hot in memory
    Worker,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Estimate { dir, json } => {
            let summary = estimate_directory(&dir)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            } else {
                println!("Prism Folder Estimate: {}", summary.root.display());
                println!("  Total files     : {}", summary.total_files);
                println!(
                    "  Total size      : {:.2} MB",
                    (summary.total_bytes as f64) / (1024.0 * 1024.0)
                );
                println!("  Estimated chunks: {}", summary.estimated_chunks);
                println!("  Estimated time  : {:.1}s", summary.estimated_seconds);
                for (k, v) in &summary.by_kind {
                    println!("    - {:<12}: {}", k, v);
                }
            }
        }

        Commands::DownloadModel { dir } => {
            let target = dir.unwrap_or_else(ModelFiles::default_directory);
            eprintln!("Ensuring EmbeddingGemma 2 models in {} ...", target.display());
            let mut last_pct = 0u64;
            let paths = ModelFiles::ensure(
                Some(&target),
                Some(&mut |done, total, name| {
                    let pct = (done * 100) / total.max(1);
                    if pct >= last_pct + 5 || done == total {
                        eprintln!("  [{}%] {}", pct, name);
                        last_pct = pct;
                    }
                }),
            )?;
            println!("Models ready:");
            println!("  Text  : {}", paths.model_path.display());
            println!("  Media : {}", paths.mmproj_path.display());
        }

        Commands::Index { dir, force, json } => {
            let models = LlamaModels::load_default(true)?;
            let embedder = LlamaEmbedder::new(models, None)?;
            let asr_engine = SemanticAsrEngine::new(&embedder)?;

            let stats = Indexer::index_directory(
                &dir,
                &embedder,
                Some(&asr_engine),
                force,
                Some(&mut |done, total, rel| {
                    if !json {
                        eprintln!("  [{}/{}] {}", done, total, rel);
                    }
                }),
            )?;

            if json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!(
                    "Indexed {} files ({} chunks: {} text, {} image, {} audio) in {:.0} ms",
                    stats.indexed_files,
                    stats.total_chunks,
                    stats.text_chunks,
                    stats.image_chunks,
                    stats.audio_chunks,
                    stats.elapsed_ms
                );
            }
        }

        Commands::Search {
            query,
            dir,
            audio,
            mic,
            input,
            mode,
            kind,
            limit,
            verbatim,
            json,
        } => {
            let search_mode = match mode.to_lowercase().as_str() {
                "semantic" | "sem" => SearchMode::Semantic,
                "lexical" | "lex" | "words" => SearchMode::Lexical,
                _ => SearchMode::Hybrid,
            };
            let kinds: Vec<SearchKind> = kind
                .iter()
                .filter_map(|k| SearchKind::from_str(k))
                .collect();
            let filter = SearchFilter {
                kinds,
                path_prefix: None,
                limit,
            };

            let searcher = Searcher::open(&dir)?;

            let response = if let Some(audio_path) = audio {
                let models = LlamaModels::load_default(true)?;
                let embedder = LlamaEmbedder::new(models, None)?;
                let asr_engine = SemanticAsrEngine::new(&embedder)?;
                searcher.search_audio_file(
                    &embedder,
                    &asr_engine,
                    &audio_path,
                    verbatim,
                    query.as_deref(),
                    &filter,
                )?
            } else if let Some(dur) = mic {
                eprintln!("Recording microphone for {:.1}s ...", dur);
                let samples = record_microphone_16k_mono(dur)?;
                let models = LlamaModels::load_default(true)?;
                let embedder = LlamaEmbedder::new(models, None)?;
                let asr_engine = SemanticAsrEngine::new(&embedder)?;
                searcher.search_audio_samples(
                    &embedder,
                    &asr_engine,
                    &samples,
                    query.as_deref(),
                    &filter,
                )?
            } else if let Some(input_path) = input {
                let models = LlamaModels::load_default(true)?;
                let embedder = LlamaEmbedder::new(models, None)?;
                let asr_engine = SemanticAsrEngine::new(&embedder)?;
                searcher.search_by_file(&embedder, &asr_engine, &input_path, &filter)?
            } else {
                let q = query.unwrap_or_default();
                if search_mode == SearchMode::Lexical {
                    searcher.search_text(None, &q, search_mode, &filter)?
                } else {
                    let models = LlamaModels::load_default(false)?;
                    let embedder = LlamaEmbedder::new(models, None)?;
                    searcher.search_text(Some(&embedder), &q, search_mode, &filter)?
                }
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&response)?);
            } else {
                if let Some(ref tr) = response.decoded_audio_transcript {
                    println!("Semantic Audio Decode: \"{}\"", tr);
                    if !response.distilled_audio_keywords.is_empty() {
                        let kws = response
                            .distilled_audio_keywords
                            .iter()
                            .take(6)
                            .map(|k| format!("{} (z={:.2})", k.term, k.z_score))
                            .collect::<Vec<_>>()
                            .join(", ");
                        println!("Distilled 768-D Keywords: {}", kws);
                    }
                }
                println!(
                    "Found {} results for \"{}\" ({:.1} ms):",
                    response.hits.len(),
                    response.query,
                    response.elapsed_ms
                );
                for (idx, hit) in response.hits.iter().enumerate() {
                    let loc = hit
                        .origin
                        .format_label()
                        .map(|l| format!(" [{}]", l))
                        .unwrap_or_default();
                    println!(
                        "  {}. {} ({}){}",
                        idx + 1,
                        hit.relative_path,
                        hit.kind.as_str(),
                        loc
                    );
                    println!(
                        "     score={:.2} (z={:.2}, cos={:.3}, lex={:.2}) | {}",
                        hit.score, hit.semantic_z, hit.raw_cosine, hit.lexical_coverage, hit.snippet
                    );
                }
            }
        }

        Commands::Contain {
            file,
            text,
            dir,
            limit,
            json,
        } => {
            let models = LlamaModels::load_default(true)?;
            let embedder = LlamaEmbedder::new(models, None)?;
            let asr_engine = SemanticAsrEngine::new(&embedder)?;

            let response = if let Some(ref t) = text {
                ContainmentEngine::verify_text(&dir, &embedder, t, limit)?
            } else if let Some(ref f) = file {
                if f.exists() {
                    ContainmentEngine::verify_file(&dir, &embedder, &asr_engine, f, limit)?
                } else {
                    // Treat non-existent file arg as a raw text query for convenience
                    ContainmentEngine::verify_text(
                        &dir,
                        &embedder,
                        &f.to_string_lossy(),
                        limit,
                    )?
                }
            } else {
                anyhow::bail!("Provide either a file path or --text <snippet> to check containment");
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&response)?);
            } else {
                println!(
                    "Part-of-File Check for \"{}\" ({}):",
                    response.query_description, response.query_modality
                );
                println!(
                    "  Contained in Index: {} (best confidence: {:.1}%, {:.1} ms)",
                    if response.is_contained { "YES" } else { "NO" },
                    response.best_confidence * 100.0,
                    response.elapsed_ms
                );
                for (idx, hit) in response.hits.iter().enumerate() {
                    println!(
                        "  {}. [{}] {} — {} (confidence: {:.1}%)",
                        idx + 1,
                        hit.verdict.label(),
                        hit.relative_path,
                        hit.location_summary,
                        hit.confidence * 100.0
                    );
                    let mut metrics = vec![format!("sem={:.3}", hit.semantic_similarity)];
                    if let Some(ah) = hit.perceptual_hash_similarity {
                        metrics.push(format!("aHash={:.3}", ah));
                    }
                    if let Some(lx) = hit.lexical_containment {
                        metrics.push(format!("lex={:.3}", lx));
                    }
                    if let Some(ta) = hit.temporal_alignment {
                        metrics.push(format!("temp={:.3}", ta));
                    }
                    println!("     {} | {}", metrics.join(", "), hit.snippet);
                }
            }
        }

        Commands::Asr {
            audio,
            mic,
            dir,
            verbatim,
            json,
        } => {
            let models = LlamaModels::load_default(true)?;
            let embedder = LlamaEmbedder::new(models, None)?;
            let asr_engine = SemanticAsrEngine::new(&embedder)?;

            let mut res = if let Some(ref path) = audio {
                asr_engine.decode_file(&embedder, path, verbatim)?
            } else if let Some(dur) = mic {
                eprintln!("Recording microphone for {:.1}s ...", dur);
                let samples = record_microphone_16k_mono(dur)?;
                asr_engine.decode_samples(&embedder, &samples)?
            } else {
                anyhow::bail!("Provide an audio file path or --mic <seconds>");
            };

            // If an indexed directory is provided, ground the ASR with Semantic Pseudo-Relevance Feedback!
            if let Some(ref d) = dir {
                if IndexStore::exists_in(d) {
                    if let Ok(searcher) = Searcher::open(d) {
                        let resp = searcher.search_audio_samples(
                            &embedder,
                            &asr_engine,
                            &[],
                            Some(&res.effective_query()),
                            &SearchFilter {
                                kinds: Vec::new(),
                                path_prefix: None,
                                limit: 6,
                            },
                        );
                        if let Ok(r) = resp {
                            let cands: Vec<String> = r
                                .hits
                                .iter()
                                .map(|h| format!("{} {}", h.file_name, h.snippet))
                                .collect();
                            if let Ok(prf) =
                                asr_engine.distill_corpus_keywords(&embedder, &res, &cands, 8)
                            {
                                res.distilled_keywords = prf;
                            }
                        }
                    }
                }
            }

            if json {
                println!("{}", serde_json::to_string_pretty(&res)?);
            } else {
                println!("EmbeddingGemma 2 Semantic-Space ASR ({:.2}s):", res.duration_s);
                println!("  Semantic Transcript : {}", res.semantic_transcript);
                if let Some(ref v) = res.verbatim_transcript {
                    println!("  Verbatim Transcript : {}", v);
                }
                println!("  Top 768-D Keywords  :");
                for kw in &res.distilled_keywords {
                    println!(
                        "    - {:<18} z={:>5.2}  cos={:.4}",
                        kw.term, kw.z_score, kw.cosine
                    );
                }
                println!("  Acoustic Sub-Windows ({}):", res.windows.len());
                for w in &res.windows {
                    let toks = w
                        .top_tokens
                        .iter()
                        .take(3)
                        .map(|t| format!("{}({:.2})", t.term, t.z_score))
                        .collect::<Vec<_>>()
                        .join(", ");
                    println!(
                        "    [{:>5.2}s - {:>5.2}s] rms={:.4} -> {}",
                        w.start_s, w.end_s, w.rms, toks
                    );
                }
            }
        }

        Commands::Status { dir, json } => {
            let store = IndexStore::open(&dir)?;
            let stats = store.stats(0.0)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("Prism Index Status: {}", stats.root);
                println!("  Indexed files : {}", stats.indexed_files);
                println!(
                    "  Total chunks  : {} ({} text, {} image, {} audio)",
                    stats.total_chunks, stats.text_chunks, stats.image_chunks, stats.audio_chunks
                );
                println!("  Vectors size  : {:.2} KB", stats.vector_bytes as f64 / 1024.0);
                println!("  SQLite DB size: {:.2} KB", stats.db_bytes as f64 / 1024.0);
                println!("  Models ready  : {}", ModelFiles::is_ready(None));
            }
        }

        Commands::Speed { json } => {
            let models = LlamaModels::load_default(true)?;
            let embedder = LlamaEmbedder::new(models, None)?;

            let t0 = Instant::now();
            let sample_texts = vec![
                "title: Architecture | text: Prism unifies text, image, audio, and video retrieval in a 768-dimensional semantic space.",
                "title: Search | text: Modality-centered Z-score calibration aligns cross-modal cosine similarities.",
                "title: ASR | text: Spoken audio queries are projected onto the semantic lexicon without intermediate transcription latency.",
                "title: Containment | text: Perceptual 64-bit hashes and 768-D embeddings verify exact part-of-file relationships.",
            ];
            let _ = embedder.embed_texts(&sample_texts)?;
            let text_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let dummy_rgb = vec![128u8; 384 * 384 * 3];
            let _ = embedder.embed_image_rgb(&dummy_rgb, 384, 384, ImageSizing::FRAME_BUDGET)?;
            let image_ms = t1.elapsed().as_secs_f64() * 1000.0;

            let t2 = Instant::now();
            let dummy_audio = vec![0.01f32; 16_000 * 2];
            let _ = embedder.embed_audio_samples(&dummy_audio)?;
            let audio_ms = t2.elapsed().as_secs_f64() * 1000.0;

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "text_batch_4_ms": text_ms,
                        "text_per_doc_ms": text_ms / 4.0,
                        "image_frame_ms": image_ms,
                        "audio_2s_ms": audio_ms,
                    }))?
                );
            } else {
                println!("Prism EmbeddingGemma 2 Speed Benchmark:");
                println!(
                    "  Text (batch of 4) : {:.1} ms ({:.1} ms/chunk)",
                    text_ms,
                    text_ms / 4.0
                );
                println!("  Image (70 tokens) : {:.1} ms", image_ms);
                println!("  Audio (2s 16kHz)  : {:.1} ms", audio_ms);
            }
        }

        Commands::Worker => {
            run_stdio_worker()?;
        }
    }

    Ok(())
}

/// Persistent JSON-RPC over stdio worker keeping EmbeddingGemma 2 models hot in memory.
fn run_stdio_worker() -> Result<()> {
    let models = LlamaModels::load_default(true)?;
    let embedder = LlamaEmbedder::new(models, None)?;
    let asr_engine = SemanticAsrEngine::new(&embedder)?;

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let req: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let err = json!({"ok": false, "error": e.to_string()});
                writeln!(stdout, "{}", err)?;
                stdout.flush()?;
                continue;
            }
        };

        let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
        let cmd = req.get("cmd").and_then(|v| v.as_str()).unwrap_or("");
        let dir = PathBuf::from(req.get("dir").and_then(|v| v.as_str()).unwrap_or("."));

        let result: Result<serde_json::Value> = match cmd {
            "ping" => Ok(json!({"pong": true})),
            "status" => {
                let store = IndexStore::open(&dir)?;
                Ok(serde_json::to_value(store.stats(0.0)?)?)
            }
            "index" => {
                let force = req.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
                let stats =
                    Indexer::index_directory(&dir, &embedder, Some(&asr_engine), force, None)?;
                Ok(serde_json::to_value(stats)?)
            }
            "search" => {
                let query = req.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let limit = req.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let filter = SearchFilter {
                    kinds: Vec::new(),
                    path_prefix: None,
                    limit,
                };
                let searcher = Searcher::open(&dir)?;
                if let Some(audio_path) = req.get("audio").and_then(|v| v.as_str()) {
                    let resp = searcher.search_audio_file(
                        &embedder,
                        &asr_engine,
                        Path::new(audio_path),
                        false,
                        if query.is_empty() { None } else { Some(query) },
                        &filter,
                    )?;
                    Ok(serde_json::to_value(resp)?)
                } else {
                    let resp = searcher.search_text(
                        Some(&embedder),
                        query,
                        SearchMode::Hybrid,
                        &filter,
                    )?;
                    Ok(serde_json::to_value(resp)?)
                }
            }
            "contain" => {
                let limit = req.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                if let Some(file_path) = req.get("file").and_then(|v| v.as_str()) {
                    let resp = ContainmentEngine::verify_file(
                        &dir,
                        &embedder,
                        &asr_engine,
                        Path::new(file_path),
                        limit,
                    )?;
                    Ok(serde_json::to_value(resp)?)
                } else if let Some(txt) = req.get("text").and_then(|v| v.as_str()) {
                    let resp = ContainmentEngine::verify_text(&dir, &embedder, txt, limit)?;
                    Ok(serde_json::to_value(resp)?)
                } else {
                    anyhow::bail!("Missing file or text for contain")
                }
            }
            _ => anyhow::bail!("Unknown worker command: {}", cmd),
        };

        let reply = match result {
            Ok(data) => json!({"id": id, "ok": true, "data": data}),
            Err(e) => json!({"id": id, "ok": false, "error": e.to_string()}),
        };
        writeln!(stdout, "{}", reply)?;
        stdout.flush()?;
    }

    Ok(())
}
