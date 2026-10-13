use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use prism_asr::SemanticAsrEngine;
use prism_llama::LlamaEmbedder;
use rayon::prelude::*;

use crate::extractors::{extract_file, ExtractedPayload, ExtractedSegment};
use crate::model::{ChunkModality, ContentOrigin, IndexStats};
use crate::rules::{crawl_directory, crawl_os_roots, CrawledFile};
use crate::search_text::SearchText;
use crate::store::{IndexStore, NewChunkRecord};

pub struct Indexer;

impl Indexer {
    /// Phase 1 Instant OS Sync: Crawl OS directories (`~/Documents`, `~/Downloads`, `~/Desktop`, `~/Pictures`, `~/Music`, `~/Videos`, etc.)
    /// and sync all files into the Global OS Index (`~/.local/share/prism/os_index`) in `<300ms` so every file across the OS
    /// is immediately searchable by filename, folder path, and fast text content.
    pub fn sync_os_roots_fast(roots: &[PathBuf]) -> Result<IndexStats> {
        let start = Instant::now();
        let mut store = IndexStore::open_global()?;
        let crawled = crawl_os_roots(roots)?;
        let _ = store.sync_files_fast(&crawled)?;
        for root in roots {
            let _ = store.import_folder_store(root);
        }
        store.stats(start.elapsed().as_secs_f64() * 1000.0)
    }

    /// Phase 2 Background Multimodal Embedding: Embed up to `batch_size` pending (`embedded = 0`) files in the Global OS Index
    /// using EmbeddingGemma 2 (Text + Vision + Audio Conformer + 768-D Semantic ASR).
    /// Returns `(embedded_in_this_batch, remaining_pending_files)`.
    pub fn embed_pending_global_batch(
        embedder: &LlamaEmbedder,
        asr_engine: Option<&SemanticAsrEngine>,
        batch_size: usize,
    ) -> Result<(usize, usize)> {
        let mut store = IndexStore::open_global()?;
        let (pending, total_pending) = store.pending_unembedded_files(batch_size)?;
        if pending.is_empty() {
            return Ok((0, 0));
        }
        let count = pending.len();
        Self::extract_embed_and_store(&mut store, pending, embedder, asr_engine, None)?;
        Ok((count, total_pending.saturating_sub(count)))
    }

    pub fn index_directory(
        root: &Path,
        embedder: &LlamaEmbedder,
        asr_engine: Option<&SemanticAsrEngine>,
        force: bool,
        progress: Option<&mut dyn FnMut(usize, usize, &str)>,
    ) -> Result<IndexStats> {
        let start = Instant::now();
        let root_str = root.to_string_lossy();
        if IndexStore::is_global_dir(&root_str) {
            let mut store = IndexStore::open_global()?;
            if force {
                store.clear_all()?;
            }
            let roots = IndexStore::default_os_roots();
            let crawled = crawl_os_roots(&roots)?;
            let _ = store.sync_files_fast(&crawled)?;
            let (pending, _) = store.pending_unembedded_files(48)?;
            if !pending.is_empty() {
                Self::extract_embed_and_store(&mut store, pending, embedder, asr_engine, progress)?;
            }
            return store.stats(start.elapsed().as_secs_f64() * 1000.0);
        }

        let mut store = IndexStore::open(root)?;
        if force {
            store.clear_all()?;
        }

        let crawled = crawl_directory(root)?;
        let existing = store.existing_files_map()?;

        // 1. Prune deleted files
        let current_rel: HashSet<&str> = crawled.iter().map(|f| f.relative_path.as_str()).collect();
        for (rel, rec) in &existing {
            if !current_rel.contains(rel.as_str()) {
                store.remove_file(rec.id)?;
            }
        }

        // 2. Filter files that need indexing
        let dirty_files: Vec<CrawledFile> = crawled
            .into_iter()
            .filter(|f| {
                if force {
                    return true;
                }
                match existing.get(&f.relative_path) {
                    Some(rec) => {
                        !rec.embedded
                            || rec.size_bytes != f.size_bytes
                            || rec.modified_unix != f.modified_unix
                    }
                    None => true,
                }
            })
            .collect();

        if dirty_files.is_empty() {
            return store.stats(start.elapsed().as_secs_f64() * 1000.0);
        }

        Self::extract_embed_and_store(&mut store, dirty_files, embedder, asr_engine, progress)?;
        store.stats(start.elapsed().as_secs_f64() * 1000.0)
    }

    fn extract_embed_and_store(
        store: &mut IndexStore,
        files_to_process: Vec<CrawledFile>,
        embedder: &LlamaEmbedder,
        asr_engine: Option<&SemanticAsrEngine>,
        mut progress: Option<&mut dyn FnMut(usize, usize, &str)>,
    ) -> Result<()> {
        let total_dirty = files_to_process.len();
        if total_dirty == 0 {
            return Ok(());
        }

        // Stage 1: Parallel extraction across CPU cores
        let extracted_per_file: Vec<(CrawledFile, Vec<ExtractedSegment>)> = files_to_process
            .into_par_iter()
            .map(|file| {
                let segs = extract_file(&file).unwrap_or_default();
                (file, segs)
            })
            .collect();

        // Enrich Semantic ASR lexicon with domain terms from extracted text/filenames
        if let Some(asr) = asr_engine {
            let mut domain_terms = Vec::new();
            let mut seen = HashSet::new();
            for (file, segs) in &extracted_per_file {
                for tok in SearchText::fts_tokens(&file.file_name) {
                    if tok.len() >= 3 && tok.len() <= 24 && seen.insert(tok.clone()) {
                        domain_terms.push(tok);
                    }
                }
                for seg in segs.iter().take(4) {
                    if matches!(seg.payload, ExtractedPayload::Text { .. }) {
                        for tok in SearchText::fts_tokens(&seg.text_content).into_iter().take(25) {
                            if tok.len() >= 4 && tok.len() <= 24 && seen.insert(tok.clone()) {
                                domain_terms.push(tok);
                            }
                        }
                    }
                }
                if domain_terms.len() >= 256 {
                    break;
                }
            }
            let _ = asr.enrich_vocabulary(embedder, &domain_terms);
        }

        // Stage 2: Embed & store per file
        for (idx, (file, segments)) in extracted_per_file.into_iter().enumerate() {
            if let Some(cb) = progress.as_mut() {
                cb(idx + 1, total_dirty, &file.relative_path);
            }
            if segments.is_empty() {
                // Keep a fallback lexical chunk so the file is marked embedded=1 and searchable by name
                let fallback_chunk = NewChunkRecord {
                    chunk_index: 0,
                    modality: ChunkModality::Text,
                    sets_scale: false,
                    vector: None,
                    ahash: None,
                    text_content: format!("{} ({})", file.file_name, file.kind.as_str()),
                    origin: ContentOrigin::none(),
                };
                store.upsert_file_with_chunks(
                    &file.relative_path,
                    &file.file_name,
                    file.kind,
                    file.size_bytes,
                    file.modified_unix,
                    vec![fallback_chunk],
                )?;
                continue;
            }

            let mut text_indices = Vec::new();
            let mut text_prompts = Vec::new();
            let mut image_indices = Vec::new();
            let mut audio_indices = Vec::new();

            for (s_idx, seg) in segments.iter().enumerate() {
                match &seg.payload {
                    ExtractedPayload::Text { prompt_title } => {
                        text_indices.push(s_idx);
                        text_prompts.push(LlamaEmbedder::document_prompt(
                            &seg.text_content,
                            Some(prompt_title.as_str()),
                        ));
                    }
                    ExtractedPayload::Image { .. } => {
                        image_indices.push(s_idx);
                    }
                    ExtractedPayload::Audio { .. } => {
                        audio_indices.push(s_idx);
                    }
                    ExtractedPayload::LexicalOnly => {}
                }
            }

            let mut vectors_by_seg: Vec<Option<Vec<f32>>> = vec![None; segments.len()];

            if !text_prompts.is_empty() {
                if let Ok(vecs) = embedder.embed_texts(&text_prompts) {
                    for (s_idx, vec) in text_indices.into_iter().zip(vecs.into_iter()) {
                        vectors_by_seg[s_idx] = Some(vec);
                    }
                }
            }

            if !image_indices.is_empty() {
                let img_refs: Vec<(&[u8], usize, usize)> = image_indices
                    .iter()
                    .filter_map(|&s_idx| match &segments[s_idx].payload {
                        ExtractedPayload::Image {
                            rgb,
                            width,
                            height,
                            ..
                        } => Some((rgb.as_slice(), *width, *height)),
                        _ => None,
                    })
                    .collect();
                if let Ok(vecs) = embedder.embed_images(&img_refs) {
                    for (s_idx, vec) in image_indices.into_iter().zip(vecs.into_iter()) {
                        vectors_by_seg[s_idx] = Some(vec);
                    }
                }
            }

            if !audio_indices.is_empty() {
                let win_refs: Vec<&[f32]> = audio_indices
                    .iter()
                    .filter_map(|&s_idx| match &segments[s_idx].payload {
                        ExtractedPayload::Audio { samples_16k } => Some(samples_16k.as_slice()),
                        _ => None,
                    })
                    .collect();
                if let Ok(vecs) = embedder.embed_audio_windows(&win_refs) {
                    for (s_idx, vec) in audio_indices.into_iter().zip(vecs.into_iter()) {
                        vectors_by_seg[s_idx] = Some(vec);
                    }
                }
            }

            let mut new_chunks = Vec::with_capacity(segments.len());
            for (s_idx, seg) in segments.into_iter().enumerate() {
                let ahash = match &seg.payload {
                    ExtractedPayload::Image { ahash, .. } => Some(*ahash),
                    _ => None,
                };
                let mut text_content = seg.text_content;

                // Index-on-the-Fly Semantic-Space ASR for Audio/Video soundtracks!
                if let (ExtractedPayload::Audio { samples_16k }, Some(asr)) =
                    (&seg.payload, asr_engine)
                {
                    if let Ok(asr_res) = asr.decode_samples(embedder, samples_16k) {
                        let kw_line = asr_res
                            .distilled_keywords
                            .iter()
                            .take(8)
                            .map(|m| m.term.as_str())
                            .collect::<Vec<_>>()
                            .join(" ");
                        if !kw_line.is_empty() {
                            text_content = format!(
                                "{} · Semantic ASR: {} ({})",
                                text_content, asr_res.semantic_transcript, kw_line
                            );
                        }
                    }
                }

                new_chunks.push(NewChunkRecord {
                    chunk_index: s_idx,
                    modality: seg.modality,
                    sets_scale: seg.sets_scale,
                    vector: vectors_by_seg[s_idx].take(),
                    ahash,
                    text_content,
                    origin: seg.origin,
                });
            }

            store.upsert_file_with_chunks(
                &file.relative_path,
                &file.file_name,
                file.kind,
                file.size_bytes,
                file.modified_unix,
                new_chunks,
            )?;
        }

        Ok(())
    }
}

