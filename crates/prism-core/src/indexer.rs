use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use prism_asr::SemanticAsrEngine;
use prism_llama::LlamaEmbedder;
use rayon::prelude::*;

use crate::extractors::{extract_file, ExtractedPayload, ExtractedSegment};
use crate::model::IndexStats;
use crate::rules::{crawl_directory, CrawledFile};
use crate::search_text::SearchText;
use crate::store::{IndexStore, NewChunkRecord};

pub struct Indexer;

impl Indexer {
    pub fn index_directory(
        root: &Path,
        embedder: &LlamaEmbedder,
        asr_engine: Option<&SemanticAsrEngine>,
        force: bool,
        mut progress: Option<&mut dyn FnMut(usize, usize, &str)>,
    ) -> Result<IndexStats> {
        let start = Instant::now();
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
                        rec.size_bytes != f.size_bytes || rec.modified_unix != f.modified_unix
                    }
                    None => true,
                }
            })
            .collect();

        let total_dirty = dirty_files.len();
        if total_dirty == 0 {
            return store.stats(start.elapsed().as_secs_f64() * 1000.0);
        }

        // 3. Stage 1: Parallel extraction across CPU cores
        let extracted_per_file: Vec<(CrawledFile, Vec<ExtractedSegment>)> = dirty_files
            .into_par_iter()
            .map(|file| {
                let segs = extract_file(&file).unwrap_or_default();
                (file, segs)
            })
            .collect();

        // 4. Enrich Semantic ASR lexicon with domain terms from extracted text/filenames
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

        // 5. Stage 2: Embed & store per file
        for (idx, (file, segments)) in extracted_per_file.into_iter().enumerate() {
            if let Some(cb) = progress.as_mut() {
                cb(idx + 1, total_dirty, &file.relative_path);
            }
            if segments.is_empty() {
                continue;
            }

            // Batch text prompts for this file
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

        store.stats(start.elapsed().as_secs_f64() * 1000.0)
    }
}
