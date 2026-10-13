use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use prism_asr::SemanticAsrEngine;
use prism_llama::{cosine_similarity, ImageSizing, LlamaEmbedder};

use crate::extractors::{
    extract_file, extract_video_frame_at, probe_media_duration, ExtractedPayload,
};
use crate::model::{
    ChunkModality, ContainmentHit, ContainmentResponse, ContainmentVerdict, SearchKind,
};
use crate::rules::{classify_path, CrawledFile};
use crate::search_text::SearchText;
use crate::store::{IndexStore, IndexedChunkRow};
use crate::vectors::compute_modality_z_scores;

pub struct ContainmentEngine;

impl ContainmentEngine {
    /// Check whether `input_path` (any Image, Video, Audio, Text, PDF, Office doc, or Code file)
    /// is contained within or part of any indexed file in `root`.
    pub fn verify_file(
        root: &Path,
        embedder: &LlamaEmbedder,
        asr_engine: &SemanticAsrEngine,
        input_path: &Path,
        limit: usize,
    ) -> Result<ContainmentResponse> {
        let start = Instant::now();
        let (store, scope_prefix) = IndexStore::open_target(&root.to_string_lossy())?;
        let mut chunks = store.load_all_chunks()?;
        if let Some(ref prefix) = scope_prefix {
            chunks.retain(|c| c.relative_path.starts_with(prefix));
        }
        let kind = classify_path(input_path).unwrap_or(SearchKind::Text);
        let file_name = input_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| input_path.to_string_lossy().to_string());

        let hits = match kind {
            SearchKind::Image => {
                Self::verify_image_input(&store, &chunks, embedder, input_path, limit)?
            }
            SearchKind::Video => {
                Self::verify_video_input(&store, &chunks, embedder, input_path, limit)?
            }
            SearchKind::Audio => {
                Self::verify_audio_input(&store, &chunks, embedder, asr_engine, input_path, limit)?
            }
            _ => Self::verify_document_or_text_file(&store, &chunks, embedder, input_path, kind, limit)?,
        };

        let best_confidence = hits.first().map(|h| h.confidence).unwrap_or(0.0);
        let is_contained = hits.iter().any(|h| {
            matches!(
                h.verdict,
                ContainmentVerdict::ExactPart
                    | ContainmentVerdict::NearDuplicatePart
                    | ContainmentVerdict::SemanticPart
            ) && h.confidence >= 0.62
        });

        Ok(ContainmentResponse {
            query_description: file_name,
            query_modality: kind.as_str().to_string(),
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            is_contained,
            best_confidence,
            hits,
        })
    }

    /// Check whether a raw text snippet / passage is part of any indexed file in `root`.
    pub fn verify_text(
        root: &Path,
        embedder: &LlamaEmbedder,
        text_snippet: &str,
        limit: usize,
    ) -> Result<ContainmentResponse> {
        let start = Instant::now();
        let (store, scope_prefix) = IndexStore::open_target(&root.to_string_lossy())?;
        let mut chunks = store.load_all_chunks()?;
        if let Some(ref prefix) = scope_prefix {
            chunks.retain(|c| c.relative_path.starts_with(prefix));
        }

        let hits = Self::score_text_containment(&store, &chunks, embedder, text_snippet, limit)?;
        let best_confidence = hits.first().map(|h| h.confidence).unwrap_or(0.0);
        let is_contained = hits.iter().any(|h| {
            matches!(
                h.verdict,
                ContainmentVerdict::ExactPart
                    | ContainmentVerdict::NearDuplicatePart
                    | ContainmentVerdict::SemanticPart
            ) && h.confidence >= 0.62
        });

        Ok(ContainmentResponse {
            query_description: SearchText::collapsed(text_snippet, 64),
            query_modality: "text".to_string(),
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            is_contained,
            best_confidence,
            hits,
        })
    }

    fn verify_image_input(
        store: &IndexStore,
        chunks: &[IndexedChunkRow],
        embedder: &LlamaEmbedder,
        image_path: &Path,
        limit: usize,
    ) -> Result<Vec<ContainmentHit>> {
        let (q_vec, q_ahash) =
            embedder.embed_image_path(image_path, ImageSizing::PHOTO_BUDGET)?;

        let slots: Vec<Option<usize>> = chunks.iter().map(|c| c.vector_slot).collect();
        let raw_scores = store.vectors.score_slots(&q_vec, &slots);
        let modalities: Vec<ChunkModality> = chunks.iter().map(|c| c.modality).collect();
        let sets_scale: Vec<bool> = chunks.iter().map(|c| c.sets_scale).collect();
        let z_scores = compute_modality_z_scores(&raw_scores, &modalities, &sets_scale);

        let mut hits = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let sem = (raw_scores[i] as f64).clamp(0.0, 1.0);
            let ahash_sim = chunk
                .ahash
                .map(|ch| ImageSizing::ahash_similarity(q_ahash, ch) as f64);

            let (confidence, verdict) = if chunk.modality == ChunkModality::Image {
                let ah = ahash_sim.unwrap_or(0.0);
                let conf = (0.68 * sem + 0.32 * ah).max(sem);
                let v = if sem >= 0.95 || ah >= 0.92 {
                    ContainmentVerdict::ExactPart
                } else if conf >= 0.78 || (sem >= 0.72 && ah >= 0.75) {
                    ContainmentVerdict::NearDuplicatePart
                } else if sem >= 0.55 || z_scores[i] >= 1.8 {
                    ContainmentVerdict::SemanticPart
                } else {
                    ContainmentVerdict::RelatedMatch
                };
                (conf, v)
            } else {
                // Cross-modal containment (e.g. query image described in a document or audio clip)
                let z_norm = (z_scores[i] / 3.5).clamp(0.0, 0.92);
                let conf = z_norm.max(sem);
                let v = if z_scores[i] >= 2.2 {
                    ContainmentVerdict::SemanticPart
                } else {
                    ContainmentVerdict::RelatedMatch
                };
                (conf, v)
            };

            if confidence < 0.25 {
                continue;
            }

            hits.push(build_containment_hit(
                store,
                chunk,
                verdict,
                confidence,
                sem,
                ahash_sim,
                None,
                None,
                &chunk.text_content,
            ));
        }

        dedup_and_sort_hits(hits, limit)
    }

    fn verify_video_input(
        store: &IndexStore,
        chunks: &[IndexedChunkRow],
        embedder: &LlamaEmbedder,
        video_path: &Path,
        limit: usize,
    ) -> Result<Vec<ContainmentHit>> {
        let dur = probe_media_duration(video_path).unwrap_or(6.0).max(1.0);
        let sample_count = ((dur / 2.0).ceil() as usize).clamp(1, 6);
        let step = dur / (sample_count as f64);

        let mut frame_vecs = Vec::new();
        let mut frame_hashes = Vec::new();
        for i in 0..sample_count {
            let t = (i as f64 + 0.3) * step;
            if let Ok((rgb, w, h, ahash)) =
                extract_video_frame_at(video_path, t, ImageSizing::FRAME_BUDGET)
            {
                if let Ok(mut vecs) = embedder.embed_images(&[(&rgb, w, h)]) {
                    if let Some(v) = vecs.pop() {
                        frame_vecs.push(v);
                        frame_hashes.push(ahash);
                    }
                }
            }
        }

        let audio_vec = prism_asr::load_audio_16k_mono(video_path)
            .ok()
            .filter(|s| !s.is_empty())
            .and_then(|s| embedder.embed_audio_samples(&s).ok());

        let mut hits = Vec::new();
        for chunk in chunks {
            let Some(slot) = chunk.vector_slot else {
                continue;
            };
            let Some(c_vec) = store.vectors.get_vector(slot) else {
                continue;
            };

            let mut best_sem = 0.0f64;
            let mut sum_sem = 0.0f64;
            for fv in &frame_vecs {
                let s = cosine_similarity(fv, &c_vec) as f64;
                if s > best_sem {
                    best_sem = s;
                }
                sum_sem += s;
            }
            let mean_sem = if !frame_vecs.is_empty() {
                sum_sem / (frame_vecs.len() as f64)
            } else {
                0.0
            };

            if let Some(ref av) = audio_vec {
                if chunk.modality == ChunkModality::Audio {
                    let s = cosine_similarity(av, &c_vec) as f64;
                    if s > best_sem {
                        best_sem = s;
                    }
                }
            }

            let best_ahash = chunk.ahash.and_then(|ch| {
                frame_hashes
                    .iter()
                    .map(|&fh| ImageSizing::ahash_similarity(fh, ch) as f64)
                    .fold(None, |acc: Option<f64>, x| Some(acc.map_or(x, |a| a.max(x))))
            });

            let ah = best_ahash.unwrap_or(0.0);
            let temporal = if frame_vecs.len() > 1 {
                Some((0.6 * best_sem + 0.4 * mean_sem).clamp(0.0, 1.0))
            } else {
                None
            };
            let confidence = (0.65 * best_sem + 0.35 * ah).max(best_sem);
            if confidence < 0.25 {
                continue;
            }

            let verdict = if best_sem >= 0.94 || ah >= 0.90 {
                ContainmentVerdict::ExactPart
            } else if confidence >= 0.76 {
                ContainmentVerdict::NearDuplicatePart
            } else if best_sem >= 0.52 {
                ContainmentVerdict::SemanticPart
            } else {
                ContainmentVerdict::RelatedMatch
            };

            hits.push(build_containment_hit(
                store,
                chunk,
                verdict,
                confidence,
                best_sem,
                best_ahash,
                None,
                temporal,
                &chunk.text_content,
            ));
        }

        dedup_and_sort_hits(hits, limit)
    }

    fn verify_audio_input(
        store: &IndexStore,
        chunks: &[IndexedChunkRow],
        embedder: &LlamaEmbedder,
        asr_engine: &SemanticAsrEngine,
        audio_path: &Path,
        limit: usize,
    ) -> Result<Vec<ContainmentHit>> {
        let asr = asr_engine.decode_file(embedder, audio_path, false)?;
        let slots: Vec<Option<usize>> = chunks.iter().map(|c| c.vector_slot).collect();
        let raw_scores = store.vectors.score_slots(&asr.audio_embedding, &slots);
        let modalities: Vec<ChunkModality> = chunks.iter().map(|c| c.modality).collect();
        let sets_scale: Vec<bool> = chunks.iter().map(|c| c.sets_scale).collect();
        let z_scores = compute_modality_z_scores(&raw_scores, &modalities, &sets_scale);

        let mut hits = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let holistic_sem = (raw_scores[i] as f64).clamp(0.0, 1.0);
            let mut max_win_sem = holistic_sem;
            if let Some(slot) = chunk.vector_slot {
                if let Some(c_vec) = store.vectors.get_vector(slot) {
                    for w in &asr.windows {
                        let s = cosine_similarity(&w.embedding, &c_vec) as f64;
                        if s > max_win_sem {
                            max_win_sem = s;
                        }
                    }
                }
            }

            let lex = if !asr.semantic_transcript.is_empty() {
                Some(SearchText::lexical_containment(
                    &asr.semantic_transcript,
                    &chunk.text_content,
                ))
            } else {
                None
            };

            let (confidence, verdict) = if chunk.modality == ChunkModality::Audio {
                let conf = (0.55 * holistic_sem + 0.45 * max_win_sem).max(max_win_sem);
                let v = if conf >= 0.94 {
                    ContainmentVerdict::ExactPart
                } else if conf >= 0.76 {
                    ContainmentVerdict::NearDuplicatePart
                } else if conf >= 0.52 || z_scores[i] >= 1.8 {
                    ContainmentVerdict::SemanticPart
                } else {
                    ContainmentVerdict::RelatedMatch
                };
                (conf, v)
            } else {
                // Cross-modal: spoken audio matching a text passage or image
                let z_norm = (z_scores[i] / 3.2).clamp(0.0, 0.92);
                let lex_val = lex.unwrap_or(0.0);
                let conf = z_norm.max(0.6 * z_norm + 0.4 * lex_val).max(holistic_sem);
                let v = if z_scores[i] >= 2.0 || lex_val >= 0.65 {
                    ContainmentVerdict::SemanticPart
                } else {
                    ContainmentVerdict::RelatedMatch
                };
                (conf, v)
            };

            if confidence < 0.25 {
                continue;
            }

            hits.push(build_containment_hit(
                store,
                chunk,
                verdict,
                confidence,
                max_win_sem,
                None,
                lex,
                Some(max_win_sem),
                &chunk.text_content,
            ));
        }

        dedup_and_sort_hits(hits, limit)
    }

    fn verify_document_or_text_file(
        store: &IndexStore,
        chunks: &[IndexedChunkRow],
        embedder: &LlamaEmbedder,
        path: &Path,
        kind: SearchKind,
        limit: usize,
    ) -> Result<Vec<ContainmentHit>> {
        let file_name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let ext = path
            .extension()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let crawled = CrawledFile {
            path: path.to_path_buf(),
            relative_path: file_name.clone(),
            file_name,
            extension: ext,
            kind,
            size_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
            modified_unix: 0,
        };

        let segments = extract_file(&crawled).unwrap_or_default();
        let combined_text = segments
            .iter()
            .filter_map(|s| match s.payload {
                ExtractedPayload::Text { .. } | ExtractedPayload::LexicalOnly => {
                    Some(s.text_content.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if !combined_text.trim().is_empty() {
            return Self::score_text_containment(store, chunks, embedder, &combined_text, limit);
        }

        // Fallback if scanned PDF with image pages
        for seg in segments {
            if let ExtractedPayload::Image {
                rgb,
                width,
                height,
                ahash: _,
            } = seg.payload
            {
                let vec = embedder.embed_images(&[(&rgb, width, height)])?[0].clone();
                let slots: Vec<Option<usize>> = chunks.iter().map(|c| c.vector_slot).collect();
                let scores = store.vectors.score_slots(&vec, &slots);
                let mut hits = Vec::new();
                for (i, chunk) in chunks.iter().enumerate() {
                    let sem = scores[i] as f64;
                    if sem >= 0.45 {
                        hits.push(build_containment_hit(
                            store,
                            chunk,
                            if sem >= 0.90 {
                                ContainmentVerdict::ExactPart
                            } else {
                                ContainmentVerdict::SemanticPart
                            },
                            sem,
                            sem,
                            None,
                            None,
                            None,
                            &chunk.text_content,
                        ));
                    }
                }
                return dedup_and_sort_hits(hits, limit);
            }
        }

        Ok(Vec::new())
    }

    fn score_text_containment(
        store: &IndexStore,
        chunks: &[IndexedChunkRow],
        embedder: &LlamaEmbedder,
        needle_text: &str,
        limit: usize,
    ) -> Result<Vec<ContainmentHit>> {
        let query_slice: String = needle_text.chars().take(1200).collect();
        let doc_vec = embedder.embed_document(&query_slice, None)?;
        let query_vec = embedder.embed_query(&query_slice)?;

        let slots: Vec<Option<usize>> = chunks.iter().map(|c| c.vector_slot).collect();
        let doc_scores = store.vectors.score_slots(&doc_vec, &slots);
        let q_scores = store.vectors.score_slots(&query_vec, &slots);

        let mut hits = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let sem = (doc_scores[i].max(q_scores[i]) as f64).clamp(0.0, 1.0);
            let lex = SearchText::lexical_containment(&query_slice, &chunk.text_content);

            let confidence = if lex >= 0.95 {
                (0.75 * lex + 0.25 * sem).max(0.95)
            } else {
                (0.55 * lex + 0.45 * sem).max(sem)
            };

            if confidence < 0.25 {
                continue;
            }

            let verdict = if lex >= 0.92 || sem >= 0.95 {
                ContainmentVerdict::ExactPart
            } else if lex >= 0.65 || confidence >= 0.78 {
                ContainmentVerdict::NearDuplicatePart
            } else if sem >= 0.55 || lex >= 0.40 {
                ContainmentVerdict::SemanticPart
            } else {
                ContainmentVerdict::RelatedMatch
            };

            hits.push(build_containment_hit(
                store,
                chunk,
                verdict,
                confidence,
                sem,
                None,
                Some(lex),
                None,
                &SearchText::snippet(&chunk.text_content, &query_slice, 180),
            ));
        }

        dedup_and_sort_hits(hits, limit)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_containment_hit(
    store: &IndexStore,
    chunk: &IndexedChunkRow,
    verdict: ContainmentVerdict,
    confidence: f64,
    semantic_similarity: f64,
    perceptual_hash_similarity: Option<f64>,
    lexical_containment: Option<f64>,
    temporal_alignment: Option<f64>,
    snippet: &str,
) -> ContainmentHit {
    let abs_path = store
        .root()
        .join(&chunk.relative_path)
        .to_string_lossy()
        .to_string();
    let location_summary = chunk
        .origin
        .format_label()
        .unwrap_or_else(|| "Full file".to_string());

    ContainmentHit {
        path: abs_path,
        relative_path: chunk.relative_path.clone(),
        file_name: chunk.file_name.clone(),
        kind: chunk.kind,
        modality: chunk.modality,
        verdict,
        confidence: confidence.clamp(0.0, 1.0),
        semantic_similarity,
        perceptual_hash_similarity,
        lexical_containment,
        temporal_alignment,
        origin: chunk.origin.clone(),
        location_summary,
        snippet: snippet.to_string(),
    }
}

fn dedup_and_sort_hits(hits: Vec<ContainmentHit>, limit: usize) -> Result<Vec<ContainmentHit>> {
    let mut best_by_path: HashMap<String, ContainmentHit> = HashMap::new();
    for h in hits {
        match best_by_path.get(&h.relative_path) {
            Some(prev) if prev.confidence >= h.confidence => {}
            _ => {
                best_by_path.insert(h.relative_path.clone(), h);
            }
        }
    }
    let mut out: Vec<ContainmentHit> = best_by_path.into_values().collect();
    out.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(if limit == 0 { 10 } else { limit });
    Ok(out)
}
