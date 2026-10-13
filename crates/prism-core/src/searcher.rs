use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use prism_asr::SemanticAsrEngine;
use prism_llama::{ImageSizing, LlamaEmbedder};

use crate::model::{
    ChunkModality, SearchFilter, SearchHit, SearchKind, SearchMode, SearchResponse,
};
use crate::rules::classify_path;
use crate::search_text::SearchText;
use crate::store::{IndexStore, IndexedChunkRow};
use crate::vectors::compute_modality_z_scores;

pub struct Searcher {
    store: IndexStore,
}

impl Searcher {
    pub fn open(root: &Path) -> Result<Self> {
        let store = IndexStore::open(root)?;
        Ok(Self { store })
    }

    pub fn store(&self) -> &IndexStore {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut IndexStore {
        &mut self.store
    }

    /// Perform hybrid, semantic, or lexical search for a text query.
    pub fn search_text(
        &self,
        embedder: Option<&LlamaEmbedder>,
        query: &str,
        mode: SearchMode,
        filter: &SearchFilter,
    ) -> Result<SearchResponse> {
        let start = Instant::now();
        let query_vec = if mode != SearchMode::Lexical {
            if let Some(emb) = embedder {
                Some(emb.embed_query(query)?)
            } else {
                None
            }
        } else {
            None
        };

        let hits = self.rank_chunks(query, query_vec.as_deref(), mode, filter)?;
        Ok(SearchResponse {
            query: query.to_string(),
            mode,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            decoded_audio_transcript: None,
            distilled_audio_keywords: Vec::new(),
            hits,
        })
    }

    /// Perform spoken audio / voice query search exploiting EmbeddingGemma 2's 768-D semantic space:
    /// 1. Direct 768-D Audio-to-Multimodal Vector Search across Text, Image, PDF, Code, Audio, and Video chunks
    /// 2. Sub-Window Semantic Lexicon Decoding (`SemanticAsrEngine`)
    /// 3. Semantic Pseudo-Relevance Keyword Distillation from top semantic index matches for FTS5 hybrid ranking.
    pub fn search_audio_file(
        &self,
        embedder: &LlamaEmbedder,
        asr_engine: &SemanticAsrEngine,
        audio_path: &Path,
        verbatim: bool,
        text_hint: Option<&str>,
        filter: &SearchFilter,
    ) -> Result<SearchResponse> {
        let start = Instant::now();
        let mut asr = asr_engine.decode_file(embedder, audio_path, verbatim)?;
        self.finish_audio_search(embedder, asr_engine, &mut asr, text_hint, filter, start)
    }

    pub fn search_audio_samples(
        &self,
        embedder: &LlamaEmbedder,
        asr_engine: &SemanticAsrEngine,
        samples_16k: &[f32],
        text_hint: Option<&str>,
        filter: &SearchFilter,
    ) -> Result<SearchResponse> {
        let start = Instant::now();
        let mut asr = asr_engine.decode_samples(embedder, samples_16k)?;
        self.finish_audio_search(embedder, asr_engine, &mut asr, text_hint, filter, start)
    }

    fn finish_audio_search(
        &self,
        embedder: &LlamaEmbedder,
        asr_engine: &SemanticAsrEngine,
        asr: &mut prism_asr::SemanticAsrResult,
        text_hint: Option<&str>,
        filter: &SearchFilter,
        start: Instant,
    ) -> Result<SearchResponse> {
        // Step 1: First-pass direct 768-D audio-to-multimodal semantic search
        let mut probe_filter = filter.clone();
        probe_filter.limit = 8;
        let initial_hits = self.rank_chunks(
            &asr.effective_query(),
            Some(&asr.audio_embedding),
            SearchMode::Semantic,
            &probe_filter,
        )?;

        // Step 2: Semantic Pseudo-Relevance Keyword Distillation over top candidate hits
        let candidate_texts: Vec<String> = initial_hits
            .iter()
            .take(6)
            .map(|h| format!("{} {}", h.file_name, h.snippet))
            .collect();
        if !candidate_texts.is_empty() {
            if let Ok(prf_keywords) =
                asr_engine.distill_corpus_keywords(embedder, asr, &candidate_texts, 8)
            {
                // Merge high-Z PRF keywords into distilled_keywords & semantic_transcript
                let mut existing: HashSet<String> = asr
                    .distilled_keywords
                    .iter()
                    .map(|k| k.term.clone())
                    .collect();
                for kw in prf_keywords {
                    if kw.z_score >= 1.2 && existing.insert(kw.term.clone()) {
                        asr.distilled_keywords.push(kw);
                    }
                }
                asr.distilled_keywords.sort_by(|a, b| {
                    b.z_score
                        .partial_cmp(&a.z_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                asr.distilled_keywords.truncate(10);
            }
        }

        let top_kw_str = asr
            .distilled_keywords
            .iter()
            .take(5)
            .map(|k| k.term.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let effective_query = match text_hint {
            Some(h) if !h.trim().is_empty() => format!("{} {}", h.trim(), top_kw_str),
            _ => {
                if !top_kw_str.is_empty() {
                    top_kw_str
                } else {
                    asr.effective_query()
                }
            }
        };

        // Blend direct audio 768-D vector (75%) with distilled semantic query vector (25%) for rock-solid multimodal retrieval
        let mut query_vec = asr.audio_embedding.clone();
        if !effective_query.trim().is_empty() {
            if let Ok(txt_vec) = embedder.embed_query(&effective_query) {
                for (a, &t) in query_vec.iter_mut().zip(txt_vec.iter()) {
                    *a = 0.75 * (*a) + 0.25 * t;
                }
                let norm: f32 = query_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut query_vec {
                        *x /= norm;
                    }
                }
            }
        }

        let hits = self.rank_chunks(
            &effective_query,
            Some(&query_vec),
            SearchMode::Hybrid,
            filter,
        )?;

        let display_transcript = asr
            .verbatim_transcript
            .clone()
            .unwrap_or_else(|| effective_query.clone());

        Ok(SearchResponse {
            query: effective_query,
            mode: SearchMode::Hybrid,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            decoded_audio_transcript: Some(display_transcript),
            distilled_audio_keywords: asr.distilled_keywords.clone(),
            hits,
        })
    }

    /// Search the index using any file (Image, Video, Audio, Document, Text) as the query input.
    pub fn search_by_file(
        &self,
        embedder: &LlamaEmbedder,
        asr_engine: &SemanticAsrEngine,
        input_path: &Path,
        filter: &SearchFilter,
    ) -> Result<SearchResponse> {
        let start = Instant::now();
        let kind = classify_path(input_path).unwrap_or(SearchKind::Text);
        let file_name = input_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "query".to_string());

        match kind {
            SearchKind::Audio => {
                self.search_audio_file(embedder, asr_engine, input_path, false, None, filter)
            }
            SearchKind::Image => {
                let (vec, _ahash) =
                    embedder.embed_image_path(input_path, ImageSizing::PHOTO_BUDGET)?;
                let hits = self.rank_chunks(&file_name, Some(&vec), SearchMode::Semantic, filter)?;
                Ok(SearchResponse {
                    query: format!("[Image: {}]", file_name),
                    mode: SearchMode::Semantic,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                    decoded_audio_transcript: None,
                    distilled_audio_keywords: Vec::new(),
                    hits,
                })
            }
            _ => {
                let text = std::fs::read_to_string(input_path).unwrap_or_else(|_| file_name.clone());
                let snippet: String = text.chars().take(600).collect();
                let vec = embedder.embed_query(&snippet)?;
                let hits = self.rank_chunks(&snippet, Some(&vec), SearchMode::Hybrid, filter)?;
                Ok(SearchResponse {
                    query: format!("[File: {}]", file_name),
                    mode: SearchMode::Hybrid,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                    decoded_audio_transcript: None,
                    distilled_audio_keywords: Vec::new(),
                    hits,
                })
            }
        }
    }

    fn rank_chunks(
        &self,
        query: &str,
        query_vec: Option<&[f32]>,
        mode: SearchMode,
        filter: &SearchFilter,
    ) -> Result<Vec<SearchHit>> {
        let chunks = self.store.load_all_chunks()?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Semantic scoring + Modality-Centered Z-score calibration
        let (raw_scores, z_scores) = if let Some(qvec) = query_vec {
            let slots: Vec<Option<usize>> = chunks.iter().map(|c| c.vector_slot).collect();
            let raw = self.store.vectors.score_slots(qvec, &slots);
            let modalities: Vec<ChunkModality> = chunks.iter().map(|c| c.modality).collect();
            let sets_scale: Vec<bool> = chunks.iter().map(|c| c.sets_scale).collect();
            let z = compute_modality_z_scores(&raw, &modalities, &sets_scale);
            (raw, z)
        } else {
            (vec![0.0; chunks.len()], vec![0.0; chunks.len()])
        };

        // 2. Lexical FTS5 + token coverage scoring
        let query_tokens = SearchText::fts_tokens(query);
        let fts_bm25 = if mode != SearchMode::Semantic && !query_tokens.is_empty() {
            self.store.query_fts_matches(query).unwrap_or_default()
        } else {
            HashMap::new()
        };

        let kind_hints = detect_kind_hints(&query_tokens);

        let (meaning_weight, words_weight) = match mode {
            SearchMode::Semantic => (1.0, 0.0),
            SearchMode::Lexical => (0.0, 4.0),
            SearchMode::Hybrid => (1.0, 2.2),
        };

        // Deduplicate by file_id (keep highest-scoring chunk per file)
        let mut best_by_file: HashMap<i64, SearchHit> = HashMap::new();

        for (i, chunk) in chunks.iter().enumerate() {
            if !filter.kinds.is_empty() && !filter.kinds.contains(&chunk.kind) {
                continue;
            }
            if let Some(ref prefix) = filter.path_prefix {
                if !chunk.relative_path.starts_with(prefix) {
                    continue;
                }
            }

            let z = z_scores[i];
            let raw_cos = raw_scores[i] as f64;
            let coverage = compute_lexical_coverage(&query_tokens, chunk, fts_bm25.get(&chunk.chunk_id).copied());

            let hint_bonus = if kind_hints.contains(&chunk.kind) && z > 0.5 {
                0.85
            } else {
                0.0
            };

            let score = meaning_weight * z.max(0.0)
                + words_weight * coverage.powf(1.5)
                + hint_bonus;

            if mode == SearchMode::Lexical && coverage <= 0.0 {
                continue;
            }
            if score <= 0.05 && raw_cos <= 0.05 {
                continue;
            }

            let abs_path = self
                .store
                .root()
                .join(&chunk.relative_path)
                .to_string_lossy()
                .to_string();

            let hit = SearchHit {
                file_id: chunk.file_id,
                chunk_id: chunk.chunk_id,
                path: abs_path,
                relative_path: chunk.relative_path.clone(),
                file_name: chunk.file_name.clone(),
                kind: chunk.kind,
                modality: chunk.modality,
                score,
                semantic_z: z,
                raw_cosine: raw_cos,
                lexical_coverage: coverage,
                snippet: SearchText::snippet(&chunk.text_content, query, 180),
                origin: chunk.origin.clone(),
                modified_unix: chunk.modified_unix,
                size_bytes: chunk.size_bytes,
            };

            match best_by_file.get(&chunk.file_id) {
                Some(prev) if prev.score >= hit.score => {}
                _ => {
                    best_by_file.insert(chunk.file_id, hit);
                }
            }
        }

        let mut hits: Vec<SearchHit> = best_by_file.into_values().collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let limit = if filter.limit == 0 { 20 } else { filter.limit };
        hits.truncate(limit);
        Ok(hits)
    }
}

fn compute_lexical_coverage(
    query_tokens: &[String],
    chunk: &IndexedChunkRow,
    bm25: Option<f64>,
) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let hay = SearchText::normalized(&format!(
        "{} {} {}",
        chunk.file_name, chunk.relative_path, chunk.text_content
    ))
    .to_lowercase();

    let mut matched = 0usize;
    for tok in query_tokens {
        if hay.contains(tok.as_str()) {
            matched += 1;
        }
    }
    let ratio = (matched as f64) / (query_tokens.len() as f64);
    let bm25_boost = bm25.map(|b| (b / 10.0).clamp(0.0, 0.35)).unwrap_or(0.0);
    (ratio + bm25_boost).clamp(0.0, 1.25)
}

fn detect_kind_hints(tokens: &[String]) -> HashSet<SearchKind> {
    let mut set = HashSet::new();
    for t in tokens {
        match t.as_str() {
            "photo" | "photos" | "image" | "images" | "picture" | "pictures" | "screenshot" => {
                set.insert(SearchKind::Image);
            }
            "audio" | "voice" | "speech" | "recording" | "podcast" | "song" | "music" => {
                set.insert(SearchKind::Audio);
            }
            "video" | "videos" | "clip" | "movie" | "footage" => {
                set.insert(SearchKind::Video);
            }
            "pdf" | "document" | "paper" | "report" | "contract" | "invoice" => {
                set.insert(SearchKind::Document);
            }
            "spreadsheet" | "sheet" | "excel" | "csv" | "table" | "budget" => {
                set.insert(SearchKind::Spreadsheet);
            }
            "slide" | "slides" | "presentation" | "deck" => {
                set.insert(SearchKind::Presentation);
            }
            "code" | "function" | "struct" | "class" | "script" => {
                set.insert(SearchKind::Code);
            }
            _ => {}
        }
    }
    set
}
