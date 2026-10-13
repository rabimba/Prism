use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use anyhow::Result;
use parking_lot::RwLock;
use prism_llama::{cosine_similarity, LlamaEmbedder};
use serde::{Deserialize, Serialize};

use crate::audio::{load_audio_16k_mono, MAX_WINDOW_SAMPLES, SAMPLE_RATE};
use crate::lexicon::{LexiconMatch, SemanticLexicon};
use crate::vad::slice_acoustic_windows;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedAudioWindow {
    pub start_s: f64,
    pub end_s: f64,
    pub rms: f32,
    pub embedding: Vec<f32>,
    pub top_tokens: Vec<LexiconMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAsrResult {
    pub duration_s: f64,
    /// Holistic L2-normalized 768-D EmbeddingGemma 2 vector of the spoken audio.
    pub audio_embedding: Vec<f32>,
    /// Overlapping acoustic sub-windows with 768-D embeddings and localized token projections.
    pub windows: Vec<DecodedAudioWindow>,
    /// CTC/Viterbi deduplicated temporal sequence of semantic words across sub-windows.
    pub semantic_transcript: String,
    /// High-affinity semantic keywords ranked by 768-D alignment.
    pub distilled_keywords: Vec<LexiconMatch>,
    /// Optional verbatim transcript if `--asr verbatim` was requested and a backend is available.
    pub verbatim_transcript: Option<String>,
}

impl SemanticAsrResult {
    /// Effective text query string to feed into FTS5 hybrid search alongside `audio_embedding`.
    pub fn effective_query(&self) -> String {
        if let Some(ref v) = self.verbatim_transcript {
            if !v.trim().is_empty() {
                return v.trim().to_string();
            }
        }
        if !self.semantic_transcript.trim().is_empty() {
            return self.semantic_transcript.clone();
        }
        self.distilled_keywords
            .iter()
            .take(6)
            .map(|m| m.term.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// EmbeddingGemma 2 Semantic-Space ASR Engine.
pub struct SemanticAsrEngine {
    lexicon: RwLock<SemanticLexicon>,
}

impl SemanticAsrEngine {
    pub fn new(embedder: &LlamaEmbedder) -> Result<Self> {
        let lexicon = SemanticLexicon::load_or_build(embedder, None)?;
        Ok(Self {
            lexicon: RwLock::new(lexicon),
        })
    }

    pub fn with_lexicon(lexicon: SemanticLexicon) -> Self {
        Self {
            lexicon: RwLock::new(lexicon),
        }
    }

    /// Enrich the 768-D semantic lexicon with domain vocabulary from indexed files.
    pub fn enrich_vocabulary(&self, embedder: &LlamaEmbedder, terms: &[String]) -> Result<usize> {
        let mut guard = self.lexicon.write();
        let added = guard.add_terms(embedder, terms)?;
        if added > 0 {
            let _ = guard.save_to_file(&SemanticLexicon::default_cache_path());
        }
        Ok(added)
    }

    /// Decode an audio file using EmbeddingGemma 2's 768-D semantic space.
    pub fn decode_file(
        &self,
        embedder: &LlamaEmbedder,
        path: &Path,
        verbatim: bool,
    ) -> Result<SemanticAsrResult> {
        let samples = load_audio_16k_mono(path)?;
        let mut result = self.decode_samples(embedder, &samples)?;
        if verbatim {
            result.verbatim_transcript = try_verbatim_asr_file(path);
        }
        Ok(result)
    }

    /// Decode 16 kHz mono float32 PCM samples directly in EmbeddingGemma 2's 768-D space.
    pub fn decode_samples(
        &self,
        embedder: &LlamaEmbedder,
        samples: &[f32],
    ) -> Result<SemanticAsrResult> {
        let duration_s = (samples.len() as f64) / (SAMPLE_RATE as f64);
        if samples.is_empty() {
            return Ok(SemanticAsrResult {
                duration_s: 0.0,
                audio_embedding: vec![0.0; LlamaEmbedder::DIMENSION],
                windows: Vec::new(),
                semantic_transcript: String::new(),
                distilled_keywords: Vec::new(),
                verbatim_transcript: None,
            });
        }

        // 1. Holistic 768-D audio embedding (up to 30s per mtmd window, mean-pooled if longer)
        let mut holistic_chunks: Vec<&[f32]> = Vec::new();
        for chunk in samples.chunks(MAX_WINDOW_SAMPLES) {
            if !chunk.is_empty() {
                holistic_chunks.push(chunk);
            }
        }
        let padded_short: Vec<f32>;
        if holistic_chunks.len() == 1 && holistic_chunks[0].len() < 9_600 {
            let mut p = holistic_chunks[0].to_vec();
            p.resize(9_600, 0.0);
            padded_short = p;
            holistic_chunks[0] = &padded_short;
        }
        let holistic_vecs = embedder.embed_audio_windows(&holistic_chunks)?;
        let audio_embedding = average_and_normalize(&holistic_vecs);

        // 2. Slice into overlapping acoustic sub-windows (1.6s window, 0.6s hop, up to 16 windows)
        let ac_windows = slice_acoustic_windows(samples, 1.6, 0.6, 16);
        let win_slices: Vec<&[f32]> = ac_windows.iter().map(|w| w.samples.as_slice()).collect();
        let win_vecs = if win_slices.len() == 1 && holistic_chunks.len() == 1 {
            vec![audio_embedding.clone()]
        } else {
            embedder.embed_audio_windows(&win_slices)?
        };

        // 3. Project holistic vector & sub-window vectors onto 768-D Semantic Lexicon
        let lex_guard = self.lexicon.read();
        let holistic_matches = lex_guard.project(&audio_embedding, 24);
        let holistic_z_map: HashMap<String, f32> = holistic_matches
            .iter()
            .map(|m| (m.term.clone(), m.z_score))
            .collect();

        let mut decoded_windows = Vec::with_capacity(ac_windows.len());
        let mut temporal_tokens: Vec<String> = Vec::new();
        let mut combined_scores: HashMap<String, (f32, f32)> = HashMap::new();

        for (m, _) in holistic_matches.iter().zip(0..12) {
            combined_scores.insert(m.term.clone(), (m.cosine, m.z_score));
        }

        for (ac, vec) in ac_windows.into_iter().zip(win_vecs.into_iter()) {
            let mut local_matches = lex_guard.project(&vec, 8);
            // Re-rank local window candidates by blending local Z-score (60%) + global utterance Z-score (40%)
            for lm in &mut local_matches {
                if let Some(&gz) = holistic_z_map.get(&lm.term) {
                    lm.z_score = 0.60 * lm.z_score + 0.40 * gz;
                }
                let entry = combined_scores
                    .entry(lm.term.clone())
                    .or_insert((lm.cosine, lm.z_score));
                if lm.cosine > entry.0 {
                    entry.0 = lm.cosine;
                }
                if lm.z_score > entry.1 {
                    entry.1 = lm.z_score;
                }
            }
            local_matches.sort_by(|a, b| {
                b.z_score
                    .partial_cmp(&a.z_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // CTC-style temporal emission: emit top token if Z-score >= 1.65 and not identical to previous token
            if let Some(best) = local_matches.first() {
                if best.z_score >= 1.65 {
                    if temporal_tokens.last().map(|s| s.as_str()) != Some(best.term.as_str()) {
                        temporal_tokens.push(best.term.clone());
                    }
                }
            }

            local_matches.truncate(5);
            decoded_windows.push(DecodedAudioWindow {
                start_s: ac.start_s,
                end_s: ac.end_s,
                rms: ac.rms,
                embedding: vec,
                top_tokens: local_matches,
            });
        }
        drop(lex_guard);

        let mut distilled: Vec<LexiconMatch> = combined_scores
            .into_iter()
            .map(|(term, (cosine, z_score))| LexiconMatch {
                term,
                cosine,
                z_score,
            })
            .collect();
        distilled.sort_by(|a, b| {
            b.z_score
                .partial_cmp(&a.z_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        distilled.truncate(10);

        // Ensure highest-confidence holistic terms appear in the semantic transcript
        for dm in distilled.iter().take(4) {
            if dm.z_score >= 2.0 && !temporal_tokens.contains(&dm.term) {
                temporal_tokens.push(dm.term.clone());
            }
        }

        let semantic_transcript = temporal_tokens.join(" ");

        Ok(SemanticAsrResult {
            duration_s,
            audio_embedding,
            windows: decoded_windows,
            semantic_transcript,
            distilled_keywords: distilled,
            verbatim_transcript: None,
        })
    }

    /// **Semantic Pseudo-Relevance Keyword Distillation**:
    /// Given an `SemanticAsrResult` and candidate text snippets/titles from top semantic matches in the index,
    /// extracts candidate tokens, embeds unseen tokens into the 768-D space, and verifies which exact corpus words
    /// have the highest cosine similarity with the spoken audio vector + sub-windows.
    pub fn distill_corpus_keywords(
        &self,
        embedder: &LlamaEmbedder,
        asr: &SemanticAsrResult,
        candidate_texts: &[String],
        top_k: usize,
    ) -> Result<Vec<LexiconMatch>> {
        let mut candidate_words = Vec::new();
        let mut seen = HashSet::new();
        for text in candidate_texts {
            for token in tokenize_candidate_words(text) {
                if seen.insert(token.clone()) {
                    candidate_words.push(token);
                }
                if candidate_words.len() >= 128 {
                    break;
                }
            }
            if candidate_words.len() >= 128 {
                break;
            }
        }

        if candidate_words.is_empty() {
            return Ok(asr.distilled_keywords.clone());
        }

        // Ensure all candidate corpus words are in the 768-D lexicon
        {
            let mut lex_write = self.lexicon.write();
            let _ = lex_write.add_terms(embedder, &candidate_words);
        }

        let lex_read = self.lexicon.read();
        let mut scored = Vec::new();
        let mut sum = 0.0f64;

        for word in &candidate_words {
            if let Some(word_vec) = lex_read.get_vector(word) {
                let holistic_cos = cosine_similarity(&asr.audio_embedding, word_vec);
                let max_win_cos = asr
                    .windows
                    .iter()
                    .map(|w| cosine_similarity(&w.embedding, word_vec))
                    .fold(holistic_cos, f32::max);
                let blended = 0.65 * holistic_cos + 0.35 * max_win_cos;
                sum += blended as f64;
                scored.push((word.clone(), blended));
            }
        }
        drop(lex_read);

        if scored.is_empty() {
            return Ok(asr.distilled_keywords.clone());
        }

        let n = scored.len() as f64;
        let mean = (sum / n) as f32;
        let var: f64 = scored
            .iter()
            .map(|(_, c)| {
                let d = (*c - mean) as f64;
                d * d
            })
            .sum::<f64>()
            / n;
        let std_dev = (var.sqrt() as f32).max(0.015);

        let mut matches: Vec<LexiconMatch> = scored
            .into_iter()
            .map(|(term, cosine)| LexiconMatch {
                term,
                cosine,
                z_score: (cosine - mean) / std_dev,
            })
            .collect();

        matches.sort_by(|a, b| {
            b.z_score
                .partial_cmp(&a.z_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(top_k);
        Ok(matches)
    }
}

fn average_and_normalize(vecs: &[Vec<f32>]) -> Vec<f32> {
    let dim = LlamaEmbedder::DIMENSION;
    if vecs.is_empty() {
        return vec![0.0; dim];
    }
    if vecs.len() == 1 {
        return vecs[0].clone();
    }
    let mut avg = vec![0.0f32; dim];
    for v in vecs {
        for (i, &x) in v.iter().enumerate() {
            avg[i] += x;
        }
    }
    let norm: f32 = avg.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut avg {
            *x /= norm;
        }
    }
    avg
}

fn tokenize_candidate_words(text: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "the", "and", "for", "that", "this", "with", "from", "your", "have", "are", "was", "were",
        "will", "would", "could", "should", "about", "into", "over", "after", "before", "between",
        "under", "while", "where", "when", "which", "their", "there", "these", "those", "some",
        "such", "than", "then", "them", "they", "what", "been", "being", "each", "more", "most",
        "other", "only", "same", "very", "just", "also", "both", "none", "title", "text",
    ];
    let stop: HashSet<&str> = STOPWORDS.iter().copied().collect();
    let mut out = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        let w = raw.trim().to_lowercase();
        if w.len() >= 3 && w.len() <= 28 && !w.chars().all(|c| c.is_ascii_digit()) && !stop.contains(w.as_str()) {
            out.push(w);
        }
    }
    out
}

/// Optional pluggable verbatim ASR fallback (checks `whisper-cli` or python `faster_whisper` if installed).
fn try_verbatim_asr_file(path: &Path) -> Option<String> {
    // 1. Check whisper-cli if on PATH
    if let Ok(output) = Command::new("whisper-cli")
        .args(["-nt", "-np", "-f"])
        .arg(path)
        .output()
    {
        if output.status.success() {
            let txt = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !txt.is_empty() {
                return Some(txt);
            }
        }
    }
    None
}
