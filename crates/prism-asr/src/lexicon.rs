use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use prism_llama::{cosine_similarity, LlamaEmbedder, ModelFiles};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconMatch {
    pub term: String,
    pub cosine: f32,
    pub z_score: f32,
}

#[derive(Serialize, Deserialize)]
struct CachedLexiconFile {
    version: u32,
    terms: Vec<String>,
    /// Flattened 768-D vectors encoded as f16 bits (`u16`) to keep cache compact (~1 MB).
    vectors_f16: Vec<u16>,
}

/// 768-D EmbeddingGemma 2 Semantic Lexicon for zero-transcription Semantic-Space ASR
/// and cross-modal keyword distillation.
pub struct SemanticLexicon {
    terms: Vec<String>,
    term_index: HashMap<String, usize>,
    vectors: Vec<Vec<f32>>,
}

impl SemanticLexicon {
    pub const CACHE_VERSION: u32 = 1;
    pub const CACHE_FILENAME: &'static str = "prism_semantic_lexicon_v1.bin";

    pub fn default_cache_path() -> PathBuf {
        ModelFiles::default_directory().join(Self::CACHE_FILENAME)
    }

    /// Load the cached 768-D semantic lexicon from disk, or build it using `embedder` if not yet cached.
    pub fn load_or_build(embedder: &LlamaEmbedder, cache_path: Option<&Path>) -> Result<Self> {
        let path = cache_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(Self::default_cache_path);

        if path.exists() {
            if let Ok(lex) = Self::load_from_file(&path) {
                if !lex.terms.is_empty() {
                    return Ok(lex);
                }
            }
        }

        let core_terms: Vec<String> = CORE_VOCABULARY
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let mut lex = Self::empty();
        lex.add_terms(embedder, &core_terms)?;
        let _ = lex.save_to_file(&path);
        Ok(lex)
    }

    pub fn empty() -> Self {
        Self {
            terms: Vec::new(),
            term_index: HashMap::new(),
            vectors: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Embed new terms into the 768-D semantic space using `LlamaEmbedder::query_prompt`
    /// (the trained dual of `mtmd` audio/image embeddings in EmbeddingGemma 2).
    pub fn add_terms(&mut self, embedder: &LlamaEmbedder, candidate_terms: &[String]) -> Result<usize> {
        let mut new_terms = Vec::new();
        let mut seen = HashSet::new();
        for raw in candidate_terms {
            let clean = raw.trim().to_lowercase();
            if clean.len() < 2 || clean.len() > 48 {
                continue;
            }
            if !self.term_index.contains_key(&clean) && seen.insert(clean.clone()) {
                new_terms.push(clean);
            }
        }

        if new_terms.is_empty() {
            return Ok(0);
        }

        let prompts: Vec<String> = new_terms
            .iter()
            .map(|t| LlamaEmbedder::query_prompt(t))
            .collect();

        // Batch in chunks of 64 to keep memory bounded
        let mut added = 0usize;
        for (term_chunk, prompt_chunk) in new_terms.chunks(64).zip(prompts.chunks(64)) {
            let vecs = embedder.embed_texts(prompt_chunk)?;
            for (term, vec) in term_chunk.iter().zip(vecs.into_iter()) {
                let idx = self.terms.len();
                self.term_index.insert(term.clone(), idx);
                self.terms.push(term.clone());
                self.vectors.push(vec);
                added += 1;
            }
        }

        Ok(added)
    }

    /// Project a 768-D audio (or image/multimodal) vector onto the semantic lexicon,
    /// returning the top-K terms ranked by hubness-corrected Z-score and cosine similarity.
    pub fn project(&self, audio_vec: &[f32], top_k: usize) -> Vec<LexiconMatch> {
        if self.terms.is_empty() || audio_vec.len() != LlamaEmbedder::DIMENSION || top_k == 0 {
            return Vec::new();
        }

        let mut raw_scores = Vec::with_capacity(self.terms.len());
        let mut sum = 0.0f64;
        for vec in &self.vectors {
            let c = cosine_similarity(audio_vec, vec);
            raw_scores.push(c);
            sum += c as f64;
        }

        let n = self.terms.len() as f64;
        let mean = (sum / n) as f32;
        let mut var_sum = 0.0f64;
        for &c in &raw_scores {
            let d = (c - mean) as f64;
            var_sum += d * d;
        }
        let std_dev = ((var_sum / n).sqrt() as f32).max(0.015);

        let mut matches: Vec<LexiconMatch> = self
            .terms
            .iter()
            .zip(raw_scores.into_iter())
            .map(|(term, cosine)| LexiconMatch {
                term: term.clone(),
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
        matches
    }

    pub fn get_vector(&self, term: &str) -> Option<&[f32]> {
        let idx = *self.term_index.get(&term.to_lowercase())?;
        self.vectors.get(idx).map(|v| v.as_slice())
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let mut vectors_f16 = Vec::with_capacity(self.vectors.len() * LlamaEmbedder::DIMENSION);
        for v in &self.vectors {
            for &x in v {
                vectors_f16.push(half::f16::from_f32(x).to_bits());
            }
        }
        let payload = CachedLexiconFile {
            version: Self::CACHE_VERSION,
            terms: self.terms.clone(),
            vectors_f16,
        };
        let bytes = serde_json::to_vec(&payload)?;
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let payload: CachedLexiconFile = serde_json::from_slice(&bytes)?;
        if payload.version != Self::CACHE_VERSION {
            anyhow::bail!("Incompatible lexicon cache version");
        }
        let dim = LlamaEmbedder::DIMENSION;
        if payload.vectors_f16.len() != payload.terms.len() * dim {
            anyhow::bail!("Corrupt lexicon cache size");
        }
        let mut vectors = Vec::with_capacity(payload.terms.len());
        let mut term_index = HashMap::with_capacity(payload.terms.len());
        for (i, term) in payload.terms.iter().enumerate() {
            term_index.insert(term.clone(), i);
            let slice = &payload.vectors_f16[i * dim..(i + 1) * dim];
            let vec: Vec<f32> = slice
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect();
            vectors.push(vec);
        }
        Ok(Self {
            terms: payload.terms,
            term_index,
            vectors,
        })
    }
}

pub const CORE_VOCABULARY: &[&str] = &[
    // Nature, geography & photography
    "mountain", "lake", "landscape", "alpine", "snow", "peaks", "forest", "ocean", "beach",
    "sunset", "sunrise", "river", "valley", "desert", "canyon", "island", "waterfall", "clouds",
    "sky", "trees", "flowers", "garden", "wildlife", "bird", "dog", "cat", "animal", "portrait",
    "photograph", "camera", "panorama", "reflection", "horizon", "winter", "summer", "autumn",
    "spring", "rain", "storm", "weather", "outdoor", "travel", "vacation", "hiking", "camping",
    // Documents, business & finance
    "document", "report", "summary", "invoice", "receipt", "contract", "agreement", "proposal",
    "presentation", "slides", "spreadsheet", "budget", "finance", "revenue", "quarterly",
    "annual", "profit", "expense", "tax", "deduction", "accounting", "balance", "statement",
    "schedule", "roadmap", "timeline", "meeting", "minutes", "notes", "memo", "letter", "resume",
    "curriculum", "vitae", "certificate", "license", "passport", "identity", "medical", "record",
    "prescription", "insurance", "policy", "legal", "patent", "trademark", "copyright",
    // Software engineering, systems & AI
    "code", "source", "function", "method", "class", "struct", "interface", "module", "package",
    "library", "framework", "algorithm", "database", "query", "index", "table", "schema",
    "migration", "server", "client", "api", "endpoint", "request", "response", "authentication",
    "security", "encryption", "password", "token", "certificate", "terminal", "console", "shell",
    "command", "script", "compiler", "binary", "executable", "process", "thread", "memory",
    "performance", "benchmark", "latency", "throughput", "error", "exception", "bug", "debug",
    "test", "unit", "integration", "deployment", "container", "docker", "kubernetes", "cloud",
    "linux", "macos", "windows", "android", "ios", "rust", "python", "swift", "javascript",
    "typescript", "golang", "java", "cpp", "html", "css", "json", "yaml", "markdown", "sql",
    "git", "repository", "commit", "branch", "merge", "patch", "neural", "network", "model",
    "embedding", "vector", "semantic", "search", "retrieval", "ranking", "transformer", "attention",
    "quantization", "inference", "training", "dataset", "tokenization", "multimodal", "vision",
    // Audio, speech & music
    "audio", "speech", "voice", "recording", "podcast", "interview", "conversation", "dialogue",
    "lecture", "keynote", "webinar", "conference", "discussion", "announcement", "narration",
    "audiobook", "music", "song", "melody", "instrument", "piano", "guitar", "drums", "orchestra",
    "concert", "acoustic", "soundtrack", "microphone", "speaker", "lyrics", "transcript",
    // Video, screen & UI
    "video", "clip", "movie", "film", "animation", "screencast", "tutorial", "demo", "walkthrough",
    "footage", "frame", "scene", "subtitle", "caption", "screenshot", "window", "browser",
    "website", "webpage", "dashboard", "button", "menu", "dialog", "settings", "configuration",
    "diagram", "flowchart", "architecture", "chart", "graph", "plot", "histogram", "heatmap",
    "table", "figure", "illustration", "drawing", "sketch", "wireframe", "mockup", "logo", "icon",
    // Science, research & education
    "research", "paper", "article", "journal", "publication", "thesis", "dissertation", "abstract",
    "introduction", "methodology", "experiment", "results", "discussion", "conclusion",
    "references", "citation", "bibliography", "mathematics", "equation", "formula", "theorem",
    "proof", "statistics", "probability", "physics", "chemistry", "biology", "medicine", "genome",
    "climate", "energy", "battery", "hardware", "processor", "gpu", "npu", "sensor", "robotics",
    // Common spoken search verbs & modifiers
    "find", "search", "show", "open", "locate", "match", "contain", "verify", "check", "compare",
    "recent", "latest", "first", "last", "important", "draft", "final", "signed", "scanned",
    "color", "black", "white", "blue", "green", "red", "yellow", "dark", "bright", "large", "small",
];
