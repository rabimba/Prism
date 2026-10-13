use std::collections::BTreeMap;
use std::path::PathBuf;

use prism_asr::LexiconMatch;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchKind {
    Text,
    Code,
    Document,
    Spreadsheet,
    Presentation,
    Image,
    Audio,
    Video,
    Archive,
}

impl SearchKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Code => "code",
            Self::Document => "document",
            Self::Spreadsheet => "spreadsheet",
            Self::Presentation => "presentation",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Archive => "archive",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "text" | "txt" | "md" => Some(Self::Text),
            "code" | "src" => Some(Self::Code),
            "document" | "doc" | "pdf" => Some(Self::Document),
            "spreadsheet" | "sheet" | "xls" | "csv" => Some(Self::Spreadsheet),
            "presentation" | "slides" | "ppt" => Some(Self::Presentation),
            "image" | "photo" | "picture" | "img" => Some(Self::Image),
            "audio" | "sound" | "music" | "voice" => Some(Self::Audio),
            "video" | "movie" | "clip" => Some(Self::Video),
            "archive" | "zip" => Some(Self::Archive),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkModality {
    None,
    Text,
    Image,
    Audio,
}

impl ChunkModality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "text" => Self::Text,
            "image" => Self::Image,
            "audio" => Self::Audio,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Hybrid,
    Semantic,
    Lexical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentOrigin {
    pub page: Option<u32>,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub start_seconds: Option<f64>,
    pub end_seconds: Option<f64>,
}

impl ContentOrigin {
    pub fn none() -> Self {
        Self {
            page: None,
            line_start: None,
            line_end: None,
            start_seconds: None,
            end_seconds: None,
        }
    }

    pub fn page(p: u32) -> Self {
        Self {
            page: Some(p),
            line_start: None,
            line_end: None,
            start_seconds: None,
            end_seconds: None,
        }
    }

    pub fn lines(start: u32, end: u32) -> Self {
        Self {
            page: None,
            line_start: Some(start),
            line_end: Some(end),
            start_seconds: None,
            end_seconds: None,
        }
    }

    pub fn time_range(start_s: f64, end_s: f64) -> Self {
        Self {
            page: None,
            line_start: None,
            line_end: None,
            start_seconds: Some(start_s),
            end_seconds: Some(end_s),
        }
    }

    pub fn format_label(&self) -> Option<String> {
        if let Some(p) = self.page {
            return Some(format!("Page {}", p));
        }
        if let (Some(s), Some(e)) = (self.start_seconds, self.end_seconds) {
            return Some(format!("{}–{}", format_timestamp(s), format_timestamp(e)));
        }
        if let (Some(l1), Some(l2)) = (self.line_start, self.line_end) {
            if l1 == l2 {
                return Some(format!("Line {}", l1));
            }
            return Some(format!("Lines {}–{}", l1, l2));
        }
        None
    }
}

pub fn format_timestamp(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let mins = total / 60;
    let secs = total % 60;
    if mins >= 60 {
        let hrs = mins / 60;
        let rem_mins = mins % 60;
        format!("{}:{:02}:{:02}", hrs, rem_mins, secs)
    } else {
        format!("{}:{:02}", mins, secs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Landmark {
    pub label: String,
    pub text: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilter {
    pub kinds: Vec<SearchKind>,
    pub path_prefix: Option<String>,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub file_id: i64,
    pub chunk_id: i64,
    pub path: String,
    pub relative_path: String,
    pub file_name: String,
    pub kind: SearchKind,
    pub modality: ChunkModality,
    pub score: f64,
    pub semantic_z: f64,
    pub raw_cosine: f64,
    pub lexical_coverage: f64,
    pub snippet: String,
    pub origin: ContentOrigin,
    pub modified_unix: i64,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub query: String,
    pub mode: SearchMode,
    pub elapsed_ms: f64,
    pub decoded_audio_transcript: Option<String>,
    pub distilled_audio_keywords: Vec<LexiconMatch>,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainmentVerdict {
    ExactPart,
    NearDuplicatePart,
    SemanticPart,
    RelatedMatch,
}

impl ContainmentVerdict {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ExactPart => "EXACT PART OF FILE",
            Self::NearDuplicatePart => "NEAR-DUPLICATE PART OF FILE",
            Self::SemanticPart => "SEMANTIC PART OF FILE",
            Self::RelatedMatch => "RELATED MATCH",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainmentHit {
    pub path: String,
    pub relative_path: String,
    pub file_name: String,
    pub kind: SearchKind,
    pub modality: ChunkModality,
    pub verdict: ContainmentVerdict,
    pub confidence: f64,
    pub semantic_similarity: f64,
    pub perceptual_hash_similarity: Option<f64>,
    pub lexical_containment: Option<f64>,
    pub temporal_alignment: Option<f64>,
    pub origin: ContentOrigin,
    pub location_summary: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainmentResponse {
    pub query_description: String,
    pub query_modality: String,
    pub elapsed_ms: f64,
    pub is_contained: bool,
    pub best_confidence: f64,
    pub hits: Vec<ContainmentHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimateSummary {
    pub root: PathBuf,
    pub total_files: usize,
    pub total_bytes: u64,
    pub by_kind: BTreeMap<String, usize>,
    pub estimated_chunks: usize,
    pub estimated_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub root: String,
    pub indexed_files: usize,
    pub total_chunks: usize,
    pub text_chunks: usize,
    pub image_chunks: usize,
    pub audio_chunks: usize,
    pub vector_bytes: u64,
    pub db_bytes: u64,
    pub elapsed_ms: f64,
}
