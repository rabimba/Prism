pub mod containment;
pub mod extractors;
pub mod indexer;
pub mod model;
pub mod rules;
pub mod search_text;
pub mod searcher;
pub mod store;
pub mod vectors;

pub use containment::ContainmentEngine;
pub use indexer::Indexer;
pub use model::{
    ChunkModality, ContainmentHit, ContainmentResponse, ContainmentVerdict, ContentOrigin,
    EstimateSummary, IndexStats, Landmark, SearchFilter, SearchHit, SearchKind, SearchMode,
    SearchResponse,
};
pub use rules::{classify_extension, classify_path, crawl_directory, estimate_directory, CrawledFile};
pub use search_text::{PageTextProblem, SearchText};
pub use searcher::Searcher;
pub use store::{IndexStore, DB_FILENAME, VECTORS_FILENAME};
pub use vectors::{compute_modality_z_scores, VectorStore, VECTOR_BYTES, VECTOR_DIM};
