use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use prism_asr::SemanticAsrEngine;
use prism_core::{
    ContainmentEngine, ContainmentVerdict, Indexer, SearchFilter, SearchMode, Searcher,
};
use prism_llama::{LlamaEmbedder, LlamaModels, ModelFiles};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_end_to_end_multimodal_index_search_asr_and_containment() -> Result<()> {
    if !ModelFiles::is_ready(None) {
        return Ok(());
    }
    let ref_dir = workspace_root().join("tests").join("reference");
    if !ref_dir.exists() {
        return Ok(());
    }

    let tmp = tempfile::tempdir()?;
    let root = tmp.path();

    // Populate multimodal testbed
    fs::copy(ref_dir.join("photo.jpg"), root.join("alpine_photo.jpg"))?;
    fs::copy(ref_dir.join("screen.png"), root.join("terminal_screen.png"))?;
    fs::copy(ref_dir.join("speech.wav"), root.join("voice_note.wav"))?;
    fs::write(
        root.join("tokyo_itinerary.md"),
        "# Tokyo Itinerary\nFlight to Tokyo Haneda on November 3. Then the train to Kamakura to see the Great Buddha, and back for dinner in Shibuya.\n",
    )?;
    fs::write(
        root.join("engine.rs"),
        "pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {\n    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()\n}\n",
    )?;

    let models = LlamaModels::load_default(true)?;
    let embedder = LlamaEmbedder::new(models, None)?;
    let asr = SemanticAsrEngine::new(&embedder)?;

    // 1. Index folder
    let stats = Indexer::index_directory(root, &embedder, Some(&asr), true, None)?;
    assert_eq!(stats.indexed_files, 5);
    assert!(stats.text_chunks >= 2);
    assert_eq!(stats.image_chunks, 2);
    assert_eq!(stats.audio_chunks, 1);

    // 2. Hybrid Text Search
    let searcher = Searcher::open(root)?;
    let filter = SearchFilter {
        kinds: Vec::new(),
        path_prefix: None,
        limit: 5,
    };
    let text_resp = searcher.search_text(
        Some(&embedder),
        "Kamakura Great Buddha",
        SearchMode::Hybrid,
        &filter,
    )?;
    assert!(!text_resp.hits.is_empty());
    assert_eq!(text_resp.hits[0].file_name, "tokyo_itinerary.md");

    // 3. Spoken Audio Search
    let audio_resp = searcher.search_audio_file(
        &embedder,
        &asr,
        &ref_dir.join("speech.wav"),
        false,
        None,
        &filter,
    )?;
    assert!(!audio_resp.hits.is_empty());
    assert!(audio_resp.decoded_audio_transcript.is_some());
    assert!(!audio_resp.distilled_audio_keywords.is_empty());

    // 4. Part-of-File Containment: Image
    let img_contain =
        ContainmentEngine::verify_file(root, &embedder, &asr, &ref_dir.join("photo.jpg"), 5)?;
    assert!(img_contain.is_contained);
    assert_eq!(img_contain.hits[0].file_name, "alpine_photo.jpg");
    assert_eq!(img_contain.hits[0].verdict, ContainmentVerdict::ExactPart);
    assert!(img_contain.hits[0].confidence >= 0.99);

    // 5. Part-of-File Containment: Audio
    let aud_contain =
        ContainmentEngine::verify_file(root, &embedder, &asr, &ref_dir.join("speech.wav"), 5)?;
    assert!(aud_contain.is_contained);
    assert_eq!(aud_contain.hits[0].file_name, "voice_note.wav");
    assert_eq!(aud_contain.hits[0].verdict, ContainmentVerdict::ExactPart);

    // 6. Part-of-File Containment: Text
    let txt_contain = ContainmentEngine::verify_text(
        root,
        &embedder,
        "train to Kamakura to see the Great Buddha",
        5,
    )?;
    assert!(txt_contain.is_contained);
    assert_eq!(txt_contain.hits[0].file_name, "tokyo_itinerary.md");
    assert_eq!(txt_contain.hits[0].verdict, ContainmentVerdict::ExactPart);

    Ok(())
}
