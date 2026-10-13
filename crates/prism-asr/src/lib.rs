pub mod audio;
pub mod decoder;
pub mod lexicon;
pub mod vad;

pub use audio::{
    decode_wav_bytes_16k_mono, load_audio_16k_mono, load_ffmpeg_16k_mono,
    record_microphone_16k_mono, resample_linear, MAX_WINDOW_SAMPLES, SAMPLE_RATE,
};
pub use decoder::{DecodedAudioWindow, SemanticAsrEngine, SemanticAsrResult};
pub use lexicon::{LexiconMatch, SemanticLexicon, CORE_VOCABULARY};
pub use vad::{compute_rms, compute_zcr, slice_30s_index_windows, slice_acoustic_windows, AcousticWindow};

#[cfg(test)]
mod tests {
    use super::*;
    use prism_llama::{LlamaEmbedder, LlamaModels, ModelFiles};
    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn test_semantic_asr_on_reference_speech() -> anyhow::Result<()> {
        if !ModelFiles::is_ready(None) {
            return Ok(());
        }
        let speech_path = workspace_root()
            .join("tests")
            .join("reference")
            .join("speech.wav");
        if !speech_path.exists() {
            return Ok(());
        }

        let models = LlamaModels::load_default(true)?;
        let embedder = LlamaEmbedder::new(models, None)?;
        let engine = SemanticAsrEngine::new(&embedder)?;

        let res = engine.decode_file(&embedder, &speech_path, false)?;
        assert_eq!(res.audio_embedding.len(), LlamaEmbedder::DIMENSION);
        assert!(!res.windows.is_empty());
        assert!(!res.distilled_keywords.is_empty());

        // Test Semantic Pseudo-Relevance Keyword Distillation against candidate sentences
        let candidates = vec![
            "A calm alpine lake reflecting snow-capped mountain peaks.".to_string(),
            "Quarterly revenue spreadsheet and corporate tax deductions.".to_string(),
        ];
        let prf = engine.distill_corpus_keywords(&embedder, &res, &candidates, 5)?;
        assert!(!prf.is_empty());
        Ok(())
    }
}
