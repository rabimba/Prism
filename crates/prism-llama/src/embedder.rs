use std::ffi::{c_char, c_void, CString};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::image_sizing::ImageSizing;
use crate::models::ModelFiles;

extern "C" {
    fn prism_llama_init_backend();
    fn prism_llama_load_model(path: *const c_char) -> *mut c_void;
    fn prism_llama_free_model(model: *mut c_void);
    fn prism_llama_model_n_embd(model: *const c_void) -> i32;
    fn prism_mtmd_init(mmproj_path: *const c_char, model: *const c_void, n_threads: i32) -> *mut c_void;
    fn prism_mtmd_free(mtmd: *mut c_void);
    fn prism_llama_new_context(model: *mut c_void, max_tokens: u32, n_threads: i32) -> *mut c_void;
    fn prism_llama_free_context(ctx: *mut c_void);
    fn prism_llama_embed_texts(
        ctx: *mut c_void,
        model: *const c_void,
        texts: *const *const c_char,
        text_lens: *const usize,
        n_texts: usize,
        max_tokens: u32,
        out_embeddings: *mut f32,
    ) -> i32;
    fn prism_mtmd_bitmap_from_rgb(nx: u32, ny: u32, rgb_data: *const u8) -> *mut c_void;
    fn prism_mtmd_bitmap_from_audio(n_samples: usize, pcm_f32: *const f32) -> *mut c_void;
    fn prism_mtmd_bitmap_from_file(mtmd: *mut c_void, path: *const c_char) -> *mut c_void;
    fn prism_mtmd_bitmap_free(bmp: *mut c_void);
    fn prism_mtmd_embed_bitmaps(
        ctx: *mut c_void,
        mtmd: *mut c_void,
        model: *const c_void,
        bitmaps: *const *mut c_void,
        n_bitmaps: usize,
        max_tokens: u32,
        out_embeddings: *mut f32,
    ) -> i32;
}

static BACKEND_INIT: Lazy<()> = Lazy::new(|| unsafe {
    prism_llama_init_backend();
});

fn default_threads() -> i32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4);
    (cores - 2).clamp(1, 8)
}

/// Loaded model weights (`embeddinggemma-2` text + optional `mmproj` vision/audio).
pub struct LlamaModels {
    pub(crate) model: *mut c_void,
    pub(crate) mtmd: *mut c_void,
    pub(crate) mtmd_lock: Mutex<()>,
}

unsafe impl Send for LlamaModels {}
unsafe impl Sync for LlamaModels {}

impl LlamaModels {
    pub fn load(model_path: &Path, mmproj_path: Option<&Path>, threads: Option<i32>) -> Result<Arc<Self>> {
        Lazy::force(&BACKEND_INIT);
        let n_threads = threads.unwrap_or_else(default_threads);

        let c_model = CString::new(model_path.to_string_lossy().as_bytes())
            .context("Invalid model path")?;
        let model = unsafe { prism_llama_load_model(c_model.as_ptr()) };
        if model.is_null() {
            anyhow::bail!("Failed to load llama model from {}", model_path.display());
        }

        let dim = unsafe { prism_llama_model_n_embd(model) };
        if dim as usize != LlamaEmbedder::DIMENSION {
            unsafe { prism_llama_free_model(model) };
            anyhow::bail!(
                "Unexpected model embedding dimension: got {}, expected {}",
                dim,
                LlamaEmbedder::DIMENSION
            );
        }

        let mtmd = if let Some(mp) = mmproj_path {
            let c_mmproj = CString::new(mp.to_string_lossy().as_bytes())
                .context("Invalid mmproj path")?;
            let m = unsafe { prism_mtmd_init(c_mmproj.as_ptr(), model, n_threads) };
            if m.is_null() {
                unsafe { prism_llama_free_model(model) };
                anyhow::bail!("Failed to load mtmd projector from {}", mp.display());
            }
            m
        } else {
            std::ptr::null_mut()
        };

        Ok(Arc::new(Self {
            model,
            mtmd,
            mtmd_lock: Mutex::new(()),
        }))
    }

    pub fn load_default(with_media: bool) -> Result<Arc<Self>> {
        let paths = ModelFiles::ensure(None, None)?;
        Self::load(
            &paths.model_path,
            if with_media {
                Some(paths.mmproj_path.as_path())
            } else {
                None
            },
            None,
        )
    }
}

impl Drop for LlamaModels {
    fn drop(&mut self) {
        unsafe {
            if !self.mtmd.is_null() {
                prism_mtmd_free(self.mtmd);
            }
            if !self.model.is_null() {
                prism_llama_free_model(self.model);
            }
        }
    }
}

/// EmbeddingGemma 2 multimodal embedder producing 768-dimensional L2-normalized vectors
/// for text, images, and 16 kHz mono audio in a single shared vector space.
pub struct LlamaEmbedder {
    models: Arc<LlamaModels>,
    ctx: Mutex<*mut c_void>,
    max_tokens: u32,
}

unsafe impl Send for LlamaEmbedder {}
unsafe impl Sync for LlamaEmbedder {}

impl LlamaEmbedder {
    pub const DIMENSION: usize = 768;
    pub const MAX_TEXT_TOKENS: u32 = 512;
    pub const MAX_MEDIA_TOKENS: u32 = 2048;

    pub const PHOTO_BUDGET: usize = ImageSizing::PHOTO_BUDGET;
    pub const SCREENSHOT_BUDGET: usize = ImageSizing::SCREENSHOT_BUDGET;
    pub const PAGE_BUDGET: usize = ImageSizing::PAGE_BUDGET;
    pub const FRAME_BUDGET: usize = ImageSizing::FRAME_BUDGET;

    pub fn new(models: Arc<LlamaModels>, threads: Option<i32>) -> Result<Self> {
        let n_threads = threads.unwrap_or_else(default_threads);
        let max_tokens = if models.mtmd.is_null() {
            Self::MAX_TEXT_TOKENS
        } else {
            Self::MAX_MEDIA_TOKENS
        };
        let ctx = unsafe { prism_llama_new_context(models.model, max_tokens, n_threads) };
        if ctx.is_null() {
            anyhow::bail!("Failed to create llama_context");
        }
        Ok(Self {
            models,
            ctx: Mutex::new(ctx),
            max_tokens,
        })
    }

    pub fn from_paths(model_path: &Path, mmproj_path: Option<&Path>, threads: Option<i32>) -> Result<Self> {
        let models = LlamaModels::load(model_path, mmproj_path, threads)?;
        Self::new(models, threads)
    }

    pub fn models(&self) -> &Arc<LlamaModels> {
        &self.models
    }

    pub fn query_prompt(text: &str) -> String {
        format!("task: search result | query: {}", text)
    }

    pub fn code_query_prompt(text: &str) -> String {
        format!("task: code retrieval | query: {}", text)
    }

    pub fn document_prompt(text: &str, title: Option<&str>) -> String {
        let tag = match title {
            Some(t) if !t.trim().is_empty() => t.trim(),
            _ => "none",
        };
        format!("title: {} | text: {}", tag, text)
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prompt = Self::query_prompt(text);
        let mut res = self.embed_texts(&[prompt])?;
        Ok(res.pop().unwrap_or_else(|| vec![0.0; Self::DIMENSION]))
    }

    pub fn embed_code_query(&self, text: &str) -> Result<Vec<f32>> {
        let prompt = Self::code_query_prompt(text);
        let mut res = self.embed_texts(&[prompt])?;
        Ok(res.pop().unwrap_or_else(|| vec![0.0; Self::DIMENSION]))
    }

    pub fn embed_document(&self, text: &str, title: Option<&str>) -> Result<Vec<f32>> {
        let prompt = Self::document_prompt(text, title);
        let mut res = self.embed_texts(&[prompt])?;
        Ok(res.pop().unwrap_or_else(|| vec![0.0; Self::DIMENSION]))
    }

    pub fn embed_texts<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let ptrs: Vec<*const c_char> = texts
            .iter()
            .map(|s| s.as_ref().as_ptr() as *const c_char)
            .collect();
        let lens: Vec<usize> = texts.iter().map(|s| s.as_ref().len()).collect();
        let mut flat = vec![0.0f32; texts.len() * Self::DIMENSION];

        let ctx_guard = self.ctx.lock();
        let rc = unsafe {
            prism_llama_embed_texts(
                *ctx_guard,
                self.models.model,
                ptrs.as_ptr(),
                lens.as_ptr(),
                texts.len(),
                self.max_tokens,
                flat.as_mut_ptr(),
            )
        };
        drop(ctx_guard);

        if rc != 0 {
            anyhow::bail!("prism_llama_embed_texts failed with code {}", rc);
        }

        Ok(flat
            .chunks_exact(Self::DIMENSION)
            .map(|c| c.to_vec())
            .collect())
    }

    /// Embed pre-sized RGB8 images `(&[u8], width, height)`.
    pub fn embed_images(&self, images: &[(&[u8], usize, usize)]) -> Result<Vec<Vec<f32>>> {
        if self.models.mtmd.is_null() {
            anyhow::bail!("Media projector (mmproj) is not loaded");
        }
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut bitmaps = Vec::with_capacity(images.len());
        for &(rgb, w, h) in images {
            if rgb.len() != w * h * 3 || w == 0 || h == 0 {
                for b in bitmaps {
                    unsafe { prism_mtmd_bitmap_free(b) };
                }
                anyhow::bail!("Invalid RGB buffer size {} for {}x{}", rgb.len(), w, h);
            }
            let bmp = unsafe { prism_mtmd_bitmap_from_rgb(w as u32, h as u32, rgb.as_ptr()) };
            if bmp.is_null() {
                for b in bitmaps {
                    unsafe { prism_mtmd_bitmap_free(b) };
                }
                anyhow::bail!("mtmd_bitmap_init failed");
            }
            bitmaps.push(bmp);
        }

        let res = self.eval_bitmaps(&bitmaps);
        for b in bitmaps {
            unsafe { prism_mtmd_bitmap_free(b) };
        }
        res
    }

    /// Resize arbitrary RGB8 image to `budget` and embed.
    pub fn embed_image_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        budget: usize,
    ) -> Result<Vec<f32>> {
        let (out_w, out_h) = ImageSizing::size(width, height, budget)?;
        let resized = ImageSizing::resize_rgb(rgb, width, height, out_w, out_h);
        let mut vecs = self.embed_images(&[(&resized, out_w, out_h)])?;
        Ok(vecs.pop().unwrap_or_else(|| vec![0.0; Self::DIMENSION]))
    }

    /// Load an image file from disk, apply EXIF orientation, compute 64-bit perceptual aHash,
    /// resize to `budget`, and embed into 768-D space.
    pub fn embed_image_path(&self, path: &Path, budget: usize) -> Result<(Vec<f32>, u64)> {
        let (resized, out_w, out_h, ahash) = ImageSizing::load_and_prepare_file(path, budget)?;
        let mut vecs = self.embed_images(&[(&resized, out_w, out_h)])?;
        let vec = vecs.pop().unwrap_or_else(|| vec![0.0; Self::DIMENSION]);
        Ok((vec, ahash))
    }

    /// Embed audio files (WAV/MP3/FLAC) via `mtmd_helper_bitmap_init_from_file`.
    pub fn embed_audio(&self, paths: &[&Path]) -> Result<Vec<Vec<f32>>> {
        if self.models.mtmd.is_null() {
            anyhow::bail!("Media projector (mmproj) is not loaded");
        }
        if paths.is_empty() {
            return Ok(Vec::new());
        }

        let mut bitmaps = Vec::with_capacity(paths.len());
        for p in paths {
            let c_path = CString::new(p.to_string_lossy().as_bytes())
                .context("Invalid audio path")?;
            let bmp = unsafe { prism_mtmd_bitmap_from_file(self.models.mtmd, c_path.as_ptr()) };
            if bmp.is_null() {
                for b in bitmaps {
                    unsafe { prism_mtmd_bitmap_free(b) };
                }
                anyhow::bail!("Failed to load audio file {}", p.display());
            }
            bitmaps.push(bmp);
        }

        let res = self.eval_bitmaps(&bitmaps);
        for b in bitmaps {
            unsafe { prism_mtmd_bitmap_free(b) };
        }
        res
    }

    /// Embed a single 16 kHz mono float32 PCM window directly in memory (zero disk I/O).
    pub fn embed_audio_samples(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut vecs = self.embed_audio_windows(&[samples])?;
        Ok(vecs.pop().unwrap_or_else(|| vec![0.0; Self::DIMENSION]))
    }

    /// Embed a batch of 16 kHz mono float32 PCM windows directly in memory using `mtmd_bitmap_init_from_audio`.
    pub fn embed_audio_windows(&self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if self.models.mtmd.is_null() {
            anyhow::bail!("Media projector (mmproj) is not loaded");
        }
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        let mut bitmaps = Vec::with_capacity(windows.len());
        for &win in windows {
            if win.is_empty() {
                for b in bitmaps {
                    unsafe { prism_mtmd_bitmap_free(b) };
                }
                anyhow::bail!("Empty audio PCM window");
            }
            let bmp = unsafe { prism_mtmd_bitmap_from_audio(win.len(), win.as_ptr()) };
            if bmp.is_null() {
                for b in bitmaps {
                    unsafe { prism_mtmd_bitmap_free(b) };
                }
                anyhow::bail!("mtmd_bitmap_init_from_audio failed");
            }
            bitmaps.push(bmp);
        }

        let res = self.eval_bitmaps(&bitmaps);
        for b in bitmaps {
            unsafe { prism_mtmd_bitmap_free(b) };
        }
        res
    }

    fn eval_bitmaps(&self, bitmaps: &[*mut c_void]) -> Result<Vec<Vec<f32>>> {
        if bitmaps.is_empty() {
            return Ok(Vec::new());
        }
        let mut flat = vec![0.0f32; bitmaps.len() * Self::DIMENSION];
        let _mtmd_guard = self.models.mtmd_lock.lock();
        let ctx_guard = self.ctx.lock();
        let rc = unsafe {
            prism_mtmd_embed_bitmaps(
                *ctx_guard,
                self.models.mtmd,
                self.models.model,
                bitmaps.as_ptr(),
                bitmaps.len(),
                Self::MAX_MEDIA_TOKENS,
                flat.as_mut_ptr(),
            )
        };
        drop(ctx_guard);
        drop(_mtmd_guard);

        if rc != 0 {
            anyhow::bail!("prism_mtmd_embed_bitmaps failed with code {}", rc);
        }

        Ok(flat
            .chunks_exact(Self::DIMENSION)
            .map(|c| c.to_vec())
            .collect())
    }
}

impl Drop for LlamaEmbedder {
    fn drop(&mut self) {
        let ctx = *self.ctx.lock();
        unsafe {
            if !ctx.is_null() {
                prism_llama_free_context(ctx);
            }
        }
    }
}

/// Cosine similarity between two L2-normalized (or general) vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct ReferenceFile {
        vector_version: u32,
        made_with: String,
        vectors: HashMap<String, Vec<f32>>,
    }

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn test_reference_vectors_parity() -> Result<()> {
        if !ModelFiles::is_ready(None) {
            eprintln!("Skipping reference vector test because GGUF models are not downloaded yet");
            return Ok(());
        }
        let ref_dir = workspace_root().join("tests").join("reference");
        let vectors_path = ref_dir.join("vectors.json");
        if !vectors_path.exists() {
            return Ok(());
        }

        let data = std::fs::read_to_string(&vectors_path)?;
        let reference: ReferenceFile = serde_json::from_str(&data)?;
        assert_eq!(reference.vector_version, 1, "{}", reference.made_with);

        let models = LlamaModels::load_default(true)?;
        let embedder = LlamaEmbedder::new(models, None)?;

        let doc_long_body = (1..=60)
            .map(|i| {
                format!(
                    "Note {}: the review moved to room {} because the projector in the main hall failed again, and the team agreed to send the slides the night before.",
                    i,
                    i % 7 + 2
                )
            })
            .collect::<Vec<_>>()
            .join(" ");

        let text_cases: Vec<(&str, String)> = vec![
            ("query", LlamaEmbedder::query_prompt("payment declined error")),
            ("query-bn", LlamaEmbedder::query_prompt("ইলিশ মাছ রান্না")),
            ("query-ar", LlamaEmbedder::query_prompt("متى تغلق المكتبة")),
            (
                "document",
                LlamaEmbedder::document_prompt(
                    "Flight to Tokyo Haneda on November 3. Then the train to Kamakura to see the Great Buddha, and back for dinner in Shibuya.",
                    Some("Tokyo itinerary"),
                ),
            ),
            (
                "document-long",
                LlamaEmbedder::document_prompt(&doc_long_body, None),
            ),
        ];

        // 1. Text embeddings match reference within Q8_0 tolerance (cosine >= 0.999)
        let prompts: Vec<&str> = text_cases.iter().map(|(_, p)| p.as_str()).collect();
        let text_vecs = embedder.embed_texts(&prompts)?;
        for ((name, _), got) in text_cases.iter().zip(text_vecs.iter()) {
            let expected = reference.vectors.get(*name).expect("Missing reference key");
            let cos = cosine_similarity(got, expected);
            assert!(
                cos >= 0.999,
                "Text '{}' cosine {} < 0.999",
                name,
                cos
            );
        }

        // 2. Image embeddings match reference within Q8_0 tolerance (cosine >= 0.999)
        let image_cases: &[(&str, &str, usize)] = &[
            ("screen-280", "screen.png", 280),
            ("photo-140", "photo.jpg", 140),
            ("frame-70", "photo.jpg", 70),
        ];
        for &(name, file, budget) in image_cases {
            let img_path = ref_dir.join(file);
            let (got, _ahash) = embedder.embed_image_path(&img_path, budget)?;
            let expected = reference.vectors.get(name).expect("Missing reference key");
            let cos = cosine_similarity(&got, expected);
            assert!(
                cos >= 0.999,
                "Image '{}' cosine {} < 0.999",
                name,
                cos
            );
        }

        // 3. Audio embeddings match reference within Q8_0 tolerance (cosine >= 0.999)
        let audio_path = ref_dir.join("speech.wav");
        let got_audio = &embedder.embed_audio(&[audio_path.as_path()])?[0];
        let expected_audio = reference.vectors.get("speech").expect("Missing speech key");
        let cos_audio = cosine_similarity(got_audio, expected_audio);
        assert!(
            cos_audio >= 0.999,
            "Audio 'speech' cosine {} < 0.999",
            cos_audio
        );

        Ok(())
    }
}
