# Prism

**Prism** is a fast, local-first, cross-platform (**Linux** & **macOS**) multimodal search engine, **Semantic-Space ASR**, and **Part-of-File Containment Verifier** written in 100% Rust on top of Google DeepMind's **EmbeddingGemma 2** (`ggml-org/embeddinggemma-2-GGUF`) and a custom-patched `llama.cpp` + `mtmd` runtime.

Prism indexes folders of **Text, Source Code, Jupyter Notebooks, PDFs, Office Documents (`.docx`, `.pptx`, `.xlsx`, `.odt`, `.epub`), Images (`.jpg`, `.png`, `.webp`, `.heic`, `.avif`, `.tiff`), Audio (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`), and Video (`.mp4`, `.mkv`, `.mov`, `.webm`)** into a unified **768-dimensional semantic vector space** (`.prism.vectors.f16`) paired with a Unicode-normalized SQLite WAL + FTS5 lexical index (`.prism.db`).

---

## Key Capabilities

### 1. Unified 768-D Multimodal Hybrid Search (Text + Vision + Audio + Code)
- **Single Shared 768-D Space**: Uses `embeddinggemma-2-Q8_0.gguf` and `mmproj-embeddinggemma-2-Q8_0.gguf` so text queries, source code, photos, screenshots, PDF pages, spoken audio clips, and video keyframes all live in the same 768-dimensional L2-normalized vector space.
- **Modality-Centered Z-Score Calibration**: Eliminates cross-modal similarity distribution drift by centering raw cosine scores against each modality's mean (`text`, `image`, `audio`) and normalizing by global standard deviation before blending with FTS5 lexical coverage (`pow(coverage, 1.5)`).
- **Pillow-Exact 48px Bicubic Image Resizing**: Implements a 22-bit fixed-point separable Mitchell-Netravali/Catmull-Rom (`a = -0.5`) resizer aligned to Gemma 4's `16 × 3 = 48px` patch-pooling grid (`cosine >= 0.999` parity with Python reference embeddings).

### 2. EmbeddingGemma 2 Semantic-Space ASR (`prism-asr`)
Prism exploits EmbeddingGemma 2's native `mtmd` audio encoder and shared 768-D text-audio space to perform **speech-to-intent and semantic keyword decoding** without requiring a heavy external ASR model:
1. **Direct 768-D Audio-to-Multimodal Vector Search**: Embeds 16 kHz mono spoken queries directly into the 768-D space (`mtmd_bitmap_init_from_audio`) to retrieve matching images, PDFs, code, documents, audio, and video with zero transcription latency.
2. **Sub-Window Acoustic Slicing & 768-D Lexicon Projection**: Slices active speech via RMS/ZCR voice activity detection into overlapping acoustic sub-windows (`1.6s` window, `0.6s` hop), embeds each window in a single `mtmd` batch, projects against a cached + corpus-enriched 768-D vocabulary bank with hubness-corrected Z-scores, and runs CTC-style temporal deduplication.
3. **Semantic Pseudo-Relevance Keyword Distillation**: Uses the spoken audio vector's top semantic matches in the index to extract candidate domain terms, verifies their 768-D cosine alignment against the spoken utterance and its sub-windows, and injects the verified keywords into FTS5 hybrid ranking.
4. **Index-on-the-Fly Audio Tagging**: Automatically runs Semantic-Space ASR on 30-second audio/video soundtrack windows during `prism index`, making spoken topics searchable via FTS5 alongside their 768-D audio vectors.

### 3. Any-to-Any "Part-of-File" Containment Verification (`prism contain`)
Pass **any Image, Video clip, Audio clip, Document, or Text passage** to verify whether—and where—it is contained inside the indexed files:
- **Image-in-File**: Combines 768-D EmbeddingGemma 2 visual similarity with 64-bit 8×8 perceptual average hashing (`aHash`) to detect whether an image is an exact match, crop/re-encode, rasterized PDF page (`Page N`), or video frame (`MM:SS–MM:SS`).
- **Video-in-Video / Media**: Extracts multi-frame keyframe sequences + audio tracks and aligns them against indexed videos and images to localize exact `[start, end]` timestamps.
- **Audio-in-Audio / Video / Text**: Matches overlapping 768-D acoustic sub-windows and Semantic ASR tokens against indexed audio/video soundtracks and documents.
- **Text / Document-in-File**: Combines normalized substring & 3-gram shingle recall with 768-D passage embeddings to pinpoint exact line ranges (`Lines X–Y`) or pages.

---

## Workspace Architecture

| Crate | Description |
| :--- | :--- |
| [`crates/prism-llama`](crates/prism-llama) | C shim + Rust FFI to patched `llama.cpp` (`b11461`) & `mtmd`, `ModelFiles` GGUF manager, `LlamaEmbedder`, and Pillow-exact `ImageSizing` + 64-bit `aHash` |
| [`crates/prism-asr`](crates/prism-asr) | 16 kHz audio loader/resampler, acoustic VAD slicer, 768-D `SemanticLexicon`, temporal sub-window decoder, and Semantic PRF keyword distillation |
| [`crates/prism-core`](crates/prism-core) | `IndexStore` (SQLite WAL + FTS5 + `mmap` `f16` vectors), multimodal extractors (PDFium FFI, Office XML, EXIF images, FFmpeg audio/video), `Indexer`, `Searcher`, and `ContainmentEngine` |
| [`crates/prism-cli`](crates/prism-cli) | `prism` command-line interface and persistent JSON-RPC stdio worker |
| [`crates/prism-app`](crates/prism-app) | Cross-platform Tauri v2 desktop application for Linux and macOS |

---

## Quick Start (Linux & macOS)

### Prerequisites
- **Rust** (`cargo`, `rustc` 1.80+)
- **CMake** + **Ninja** + C/C++ compiler (`clang` or `gcc`)
- **FFmpeg** (`ffmpeg` and `ffprobe` on `PATH` for audio/video/HEIC/AVIF decoding)

### Build
```bash
./build.sh
```
This automatically fetches `Vendor/pdfium`, builds patched `llama.cpp` + `mtmd` into `Vendor/llama` (with Metal on macOS or CPU/SIMD on Linux), and compiles `target/release/prism`.

### CLI Usage
```bash
# 1. Download & verify EmbeddingGemma 2 GGUF models (~865 MB total)
./target/release/prism download-model

# 2. Estimate indexing cost for a directory
./target/release/prism estimate /path/to/folder

# 3. Index a folder (Text, Code, PDFs, Office docs, Images, Audio, Video)
./target/release/prism index /path/to/folder

# 4. Hybrid text search
./target/release/prism search "calm alpine mountain lake" -d /path/to/folder

# 5. Spoken Audio / Voice search (exploiting 768-D EmbeddingGemma 2 space + Semantic ASR)
./target/release/prism search --audio speech.wav -d /path/to/folder

# 6. Standalone 768-D Semantic-Space ASR breakdown
./target/release/prism asr speech.wav -d /path/to/folder

# 7. Verify if an Image, Video, Audio clip, or Text snippet is part of any indexed file
./target/release/prism contain photo.jpg -d /path/to/folder
./target/release/prism contain clip.mp4 -d /path/to/folder
./target/release/prism contain speech.wav -d /path/to/folder
./target/release/prism contain --text "exact quote or code snippet" -d /path/to/folder
```

---

## License
Apache-2.0
