use std::ffi::{c_char, c_int, c_ulong, c_void, CString};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use prism_asr::{load_audio_16k_mono, slice_30s_index_windows};
use prism_llama::ImageSizing;
use quick_xml::events::Event;
use quick_xml::Reader as XmlReader;
use serde_json::Value;
use zip::ZipArchive;

use crate::model::{ChunkModality, ContentOrigin, SearchKind};
use crate::rules::CrawledFile;
use crate::search_text::SearchText;

#[derive(Debug, Clone)]
pub enum ExtractedPayload {
    Text {
        prompt_title: String,
    },
    Image {
        rgb: Vec<u8>,
        width: usize,
        height: usize,
        ahash: u64,
    },
    Audio {
        samples_16k: Vec<f32>,
    },
    LexicalOnly,
}

#[derive(Debug, Clone)]
pub struct ExtractedSegment {
    pub modality: ChunkModality,
    pub sets_scale: bool,
    pub text_content: String,
    pub origin: ContentOrigin,
    pub payload: ExtractedPayload,
}

/// Extract multimodal segments from a crawled file.
pub fn extract_file(file: &CrawledFile) -> Result<Vec<ExtractedSegment>> {
    let ext = file.extension.as_str();
    match file.kind {
        SearchKind::Text | SearchKind::Code => {
            if ext == "ipynb" {
                extract_ipynb(&file.path, &file.file_name)
            } else {
                extract_text_or_code(&file.path, &file.file_name, ext)
            }
        }
        SearchKind::Spreadsheet => {
            if matches!(ext, "xlsx" | "ods") {
                extract_office_zip(&file.path, &file.file_name, ext)
            } else {
                extract_text_or_code(&file.path, &file.file_name, ext)
            }
        }
        SearchKind::Presentation => extract_office_zip(&file.path, &file.file_name, ext),
        SearchKind::Document => match ext {
            "pdf" => extract_pdf(&file.path, &file.file_name),
            "docx" | "odt" | "epub" => extract_office_zip(&file.path, &file.file_name, ext),
            "html" | "htm" | "xhtml" => extract_html(&file.path, &file.file_name),
            "rtf" => extract_rtf(&file.path, &file.file_name),
            _ => extract_text_or_code(&file.path, &file.file_name, ext),
        },
        SearchKind::Image => extract_image(&file.path, &file.file_name),
        SearchKind::Audio => extract_audio(&file.path, &file.file_name),
        SearchKind::Video => extract_video(&file.path, &file.file_name),
        SearchKind::Archive => Ok(Vec::new()),
    }
}

fn extract_text_or_code(
    path: &Path,
    file_name: &str,
    ext: &str,
) -> Result<Vec<ExtractedSegment>> {
    let bytes = fs::read(path)?;
    // Skip binary files masquerading as text
    if bytes.iter().take(4096).any(|&b| b == 0) {
        return Ok(Vec::new());
    }
    let text = String::from_utf8_lossy(&bytes).to_string();
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut segments = chunk_text_by_lines(&text, file_name, 1400, 200);
    // Attach structural landmarks summary if code/markdown has many symbols
    let landmarks = SearchText::extract_landmarks(&text, ext);
    if !landmarks.is_empty() && segments.len() > 1 {
        let summary = landmarks
            .iter()
            .take(40)
            .map(|l| format!("{}: {}", l.label, l.text))
            .collect::<Vec<_>>()
            .join("\n");
        segments.push(ExtractedSegment {
            modality: ChunkModality::None,
            sets_scale: false,
            text_content: summary,
            origin: ContentOrigin::none(),
            payload: ExtractedPayload::LexicalOnly,
        });
    }
    Ok(segments)
}

pub fn chunk_text_by_lines(
    text: &str,
    title: &str,
    target_chars: usize,
    overlap_chars: usize,
) -> Vec<ExtractedSegment> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut start_line = 0usize;
    while start_line < lines.len() {
        let mut end_line = start_line;
        let mut char_count = 0usize;
        let mut buf = String::new();

        while end_line < lines.len() {
            let line = lines[end_line];
            let line_len = line.len() + 1;
            if char_count > 0 && char_count + line_len > target_chars {
                break;
            }
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(line);
            char_count += line_len;
            end_line += 1;
        }

        if !buf.trim().is_empty() {
            segments.push(ExtractedSegment {
                modality: ChunkModality::Text,
                sets_scale: true,
                text_content: buf,
                origin: ContentOrigin::lines((start_line + 1) as u32, end_line as u32),
                payload: ExtractedPayload::Text {
                    prompt_title: title.to_string(),
                },
            });
        }

        if end_line >= lines.len() {
            break;
        }

        // Step back by `overlap_chars` for context continuity
        let mut back_chars = 0usize;
        let mut next_start = end_line;
        while next_start > start_line + 1 {
            let prev_len = lines[next_start - 1].len() + 1;
            if back_chars + prev_len > overlap_chars {
                break;
            }
            back_chars += prev_len;
            next_start -= 1;
        }
        start_line = next_start.max(start_line + 1);
    }

    segments
}

fn extract_ipynb(path: &Path, file_name: &str) -> Result<Vec<ExtractedSegment>> {
    let raw = fs::read_to_string(path)?;
    let json: Value = serde_json::from_str(&raw)?;
    let mut combined = String::new();
    if let Some(cells) = json.get("cells").and_then(|c| c.as_array()) {
        for (idx, cell) in cells.iter().enumerate() {
            let cell_type = cell
                .get("cell_type")
                .and_then(|v| v.as_str())
                .unwrap_or("code");
            if let Some(source) = cell.get("source") {
                let text = match source {
                    Value::String(s) => s.clone(),
                    Value::Array(arr) => arr
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                };
                if !text.trim().is_empty() {
                    combined.push_str(&format!("# Cell {} ({})\n{}\n\n", idx + 1, cell_type, text));
                }
            }
        }
    }
    Ok(chunk_text_by_lines(&combined, file_name, 1400, 180))
}

fn extract_html(path: &Path, file_name: &str) -> Result<Vec<ExtractedSegment>> {
    let raw = fs::read_to_string(path)?;
    let text = strip_xml_tags(&raw);
    Ok(chunk_text_by_lines(&text, file_name, 1400, 180))
}

fn extract_rtf(path: &Path, file_name: &str) -> Result<Vec<ExtractedSegment>> {
    let raw = fs::read_to_string(path)?;
    let mut out = String::with_capacity(raw.len());
    let mut in_control = false;
    for c in raw.chars() {
        match c {
            '\\' => in_control = true,
            ' ' | '\n' | '\r' | '{' | '}' => {
                if in_control && c == ' ' {
                    out.push(' ');
                } else if c == '\n' {
                    out.push('\n');
                }
                in_control = false;
            }
            _ => {
                if !in_control {
                    out.push(c);
                }
            }
        }
    }
    Ok(chunk_text_by_lines(&out, file_name, 1400, 180))
}

fn extract_office_zip(path: &Path, file_name: &str, ext: &str) -> Result<Vec<ExtractedSegment>> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(BufReader::new(file))?;

    let mut xml_names: Vec<String> = Vec::new();
    for i in 0..archive.len() {
        if let Ok(entry) = archive.by_index(i) {
            let name = entry.name().to_string();
            let keep = match ext {
                "docx" => name == "word/document.xml",
                "pptx" => name.starts_with("ppt/slides/slide") && name.ends_with(".xml"),
                "xlsx" => name == "xl/sharedStrings.xml" || (name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml")),
                "odt" | "ods" | "odp" => name == "content.xml",
                "epub" => name.ends_with(".xhtml") || name.ends_with(".html") || name.ends_with(".htm"),
                _ => name.ends_with(".xml"),
            };
            if keep {
                xml_names.push(name);
            }
        }
    }
    xml_names.sort();

    let mut segments = Vec::new();
    for (idx, name) in xml_names.iter().enumerate() {
        let Ok(mut entry) = archive.by_name(name) else {
            continue;
        };
        let mut xml_str = String::new();
        if entry.read_to_string(&mut xml_str).is_ok() {
            let extracted = strip_xml_tags(&xml_str);
            if !extracted.trim().is_empty() {
                if ext == "pptx" {
                    segments.push(ExtractedSegment {
                        modality: ChunkModality::Text,
                        sets_scale: true,
                        text_content: extracted,
                        origin: ContentOrigin::page((idx + 1) as u32),
                        payload: ExtractedPayload::Text {
                            prompt_title: format!("{} (Slide {})", file_name, idx + 1),
                        },
                    });
                } else {
                    segments.extend(chunk_text_by_lines(&extracted, file_name, 1400, 180));
                }
            }
        }
    }

    Ok(segments)
}

fn strip_xml_tags(xml: &str) -> String {
    let mut reader = XmlReader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut out = String::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Text(e)) => {
                if let Ok(txt) = e.unescape() {
                    if !txt.trim().is_empty() {
                        if !out.is_empty() && !out.ends_with('\n') {
                            out.push(' ');
                        }
                        out.push_str(txt.trim());
                    }
                }
            }
            Ok(Event::End(e)) => {
                let local = e.local_name();
                let tag = std::str::from_utf8(local.as_ref()).unwrap_or("");
                if matches!(tag, "p" | "div" | "li" | "tr" | "h1" | "h2" | "h3" | "h4" | "br") {
                    if !out.ends_with('\n') {
                        out.push('\n');
                    }
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out
}

fn extract_image(path: &Path, file_name: &str) -> Result<Vec<ExtractedSegment>> {
    let is_screenshot = {
        let lower = file_name.to_lowercase();
        lower.contains("screen") || lower.contains("shot") || lower.contains("capture")
    };
    let budget = if is_screenshot {
        ImageSizing::SCREENSHOT_BUDGET
    } else {
        ImageSizing::PHOTO_BUDGET
    };

    let (rgb, width, height, ahash) = ImageSizing::load_and_prepare_file(path, budget)?;
    let exif_summary = extract_exif_summary(path, file_name, width, height);

    Ok(vec![ExtractedSegment {
        modality: ChunkModality::Image,
        sets_scale: true,
        text_content: exif_summary,
        origin: ContentOrigin::none(),
        payload: ExtractedPayload::Image {
            rgb,
            width,
            height,
            ahash,
        },
    }])
}

fn extract_exif_summary(path: &Path, file_name: &str, w: usize, h: usize) -> String {
    let mut parts = vec![file_name.to_string(), format!("{}x{}", w, h)];
    if let Ok(file) = File::open(path) {
        let mut bufreader = BufReader::new(&file);
        let reader = exif::Reader::new();
        if let Ok(exif) = reader.read_from_container(&mut bufreader) {
            for tag in [
                exif::Tag::Make,
                exif::Tag::Model,
                exif::Tag::DateTimeOriginal,
                exif::Tag::ImageDescription,
                exif::Tag::Software,
            ] {
                if let Some(field) = exif.get_field(tag, exif::In::PRIMARY) {
                    let val = field.display_value().with_unit(&exif).to_string();
                    let trimmed = val.trim_matches('"').trim();
                    if !trimmed.is_empty() {
                        parts.push(trimmed.to_string());
                    }
                }
            }
        }
    }
    parts.join(" · ")
}

fn extract_audio(path: &Path, file_name: &str) -> Result<Vec<ExtractedSegment>> {
    let samples = load_audio_16k_mono(path)?;
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let windows = slice_30s_index_windows(&samples);
    let mut segments = Vec::with_capacity(windows.len());
    for (start_s, end_s, win_samples) in windows {
        segments.push(ExtractedSegment {
            modality: ChunkModality::Audio,
            sets_scale: true,
            text_content: format!(
                "{} (Audio {:.0}s–{:.0}s)",
                file_name, start_s, end_s
            ),
            origin: ContentOrigin::time_range(start_s, end_s),
            payload: ExtractedPayload::Audio {
                samples_16k: win_samples,
            },
        });
    }
    Ok(segments)
}

fn extract_video(path: &Path, file_name: &str) -> Result<Vec<ExtractedSegment>> {
    let duration = probe_media_duration(path).unwrap_or(10.0).max(1.0);
    let mut segments = Vec::new();

    // 1. Extract keyframes (up to 10 frames across the video)
    let num_frames = ((duration / 4.0).ceil() as usize).clamp(1, 10);
    let step = duration / (num_frames as f64);

    for i in 0..num_frames {
        let t = (i as f64 + 0.3) * step;
        let start_s = (i as f64) * step;
        let end_s = ((i + 1) as f64 * step).min(duration);
        if let Ok((rgb, width, height, ahash)) =
            extract_video_frame_at(path, t, ImageSizing::FRAME_BUDGET)
        {
            segments.push(ExtractedSegment {
                modality: ChunkModality::Image,
                sets_scale: true,
                text_content: format!(
                    "{} (Frame {:.1}s–{:.1}s)",
                    file_name, start_s, end_s
                ),
                origin: ContentOrigin::time_range(start_s, end_s),
                payload: ExtractedPayload::Image {
                    rgb,
                    width,
                    height,
                    ahash,
                },
            });
        }
    }

    // 2. Extract 16kHz audio track windows if the video has an audio stream
    if let Ok(samples) = load_audio_16k_mono(path) {
        if !samples.is_empty() {
            for (start_s, end_s, win_samples) in slice_30s_index_windows(&samples) {
                segments.push(ExtractedSegment {
                    modality: ChunkModality::Audio,
                    sets_scale: true,
                    text_content: format!(
                        "{} (Soundtrack {:.0}s–{:.0}s)",
                        file_name, start_s, end_s
                    ),
                    origin: ContentOrigin::time_range(start_s, end_s),
                    payload: ExtractedPayload::Audio {
                        samples_16k: win_samples,
                    },
                });
            }
        }
    }

    Ok(segments)
}

pub fn probe_media_duration(path: &Path) -> Option<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .ok()
}

pub fn extract_video_frame_at(
    path: &Path,
    timestamp_s: f64,
    budget: usize,
) -> Result<(Vec<u8>, usize, usize, u64)> {
    let output = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-v",
            "error",
            "-ss",
            &format!("{:.3}", timestamp_s.max(0.0)),
            "-i",
        ])
        .arg(path)
        .args([
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ])
        .output()?;
    if !output.status.success() || output.stdout.is_empty() {
        anyhow::bail!("Failed to extract video frame at {:.2}s", timestamp_s);
    }
    ImageSizing::load_and_prepare_bytes(&output.stdout, budget)
}

// ============================================================================
// PDF Extraction via Dynamic PDFium FFI (`Vendor/pdfium/lib/libpdfium.*`)
// ============================================================================

struct PdfiumLibrary {
    _lib: libloading::Library,
    load_document: unsafe extern "C" fn(*const c_char, *const c_char) -> *mut c_void,
    close_document: unsafe extern "C" fn(*mut c_void),
    get_page_count: unsafe extern "C" fn(*mut c_void) -> c_int,
    load_page: unsafe extern "C" fn(*mut c_void, c_int) -> *mut c_void,
    close_page: unsafe extern "C" fn(*mut c_void),
    get_page_width: unsafe extern "C" fn(*mut c_void) -> f32,
    get_page_height: unsafe extern "C" fn(*mut c_void) -> f32,
    text_load_page: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    text_close_page: unsafe extern "C" fn(*mut c_void),
    text_count_chars: unsafe extern "C" fn(*mut c_void) -> c_int,
    text_get_text: unsafe extern "C" fn(*mut c_void, c_int, c_int, *mut u16) -> c_int,
    bitmap_create: unsafe extern "C" fn(c_int, c_int, c_int) -> *mut c_void,
    bitmap_fill_rect: unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int, c_int, c_ulong) -> c_int,
    render_page_bitmap:
        unsafe extern "C" fn(*mut c_void, *mut c_void, c_int, c_int, c_int, c_int, c_int, c_int),
    bitmap_get_buffer: unsafe extern "C" fn(*mut c_void) -> *const u8,
    bitmap_get_stride: unsafe extern "C" fn(*mut c_void) -> c_int,
    bitmap_destroy: unsafe extern "C" fn(*mut c_void),
}

unsafe impl Send for PdfiumLibrary {}
unsafe impl Sync for PdfiumLibrary {}

static PDFIUM: Lazy<Option<Mutex<PdfiumLibrary>>> = Lazy::new(|| {
    let candidates = pdfium_candidate_paths();
    for path in candidates {
        if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
            unsafe {
                if let Ok(init_fn) = lib.get::<unsafe extern "C" fn()>(b"FPDF_InitLibrary\0") {
                    init_fn();
                }
                let Ok(load_document) = lib
                    .get::<unsafe extern "C" fn(*const c_char, *const c_char) -> *mut c_void>(
                        b"FPDF_LoadDocument\0",
                    )
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(close_document) = lib
                    .get::<unsafe extern "C" fn(*mut c_void)>(b"FPDF_CloseDocument\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(get_page_count) = lib
                    .get::<unsafe extern "C" fn(*mut c_void) -> c_int>(b"FPDF_GetPageCount\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(load_page) = lib
                    .get::<unsafe extern "C" fn(*mut c_void, c_int) -> *mut c_void>(
                        b"FPDF_LoadPage\0",
                    )
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(close_page) = lib
                    .get::<unsafe extern "C" fn(*mut c_void)>(b"FPDF_ClosePage\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(get_page_width) = lib
                    .get::<unsafe extern "C" fn(*mut c_void) -> f32>(b"FPDF_GetPageWidthF\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(get_page_height) = lib
                    .get::<unsafe extern "C" fn(*mut c_void) -> f32>(b"FPDF_GetPageHeightF\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(text_load_page) = lib
                    .get::<unsafe extern "C" fn(*mut c_void) -> *mut c_void>(b"FPDFText_LoadPage\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(text_close_page) = lib
                    .get::<unsafe extern "C" fn(*mut c_void)>(b"FPDFText_ClosePage\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(text_count_chars) = lib
                    .get::<unsafe extern "C" fn(*mut c_void) -> c_int>(b"FPDFText_CountChars\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(text_get_text) = lib
                    .get::<unsafe extern "C" fn(*mut c_void, c_int, c_int, *mut u16) -> c_int>(
                        b"FPDFText_GetText\0",
                    )
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(bitmap_create) = lib
                    .get::<unsafe extern "C" fn(c_int, c_int, c_int) -> *mut c_void>(
                        b"FPDFBitmap_Create\0",
                    )
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(bitmap_fill_rect) = lib
                    .get::<unsafe extern "C" fn(
                        *mut c_void,
                        c_int,
                        c_int,
                        c_int,
                        c_int,
                        c_ulong,
                    ) -> c_int>(b"FPDFBitmap_FillRect\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(render_page_bitmap) = lib
                    .get::<unsafe extern "C" fn(
                        *mut c_void,
                        *mut c_void,
                        c_int,
                        c_int,
                        c_int,
                        c_int,
                        c_int,
                        c_int,
                    )>(b"FPDF_RenderPageBitmap\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(bitmap_get_buffer) = lib
                    .get::<unsafe extern "C" fn(*mut c_void) -> *const u8>(
                        b"FPDFBitmap_GetBuffer\0",
                    )
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(bitmap_get_stride) = lib
                    .get::<unsafe extern "C" fn(*mut c_void) -> c_int>(b"FPDFBitmap_GetStride\0")
                    .map(|f| *f)
                else {
                    continue;
                };
                let Ok(bitmap_destroy) = lib
                    .get::<unsafe extern "C" fn(*mut c_void)>(b"FPDFBitmap_Destroy\0")
                    .map(|f| *f)
                else {
                    continue;
                };

                return Some(Mutex::new(PdfiumLibrary {
                    _lib: lib,
                    load_document,
                    close_document,
                    get_page_count,
                    load_page,
                    close_page,
                    get_page_width,
                    get_page_height,
                    text_load_page,
                    text_close_page,
                    text_count_chars,
                    text_get_text,
                    bitmap_create,
                    bitmap_fill_rect,
                    render_page_bitmap,
                    bitmap_get_buffer,
                    bitmap_get_stride,
                    bitmap_destroy,
                }));
            }
        }
    }
    None
});

fn pdfium_candidate_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    #[cfg(target_os = "macos")]
    let lib_name = "libpdfium.dylib";
    #[cfg(not(target_os = "macos"))]
    let lib_name = "libpdfium.so";

    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf());
    if let Some(ws) = workspace {
        paths.push(ws.join("Vendor").join("pdfium").join("lib").join(lib_name));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join(lib_name));
            paths.push(dir.join("../lib").join(lib_name));
            paths.push(dir.join("../Frameworks").join(lib_name));
        }
    }
    paths.push(PathBuf::from(lib_name));
    paths
}

fn extract_pdf(path: &Path, file_name: &str) -> Result<Vec<ExtractedSegment>> {
    if let Some(ref pdfium_mutex) = *PDFIUM {
        let pdfium = pdfium_mutex.lock();
        let c_path = CString::new(path.to_string_lossy().as_bytes())?;
        let doc = unsafe { (pdfium.load_document)(c_path.as_ptr(), std::ptr::null()) };
        if !doc.is_null() {
            let page_count = unsafe { (pdfium.get_page_count)(doc) }.max(0) as usize;
            let mut segments = Vec::new();

            for page_idx in 0..page_count.min(200) {
                let page = unsafe { (pdfium.load_page)(doc, page_idx as c_int) };
                if page.is_null() {
                    continue;
                }
                let page_num = (page_idx + 1) as u32;

                // 1. Extract page text
                let mut page_text = String::new();
                let text_page = unsafe { (pdfium.text_load_page)(page) };
                if !text_page.is_null() {
                    let count = unsafe { (pdfium.text_count_chars)(text_page) };
                    if count > 0 {
                        let mut utf16 = vec![0u16; (count as usize) + 2];
                        let written = unsafe {
                            (pdfium.text_get_text)(text_page, 0, count, utf16.as_mut_ptr())
                        };
                        if written > 1 {
                            page_text = String::from_utf16_lossy(&utf16[..(written as usize - 1)]);
                        }
                    }
                    unsafe { (pdfium.text_close_page)(text_page) };
                }

                let has_problem = SearchText::page_text_problem(&page_text).is_some();
                if !has_problem {
                    segments.push(ExtractedSegment {
                        modality: ChunkModality::Text,
                        sets_scale: true,
                        text_content: page_text.trim().to_string(),
                        origin: ContentOrigin::page(page_num),
                        payload: ExtractedPayload::Text {
                            prompt_title: format!("{} (Page {})", file_name, page_num),
                        },
                    });
                } else {
                    // Render page bitmap for visual embedding + aHash
                    let pw = unsafe { (pdfium.get_page_width)(page) }.max(72.0) as usize;
                    let ph = unsafe { (pdfium.get_page_height)(page) }.max(72.0) as usize;
                    if let Ok((out_w, out_h)) =
                        ImageSizing::size(pw, ph, ImageSizing::PAGE_BUDGET)
                    {
                        let bmp =
                            unsafe { (pdfium.bitmap_create)(out_w as c_int, out_h as c_int, 0) };
                        if !bmp.is_null() {
                            unsafe {
                                (pdfium.bitmap_fill_rect)(
                                    bmp,
                                    0,
                                    0,
                                    out_w as c_int,
                                    out_h as c_int,
                                    0xFFFFFFFF,
                                );
                                (pdfium.render_page_bitmap)(
                                    bmp,
                                    page,
                                    0,
                                    0,
                                    out_w as c_int,
                                    out_h as c_int,
                                    0,
                                    0,
                                );
                            }
                            let ptr = unsafe { (pdfium.bitmap_get_buffer)(bmp) };
                            let stride = unsafe { (pdfium.bitmap_get_stride)(bmp) } as usize;
                            if !ptr.is_null() && stride >= out_w * 4 {
                                let bgra =
                                    unsafe { std::slice::from_raw_parts(ptr, stride * out_h) };
                                let mut rgb = vec![0u8; out_w * out_h * 3];
                                for y in 0..out_h {
                                    let row = &bgra[y * stride..y * stride + out_w * 4];
                                    for x in 0..out_w {
                                        let s = x * 4;
                                        let d = (y * out_w + x) * 3;
                                        rgb[d] = row[s + 2];
                                        rgb[d + 1] = row[s + 1];
                                        rgb[d + 2] = row[s];
                                    }
                                }
                                let ahash = ImageSizing::compute_ahash(&rgb, out_w, out_h);
                                let fallback_txt = if page_text.trim().is_empty() {
                                    format!("{} (Page {})", file_name, page_num)
                                } else {
                                    page_text.trim().to_string()
                                };
                                segments.push(ExtractedSegment {
                                    modality: ChunkModality::Image,
                                    sets_scale: true,
                                    text_content: fallback_txt,
                                    origin: ContentOrigin::page(page_num),
                                    payload: ExtractedPayload::Image {
                                        rgb,
                                        width: out_w,
                                        height: out_h,
                                        ahash,
                                    },
                                });
                            }
                            unsafe { (pdfium.bitmap_destroy)(bmp) };
                        }
                    }
                }

                unsafe { (pdfium.close_page)(page) };
            }

            unsafe { (pdfium.close_document)(doc) };
            return Ok(segments);
        }
    }

    // Fallback if PDFium couldn't load: try `pdftotext` if available
    if let Ok(output) = Command::new("pdftotext").arg(path).arg("-").output() {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout).to_string();
            return Ok(chunk_text_by_lines(&text, file_name, 1400, 180));
        }
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_image_exif_orientation_and_formats() -> Result<()> {
        let pics = workspace_root().join("tests").join("pictures");
        if !pics.exists() {
            return Ok(());
        }
        // pattern-o6.jpg has EXIF Orientation = 6 (90 deg CW); verify decoding & orientation
        let jpg_path = pics.join("pattern-o6.jpg");
        if jpg_path.exists() {
            let orient = ImageSizing::read_exif_orientation(&jpg_path);
            assert_eq!(orient, 6);
            let (rgb, w, h, ahash) =
                ImageSizing::load_and_prepare_file(&jpg_path, ImageSizing::PHOTO_BUDGET)?;
            assert_eq!(rgb.len(), w * h * 3);
            assert_ne!(ahash, 0);
        }
        Ok(())
    }
}
