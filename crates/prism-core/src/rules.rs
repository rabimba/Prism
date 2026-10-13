use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::Result;
use ignore::WalkBuilder;

use crate::model::{EstimateSummary, SearchKind};

#[derive(Debug, Clone)]
pub struct CrawledFile {
    pub path: PathBuf,
    pub relative_path: String,
    pub file_name: String,
    pub extension: String,
    pub kind: SearchKind,
    pub size_bytes: u64,
    pub modified_unix: i64,
}

pub const DEFAULT_IGNORED_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    ".DS_Store",
    "node_modules",
    "target",
    ".build",
    "build",
    "dist",
    "Vendor",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
    ".nuxt",
    ".gradle",
    ".idea",
    ".vscode",
    "Pods",
    "DerivedData",
    "snap",
    "site-packages",
    "dist-packages",
    "nltk_data",
    "CMakeFiles",
    "third_party",
    "3rdparty",
    "external",
];

pub const IGNORED_DIR_PREFIXES: &[&str] = &[
    "go1.",
    "android-studio",
    "tor-browser",
    "scrcpy-",
    "support_material",
    "Comment Letters_PDF",
    "cmake-",
    "miniforge",
    "miniconda",
    "anaconda",
];

/// Maximum indexable files taken from a single directory before treating the rest as a bulk dataset (matches DigUp dataset skipping).
pub const MAX_FILES_PER_SINGLE_DIR: usize = 250;

pub const INDEX_DB_NAMES: &[&str] = &[
    ".prism.db",
    ".prism.db-wal",
    ".prism.db-shm",
    ".prism.vectors.f16",
    ".digup.db",
    ".digup.db-wal",
    ".digup.db-shm",
    ".digup.vectors.f16",
];

/// Classify a file path into its `SearchKind` based on extension or filename.
pub fn classify_path(path: &Path) -> Option<SearchKind> {
    let file_name = path.file_name()?.to_str()?;
    if file_name.starts_with('.') || INDEX_DB_NAMES.contains(&file_name) {
        return None;
    }

    let lower_name = file_name.to_lowercase();
    if matches!(
        lower_name.as_str(),
        "makefile" | "dockerfile" | "cmakelists.txt" | "justfile" | "vagrantfile"
    ) {
        return Some(SearchKind::Code);
    }
    if matches!(
        lower_name.as_str(),
        "readme" | "license" | "copying" | "changelog" | "authors" | "notice" | "todo"
    ) {
        return Some(SearchKind::Text);
    }

    let ext = path.extension()?.to_str()?.to_lowercase();
    classify_extension(&ext)
}

pub fn classify_extension(ext: &str) -> Option<SearchKind> {
    match ext {
        // Plain text & markup
        "txt" | "md" | "markdown" | "mdx" | "rst" | "org" | "tex" | "log" | "ini" | "cfg"
        | "conf" | "env" => Some(SearchKind::Text),

        // Source code, scripts, config & notebooks
        "rs" | "py" | "pyi" | "ipynb" | "swift" | "c" | "h" | "cc" | "cpp" | "cxx" | "hpp"
        | "hxx" | "m" | "mm" | "go" | "java" | "kt" | "kts" | "scala" | "js" | "jsx" | "mjs"
        | "cjs" | "ts" | "tsx" | "vue" | "svelte" | "rb" | "php" | "cs" | "fs" | "vb" | "lua"
        | "pl" | "pm" | "r" | "jl" | "dart" | "zig" | "nim" | "ex" | "exs" | "erl" | "hrl"
        | "hs" | "ml" | "mli" | "clj" | "cljs" | "lisp" | "scm" | "sh" | "bash" | "zsh"
        | "fish" | "ps1" | "bat" | "cmd" | "sql" | "graphql" | "gql" | "proto" | "toml"
        | "yaml" | "yml" | "json" | "jsonc" | "xml" | "css" | "scss" | "sass" | "less"
        | "cmake" | "gradle" | "nix" | "tf" | "hcl" => Some(SearchKind::Code),

        // Documents & rich text
        "pdf" | "docx" | "doc" | "odt" | "rtf" | "epub" | "html" | "htm" | "xhtml" | "pages" => {
            Some(SearchKind::Document)
        }

        // Spreadsheets & tabular data
        "xlsx" | "xls" | "ods" | "csv" | "tsv" | "numbers" => Some(SearchKind::Spreadsheet),

        // Presentations
        "pptx" | "ppt" | "odp" | "key" => Some(SearchKind::Presentation),

        // Images
        "jpg" | "jpeg" | "png" | "webp" | "gif" | "bmp" | "tif" | "tiff" | "heic" | "heif"
        | "avif" | "ico" | "pnm" | "ppm" | "pgm" => Some(SearchKind::Image),

        // Audio
        "mp3" | "wav" | "flac" | "m4a" | "aac" | "ogg" | "oga" | "opus" | "wma" | "aiff"
        | "aif" => Some(SearchKind::Audio),

        // Video
        "mp4" | "m4v" | "mkv" | "mov" | "webm" | "avi" | "flv" | "wmv" | "mpeg" | "mpg" | "m2ts"
        | "mts" => {
            Some(SearchKind::Video)
        }

        _ => None,
    }
}

fn crawl_internal(root: &Path, use_absolute_paths: bool) -> Result<Vec<CrawledFile>> {
    let canon_root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut builder = WalkBuilder::new(&canon_root);
    builder
        .hidden(true)
        .follow_links(false)
        .max_depth(Some(8))
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .add_custom_ignore_filename(".prismignore")
        .add_custom_ignore_filename(".digupignore");

    builder.filter_entry(|entry| {
        if entry.depth() == 0 {
            return true;
        }
        if let Some(name) = entry.file_name().to_str() {
            if DEFAULT_IGNORED_DIRS.contains(&name) || INDEX_DB_NAMES.contains(&name) {
                return false;
            }
            if entry
                .file_type()
                .map(|ft| ft.is_dir())
                .unwrap_or(false)
                && IGNORED_DIR_PREFIXES.iter().any(|p| name.starts_with(p))
            {
                return false;
            }
        }
        true
    });

    let mut files = Vec::new();
    let mut per_parent_count: std::collections::HashMap<PathBuf, usize> =
        std::collections::HashMap::new();

    for result in builder.build() {
        let Ok(entry) = result else {
            continue;
        };
        let Some(ft) = entry.file_type() else {
            continue;
        };
        if !ft.is_file() {
            continue;
        }
        let path = entry.path().to_path_buf();
        let Some(kind) = classify_path(&path) else {
            continue;
        };
        if let Some(parent) = path.parent() {
            let count = per_parent_count.entry(parent.to_path_buf()).or_insert(0);
            if *count >= MAX_FILES_PER_SINGLE_DIR {
                continue;
            }
            *count += 1;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let size_bytes = meta.len();
        if size_bytes == 0 {
            continue;
        }
        let modified_unix = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let relative_path = if use_absolute_paths {
            path.to_string_lossy().replace('\\', "/")
        } else {
            path.strip_prefix(&canon_root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/")
        };
        let file_name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let extension = path
            .extension()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        files.push(CrawledFile {
            path,
            relative_path,
            file_name,
            extension,
            kind,
            size_bytes,
            modified_unix,
        });
    }

    files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(files)
}

/// Crawl a directory respecting `.gitignore`, `.digupignore`, `.prismignore`, and default exclusions.
pub fn crawl_directory(root: &Path) -> Result<Vec<CrawledFile>> {
    crawl_internal(root, false)
}

/// Crawl multiple OS root directories storing canonical absolute paths in `relative_path` for the Global OS Index.
pub fn crawl_os_roots(roots: &[PathBuf]) -> Result<Vec<CrawledFile>> {
    let mut all_files = Vec::new();
    let mut seen_paths = std::collections::HashSet::new();
    for root in roots {
        if !root.exists() || !root.is_dir() {
            continue;
        }
        if let Ok(files) = crawl_internal(root, true) {
            for f in files {
                if seen_paths.insert(f.relative_path.clone()) {
                    all_files.push(f);
                }
            }
        }
    }
    all_files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(all_files)
}

/// Estimate indexing cost and file breakdown for a directory (or ALL_OS).
pub fn estimate_directory(root: &Path) -> Result<EstimateSummary> {
    let files = if crate::store::IndexStore::is_global_dir(&root.to_string_lossy()) {
        crawl_os_roots(&crate::store::IndexStore::default_os_roots())?
    } else {
        crawl_directory(root)?
    };
    let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_bytes = 0u64;
    let mut estimated_chunks = 0usize;
    let mut estimated_seconds = 0.0f64;

    for f in &files {
        total_bytes += f.size_bytes;
        *by_kind.entry(f.kind.as_str().to_string()).or_insert(0) += 1;
        match f.kind {
            SearchKind::Text | SearchKind::Code | SearchKind::Spreadsheet => {
                let chunks = ((f.size_bytes as usize) / 1800).clamp(1, 64);
                estimated_chunks += chunks;
                estimated_seconds += (chunks as f64) * 0.015;
            }
            SearchKind::Document | SearchKind::Presentation => {
                let chunks = ((f.size_bytes as usize) / 8000).clamp(1, 120);
                estimated_chunks += chunks;
                estimated_seconds += (chunks as f64) * 0.035;
            }
            SearchKind::Image => {
                estimated_chunks += 1;
                estimated_seconds += 0.12;
            }
            SearchKind::Audio => {
                let chunks = ((f.size_bytes as usize) / (30 * 32_000)).clamp(1, 60);
                estimated_chunks += chunks;
                estimated_seconds += (chunks as f64) * 0.18;
            }
            SearchKind::Video => {
                let chunks = ((f.size_bytes as usize) / (1024 * 1024)).clamp(2, 60);
                estimated_chunks += chunks;
                estimated_seconds += (chunks as f64) * 0.22;
            }
            SearchKind::Archive => {}
        }
    }

    Ok(EstimateSummary {
        root: root.to_path_buf(),
        total_files: files.len(),
        total_bytes,
        by_kind,
        estimated_chunks,
        estimated_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_extensions() {
        assert_eq!(classify_path(Path::new("src/main.rs")), Some(SearchKind::Code));
        assert_eq!(classify_path(Path::new("notes.md")), Some(SearchKind::Text));
        assert_eq!(classify_path(Path::new("paper.pdf")), Some(SearchKind::Document));
        assert_eq!(classify_path(Path::new("photo.jpg")), Some(SearchKind::Image));
        assert_eq!(classify_path(Path::new("speech.wav")), Some(SearchKind::Audio));
        assert_eq!(classify_path(Path::new("clip.mp4")), Some(SearchKind::Video));
        assert_eq!(classify_path(Path::new(".hidden")), None);
    }
}
