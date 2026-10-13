use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

use crate::model::{ChunkModality, ContentOrigin, IndexStats, SearchKind};
use crate::rules::CrawledFile;
use crate::search_text::SearchText;
use crate::vectors::VectorStore;

pub const DB_FILENAME: &str = ".prism.db";
pub const VECTORS_FILENAME: &str = ".prism.vectors.f16";
pub const ALL_OS_SCOPE: &str = "ALL_OS";

#[derive(Debug, Clone)]
pub struct StoredFileRecord {
    pub id: i64,
    pub relative_path: String,
    pub file_name: String,
    pub kind: SearchKind,
    pub size_bytes: u64,
    pub modified_unix: i64,
    pub embedded: bool,
}

#[derive(Debug, Clone)]
pub struct NewChunkRecord {
    pub chunk_index: usize,
    pub modality: ChunkModality,
    pub sets_scale: bool,
    pub vector: Option<Vec<f32>>,
    pub ahash: Option<u64>,
    pub text_content: String,
    pub origin: ContentOrigin,
}

#[derive(Debug, Clone)]
pub struct IndexedChunkRow {
    pub chunk_id: i64,
    pub file_id: i64,
    pub relative_path: String,
    pub file_name: String,
    pub kind: SearchKind,
    pub size_bytes: u64,
    pub modified_unix: i64,
    pub modality: ChunkModality,
    pub sets_scale: bool,
    pub vector_slot: Option<usize>,
    pub ahash: Option<u64>,
    pub text_content: String,
    pub origin: ContentOrigin,
}

pub struct IndexStore {
    root: PathBuf,
    db_path: PathBuf,
    conn: Connection,
    pub vectors: VectorStore,
}

impl IndexStore {
    /// Centralized OS-wide index directory (`~/.local/share/prism/os_index` on Linux, `~/Library/Application Support/prism/os_index` on macOS).
    pub fn global_store_dir() -> PathBuf {
        let base = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
        base.join("prism").join("os_index")
    }

    pub fn is_global_dir(dir: &str) -> bool {
        let t = dir.trim();
        t.is_empty()
            || t.eq_ignore_ascii_case(ALL_OS_SCOPE)
            || t.eq_ignore_ascii_case("all")
            || t == "*"
            || Path::new(t) == Self::global_store_dir()
    }

    /// Standard OS user directories indexed by default for OS-wide search (matches DigUp Setup offered roots).
    pub fn default_os_roots() -> Vec<PathBuf> {
        let mut roots = Vec::new();
        let candidates = [
            dirs::document_dir(),
            dirs::download_dir(),
            dirs::desktop_dir(),
            dirs::picture_dir(),
            dirs::video_dir(),
            dirs::audio_dir(),
        ];
        let mut seen = HashSet::new();
        for dir_opt in candidates.into_iter().flatten() {
            if dir_opt.is_dir() {
                let canon = fs::canonicalize(&dir_opt).unwrap_or(dir_opt);
                if seen.insert(canon.clone()) {
                    roots.push(canon);
                }
            }
        }
        let demo = PathBuf::from("/tmp/prism_live_demo/corpus");
        if demo.is_dir() {
            let canon = fs::canonicalize(&demo).unwrap_or(demo);
            if seen.insert(canon.clone()) {
                roots.push(canon);
            }
        }
        roots
    }

    pub fn open_global() -> Result<Self> {
        let dir = Self::global_store_dir();
        fs::create_dir_all(&dir)?;
        Self::open(&dir)
    }

    /// Open either the Global OS Index (`ALL_OS` or a folder covered by the global index) or a folder-local `.prism.db`.
    /// Returns `(IndexStore, Option<scope_path_prefix>)`.
    pub fn open_target(dir: &str) -> Result<(Self, Option<String>)> {
        if Self::is_global_dir(dir) {
            return Ok((Self::open_global()?, None));
        }
        let p = Path::new(dir);
        let canon = fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        if canon == Self::global_store_dir() {
            return Ok((Self::open_global()?, None));
        }
        // If this folder has its own standalone index (e.g., `/tmp/prism_live_demo/corpus` or a test temp dir),
        // check if it has files; otherwise scope into the Global OS Index if the Global OS Index exists.
        if Self::exists_in(&canon) {
            let local_store = Self::open(&canon)?;
            if let Ok(st) = local_store.stats(0.0) {
                if st.indexed_files > 0 {
                    return Ok((local_store, None));
                }
            }
        }
        let global_dir = Self::global_store_dir();
        if Self::exists_in(&global_dir) {
            let prefix = canon.to_string_lossy().replace('\\', "/");
            let global_store = Self::open_global()?;
            if let Ok(st) = global_store.stats_scoped(Some(&prefix), 0.0) {
                if st.indexed_files > 0 {
                    return Ok((global_store, Some(prefix)));
                }
            }
        }
        Ok((Self::open(&canon)?, None))
    }

    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root).ok();
        let canon_root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        let db_path = canon_root.join(DB_FILENAME);
        let vec_path = canon_root.join(VECTORS_FILENAME);

        let conn = Connection::open(&db_path)
            .with_context(|| format!("Failed to open index DB {}", db_path.display()))?;

        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;
             PRAGMA temp_store = MEMORY;
             CREATE TABLE IF NOT EXISTS meta (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS files (
                 id INTEGER PRIMARY KEY,
                 relative_path TEXT NOT NULL UNIQUE,
                 file_name TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 size_bytes INTEGER NOT NULL,
                 modified_unix INTEGER NOT NULL,
                 embedded INTEGER NOT NULL DEFAULT 1
             );
             CREATE TABLE IF NOT EXISTS chunks (
                 id INTEGER PRIMARY KEY,
                 file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                 chunk_index INTEGER NOT NULL,
                 modality TEXT NOT NULL,
                 sets_scale INTEGER NOT NULL DEFAULT 1,
                 vector_slot INTEGER,
                 ahash INTEGER,
                 text_content TEXT NOT NULL,
                 page INTEGER,
                 line_start INTEGER,
                 line_end INTEGER,
                 start_seconds REAL,
                 end_seconds REAL
             );
             CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
             CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                 title,
                 body,
                 tokenize=\"porter unicode61 remove_diacritics 2 categories 'L* N* Co M*'\"
             );",
        )?;

        // Migrate older DBs that lacked `embedded` column
        let _ = conn.execute(
            "ALTER TABLE files ADD COLUMN embedded INTEGER NOT NULL DEFAULT 1",
            [],
        );

        let vectors = VectorStore::open(&vec_path)?;
        Ok(Self {
            root: canon_root,
            db_path,
            conn,
            vectors,
        })
    }

    pub fn exists_in(root: &Path) -> bool {
        root.join(DB_FILENAME).exists()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn clear_all(&mut self) -> Result<()> {
        self.conn.execute_batch(
            "DELETE FROM chunks_fts;
             DELETE FROM chunks;
             DELETE FROM files;",
        )?;
        self.vectors.clear()?;
        Ok(())
    }

    pub fn existing_files_map(&self) -> Result<HashMap<String, StoredFileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, relative_path, file_name, kind, size_bytes, modified_unix, embedded FROM files",
        )?;
        let rows = stmt.query_map([], |row| {
            let kind_str: String = row.get(3)?;
            let emb_int: i64 = row.get(6).unwrap_or(1);
            Ok(StoredFileRecord {
                id: row.get(0)?,
                relative_path: row.get(1)?,
                file_name: row.get(2)?,
                kind: SearchKind::from_str(&kind_str).unwrap_or(SearchKind::Text),
                size_bytes: row.get::<_, i64>(4)? as u64,
                modified_unix: row.get(5)?,
                embedded: emb_int != 0,
            })
        })?;
        let mut map = HashMap::new();
        for r in rows {
            let rec = r?;
            map.insert(rec.relative_path.clone(), rec);
        }
        Ok(map)
    }

    /// Phase 1 Instant OS Sync: Reconciles crawled files into SQLite + FTS5 in a single fast transaction (`<250ms`),
    /// making every file across the OS immediately searchable by filename, path, and fast text content (`embedded = 0`),
    /// while queuing files for background 768-D multimodal embedding.
    pub fn sync_files_fast(&mut self, crawled: &[CrawledFile]) -> Result<usize> {
        let existing = self.existing_files_map()?;
        let current_rel: HashSet<&str> = crawled.iter().map(|f| f.relative_path.as_str()).collect();

        let tx = self.conn.transaction()?;
        for (rel, rec) in &existing {
            if !current_rel.contains(rel.as_str()) {
                tx.execute(
                    "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?1)",
                    params![rec.id],
                )?;
                tx.execute("DELETE FROM chunks WHERE file_id = ?1", params![rec.id])?;
                tx.execute("DELETE FROM files WHERE id = ?1", params![rec.id])?;
            }
        }

        let mut synced_count = 0usize;
        for f in crawled {
            let needs_sync = match existing.get(&f.relative_path) {
                Some(rec) => rec.size_bytes != f.size_bytes || rec.modified_unix != f.modified_unix,
                None => true,
            };
            if !needs_sync {
                continue;
            }

            if let Some(rec) = existing.get(&f.relative_path) {
                tx.execute(
                    "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?1)",
                    params![rec.id],
                )?;
                tx.execute("DELETE FROM chunks WHERE file_id = ?1", params![rec.id])?;
                tx.execute("DELETE FROM files WHERE id = ?1", params![rec.id])?;
            }

            tx.execute(
                "INSERT INTO files (relative_path, file_name, kind, size_bytes, modified_unix, embedded)
                 VALUES (?1, ?2, ?3, ?4, ?5, 0)",
                params![
                    f.relative_path,
                    f.file_name,
                    f.kind.as_str(),
                    f.size_bytes as i64,
                    f.modified_unix
                ],
            )?;
            let file_id = tx.last_insert_rowid();

            let (fast_text, page, line_start, line_end) = extract_fast_preview_text(f);
            let modality = match f.kind {
                SearchKind::Image => ChunkModality::Image,
                SearchKind::Audio => ChunkModality::Audio,
                _ => ChunkModality::Text,
            };

            tx.execute(
                "INSERT INTO chunks (
                     file_id, chunk_index, modality, sets_scale, vector_slot, ahash,
                     text_content, page, line_start, line_end, start_seconds, end_seconds
                 ) VALUES (?1, 0, ?2, 0, NULL, NULL, ?3, ?4, ?5, ?6, NULL, NULL)",
                params![file_id, modality.as_str(), fast_text, page, line_start, line_end],
            )?;
            let chunk_id = tx.last_insert_rowid();
            let norm_title =
                SearchText::normalized(&format!("{} {}", f.file_name, f.relative_path));
            let norm_body = SearchText::normalized(&fast_text);
            tx.execute(
                "INSERT INTO chunks_fts (rowid, title, body) VALUES (?1, ?2, ?3)",
                params![chunk_id, norm_title, norm_body],
            )?;
            synced_count += 1;
        }

        // Ensure existing unembedded PDF/Image/Video/Audio rows have accurate origin metadata
        let _ = tx.execute(
            "UPDATE chunks
             SET line_start = NULL,
                 page = CASE
                     WHEN (SELECT kind FROM files WHERE files.id = chunks.file_id) IN ('document', 'presentation') THEN 1
                     ELSE NULL
                 END
             WHERE vector_slot IS NULL
               AND (SELECT kind FROM files WHERE files.id = chunks.file_id) NOT IN ('text', 'code', 'spreadsheet')",
            [],
        );

        tx.commit()?;
        Ok(synced_count)
    }

    /// Import already-embedded files & 768-D vectors from a folder-specific `.prism.db` into this Global OS Index
    /// without re-running the model (`<5ms`).
    pub fn import_folder_store(&mut self, folder_root: &Path) -> Result<usize> {
        let canon_folder = fs::canonicalize(folder_root).unwrap_or_else(|_| folder_root.to_path_buf());
        if !Self::exists_in(&canon_folder) || canon_folder == self.root {
            return Ok(0);
        }
        let src_store = Self::open(&canon_folder)?;
        let src_chunks = src_store.load_all_chunks()?;
        if src_chunks.is_empty() {
            return Ok(0);
        }

        let existing = self.existing_files_map()?;
        let mut by_file: HashMap<i64, Vec<IndexedChunkRow>> = HashMap::new();
        for c in src_chunks {
            by_file.entry(c.file_id).or_default().push(c);
        }

        let mut imported = 0usize;
        for (_fid, rows) in by_file {
            let first = &rows[0];
            let abs_path = canon_folder
                .join(&first.relative_path)
                .to_string_lossy()
                .replace('\\', "/");

            if let Some(rec) = existing.get(&abs_path) {
                if rec.embedded
                    && rec.size_bytes == first.size_bytes
                    && rec.modified_unix == first.modified_unix
                {
                    continue;
                }
            }

            let file_name = first.file_name.clone();
            let kind = first.kind;
            let size_bytes = first.size_bytes;
            let modified_unix = first.modified_unix;

            let mut new_chunks = Vec::with_capacity(rows.len());
            for (idx, r) in rows.into_iter().enumerate() {
                let vec = r.vector_slot.and_then(|s| src_store.vectors.get_vector(s));
                new_chunks.push(NewChunkRecord {
                    chunk_index: idx,
                    modality: r.modality,
                    sets_scale: r.sets_scale,
                    vector: vec,
                    ahash: r.ahash,
                    text_content: r.text_content,
                    origin: r.origin,
                });
            }

            self.upsert_file_with_chunks(
                &abs_path,
                &file_name,
                kind,
                size_bytes,
                modified_unix,
                new_chunks,
            )?;
            imported += 1;
        }
        Ok(imported)
    }

    /// Return up to `limit` files that are fast-synced (`embedded = 0`) and awaiting 768-D multimodal embedding,
    /// prioritized by importance (demo corpus first, then Documents/Text/PDFs/Images/Audio/Video, newest first),
    /// along with the total remaining count.
    pub fn pending_unembedded_files(&self, limit: usize) -> Result<(Vec<CrawledFile>, usize)> {
        let total_pending: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM files WHERE embedded = 0",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if total_pending == 0 || limit == 0 {
            return Ok((Vec::new(), 0));
        }

        let mut stmt = self.conn.prepare(
            "SELECT relative_path, file_name, kind, size_bytes, modified_unix
             FROM files
             WHERE embedded = 0
             ORDER BY
                 CASE WHEN relative_path LIKE '/tmp/%' THEN 1 ELSE 0 END ASC,
                 modified_unix DESC
             LIMIT ?1",
        )?;

        let rows = stmt.query_map(params![limit as i64], |row| {
            let rel: String = row.get(0)?;
            let fname: String = row.get(1)?;
            let kind_str: String = row.get(2)?;
            let size_bytes = row.get::<_, i64>(3)? as u64;
            let modified_unix: i64 = row.get(4)?;
            let abs_path = self.root.join(&rel);
            let ext = abs_path
                .extension()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            Ok(CrawledFile {
                path: abs_path,
                relative_path: rel,
                file_name: fname,
                extension: ext,
                kind: SearchKind::from_str(&kind_str).unwrap_or(SearchKind::Text),
                size_bytes,
                modified_unix,
            })
        })?;

        let mut out = Vec::new();
        for r in rows {
            if let Ok(cf) = r {
                out.push(cf);
            }
        }
        Ok((out, total_pending as usize))
    }

    pub fn remove_file(&mut self, file_id: i64) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?1)",
            params![file_id],
        )?;
        tx.execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
        tx.execute("DELETE FROM files WHERE id = ?1", params![file_id])?;
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_file_with_chunks(
        &mut self,
        relative_path: &str,
        file_name: &str,
        kind: SearchKind,
        size_bytes: u64,
        modified_unix: i64,
        chunks: Vec<NewChunkRecord>,
    ) -> Result<i64> {
        if let Some(existing_id) = self
            .conn
            .query_row(
                "SELECT id FROM files WHERE relative_path = ?1",
                params![relative_path],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
        {
            self.remove_file(existing_id)?;
        }

        // Collect all vectors first and append to VectorStore
        let mut vec_batch = Vec::new();
        let mut chunk_slots = Vec::with_capacity(chunks.len());
        for c in &chunks {
            if let Some(ref v) = c.vector {
                chunk_slots.push(Some(vec_batch.len()));
                vec_batch.push(v.clone());
            } else {
                chunk_slots.push(None);
            }
        }
        let base_slot = self.vectors.append_batch(&vec_batch)?;

        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO files (relative_path, file_name, kind, size_bytes, modified_unix, embedded)
             VALUES (?1, ?2, ?3, ?4, ?5, 1)",
            params![
                relative_path,
                file_name,
                kind.as_str(),
                size_bytes as i64,
                modified_unix
            ],
        )?;
        let file_id = tx.last_insert_rowid();

        let norm_title = SearchText::normalized(&format!("{} {}", file_name, relative_path));

        for (c, rel_slot) in chunks.into_iter().zip(chunk_slots.into_iter()) {
            let abs_slot = rel_slot.map(|s| (base_slot + s) as i64);
            let ahash_i64 = c.ahash.map(|h| h as i64);
            tx.execute(
                "INSERT INTO chunks (
                     file_id, chunk_index, modality, sets_scale, vector_slot, ahash,
                     text_content, page, line_start, line_end, start_seconds, end_seconds
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    file_id,
                    c.chunk_index as i64,
                    c.modality.as_str(),
                    if c.sets_scale { 1i64 } else { 0i64 },
                    abs_slot,
                    ahash_i64,
                    c.text_content,
                    c.origin.page.map(|p| p as i64),
                    c.origin.line_start.map(|l| l as i64),
                    c.origin.line_end.map(|l| l as i64),
                    c.origin.start_seconds,
                    c.origin.end_seconds,
                ],
            )?;
            let chunk_id = tx.last_insert_rowid();
            let norm_body = SearchText::normalized(&c.text_content);
            tx.execute(
                "INSERT INTO chunks_fts (rowid, title, body) VALUES (?1, ?2, ?3)",
                params![chunk_id, norm_title, norm_body],
            )?;
        }

        tx.commit()?;
        Ok(file_id)
    }

    /// Load all indexed chunks for fast in-memory hybrid scoring.
    pub fn load_all_chunks(&self) -> Result<Vec<IndexedChunkRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT
                 c.id, c.file_id, f.relative_path, f.file_name, f.kind,
                 f.size_bytes, f.modified_unix, c.modality, c.sets_scale,
                 c.vector_slot, c.ahash, c.text_content,
                 c.page, c.line_start, c.line_end, c.start_seconds, c.end_seconds
             FROM chunks c
             JOIN files f ON f.id = c.file_id
             ORDER BY c.id ASC",
        )?;

        let rows = stmt.query_map([], |row| {
            let kind_str: String = row.get(4)?;
            let mod_str: String = row.get(7)?;
            let sets_scale_int: i64 = row.get(8)?;
            let slot_opt: Option<i64> = row.get(9)?;
            let ahash_opt: Option<i64> = row.get(10)?;
            Ok(IndexedChunkRow {
                chunk_id: row.get(0)?,
                file_id: row.get(1)?,
                relative_path: row.get(2)?,
                file_name: row.get(3)?,
                kind: SearchKind::from_str(&kind_str).unwrap_or(SearchKind::Text),
                size_bytes: row.get::<_, i64>(5)? as u64,
                modified_unix: row.get(6)?,
                modality: ChunkModality::from_str(&mod_str),
                sets_scale: sets_scale_int != 0,
                vector_slot: slot_opt.map(|s| s as usize),
                ahash: ahash_opt.map(|h| h as u64),
                text_content: row.get(11)?,
                origin: ContentOrigin {
                    page: row.get::<_, Option<i64>>(12)?.map(|v| v as u32),
                    line_start: row.get::<_, Option<i64>>(13)?.map(|v| v as u32),
                    line_end: row.get::<_, Option<i64>>(14)?.map(|v| v as u32),
                    start_seconds: row.get(15)?,
                    end_seconds: row.get(16)?,
                },
            })
        })?;

        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Execute an FTS5 prefix query and return matching `chunk_id -> bm25_score` map.
    pub fn query_fts_matches(&self, query: &str) -> Result<HashMap<i64, f64>> {
        let tokens = SearchText::fts_tokens(query);
        let Some(match_expr) = SearchText::fts_match_query(&tokens) else {
            return Ok(HashMap::new());
        };

        let mut stmt = self.conn.prepare(
            "SELECT rowid, -bm25(chunks_fts, 2.0, 1.0)
             FROM chunks_fts
             WHERE chunks_fts MATCH ?1",
        )?;
        let rows = stmt.query_map(params![match_expr], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
        })?;

        let mut map = HashMap::new();
        for r in rows {
            if let Ok((id, score)) = r {
                map.insert(id, score.max(0.0));
            }
        }
        Ok(map)
    }

    pub fn stats(&self, elapsed_ms: f64) -> Result<IndexStats> {
        self.stats_scoped(None, elapsed_ms)
    }

    pub fn stats_scoped(&self, scope_prefix: Option<&str>, elapsed_ms: f64) -> Result<IndexStats> {
        let (indexed_files, total_chunks, text_chunks, image_chunks, audio_chunks) =
            if let Some(prefix) = scope_prefix {
                let like_pat = format!("{}%", prefix);
                let f_cnt: i64 = self.conn.query_row(
                    "SELECT COUNT(*) FROM files WHERE relative_path LIKE ?1",
                    params![like_pat],
                    |r| r.get(0),
                )?;
                let c_cnt: i64 = self.conn.query_row(
                    "SELECT COUNT(*) FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.relative_path LIKE ?1",
                    params![like_pat],
                    |r| r.get(0),
                )?;
                let t_cnt: i64 = self.conn.query_row(
                    "SELECT COUNT(*) FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.relative_path LIKE ?1 AND c.modality = 'text'",
                    params![like_pat],
                    |r| r.get(0),
                )?;
                let i_cnt: i64 = self.conn.query_row(
                    "SELECT COUNT(*) FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.relative_path LIKE ?1 AND c.modality = 'image'",
                    params![like_pat],
                    |r| r.get(0),
                )?;
                let a_cnt: i64 = self.conn.query_row(
                    "SELECT COUNT(*) FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.relative_path LIKE ?1 AND c.modality = 'audio'",
                    params![like_pat],
                    |r| r.get(0),
                )?;
                (f_cnt, c_cnt, t_cnt, i_cnt, a_cnt)
            } else {
                let f_cnt: i64 =
                    self.conn
                        .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
                let c_cnt: i64 =
                    self.conn
                        .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
                let t_cnt: i64 = self.conn.query_row(
                    "SELECT COUNT(*) FROM chunks WHERE modality = 'text'",
                    [],
                    |r| r.get(0),
                )?;
                let i_cnt: i64 = self.conn.query_row(
                    "SELECT COUNT(*) FROM chunks WHERE modality = 'image'",
                    [],
                    |r| r.get(0),
                )?;
                let a_cnt: i64 = self.conn.query_row(
                    "SELECT COUNT(*) FROM chunks WHERE modality = 'audio'",
                    [],
                    |r| r.get(0),
                )?;
                (f_cnt, c_cnt, t_cnt, i_cnt, a_cnt)
            };

        let db_bytes = fs::metadata(&self.db_path).map(|m| m.len()).unwrap_or(0);
        let root_label = scope_prefix
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.root.to_string_lossy().to_string());

        Ok(IndexStats {
            root: root_label,
            indexed_files: indexed_files as usize,
            total_chunks: total_chunks as usize,
            text_chunks: text_chunks as usize,
            image_chunks: image_chunks as usize,
            audio_chunks: audio_chunks as usize,
            vector_bytes: self.vectors.size_bytes(),
            db_bytes,
            elapsed_ms,
        })
    }

    pub fn kind_counts_scoped(&self, scope_prefix: Option<&str>) -> Result<HashMap<String, usize>> {
        let mut counts = HashMap::new();
        if let Some(prefix) = scope_prefix {
            let like_pat = format!("{}%", prefix);
            let mut stmt = self.conn.prepare(
                "SELECT kind, COUNT(*) FROM files WHERE relative_path LIKE ?1 GROUP BY kind",
            )?;
            let rows = stmt.query_map(params![like_pat], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize))
            })?;
            for row in rows.flatten() {
                counts.insert(row.0, row.1);
            }
        } else {
            let mut stmt = self
                .conn
                .prepare("SELECT kind, COUNT(*) FROM files GROUP BY kind")?;
            let rows = stmt.query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize))
            })?;
            for row in rows.flatten() {
                counts.insert(row.0, row.1);
            }
        }
        Ok(counts)
    }
}

fn extract_fast_preview_text(
    file: &CrawledFile,
) -> (String, Option<i64>, Option<i64>, Option<i64>) {
    if matches!(
        file.kind,
        SearchKind::Text | SearchKind::Code | SearchKind::Spreadsheet
    ) && file.size_bytes <= 1024 * 1024
    {
        if let Ok(mut f) = fs::File::open(&file.path) {
            let mut buf = vec![0u8; (file.size_bytes as usize).min(8192)];
            if let Ok(n) = f.read(&mut buf) {
                buf.truncate(n);
                let text = String::from_utf8_lossy(&buf);
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    let snippet: String = trimmed.chars().take(2400).collect();
                    let line_count = snippet.lines().count().max(1) as i64;
                    return (snippet, None, Some(1), Some(line_count));
                }
            }
        }
    }
    let parent_display = file
        .path
        .parent()
        .map(|p| {
            let s = p.to_string_lossy().to_string();
            if let Some(home) = dirs::home_dir() {
                let h = home.to_string_lossy();
                if s.starts_with(h.as_ref()) {
                    return format!("~{}", &s[h.len()..]);
                }
            }
            s
        })
        .unwrap_or_default();
    let label = format!("{} · {}", file.file_name, parent_display);
    match file.kind {
        SearchKind::Document | SearchKind::Presentation => (label, Some(1), None, None),
        SearchKind::Text | SearchKind::Code | SearchKind::Spreadsheet => {
            (label, None, Some(1), Some(1))
        }
        _ => (label, None, None, None),
    }
}


