use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

use crate::model::{ChunkModality, ContentOrigin, IndexStats, SearchKind};
use crate::search_text::SearchText;
use crate::vectors::VectorStore;

pub const DB_FILENAME: &str = ".prism.db";
pub const VECTORS_FILENAME: &str = ".prism.vectors.f16";

#[derive(Debug, Clone)]
pub struct StoredFileRecord {
    pub id: i64,
    pub relative_path: String,
    pub file_name: String,
    pub kind: SearchKind,
    pub size_bytes: u64,
    pub modified_unix: i64,
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
    pub fn open(root: &Path) -> Result<Self> {
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
                 modified_unix INTEGER NOT NULL
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
            "SELECT id, relative_path, file_name, kind, size_bytes, modified_unix FROM files",
        )?;
        let rows = stmt.query_map([], |row| {
            let kind_str: String = row.get(3)?;
            Ok(StoredFileRecord {
                id: row.get(0)?,
                relative_path: row.get(1)?,
                file_name: row.get(2)?,
                kind: SearchKind::from_str(&kind_str).unwrap_or(SearchKind::Text),
                size_bytes: row.get::<_, i64>(4)? as u64,
                modified_unix: row.get(5)?,
            })
        })?;
        let mut map = HashMap::new();
        for r in rows {
            let rec = r?;
            map.insert(rec.relative_path.clone(), rec);
        }
        Ok(map)
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
            "INSERT INTO files (relative_path, file_name, kind, size_bytes, modified_unix)
             VALUES (?1, ?2, ?3, ?4, ?5)",
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
        let indexed_files: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        let total_chunks: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        let text_chunks: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE modality = 'text'",
            [],
            |r| r.get(0),
        )?;
        let image_chunks: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE modality = 'image'",
            [],
            |r| r.get(0),
        )?;
        let audio_chunks: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE modality = 'audio'",
            [],
            |r| r.get(0),
        )?;
        let db_bytes = fs::metadata(&self.db_path).map(|m| m.len()).unwrap_or(0);

        Ok(IndexStats {
            root: self.root.to_string_lossy().to_string(),
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
}
