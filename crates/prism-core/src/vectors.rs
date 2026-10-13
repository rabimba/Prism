use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use half::f16;
use memmap2::Mmap;
use prism_llama::LlamaEmbedder;
use rayon::prelude::*;

use crate::model::ChunkModality;

pub const VECTOR_DIM: usize = LlamaEmbedder::DIMENSION; // 768
pub const VECTOR_BYTES: usize = VECTOR_DIM * 2; // 1536 bytes per f16 vector

/// Memory-mapped Float16 vector store (`.prism.vectors.f16`).
pub struct VectorStore {
    path: PathBuf,
    file: File,
    mmap: Option<Mmap>,
    count: usize,
}

impl VectorStore {
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("Failed to open vector file {}", path.display()))?;

        let len = file.metadata()?.len() as usize;
        let count = len / VECTOR_BYTES;
        let mmap = if len >= VECTOR_BYTES {
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };

        Ok(Self {
            path: path.to_path_buf(),
            file,
            mmap,
            count,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn size_bytes(&self) -> u64 {
        (self.count * VECTOR_BYTES) as u64
    }

    pub fn clear(&mut self) -> Result<()> {
        self.mmap = None;
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.count = 0;
        Ok(())
    }

    /// Append a batch of 768-D f32 vectors as f16, returning their starting vector slot index.
    pub fn append_batch(&mut self, vectors: &[Vec<f32>]) -> Result<usize> {
        if vectors.is_empty() {
            return Ok(self.count);
        }
        let start_idx = self.count;
        let mut bytes = Vec::with_capacity(vectors.len() * VECTOR_BYTES);
        for v in vectors {
            assert_eq!(v.len(), VECTOR_DIM);
            for &x in v {
                let bits = f16::from_f32(x).to_bits();
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
        }

        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&bytes)?;
        self.file.flush()?;
        self.count += vectors.len();
        self.remap()?;
        Ok(start_idx)
    }

    pub fn remap(&mut self) -> Result<()> {
        let len = self.file.metadata()?.len() as usize;
        self.count = len / VECTOR_BYTES;
        if len >= VECTOR_BYTES {
            self.mmap = Some(unsafe { Mmap::map(&self.file)? });
        } else {
            self.mmap = None;
        }
        Ok(())
    }

    /// Read a single 768-D vector at `slot` back into f32.
    pub fn get_vector(&self, slot: usize) -> Option<Vec<f32>> {
        let mmap = self.mmap.as_ref()?;
        let start = slot.checked_mul(VECTOR_BYTES)?;
        let end = start.checked_add(VECTOR_BYTES)?;
        if end > mmap.len() {
            return None;
        }
        let slice = &mmap[start..end];
        let mut out = Vec::with_capacity(VECTOR_DIM);
        for chunk in slice.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            out.push(f16::from_bits(bits).to_f32());
        }
        Some(out)
    }

    /// Compute dot product (cosine similarity for L2-normalized vectors) between `query`
    /// and a specific `slot`.
    #[inline]
    pub fn dot_slot(&self, query: &[f32], slot: usize) -> f32 {
        let Some(mmap) = self.mmap.as_ref() else {
            return 0.0;
        };
        let start = slot * VECTOR_BYTES;
        let end = start + VECTOR_BYTES;
        if end > mmap.len() || query.len() != VECTOR_DIM {
            return 0.0;
        }
        dot_f32_f16_bytes(query, &mmap[start..end])
    }

    /// Score a list of vector slots against `query` in parallel.
    pub fn score_slots(&self, query: &[f32], slots: &[Option<usize>]) -> Vec<f32> {
        let Some(mmap) = self.mmap.as_ref() else {
            return vec![0.0; slots.len()];
        };
        let raw_bytes: &[u8] = &mmap[..];
        if slots.len() < 256 {
            slots
                .iter()
                .map(|opt| match opt {
                    Some(s) if (*s + 1) * VECTOR_BYTES <= raw_bytes.len() => {
                        let start = *s * VECTOR_BYTES;
                        dot_f32_f16_bytes(query, &raw_bytes[start..start + VECTOR_BYTES])
                    }
                    _ => 0.0,
                })
                .collect()
        } else {
            slots
                .par_iter()
                .map(|opt| match opt {
                    Some(s) if (*s + 1) * VECTOR_BYTES <= raw_bytes.len() => {
                        let start = *s * VECTOR_BYTES;
                        dot_f32_f16_bytes(query, &raw_bytes[start..start + VECTOR_BYTES])
                    }
                    _ => 0.0,
                })
                .collect()
        }
    }
}

#[inline]
fn dot_f32_f16_bytes(query: &[f32], bytes: &[u8]) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let mut i = 0usize;
    while i + 4 <= VECTOR_DIM {
        let b = i * 2;
        let v0 = f16::from_bits(u16::from_le_bytes([bytes[b], bytes[b + 1]])).to_f32();
        let v1 = f16::from_bits(u16::from_le_bytes([bytes[b + 2], bytes[b + 3]])).to_f32();
        let v2 = f16::from_bits(u16::from_le_bytes([bytes[b + 4], bytes[b + 5]])).to_f32();
        let v3 = f16::from_bits(u16::from_le_bytes([bytes[b + 6], bytes[b + 7]])).to_f32();

        sum0 += query[i] * v0;
        sum1 += query[i + 1] * v1;
        sum2 += query[i + 2] * v2;
        sum3 += query[i + 3] * v3;
        i += 4;
    }
    (sum0 + sum1) + (sum2 + sum3)
}

/// Exact port of `Searcher.meaning(scores:modalities:setsScale:)`:
/// Centers each similarity score by its modality's mean (when modality count >= 20, else global mean)
/// and normalizes by the global standard deviation (`min_std_dev = 0.02`).
pub fn compute_modality_z_scores(
    scores: &[f32],
    modalities: &[ChunkModality],
    sets_scale: &[bool],
) -> Vec<f64> {
    assert_eq!(scores.len(), modalities.len());
    assert_eq!(scores.len(), sets_scale.len());
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }

    const MIN_MODALITY_COUNT: usize = 20;
    const MIN_STD_DEV: f64 = 0.02;

    let mut sums = [0.0f64; 4];
    let mut counts = [0usize; 4];

    for i in 0..n {
        if sets_scale[i] && modalities[i] != ChunkModality::None {
            let m = modality_index(modalities[i]);
            sums[m] += scores[i] as f64;
            counts[m] += 1;
        }
    }

    let total_count: usize = counts.iter().sum();
    if total_count == 0 {
        return vec![0.0; n];
    }

    let global_mean: f64 = sums.iter().sum::<f64>() / (total_count as f64);
    let mut means = [global_mean; 4];
    for m in 0..4 {
        if counts[m] >= MIN_MODALITY_COUNT {
            means[m] = sums[m] / (counts[m] as f64);
        }
    }

    let mut centered = vec![0.0f64; n];
    for i in 0..n {
        if modalities[i] != ChunkModality::None {
            centered[i] = (scores[i] as f64) - means[modality_index(modalities[i])];
        }
    }

    let mut mean = 0.0f64;
    for i in 0..n {
        if sets_scale[i] && modalities[i] != ChunkModality::None {
            mean += centered[i];
        }
    }
    mean /= total_count as f64;

    let mut variance = 0.0f64;
    for i in 0..n {
        if sets_scale[i] && modalities[i] != ChunkModality::None {
            let d = centered[i] - mean;
            variance += d * d;
        }
    }
    variance /= total_count as f64;
    let std_dev = variance.sqrt().max(MIN_STD_DEV);

    let mut z = vec![0.0f64; n];
    for i in 0..n {
        if modalities[i] != ChunkModality::None {
            z[i] = (centered[i] - mean) / std_dev;
        }
    }
    z
}

#[inline]
fn modality_index(m: ChunkModality) -> usize {
    match m {
        ChunkModality::None => 0,
        ChunkModality::Text => 1,
        ChunkModality::Image => 2,
        ChunkModality::Audio => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_store_roundtrip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("test.vectors.f16");
        let mut store = VectorStore::open(&path)?;

        let mut v1 = vec![0.0f32; VECTOR_DIM];
        v1[0] = 0.6;
        v1[1] = 0.8;
        let slot = store.append_batch(&[v1.clone()])?;
        assert_eq!(slot, 0);
        assert_eq!(store.len(), 1);

        let dot = store.dot_slot(&v1, 0);
        assert!((dot - 1.0).abs() < 1e-2);
        Ok(())
    }
}
