use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct ModelFileSpec {
    pub name: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub model_path: PathBuf,
    pub mmproj_path: PathBuf,
}

pub struct ModelFiles;

impl ModelFiles {
    pub const REPO: &'static str = "ggml-org/embeddinggemma-2-GGUF";
    pub const REVISION: &'static str = "bfcd298762cc34d0357ece5ebdd31791a3a374d8";

    pub const FILES: [ModelFileSpec; 2] = [
        ModelFileSpec {
            name: "embeddinggemma-2-Q8_0.gguf",
            sha256: "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135",
            bytes: 309_855_456,
        },
        ModelFileSpec {
            name: "mmproj-embeddinggemma-2-Q8_0.gguf",
            sha256: "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f",
            bytes: 554_821_024,
        },
    ];

    pub const TOTAL_BYTES: u64 = 309_855_456 + 554_821_024;

    pub fn default_directory() -> PathBuf {
        if let Ok(custom) = std::env::var("PRISM_MODELS_DIR") {
            if !custom.trim().is_empty() {
                return PathBuf::from(custom);
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Some(home) = dirs::home_dir() {
                return home
                    .join("Library")
                    .join("Application Support")
                    .join("Prism")
                    .join("models");
            }
        }
        if let Some(data_dir) = dirs::data_local_dir() {
            return data_dir.join("prism").join("models");
        }
        PathBuf::from(".prism-models")
    }

    pub fn paths_in(directory: &Path) -> ModelPaths {
        ModelPaths {
            model_path: directory.join(Self::FILES[0].name),
            mmproj_path: directory.join(Self::FILES[1].name),
        }
    }

    pub fn is_ready(directory: Option<&Path>) -> bool {
        let dir = directory
            .map(|p| p.to_path_buf())
            .unwrap_or_else(Self::default_directory);
        for spec in &Self::FILES {
            let path = dir.join(spec.name);
            let Ok(meta) = fs::metadata(&path) else {
                return false;
            };
            if meta.len() != spec.bytes {
                return false;
            }
        }
        true
    }

    pub fn download_url(file_name: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}?download=true",
            Self::REPO,
            Self::REVISION,
            file_name
        )
    }

    pub fn ensure(
        directory: Option<&Path>,
        mut progress: Option<&mut dyn FnMut(u64, u64, &str)>,
    ) -> Result<ModelPaths> {
        let dir = directory
            .map(|p| p.to_path_buf())
            .unwrap_or_else(Self::default_directory);
        fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create models directory {}", dir.display()))?;

        let mut completed: u64 = 0;
        for spec in &Self::FILES {
            let dest = dir.join(spec.name);
            if let Ok(meta) = fs::metadata(&dest) {
                if meta.len() == spec.bytes {
                    completed += spec.bytes;
                    if let Some(cb) = progress.as_mut() {
                        cb(completed, Self::TOTAL_BYTES, spec.name);
                    }
                    continue;
                }
            }

            let part = dir.join(format!("{}.part", spec.name));
            let _ = fs::remove_file(&part);
            let url = Self::download_url(spec.name);
            let response = ureq::get(&url)
                .call()
                .with_context(|| format!("Failed to download {}", url))?;

            let mut reader = response.into_reader();
            let mut out = File::create(&part)
                .with_context(|| format!("Failed to create {}", part.display()))?;
            let mut hasher = Sha256::new();
            let mut buf = vec![0u8; 256 * 1024];
            let mut file_written: u64 = 0;

            loop {
                let n = reader.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n])?;
                hasher.update(&buf[..n]);
                file_written += n as u64;
                if let Some(cb) = progress.as_mut() {
                    cb(completed + file_written, Self::TOTAL_BYTES, spec.name);
                }
            }
            out.flush()?;
            drop(out);

            let digest = hex::encode(hasher.finalize());
            if !digest.eq_ignore_ascii_case(spec.sha256) {
                let _ = fs::remove_file(&part);
                anyhow::bail!(
                    "SHA-256 mismatch for {}: expected {}, got {}",
                    spec.name,
                    spec.sha256,
                    digest
                );
            }

            fs::rename(&part, &dest)?;
            completed += spec.bytes;
        }

        Ok(Self::paths_in(&dir))
    }
}
