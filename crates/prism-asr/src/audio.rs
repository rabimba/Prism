use std::io::Cursor;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};

pub const SAMPLE_RATE: u32 = 16_000;
pub const MAX_WINDOW_SECONDS: f64 = 30.0;
pub const MAX_WINDOW_SAMPLES: usize = (SAMPLE_RATE as usize) * 30;

/// Decode an audio or video file into 16 kHz mono float32 PCM samples in `[-1.0, 1.0]`.
pub fn load_audio_16k_mono(path: &Path) -> Result<Vec<f32>> {
    // Try fast native WAV reader first if `.wav`
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
    {
        if let Ok(samples) = load_wav_16k_mono(path) {
            if !samples.is_empty() {
                return Ok(samples);
            }
        }
    }

    load_ffmpeg_16k_mono(path, None, None)
}

/// Decode a specific `[start_s, start_s + duration_s]` slice of an audio/video file into 16 kHz mono float32.
pub fn load_ffmpeg_16k_mono(
    path: &Path,
    start_s: Option<f64>,
    duration_s: Option<f64>,
) -> Result<Vec<f32>> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-v", "error"]);
    if let Some(s) = start_s {
        if s > 0.0 {
            cmd.args(["-ss", &format!("{:.3}", s)]);
        }
    }
    cmd.arg("-i").arg(path);
    if let Some(d) = duration_s {
        if d > 0.0 {
            cmd.args(["-t", &format!("{:.3}", d)]);
        }
    }
    cmd.args([
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]);

    let output = cmd
        .output()
        .with_context(|| format!("Failed to invoke ffmpeg on {}", path.display()))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg failed on {}: {}", path.display(), err.trim());
    }
    Ok(bytes_to_f32_samples(&output.stdout))
}

/// Decode an in-memory WAV byte buffer into 16 kHz mono float32 samples.
pub fn decode_wav_bytes_16k_mono(bytes: &[u8]) -> Result<Vec<f32>> {
    let reader = hound::WavReader::new(Cursor::new(bytes)).context("Invalid WAV buffer")?;
    read_hound_to_16k_mono(reader)
}

fn load_wav_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    read_hound_to_16k_mono(reader)
}

fn read_hound_to_16k_mono<R: std::io::Read>(mut reader: hound::WavReader<R>) -> Result<Vec<f32>> {
    let spec = reader.spec();
    let channels = (spec.channels as usize).max(1);
    let raw_mono: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader.samples::<f32>().filter_map(|s| s.ok()).collect();
            downmix_to_mono(&samples, channels)
        }
        hound::SampleFormat::Int => {
            let max_val = ((1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32).max(1.0);
            let samples: Vec<f32> = reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|v| (v as f32) / max_val)
                .collect();
            downmix_to_mono(&samples, channels)
        }
    };

    if spec.sample_rate == SAMPLE_RATE {
        Ok(raw_mono)
    } else {
        Ok(resample_linear(&raw_mono, spec.sample_rate, SAMPLE_RATE))
    }
}

fn downmix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let mut out = Vec::with_capacity(samples.len() / channels);
    for frame in samples.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        out.push(sum / (channels as f32));
    }
    out
}

pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = (from_rate as f64) / (to_rate as f64);
    let out_len = ((samples.len() as f64) / ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - (idx as f64)) as f32;
        let s0 = samples.get(idx).copied().unwrap_or(0.0);
        let s1 = samples.get(idx + 1).copied().unwrap_or(s0);
        out.push(s0 + frac * (s1 - s0));
    }
    out
}

pub fn bytes_to_f32_samples(bytes: &[u8]) -> Vec<f32> {
    let count = bytes.len() / 4;
    let mut out = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

/// Record audio from the default system microphone for `duration_s` seconds into 16 kHz mono float32.
pub fn record_microphone_16k_mono(duration_s: f64) -> Result<Vec<f32>> {
    let dur = duration_s.clamp(1.0, 30.0);
    #[cfg(target_os = "macos")]
    let input_args = ["-f", "avfoundation", "-i", ":0"];
    #[cfg(not(target_os = "macos"))]
    let input_args = ["-f", "pulse", "-i", "default"];

    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error"])
        .args(input_args)
        .args([
            "-t",
            &format!("{:.2}", dur),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ])
        .output()
        .context("Failed to record microphone via ffmpeg")?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Microphone capture failed: {}", err.trim());
    }
    Ok(bytes_to_f32_samples(&output.stdout))
}
