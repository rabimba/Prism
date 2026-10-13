use crate::audio::SAMPLE_RATE;

#[derive(Debug, Clone)]
pub struct AcousticWindow {
    pub start_s: f64,
    pub end_s: f64,
    pub rms: f32,
    pub zcr: f32,
    pub samples: Vec<f32>,
}

/// Compute root-mean-square energy of a PCM window.
pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    ((sum_sq / (samples.len() as f64)).sqrt()) as f32
}

/// Compute zero-crossing rate in [0.0, 1.0].
pub fn compute_zcr(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0usize;
    for pair in samples.windows(2) {
        if (pair[0] >= 0.0) != (pair[1] >= 0.0) {
            crossings += 1;
        }
    }
    (crossings as f32) / ((samples.len() - 1) as f32)
}

/// Slice 16 kHz mono PCM audio into overlapping speech-active acoustic windows for
/// EmbeddingGemma 2 Semantic-Space ASR and temporal audio alignment.
pub fn slice_acoustic_windows(
    samples: &[f32],
    window_s: f64,
    hop_s: f64,
    max_windows: usize,
) -> Vec<AcousticWindow> {
    if samples.is_empty() {
        return Vec::new();
    }

    let sr = SAMPLE_RATE as f64;
    let win_len = ((window_s * sr).round() as usize).max(8_000); // at least 0.5s
    let hop_len = ((hop_s * sr).round() as usize).max(4_000); // at least 0.25s
    let min_samples = 9_600; // 0.6s minimum for mtmd audio encoder stability

    let global_rms = compute_rms(samples);
    let silence_threshold = (global_rms * 0.18).max(0.0025);

    if samples.len() <= win_len {
        let mut padded = samples.to_vec();
        if padded.len() < min_samples {
            padded.resize(min_samples, 0.0);
        }
        return vec![AcousticWindow {
            start_s: 0.0,
            end_s: (samples.len() as f64) / sr,
            rms: global_rms,
            zcr: compute_zcr(samples),
            samples: padded,
        }];
    }

    let mut windows = Vec::new();
    let mut pos = 0usize;
    while pos < samples.len() {
        let end = (pos + win_len).min(samples.len());
        let slice = &samples[pos..end];
        let rms = compute_rms(slice);
        if rms >= silence_threshold || windows.is_empty() {
            let mut buf = slice.to_vec();
            if buf.len() < min_samples {
                buf.resize(min_samples, 0.0);
            }
            windows.push(AcousticWindow {
                start_s: (pos as f64) / sr,
                end_s: (end as f64) / sr,
                rms,
                zcr: compute_zcr(slice),
                samples: buf,
            });
        }
        if end == samples.len() {
            break;
        }
        pos += hop_len;
    }

    if windows.len() > max_windows && max_windows > 0 {
        // Subsample evenly across time while keeping highest-energy speech windows
        let step = (windows.len() as f64) / (max_windows as f64);
        let mut selected = Vec::with_capacity(max_windows);
        for i in 0..max_windows {
            let idx = ((i as f64) * step).floor() as usize;
            if let Some(w) = windows.get(idx) {
                selected.push(w.clone());
            }
        }
        return selected;
    }

    windows
}

/// Slice a long audio stream into 30-second windows for indexing (`MAX_WINDOW_SAMPLES`).
pub fn slice_30s_index_windows(samples: &[f32]) -> Vec<(f64, f64, Vec<f32>)> {
    if samples.is_empty() {
        return Vec::new();
    }
    let sr = SAMPLE_RATE as f64;
    let win_len = (30.0 * sr) as usize;
    let min_samples = 9_600;
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < samples.len() {
        let end = (pos + win_len).min(samples.len());
        let slice = &samples[pos..end];
        if compute_rms(slice) >= 0.001 || out.is_empty() {
            let mut buf = slice.to_vec();
            if buf.len() < min_samples {
                buf.resize(min_samples, 0.0);
            }
            out.push(((pos as f64) / sr, (end as f64) / sr, buf));
        }
        pos = end;
    }
    out
}
