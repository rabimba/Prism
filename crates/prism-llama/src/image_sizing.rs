use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};

/// Matches Gemma 4 vision patch size (16) * pooling kernel (3) = 48px alignment.
pub struct ImageSizing;

impl ImageSizing {
    pub const PATCH: usize = 16;
    pub const POOLING: usize = 3;
    pub const SIDE: usize = Self::PATCH * Self::POOLING; // 48
    pub const MAX_ASPECT_RATIO: f64 = 200.0;
    pub const VALID_BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];

    pub const PHOTO_BUDGET: usize = 140;
    pub const SCREENSHOT_BUDGET: usize = 280;
    pub const PAGE_BUDGET: usize = 280;
    pub const FRAME_BUDGET: usize = 70;

    /// Aspect-preserving dimensions for `width x height` within `budget` soft tokens.
    /// Exact port of `ImageSizing.size(width:height:budget:)` in Swift / HuggingFace Gemma4ImageProcessor.
    pub fn size(width: usize, height: usize, budget: usize) -> Result<(usize, usize)> {
        if !Self::VALID_BUDGETS.contains(&budget) {
            anyhow::bail!("Invalid token budget: {}", budget);
        }
        if width == 0 || height == 0 {
            anyhow::bail!("Invalid image dimensions: {}x{}", width, height);
        }
        let ratio = (width.max(height) as f64) / (width.min(height) as f64);
        if ratio > Self::MAX_ASPECT_RATIO {
            anyhow::bail!("Extreme aspect ratio: {:.2}", ratio);
        }

        let target_px = budget * (Self::POOLING * Self::POOLING) * (Self::PATCH * Self::PATCH);
        let factor = ((target_px as f64) / ((height as f64) * (width as f64))).sqrt();

        let ideal_h = factor * (height as f64);
        let ideal_w = factor * (width as f64);
        let side = Self::SIDE;

        let mut h = ((ideal_h / (side as f64)).floor() as usize) * side;
        let mut w = ((ideal_w / (side as f64)).floor() as usize) * side;

        if h == 0 && w == 0 {
            anyhow::bail!("Dimensions too small after quantization: {}x{}", width, height);
        }

        let max_side = budget * side;
        if h == 0 {
            h = side;
            w = ((width / height) * side).min(max_side);
        } else if w == 0 {
            w = side;
            h = ((height / width) * side).min(max_side);
        }

        if h * w > target_px {
            anyhow::bail!("Quantized size {}x{} exceeds target_px {}", w, h, target_px);
        }

        Ok((w, h))
    }

    /// Resize interleaved RGB8 (`width * height * 3` bytes) to `out_width x out_height`
    /// using Pillow's separable Mitchell-Netravali / Catmull-Rom bicubic filter (`a = -0.5`)
    /// with 22-bit fixed-point arithmetic.
    pub fn resize_rgb(
        rgb: &[u8],
        width: usize,
        height: usize,
        out_width: usize,
        out_height: usize,
    ) -> Vec<u8> {
        assert_eq!(rgb.len(), width * height * 3);
        if width == out_width && height == out_height {
            return rgb.to_vec();
        }

        let horiz = Self::make_table(width, out_width);
        let vert = Self::make_table(height, out_height);

        // Pass 1: horizontal (width -> out_width) for each of the `height` rows.
        let mut temp = vec![0u8; out_width * height * 3];
        let h_size = horiz.ksize;
        for y in 0..height {
            let row_in = y * width * 3;
            let row_out = y * out_width * 3;
            for x in 0..out_width {
                let xmin = horiz.bounds[x * 2] as usize;
                let xmax = horiz.bounds[x * 2 + 1] as usize;
                let k_off = x * h_size;
                let mut ss0: i32 = 1 << 21;
                let mut ss1: i32 = 1 << 21;
                let mut ss2: i32 = 1 << 21;
                for xx in 0..xmax {
                    let w = horiz.kk[k_off + xx];
                    let p = row_in + (xmin + xx) * 3;
                    ss0 = ss0.wrapping_add((rgb[p] as i32).wrapping_mul(w));
                    ss1 = ss1.wrapping_add((rgb[p + 1] as i32).wrapping_mul(w));
                    ss2 = ss2.wrapping_add((rgb[p + 2] as i32).wrapping_mul(w));
                }
                let d = row_out + x * 3;
                temp[d] = Self::clip8(ss0);
                temp[d + 1] = Self::clip8(ss1);
                temp[d + 2] = Self::clip8(ss2);
            }
        }

        if height == out_height {
            return temp;
        }

        // Pass 2: vertical (height -> out_height) across `out_width * 3` columns.
        let mut out = vec![0u8; out_width * out_height * 3];
        let v_size = vert.ksize;
        let stride = out_width * 3;
        for y in 0..out_height {
            let ymin = vert.bounds[y * 2] as usize;
            let ymax = vert.bounds[y * 2 + 1] as usize;
            let k_off = y * v_size;
            let row_out = y * stride;
            for x in 0..out_width {
                let mut ss0: i32 = 1 << 21;
                let mut ss1: i32 = 1 << 21;
                let mut ss2: i32 = 1 << 21;
                let col = x * 3;
                for yy in 0..ymax {
                    let w = vert.kk[k_off + yy];
                    let p = (ymin + yy) * stride + col;
                    ss0 = ss0.wrapping_add((temp[p] as i32).wrapping_mul(w));
                    ss1 = ss1.wrapping_add((temp[p + 1] as i32).wrapping_mul(w));
                    ss2 = ss2.wrapping_add((temp[p + 2] as i32).wrapping_mul(w));
                }
                let d = row_out + col;
                out[d] = Self::clip8(ss0);
                out[d + 1] = Self::clip8(ss1);
                out[d + 2] = Self::clip8(ss2);
            }
        }

        out
    }

    #[inline]
    fn bicubic(x: f64) -> f64 {
        let a = -0.5_f64;
        let ax = x.abs();
        if ax < 1.0 {
            ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0
        } else if ax < 2.0 {
            (((ax - 5.0) * ax + 8.0) * ax - 4.0) * a
        } else {
            0.0
        }
    }

    #[inline]
    fn clip8(v: i32) -> u8 {
        let shifted = v >> 22;
        if shifted <= 0 {
            0
        } else if shifted >= 255 {
            255
        } else {
            shifted as u8
        }
    }

    fn make_table(in_size: usize, out_size: usize) -> FilterTable {
        let scale = (in_size as f64) / (out_size as f64);
        let filter_scale = scale.max(1.0);
        let support = 2.0 * filter_scale;
        let ksize = (support.ceil() as usize) * 2 + 1;

        let mut bounds = vec![0i32; out_size * 2];
        let mut kk = vec![0i32; out_size * ksize];
        let mut weights = vec![0.0f64; ksize];
        let scale_fixed = (1i64 << 22) as f64;

        for out in 0..out_size {
            let center = ((out as f64) + 0.5) * scale;
            let first = ((center - support + 0.5) as isize).max(0) as usize;
            let count = ((center + support + 0.5) as isize)
                .min(in_size as isize)
                .saturating_sub(first as isize) as usize;

            let mut total = 0.0_f64;
            for x in 0..ksize {
                weights[x] = if x < count {
                    Self::bicubic(((x + first) as f64 - center + 0.5) / filter_scale)
                } else {
                    0.0
                };
                total += weights[x];
            }

            let k_base = out * ksize;
            for x in 0..ksize {
                let w = if x < count && total != 0.0 {
                    weights[x] / total
                } else {
                    weights[x]
                };
                let fixed = (w * scale_fixed + if w < 0.0 { -0.5 } else { 0.5 }) as i32;
                kk[k_base + x] = fixed;
            }
            bounds[out * 2] = first as i32;
            bounds[out * 2 + 1] = count as i32;
        }

        FilterTable {
            ksize,
            bounds,
            kk,
        }
    }

    /// Compute a 64-bit 8x8 perceptual Average Hash (aHash) from interleaved RGB bytes.
    pub fn compute_ahash(rgb: &[u8], width: usize, height: usize) -> u64 {
        if width == 0 || height == 0 || rgb.len() < width * height * 3 {
            return 0;
        }
        let small = Self::resize_rgb(rgb, width, height, 8, 8);
        let mut luma = [0u32; 64];
        let mut sum = 0u32;
        for i in 0..64 {
            let r = small[i * 3] as u32;
            let g = small[i * 3 + 1] as u32;
            let b = small[i * 3 + 2] as u32;
            let y = (r * 77 + g * 150 + b * 29) >> 8;
            luma[i] = y;
            sum += y;
        }
        let avg = sum / 64;
        let mut hash = 0u64;
        for (i, &y) in luma.iter().enumerate() {
            if y >= avg {
                hash |= 1u64 << i;
            }
        }
        hash
    }

    /// Normalized Hamming similarity in [0.0, 1.0] between two 64-bit perceptual hashes.
    pub fn ahash_similarity(a: u64, b: u64) -> f32 {
        let dist = (a ^ b).count_ones();
        1.0 - (dist as f32 / 64.0)
    }

    /// Apply EXIF orientation (1..=8) to a DynamicImage.
    pub fn apply_exif_orientation(img: DynamicImage, orientation: u32) -> DynamicImage {
        match orientation {
            2 => img.fliph(),
            3 => img.rotate180(),
            4 => img.flipv(),
            5 => img.rotate90().fliph(),
            6 => img.rotate90(),
            7 => img.rotate270().fliph(),
            8 => img.rotate270(),
            _ => img,
        }
    }

    /// Read EXIF orientation tag (1..=8) from a file if present.
    pub fn read_exif_orientation(path: &Path) -> u32 {
        let Ok(file) = File::open(path) else {
            return 1;
        };
        let mut bufreader = BufReader::new(&file);
        let exifreader = exif::Reader::new();
        let Ok(exif) = exifreader.read_from_container(&mut bufreader) else {
            return 1;
        };
        if let Some(field) = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY) {
            if let Some(v) = field.value.get_uint(0) {
                return v;
            }
        }
        1
    }

    /// Load an image from disk, apply EXIF orientation, resize to the target token budget
    /// using Pillow-compatible bicubic resampling, and return `(resized_rgb, out_w, out_h, ahash)`.
    pub fn load_and_prepare_file(path: &Path, budget: usize) -> Result<(Vec<u8>, usize, usize, u64)> {
        let orientation = Self::read_exif_orientation(path);
        let decoded = Self::decode_image_with_fallback(path)?;
        let oriented = Self::apply_exif_orientation(decoded, orientation);
        let (orig_w, orig_h) = oriented.dimensions();
        let rgb = oriented.to_rgb8().into_raw();
        let ahash = Self::compute_ahash(&rgb, orig_w as usize, orig_h as usize);
        let (out_w, out_h) = Self::size(orig_w as usize, orig_h as usize, budget)?;
        let resized = Self::resize_rgb(&rgb, orig_w as usize, orig_h as usize, out_w, out_h);
        Ok((resized, out_w, out_h, ahash))
    }

    /// Decode an image using the `image` crate, falling back to `ffmpeg` for container formats
    /// like HEIC/AVIF when native decoders are not compiled in.
    pub fn decode_image_with_fallback(path: &Path) -> Result<DynamicImage> {
        if let Ok(reader) = image::ImageReader::open(path) {
            if let Ok(with_guessed) = reader.with_guessed_format() {
                if let Ok(img) = with_guessed.decode() {
                    return Ok(img);
                }
            }
        }
        let output = std::process::Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-i",
                &path.to_string_lossy(),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ])
            .output()
            .with_context(|| format!("Failed to decode image {}", path.display()))?;
        if !output.status.success() || output.stdout.is_empty() {
            anyhow::bail!("Failed to decode image {}", path.display());
        }
        let img = image::load_from_memory(&output.stdout)
            .with_context(|| format!("Failed to load ffmpeg-decoded image {}", path.display()))?;
        Ok(img)
    }

    /// Load an image from in-memory bytes and prepare for embedding.
    pub fn load_and_prepare_bytes(bytes: &[u8], budget: usize) -> Result<(Vec<u8>, usize, usize, u64)> {
        let mut orientation = 1;
        let exifreader = exif::Reader::new();
        if let Ok(exif) = exifreader.read_from_container(&mut Cursor::new(bytes)) {
            if let Some(field) = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY) {
                if let Some(v) = field.value.get_uint(0) {
                    orientation = v;
                }
            }
        }
        let decoded = image::load_from_memory(bytes).context("Failed to decode image bytes")?;
        let oriented = Self::apply_exif_orientation(decoded, orientation);
        let (orig_w, orig_h) = oriented.dimensions();
        let rgb = oriented.to_rgb8().into_raw();
        let ahash = Self::compute_ahash(&rgb, orig_w as usize, orig_h as usize);
        let (out_w, out_h) = Self::size(orig_w as usize, orig_h as usize, budget)?;
        let resized = Self::resize_rgb(&rgb, orig_w as usize, orig_h as usize, out_w, out_h);
        Ok((resized, out_w, out_h, ahash))
    }
}

struct FilterTable {
    ksize: usize,
    bounds: Vec<i32>,
    kk: Vec<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_7X5: [u8; 105] = [
        0, 115, 230, 3, 118, 233, 6, 121, 236, 9, 124, 239, 12, 127, 242, 15, 130, 245, 18, 133,
        248, 77, 192, 51, 80, 195, 54, 83, 198, 57, 86, 201, 60, 89, 204, 63, 92, 207, 66, 95,
        210, 69, 154, 13, 128, 157, 16, 131, 160, 19, 134, 163, 22, 137, 166, 25, 140, 169, 28,
        143, 172, 31, 146, 231, 90, 205, 234, 93, 208, 237, 96, 211, 240, 99, 214, 243, 102, 217,
        246, 105, 220, 249, 108, 223, 52, 167, 26, 55, 170, 29, 58, 173, 32, 61, 176, 35, 64, 179,
        38, 67, 182, 41, 70, 185, 44,
    ];

    #[test]
    fn test_aspect_preserving_dimensions() {
        assert_eq!(ImageSizing::size(1440, 900, 280).unwrap(), (1008, 624));
        assert_eq!(ImageSizing::size(1440, 900, 140).unwrap(), (672, 432));
        assert_eq!(ImageSizing::size(1440, 900, 70).unwrap(), (480, 288));
        assert_eq!(ImageSizing::size(768, 768, 280).unwrap(), (768, 768));
        assert_eq!(ImageSizing::size(768, 768, 140).unwrap(), (528, 528));
        assert_eq!(ImageSizing::size(300, 300, 140).unwrap(), (528, 528));
    }

    #[test]
    fn test_invalid_inputs() {
        assert!(ImageSizing::size(0, 100, 280).is_err());
        assert!(ImageSizing::size(100, 0, 280).is_err());
        assert!(ImageSizing::size(100, 100, 999).is_err());
        assert!(ImageSizing::size(10000, 10, 280).is_err());
    }

    #[test]
    fn test_bicubic_resize_matches_pillow() {
        let down = ImageSizing::resize_rgb(&SOURCE_7X5, 7, 5, 4, 3);
        let expected_down: [u8; 36] = [
            28, 151, 152, 33, 156, 157, 39, 162, 163, 44, 167, 168, 165, 68, 129, 170, 73, 134,
            176, 79, 140, 181, 84, 145, 132, 133, 106, 137, 138, 111, 143, 144, 117, 148, 149, 122,
        ];
        assert_eq!(down, expected_down);

        let up = ImageSizing::resize_rgb(&SOURCE_7X5, 7, 5, 10, 6);
        let expected_up: [u8; 180] = [
            0, 112, 237, 0, 113, 238, 1, 116, 241, 3, 118, 243, 5, 120, 245, 7, 122, 247, 9, 124,
            249, 11, 126, 251, 14, 129, 254, 15, 130, 255, 55, 187, 85, 56, 188, 86, 59, 191, 89,
            61, 193, 91, 63, 195, 93, 65, 197, 95, 67, 199, 97, 69, 201, 99, 72, 204, 102, 73, 205,
            103, 122, 82, 83, 123, 83, 84, 126, 86, 87, 128, 88, 89, 130, 90, 91, 132, 92, 93, 134,
            94, 95, 136, 96, 97, 139, 99, 100, 140, 100, 101, 199, 27, 173, 200, 28, 174, 203, 31,
            177, 205, 33, 179, 207, 35, 181, 209, 37, 183, 211, 39, 185, 213, 41, 187, 216, 44,
            190, 217, 45, 191, 197, 112, 171, 198, 113, 172, 201, 116, 175, 203, 118, 177, 205,
            120, 179, 207, 122, 181, 209, 124, 183, 211, 126, 185, 214, 129, 188, 215, 130, 189,
            45, 170, 19, 46, 171, 20, 49, 174, 23, 51, 176, 25, 53, 178, 27, 55, 180, 29, 57, 182,
            31, 59, 184, 33, 62, 187, 36, 63, 188, 37,
        ];
        assert_eq!(up, expected_up);
    }
}
