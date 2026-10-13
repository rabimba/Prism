use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

use crate::model::Landmark;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageTextProblem {
    Empty,
    ReplacementChars,
    ControlChars,
    LowLetterDensity,
    FragmentedTokens,
}

pub struct SearchText;

impl SearchText {
    /// Normalize text for indexing and querying:
    /// - NFKC compatibility composition
    /// - Convert U+200B (ZWSP) to space
    /// - Strip zero-width joiners, directional marks, soft hyphens, and BOM
    /// - Strip Arabic tatweel & harakat, fold alef variants to bare alef (U+0627)
    pub fn normalized(text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let nfkc: String = text.nfkc().collect();
        let mut out = String::with_capacity(nfkc.len());
        for c in nfkc.chars() {
            let v = c as u32;
            if v == 0x200B {
                out.push(' ');
                continue;
            }
            if Self::is_stripped_format_scalar(v) || Self::is_arabic_ignorable(v) {
                continue;
            }
            if Self::is_alef_variant(v) {
                out.push('\u{0627}');
                continue;
            }
            out.push(c);
        }
        out
    }

    #[inline]
    fn is_stripped_format_scalar(v: u32) -> bool {
        matches!(
            v,
            0x00AD
                | 0x061C
                | 0x200C..=0x200F
                | 0x202A..=0x202E
                | 0x2060..=0x2064
                | 0x2066..=0x2069
                | 0xFEFF
        )
    }

    #[inline]
    fn is_arabic_ignorable(v: u32) -> bool {
        v == 0x0640 || (0x064B..=0x065F).contains(&v) || v == 0x0670
    }

    #[inline]
    fn is_alef_variant(v: u32) -> bool {
        matches!(v, 0x0622 | 0x0623 | 0x0625 | 0x0671)
    }

    #[inline]
    pub fn is_token_char(c: char) -> bool {
        if c.is_alphanumeric() {
            return true;
        }
        let v = c as u32;
        // Include Unicode combining marks (M*) and Private Use (Co) so Indic/Bengali/Arabic/CJK clusters stay intact
        matches!(
            v,
            0x0300..=0x036F
                | 0x0591..=0x05C7
                | 0x0610..=0x061A
                | 0x064B..=0x065F
                | 0x0670
                | 0x06D6..=0x06ED
                | 0x0900..=0x0903
                | 0x093A..=0x094F
                | 0x0951..=0x0957
                | 0x0962..=0x0963
                | 0x0981..=0x0983
                | 0x09BC..=0x09CD
                | 0x09D7
                | 0x09E2..=0x09E3
                | 0xE000..=0xF8FF
        )
    }

    /// Tokenize a query string using the same rules as the FTS5 unicode61 configuration.
    pub fn fts_tokens(query: &str) -> Vec<String> {
        let prepared = Self::normalized(query).to_lowercase();
        let mut tokens = Vec::new();
        let mut current = String::new();
        for c in prepared.chars() {
            if Self::is_token_char(c) {
                current.push(c);
            } else if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// Build a safe FTS5 MATCH expression where any token matches as a prefix.
    pub fn fts_match_query(tokens: &[String]) -> Option<String> {
        if tokens.is_empty() {
            return None;
        }
        let parts: Vec<String> = tokens
            .iter()
            .map(|t| {
                let escaped = t.replace('"', "\"\"");
                format!("\"{}\"*", escaped)
            })
            .collect();
        Some(parts.join(" OR "))
    }

    /// Extract a concise snippet centered around the best matching line for `query`.
    pub fn snippet(text: &str, query: &str, max_chars: usize) -> String {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return String::new();
        }

        let norm_query = Self::normalized(query).to_lowercase();
        let tokens: Vec<String> = norm_query
            .split(|c: char| !Self::is_token_char(c))
            .filter(|s| s.chars().count() >= 2)
            .map(|s| s.to_string())
            .collect();

        let lines: Vec<&str> = trimmed
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if lines.is_empty() {
            return Self::collapsed(trimmed, max_chars);
        }

        if tokens.is_empty() {
            let joined = lines.iter().take(3).copied().collect::<Vec<_>>().join(" ");
            return Self::collapsed(&joined, max_chars);
        }

        let mut best_line = lines[0];
        let mut best_hits = 0usize;
        for &line in &lines {
            let lower = Self::normalized(line).to_lowercase();
            let hits = tokens.iter().filter(|t| lower.contains(t.as_str())).count();
            let line_len = line.chars().count();
            let best_len = best_line.chars().count();
            if hits > best_hits || (hits == best_hits && hits > 0 && best_len < 30 && line_len >= 30) {
                best_hits = hits;
                best_line = line;
            }
        }

        if best_hits == 0 {
            let joined = lines.iter().take(3).copied().collect::<Vec<_>>().join(" ");
            return Self::collapsed(&joined, max_chars);
        }

        let collapsed_line = best_line.split_whitespace().collect::<Vec<_>>().join(" ");
        let chars: Vec<char> = collapsed_line.chars().collect();
        if chars.len() <= max_chars {
            return collapsed_line;
        }

        let lower_line = Self::normalized(&collapsed_line).to_lowercase();
        for token in &tokens {
            if let Some(byte_pos) = lower_line.find(token.as_str()) {
                let char_pos = lower_line[..byte_pos].chars().count();
                let half = max_chars / 2;
                let start = char_pos.saturating_sub(half);
                let end = (start + max_chars).min(chars.len());
                let slice: String = chars[start..end].iter().collect();
                let prefix = if start > 0 { "…" } else { "" };
                let suffix = if end < chars.len() { "…" } else { "" };
                return format!("{}{}{}", prefix, slice, suffix);
            }
        }

        let head: String = chars.into_iter().take(max_chars).collect();
        format!("{}…", head)
    }

    pub fn collapsed(text: &str, max_chars: usize) -> String {
        let single = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let chars: Vec<char> = single.chars().collect();
        if chars.len() <= max_chars {
            single
        } else {
            let head: String = chars.into_iter().take(max_chars).collect();
            format!("{}…", head)
        }
    }

    /// Detect whether extracted PDF page text is empty or garbled and needs OCR/vision fallback.
    pub fn page_text_problem(page_text: &str) -> Option<PageTextProblem> {
        let mut total = 0usize;
        let mut non_ws = 0usize;
        let mut letters = 0usize;
        let mut digits = 0usize;
        let mut replacement = 0usize;
        let mut control = 0usize;

        for c in page_text.chars() {
            total += 1;
            let v = c as u32;
            if v == 0xFFFD || (0xE000..=0xF8FF).contains(&v) {
                replacement += 1;
            } else if c.is_whitespace() {
                continue;
            } else if v < 0x20 || (0x7F..=0x9F).contains(&v) {
                control += 1;
            } else {
                non_ws += 1;
                if c.is_alphabetic() {
                    letters += 1;
                } else if c.is_numeric() {
                    digits += 1;
                }
            }
        }

        if non_ws < 24 {
            return Some(PageTextProblem::Empty);
        }
        if replacement >= 2 && replacement * 100 >= non_ws {
            return Some(PageTextProblem::ReplacementChars);
        }
        if control * 50 > total {
            return Some(PageTextProblem::ControlChars);
        }
        let alnum = letters + digits;
        if alnum * 5 < non_ws * 2 {
            return Some(PageTextProblem::LowLetterDensity);
        }

        let words: Vec<&str> = page_text.split_whitespace().collect();
        if words.len() >= 12 {
            let single_char = words.iter().filter(|w| w.chars().count() == 1).count();
            if single_char * 5 > words.len() * 2 {
                return Some(PageTextProblem::FragmentedTokens);
            }
        }

        None
    }

    /// Compute lexical & shingle containment of `needle` inside `haystack` in `[0.0, 1.0]`.
    pub fn lexical_containment(needle: &str, haystack: &str) -> f64 {
        let norm_needle = Self::normalized(needle).to_lowercase();
        let norm_haystack = Self::normalized(haystack).to_lowercase();

        let clean_n = norm_needle.split_whitespace().collect::<Vec<_>>().join(" ");
        let clean_h = norm_haystack.split_whitespace().collect::<Vec<_>>().join(" ");

        if clean_n.is_empty() || clean_h.is_empty() {
            return 0.0;
        }
        if clean_h.contains(&clean_n) {
            return 1.0;
        }

        let n_tokens = Self::fts_tokens(&clean_n);
        let h_tokens = Self::fts_tokens(&clean_h);
        if n_tokens.is_empty() || h_tokens.is_empty() {
            return 0.0;
        }

        let h_set: HashSet<&str> = h_tokens.iter().map(|s| s.as_str()).collect();
        let unigram_hits = n_tokens
            .iter()
            .filter(|t| h_set.contains(t.as_str()))
            .count();
        let unigram_cov = (unigram_hits as f64) / (n_tokens.len() as f64);

        if n_tokens.len() < 3 || h_tokens.len() < 3 {
            return unigram_cov;
        }

        let mut h_trigrams = HashSet::new();
        for win in h_tokens.windows(3) {
            h_trigrams.insert((win[0].as_str(), win[1].as_str(), win[2].as_str()));
        }
        let n_trigram_total = n_tokens.len() - 2;
        let mut trigram_hits = 0usize;
        for win in n_tokens.windows(3) {
            if h_trigrams.contains(&(win[0].as_str(), win[1].as_str(), win[2].as_str())) {
                trigram_hits += 1;
            }
        }
        let trigram_cov = (trigram_hits as f64) / (n_trigram_total as f64);

        0.45 * unigram_cov + 0.55 * trigram_cov
    }

    /// Extract structural landmarks (Markdown headings, function/class/struct definitions) from source/text.
    pub fn extract_landmarks(text: &str, ext: &str) -> Vec<Landmark> {
        let mut landmarks = Vec::new();
        for (line_idx, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let line_no = line_idx + 1;
            if matches!(ext, "md" | "markdown" | "mdx" | "rst") {
                if line.starts_with('#') {
                    let heading = line.trim_start_matches('#').trim();
                    if !heading.is_empty() {
                        landmarks.push(Landmark {
                            label: format!("Line {}", line_no),
                            text: heading.to_string(),
                        });
                    }
                }
            } else {
                for prefix in &[
                    "pub fn ", "fn ", "async fn ", "pub async fn ", "pub struct ", "struct ",
                    "pub enum ", "enum ", "pub trait ", "trait ", "impl ", "def ", "async def ",
                    "class ", "function ", "export function ", "export class ", "interface ",
                    "type ", "func ",
                ] {
                    if line.starts_with(prefix) {
                        let sig = line.trim_end_matches('{').trim_end_matches(':').trim();
                        landmarks.push(Landmark {
                            label: format!("Line {}", line_no),
                            text: sig.to_string(),
                        });
                        break;
                    }
                }
            }
        }
        landmarks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_nfkc_and_arabic_and_bengali() {
        // Fullwidth ASCII -> standard ASCII
        assert_eq!(SearchText::normalized("Ｈｅｌｌｏ １２３"), "Hello 123");
        // Zero-width space -> space, soft-hyphen & ZWJ stripped
        assert_eq!(SearchText::normalized("foo\u{200B}bar\u{00AD}baz"), "foo barbaz");
        // Arabic harakat stripped & alef variants folded
        let arabic_with_diacritics = "أَلسَّلَامُ";
        let arabic_bare = "السلام";
        assert_eq!(
            SearchText::normalized(arabic_with_diacritics),
            SearchText::normalized(arabic_bare)
        );
        // Bengali vowel signs preserved in tokens
        let bengali = "বাংলা ভাষা অনুসন্ধান";
        let tokens = SearchText::fts_tokens(bengali);
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], "বাংলা");
    }

    #[test]
    fn test_page_text_problem_detection() {
        assert_eq!(
            SearchText::page_text_problem("too short"),
            Some(PageTextProblem::Empty)
        );
        assert_eq!(
            SearchText::page_text_problem(
                "This is a clean paragraph with plenty of readable English words and sentences on the page."
            ),
            None
        );
        assert_eq!(
            SearchText::page_text_problem("a b c d e f g h i j k l m n o p q r s t u v w x y z"),
            Some(PageTextProblem::FragmentedTokens)
        );
    }

    #[test]
    fn test_lexical_containment() {
        let doc = "Google DeepMind introduced EmbeddingGemma 2 for multimodal search across text, images, and audio.";
        let quote = "EmbeddingGemma 2 for multimodal search";
        assert_eq!(SearchText::lexical_containment(quote, doc), 1.0);
    }
}
