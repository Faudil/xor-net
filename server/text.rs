//! Text post-processing for streamed generation: incremental detokenization
//! and stop-string hold-back. Both are independent of the tokenizer type so
//! they can be tested with a mock decoder.

/// Incremental detokenizer using the prefix/read-offset scheme (as in HF
/// text-generation-inference): decode a small window of trailing tokens and
/// emit only the new suffix, and only once it no longer ends in U+FFFD (an
/// incomplete multi-byte UTF-8 sequence split across tokens).
pub struct IncrementalDetok {
    ids: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
}

impl IncrementalDetok {
    /// `context` are a few trailing prompt tokens. They are never emitted;
    /// they only give the decoder left context (so e.g. a SentencePiece
    /// leading-space strip doesn't eat the first generated space).
    pub fn new(context: &[u32]) -> Self {
        Self {
            ids: context.to_vec(),
            prefix_offset: 0,
            read_offset: context.len(),
        }
    }

    /// Push one generated token; returns the newly stable text (possibly
    /// empty).
    pub fn push<E>(
        &mut self,
        id: u32,
        decode: &impl Fn(&[u32]) -> Result<String, E>,
    ) -> Result<String, E> {
        self.ids.push(id);
        self.step(decode, false)
    }

    /// Flush whatever is left at the end of generation (even if it ends in
    /// U+FFFD).
    pub fn finish<E>(
        &mut self,
        decode: &impl Fn(&[u32]) -> Result<String, E>,
    ) -> Result<String, E> {
        self.step(decode, true)
    }

    fn step<E>(
        &mut self,
        decode: &impl Fn(&[u32]) -> Result<String, E>,
        force: bool,
    ) -> Result<String, E> {
        if self.read_offset >= self.ids.len() {
            return Ok(String::new());
        }
        let prefix = decode(&self.ids[self.prefix_offset..self.read_offset])?;
        let full = decode(&self.ids[self.prefix_offset..])?;
        if full.len() > prefix.len() && (force || !full.ends_with('\u{FFFD}')) {
            // `full` normally starts with `prefix`; if the decoder rewrote the
            // boundary, fall back to the common prefix so we never panic on a
            // non-char-boundary slice.
            let cut = if full.starts_with(&prefix) {
                prefix.len()
            } else {
                common_prefix_len(&prefix, &full)
            };
            let delta = full[cut..].to_string();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.ids.len();
            Ok(delta)
        } else {
            if force {
                self.prefix_offset = self.read_offset;
                self.read_offset = self.ids.len();
            }
            Ok(String::new())
        }
    }
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = 0;
    for ((i, ca), cb) in a.char_indices().zip(b.chars()) {
        if ca != cb {
            return i;
        }
        n = i + ca.len_utf8();
    }
    n
}

/// Outcome of feeding text to a [`StopBuffer`].
#[derive(Debug, PartialEq, Eq)]
pub enum StopOutcome {
    /// Text that is safe to emit now (may be empty).
    Emit(String),
    /// A stop string matched: emit the text before it, then stop.
    Stopped { emit: String, word: String },
}

/// Holds back any suffix that could be the start of a stop string, and cuts
/// the output at the first full match (the stop text itself is never
/// emitted).
pub struct StopBuffer {
    stops: Vec<String>,
    pending: String,
}

impl StopBuffer {
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            pending: String::new(),
        }
    }

    pub fn push(&mut self, text: &str) -> StopOutcome {
        self.pending.push_str(text);
        if self.stops.is_empty() {
            return StopOutcome::Emit(std::mem::take(&mut self.pending));
        }
        // Earliest match wins; on equal positions, the longest stop.
        let mut best: Option<(usize, &str)> = None;
        for s in &self.stops {
            if let Some(pos) = self.pending.find(s.as_str()) {
                let better = match best {
                    None => true,
                    Some((bp, bs)) => pos < bp || (pos == bp && s.len() > bs.len()),
                };
                if better {
                    best = Some((pos, s.as_str()));
                }
            }
        }
        if let Some((pos, word)) = best {
            let word = word.to_string();
            let emit = self.pending[..pos].to_string();
            self.pending.clear();
            return StopOutcome::Stopped { emit, word };
        }
        // Hold back the longest suffix that is a proper prefix of a stop.
        let mut hold = 0usize;
        for s in &self.stops {
            let max = s.len().saturating_sub(1).min(self.pending.len());
            for k in (hold + 1..=max).rev() {
                let start = self.pending.len() - k;
                if self.pending.is_char_boundary(start) && s.starts_with(&self.pending[start..]) {
                    hold = k;
                    break;
                }
            }
        }
        let cut = self.pending.len() - hold;
        let emit = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        StopOutcome::Emit(emit)
    }

    /// End of generation without a stop match: release held-back text.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock byte-level tokenizer: each id is one byte; decoding is lossy UTF-8
    /// (exactly how a byte-level BPE behaves on a split code point).
    fn decode(ids: &[u32]) -> Result<String, ()> {
        let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn ids(s: &str) -> Vec<u32> {
        s.bytes().map(u32::from).collect()
    }

    #[test]
    fn multibyte_is_never_split() {
        let text = "héllo 🦀 wörld €";
        let mut d = IncrementalDetok::new(&ids("ctx "));
        let mut out = String::new();
        for id in ids(text) {
            let delta = d.push(id, &decode).unwrap();
            assert!(!delta.contains('\u{FFFD}'), "emitted replacement char");
            out.push_str(&delta);
        }
        out.push_str(&d.finish(&decode).unwrap());
        assert_eq!(out, text);
    }

    #[test]
    fn per_token_decode_would_corrupt() {
        // Demonstrates the bug the incremental decoder fixes.
        let naive: String = ids("é").iter().map(|&i| decode(&[i]).unwrap()).collect();
        assert_ne!(naive, "é");
    }

    #[test]
    fn finish_flushes_incomplete_tail() {
        let mut d = IncrementalDetok::new(&[]);
        assert_eq!(d.push(b'a' as u32, &decode).unwrap(), "a");
        assert_eq!(d.push(0xE2, &decode).unwrap(), "");
        assert_eq!(d.finish(&decode).unwrap(), "\u{FFFD}");
        assert_eq!(d.finish(&decode).unwrap(), "");
    }

    fn run(stops: &[&str], chunks: &[&str]) -> (Vec<String>, Option<String>) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut b = StopBuffer::new(&stops);
        let mut out = Vec::new();
        for c in chunks {
            match b.push(c) {
                StopOutcome::Emit(e) => out.push(e),
                StopOutcome::Stopped { emit, word } => {
                    out.push(emit);
                    return (out, Some(word));
                }
            }
        }
        out.push(b.finish());
        (out, None)
    }

    #[test]
    fn stop_holdback_across_chunks() {
        let (out, w) = run(&["\nUser:"], &["Hello", "\n", "Us", "er:", " more"]);
        assert_eq!(w.as_deref(), Some("\nUser:"));
        assert_eq!(out.concat(), "Hello");
        // The "\n" and "Us" pieces were held back, not emitted early.
        assert_eq!(out, vec!["Hello", "", "", ""]);
    }

    #[test]
    fn stop_false_alarm_is_released() {
        let (out, w) = run(&["STOP"], &["ab", "ST", "OX", "yz"]);
        assert_eq!(w, None);
        assert_eq!(out.concat(), "abSTOXyz");
        assert_eq!(out[1], ""); // "ST" held back
        // Partial match at the very end is released by finish().
        let (out, w) = run(&["STOP"], &["abST"]);
        assert_eq!(w, None);
        assert_eq!(out, vec!["ab", "ST"]);
    }

    #[test]
    fn stop_earliest_match_and_multibyte() {
        let (out, w) = run(&["é!", "b"], &["aé", "!b"]);
        assert_eq!(w.as_deref(), Some("é!"));
        assert_eq!(out.concat(), "a");
        let (out, w) = run(&["xyz"], &["no stops here"]);
        assert_eq!((out.concat(), w), ("no stops here".to_string(), None));
        let (out, w) = run(&[], &["a", "b"]);
        assert_eq!((out.concat(), w), ("ab".to_string(), None));
    }
}
