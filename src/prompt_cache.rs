//! Cross-request prompt (prefix) cache.
//!
//! A [`PromptCache`] stores [`StateSnapshot`]s of model states keyed by the
//! token prefix they encode, so a request whose prompt starts with a cached
//! prefix (a shared system prompt, an earlier turn of the same chat) restores
//! that snapshot and only prefills the rest. It is thread safe and meant to
//! be shared (`Arc`) by every [`Generator`](crate::generation::Generator) of
//! one model, e.g. all slots of a server; see
//! [`Generator::with_prompt_cache`](crate::generation::Generator::with_prompt_cache).
//!
//! Design:
//! - **Entries** are `(tokens, snapshot)` pairs. A snapshot covers exactly
//!   its tokens, so it is only usable by a prompt that starts with *all* of
//!   them (recurrent states cannot be cut shorter). Restoring is exact.
//! - **Lookup** returns the longest entry that is a prefix of the prompt.
//!   Every entry is indexed under the chained hash of each of its complete
//!   [`BLOCK`]-token blocks (like vLLM / llama.cpp block hashing), so a lookup
//!   hashes the prompt block by block until no entry shares that many blocks,
//!   then verifies the (few) candidates of the deepest matching block token by
//!   token. No linear scan over entries; hash collisions are harmless.
//! - **Budget**: entries are charged their snapshot bytes plus their tokens;
//!   inserting evicts least-recently-used entries until the total fits.
//!   An entry larger than the whole budget is rejected.
//! - Entries are keyed by model as well (the `Arc` pointer of the model), so
//!   a cache accidentally shared by two models never mixes their states.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::Serialize;

use crate::models::traits::{CausalLM, StateSnapshot};

/// Tokens per hashed block of the lookup index.
pub const BLOCK: usize = 64;

/// Default minimum entry length (tokens): shorter prefixes are cheap to
/// recompute and not worth a snapshot.
pub const DEFAULT_MIN_TOKENS: usize = 64;

/// Upper bound of [`PromptCache::default_budget`].
pub const DEFAULT_MAX_BUDGET_BYTES: usize = 1 << 30;

/// Bookkeeping charged per entry on top of its snapshot and tokens.
const ENTRY_OVERHEAD: usize = 256;

/// One cached prefix: its tokens and the model state after them.
#[derive(Debug)]
pub struct CacheEntry {
    model: usize,
    tokens: Vec<u32>,
    snapshot: StateSnapshot,
}

impl CacheEntry {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    pub fn snapshot(&self) -> &StateSnapshot {
        &self.snapshot
    }
    fn bytes(&self) -> usize {
        self.snapshot.bytes() + self.tokens.len() * 4 + ENTRY_OVERHEAD
    }
}

/// Counters since the cache was created (see [`PromptCache::stats`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PromptCacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub budget_bytes: usize,
    pub min_tokens: usize,
    /// Lookups for prompts long enough to possibly hit.
    pub lookups: u64,
    /// Entries actually restored (see [`PromptCache::record_hit`]).
    pub hits: u64,
    /// Prompt tokens not recomputed thanks to hits.
    pub hit_tokens: u64,
    pub inserts: u64,
    pub evictions: u64,
    /// Inserts refused because the entry alone exceeds the budget.
    pub rejected: u64,
}

struct Slot {
    entry: Arc<CacheEntry>,
    last_used: u64,
    /// Index keys this entry is listed under (one per complete block).
    keys: Vec<u64>,
}

#[derive(Default)]
struct Inner {
    slots: HashMap<u64, Slot>,
    /// Chained block hash -> ids of the entries sharing that many blocks.
    index: HashMap<u64, Vec<u64>>,
    /// Entries shorter than one block (only with `min_tokens < BLOCK`).
    short: Vec<u64>,
    bytes: usize,
    next_id: u64,
    tick: u64,
    stats: PromptCacheStats,
}

/// Thread-safe LRU cache of model-state snapshots keyed by token prefix.
/// See the [module docs](self).
pub struct PromptCache {
    budget: usize,
    min_tokens: usize,
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for PromptCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptCache")
            .field("stats", &self.stats())
            .finish()
    }
}

fn seed(model: usize) -> u64 {
    let mut h = DefaultHasher::new();
    model.hash(&mut h);
    h.finish()
}

fn chain(prev: u64, block: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    prev.hash(&mut h);
    block.hash(&mut h);
    h.finish()
}

/// Chained hashes of every complete block of `tokens`.
fn block_keys(model: usize, tokens: &[u32]) -> impl Iterator<Item = u64> + '_ {
    tokens
        .as_chunks::<BLOCK>()
        .0
        .iter()
        .scan(seed(model), |h, b| {
            *h = chain(*h, b);
            Some(*h)
        })
}

/// The key a [`PromptCache`] uses for `model`'s entries.
pub fn model_key(model: &Arc<dyn CausalLM>) -> usize {
    Arc::as_ptr(model) as *const () as usize
}

impl PromptCache {
    /// A cache holding at most `budget_bytes` of entries, each at least
    /// `min_tokens` long (clamped to `>= 1`).
    pub fn new(budget_bytes: usize, min_tokens: usize) -> Self {
        let min_tokens = min_tokens.max(1);
        Self {
            budget: budget_bytes,
            min_tokens,
            inner: Mutex::new(Inner {
                stats: PromptCacheStats {
                    budget_bytes,
                    min_tokens,
                    ..Default::default()
                },
                ..Default::default()
            }),
        }
    }

    /// The default byte budget: a quarter of the currently available memory
    /// (`MemAvailable` on Linux), capped at [`DEFAULT_MAX_BUDGET_BYTES`]
    /// (1 GiB); 1 GiB where available memory cannot be read.
    pub fn default_budget() -> usize {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
                let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
                Some(kb * 1024 / 4)
            })
            .unwrap_or(DEFAULT_MAX_BUDGET_BYTES)
            .min(DEFAULT_MAX_BUDGET_BYTES)
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget
    }

    pub fn min_tokens(&self) -> usize {
        self.min_tokens
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The longest entry of `model` whose tokens are a prefix of `prompt`
    /// and at most `max_len` long (callers pass `prompt.len() - 1` so at
    /// least one token is left to prefill). Marks it recently used.
    pub fn lookup(&self, model: usize, prompt: &[u32], max_len: usize) -> Option<Arc<CacheEntry>> {
        let max_len = max_len.min(prompt.len());
        if max_len < self.min_tokens {
            return None;
        }
        let mut g = self.lock();
        g.stats.lookups += 1;
        if g.slots.is_empty() {
            return None;
        }
        let mut keys = Vec::new();
        for key in block_keys(model, &prompt[..max_len]) {
            if !g.index.contains_key(&key) {
                break;
            }
            keys.push(key);
        }
        let pick = |g: &Inner, ids: &[u64]| -> Option<u64> {
            ids.iter()
                .filter_map(|id| Some((*id, &g.slots.get(id)?.entry)))
                .filter(|(_, e)| e.model == model && e.len() <= max_len)
                .filter(|(_, e)| prompt.starts_with(&e.tokens))
                .max_by_key(|(_, e)| e.len())
                .map(|(id, _)| id)
        };
        // An entry verified under the deepest shared block is longer than
        // any prefix entry listed only under shallower blocks.
        let found = keys
            .iter()
            .rev()
            .find_map(|k| pick(&g, &g.index[k]))
            .or_else(|| pick(&g, &g.short));
        let id = found?;
        g.tick += 1;
        let tick = g.tick;
        let slot = g.slots.get_mut(&id)?;
        slot.last_used = tick;
        Some(Arc::clone(&slot.entry))
    }

    /// Count a hit: a looked-up entry of `tokens` tokens was restored (the
    /// [`Generator`](crate::generation::Generator) calls this; a lookup alone
    /// is not a hit, since the caller's own state may cover more).
    pub fn record_hit(&self, tokens: usize) {
        let mut g = self.lock();
        g.stats.hits += 1;
        g.stats.hit_tokens += tokens as u64;
    }

    fn find_exact(g: &Inner, model: usize, tokens: &[u32]) -> Option<u64> {
        let ids = match block_keys(model, tokens).last() {
            Some(k) => g.index.get(&k)?,
            None => &g.short,
        };
        ids.iter().copied().find(|id| {
            g.slots
                .get(id)
                .is_some_and(|s| s.entry.model == model && s.entry.tokens == tokens)
        })
    }

    /// Whether an entry for exactly `tokens` exists (refreshing it if so).
    /// Cheap: check this before taking a snapshot to insert.
    pub fn contains(&self, model: usize, tokens: &[u32]) -> bool {
        let mut g = self.lock();
        let Some(id) = Self::find_exact(&g, model, tokens) else {
            return false;
        };
        g.tick += 1;
        let tick = g.tick;
        if let Some(s) = g.slots.get_mut(&id) {
            s.last_used = tick;
        }
        true
    }

    /// Whether an entry for `tokens` would be accepted (long enough).
    pub fn wants(&self, tokens: usize) -> bool {
        tokens >= self.min_tokens
    }

    /// Insert `snapshot` (which must cover exactly `tokens`), evicting
    /// least-recently-used entries to stay within the budget. Returns
    /// `false` if it was not inserted: too short, already cached, length
    /// mismatch, or larger than the whole budget.
    pub fn insert(&self, model: usize, tokens: &[u32], snapshot: StateSnapshot) -> bool {
        if tokens.len() < self.min_tokens || snapshot.len() != tokens.len() {
            return false;
        }
        let entry = CacheEntry {
            model,
            tokens: tokens.to_vec(),
            snapshot,
        };
        let bytes = entry.bytes();
        let mut g = self.lock();
        if bytes > self.budget {
            g.stats.rejected += 1;
            return false;
        }
        if Self::find_exact(&g, model, tokens).is_some() {
            return false;
        }
        while g.bytes + bytes > self.budget {
            let Some(victim) = g
                .slots
                .iter()
                .min_by_key(|(_, s)| s.last_used)
                .map(|(id, _)| *id)
            else {
                break;
            };
            Self::remove(&mut g, victim);
            g.stats.evictions += 1;
        }
        let id = g.next_id;
        g.next_id += 1;
        g.tick += 1;
        let keys: Vec<u64> = block_keys(model, tokens).collect();
        if keys.is_empty() {
            g.short.push(id);
        }
        for k in &keys {
            g.index.entry(*k).or_default().push(id);
        }
        let last_used = g.tick;
        g.slots.insert(
            id,
            Slot {
                entry: Arc::new(entry),
                last_used,
                keys,
            },
        );
        g.bytes += bytes;
        g.stats.inserts += 1;
        true
    }

    fn remove(g: &mut Inner, id: u64) {
        let Some(slot) = g.slots.remove(&id) else {
            return;
        };
        g.bytes -= slot.entry.bytes();
        if slot.keys.is_empty() {
            g.short.retain(|&x| x != id);
        }
        for k in &slot.keys {
            if let Some(ids) = g.index.get_mut(k) {
                ids.retain(|&x| x != id);
                if ids.is_empty() {
                    g.index.remove(k);
                }
            }
        }
    }

    /// Drop every entry (counters are kept).
    pub fn clear(&self) {
        let mut g = self.lock();
        let ids: Vec<u64> = g.slots.keys().copied().collect();
        for id in ids {
            Self::remove(&mut g, id);
        }
    }

    pub fn stats(&self) -> PromptCacheStats {
        let g = self.lock();
        PromptCacheStats {
            entries: g.slots.len(),
            bytes: g.bytes,
            ..g.stats
        }
    }

    /// Lengths of the cached entries, longest first (diagnostics).
    pub fn entry_lens(&self) -> Vec<usize> {
        let g = self.lock();
        let mut v: Vec<usize> = g.slots.values().map(|s| s.entry.len()).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(len: usize, bytes: usize) -> StateSnapshot {
        StateSnapshot::new(len, bytes, ())
    }

    fn toks(n: usize, salt: u32) -> Vec<u32> {
        (0..n as u32).map(|i| i * 7 + salt).collect()
    }

    #[test]
    fn longest_prefix_wins() {
        let c = PromptCache::new(1 << 20, 8);
        let base = toks(300, 0);
        for &n in &[10, 64, 100, 128, 200, 250] {
            assert!(c.insert(1, &base[..n], snap(n, 100)));
        }
        assert!(!c.insert(1, &base[..100], snap(100, 100)), "duplicate");
        for (plen, want) in [
            (300, Some(250)),
            (250, Some(200)),
            (201, Some(200)),
            (129, Some(128)),
            (101, Some(100)),
            (64, Some(10)),
            (11, Some(10)),
            (10, None),
        ] {
            let got = c.lookup(1, &base[..plen], plen - 1).map(|e| e.len());
            assert_eq!(got, want, "prompt {plen}");
        }
        // Diverging after 150 tokens: the 128 entry is the longest prefix.
        let mut p = base.clone();
        p[150] = 9999;
        assert_eq!(c.lookup(1, &p, p.len() - 1).unwrap().len(), 128);
        // Diverging inside the first block.
        p[3] = 9999;
        assert!(c.lookup(1, &p, p.len() - 1).is_none());
        // Other model.
        assert!(c.lookup(2, &base, 299).is_none());
        assert!(c.contains(1, &base[..64]) && !c.contains(1, &base[..65]));
    }

    #[test]
    fn lru_eviction_respects_budget() {
        let per = 1000 + 64 * 4 + ENTRY_OVERHEAD;
        let c = PromptCache::new(3 * per, 1);
        let seqs: Vec<Vec<u32>> = (0..5).map(|i| toks(64, 1000 * i)).collect();
        for s in &seqs[..3] {
            assert!(c.insert(1, s, snap(64, 1000)));
        }
        assert_eq!(c.stats().entries, 3);
        // Touch seq 0 so seq 1 is the LRU.
        assert!(
            c.lookup(1, &[seqs[0].clone(), vec![5]].concat(), 64)
                .is_some()
        );
        assert!(c.insert(1, &seqs[3], snap(64, 1000)));
        let st = c.stats();
        assert_eq!((st.entries, st.evictions), (3, 1));
        assert!(st.bytes <= st.budget_bytes);
        assert!(c.contains(1, &seqs[0]) && !c.contains(1, &seqs[1]));
        // Too large for the whole budget.
        assert!(!c.insert(1, &seqs[4], snap(64, 10 * per)));
        assert_eq!(c.stats().rejected, 1);
        // Shorter than min_tokens / length mismatch.
        let c2 = PromptCache::new(1 << 20, 64);
        assert!(!c2.insert(1, &seqs[0][..63], snap(63, 1)));
        assert!(!c2.insert(1, &seqs[0], snap(10, 1)));
        c.clear();
        assert_eq!((c.stats().entries, c.stats().bytes), (0, 0));
    }

    #[test]
    fn concurrent_access() {
        let c = Arc::new(PromptCache::new(40 * (500 * 4 + 4096), 16));
        let shared = toks(256, 3);
        let handles: Vec<_> = (0..8u32)
            .map(|t| {
                let c = Arc::clone(&c);
                let shared = shared.clone();
                std::thread::spawn(move || {
                    for i in 0..200u32 {
                        let n = 16 + ((i * 37 + t * 11) % 240) as usize;
                        let mut p = shared[..n].to_vec();
                        if i % 3 == 0 {
                            p.push(10_000 + t * 1000 + i);
                        }
                        let _ = c.insert(1, &p, snap(p.len(), 4096));
                        let q = [&shared[..], &[1u32, 2, 3][..]].concat();
                        if let Some(e) = c.lookup(1, &q, q.len() - 1) {
                            assert!(q.starts_with(e.tokens()));
                            c.record_hit(e.len());
                        }
                        let st = c.stats();
                        assert!(st.bytes <= st.budget_bytes);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let st = c.stats();
        assert!(st.entries > 0 && st.bytes <= st.budget_bytes && st.hits > 0);
    }
}
