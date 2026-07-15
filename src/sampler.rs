use std::time::{SystemTime, UNIX_EPOCH};

pub struct Sampler {
    pub seed: u64,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    rng_state: u64,
    // Reused index scratch buffer so we don't reallocate ~1 MB every token.
    idx: Vec<usize>,
}

impl Sampler {
    pub fn new(seed: u64, temperature: Option<f32>, top_p: Option<f32>, repetition_penalty: f32) -> Self {
        let actual_seed = if seed == 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
        } else {
            seed
        };
        // Avoid 0 as xorshift state
        let rng_state = if actual_seed == 0 { 1 } else { actual_seed };
        Self {
            seed: actual_seed,
            temperature,
            top_p,
            repetition_penalty,
            rng_state,
            idx: Vec::new(),
        }
    }

    fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let val = self.rng_state as u32;
        (val as f32) / (u32::MAX as f32)
    }

    pub fn sample(&mut self, logits: &mut [f32], context_tokens: &[u32]) -> anyhow::Result<u32> {
        if self.repetition_penalty > 1.0 {
            let mut applied = std::collections::HashSet::new();
            for &tok in context_tokens {
                let tok = tok as usize;
                if tok < logits.len() && applied.insert(tok) {
                    let val = logits[tok];
                    if val > 0.0 {
                        logits[tok] = val / self.repetition_penalty;
                    } else {
                        logits[tok] = val * self.repetition_penalty;
                    }
                }
            }
        }
        
        let temp = self.temperature.unwrap_or(0.0);
        if temp == 0.0 {
            // Greedy argmax
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;
            for (i, &val) in logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            return Ok(max_idx as u32);
        }

        // Apply temperature in place.
        let inv_temp = 1.0 / temp;
        for v in logits.iter_mut() {
            *v *= inv_temp;
        }

        // Numerical-stability max (also the greedy candidate).
        let mut max_l = f32::NEG_INFINITY;
        let mut max_idx = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_l {
                max_l = v;
                max_idx = i;
            }
        }

        let top_p = self.top_p.unwrap_or(1.0);

        if top_p >= 1.0 {
            // Full softmax over the whole distribution.
            let mut sum_exp = 0.0;
            for &v in logits.iter() {
                sum_exp += (v - max_l).exp();
            }
            let r = self.random_f32() * sum_exp;
            let mut acc = 0.0;
            for (i, &v) in logits.iter().enumerate() {
                acc += (v - max_l).exp();
                if r <= acc {
                    return Ok(i as u32);
                }
            }
            return Ok(max_idx as u32);
        }

        // Top-p: keep only the smallest set of top tokens whose cumulative
        // probability reaches `top_p`. Use a partial selection (O(n)) instead
        // of sorting all `vocab` logits every token.
        let vocab = logits.len();
        let mut cand = 1024usize.min(vocab);
        loop {
            // Reused index buffer; `select_nth_unstable_by` partitions so the
            // largest `cand` indices land in the tail.
            self.idx.clear();
            for i in 0..vocab {
                self.idx.push(i);
            }
            let split = vocab - cand;
            self.idx.select_nth_unstable_by(split, |&a, &b| {
                logits[a]
                    .partial_cmp(&logits[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Sort the top-`cand` tail descending by logit value.
            let tail = &mut self.idx[split..];
            tail.sort_unstable_by(|&a, &b| {
                logits[b]
                    .partial_cmp(&logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut sum_exp = 0.0;
            for &i in tail.iter() {
                sum_exp += (logits[i] - max_l).exp();
            }
            let mut cumulative = 0.0;
            let mut covered = false;
            for &i in tail.iter() {
                cumulative += (logits[i] - max_l).exp() / sum_exp;
                if cumulative >= top_p {
                    covered = true;
                    break;
                }
            }

            if covered || cand == vocab {
                // Collect (index, prob) locally so we can drop the `tail`
                // borrow before calling `random_f32` (which also needs `&mut self`).
                let mut chosen: Vec<(usize, f32)> = Vec::with_capacity(tail.len());
                let mut final_sum = 0.0;
                for &i in tail.iter() {
                    let p = (logits[i] - max_l).exp();
                    final_sum += p;
                    chosen.push((i, p));
                }
                let r = self.random_f32() * final_sum;
                let mut acc = 0.0;
                for (i, p) in &chosen {
                    acc += p;
                    if r <= acc {
                        return Ok(*i as u32);
                    }
                }
                return Ok(chosen.last().unwrap().0 as u32);
            }
            // Distribution too flat for `cand` tokens; widen and retry.
            cand = (cand * 2).min(vocab);
        }
    }
}
