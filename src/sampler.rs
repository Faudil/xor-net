use std::time::{SystemTime, UNIX_EPOCH};

pub struct Sampler {
    pub seed: u64,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    rng_state: u64,
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

        // Apply temperature
        let mut scaled_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l / temp))
            .collect();

        // Sort descending
        scaled_logits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax & Top-P
        let max_l = scaled_logits[0].1;
        let mut sum_exp = 0.0;
        for i in 0..scaled_logits.len() {
            sum_exp += (scaled_logits[i].1 - max_l).exp();
        }

        let mut probabilities = Vec::with_capacity(scaled_logits.len());
        let top_p = self.top_p.unwrap_or(1.0);
        let mut cumulative_prob = 0.0;
        
        for (i, l) in scaled_logits {
            let p = (l - max_l).exp() / sum_exp;
            probabilities.push((i, p));
            cumulative_prob += p;
            if cumulative_prob >= top_p {
                break;
            }
        }

        // Renormalize probabilities after top-p filtering
        let mut final_sum = 0.0;
        for &(_, p) in &probabilities {
            final_sum += p;
        }

        let r = self.random_f32() * final_sum;
        let mut acc = 0.0;
        for &(i, p) in &probabilities {
            acc += p;
            if r <= acc {
                return Ok(i as u32);
            }
        }

        Ok(probabilities.last().unwrap().0 as u32)
    }
}
