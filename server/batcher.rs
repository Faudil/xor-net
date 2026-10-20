//! Continuous-batching scheduler (`--scheduler batch`).
//!
//! One engine thread owns every active sequence (at most one per slot,
//! `--max-batch` slots). Each step it gathers the next token of every
//! decoding sequence plus prefill chunks of newly admitted ones (bounded by
//! `--batch-prefill` tokens while others decode), runs one batched forward
//! (`BatchedCausalLM::forward_batch`: each weight block is read once for
//! all sequences) on the global rayon pool, then lets each sequence sample
//! with its own sampler and streams its deltas.
//!
//! The slot workers of `serve` stay the request-facing side: a worker calls
//! [`BatchRunner::generate`], which submits the request to the engine thread
//! and relays its events until it finishes, so queueing, slot affinity,
//! cancellation and the HTTP layer are shared with the slot scheduler.
//! Per-request semantics (stop strings, EOS, `max_tokens`, reasoning split,
//! usage, prompt-cache restore / insertion, state reuse per slot) live in
//! [`xor_net::batch_session::DecodeSession`].

use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, PoisonError, TryLockError};
use std::thread;
use std::time::Duration;

use anyhow::anyhow;
use xor_net::batch_session::{DecodeSession, SessionOutcome, SessionShared, step};
use xor_net::generation::{GenerationParams, StreamEvent, common_prefix};
use xor_net::models::batched::{BatchItem, BatchedCausalLM, forward_batch_looping};
use xor_net::models::traits::{Logits, ModelState};

/// How often a waiting worker polls its request for cancellation.
const CANCEL_POLL: Duration = Duration::from_millis(50);

/// Scheduler knobs.
#[derive(Clone, Copy, Debug)]
pub struct BatchOptions {
    /// Concurrent sequences (= slots).
    pub max_batch: usize,
    /// Largest prefill chunk of one sequence per step.
    pub prefill_chunk: usize,
    /// Prefill tokens per step (all sequences) while others are decoding:
    /// bounds the extra latency a newly admitted prompt adds to their steps.
    pub prefill_budget: usize,
}

enum Event {
    Piece(bool, String),
    Done(Result<SessionOutcome, String>),
}

struct Submit {
    slot: usize,
    prompt: Vec<u32>,
    params: GenerationParams,
    boundaries: Vec<usize>,
    events: mpsc::Sender<Event>,
    cancel: Arc<AtomicBool>,
}

struct Active {
    slot: usize,
    session: DecodeSession,
    events: mpsc::Sender<Event>,
    cancel: Arc<AtomicBool>,
}

/// Handle of the engine thread; shuts it down when dropped.
pub struct BatchRunner {
    tx: Mutex<Option<mpsc::Sender<Submit>>>,
    /// Tokens held by each slot's idle state (for slot affinity).
    held: Arc<Vec<Mutex<Vec<u32>>>>,
    handle: Mutex<Option<thread::JoinHandle<()>>>,
    max_batch: usize,
}

type SlotState = Option<(Box<dyn ModelState>, Vec<u32>)>;

impl BatchRunner {
    /// Start the engine thread. `batched` is the model's batched forward;
    /// without one, each step forwards the sequences one by one (correct,
    /// but without shared weight reads).
    pub fn spawn(
        shared: SessionShared,
        batched: Option<Arc<dyn BatchedCausalLM>>,
        opts: BatchOptions,
    ) -> anyhow::Result<Self> {
        let max_batch = opts.max_batch.max(1);
        let held: Arc<Vec<Mutex<Vec<u32>>>> =
            Arc::new((0..max_batch).map(|_| Mutex::new(Vec::new())).collect());
        let (tx, rx) = mpsc::channel::<Submit>();
        let held2 = Arc::clone(&held);
        let handle = thread::Builder::new()
            .name("batch-engine".into())
            .spawn(move || engine_loop(&shared, batched.as_deref(), opts, &rx, &held2))?;
        Ok(Self {
            tx: Mutex::new(Some(tx)),
            held,
            handle: Mutex::new(Some(handle)),
            max_batch,
        })
    }

    /// Leading tokens of `ids` that slot `slot`'s idle state holds (0 while
    /// the slot is busy or unknown). Never blocks.
    pub fn affinity(&self, slot: usize, ids: &[u32]) -> usize {
        match self.held.get(slot).map(Mutex::try_lock) {
            Some(Ok(h)) => common_prefix(&h, ids),
            Some(Err(TryLockError::Poisoned(p))) => common_prefix(&p.into_inner(), ids),
            _ => 0,
        }
    }

    /// Run one request on slot `slot` (blocking): submits it to the engine
    /// thread and relays its text / reasoning pieces to `on_event` until it
    /// finishes. `on_event` returning `Break` cancels the request; it is
    /// also polled with an empty text event while no output arrives (so a
    /// caller's cancellation check runs during long prefills).
    pub fn generate(
        &self,
        slot: usize,
        prompt: &[u32],
        params: &GenerationParams,
        boundaries: &[usize],
        on_event: &mut dyn FnMut(StreamEvent<'_>) -> ControlFlow<()>,
    ) -> anyhow::Result<SessionOutcome> {
        anyhow::ensure!(slot < self.max_batch, "no batch slot {slot}");
        let (etx, erx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let submit = Submit {
            slot,
            prompt: prompt.to_vec(),
            params: params.clone(),
            boundaries: boundaries.to_vec(),
            events: etx,
            cancel: Arc::clone(&cancel),
        };
        {
            let tx = self.tx.lock().unwrap_or_else(PoisonError::into_inner);
            tx.as_ref()
                .ok_or_else(|| anyhow!("batch engine stopped"))?
                .send(submit)
                .map_err(|_| anyhow!("batch engine stopped"))?;
        }
        let mut cancelled = false;
        loop {
            let ev = match erx.recv_timeout(CANCEL_POLL) {
                Ok(ev) => ev,
                Err(RecvTimeoutError::Timeout) => {
                    if !cancelled && on_event(StreamEvent::Text("")).is_break() {
                        cancelled = true;
                        cancel.store(true, Ordering::Relaxed);
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(anyhow!("batch engine stopped"));
                }
            };
            match ev {
                Event::Piece(reasoning, text) => {
                    if cancelled {
                        continue;
                    }
                    let ev = if reasoning {
                        StreamEvent::Reasoning(&text)
                    } else {
                        StreamEvent::Text(&text)
                    };
                    if on_event(ev).is_break() {
                        cancelled = true;
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
                Event::Done(r) => return r.map_err(|e| anyhow!(e)),
            }
        }
    }
}

impl Drop for BatchRunner {
    fn drop(&mut self) {
        self.tx
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(h) = self
            .handle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            let _ = h.join();
        }
    }
}

fn admit(
    shared: &SessionShared,
    s: Submit,
    states: &mut [SlotState],
    held: &[Mutex<Vec<u32>>],
    active: &mut Vec<Active>,
) {
    let slot = s.slot;
    if s.slot >= states.len() || active.iter().any(|a| a.slot == slot) {
        let _ = s
            .events
            .send(Event::Done(Err(format!("batch slot {slot} unavailable"))));
        return;
    }
    held[slot]
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
    let state = states[slot].take();
    match DecodeSession::start(shared, &s.prompt, &s.params, &s.boundaries, state) {
        Ok(session) => active.push(Active {
            slot,
            session,
            events: s.events,
            cancel: s.cancel,
        }),
        Err(e) => {
            let _ = s.events.send(Event::Done(Err(format!("{e:#}"))));
        }
    }
}

fn engine_loop(
    shared: &SessionShared,
    batched: Option<&dyn BatchedCausalLM>,
    opts: BatchOptions,
    rx: &mpsc::Receiver<Submit>,
    held: &[Mutex<Vec<u32>>],
) {
    let model = Arc::clone(&shared.model);
    let forward = |items: &mut [BatchItem<'_>]| -> anyhow::Result<Vec<Logits>> {
        match batched {
            Some(b) => b.forward_batch(items),
            None => forward_batch_looping(&*model, items),
        }
    };
    let mut states: Vec<SlotState> = (0..held.len()).map(|_| None).collect();
    let mut active: Vec<Active> = Vec::new();
    loop {
        // Admit new requests (block while idle).
        if active.is_empty() {
            match rx.recv() {
                Ok(s) => admit(shared, s, &mut states, held, &mut active),
                Err(_) => return,
            }
        }
        while let Ok(s) = rx.try_recv() {
            admit(shared, s, &mut states, held, &mut active);
        }
        for a in active.iter_mut() {
            if a.cancel.load(Ordering::Relaxed) {
                a.session.cancel();
            }
        }
        let decoding = active
            .iter()
            .any(|a| !a.session.is_finished() && !a.session.is_prefilling());
        let budget = if decoding {
            opts.prefill_budget
        } else {
            opts.prefill_chunk.max(opts.prefill_budget)
        };
        {
            let mut refs: Vec<&mut DecodeSession> =
                active.iter_mut().map(|a| &mut a.session).collect();
            step(&forward, &mut refs, opts.prefill_chunk.max(1), budget);
        }
        // Stream deltas; retire finished sequences.
        let mut i = 0;
        while i < active.len() {
            let a = &mut active[i];
            for (r, text) in a.session.take_events() {
                if a.events.send(Event::Piece(r, text)).is_err() {
                    a.session.cancel();
                }
            }
            if !a.session.is_finished() {
                i += 1;
                continue;
            }
            let a = active.remove(i);
            let result = match (a.session.error(), a.session.outcome()) {
                (Some(e), _) => Err(e.to_owned()),
                (None, Some(o)) => Ok(o),
                (None, None) => Err("session ended without an outcome".to_owned()),
            };
            let state = a.session.into_state();
            if let Some((_, fed)) = &state {
                let mut h = held[a.slot].lock().unwrap_or_else(PoisonError::into_inner);
                h.clear();
                h.extend_from_slice(fed);
            }
            states[a.slot] = state;
            let _ = a.events.send(Event::Done(result));
        }
    }
}
