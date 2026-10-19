use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const PARALLEL_THRESHOLD: usize = 32768;
pub const CHUNK_SIZE: usize = 4096;

/// Allocate a `Vec<T>` of length `size` whose elements are all `T::default()`.
///
/// Historically this returned *uninitialised* memory via `set_len`, which is
/// undefined behaviour as soon as an element is read (and `clippy::uninit_vec`
/// rejects it). It now zero/default-initialises. For the numeric types the
/// engine uses (`f32`, `i8`, `u8`, ...) `vec![0; n]` lowers to `calloc`, so a
/// large allocation is served by already-zeroed pages and costs no more than
/// the old version (the page faults happen on first touch either way).
pub fn uninit_vec<T: Copy + Default>(size: usize) -> Vec<T> {
    vec![T::default(); size]
}

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

pub fn get_num_threads() -> usize {
    let mut val = NUM_THREADS.load(Ordering::Relaxed);
    if val == 0 {
        val = rayon::current_num_threads().max(1);
        NUM_THREADS.store(val, Ordering::Relaxed);
    }
    val
}

/// Runtime CPU feature flags, detected once per process.
///
/// Every SIMD dispatcher in the crate goes through [`cpu_features`] so that
/// (a) the check that guards an `unsafe` `#[target_feature]` kernel is spelled
/// in one place and always covers *every* feature the kernel enables, and
/// (b) the per-call cost is a single relaxed load instead of several
/// `is_x86_feature_detected!` probes. On non-x86 targets all x86 flags are
/// `false`, so the portable scalar fallbacks are used.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub popcnt: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,
}

impl CpuFeatures {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::is_x86_feature_detected!("avx2"),
                fma: std::is_x86_feature_detected!("fma"),
                popcnt: std::is_x86_feature_detected!("popcnt"),
                avx512f: std::is_x86_feature_detected!("avx512f"),
                avx512bw: std::is_x86_feature_detected!("avx512bw"),
                avx512vnni: std::is_x86_feature_detected!("avx512vnni"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::default()
        }
    }

    /// `avx2` + `fma` (FMA3 is a separate CPUID bit from AVX2).
    #[inline(always)]
    pub fn avx2_fma(&self) -> bool {
        self.avx2 && self.fma
    }

    /// `avx512f` + `avx512bw` (byte/word ops such as `_mm512_shuffle_epi8`).
    #[inline(always)]
    pub fn avx512bw(&self) -> bool {
        self.avx512f && self.avx512bw
    }

    /// `avx512f` + `avx512bw` + `avx512vnni` (`vpdpbusd` on zmm registers).
    #[inline(always)]
    pub fn avx512_vnni(&self) -> bool {
        self.avx512f && self.avx512bw && self.avx512vnni
    }
}

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// The process-wide [`CpuFeatures`], detected on first use.
#[inline(always)]
pub fn cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(CpuFeatures::detect)
}
