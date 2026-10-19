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
    /// F16C (`vcvtph2ps` / `vcvtps2ph` on xmm/ymm).
    pub f16c: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512dq: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
}

impl CpuFeatures {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let detected = Self {
                avx2: std::is_x86_feature_detected!("avx2"),
                fma: std::is_x86_feature_detected!("fma"),
                popcnt: std::is_x86_feature_detected!("popcnt"),
                f16c: std::is_x86_feature_detected!("f16c"),
                avx512f: std::is_x86_feature_detected!("avx512f"),
                avx512bw: std::is_x86_feature_detected!("avx512bw"),
                avx512dq: std::is_x86_feature_detected!("avx512dq"),
                avx512vl: std::is_x86_feature_detected!("avx512vl"),
                avx512vnni: std::is_x86_feature_detected!("avx512vnni"),
            };
            detected.capped(std::env::var("XORNET_FORCE_ISA").ok().as_deref())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::default()
        }
    }

    /// Mask out features above the ISA named by `XORNET_FORCE_ISA`
    /// (`scalar`, `avx2`, `avx512bw`; anything else, or unset, keeps all).
    /// It can only *lower* the dispatch level, never claim a feature the CPU
    /// lacks, so it is safe to honour in release builds. CI uses it to run the
    /// whole test-suite through the AVX2 and scalar paths on AVX-512 runners.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    fn capped(mut self, isa: Option<&str>) -> Self {
        match isa.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("scalar") => self = Self::default(),
            Some("avx2") => {
                self.avx512f = false;
                self.avx512bw = false;
                self.avx512dq = false;
                self.avx512vl = false;
                self.avx512vnni = false;
            }
            Some("avx512bw") => self.avx512vnni = false,
            _ => {}
        }
        self
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

    /// AVX2 + FMA + F16C + POPCNT: the tier `with_best_simd!` compiles for
    /// as `avx2` (the x86-64-v3 vector features).
    #[inline(always)]
    pub fn avx2_tier(&self) -> bool {
        self.avx2 && self.fma && self.f16c && self.popcnt
    }

    /// AVX-512 F/BW/DQ/VL on top of [`Self::avx2_tier`] (x86-64-v4 level).
    #[inline(always)]
    pub fn avx512_tier(&self) -> bool {
        self.avx2_tier() && self.avx512f && self.avx512bw && self.avx512dq && self.avx512vl
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

/// Evaluate `$body` compiled for the best x86_64 SIMD tier this CPU supports
/// (AVX-512 F/BW/DQ/VL with or without VNNI, AVX2+FMA+F16C+POPCNT, or the
/// baseline), so plain Rust loops in hot non-kernel code are auto-vectorized
/// for the host without building the whole crate with `target-cpu=native`.
///
/// The body is expanded once per tier, each copy as a closure with a single
/// call site inside a `#[target_feature]` trampoline, so LLVM inlines it
/// there and code-generates it with those features (a closure shared by
/// several trampolines would stay out of line and get baseline codegen).
/// Callees of the body only benefit if they are inlined too, so mark small
/// helpers `#[inline(always)]`. Keep whole loops inside the body so the
/// dispatch is paid once per call.
///
/// Prefer a body that is a call to an `#[inline(always)]` function taking the
/// slices as arguments: values a closure captures by reference may alias as
/// far as LLVM knows (it then keeps scalar loads/stores and does not
/// vectorize), while `&mut` / `&` *parameters* are `noalias`.
macro_rules! with_best_simd {
    ($body:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            let c = $crate::util::cpu_features();
            // The closures are built outside the `unsafe` blocks so `$body`
            // gets no unsafe superpowers from the macro.
            if c.avx512_tier() && c.avx512vnni {
                let f = || $body;
                // SAFETY: every feature the trampoline enables was detected.
                unsafe { $crate::util::run_avx512vnni(f) }
            } else if c.avx512_tier() {
                let f = || $body;
                // SAFETY: as above.
                unsafe { $crate::util::run_avx512(f) }
            } else if c.avx2_tier() {
                let f = || $body;
                // SAFETY: as above.
                unsafe { $crate::util::run_avx2(f) }
            } else {
                $body
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $body
        }
    }};
}
pub(crate) use with_best_simd;

/// AVX-512 + VNNI trampoline for `with_best_simd!`.
///
/// # Safety
/// [`CpuFeatures::avx512_tier`] and `avx512vnni` must hold.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma,f16c,popcnt")]
pub(crate) unsafe fn run_avx512vnni<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// AVX-512 trampoline for `with_best_simd!`.
///
/// # Safety
/// [`CpuFeatures::avx512_tier`] must hold.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma,f16c,popcnt")]
pub(crate) unsafe fn run_avx512<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// AVX2 trampoline for `with_best_simd!`.
///
/// # Safety
/// [`CpuFeatures::avx2_tier`] must hold.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,popcnt")]
pub(crate) unsafe fn run_avx2<R>(f: impl FnOnce() -> R) -> R {
    f()
}

#[cfg(test)]
mod tests {
    use super::CpuFeatures;

    #[test]
    fn force_isa_only_lowers() {
        let all = CpuFeatures {
            avx2: true,
            fma: true,
            popcnt: true,
            f16c: true,
            avx512f: true,
            avx512bw: true,
            avx512dq: true,
            avx512vl: true,
            avx512vnni: true,
        };
        assert_eq!(all.capped(None), all);
        assert_eq!(all.capped(Some("bogus")), all);
        assert_eq!(all.capped(Some("scalar")), CpuFeatures::default());
        let avx2 = all.capped(Some(" AVX2 "));
        assert!(avx2.avx2_fma() && !avx2.avx512bw() && !avx2.avx512_vnni());
        let bw = all.capped(Some("avx512bw"));
        assert!(bw.avx512bw() && !bw.avx512_vnni());
        // Never raises: a CPU without AVX2 stays without it.
        assert_eq!(
            CpuFeatures::default().capped(Some("avx2")),
            CpuFeatures::default()
        );
    }
}
