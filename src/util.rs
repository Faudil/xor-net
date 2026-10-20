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

// ---------------------------------------------------------------------------
// Transparent huge pages for weight buffers
// ---------------------------------------------------------------------------

/// Size of an x86_64 / aarch64 (4 KiB granule) PMD huge page.
const HUGE_PAGE: usize = 2 << 20;

/// Whether `XORNET_HUGEPAGES=1` asks for weights in transparent huge pages:
/// `.xq1` weights are then copied at load into a THP-backed anonymous buffer
/// instead of being used from the file mapping (file-backed THP is not
/// available on most kernels / filesystems), and GGUF weights are repacked
/// into THP-backed buffers. Linux only; elsewhere it has no effect.
pub fn hugepages_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(target_os = "linux")
            && std::env::var("XORNET_HUGEPAGES").is_ok_and(|v| v.trim() == "1")
    })
}

/// A zero-initialised anonymous buffer whose start is 2 MiB aligned and that
/// is marked `MADV_HUGEPAGE` before first touch, so the kernel backs it with
/// transparent huge pages when `/sys/kernel/mm/transparent_hugepage/enabled`
/// is `always` or `madvise` (fewer TLB misses when streaming weights).
/// Elsewhere, or if the advice fails, it is ordinary anonymous memory.
pub struct HugeBuf {
    map: memmap2::MmapMut,
    off: usize,
    len: usize,
}

impl HugeBuf {
    pub fn new(len: usize) -> std::io::Result<Self> {
        // Over-allocate one huge page so the used range can start on a
        // huge-page boundary.
        let map = memmap2::MmapOptions::new()
            .len(
                len.checked_add(HUGE_PAGE)
                    .ok_or(std::io::ErrorKind::OutOfMemory)?,
            )
            .map_anon()?;
        let off = (map.as_ptr() as usize).next_multiple_of(HUGE_PAGE) - map.as_ptr() as usize;
        #[cfg(target_os = "linux")]
        {
            // Best effort: without THP support this is ordinary memory.
            let _ = map.advise_range(memmap2::Advice::HugePage, off, len);
        }
        Ok(Self { map, off, len })
    }

    /// A [`HugeBuf`] holding a copy of `src`.
    pub fn copy_of(src: &[u8]) -> std::io::Result<Self> {
        let mut b = Self::new(src.len())?;
        b.as_mut_slice().copy_from_slice(src);
        Ok(b)
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.map[self.off..self.off + self.len]
    }
}

impl AsRef<[u8]> for HugeBuf {
    fn as_ref(&self) -> &[u8] {
        &self.map[self.off..self.off + self.len]
    }
}

// ---------------------------------------------------------------------------
// Thread pinning
// ---------------------------------------------------------------------------

/// CPUs to pin `n` pool workers to, one per physical core first (the lowest
/// numbered SMT sibling of each core), then the remaining siblings; only CPUs
/// in the process affinity mask are used. `None` if the topology cannot be
/// read (non-Linux, no sysfs) or there are fewer allowed CPUs than `n`.
pub fn pin_plan(n: usize) -> Option<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        let allowed = affinity::allowed_cpus()?;
        let mut primaries = Vec::new();
        let mut secondaries = Vec::new();
        for &cpu in &allowed {
            let sib = std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ))
            .ok()?;
            let sibs = parse_cpu_list(sib.trim())?;
            // The first allowed sibling of a core stands for the core.
            let first = sibs.iter().copied().find(|c| allowed.contains(c))?;
            if first == cpu {
                primaries.push(cpu);
            } else {
                secondaries.push(cpu);
            }
        }
        primaries.extend(secondaries);
        (primaries.len() >= n).then(|| primaries[..n].to_vec())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = n;
        None
    }
}

/// Parse a sysfs CPU list such as `0,6` or `0-3,8-11`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut v = Vec::new();
    for part in s.split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => v.extend(a.trim().parse::<usize>().ok()?..=b.trim().parse().ok()?),
            None => v.push(part.trim().parse().ok()?),
        }
    }
    Some(v)
}

/// Pin the calling thread to `cpu`. Returns whether it succeeded (always
/// `false` off Linux).
pub fn pin_current_thread(cpu: usize) -> bool {
    #[cfg(target_os = "linux")]
    {
        affinity::pin(cpu)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu;
        false
    }
}

#[cfg(target_os = "linux")]
mod affinity {
    /// CPUs in the calling thread's affinity mask.
    pub fn allowed_cpus() -> Option<Vec<usize>> {
        // SAFETY: `cpu_set_t` is plain data; the kernel writes at most
        // `size_of::<cpu_set_t>()` bytes into it.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
                return None;
            }
            let max = 8 * std::mem::size_of::<libc::cpu_set_t>();
            Some((0..max).filter(|&c| libc::CPU_ISSET(c, &set)).collect())
        }
    }

    pub fn pin(cpu: usize) -> bool {
        if cpu >= 8 * std::mem::size_of::<libc::cpu_set_t>() {
            return false;
        }
        // SAFETY: as above; pid 0 is the calling thread.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
        }
    }
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
    /// AVX-VNNI: VEX-encoded `vpdpbusd` on ymm (Alder Lake+, Zen 5+).
    pub avxvnni: bool,
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
                avxvnni: std::is_x86_feature_detected!("avxvnni"),
            };
            detected.capped(std::env::var("XORNET_FORCE_ISA").ok().as_deref())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::default()
        }
    }

    /// Mask out features above the ISA named by `XORNET_FORCE_ISA`
    /// (`scalar`, `avx2`, `avxvnni`, `avx512bw`; anything else, or unset,
    /// keeps all). `avx2` is plain AVX2 (no AVX-VNNI); `avxvnni` is AVX2 plus
    /// AVX-VNNI without AVX-512.
    /// It can only *lower* the dispatch level, never claim a feature the CPU
    /// lacks, so it is safe to honour in release builds. CI uses it to run the
    /// whole test-suite through the AVX2 and scalar paths on AVX-512 runners.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    fn capped(mut self, isa: Option<&str>) -> Self {
        match isa.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("scalar") => self = Self::default(),
            Some(level @ ("avx2" | "avxvnni")) => {
                if level == "avx2" {
                    self.avxvnni = false;
                }
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

    /// [`Self::avx2_tier`] + AVX-VNNI (`vpdpbusd` on ymm registers).
    #[inline(always)]
    pub fn avx_vnni(&self) -> bool {
        self.avx2_tier() && self.avxvnni
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
    use super::{CpuFeatures, parse_cpu_list};

    #[test]
    fn cpu_list_parsing() {
        assert_eq!(parse_cpu_list("0,6"), Some(vec![0, 6]));
        assert_eq!(parse_cpu_list("0-3,8-9"), Some(vec![0, 1, 2, 3, 8, 9]));
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list("x"), None);
    }

    #[test]
    fn huge_buf_is_aligned_and_zeroed() {
        let mut b = super::HugeBuf::new(3 << 20).unwrap();
        assert_eq!(b.as_ref().as_ptr() as usize % (2 << 20), 0);
        assert!(b.as_ref().iter().all(|&x| x == 0));
        b.as_mut_slice()[12345] = 7;
        assert_eq!(b.as_ref()[12345], 7);
        let c = super::HugeBuf::copy_of(&[1, 2, 3]).unwrap();
        assert_eq!(c.as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn pin_plan_prefers_physical_cores() {
        if let Some(p) = super::pin_plan(1) {
            assert_eq!(p.len(), 1);
        }
    }

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
            avxvnni: true,
        };
        assert_eq!(all.capped(None), all);
        assert_eq!(all.capped(Some("bogus")), all);
        assert_eq!(all.capped(Some("scalar")), CpuFeatures::default());
        let avx2 = all.capped(Some(" AVX2 "));
        assert!(avx2.avx2_fma() && !avx2.avx512bw() && !avx2.avx512_vnni() && !avx2.avx_vnni());
        let vnni = all.capped(Some("avxvnni"));
        assert!(vnni.avx_vnni() && !vnni.avx512bw());
        let bw = all.capped(Some("avx512bw"));
        assert!(bw.avx512bw() && !bw.avx512_vnni());
        // Never raises: a CPU without AVX2 stays without it.
        assert_eq!(
            CpuFeatures::default().capped(Some("avx2")),
            CpuFeatures::default()
        );
    }
}
