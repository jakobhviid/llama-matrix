//! GPU memory sensing behind a small trait, with per-platform backends. The rest
//! of the tool is platform-agnostic: it asks for `total_gb()` / `used_gb()` and
//! doesn't care how they're read.
//!
//! Backends: AMD `amdgpu` sysfs (unified VRAM + GTT), NVIDIA via `nvidia-smi`, and
//! Apple Silicon (macOS) unified memory via `ioreg` + `sysctl`. When no backend is
//! available, `measure` can't run — but `build` still works from a supplied
//! `--budget` and an existing measurement store, so the pure half never needs a
//! sensor.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};

/// Bytes per GiB (all footprints are GiB: bytes / 1073741824).
pub const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// A source of live GPU memory occupancy, summed across pools into one number.
pub trait GpuMemory {
    /// A human label for the detected device (shown by `setup`/`measure`).
    fn label(&self) -> String;
    /// Physical pool total, in GB.
    fn total_gb(&self) -> Result<f64>;
    /// Currently occupied, in GB.
    fn used_gb(&self) -> Result<f64>;
    /// The `(vram, gtt)` breakdown of [`GpuMemory::used_gb`], for a device with
    /// distinct pools.
    ///
    /// `None` means "this backend cannot report a split", never "the split is
    /// zero": unified memory is one pool by construction, and a discrete NVIDIA
    /// card is all VRAM. `measure` records the split only when it is present, so a
    /// per-pool `0` in the store is always a real reading (see
    /// `cache::Measurement`). Defaults to `None` so a new backend has to opt in
    /// rather than silently report zeros.
    fn used_split_gb(&self) -> Option<(f64, f64)> {
        None
    }
}

/// AMD `amdgpu` sysfs backend. Unified-memory APUs expose a VRAM carve-out plus a
/// GPU-accessible system-RAM (GTT) pool; models spill VRAM→GTT, so occupancy is
/// their sum. Discrete AMD cards expose only VRAM (GTT reads as 0/absent).
pub struct AmdSysfs {
    device_dir: PathBuf,
}

impl AmdSysfs {
    /// Find the first DRM card that exposes amdgpu memory counters.
    pub fn detect() -> Option<Self> {
        for card_index in 0..16 {
            let device_dir = PathBuf::from(format!("/sys/class/drm/card{card_index}/device"));
            if device_dir.join("mem_info_vram_total").exists() {
                return Some(AmdSysfs { device_dir });
            }
        }
        None
    }

    /// Construct against an explicit device directory (used in tests).
    pub fn at(device_dir: impl Into<PathBuf>) -> Self {
        AmdSysfs {
            device_dir: device_dir.into(),
        }
    }

    fn read_bytes(&self, counter: &str) -> Result<u64> {
        let path = self.device_dir.join(counter);
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        text.trim()
            .parse::<u64>()
            .with_context(|| format!("parsing {}", path.display()))
    }

    fn read_bytes_or_zero(&self, counter: &str) -> u64 {
        self.read_bytes(counter).unwrap_or(0)
    }
}

impl GpuMemory for AmdSysfs {
    fn label(&self) -> String {
        format!("AMD amdgpu ({})", self.device_dir.display())
    }

    fn total_gb(&self) -> Result<f64> {
        let vram = self.read_bytes("mem_info_vram_total")?;
        let gtt = self.read_bytes_or_zero("mem_info_gtt_total");
        Ok((vram + gtt) as f64 / BYTES_PER_GIB)
    }

    fn used_gb(&self) -> Result<f64> {
        let vram = self.read_bytes("mem_info_vram_used")?;
        let gtt = self.read_bytes_or_zero("mem_info_gtt_used");
        Ok((vram + gtt) as f64 / BYTES_PER_GIB)
    }

    /// Both counters are already read for the sum, so the split costs nothing
    /// extra. A discrete card has no GTT pool, and reads a true 0.
    fn used_split_gb(&self) -> Option<(f64, f64)> {
        let vram = self.read_bytes("mem_info_vram_used").ok()?;
        let gtt = self.read_bytes_or_zero("mem_info_gtt_used");
        Some((vram as f64 / BYTES_PER_GIB, gtt as f64 / BYTES_PER_GIB))
    }
}

/// NVIDIA backend via `nvidia-smi`. Sums across all visible GPUs (v1 treats the
/// pool as one budget; per-device budgets are a multi-GPU roadmap item).
pub struct NvidiaSmi;

impl NvidiaSmi {
    pub fn detect() -> Option<Self> {
        let ran = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output();
        match ran {
            Ok(output) if output.status.success() => Some(NvidiaSmi),
            _ => None,
        }
    }

    fn query_sum_gb(field: &str) -> Result<f64> {
        let output = Command::new("nvidia-smi")
            .args([&format!("--query-gpu={field}"), "--format=csv,noheader,nounits"])
            .output()
            .context("running nvidia-smi")?;
        if !output.status.success() {
            bail!("nvidia-smi exited with an error");
        }
        let text = String::from_utf8_lossy(&output.stdout);
        // nvidia-smi reports MiB; sum every GPU line, convert to GiB.
        let mut total_mib = 0.0;
        for line in text.lines() {
            if let Ok(value) = line.trim().parse::<f64>() {
                total_mib += value;
            }
        }
        Ok(total_mib / 1024.0)
    }
}

impl GpuMemory for NvidiaSmi {
    fn label(&self) -> String {
        "NVIDIA (nvidia-smi)".to_string()
    }

    fn total_gb(&self) -> Result<f64> {
        Self::query_sum_gb("memory.total")
    }

    fn used_gb(&self) -> Result<f64> {
        Self::query_sum_gb("memory.used")
    }
}

/// Apple Silicon backend (macOS). CPU and GPU share one unified-memory pool, so a
/// model - llama.cpp Metal or MLX - allocates from the same pool the sensor reads.
/// `total` is the physical unified pool (`sysctl hw.memsize`); `used` is the GPU's
/// in-use system memory read from `ioreg` (the `IOAccelerator` "In use system
/// memory" counter), which needs no sudo. This is what lets `measure` size MLX and
/// Metal models on a Mac. The GPU shares the pool with the OS, so reserve headroom
/// with `budget`/`margin` rather than planning against the full total.
#[cfg(target_os = "macos")]
pub struct AppleSilicon {
    chip: String,
}

#[cfg(target_os = "macos")]
impl AppleSilicon {
    /// Present on every Apple Silicon Mac; the in-use read is the real capability
    /// probe, so detection succeeds only if that counter can be read.
    pub fn detect() -> Option<Self> {
        gpu_in_use_bytes().ok()?;
        let chip = sysctl_string("machdep.cpu.brand_string")
            .unwrap_or_else(|| "Apple Silicon".to_string());
        Some(AppleSilicon { chip })
    }
}

#[cfg(target_os = "macos")]
impl GpuMemory for AppleSilicon {
    fn label(&self) -> String {
        format!("{} (Metal unified memory)", self.chip)
    }

    fn total_gb(&self) -> Result<f64> {
        Ok(phys_mem_bytes()? as f64 / BYTES_PER_GIB)
    }

    fn used_gb(&self) -> Result<f64> {
        Ok(gpu_in_use_bytes()? as f64 / BYTES_PER_GIB)
    }
}

/// Extract the GPU's in-use unified memory (bytes) from `ioreg` IOAccelerator text.
/// Matches the exact `"In use system memory"=<n>` counter, never the sibling
/// `"In use system memory (driver)"` key (whose name has no `"=` after `memory`).
#[cfg(target_os = "macos")]
fn parse_in_use(ioreg_text: &str) -> Option<u64> {
    let needle = "\"In use system memory\"=";
    let start = ioreg_text.find(needle)? + needle.len();
    let digits: String = ioreg_text[start..]
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// The GPU's in-use unified memory in bytes, via `ioreg -r -d 1 -c IOAccelerator`.
#[cfg(target_os = "macos")]
fn gpu_in_use_bytes() -> Result<u64> {
    let output = Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
        .context("running ioreg")?;
    if !output.status.success() {
        bail!("ioreg exited with an error");
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_in_use(&text)
        .ok_or_else(|| anyhow::anyhow!("no 'In use system memory' counter in ioreg output"))
}

/// The physical unified-memory pool in bytes, via `sysctl -n hw.memsize`.
#[cfg(target_os = "macos")]
fn phys_mem_bytes() -> Result<u64> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .context("running sysctl hw.memsize")?;
    if !output.status.success() {
        bail!("sysctl hw.memsize failed");
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .context("parsing hw.memsize")
}

/// A trimmed `sysctl -n <name>` string, or None if it can't be read.
#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    let output = Command::new("sysctl").args(["-n", name]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!value.is_empty()).then_some(value)
}

/// Host RAM, which is a different resource from the GPU pool even on a box where
/// the two are carved out of the same chips.
///
/// `used` is `total - available`, not `total - free`: page cache the kernel can
/// reclaim under pressure is not memory anyone is holding, and counting it would
/// make every box look full. What is left is close to the sum of what processes
/// have actually dirtied, which is the quantity an OOM kill is decided on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostMemory {
    pub total_gb: f64,
    pub used_gb: f64,
}

/// Read host RAM, or `None` where there is no way to (an unsupported OS, an
/// unreadable `/proc`).
///
/// `None` is a first-class answer, not a failure: the host dimension is optional
/// throughout, and a box that cannot report it gets no host budgeting rather than a
/// guessed one (Principle 2).
pub fn host_memory() -> Option<HostMemory> {
    read_host_memory()
}

#[cfg(target_os = "linux")]
fn read_host_memory() -> Option<HostMemory> {
    parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// macOS has no `MemAvailable`. `hw.memsize` minus the pages `vm_stat` calls free,
/// inactive, speculative and purgeable is the same idea: everything the kernel could
/// hand out without evicting anyone's working set.
#[cfg(target_os = "macos")]
fn read_host_memory() -> Option<HostMemory> {
    let total = phys_mem_bytes().ok()? as f64 / BYTES_PER_GIB;
    let available = macos_available_bytes()? as f64 / BYTES_PER_GIB;
    Some(HostMemory { total_gb: total, used_gb: (total - available).max(0.0) })
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn read_host_memory() -> Option<HostMemory> {
    None
}

/// `MemTotal` and `MemAvailable` out of `/proc/meminfo` (both in kB) as GB.
///
/// `MemAvailable` is the kernel's own estimate of what can be handed out without
/// swapping, which is exactly the question being asked and is not reconstructible
/// from free + cached (some cache is not reclaimable). A kernel too old to publish
/// it reports nothing rather than a worse substitute.
#[cfg(target_os = "linux")]
fn parse_meminfo(meminfo: &str) -> Option<HostMemory> {
    let field = |name: &str| -> Option<f64> {
        let line = meminfo.lines().find(|line| line.starts_with(name))?;
        let kilobytes: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kilobytes * 1024.0 / BYTES_PER_GIB)
    };
    let total = field("MemTotal:")?;
    let available = field("MemAvailable:")?;
    Some(HostMemory { total_gb: total, used_gb: (total - available).max(0.0) })
}

/// Reclaimable bytes from `vm_stat`: free + inactive + speculative + purgeable.
#[cfg(target_os = "macos")]
fn macos_available_bytes() -> Option<u64> {
    let output = Command::new("vm_stat").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let page_size: u64 = text
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let pages = |name: &str| -> u64 {
        text.lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.rsplit(':').next())
            .map(|value| value.trim().trim_end_matches('.'))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    };
    Some(
        (pages("Pages free")
            + pages("Pages inactive")
            + pages("Pages speculative")
            + pages("Pages purgeable"))
            * page_size,
    )
}

/// Auto-select a backend: Apple Silicon (on macOS) first, then AMD sysfs, then
/// NVIDIA. Errors if none is present (the caller should fall back to a
/// configured/`--budget` value).
pub fn detect() -> Result<Box<dyn GpuMemory>> {
    #[cfg(target_os = "macos")]
    {
        if let Some(apple) = AppleSilicon::detect() {
            return Ok(Box::new(apple));
        }
    }
    if let Some(amd) = AmdSysfs::detect() {
        return Ok(Box::new(amd));
    }
    if let Some(nvidia) = NvidiaSmi::detect() {
        return Ok(Box::new(nvidia));
    }
    bail!(
        "no supported GPU memory sensor found (Apple Silicon, AMD amdgpu sysfs, or NVIDIA \
         nvidia-smi) — pass --budget or set it in llama-matrix.toml to skip detection"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amd_sysfs_sums_vram_and_gtt() {
        let dir = tempfile::tempdir().unwrap();
        let device = dir.path();
        std::fs::write(device.join("mem_info_vram_total"), (96u64 * 1_073_741_824).to_string()).unwrap();
        std::fs::write(device.join("mem_info_gtt_total"), (16u64 * 1_073_741_824).to_string()).unwrap();
        std::fs::write(device.join("mem_info_vram_used"), (10u64 * 1_073_741_824).to_string()).unwrap();
        std::fs::write(device.join("mem_info_gtt_used"), 1_073_741_824u64.to_string()).unwrap();

        let amd = AmdSysfs::at(device);
        assert!((amd.total_gb().unwrap() - 112.0).abs() < 1e-6);
        assert!((amd.used_gb().unwrap() - 11.0).abs() < 1e-6);

        // The per-pool split is reported too, and sums to the total occupancy.
        let (vram, gtt) = amd.used_split_gb().expect("amdgpu exposes both pools");
        assert!((vram - 10.0).abs() < 1e-6, "vram {vram}");
        assert!((gtt - 1.0).abs() < 1e-6, "gtt {gtt}");
        assert!((vram + gtt - amd.used_gb().unwrap()).abs() < 1e-6);
    }

    #[test]
    fn a_backend_without_pools_reports_no_split() {
        // The trait default: a backend that cannot separate pools returns None, so
        // `measure` omits the fields rather than writing zeros into them.
        struct Unified;
        impl GpuMemory for Unified {
            fn label(&self) -> String {
                "unified".into()
            }
            fn total_gb(&self) -> Result<f64> {
                Ok(48.0)
            }
            fn used_gb(&self) -> Result<f64> {
                Ok(12.0)
            }
        }
        assert_eq!(Unified.used_split_gb(), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn apple_silicon_parses_in_use_and_reads_the_pool() {
        // The parser must pick the exact counter, not the sibling "(driver)" key.
        let sample = r#""In use system memory (driver)"=0,"Alloc system memory"=10098065408,"In use system memory"=5449367552}"#;
        assert_eq!(parse_in_use(sample), Some(5_449_367_552));

        // A real read on this Mac: a positive pool, and occupancy within it.
        let apple = AppleSilicon::detect().expect("Apple Silicon backend detects on macOS");
        let total = apple.total_gb().unwrap();
        let used = apple.used_gb().unwrap();
        assert!(total > 1.0, "total {total} GB");
        assert!(used >= 0.0 && used <= total, "used {used} / total {total}");
    }

    #[test]
    fn detect_reads_the_local_gpu_when_present() {
        // On a box with amdgpu (or NVIDIA), detection must return a positive total.
        // A no-op on CI runners with no GPU.
        if std::path::Path::new("/sys/class/drm/card0/device/mem_info_vram_total").exists() {
            let gpu = detect().unwrap();
            assert!(gpu.total_gb().unwrap() > 1.0, "detected {}", gpu.label());
        }
    }
}
