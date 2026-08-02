//! GPU memory sensing behind a small trait, with per-platform backends. The rest
//! of the tool is platform-agnostic: it asks for `total_gb()` / `used_gb()` and
//! doesn't care how they're read.
//!
//! v1 backends: AMD `amdgpu` sysfs (unified VRAM + GTT) and NVIDIA via
//! `nvidia-smi`. When no backend is available, `measure` can't run — but `build`
//! still works from a supplied `--budget` and an existing measurement store, so
//! the pure half never needs a sensor.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};

/// Bytes per GiB (all footprints are GiB: bytes / 1073741824).
const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// A source of live GPU memory occupancy, summed across pools into one number.
pub trait GpuMemory {
    /// A human label for the detected device (shown by `setup`/`measure`).
    fn label(&self) -> String;
    /// Physical pool total, in GB.
    fn total_gb(&self) -> Result<f64>;
    /// Currently occupied, in GB.
    fn used_gb(&self) -> Result<f64>;
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

/// Auto-select a backend: AMD sysfs first, then NVIDIA. Errors if neither is
/// present (the caller should fall back to a configured/`--budget` value).
pub fn detect() -> Result<Box<dyn GpuMemory>> {
    if let Some(amd) = AmdSysfs::detect() {
        return Ok(Box::new(amd));
    }
    if let Some(nvidia) = NvidiaSmi::detect() {
        return Ok(Box::new(nvidia));
    }
    bail!(
        "no supported GPU memory sensor found (AMD amdgpu sysfs or NVIDIA nvidia-smi) — \
         pass --budget or set it in llama-matrix.toml to skip detection"
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
