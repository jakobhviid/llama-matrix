//! Phase 1 — the solo-footprint sweep. Loads each model alone, reads real GPU
//! occupancy after allocation stabilizes, and records the delta over an empty
//! baseline into the per-model store, keyed by param-hash. GPU-touching, slow,
//! and lockfile-guarded. See ARCHITECTURE.md §2.1.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::cache::{BoxMeta, Measurement, ModelStore, Store};
use crate::model::{ModelRecord, ModelType};
use crate::param_hash::memory_cmd;
use crate::platform::{self, GpuMemory};
use crate::policy::Policy;

/// Knobs for a sweep.
pub struct MeasureOptions {
    pub endpoint: String,
    /// Re-measure even on a cache hit.
    pub force: bool,
    /// Restrict to these ids (None = the whole config worklist).
    pub only: Option<Vec<String>>,
    /// Per-model load timeout.
    pub load_timeout: Duration,
}

/// One model that could not be measured, and why (surfaced in the `--json` report
/// and the human failure list).
#[derive(Debug, Clone, Serialize)]
pub struct Failure {
    pub id: String,
    pub reason: String,
}

/// What a sweep did. This is the `measure --json` document (see the collect/render
/// split, ../../CLI-PATTERNS.md, DECISIONS.md D16): the CLI serializes it directly
/// and renders the human view from the same fields, so the two can't drift.
#[derive(Debug, Default, Serialize)]
pub struct MeasureSummary {
    pub measured: Vec<String>,
    pub cached: Vec<String>,
    pub failed: Vec<Failure>,
    pub skipped_missing: Vec<String>,
    pub baseline: f64,
    pub detected_total: f64,
}

/// A pid lockfile so two sweeps never share `/unload` and corrupt each other.
struct LockGuard {
    path: PathBuf,
}

impl LockGuard {
    fn acquire(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(".measure.lock");
        // A lock file records the owning pid. If that process is still alive,
        // another sweep is genuinely running - refuse. If it is gone, the lock is
        // stale (a crash or a Ctrl-C: Drop does not run on SIGINT), so reclaim it
        // rather than forcing the user to `rm` it by hand.
        if let Ok(contents) = std::fs::read_to_string(&path) {
            if let Ok(pid) = contents.trim().parse::<i32>() {
                if pid_is_alive(pid) {
                    bail!(
                        "a sweep is already running (pid {pid}); wait for it to finish, or \
                         remove {} if it is stale",
                        path.display()
                    );
                }
            }
            // an unreadable pid, or one that is gone, means a stale lock: reclaim.
        }
        std::fs::write(&path, std::process::id().to_string())?;
        Ok(LockGuard { path })
    }
}

/// Is a process with this pid still alive? `kill(pid, 0)` delivers no signal but
/// runs the kernel's existence + permission check: success or `EPERM` (alive but
/// not ours) means present; `ESRCH` means gone. Used to tell a running sweep from
/// a stale lock a crashed or interrupted one left behind.
fn pid_is_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    // SAFETY: `kill` with signal 0 only probes for the process; it delivers no
    // signal and mutates no state.
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn poll_agent() -> ureq::Agent {
    ureq::builder().timeout(Duration::from_secs(10)).build()
}

/// `{model: state}` from `GET /running` (empty on any error).
fn running(agent: &ureq::Agent, endpoint: &str) -> HashMap<String, String> {
    let url = format!("{endpoint}/running");
    let Ok(response) = agent.get(&url).call() else {
        return HashMap::new();
    };
    let Ok(text) = response.into_string() else {
        return HashMap::new();
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return HashMap::new();
    };
    let mut states = HashMap::new();
    if let Some(entries) = json.get("running").and_then(|value| value.as_array()) {
        for entry in entries {
            if let Some(model) = entry.get("model").and_then(|value| value.as_str()) {
                let state = entry.get("state").and_then(|value| value.as_str()).unwrap_or("");
                states.insert(model.to_string(), state.to_string());
            }
        }
    }
    states
}

/// Unload everything and wait until `/running` is empty, then settle.
fn unload_all(agent: &ureq::Agent, endpoint: &str, settle: Duration) {
    let posted = agent.post(&format!("{endpoint}/api/models/unload")).call();
    if posted.is_err() {
        let _ = agent.get(&format!("{endpoint}/unload")).call(); // legacy fallback
    }
    for _ in 0..20 {
        if running(agent, endpoint).is_empty() {
            break;
        }
        thread::sleep(Duration::from_secs(1));
    }
    thread::sleep(settle);
}

/// Fire the model's load-trigger on a detached thread (we poll `/running`
/// instead of awaiting — an image/chat call blocks well past the load).
fn trigger(model: &str, model_type: ModelType, endpoint: &str) -> thread::JoinHandle<()> {
    let model = model.to_string();
    let endpoint = endpoint.to_string();
    thread::spawn(move || {
        let agent = ureq::builder().timeout(Duration::from_secs(320)).build();
        let _ = match model_type {
            ModelType::Embed => agent
                .post(&format!("{endpoint}/v1/embeddings"))
                .send_json(serde_json::json!({"model": model, "input": "x"})),
            ModelType::Rerank => agent.post(&format!("{endpoint}/v1/rerank")).send_json(
                serde_json::json!({"model": model, "query": "x", "documents": ["a", "b"]}),
            ),
            ModelType::Image => agent.post(&format!("{endpoint}/v1/images/generations")).send_json(
                serde_json::json!({"model": model, "prompt": "a cat", "size": "256x256"}),
            ),
            ModelType::Stt => {
                let boundary = "----llamamatrixboundary";
                let body = multipart_wav(&model, boundary);
                agent
                    .post(&format!("{endpoint}/v1/audio/transcriptions"))
                    .set("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
                    .send_bytes(&body)
            }
            ModelType::Llm | ModelType::TtsProxy => agent
                .post(&format!("{endpoint}/v1/chat/completions"))
                .send_json(serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1
                })),
        };
    })
}

/// Poll until the model is `ready` (returns load seconds) or gives up / tears down.
fn wait_ready(
    agent: &ureq::Agent,
    endpoint: &str,
    model: &str,
    timeout: Duration,
) -> Option<f64> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        match running(agent, endpoint).get(model).map(String::as_str) {
            Some("ready") => return Some(start.elapsed().as_secs_f64()),
            Some("stopping") | Some("stopped") | Some("shutdown") => return None,
            _ => {}
        }
        thread::sleep(Duration::from_secs(2));
    }
    None
}

/// Sample occupancy until two consecutive readings are within `eps` GB (KV +
/// compute buffers finish allocating after `ready`), or `max_wait` elapses.
fn stabilize(gpu: &dyn GpuMemory, max_wait: Duration, eps: f64, hold: u32) -> f64 {
    let mut previous: Option<f64> = None;
    let mut stable = 0;
    let start = Instant::now();
    while start.elapsed() < max_wait {
        let current = gpu.used_gb().unwrap_or(0.0);
        if let Some(prev) = previous {
            if (current - prev).abs() < eps {
                stable += 1;
                if stable >= hold {
                    return current;
                }
            } else {
                stable = 0;
            }
        }
        previous = Some(current);
        thread::sleep(Duration::from_millis(1200));
    }
    gpu.used_gb().unwrap_or(0.0)
}

/// Run the sweep. Detects the GPU (errors if none — measure needs a sensor).
pub fn sweep(
    records: &[ModelRecord],
    store: &Store,
    policy: &Policy,
    options: &MeasureOptions,
) -> Result<MeasureSummary> {
    let gpu = platform::detect().context("measure needs a GPU sensor")?;
    let _lock = LockGuard::acquire(store.dir())?;
    let agent = poll_agent();
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();

    let mut summary = MeasureSummary {
        detected_total: gpu.total_gb()?,
        ..Default::default()
    };

    // empty baseline
    unload_all(&agent, &options.endpoint, Duration::from_secs(4));
    summary.baseline = gpu.used_gb()?;

    let only = options.only.as_ref();
    for record in records {
        if record.model_type == ModelType::TtsProxy {
            continue; // proxy entries allocate no GPU; footprint is hand-set
        }
        if let Some(only) = only {
            if !only.contains(&record.id) {
                continue;
            }
        }

        // pre-check the weight file exists on the host (skip a doomed load)
        if let Some(container_path) = &record.primary_file {
            let host_path = policy.to_host(container_path);
            if !Path::new(&host_path).exists() {
                summary.skipped_missing.push(record.id.clone());
                continue;
            }
        }

        // cache hit?
        if !options.force {
            if let Some(existing) = store.select(&record.id, &record.param_hash)? {
                if existing.is_ok() {
                    summary.cached.push(record.id.clone());
                    continue;
                }
            }
        }

        unload_all(&agent, &options.endpoint, Duration::from_secs(2));
        let handle = trigger(&record.id, record.model_type, &options.endpoint);
        let load_seconds = wait_ready(&agent, &options.endpoint, &record.id, options.load_timeout);

        let measurement = match load_seconds {
            None => {
                let failed = Measurement {
                    status: "FAILED".to_string(),
                    params: memory_cmd(&record.cmd),
                    measured_at: today.clone(),
                    ..Default::default()
                };
                summary.failed.push(Failure {
                    id: record.id.clone(),
                    reason: "load timed out or exited".to_string(),
                });
                failed
            }
            Some(load) => {
                let used = stabilize(gpu.as_ref(), Duration::from_secs(30), 0.03, 2);
                summary.measured.push(record.id.clone());
                Measurement {
                    status: "ok".to_string(),
                    d_total: round2(used - summary.baseline),
                    abs_total: round2(used),
                    load_s: round1(load),
                    params: memory_cmd(&record.cmd),
                    measured_at: today.clone(),
                    ..Default::default()
                }
            }
        };

        // upsert into the model's store file
        let mut model_store = store.read_model(&record.id)?.unwrap_or_else(|| ModelStore {
            model_type: record.model_type.as_str().to_string(),
            file: record.primary_file.clone(),
            measurements: Default::default(),
        });
        model_store.model_type = record.model_type.as_str().to_string();
        model_store.file = record.primary_file.clone();
        model_store.measurements.insert(record.param_hash.clone(), measurement);
        store.write_model(&record.id, &model_store)?;

        // best-effort: kill the (likely still-blocking) trigger call
        drop(handle);
        unload_all(&agent, &options.endpoint, Duration::from_secs(2));
    }

    store.write_box(&BoxMeta {
        baseline: round2(summary.baseline),
        detected_total: Some(round2(summary.detected_total)),
        date: Some(today),
        additivity_check: None,
        ..Default::default()
    })?;

    Ok(summary)
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}
fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

/// A `multipart/form-data` body: the `model` field + a tiny silent WAV file, for
/// the STT load-trigger.
fn multipart_wav(model: &str, boundary: &str) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    body.extend_from_slice(model.as_bytes());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body.extend_from_slice(&tiny_wav());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

/// 16 kHz mono 16-bit ~0.3 s of silence.
fn tiny_wav() -> Vec<u8> {
    let sample_rate: u32 = 16000;
    let samples: u32 = sample_rate * 3 / 10;
    let data_len: u32 = samples * 2;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.resize(wav.len() + data_len as usize, 0);
    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_lock_is_refused_and_a_stale_one_is_reclaimed() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join(".measure.lock");

        // Our own pid is alive, so an existing lock owned by it is refused.
        std::fs::write(&lock, std::process::id().to_string()).unwrap();
        assert!(LockGuard::acquire(dir.path()).is_err());

        // A pid that is not running leaves a stale lock, which is reclaimed.
        std::fs::write(&lock, "2147483646").unwrap();
        let guard = LockGuard::acquire(dir.path()).expect("a stale lock must be reclaimed");
        drop(guard); // Drop removes the lock on a clean exit
        assert!(!lock.exists(), "Drop should remove the lock file");
    }

    #[test]
    fn wav_has_a_valid_riff_header() {
        let wav = tiny_wav();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        // 16 kHz * 0.3 s * 2 bytes = 9600 data bytes + 44 header
        assert_eq!(wav.len(), 44 + 9600);
    }

    #[test]
    fn multipart_contains_model_and_file_parts() {
        let body = multipart_wav("whisper-1", "B");
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"model\""));
        assert!(text.contains("whisper-1"));
        assert!(text.contains("filename=\"a.wav\""));
        assert!(text.trim_end().ends_with("--B--"));
    }
}
