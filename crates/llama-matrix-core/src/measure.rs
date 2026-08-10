//! Phase 1 — the solo-footprint sweep. Loads each model alone, reads real GPU
//! occupancy after allocation stabilizes, and records the delta over an empty
//! baseline into the per-model store, keyed by param-hash. GPU-touching, slow,
//! and lockfile-guarded. See ARCHITECTURE.md §2.1.
//!
//! **`ready` is not `allocated`.** llama-swap reports a model `ready` when its
//! upstream answers HTTP, which for a lazily-allocating backend (sd-server: the
//! generation *is* the allocation) happens long before the weights are resident.
//! Sampling at that point captures a mid-load plateau, which can be under half the
//! real footprint and is indistinguishable by inspection from a settled reading, so it
//! would reach `build` as the over-declaration Principle 1 forbids. The trigger request
//! is therefore awaited: its completion is the strongest evidence any backend gives
//! that it finished allocating, and whether that evidence was obtained is recorded with
//! the measurement ([`crate::cache::Measurement::allocation_confirmed`]) rather than
//! assumed.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::cache::{BoxMeta, Measurement, ModelStore, Store};
use crate::model::{weight_files, ModelRecord, ModelType};
use crate::param_hash::memory_cmd;
use crate::platform::{self, GpuMemory, BYTES_PER_GIB};
use crate::policy::Policy;

/// How long to wait for `/running` to report a model `ready`.
pub const DEFAULT_LOAD_TIMEOUT: Duration = Duration::from_secs(300);

/// How long to wait for the load-trigger request itself to finish. Generous on
/// purpose: for an image backend this covers a full generation at the probe
/// resolution, and overrunning it does not produce a wrong number, only an
/// *unconfirmed* one that says so.
pub const DEFAULT_TRIGGER_TIMEOUT: Duration = Duration::from_secs(900);

/// How often occupancy is sampled while waiting for the trigger to finish (only to
/// track the allocation peak; the footprint itself comes from `stabilize`).
const TRIGGER_SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

/// Consecutive quiet samples required before occupancy counts as settled. Three
/// (~3.6 s) rather than two: a staged loader pauses between components, and with the
/// trigger already awaited there is nothing left to allocate, so the extra sample
/// costs a second and removes a whole class of false settle.
const STABILIZE_HOLD: u32 = 3;

/// How long to keep sampling for a settled reading before giving up (and saying so).
const STABILIZE_MAX_WAIT: Duration = Duration::from_secs(30);

/// Two readings within this many GB of each other count as quiet.
const STABILIZE_EPS: f64 = 0.03;

/// Delay between occupancy samples while looking for a settled reading.
const STABILIZE_INTERVAL: Duration = Duration::from_millis(1200);

/// Knobs for a sweep.
pub struct MeasureOptions {
    pub endpoint: String,
    /// Re-measure even on a cache hit.
    pub force: bool,
    /// Restrict to these ids (None = the whole config worklist).
    pub only: Option<Vec<String>>,
    /// Per-model load timeout (how long `/running` may take to say `ready`).
    pub load_timeout: Duration,
    /// How long the load-trigger itself may take. Completing it is what proves the
    /// allocation finished, so this is a correctness budget, not a convenience one.
    pub trigger_timeout: Duration,
    /// `WxH` for the image load-trigger: a diffusion model's allocation scales with
    /// the resolution it generates at, so this decides what the footprint means.
    pub probe_image_size: String,
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
    /// Models whose footprint was recorded without confirming that llama-swap
    /// actually loaded the command we hashed (the backend exposes no `/props`, or
    /// the endpoint predates it). Reported rather than silently implied to be
    /// verified: Principle 7.
    ///
    /// Informational and **permanent** for some backends (an image or STT server has
    /// no `/props` to ask), which is exactly why it does not escalate the headline:
    /// a warning that can never be cleared trains an operator to ignore it. The
    /// actionable sibling is `unconfirmed_allocation`.
    pub unverified_serving: Vec<String>,
    /// Models whose footprint was recorded **without confirming the allocation
    /// finished** (the load-trigger never came back, or occupancy was still moving
    /// when sampling stopped). Unlike `unverified_serving` this is clearable, and
    /// `build` treats it as policy (`on_unconfirmed`), so it does escalate.
    pub unconfirmed_allocation: Vec<Failure>,
    /// Models whose footprint came out implausibly small for the weight files their
    /// command names (below `cache::WEIGHT_FLOOR_RATIO` of the total on disk). A
    /// signal, not a verdict: partial offload legitimately sits lower.
    pub below_weight_floor: Vec<Failure>,
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

/// The model ids llama-swap currently advertises, from `GET /v1/models`.
///
/// Only ever used to *explain* a failed load, never to skip one: an `unlisted` model
/// is hidden from this roster and still perfectly loadable (SPEC §8), so absence is
/// a hint, not a verdict. `None` when the roster can't be read at all.
fn served_ids(agent: &ureq::Agent, endpoint: &str) -> Option<HashSet<String>> {
    let text = agent.get(&format!("{endpoint}/v1/models")).call().ok()?.into_string().ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let entries = json.get("data")?.as_array()?;
    Some(
        entries
            .iter()
            .filter_map(|entry| entry.get("id")?.as_str().map(str::to_string))
            .collect(),
    )
}

/// The `(context_per_slot, slots)` the **running** server actually allocated, read
/// from the model's own `/props` through llama-swap's `/upstream/<id>/` route.
///
/// This is the only way to see the live launch flags: no llama-swap endpoint (as of
/// v247) reports a model's `cmd`, so the served command has to be inferred from what
/// the loaded server says it did. `None` for a backend with no `/props` (an image or
/// STT server) or a shape we don't recognize.
fn served_context(agent: &ureq::Agent, endpoint: &str, model: &str) -> Option<(u64, u64)> {
    let url = format!("{endpoint}/upstream/{model}/props");
    let text = agent.get(&url).call().ok()?.into_string().ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    // `total_slots` is `-np`; a server built without it runs a single slot.
    let slots = json.get("total_slots").and_then(serde_json::Value::as_u64).unwrap_or(1);
    // Per-slot context. Older builds expose it only at the top level.
    let context = json
        .pointer("/default_generation_settings/n_ctx")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| json.get("n_ctx").and_then(serde_json::Value::as_u64))?;
    Some((context, slots))
}

/// The `(context, slots)` a launch command declares, where `slots` is `None` when
/// `-np` is absent (llama.cpp then picks its own default, which is **not** 1 and is
/// not knowable from the command).
///
/// `None` when there is nothing to compare: no `-c`, or `-c 0` ("take the model's
/// trained context", which only the loaded server can resolve).
fn declared_context(cmd: &str) -> Option<(u64, Option<u64>)> {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let value_of = |names: &[&str]| -> Option<u64> {
        let index = tokens.iter().position(|token| names.contains(token))?;
        tokens.get(index + 1)?.parse().ok()
    };
    let context = value_of(&["-c", "--ctx-size"])?;
    let slots = value_of(&["-np", "--parallel"]).filter(|slots| *slots > 0);
    (context > 0).then_some((context, slots))
}

/// Whether the model llama-swap loaded is the one whose command we hashed.
#[derive(Debug, PartialEq, Eq)]
enum Serving {
    /// The live server's context matches the config's.
    Confirmed,
    /// It does not: llama-swap is serving a different command than the config
    /// declares, so the footprint about to be recorded belongs to neither. Both
    /// sides are rendered for the failure message.
    Mismatch { declared: String, served: String },
    /// Nothing to compare against, so the recording is unconfirmed (not wrong).
    Unconfirmed,
}

/// Cross-check the loaded server against the config we hashed.
///
/// `measure` derives the param-hash and `params` from the config file on disk, but
/// the load runs through llama-swap, which serves whatever config **it** last
/// hot-reloaded. When the two disagree (the file was edited underneath it, the
/// reload hasn't landed, or `--config` points at a copy), the footprint would be
/// stored under the new hash while describing a command that never ran. That entry
/// never self-corrects, because the hash then looks present.
fn check_serving(
    agent: &ureq::Agent,
    endpoint: &str,
    record: &ModelRecord,
) -> Serving {
    // Only llama.cpp servers answer /props; image and STT backends do not, and a
    // proxy never loads at all.
    if !matches!(record.model_type, ModelType::Llm | ModelType::Embed | ModelType::Rerank) {
        return Serving::Unconfirmed;
    }
    let Some(declared) = declared_context(&record.cmd) else {
        return Serving::Unconfirmed;
    };
    let Some(served) = served_context(agent, endpoint, &record.id) else {
        return Serving::Unconfirmed;
    };
    compare_context(declared, served)
}

/// The pure half of [`check_serving`]: does a `(context, explicit_slots)`
/// declaration match the `(per_slot_context, slots)` a live server reports?
///
/// Deliberately agnostic about whether `-c` means the total context or the per-slot
/// context, because **it is both**, depending on flags. Measured against one
/// llama-swap v247 with one llama.cpp build:
///
/// | config | reported `n_ctx` | `total_slots` |
/// |---|---|---|
/// | `-c 262144 -np 2` | 131072 (`-c` / slots) | 2 |
/// | `-c 8192` (no `-np`) | 8192 (`-c` itself) | 4 |
///
/// So the declared context is accepted when it matches **either** the per-slot
/// figure or the reconstructed total, and the slot count is only compared when the
/// command states `-np` (its default is neither 1 nor derivable). This still catches
/// what matters: any change to `-c` makes both candidates wrong at once, which is
/// the whole failure mode (a footprint filed under a command that never ran).
fn compare_context(declared: (u64, Option<u64>), served: (u64, u64)) -> Serving {
    let (declared_context, declared_slots) = declared;
    let (served_per_slot, served_slots) = served;
    let served_total = served_per_slot.saturating_mul(served_slots);
    let rendered = |context: u64, slots: Option<u64>| match slots {
        Some(slots) => format!("-c {context} -np {slots}"),
        None => format!("-c {context}"),
    };
    let mismatch = || Serving::Mismatch {
        declared: rendered(declared_context, declared_slots),
        served: format!("{served_per_slot} per slot across {served_slots} slot(s)"),
    };

    if let Some(slots) = declared_slots {
        if slots != served_slots {
            return mismatch();
        }
    }
    // A per-slot figure derived by division can fall short of the total by at most
    // `slots - 1` tokens; the per-slot reading itself is exact.
    let matches_total = served_total.abs_diff(declared_context) <= served_slots.saturating_sub(1);
    let matches_per_slot = served_per_slot == declared_context;
    if matches_total || matches_per_slot {
        Serving::Confirmed
    } else {
        mismatch()
    }
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

/// What a load-trigger request did.
#[derive(Debug, PartialEq, Eq)]
enum TriggerOutcome {
    /// The backend answered with this HTTP status.
    Answered(u16),
    /// The request never completed (transport error, timeout, refused).
    Failed(String),
}

/// A fired load-trigger, awaited by [`await_allocation`].
///
/// The request runs on its own thread, because the load has to be in flight while
/// `/running` is polled for `ready`. The sweep then waits for it before sampling, so a
/// lazily-allocating backend is measured after it allocated, and no request from one
/// model is still running during the next model's measurement window.
struct Trigger {
    outcome: Receiver<TriggerOutcome>,
    /// When the request was fired. The wait budget is measured from here, not from
    /// when the sweep starts waiting, so a model's total cost is bounded by the
    /// trigger timeout instead of stacking it on top of the ready timeout.
    fired_at: Instant,
}

/// Fire the model's load-trigger on its own thread and hand back a [`Trigger`].
///
/// The request's own timeout matches the wait budget, so when the sweep stops waiting
/// the thread is already unwinding rather than allocating behind its back.
fn trigger(model: &str, model_type: ModelType, options: &MeasureOptions) -> Trigger {
    let model = model.to_string();
    let endpoint = options.endpoint.clone();
    let image_size = options.probe_image_size.clone();
    let timeout = options.trigger_timeout;
    let (sender, outcome) = mpsc::channel();
    thread::spawn(move || {
        let agent = ureq::builder().timeout(timeout).build();
        let response = match model_type {
            ModelType::Embed => agent
                .post(&format!("{endpoint}/v1/embeddings"))
                .send_json(serde_json::json!({"model": model, "input": "x"})),
            ModelType::Rerank => agent.post(&format!("{endpoint}/v1/rerank")).send_json(
                serde_json::json!({"model": model, "query": "x", "documents": ["a", "b"]}),
            ),
            ModelType::Image => agent.post(&format!("{endpoint}/v1/images/generations")).send_json(
                serde_json::json!({"model": model, "prompt": "a cube", "size": image_size}),
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
        // A non-2xx is an *answer*, not a transport failure: ureq models it as an
        // error, but it tells us the backend replied, and the status explains why a
        // load produced nothing worth recording.
        let outcome = match response {
            Ok(response) => TriggerOutcome::Answered(response.status()),
            Err(ureq::Error::Status(status, _)) => TriggerOutcome::Answered(status),
            Err(error) => TriggerOutcome::Failed(error.to_string()),
        };
        let _ = sender.send(outcome);
    });
    Trigger { outcome, fired_at: Instant::now() }
}

/// Whether a model's allocation is known to have finished, and the highest occupancy
/// seen while it was allocating.
#[derive(Debug, PartialEq)]
enum Allocation {
    /// The trigger completed successfully: whatever it allocated is now resident.
    Confirmed { peak: f64 },
    /// The trigger itself failed or was refused, so nothing about the reading can be
    /// trusted - the model may be half-loaded, or loaded and then torn down.
    Rejected { reason: String },
    /// It never came back inside the budget. The sensor reading may be a mid-load
    /// plateau, so it is recorded and flagged, never silently trusted.
    Unconfirmed { peak: f64, reason: String },
}

/// Wait for the load-trigger to finish, sampling occupancy for its peak meanwhile.
///
/// This is the fix for the under-measurement: for sd-server the generation request
/// *is* the allocation, so its completion - not llama-swap's `ready` - is the moment
/// a footprint becomes meaningful.
///
/// `budget` is measured from when the trigger was fired, so waiting for it never adds
/// to the time already spent waiting for `ready`; a model that goes wrong costs at
/// most one budget, not two.
fn await_allocation(trigger: &Trigger, gpu: &dyn GpuMemory, budget: Duration) -> Allocation {
    let mut peak = 0.0_f64;
    loop {
        peak = peak.max(gpu.used_gb().unwrap_or(0.0));
        match trigger.outcome.recv_timeout(TRIGGER_SAMPLE_INTERVAL) {
            Ok(TriggerOutcome::Answered(status)) if (200..300).contains(&status) => {
                return Allocation::Confirmed { peak };
            }
            Ok(TriggerOutcome::Answered(status)) => {
                return Allocation::Rejected {
                    reason: format!("the load-trigger returned HTTP {status}"),
                };
            }
            Ok(TriggerOutcome::Failed(error)) => {
                return Allocation::Rejected {
                    reason: format!("the load-trigger did not complete: {error}"),
                };
            }
            Err(RecvTimeoutError::Timeout) => {
                if trigger.fired_at.elapsed() >= budget {
                    return Allocation::Unconfirmed {
                        peak,
                        reason: format!(
                            "the load-trigger was still running {}s after it was fired, so the \
                             reading may be a mid-load plateau rather than the finished footprint",
                            budget.as_secs()
                        ),
                    };
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Allocation::Unconfirmed {
                    peak,
                    reason: "the load-trigger ended without reporting an outcome".to_string(),
                };
            }
        }
    }
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

/// A settled occupancy reading.
struct Stabilized {
    /// The occupancy accepted as the footprint, in GB.
    used: f64,
    /// Did occupancy go quiet, or did `max_wait` run out while it was still moving? A
    /// reading that never settled is not a finished allocation, so the caller records it
    /// as unconfirmed rather than as a footprint.
    settled: bool,
    /// Highest occupancy seen while sampling.
    peak: f64,
}

/// Sample occupancy every `interval` until `hold` consecutive readings are within
/// `eps` GB (KV and compute buffers finish allocating after `ready`), or `max_wait`
/// elapses. `interval` is a parameter so the sampling logic is testable in
/// milliseconds instead of in real sweep time.
fn stabilize(
    gpu: &dyn GpuMemory,
    max_wait: Duration,
    interval: Duration,
    eps: f64,
    hold: u32,
) -> Stabilized {
    let mut previous: Option<f64> = None;
    let mut stable = 0;
    let mut peak = 0.0_f64;
    let start = Instant::now();
    while start.elapsed() < max_wait {
        let current = gpu.used_gb().unwrap_or(0.0);
        peak = peak.max(current);
        if let Some(prev) = previous {
            if (current - prev).abs() < eps {
                stable += 1;
                if stable >= hold {
                    return Stabilized { used: current, settled: true, peak };
                }
            } else {
                stable = 0;
            }
        }
        previous = Some(current);
        thread::sleep(interval);
    }
    let current = gpu.used_gb().unwrap_or(0.0);
    Stabilized { used: current, settled: false, peak: peak.max(current) }
}

/// Total size in GB of the weight files the command names, host-mapped and stat'ed.
///
/// `None` when none of them can be read (an unmapped container path, a remote mount),
/// in which case no floor check is possible and none is claimed.
fn weights_gb(record: &ModelRecord, policy: &Policy) -> Option<f64> {
    let mut total = 0.0;
    let mut readable = false;
    for file in weight_files(&record.cmd) {
        if let Ok(metadata) = std::fs::metadata(policy.to_host(&file)) {
            total += metadata.len() as f64 / BYTES_PER_GIB;
            readable = true;
        }
    }
    readable.then_some(total)
}

/// Upsert one measurement into the model's store file, refreshing the type/file the
/// entry is filed under.
///
/// A `FAILED` result never overwrites an existing `ok` footprint at the same hash. A
/// measurement is data the operator paid GPU time for, the store's rule is that nothing
/// is auto-deleted (SPEC §2), and a bad load in this sweep (a rejected trigger, a
/// timeout during a `--force` re-measure) is no evidence against the stored number,
/// while clobbering it would silently drop the model out of every future matrix. The
/// failure is reported in the sweep summary either way.
fn store_measurement(store: &Store, record: &ModelRecord, measurement: Measurement) -> Result<()> {
    let mut model_store = store.read_model(&record.id)?.unwrap_or_else(|| ModelStore {
        model_type: record.model_type.as_str().to_string(),
        file: record.primary_file.clone(),
        measurements: Default::default(),
    });
    if !measurement.is_ok()
        && model_store
            .measurements
            .get(&record.param_hash)
            .is_some_and(Measurement::is_ok)
    {
        return Ok(());
    }
    model_store.model_type = record.model_type.as_str().to_string();
    model_store.file = record.primary_file.clone();
    model_store.measurements.insert(record.param_hash.clone(), measurement);
    store.write_model(&record.id, &model_store)
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

    // empty baseline (with its per-pool split, when the device has one, so the
    // recorded deltas can be split the same way)
    unload_all(&agent, &options.endpoint, Duration::from_secs(4));
    summary.baseline = gpu.used_gb()?;
    let baseline_split = gpu.used_split_gb();

    // The roster llama-swap is actually serving. A model the config declares but
    // llama-swap doesn't know is the loud half of a config mismatch: measuring it
    // would load nothing (or something else).
    let served = served_ids(&agent, &options.endpoint);

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

        // cache hit? An *unconfirmed* entry is not one: it may be a mid-load plateau
        // rather than a footprint, so re-measuring it is the cheap half of Principle 6
        // (extra work beats wrong reuse), and it self-heals without the operator having
        // to know which entries to distrust. A store holding no confirmations therefore
        // re-measures in full, and one holding them re-measures only what is suspect.
        if !options.force {
            if let Some(existing) = store.select(&record.id, &record.param_hash)? {
                if existing.is_confirmed() {
                    summary.cached.push(record.id.clone());
                    continue;
                }
            }
        }

        // Weights on disk: a floor on the footprint of a fully offloaded model, and
        // the one cross-check that needs no GPU and no cooperation from the backend.
        let weights = weights_gb(record, policy);

        unload_all(&agent, &options.endpoint, Duration::from_secs(2));
        let fired = trigger(&record.id, record.model_type, options);
        let load_seconds = wait_ready(&agent, &options.endpoint, &record.id, options.load_timeout);

        // Every path below yields at most one measurement and then falls through to a
        // single store-and-unload. `None` means record nothing at all, which is
        // reserved for a serving mismatch: that reading describes neither the config's
        // command nor the served one, so storing it is the one outcome that must not
        // happen (it would look like a present hash forever after).
        let recorded: Option<Measurement> = match load_seconds {
            None => {
                // The trigger may still be in flight. Wait it out (bounded) so a
                // request from this model can never allocate during the next model's
                // measurement window, and use whatever it reports to explain the
                // failure - "the load-trigger returned HTTP 502" beats "timed out".
                let trigger_note =
                    match await_allocation(&fired, gpu.as_ref(), options.trigger_timeout) {
                        Allocation::Rejected { reason } => Some(reason),
                        _ => None,
                    };
                // An id llama-swap doesn't advertise is the likeliest cause worth
                // naming: it usually means it is serving a different config than the
                // one being measured. Only a hint, since an `unlisted` model is
                // absent from the roster and still loadable (SPEC §8).
                let unknown_to_llama_swap =
                    served.as_ref().is_some_and(|ids| !ids.contains(&record.id));
                let base = if unknown_to_llama_swap {
                    "load timed out or exited, and llama-swap does not list this model \
                     id - it may be serving a different config than the one being \
                     measured (reload it, or point --config at the file it loaded)"
                } else {
                    "load timed out or exited"
                };
                summary.failed.push(Failure {
                    id: record.id.clone(),
                    reason: match trigger_note {
                        Some(note) => format!("{base} ({note})"),
                        None => base.to_string(),
                    },
                });
                Some(Measurement {
                    status: "FAILED".to_string(),
                    params: memory_cmd(&record.cmd),
                    measured_at: today.clone(),
                    weights_gb: weights.map(round2),
                    ..Default::default()
                })
            }
            // Is the server that just loaded running the command we hashed? On a
            // mismatch the reading belongs to neither command (Principle 2, and
            // Principle 1 downstream).
            Some(load) => match check_serving(&agent, &options.endpoint, record) {
                Serving::Mismatch { declared, served } => {
                    summary.failed.push(Failure {
                        id: record.id.clone(),
                        reason: format!(
                            "llama-swap loaded {served} while this config declares \
                             {declared}, so it is serving a different command - no \
                             footprint recorded (reload llama-swap, or measure the \
                             config it actually loaded)"
                        ),
                    });
                    None
                }
                serving => {
                    if serving == Serving::Unconfirmed {
                        summary.unverified_serving.push(record.id.clone());
                    }
                    let serving_verified = Some(serving == Serving::Confirmed);

                    // The load-trigger's completion is the allocation signal: a
                    // diffusion backend allocates *during* the generation, so `ready`
                    // is far too early. Peak occupancy is tracked while we wait.
                    let allocating =
                        match await_allocation(&fired, gpu.as_ref(), options.trigger_timeout) {
                            Allocation::Confirmed { peak } => Some((peak, None)),
                            Allocation::Unconfirmed { peak, reason } => Some((peak, Some(reason))),
                            Allocation::Rejected { reason } => {
                                summary.failed.push(Failure {
                                    id: record.id.clone(),
                                    reason: format!("{reason}, so no footprint was recorded"),
                                });
                                None
                            }
                        };

                    match allocating {
                        None => Some(Measurement {
                            status: "FAILED".to_string(),
                            params: memory_cmd(&record.cmd),
                            measured_at: today.clone(),
                            weights_gb: weights.map(round2),
                            serving_verified,
                            ..Default::default()
                        }),
                        Some((trigger_peak, trigger_note)) => {
                            let settled = stabilize(
                                gpu.as_ref(),
                                STABILIZE_MAX_WAIT,
                                STABILIZE_INTERVAL,
                                STABILIZE_EPS,
                                STABILIZE_HOLD,
                            );
                            // Confirmed needs both halves: the trigger finished *and*
                            // occupancy then stopped moving. Either one missing means
                            // the number may be incomplete, which is recorded rather
                            // than assumed away.
                            let confirmed = trigger_note.is_none() && settled.settled;
                            if !confirmed {
                                summary.unconfirmed_allocation.push(Failure {
                                    id: record.id.clone(),
                                    reason: trigger_note.unwrap_or_else(|| {
                                        format!(
                                            "occupancy was still changing after {}s of sampling, \
                                             so the footprint may be incomplete",
                                            STABILIZE_MAX_WAIT.as_secs()
                                        )
                                    }),
                                });
                            }

                            let used = settled.used;
                            // Read the split at the same settled point as the total.
                            // Both are None on a device with a single (or unified)
                            // pool, and are then omitted from the store rather than
                            // written as zeros.
                            let (d_vram, d_gtt, abs_vram, abs_gtt) =
                                match (gpu.used_split_gb(), baseline_split) {
                                    (Some((vram, gtt)), Some((base_vram, base_gtt))) => (
                                        Some(round2(vram - base_vram)),
                                        Some(round2(gtt - base_gtt)),
                                        Some(round2(vram)),
                                        Some(round2(gtt)),
                                    ),
                                    _ => (None, None, None, None),
                                };
                            let d_total = round2(used - summary.baseline);
                            let peak = trigger_peak.max(settled.peak) - summary.baseline;
                            let measurement = Measurement {
                                status: "ok".to_string(),
                                d_total,
                                abs_total: round2(used),
                                d_vram,
                                d_gtt,
                                abs_vram,
                                abs_gtt,
                                load_s: round1(load),
                                allocation_confirmed: Some(confirmed),
                                serving_verified,
                                peak_total: Some(round2(peak.max(d_total))),
                                weights_gb: weights.map(round2),
                                params: memory_cmd(&record.cmd),
                                measured_at: today.clone(),
                            };
                            if let (true, Some(ratio), Some(weights)) = (
                                measurement.below_weight_floor(),
                                measurement.weight_ratio(),
                                weights,
                            ) {
                                summary.below_weight_floor.push(Failure {
                                    id: record.id.clone(),
                                    reason: format!(
                                        "footprint {d_total:.2} GB is only {:.0}% of the \
                                         {weights:.2} GB of weight files its command names; a \
                                         fully offloaded model cannot hold much less than its \
                                         weights, so this may be under-measured (partial offload \
                                         with -ngl/-ot/--cpu-moe is a legitimate reason to sit \
                                         lower)",
                                        ratio * 100.0
                                    ),
                                });
                            }
                            summary.measured.push(record.id.clone());
                            Some(measurement)
                        }
                    }
                }
            },
        };

        if let Some(measurement) = recorded {
            store_measurement(store, record, measurement)?;
        }
        // The trigger has been awaited (or waited out) by now, so nothing of this
        // model's is still allocating; release the channel and clear the pool.
        drop(fired);
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A sensor that replays a scripted sequence of readings, so the sampling logic
    /// can be exercised without a GPU. Past the end of the script it holds the last
    /// value, or keeps climbing by `climb` - a model that is still allocating does not
    /// politely stop when the script does, and a fake that flattens out would let a
    /// never-settling reading look settled.
    struct ScriptedGpu {
        readings: Vec<f64>,
        next: AtomicUsize,
        climb: f64,
    }

    impl ScriptedGpu {
        fn new(readings: &[f64]) -> Self {
            ScriptedGpu { readings: readings.to_vec(), next: AtomicUsize::new(0), climb: 0.0 }
        }

        /// Occupancy that rises by `step` GB on every read, forever.
        fn climbing(step: f64) -> Self {
            ScriptedGpu { readings: vec![step], next: AtomicUsize::new(0), climb: step }
        }
    }

    impl GpuMemory for ScriptedGpu {
        fn label(&self) -> String {
            "scripted".to_string()
        }
        fn total_gb(&self) -> Result<f64> {
            Ok(100.0)
        }
        fn used_gb(&self) -> Result<f64> {
            let index = self.next.fetch_add(1, Ordering::SeqCst);
            let last = self.readings.len() - 1;
            Ok(match index <= last {
                true => self.readings[index],
                false => self.readings[last] + self.climb * (index - last) as f64,
            })
        }
    }

    fn trigger_holding(outcome: Option<TriggerOutcome>) -> (mpsc::Sender<TriggerOutcome>, Trigger) {
        let (sender, receiver) = mpsc::channel();
        if let Some(outcome) = outcome {
            sender.send(outcome).unwrap();
        }
        // The sender is handed back so the caller can keep the channel open: dropping
        // it would look like a finished trigger rather than a stalled one.
        (sender, Trigger { outcome: receiver, fired_at: Instant::now() })
    }

    /// The report's core failure: a reading taken while the model is still loading.
    /// `stabilize` cannot tell a mid-load plateau from a finished one - that is what
    /// awaiting the trigger is for - but it must at least never *claim* a reading
    /// settled when occupancy was still climbing when time ran out.
    #[test]
    fn stabilize_reports_whether_it_actually_settled() {
        let interval = Duration::from_millis(1);

        // Still climbing when the budget runs out: not settled, and the peak is kept.
        let climbing = ScriptedGpu::climbing(4.0);
        let result = stabilize(&climbing, Duration::from_millis(20), interval, 0.03, 3);
        assert!(!result.settled, "a moving reading must not be reported as settled");
        assert!(result.peak >= 13.0, "peak {} should track the climb", result.peak);

        // Quiet for three consecutive samples: settled, at the quiet value.
        let quiet = ScriptedGpu::new(&[4.0, 16.10, 16.11, 16.10, 16.10]);
        let result = stabilize(&quiet, Duration::from_secs(5), interval, 0.03, 3);
        assert!(result.settled);
        assert!((result.used - 16.10).abs() < 1e-9, "used {}", result.used);

        // A plateau that holds long enough is accepted, which is precisely why the
        // trigger has to be awaited first: three quiet samples mid-load look
        // identical to three quiet samples post-load.
        let plateau = ScriptedGpu::new(&[8.87, 8.87, 8.87, 8.87]);
        let result = stabilize(&plateau, Duration::from_secs(5), interval, 0.03, 3);
        assert!(result.settled && (result.used - 8.87).abs() < 1e-9);
    }

    /// The trigger's outcome decides whether a footprint counts as confirmed.
    #[test]
    fn a_finished_trigger_confirms_and_a_stalled_one_does_not() {
        let gpu = ScriptedGpu::new(&[2.0]);

        // 2xx: the request that does the allocating finished, so what is resident now
        // is the whole footprint.
        let (_sender, trigger) = trigger_holding(Some(TriggerOutcome::Answered(200)));
        assert!(matches!(
            await_allocation(&trigger, &gpu, Duration::from_secs(5)),
            Allocation::Confirmed { .. }
        ));

        // A non-2xx answer: the model may be half-loaded or already tearing down, so
        // nothing about the reading is trustworthy.
        let (_sender, trigger) = trigger_holding(Some(TriggerOutcome::Answered(500)));
        match await_allocation(&trigger, &gpu, Duration::from_secs(5)) {
            Allocation::Rejected { reason } => assert!(reason.contains("500"), "{reason}"),
            other => panic!("expected Rejected, got {other:?}"),
        }

        // A transport failure is equally disqualifying.
        let (_sender, trigger) =
            trigger_holding(Some(TriggerOutcome::Failed("connection reset".into())));
        assert!(matches!(
            await_allocation(&trigger, &gpu, Duration::from_secs(5)),
            Allocation::Rejected { .. }
        ));

        // Never answers inside the budget: unconfirmed, carrying the peak it saw
        // while waiting (9.0), not silently trusted.
        let peaky = ScriptedGpu::new(&[1.0, 9.0, 3.0]);
        let (_sender, trigger) = trigger_holding(None);
        match await_allocation(&trigger, &peaky, Duration::from_millis(600)) {
            Allocation::Unconfirmed { peak, reason } => {
                assert!((peak - 9.0).abs() < 1e-9, "peak {peak}");
                assert!(reason.contains("mid-load plateau"), "{reason}");
                assert!(reason.contains("after it was fired"), "{reason}");
            }
            other => panic!("expected Unconfirmed, got {other:?}"),
        }
    }

    /// The weights floor totals every weight file the command names, through the
    /// container→host path map, and ignores what it cannot read.
    #[test]
    fn weights_are_totalled_across_host_mapped_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("unet.gguf"), vec![0u8; 3 * 1024 * 1024]).unwrap();
        std::fs::write(dir.path().join("vae.safetensors"), vec![0u8; 1024 * 1024]).unwrap();
        let mut policy = Policy::default();
        policy.paths.insert("/sd".to_string(), dir.path().display().to_string());

        let record = ModelRecord::from_expanded(
            "img",
            "/opt/sdcpp/bin/sd-server --diffusion-model /sd/unet.gguf \
             --vae /sd/vae.safetensors --t5xxl /sd/absent.safetensors",
        );
        // 4 MiB readable; the missing file contributes nothing rather than aborting.
        let total = weights_gb(&record, &policy).expect("readable weights");
        assert!((total - 4.0 / 1024.0).abs() < 1e-9, "total {total}");

        // Nothing readable at all → no floor is claimed (an unmapped container path).
        let unmapped = ModelRecord::from_expanded(
            "img",
            "/opt/sdcpp/bin/sd-server --diffusion-model /elsewhere/u.gguf",
        );
        assert_eq!(weights_gb(&unmapped, &Policy::default()), None);
    }

    /// A failed load must not erase a footprint that was measured successfully: the
    /// store is append-and-keep, and losing the number would quietly drop the model
    /// from every future matrix.
    #[test]
    fn a_failure_never_overwrites_a_good_footprint() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path().join("measurements"));
        let record = ModelRecord::from_expanded("chat", "/app/llama-server -m /m.gguf -c 4096");

        let good = Measurement {
            status: "ok".to_string(),
            d_total: 30.0,
            allocation_confirmed: Some(true),
            ..Default::default()
        };
        store_measurement(&store, &record, good).unwrap();

        let failed = Measurement { status: "FAILED".to_string(), ..Default::default() };
        store_measurement(&store, &record, failed).unwrap();
        let kept = store.select("chat", &record.param_hash).unwrap().expect("footprint kept");
        assert_eq!(kept.d_total, 30.0);

        // With nothing to protect, the failure is recorded as usual (a `FAILED` entry
        // documents that this exact command was tried and didn't load).
        let fresh = ModelRecord::from_expanded("other", "/app/llama-server -m /o.gguf -c 4096");
        store_measurement(
            &store,
            &fresh,
            Measurement { status: "FAILED".to_string(), ..Default::default() },
        )
        .unwrap();
        let stored = store.read_model("other").unwrap().unwrap();
        assert_eq!(stored.measurements[&fresh.param_hash].status, "FAILED");
    }

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
    fn declared_context_reads_c_and_np() {
        let cmd = "/app/llama-server -m /m.gguf -ngl 99 -c 524288 -np 2 -fa on";
        assert_eq!(declared_context(cmd), Some((524288, Some(2))));
        assert_eq!(
            declared_context("/app/llama-server -m /m.gguf --ctx-size 4096 --parallel 4"),
            Some((4096, Some(4)))
        );
        // An absent `-np` stays absent: llama.cpp's default is not 1 (it was 4 on the
        // build this was validated against), so it must not be assumed.
        assert_eq!(declared_context("/app/llama-server -m /m.gguf -c 8192"), Some((8192, None)));
        // Nothing to compare: no `-c`, or `-c 0` (resolved from the model at load).
        assert_eq!(declared_context("/app/llama-server -m /m.gguf -ngl 99"), None);
        assert_eq!(declared_context("/app/llama-server -m /m.gguf -c 0 -fa on"), None);
    }

    /// Cases taken from a live llama-swap v247, so the two `-c` semantics are pinned
    /// by observation rather than by reading the flag documentation.
    #[test]
    fn a_served_context_that_differs_is_a_mismatch() {
        // Live: `-c 262144 -np 2` loads as 131072 per slot across 2 slots, so here
        // `-c` is the total.
        assert_eq!(compare_context((262144, Some(2)), (131072, 2)), Serving::Confirmed);
        // Live: `-c 8192` with no `-np` loads as 8192 per slot across 4 slots, so here
        // the same flag is the per-slot value. Assuming either reading alone would
        // fail one of these two.
        assert_eq!(compare_context((8192, None), (8192, 4)), Serving::Confirmed);

        // The failure this exists for: the config said `-c 393216 -np 2` while
        // llama-swap was still serving `-c 524288 -np 2`, and the 524288 footprint
        // was filed under the 393216 hash. Neither reading of 393216 fits.
        assert_eq!(
            compare_context((393216, Some(2)), (262144, 2)),
            Serving::Mismatch {
                declared: "-c 393216 -np 2".into(),
                served: "262144 per slot across 2 slot(s)".into(),
            }
        );
        // A one-token change is caught too (the reported reproduction).
        assert_eq!(
            compare_context((8191, None), (8192, 4)),
            Serving::Mismatch {
                declared: "-c 8191".into(),
                served: "8192 per slot across 4 slot(s)".into(),
            }
        );
        // An explicitly declared slot count that the server did not honour.
        assert_eq!(
            compare_context((524288, Some(2)), (131072, 4)),
            Serving::Mismatch {
                declared: "-c 524288 -np 2".into(),
                served: "131072 per slot across 4 slot(s)".into(),
            }
        );
        // Integer division of an odd total across slots is not a mismatch.
        assert_eq!(compare_context((8191, Some(2)), (4095, 2)), Serving::Confirmed);
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
