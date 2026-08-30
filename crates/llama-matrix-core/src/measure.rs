//! Phase 1 - the solo-footprint sweep. Loads each model alone, reads real GPU
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
use crate::param_hash::{memory_cmd, token_difference};
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

/// Occupancy above the sweep's lowest empty-pool reading that counts as "something
/// else is resident" rather than sensor noise. Two orders of magnitude above
/// `STABILIZE_EPS` and two below any model worth measuring, so it separates the two
/// without a judgement call.
const CONTENTION_EPS: f64 = 0.25;

/// How far a re-measure may move from the stored footprint before it is reported.
/// The absolute floor covers small models, where a percentage is meaningless; the
/// fraction covers large ones, where 0.25 GB is inside normal allocator variation.
const REMEASURE_EPS_GB: f64 = 0.25;
const REMEASURE_EPS_FRACTION: f64 = 0.02;

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

/// A sweep's progress, for a frontend that wants to show it.
///
/// A full sweep loads every model in turn and can run for the better part of an
/// hour, most of it inside one `wait_ready`. Reporting nothing until the summary is
/// indistinguishable from being hung, and the operator's only recourse is to watch
/// the GPU from another terminal. The core does not print (Principle 9 puts progress
/// on stderr and keeps `--json` clean), so it hands the frontend these instead.
#[derive(Debug)]
pub enum Progress<'a> {
    /// About to load model `index` of `total` (both 1-based counts of the worklist).
    Loading { index: usize, total: usize, id: &'a str },
    /// Finished with it. `outcome` is a rendered one-liner ("19.41 GB in 12.0 s",
    /// "cached", "FAILED").
    Done { index: usize, total: usize, id: &'a str, outcome: String },
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
    /// Models measured while the pool held something other than the model under
    /// test, so the footprint may include memory that is not its own
    /// (`cache::Measurement::contended`). Over-measured, therefore safe, therefore
    /// reported rather than gated: the fix is to quiesce the box and `--force`.
    pub contended: Vec<Failure>,
    /// Models whose footprint was recovered from a store file the config no longer
    /// names, rather than re-loaded. A rename is not a new footprint.
    pub adopted: Vec<Adopted>,
    /// Models whose fresh footprint disagrees with the one already stored under the
    /// same param-hash by more than a tolerance. Same box, same flags, two numbers:
    /// at most one of them is right, and until this was reported the new one simply
    /// overwrote the old with no trace that anything had changed.
    pub changed: Vec<Remeasured>,
    /// The sweep never saw the pool verifiably empty, so it could not establish the
    /// box baseline and kept the stored one. Every footprint it did record was taken
    /// against a pool holding something else.
    pub no_empty_pool: bool,
    /// The box baseline the previous sweep recorded, when it differs from this
    /// sweep's by more than `CONTENTION_EPS`.
    ///
    /// The empty pool is the one quantity on the box that should not move, so a move
    /// is worth a sentence either way. An *upward* one is the story: `/running` can
    /// report the pool empty while the device is still holding a model llama-swap
    /// has stopped accounting for, and that reading passes every check the sweep has
    /// except this one - comparison against what the same box read last time.
    pub baseline_was: Option<f64>,
    pub baseline: f64,
    pub detected_total: f64,
    /// Host RAM with nothing loaded, and the box total, when the box can report
    /// them. Recorded so `build` can check a pack against the host as well as the
    /// GPU (SPEC §7.4); `None` on a box with no way to read it.
    pub host_baseline: Option<f64>,
    pub host_total: Option<f64>,
}

/// A footprint recovered from a renamed model's orphaned store file.
#[derive(Debug, Clone, Serialize)]
pub struct Adopted {
    pub id: String,
    /// The id the store had it filed under.
    pub from: String,
    pub d_total: f64,
}

/// A re-measure that did not reproduce the stored footprint.
#[derive(Debug, Clone, Serialize)]
pub struct Remeasured {
    pub id: String,
    pub previous: f64,
    pub current: f64,
    /// When the stored number was taken, so a disagreement can be read against what
    /// else changed on the box that day.
    pub previous_measured_at: String,
}

impl Remeasured {
    /// Do two footprints of the same `(model, param-hash)` disagree by enough to be
    /// worth an operator's attention?
    fn disagree(previous: f64, current: f64) -> bool {
        let tolerance = REMEASURE_EPS_GB.max(previous.abs() * REMEASURE_EPS_FRACTION);
        (previous - current).abs() > tolerance
    }
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

/// One `GET /running` entry: what llama-swap says about a model it has loaded.
#[derive(Debug, Clone, Default)]
struct Running {
    /// `ready` | `starting` | `stopping` | ... (empty when the field is absent).
    state: String,
    /// The command llama-swap **launched**, when it reports one. This is the
    /// ground truth the serving cross-check wants: it is the served command
    /// itself, not an inference from what the loaded server says it did. Absent on
    /// a llama-swap too old to report it, which is why [`check_serving`] keeps the
    /// `/props` path as a fallback.
    cmd: Option<String>,
}

/// `{model: entry}` from `GET /running` (empty on any error).
fn running(agent: &ureq::Agent, endpoint: &str) -> HashMap<String, Running> {
    let url = format!("{endpoint}/running");
    let Ok(response) = agent.get(&url).call() else {
        return HashMap::new();
    };
    let Ok(text) = response.into_string() else {
        return HashMap::new();
    };
    parse_running(&text)
}

/// The pure half of [`running`]: `GET /running`'s body into `{model: entry}`.
fn parse_running(body: &str) -> HashMap<String, Running> {
    let mut entries = HashMap::new();
    let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
        return entries;
    };
    if let Some(array) = json.get("running").and_then(|value| value.as_array()) {
        for entry in array {
            if let Some(model) = entry.get("model").and_then(|value| value.as_str()) {
                entries.insert(
                    model.to_string(),
                    Running {
                        state: entry
                            .get("state")
                            .and_then(|value| value.as_str())
                            .unwrap_or("")
                            .to_string(),
                        cmd: entry
                            .get("cmd")
                            .and_then(|value| value.as_str())
                            .map(str::to_string)
                            .filter(|cmd| !cmd.trim().is_empty()),
                    },
                );
            }
        }
    }
    entries
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
    /// The served command matches the config's, on the memory tokens.
    Confirmed,
    /// It does not: llama-swap is serving a different command than the config
    /// declares, so the footprint about to be recorded belongs to neither.
    /// `detail` names both sides of the disagreement for the failure message.
    Mismatch { detail: String },
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
///
/// Two sources, strongest first. `GET /running` reports the command llama-swap
/// launched, which settles the question outright and for **every** backend: an
/// image or STT server has no `/props`, but llama-swap knows what it started. When
/// that field is absent (an older llama-swap), the served command is inferred from
/// what the loaded llama.cpp server says it did, via `/props`.
fn check_serving(
    agent: &ureq::Agent,
    endpoint: &str,
    record: &ModelRecord,
    served_cmd: Option<&str>,
) -> Serving {
    if let Some(served_cmd) = served_cmd {
        match compare_commands(&record.cmd, served_cmd) {
            Serving::Unconfirmed => {} // an unexpanded placeholder; fall through
            decided => return decided,
        }
    }
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

/// Compare the command the config declares against the one llama-swap launched,
/// on the **memory** tokens only ([`memory_cmd`]).
///
/// Comparing the memory command rather than the raw string is what makes this
/// usable: llama-swap assigns the port, and a `--reasoning`/`--jinja` difference is
/// footprint-neutral by construction. What is left is exactly the set of tokens the
/// param-hash is built from, so "these differ" and "the footprint would be filed
/// under a hash that never ran" are the same statement.
///
/// `Unconfirmed` when either side still carries an unexpanded `${...}`: llama-swap
/// substitutes `${PORT}`/`${PID}` at launch and the config file never can, so a
/// difference there says nothing about the memory flags.
fn compare_commands(declared_cmd: &str, served_cmd: &str) -> Serving {
    let declared = memory_cmd(declared_cmd);
    let served = memory_cmd(served_cmd);
    if declared.contains("${") || served.contains("${") {
        return Serving::Unconfirmed;
    }
    if declared == served {
        return Serving::Confirmed;
    }
    Serving::Mismatch {
        detail: format!(
            "this config declares `{}` where llama-swap launched `{}`",
            token_difference(&declared, &served),
            token_difference(&served, &declared)
        ),
    }
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
        detail: format!(
            "this config declares `{}` but the loaded server reports {served_per_slot} per slot \
             across {served_slots} slot(s)",
            rendered(declared_context, declared_slots)
        ),
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

/// The pool after an unload: its settled occupancy, who was still in it, and what
/// the host was holding at the same moment.
struct ClearedPool {
    /// Settled occupancy in GB.
    used: f64,
    /// Ids llama-swap still reported after the unload and the wait. Empty is the
    /// normal case and the only one where `used` is an *empty-pool* reading.
    residents: Vec<String>,
    /// Host RAM at the same point, when the box can report it. Read here rather than
    /// once per sweep for the same reason the GPU baseline is: a delta is only a
    /// footprint if the thing it is measured against is what was actually there.
    host: Option<platform::HostMemory>,
}

/// Unload everything and wait for the pool to actually empty: `/running` clear,
/// **then** occupancy settled.
///
/// The two waits are not redundant. `/running` is the proxy's bookkeeping, not the
/// device's occupancy: a model llama-swap has marked unloaded can still be holding
/// memory when the next sample is taken. Sleeping a fixed few seconds instead is a
/// guess that a large model outlives. Waiting for occupancy to go quiet is the same
/// positive evidence [`stabilize`] already demands after a *load*, applied to the
/// unload, and it is what makes the returned number a baseline rather than a hope.
fn clear_pool(agent: &ureq::Agent, endpoint: &str, gpu: &dyn GpuMemory) -> ClearedPool {
    let posted = agent.post(&format!("{endpoint}/api/models/unload")).call();
    if posted.is_err() {
        let _ = agent.get(&format!("{endpoint}/unload")).call(); // legacy fallback
    }
    let mut residents = Vec::new();
    for _ in 0..20 {
        residents = resident_ids(agent, endpoint, "");
        if residents.is_empty() {
            break;
        }
        thread::sleep(Duration::from_secs(1));
    }
    let used =
        stabilize(gpu, STABILIZE_MAX_WAIT, STABILIZE_INTERVAL, STABILIZE_EPS, STABILIZE_HOLD).used;
    ClearedPool { used, residents, host: platform::host_memory() }
}

/// Lower the sweep's empty-pool floor to this reading, if it is one.
///
/// A reading taken while llama-swap still reported a model resident is not an
/// empty-pool reading and must never become the box's floor: `build` treats that
/// number as always-resident, so another model's footprint filed there would inflate
/// the reserved floor of every future build.
fn note_floor(floor: &mut Option<f64>, host_floor: &mut Option<f64>, cleared: &ClearedPool) {
    if !cleared.residents.is_empty() {
        return;
    }
    *floor = Some(floor.map_or(cleared.used, |current: f64| current.min(cleared.used)));
    if let Some(host) = cleared.host {
        *host_floor = Some(host_floor.map_or(host.used_gb, |current: f64| current.min(host.used_gb)));
    }
}

/// Ids llama-swap reports resident, other than `except`, sorted so a message reads
/// the same on every sweep.
///
/// Necessary but not sufficient on its own: `/running` reflects what the proxy
/// believes, so a model it has already marked unloaded is invisible here while still
/// occupying memory. It is decisive for the question [`sweep`] asks of it, which is
/// not "is the pool clean" but "did the *set* of other residents change across this
/// model's window" - see the departure check there.
fn resident_ids(agent: &ureq::Agent, endpoint: &str, except: &str) -> Vec<String> {
    let mut ids: Vec<String> = running(agent, endpoint)
        .into_keys()
        .filter(|id| id != except)
        .collect();
    ids.sort();
    ids
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

/// The model reached `ready`: how long it took, and the command llama-swap reports
/// having launched for it (`None` on a llama-swap that does not report one).
struct Ready {
    load_s: f64,
    served_cmd: Option<String>,
}

/// Poll until the model is `ready` or gives up / tears down.
///
/// The served command is captured **here**, at the moment the model is ready, rather
/// than re-read later: it is already in the response that settled the state, and a
/// second request could catch a model llama-swap has begun to evict.
fn wait_ready(
    agent: &ureq::Agent,
    endpoint: &str,
    model: &str,
    timeout: Duration,
) -> Option<Ready> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        let entries = running(agent, endpoint);
        match entries.get(model).map(|entry| entry.state.as_str()) {
            Some("ready") => {
                return Some(Ready {
                    load_s: start.elapsed().as_secs_f64(),
                    served_cmd: entries.get(model).and_then(|entry| entry.cmd.clone()),
                })
            }
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
/// entry is filed under. Returns whether it wrote.
///
/// Two kinds of new reading are refused, both for the same reason: *more recent* is
/// not *better* when the newer number is known to be worse evidence, and a
/// measurement is GPU time the operator paid for that nothing else auto-deletes
/// (SPEC §2). Both refusals are reported in the sweep summary.
///
/// - A **`FAILED`** result never overwrites an `ok` footprint at the same hash. A
///   bad load in this sweep (a rejected trigger, a timeout during a `--force`
///   re-measure) is no evidence against the stored number, and clobbering it would
///   silently drop the model out of every future matrix.
/// - A **contended** reading never overwrites a reading recorded as clean. Something
///   else was in the pool, so the new number includes memory that is not this
///   model's, and the stored one was taken under conditions known to be better. Only
///   `contended: Some(false)` counts as clean: an entry written before the check
///   existed says nothing either way and is replaced as usual.
fn store_measurement(
    store: &Store,
    record: &ModelRecord,
    measurement: Measurement,
) -> Result<bool> {
    let mut model_store = store.read_model(&record.id)?.unwrap_or_else(|| ModelStore {
        model_type: record.model_type.as_str().to_string(),
        file: record.primary_file.clone(),
        measurements: Default::default(),
    });
    let existing = model_store.measurements.get(&record.param_hash);
    let refuse = match existing {
        Some(previous) if previous.is_ok() => {
            !measurement.is_ok()
                || (measurement.contended == Some(true) && previous.contended == Some(false))
        }
        _ => false,
    };
    if refuse {
        return Ok(false);
    }
    model_store.model_type = record.model_type.as_str().to_string();
    model_store.file = record.primary_file.clone();
    model_store.measurements.insert(record.param_hash.clone(), measurement);
    store.write_model(&record.id, &model_store)?;
    Ok(true)
}

/// Run the sweep. Detects the GPU (errors if none - measure needs a sensor).
///
/// `progress` is called as each model starts and finishes; pass `&|_| {}` to ignore
/// it. See [`Progress`] for why the core reports rather than prints.
pub fn sweep(
    records: &[ModelRecord],
    store: &Store,
    policy: &Policy,
    options: &MeasureOptions,
    progress: &dyn Fn(Progress),
) -> Result<MeasureSummary> {
    let gpu = platform::detect().context("measure needs a GPU sensor")?;
    let _lock = LockGuard::acquire(store.dir())?;
    let agent = poll_agent();
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();

    let mut summary = MeasureSummary {
        detected_total: gpu.total_gb()?,
        ..Default::default()
    };

    // The empty-pool floor: the lowest settled occupancy seen with the pool
    // **verifiably** empty, over the whole sweep. Each model's delta is taken
    // against its own pre-load baseline (below); this is the box-level number
    // `build` uses as the always-resident floor, and a minimum is the right
    // estimator for it because contamination only ever adds occupancy.
    //
    // `None` until a clean reading is obtained, so a sweep run on a box that never
    // emptied cannot quietly file a contaminated number as the box's floor.
    // Host RAM is a second, independent budget: a pack that fits the GPU can still
    // exhaust the box it runs on, and the failure presents as an unexplained upstream
    // death rather than as anything the matrix reports. It is read at exactly the
    // points the GPU pool is, so the two arithmetics rest on the same moments.
    let mut empty_pool_floor: Option<f64> = None;
    let mut host_floor: Option<f64> = None;
    let opening = clear_pool(&agent, &options.endpoint, gpu.as_ref());
    summary.host_total = opening.host.map(|host| round2(host.total_gb));
    note_floor(&mut empty_pool_floor, &mut host_floor, &opening);

    // The roster llama-swap is actually serving. A model the config declares but
    // llama-swap doesn't know is the loud half of a config mismatch: measuring it
    // would load nothing (or something else).
    let served = served_ids(&agent, &options.endpoint);

    // The worklist, resolved before the loop so progress can count against a total
    // the operator recognises. A proxy entry allocates no GPU and its footprint is
    // hand-set, so it is not work; nor is a model `--only` excludes.
    // Ids the config still names. An id in here is not an orphan, whatever the store
    // holds for it (see `Store::adoptable`).
    let live_ids: Vec<String> = records.iter().map(|record| record.id.clone()).collect();
    let worklist: Vec<&ModelRecord> = records
        .iter()
        .filter(|record| record.model_type != ModelType::TtsProxy)
        .filter(|record| {
            options.only.as_ref().is_none_or(|only| only.contains(&record.id))
        })
        .collect();
    let total = worklist.len();

    for (position, record) in worklist.iter().enumerate() {
        let index = position + 1;
        let report = |outcome: String| {
            progress(Progress::Done { index, total, id: &record.id, outcome });
        };

        // pre-check the weight file exists on the host (skip a doomed load)
        if let Some(container_path) = &record.primary_file {
            let host_path = policy.to_host(container_path);
            if !Path::new(&host_path).exists() {
                summary.skipped_missing.push(record.id.clone());
                report("skipped, weight file missing".to_string());
                continue;
            }
        }

        // cache hit? An *unconfirmed* entry is not one: it may be a mid-load plateau
        // rather than a footprint, so re-measuring it is the cheap half of Principle 6
        // (extra work beats wrong reuse), and it self-heals without the operator having
        // to know which entries to distrust. A store holding no confirmations therefore
        // re-measures in full, and one holding them re-measures only what is suspect.
        let stored = store.select(&record.id, &record.param_hash)?;
        if !options.force && stored.as_ref().is_some_and(Measurement::is_confirmed) {
            summary.cached.push(record.id.clone());
            report(format!(
                "cached, {:.2} GB",
                stored.as_ref().map_or(0.0, |entry| entry.d_total)
            ));
            continue;
        }

        // Nothing under this id, but the store may hold this exact memory command
        // under an id the config no longer has: a rename orphans a measurement file,
        // and a rename is not a new footprint. Adopting it files the number under the
        // new id, so this happens once and `build` sees it like any other hit.
        if !options.force && stored.is_none() {
            if let Some((from, measurement)) = store.adoptable(&record.param_hash, &live_ids)? {
                let d_total = measurement.d_total;
                store_measurement(store, record, measurement)?;
                summary.adopted.push(Adopted { id: record.id.clone(), from: from.clone(), d_total });
                report(format!("adopted {d_total:.2} GB from `{from}` (renamed)"));
                continue;
            }
        }

        // Weights on disk: a floor on the footprint of a fully offloaded model, and
        // the one cross-check that needs no GPU and no cooperation from the backend.
        let weights = weights_gb(record, policy);

        progress(Progress::Loading { index, total, id: &record.id });

        // This model's own baseline, read after the pool is verifiably empty. Per
        // model rather than once per sweep: a baseline that still counts a previous
        // model makes every delta taken against it SHORT, and a short delta is the
        // one error direction that OOMs (Principle 1). Reading it here also turns a
        // pool that failed to clear into a visible, per-model signal instead of a
        // silent bias spread across the rest of the sweep.
        let cleared = clear_pool(&agent, &options.endpoint, gpu.as_ref());
        let baseline_split = gpu.used_split_gb();
        note_floor(&mut empty_pool_floor, &mut host_floor, &cleared);
        let model_baseline = cleared.used;
        let host_at_baseline = cleared.host;
        let residents_before = cleared.residents;
        // Above the floor with nothing loaded means memory the proxy no longer
        // accounts for is still held: `/running` says empty, the device disagrees.
        let mut contention: Vec<String> = Vec::new();
        if let Some(floor) = empty_pool_floor {
            if residents_before.is_empty() && model_baseline > floor + CONTENTION_EPS {
                contention.push(format!(
                    "the pool held {model_baseline:.2} GB before this model loaded, {:.2} GB \
                     above the empty-pool floor seen this sweep, with llama-swap reporting \
                     nothing resident",
                    model_baseline - floor
                ));
            }
        }
        if !residents_before.is_empty() {
            contention.push(format!(
                "llama-swap still had {} resident when this model's baseline was read",
                residents_before.join(", ")
            ));
        }

        let fired = trigger(&record.id, record.model_type, options);
        let ready = wait_ready(&agent, &options.endpoint, &record.id, options.load_timeout);

        // Every path below yields at most one measurement and then falls through to a
        // single store-and-unload. `None` means record nothing at all, which is
        // reserved for a serving mismatch: that reading describes neither the config's
        // command nor the served one, so storing it is the one outcome that must not
        // happen (it would look like a present hash forever after).
        let recorded: Option<Measurement> = match ready {
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
                    pool_baseline: Some(round2(model_baseline)),
                    ..Default::default()
                })
            }
            // Is the server that just loaded running the command we hashed? On a
            // mismatch the reading belongs to neither command (Principle 2, and
            // Principle 1 downstream).
            Some(Ready { load_s: load, served_cmd }) => match check_serving(
                &agent,
                &options.endpoint,
                record,
                served_cmd.as_deref(),
            ) {
                Serving::Mismatch { detail } => {
                    summary.failed.push(Failure {
                        id: record.id.clone(),
                        reason: format!(
                            "llama-swap is serving a different command than the one being \
                             measured ({detail}) - no footprint recorded (reload llama-swap, \
                             or measure the config it actually loaded)"
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

                            // Who else was in the pool when the reading was taken?
                            // The two directions are not the same risk and must not
                            // be handled the same way.
                            let residents_after =
                                resident_ids(&agent, &options.endpoint, &record.id);
                            let departed: Vec<&String> = residents_before
                                .iter()
                                .filter(|id| !residents_after.contains(id))
                                .collect();
                            let arrived: Vec<&String> = residents_after
                                .iter()
                                .filter(|id| !residents_before.contains(id))
                                .collect();
                            if !arrived.is_empty() {
                                contention.push(format!(
                                    "llama-swap loaded {} during this model's measurement window",
                                    arrived
                                        .iter()
                                        .map(|id| id.as_str())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ));
                            }
                            // A model that was in the baseline and is gone by the
                            // sample was subtracted from the reading but is not in
                            // it: the delta is SHORT by that model's footprint. That
                            // is the one direction Principle 1 cannot tolerate, so
                            // the reading is refused outright rather than recorded
                            // with a caveat.
                            if !departed.is_empty() {
                                summary.failed.push(Failure {
                                    id: record.id.clone(),
                                    reason: format!(
                                        "{} left the pool between this model's baseline and its \
                                         reading, so the footprint would be SHORT by whatever \
                                         they held - no footprint recorded. Quiesce anything \
                                         that requests models and re-measure",
                                        departed
                                            .iter()
                                            .map(|id| id.as_str())
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    ),
                                });
                            }
                            let short_by_departure = !departed.is_empty();
                            if !contention.is_empty() {
                                summary.contended.push(Failure {
                                    id: record.id.clone(),
                                    reason: format!(
                                        "{} - a footprint is a SOLO footprint, so this one may \
                                         include memory that is not this model's (it can only \
                                         be too high, never too low). Quiesce anything that \
                                         requests models - health probes and pollers especially \
                                         - and re-measure with --force",
                                        contention.join("; ")
                                    ),
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
                            // Host delta over the same empty-pool point. A floor:
                            // the host-side prompt cache fills with use, not at
                            // load, so `build` adds the declared `-cram` cap on top.
                            let d_host = match (platform::host_memory(), host_at_baseline) {
                                (Some(now), Some(base)) => {
                                    Some(round2((now.used_gb - base.used_gb).max(0.0)))
                                }
                                _ => None,
                            };
                            let d_total = round2(used - model_baseline);
                            let peak = trigger_peak.max(settled.peak) - model_baseline;
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
                                d_host,
                                pool_baseline: Some(round2(model_baseline)),
                                contended: Some(!contention.is_empty()),
                                params: memory_cmd(&record.cmd),
                                measured_at: today.clone(),
                            };
                            // Same box, same flags, a different number. Until this
                            // was reported the fresh value simply overwrote the old
                            // one, so a configuration could be recorded twice, far
                            // apart, with nothing on disk saying so.
                            if let Some(previous) = stored.as_ref().filter(|entry| entry.is_ok()) {
                                if Remeasured::disagree(previous.d_total, d_total) {
                                    summary.changed.push(Remeasured {
                                        id: record.id.clone(),
                                        previous: previous.d_total,
                                        current: d_total,
                                        previous_measured_at: previous.measured_at.clone(),
                                    });
                                }
                            }
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
                            if short_by_departure {
                                None
                            } else {
                                summary.measured.push(record.id.clone());
                                Some(measurement)
                            }
                        }
                    }
                }
            },
        };

        let outcome = match &recorded {
            Some(measurement) if measurement.is_ok() => format!(
                "{:.2} GB in {:.1} s{}",
                measurement.d_total,
                measurement.load_s,
                if measurement.contended == Some(true) { " (contended)" } else { "" }
            ),
            Some(_) => "FAILED".to_string(),
            None => "nothing recorded".to_string(),
        };
        // `None` means there was deliberately nothing to store (a serving mismatch, a
        // departure mid-window), which the outcome text already says; only a reading
        // the store *refused* needs the extra sentence.
        let refused = match recorded {
            Some(measurement) => !store_measurement(store, record, measurement)?,
            None => false,
        };
        report(if refused {
            format!("{outcome}, not stored (the better reading already on record is kept)")
        } else {
            outcome
        });
        // The trigger has been awaited (or waited out) by now, so nothing of this
        // model's is still allocating; release the channel. The pool is cleared at
        // the top of the next model's window, which is also where its baseline is
        // read, so the unload and the reading it has to precede cannot drift apart.
        drop(fired);
    }
    // Leave the box as the sweep found it, and take one more empty-pool reading
    // while doing so: on a single-model sweep it is the only second opinion there
    // is about the floor.
    note_floor(
        &mut empty_pool_floor,
        &mut host_floor,
        &clear_pool(&agent, &options.endpoint, gpu.as_ref()),
    );
    summary.host_baseline = host_floor.map(round2);

    // The box baseline is what `build` treats as always resident, so it must come
    // from a pool that was actually empty. A sweep that never saw one (a box with a
    // client loading models throughout) keeps the stored value and says so, rather
    // than filing another model's footprint as the box's floor - which would inflate
    // the reserved floor for every future build.
    let previous_box = store.read_box()?;
    summary.baseline = match empty_pool_floor {
        Some(floor) => floor,
        None => {
            summary.no_empty_pool = true;
            previous_box.baseline
        }
    };
    // The freshly read floor is what gets written - the empty pool genuinely moves
    // when something else on the box takes or releases memory, and a stale baseline
    // is its own kind of wrong. But a move is reported, because the other thing that
    // produces one is a pool that only *looked* empty.
    if previous_box.baseline > 0.0
        && (summary.baseline - previous_box.baseline).abs() > CONTENTION_EPS
    {
        summary.baseline_was = Some(round2(previous_box.baseline));
    }

    store.write_box(&BoxMeta {
        baseline: round2(summary.baseline),
        detected_total: Some(round2(summary.detected_total)),
        host_total: summary.host_total,
        host_baseline: summary.host_baseline,
        date: Some(today),
        additivity_check: None,
        ..Default::default()
    })?;

    Ok(summary)
}

/// What a co-residency check found: the combination it loaded, what the plan
/// predicted it would occupy, and what the device actually reported.
#[derive(Debug, Clone, Serialize)]
pub struct Validation {
    /// The emitted set that was tested.
    pub set: String,
    /// The model ids loaded, in the order they were loaded.
    pub combo: Vec<String>,
    /// `baseline + Σ solo footprints`, from the plan.
    pub predicted: f64,
    /// Settled occupancy with all of them resident.
    pub measured: f64,
    /// `measured - predicted`. **Positive is the dangerous sign**: the models
    /// together hold more than their solo footprints predicted, so every declared
    /// combination is closer to the ceiling than the plan says.
    pub error: f64,
    /// The ceiling the plan was built against, so the error can be read against the
    /// slack that is supposed to absorb it.
    pub ceiling: f64,
    /// The safety margin, which is what an error this size eats into.
    pub margin: f64,
    /// Models the combination named that were not resident *and* finished allocating
    /// when the reading was taken. Non-empty means the number above is not a
    /// co-residency measurement at all, and nothing is recorded: a missing member
    /// makes the total look small, and small reads as "additive, plenty of headroom",
    /// which is the reassuring direction and the wrong one.
    pub absent: Vec<String>,
    /// Models resident that the combination did not name. Their memory is in the
    /// reading, so the error comes out too high, and a too-high error is the one that
    /// tells an operator to shrink their matrix. Nothing is recorded: a false alarm
    /// on the safety-critical output is worse than no reading.
    pub intruders: Vec<String>,
}

impl Validation {
    /// Is this a reading of what it claims to be a reading of?
    pub fn is_clean(&self) -> bool {
        self.absent.is_empty() && self.intruders.is_empty()
    }
}

/// Load one declared combination and compare what it actually occupies against what
/// the plan predicted.
///
/// This is the only step that tests the tool's central assumption. Everything else
/// measures models **alone** and then *sums*; if footprints are not additive on a box
/// (allocator fragmentation, a shared buffer, a driver that reserves per-process),
/// every declared combination is closer to the ceiling than the plan says, and the
/// error is in the direction that OOMs.
///
/// The tightest declared set is the one worth testing: it is the binding claim, and
/// anything smaller is implied by it. Loading it is not a risk the operator is not
/// already taking, because llama-swap will load exactly that combination on demand.
///
/// Requires the live config to declare the combination, since llama-swap evicts to
/// satisfy each request and will not hold models it has not been told may co-reside.
/// When it does not, the models are reported `absent` rather than a smaller number
/// being recorded as if it were the answer.
pub fn validate(
    plan: &crate::build::MatrixPlan,
    records: &[ModelRecord],
    store: &Store,
    options: &MeasureOptions,
    only: Option<&str>,
    progress: &dyn Fn(Progress),
) -> Result<Option<Validation>> {
    // Before the sensor and before the lock: rejecting a name the config does not
    // declare needs neither, and an operator who typoed a set should get told that
    // rather than told their box has no GPU.
    if let Some(name) = only {
        if !plan.sets.iter().any(|set| set.name == name) {
            bail!(
                "this config declares no set named `{name}`; `llama-matrix build` prints the \
                 names it emits"
            );
        }
    }

    let gpu = platform::detect().context("validate needs a GPU sensor")?;
    let _lock = LockGuard::acquire(store.dir())?;
    let agent = poll_agent();

    // The tightest set that names more than one loadable model, or the one named. A
    // single-model set proves nothing about additivity, and an `aux`-only set is
    // already inside every other set's prediction.
    //
    // The default is the tightest because it is the binding claim and everything
    // smaller is implied by it. `--set` exists because "implied by" is an argument
    // about footprints, and an operator with a specific worry (a diffusion server
    // that allocates transiently, a combination that failed once) should be able to
    // put that exact combination on the device instead of arguing about it.
    let by_id: HashMap<&str, &ModelRecord> =
        records.iter().map(|record| (record.id.as_str(), record)).collect();
    let Some((set, combo)) = plan
        .sets
        .iter()
        .filter(|set| only.is_none_or(|name| set.name == name))
        .filter_map(|set| {
            let combo: Vec<String> = crate::build::set_member_ids(&plan.sets, set)
                .into_iter()
                .filter(|id| {
                    by_id.get(id.as_str())
                        .is_some_and(|record| record.model_type != ModelType::TtsProxy)
                })
                .collect();
            (combo.len() > 1).then_some((set, combo))
        })
        .max_by(|left, right| {
            left.0.footprint.partial_cmp(&right.0.footprint).unwrap_or(std::cmp::Ordering::Equal)
        })
    else {
        return Ok(None);
    };

    let total = combo.len();
    clear_pool(&agent, &options.endpoint, gpu.as_ref());
    let mut fired = Vec::new();
    for (position, id) in combo.iter().enumerate() {
        progress(Progress::Loading { index: position + 1, total, id });
        let model_type = by_id[id.as_str()].model_type;
        let trigger = trigger(id, model_type, options);
        wait_ready(&agent, &options.endpoint, id, options.load_timeout);
        fired.push(trigger);
    }
    // Every trigger has to finish before the reading, for the same reason a solo
    // measurement waits for one: a lazily-allocating backend is `ready` long before
    // its weights are resident. Here it matters even more than there, because a model
    // that has not finished allocating makes the total look SMALL, and small reads as
    // "additive, plenty of headroom" - the reassuring direction, and the wrong one.
    let mut incomplete: Vec<String> = Vec::new();
    for (trigger, id) in fired.iter().zip(&combo) {
        if !matches!(
            await_allocation(trigger, gpu.as_ref(), options.trigger_timeout),
            Allocation::Confirmed { .. }
        ) {
            incomplete.push(id.clone());
        }
    }

    let resident = running(&agent, &options.endpoint);
    let mut absent: Vec<String> = combo
        .iter()
        .filter(|id| resident.get(id.as_str()).map(|entry| entry.state.as_str()) != Some("ready"))
        .cloned()
        .collect();
    for id in incomplete {
        if !absent.contains(&id) {
            absent.push(id);
        }
    }
    let intruders: Vec<String> = {
        let mut ids: Vec<String> = resident
            .keys()
            .filter(|id| !combo.iter().any(|member| member == *id))
            .cloned()
            .collect();
        ids.sort();
        ids
    };
    let settled = stabilize(
        gpu.as_ref(),
        STABILIZE_MAX_WAIT,
        STABILIZE_INTERVAL,
        STABILIZE_EPS,
        STABILIZE_HOLD,
    );
    drop(fired);

    let measured = round2(settled.used);
    let predicted = round2(set.footprint);
    // Leave the pool as `measure` leaves it: empty. This is a diagnostic, and pinning
    // the box at its ceiling afterwards is a side effect nobody asked for - on a
    // roster with `ttl: 0` those models would simply stay there.
    clear_pool(&agent, &options.endpoint, gpu.as_ref());
    let validation = Validation {
        set: set.name.clone(),
        combo,
        predicted,
        measured,
        error: round2(measured - predicted),
        ceiling: plan.ceiling,
        margin: plan.margin,
        absent,
        intruders,
    };

    // Record only a reading of exactly the combination it claims. Both ways of
    // failing that are worth refusing, in opposite directions: a missing member makes
    // the total too small (false headroom) and an extra one makes it too large (a
    // false "your footprints are not additive, shrink your matrix").
    if validation.is_clean() {
        let mut meta = store.read_box()?;
        meta.additivity_check = Some(crate::cache::AdditivityCheck {
            combo: validation.combo.clone(),
            predicted: validation.predicted,
            measured: validation.measured,
            error: validation.error,
        });
        store.write_box(&meta)?;
    }
    Ok(Some(validation))
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
                detail: "this config declares `-c 393216 -np 2` but the loaded server reports \
                         262144 per slot across 2 slot(s)"
                    .into(),
            }
        );
        // A one-token change is caught too (the reported reproduction).
        assert_eq!(
            compare_context((8191, None), (8192, 4)),
            Serving::Mismatch {
                detail: "this config declares `-c 8191` but the loaded server reports 8192 per \
                         slot across 4 slot(s)"
                    .into(),
            }
        );
        // An explicitly declared slot count that the server did not honour.
        assert_eq!(
            compare_context((524288, Some(2)), (131072, 4)),
            Serving::Mismatch {
                detail: "this config declares `-c 524288 -np 2` but the loaded server reports \
                         131072 per slot across 4 slot(s)"
                    .into(),
            }
        );
        // Integer division of an odd total across slots is not a mismatch.
        assert_eq!(compare_context((8191, Some(2)), (4095, 2)), Serving::Confirmed);
    }

    /// llama-swap v251 reports the command it launched in `GET /running`. That is
    /// the served command itself, so the cross-check no longer has to infer it from
    /// what the loaded server says it did.
    #[test]
    fn running_carries_the_launched_command() {
        let body = r#"{"running":[{"model":"embed","state":"ready",
            "cmd":"/app/llama-server -m /m.gguf --port 9050 -c 8192\n"}]}"#;
        let entries = parse_running(body);
        assert_eq!(entries["embed"].state, "ready");
        assert_eq!(
            entries["embed"].cmd.as_deref(),
            Some("/app/llama-server -m /m.gguf --port 9050 -c 8192\n")
        );

        // An older llama-swap reports no cmd; an empty one is the same as none, so a
        // blank string can never be compared against and called a mismatch.
        let older = parse_running(r#"{"running":[{"model":"embed","state":"ready"}]}"#);
        assert_eq!(older["embed"].cmd, None);
        let blank = parse_running(r#"{"running":[{"model":"embed","state":"ready","cmd":"  "}]}"#);
        assert_eq!(blank["embed"].cmd, None);
    }

    /// The comparison runs on the *memory* command, so the tokens llama-swap owns
    /// (the port) and the ones known to be footprint-neutral never raise a false
    /// mismatch, while any flag the param-hash is built from does.
    #[test]
    fn a_served_command_is_compared_on_its_memory_tokens() {
        let declared = "/app/llama-server -m /m.gguf --host 127.0.0.1 --port 9050 -ngl 99 \
                        -c 8192 -fa on --jinja";
        // llama-swap launched the same thing on another port, with a folded newline.
        let served = "/app/llama-server -m /m.gguf --host 127.0.0.1 --port 9111 -ngl 99\n \
                      -c 8192 -fa on --jinja";
        assert_eq!(compare_commands(declared, served), Serving::Confirmed);

        // A real memory flag moved: the footprint would be filed under a hash that
        // never ran, which is the whole failure being guarded.
        let restretched = declared.replace("-c 8192", "-c 262144");
        let Serving::Mismatch { detail } = compare_commands(declared, &restretched) else {
            panic!("a changed -c must be a mismatch");
        };
        assert!(detail.contains("`-c 8192`"), "{detail}");
        assert!(detail.contains("`-c 262144`"), "{detail}");

        // An unexpanded runtime placeholder is not evidence either way: llama-swap
        // substitutes it at launch and the config file never can.
        assert_eq!(
            compare_commands("/app/llama-server -m /m.gguf -c ${CTX}", "/app/llama-server -m /m.gguf -c 8192"),
            Serving::Unconfirmed
        );
    }



    /// The `/props` fallback still runs when llama-swap reports no command, and an
    /// image backend (no `/props` at all) is then unconfirmable, as before.
    #[test]
    fn without_a_served_command_the_props_path_still_decides() {
        // Confirmed straight from the command, whatever the backend: this is the
        // path that gives an image or STT server a verdict for the first time.
        let sd = ModelRecord::from_expanded(
            "z-image",
            "/opt/sdcpp/bin/sd-server --diffusion-model /sd/u.gguf --steps 8",
        );
        assert_eq!(
            check_serving(&poll_agent(), "http://127.0.0.1:1", &sd, Some(&sd.cmd)),
            Serving::Confirmed
        );
        // With no served command and no reachable `/props`, it stays unconfirmed
        // rather than being passed off as verified.
        assert_eq!(
            check_serving(&poll_agent(), "http://127.0.0.1:1", &sd, None),
            Serving::Unconfirmed
        );
    }

    /// The tolerance separates allocator jitter from a reading that changed, at
    /// both ends of the size range: 0.25 GB is noise on an 80 GB model and a fifth
    /// of a small one, so neither a flat GB nor a flat percentage works alone.
    #[test]
    fn a_remeasure_disagrees_only_outside_the_tolerance() {
        // Small model: the absolute floor decides.
        assert!(!Remeasured::disagree(1.80, 1.82));
        assert!(Remeasured::disagree(1.80, 2.20));
        // Large model: 2% of 80 GB is 1.6 GB, so ordinary variation is not a story.
        assert!(!Remeasured::disagree(80.00, 81.00));
        assert!(Remeasured::disagree(80.00, 82.00));
        // The case that motivated it: a 32.16 GB entry that came back 6.52 GB high,
        // exactly another model's footprint.
        assert!(Remeasured::disagree(32.16, 38.68));
    }

    /// Only the model under test may be resident; anything else llama-swap reports
    /// is named, sorted, so the message reads the same on every sweep.
    #[test]
    fn foreign_residents_names_everything_but_the_model_under_test() {
        let entries = parse_running(
            r#"{"running":[{"model":"under-test","state":"ready"},
                           {"model":"rag-embed","state":"ready"},
                           {"model":"aux-tts","state":"starting"}]}"#,
        );
        let mut others: Vec<String> =
            entries.into_keys().filter(|id| id != "under-test").collect();
        others.sort();
        assert_eq!(others, vec!["aux-tts".to_string(), "rag-embed".to_string()]);
    }

    /// The store keeps the better evidence, not the newer. Both refusals share the
    /// reason: a measurement is GPU time already paid for, and nothing else deletes
    /// one.
    #[test]
    fn a_worse_reading_does_not_displace_a_better_one() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path().join("measurements"));
        let record = ModelRecord::from_expanded("m", "/app/llama-server -m /m.gguf -c 4096");
        let stored = |hash: &str| {
            store.read_model("m").unwrap().unwrap().measurements.get(hash).cloned().unwrap()
        };

        let clean = Measurement {
            status: "ok".into(),
            d_total: 16.12,
            allocation_confirmed: Some(true),
            contended: Some(false),
            ..Default::default()
        };
        assert!(store_measurement(&store, &record, clean).unwrap());
        assert_eq!(stored(&record.param_hash).d_total, 16.12);

        // Contended: something else was in the pool, so this number includes memory
        // that is not the model's. Exactly the 6.52 GB an embedding model added to a
        // 16.12 GB image server on the box that raised this.
        let contended = Measurement {
            status: "ok".into(),
            d_total: 22.64,
            allocation_confirmed: Some(true),
            contended: Some(true),
            ..Default::default()
        };
        assert!(!store_measurement(&store, &record, contended.clone()).unwrap());
        assert_eq!(stored(&record.param_hash).d_total, 16.12);

        // A failed load is no evidence against a stored footprint either.
        let failed = Measurement { status: "FAILED".into(), ..Default::default() };
        assert!(!store_measurement(&store, &record, failed).unwrap());
        assert_eq!(stored(&record.param_hash).d_total, 16.12);

        // But an entry written before the check existed claims nothing about
        // contention, so it is replaced as usual rather than treated as clean.
        let unchecked = Measurement {
            status: "ok".into(),
            d_total: 16.12,
            allocation_confirmed: Some(true),
            contended: None,
            ..Default::default()
        };
        let other = ModelRecord::from_expanded("n", "/app/llama-server -m /n.gguf -c 4096");
        assert!(store_measurement(&store, &other, unchecked).unwrap());
        assert!(store_measurement(&store, &other, contended).unwrap());
        assert_eq!(
            store.read_model("n").unwrap().unwrap().measurements[&other.param_hash].d_total,
            22.64
        );
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
