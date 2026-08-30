//! The per-model, per-box measurement store: a `measurements/` directory holding
//! one JSON file per model (footprints stacked, keyed by param-hash) plus a
//! reserved `_box.json` for box-level values (baseline, detected total,
//! additivity check). Kept beside `llama-matrix.toml`, never beside the weights
//! (a footprint is a property of `(model, box)`). Retained indefinitely; pruning
//! is explicit only. See SPEC.md §2.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::model::ModelType;

/// The reserved box-level file name inside `measurements/`.
pub const BOX_FILE: &str = "_box.json";

/// The running llama-matrix version, stamped into the store (`BoxMeta::written_by`)
/// on every write so a later build knows which on-disk schema wrote the store and
/// can select a migration or flag a newer-than-me store (house guidelines D5).
pub const WRITER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Numeric `(major, minor, patch)` of a version string, ignoring any `-pre`/`+build`
/// suffix. Unparseable parts become 0, so a version we can't read never spuriously
/// compares as newer. (Ported from temper's `manifest::version_triple`.)
fn version_triple(version: &str) -> (u64, u64, u64) {
    let core = version.split(['-', '+']).next().unwrap_or(version);
    let mut parts = core.split('.').map(|part| part.parse::<u64>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

/// Is `candidate` a strictly newer version than `running`? Numeric compare, not
/// lexical, so `1.10.0` > `1.9.0`.
pub fn version_is_newer(candidate: &str, running: &str) -> bool {
    version_triple(candidate) > version_triple(running)
}

/// Box-level values that have no per-model home.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BoxMeta {
    /// Empty-pool occupancy in GB.
    #[serde(default)]
    pub baseline: f64,
    /// Physical pool total in GB at sweep time (build may override via budget).
    #[serde(default)]
    pub detected_total: Option<f64>,
    /// Host RAM total in GB, as distinct from the GPU pool even on a box where both
    /// are carved out of the same chips. `None` where the box cannot report it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_total: Option<f64>,
    /// Host RAM held with no model loaded, in GB: the OS plus everything else the
    /// box runs. The floor a pack's host cost sits on top of, and measured for the
    /// same reason `baseline` is, rather than assumed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_baseline: Option<f64>,
    /// Last sweep date (YYYY-MM-DD).
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub additivity_check: Option<AdditivityCheck>,
    /// The llama-matrix version that last wrote this store. Stamped by `write_box`
    /// (never hand-set), so a later build knows the on-disk schema and can flag a
    /// store written by a newer build than itself (house guidelines D5).
    #[serde(default)]
    pub written_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdditivityCheck {
    pub combo: Vec<String>,
    pub predicted: f64,
    pub measured: f64,
    pub error: f64,
}

/// One measured footprint, keyed in a model's file by its param-hash.
///
/// The per-pool fields (`d_vram`/`d_gtt`/`abs_vram`/`abs_gtt`) are `Option`: a
/// backend with one pool (or a unified one, where the split has no meaning) omits
/// them rather than writing `0`, so a recorded zero always means *measured zero*
/// and never *not measured*. See [`ModelStore::normalize_pool_split`] for the
/// entries written before that distinction existed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Measurement {
    /// "ok" | "FAILED".
    pub status: String,
    #[serde(default)]
    pub d_total: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d_vram: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d_gtt: Option<f64>,
    #[serde(default)]
    pub abs_total: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abs_vram: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abs_gtt: Option<f64>,
    #[serde(default)]
    pub load_s: f64,
    /// Did the load-trigger complete, proving the model finished allocating?
    ///
    /// `Some(true)`: the trigger returned and occupancy then settled, so the number
    /// describes a finished allocation. `Some(false)`: it did not, so the reading may
    /// be a mid-load plateau (a lazily-allocating backend such as sd-server is
    /// `ready` long before its weights are resident). `None`: the writer recorded no
    /// confirmation, which is exactly as much evidence as `Some(false)` and is treated
    /// the same. See SPEC §7.2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocation_confirmed: Option<bool>,
    /// Did `/props` confirm llama-swap was serving the command we hashed (SPEC §7.1)?
    /// `Some(false)`/`None` mean *unconfirmable* (no `/props` on that backend, or
    /// nothing comparable in the command), not wrong.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving_verified: Option<bool>,
    /// Highest occupancy seen while the model was allocating, as a delta over
    /// baseline in GB (so it is directly comparable to `d_total`). A diffusion step
    /// can allocate transiently above what it leaves resident; recorded for insight
    /// and for future peak budgeting. `build` plans against `d_total`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_total: Option<f64>,
    /// Total size in GB of the weight files the command names, when they were
    /// readable at measure time. A fully offloaded model cannot hold much less than
    /// its weights, which makes this a cheap, backend-agnostic sanity floor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights_gb: Option<f64>,
    /// The empty-pool occupancy this delta was taken against, in GB.
    ///
    /// Read immediately before this model loaded, so `abs_total - pool_baseline =
    /// d_total` is checkable after the fact and a baseline that still counted a
    /// previous model cannot silently shorten the delta.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pool_baseline: Option<f64>,
    /// Host RAM this model added when it loaded, in GB.
    ///
    /// A **floor**, not the whole story, and the difference matters. It is what the
    /// process had dirtied by the time it was serving: weights it copies to host,
    /// its own allocations, the runtime. It cannot include the host-side prompt
    /// cache (`-cram`, 8 GiB per llama-server by default), because that fills as
    /// prompts are processed and the load-trigger processes one tiny prompt. `build`
    /// adds the declared cache cap on top, reading it from the live command rather
    /// than from here, which is why a `-cram` change needs no re-measure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d_host: Option<f64>,
    /// Was anything other than this model in the pool during its measurement window?
    ///
    /// A footprint is a *solo* footprint, and nothing stops a client (a health probe,
    /// a RAG poller, a scheduled job) from asking llama-swap for another model
    /// mid-sweep. `Some(true)` means the sweep saw evidence of exactly that, so the
    /// number may include memory that is not this model's. `None` means the writer
    /// ran no such check.
    ///
    /// It does not gate anything, because the risk direction is favourable:
    /// contamination adds occupancy, so a contended reading is *over*-measured, and
    /// over-measuring wastes packs but never OOMs (Principle 1). It is reported so
    /// the operator can quiesce the box and `--force` the entries back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contended: Option<bool>,
    /// The hashed (memory) command, human-readable.
    #[serde(default)]
    pub params: String,
    #[serde(default)]
    pub measured_at: String,
}

/// A footprint below this fraction of its weights on disk is implausible for a
/// fully offloaded model. Not 1.0: not every component is resident at once, and two
/// verified image measurements sat at 0.97-0.98 of their file total, so the floor
/// has to clear those while still catching a half-measured load.
pub const WEIGHT_FLOOR_RATIO: f64 = 0.90;

impl Measurement {
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }

    /// Is this footprint trustworthy for the knapsack: measured, *and* the
    /// allocation it describes is known to have finished?
    ///
    /// The two halves are separate on purpose. `is_ok` says a number was recorded;
    /// this says the number is complete. An entry that is `ok` but unconfirmed may be
    /// a mid-load plateau, which under-counts the matrix - the one direction
    /// Principle 1 cannot tolerate - so `build` treats it as policy
    /// (`on_unconfirmed`) rather than as data.
    pub fn is_confirmed(&self) -> bool {
        self.is_ok() && self.allocation_confirmed == Some(true)
    }

    /// `d_total` as a fraction of the weights on disk, when both are known.
    pub fn weight_ratio(&self) -> Option<f64> {
        let weights = self.weights_gb?;
        (weights > 0.0).then_some(self.d_total / weights)
    }

    /// Is the footprint implausibly small for the weights the command loads?
    ///
    /// A warning signal, never a verdict: partial offload (`-ngl` below all layers,
    /// `-ot`, `--cpu-moe`) is a legitimate reason for a model to sit lower.
    pub fn below_weight_floor(&self) -> bool {
        self.weight_ratio().is_some_and(|ratio| ratio < WEIGHT_FLOOR_RATIO)
    }
}

/// One model's file: its type, weight file, and stacked measurements by param-hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStore {
    #[serde(rename = "type", default)]
    pub model_type: String,
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub measurements: IndexMap<String, Measurement>,
}

impl ModelStore {
    /// Is this a hand-set proxy entry (a fronted service with a placeholder `cmd`,
    /// typed `tts-proxy`)? Such models are excluded from the measure worklist, so
    /// their footprint is written by hand under a key of the operator's choosing
    /// rather than a param-hash: the one case [`Store::select`] may resolve without
    /// a hash match. See SPEC.md §2, §6.
    fn is_hand_set_proxy(&self) -> bool {
        self.model_type == ModelType::TtsProxy.as_str()
    }

    /// Rewrite a `0`/`0` pool split as "not measured" (`None`).
    ///
    /// Before the per-pool fields became `Option`, every entry this tool wrote
    /// carried a literal `0.0` in all four, because the sensor has only ever
    /// reported summed occupancy. A model that occupies a nonzero total cannot in
    /// fact hold zero in *both* pools, so that pattern is unambiguously an
    /// unpopulated field rather than a reading, and is safe to clear on read. This
    /// makes a persisted zero mean measured-zero for good, without needing a
    /// per-entry schema version (the store's `written_by` stamp is box-level).
    fn normalize_pool_split(&mut self) {
        for measurement in self.measurements.values_mut() {
            if measurement.d_total > 0.0
                && measurement.d_vram == Some(0.0)
                && measurement.d_gtt == Some(0.0)
            {
                measurement.d_vram = None;
                measurement.d_gtt = None;
            }
            if measurement.abs_total > 0.0
                && measurement.abs_vram == Some(0.0)
                && measurement.abs_gtt == Some(0.0)
            {
                measurement.abs_vram = None;
                measurement.abs_gtt = None;
            }
        }
    }
}

/// The legacy single-file schema (the reference tooling's `measurements.json`):
/// box-level values at the top level plus a `models` map whose values already
/// match [`ModelStore`]. Read only for migration.
#[derive(Debug, Deserialize)]
struct LegacyFile {
    #[serde(default)]
    baseline: f64,
    /// The legacy `budget` becomes our detected-total.
    #[serde(default)]
    budget: Option<f64>,
    #[serde(default)]
    date: Option<String>,
    #[serde(default)]
    models: IndexMap<String, ModelStore>,
    #[serde(default)]
    additivity_check: Option<AdditivityCheck>,
}

/// A handle to a `measurements/` directory.
pub struct Store {
    dir: PathBuf,
}

impl Store {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Store { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn model_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.json", file_safe(id)))
    }

    fn box_path(&self) -> PathBuf {
        self.dir.join(BOX_FILE)
    }

    /// Read the box-level file (defaults if it doesn't exist yet).
    pub fn read_box(&self) -> Result<BoxMeta> {
        let path = self.box_path();
        if !path.exists() {
            return Ok(BoxMeta::default());
        }
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&json).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn write_box(&self, meta: &BoxMeta) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let mut meta = meta.clone();
        // Stamp the writing tool's version, monotonically: never stamp *down* a
        // store a newer llama-matrix wrote (it may use a schema this build
        // predates). Mirrors temper's `stamp_version`. See the guidelines' D5.
        meta.written_by = Some(match self.read_box().ok().and_then(|prev| prev.written_by) {
            Some(prev) if version_is_newer(&prev, WRITER_VERSION) => prev,
            _ => WRITER_VERSION.to_string(),
        });
        let json = serde_json::to_string_pretty(&meta)?;
        std::fs::write(self.box_path(), json + "\n")?;
        Ok(())
    }

    pub fn read_model(&self, id: &str) -> Result<Option<ModelStore>> {
        let path = self.model_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut store: ModelStore =
            serde_json::from_str(&json).with_context(|| format!("parsing {}", path.display()))?;
        store.normalize_pool_split();
        Ok(Some(store))
    }

    pub fn write_model(&self, id: &str, store: &ModelStore) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let json = serde_json::to_string_pretty(store)?;
        std::fs::write(self.model_path(id), json + "\n")?;
        Ok(())
    }

    /// Delete a model's file from the store (explicit prune only — never automatic).
    pub fn remove_model(&self, id: &str) -> Result<()> {
        let path = self.model_path(id);
        if path.exists() {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
        Ok(())
    }

    /// Migrate a legacy single-file `measurements.json` (the reference tooling's
    /// one-blob format) into this per-model store. A no-op if the store already
    /// has models or the legacy file is absent. Returns the number of models
    /// migrated.
    pub fn migrate_legacy(&self, legacy_path: &Path) -> Result<usize> {
        if !self.list_ids().is_empty() || !legacy_path.exists() {
            return Ok(0);
        }
        let text = std::fs::read_to_string(legacy_path)
            .with_context(|| format!("reading {}", legacy_path.display()))?;
        let legacy: LegacyFile = serde_json::from_str(&text)
            .with_context(|| format!("parsing legacy {}", legacy_path.display()))?;

        self.write_box(&BoxMeta {
            baseline: legacy.baseline,
            detected_total: legacy.budget,
            date: legacy.date,
            additivity_check: legacy.additivity_check,
            ..Default::default()
        })?;
        for (id, model_store) in &legacy.models {
            // The reference tooling read both pools, so migrated entries usually
            // carry a real split; the ones it left at 0/0 are cleared, same as on read.
            let mut model_store = model_store.clone();
            model_store.normalize_pool_split();
            self.write_model(id, &model_store)?;
        }
        Ok(legacy.models.len())
    }

    /// The `ok` measurement for `(id, param_hash)`, or `None`.
    ///
    /// **A hash miss is a miss.** The param-hash covers every flag known to affect
    /// the footprint, so an entry stored under a different hash was measured under
    /// different memory flags and is not this model's footprint. Returning it would
    /// be a wrong cache hit: `measure` would report `cached` and skip the
    /// re-measure, and `build` would plan the knapsack against a stale number, which
    /// is exactly the under-count that Principle 1 (never OOM) and Principle 6 (a
    /// changed flag costs at most a harmless re-measure, never a wrong reuse) exist
    /// to prevent. A `FAILED` entry at the matching hash is a miss for the same
    /// reason: it carries no footprint.
    ///
    /// The single documented exception is a **hand-set proxy entry** (see
    /// [`ModelStore::is_hand_set_proxy`]): excluded from the measure worklist, so its
    /// footprint is keyed by hand and can never match a config-derived hash. That is
    /// resolvable without guessing because such an entry is not a measurement of
    /// flags that could have changed, and it is still required to be the model's
    /// *only* `ok` entry, so there is nothing to choose between.
    ///
    /// **Confirmation is the caller's decision.** This returns any `ok` entry at the
    /// hash, including one whose allocation was never confirmed
    /// ([`Measurement::is_confirmed`]), because the two consumers want opposite
    /// things from it: `measure` re-measures an unconfirmed entry (so a suspect
    /// number self-heals), while `build` applies `on_unconfirmed`. Folding the check
    /// in here would make an unconfirmed entry invisible and silently re-measured
    /// forever with no way to report it.
    pub fn select(&self, id: &str, param_hash: &str) -> Result<Option<Measurement>> {
        let Some(store) = self.read_model(id)? else {
            return Ok(None);
        };
        if let Some(measurement) = store.measurements.get(param_hash) {
            return Ok(measurement.is_ok().then(|| measurement.clone()));
        }
        if store.is_hand_set_proxy() {
            let hand_set: Vec<&Measurement> =
                store.measurements.values().filter(|entry| entry.is_ok()).collect();
            if let [only] = hand_set[..] {
                return Ok(Some(only.clone()));
            }
        }
        Ok(None)
    }

    /// Model ids present in the store (excluding the box file).
    pub fn list_ids(&self) -> Vec<String> {
        let mut ids = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".json") && name != BOX_FILE {
                    ids.push(name.trim_end_matches(".json").to_string());
                }
            }
        }
        ids
    }
}

/// Make an id safe as a filename (only path separators are a problem).
fn file_safe(id: &str) -> String {
    id.chars()
        .map(|character| if character == '/' || character == '\\' { '_' } else { character })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_box_and_model() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path().join("measurements"));

        store
            .write_box(&BoxMeta {
                baseline: 0.16,
                detected_total: Some(111.5),
                date: Some("2026-01-01".into()),
                additivity_check: None,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(store.read_box().unwrap().detected_total, Some(111.5));
        // write_box stamps the writing tool's version (D5); read it back.
        assert_eq!(store.read_box().unwrap().written_by.as_deref(), Some(WRITER_VERSION));

        let mut measurements = IndexMap::new();
        measurements.insert(
            "abc123".to_string(),
            Measurement {
                status: "ok".into(),
                d_total: 49.0,
                load_s: 42.0,
                ..Default::default()
            },
        );
        store
            .write_model(
                "coder-70b",
                &ModelStore {
                    model_type: "llm".into(),
                    file: Some("/m.gguf".into()),
                    measurements,
                },
            )
            .unwrap();

        // exact hash hit
        assert_eq!(store.select("coder-70b", "abc123").unwrap().unwrap().d_total, 49.0);
        assert_eq!(store.list_ids(), vec!["coder-70b".to_string()]);
    }

    /// Build a store holding one model with `entries` of `(hash, status, d_total)`.
    fn store_with(
        dir: &std::path::Path,
        model_type: &str,
        entries: &[(&str, &str, f64)],
    ) -> Store {
        let store = Store::new(dir.join("measurements"));
        let mut measurements = IndexMap::new();
        for (hash, status, d_total) in entries {
            measurements.insert(
                (*hash).to_string(),
                Measurement {
                    status: (*status).to_string(),
                    d_total: *d_total,
                    ..Default::default()
                },
            );
        }
        store
            .write_model(
                "m",
                &ModelStore {
                    model_type: model_type.to_string(),
                    file: Some("/m.gguf".into()),
                    measurements,
                },
            )
            .unwrap();
        store
    }

    /// A hash miss must never resolve to a footprint measured under other flags.
    /// The `-c`/`-np`/quant change that produces a new hash is exactly when a stale
    /// reuse would under-count the matrix and OOM the box (Principles 1 and 6).
    #[test]
    fn a_hash_miss_is_a_miss() {
        let dir = tempfile::tempdir().unwrap();

        // Exactly one ok entry: the case that used to fall back to it regardless of
        // the requested hash, so the first memory-flag change on any model was
        // silently skipped. This is the regression.
        let one = store_with(dir.path(), "llm", &[("hash-c8192", "ok", 7.0)]);
        assert!(one.select("m", "hash-c8191").unwrap().is_none());
        assert_eq!(one.select("m", "hash-c8192").unwrap().unwrap().d_total, 7.0);

        // Two or more ok entries: also a miss, and each stored hash still hits.
        let two = store_with(
            dir.path(),
            "llm",
            &[("hash-c262144", "ok", 24.61), ("hash-c524288", "ok", 28.26)],
        );
        assert!(two.select("m", "hash-c393216").unwrap().is_none());
        assert_eq!(two.select("m", "hash-c262144").unwrap().unwrap().d_total, 24.61);
        assert_eq!(two.select("m", "hash-c524288").unwrap().unwrap().d_total, 28.26);

        // A FAILED entry carries no footprint: a miss at its own hash, and no reason
        // to reach for another entry either.
        let failed = store_with(
            dir.path(),
            "llm",
            &[("hash-bad", "FAILED", 0.0), ("hash-good", "ok", 12.0)],
        );
        assert!(failed.select("m", "hash-bad").unwrap().is_none());
    }

    /// The one documented exception: a `tts-proxy` entry is hand-keyed (it is never
    /// measured, so it has no config-derived hash to match) and must still resolve,
    /// or every pack containing a fronted service would silently lose it.
    #[test]
    fn a_hand_set_proxy_entry_resolves_without_a_hash_match() {
        let dir = tempfile::tempdir().unwrap();

        let proxy = store_with(dir.path(), "tts-proxy", &[("manual-kokoro", "ok", 0.1)]);
        assert_eq!(proxy.select("m", "any-config-hash").unwrap().unwrap().d_total, 0.1);

        // The carve-out is typed, not a general "sole entry" rule: the same shape
        // typed `llm` stays a miss.
        let llm = store_with(dir.path(), "llm", &[("manual-kokoro", "ok", 0.1)]);
        assert!(llm.select("m", "any-config-hash").unwrap().is_none());

        // And it never *chooses*: two hand-set ok entries is ambiguous, so a miss.
        let ambiguous =
            store_with(dir.path(), "tts-proxy", &[("a", "ok", 0.1), ("b", "ok", 0.2)]);
        assert!(ambiguous.select("m", "any-config-hash").unwrap().is_none());
    }

    /// A `0`/`0` split against a nonzero total is an unpopulated field, not a
    /// reading, and is cleared on read so a stored zero always means measured zero.
    /// Pins the schema half of the per-pool fix; `measure` writes the split itself.
    #[test]
    fn an_unpopulated_pool_split_reads_as_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path().join("measurements"));
        std::fs::create_dir_all(store.dir()).unwrap();
        std::fs::write(
            store.dir().join("m.json"),
            r#"{"type":"llm","file":"/m.gguf","measurements":{
                 "zeroed": {"status":"ok","d_total":28.26,"d_vram":0.0,"d_gtt":0.0,
                            "abs_total":28.43,"abs_vram":0.0,"abs_gtt":0.0},
                 "real":   {"status":"ok","d_total":24.61,"d_vram":24.07,"d_gtt":0.54,
                            "abs_total":24.77,"abs_vram":24.23,"abs_gtt":0.54}}}"#,
        )
        .unwrap();

        let zeroed = store.select("m", "zeroed").unwrap().unwrap();
        assert_eq!(zeroed.d_vram, None, "0/0 against a 28.26 GB total is not a reading");
        assert_eq!(zeroed.d_gtt, None);
        assert_eq!(zeroed.abs_vram, None);
        assert_eq!(zeroed.d_total, 28.26, "the total is untouched");

        // A genuine split survives verbatim.
        let real = store.select("m", "real").unwrap().unwrap();
        assert_eq!(real.d_vram, Some(24.07));
        assert_eq!(real.d_gtt, Some(0.54));

        // An omitted split is absent from the JSON, never written back as 0.
        let rewritten = store.read_model("m").unwrap().unwrap();
        store.write_model("m", &rewritten).unwrap();
        let json = std::fs::read_to_string(store.dir().join("m.json")).unwrap();
        assert!(!json.contains("\"d_vram\": 0.0"), "an unknown split must not persist as 0");
        assert!(json.contains("\"d_vram\": 24.07"), "a real split must persist");
    }

    /// An entry carrying no `allocation_confirmed` carries no evidence that its
    /// footprint is complete, so it must not read as confirmed: every footprint in a
    /// store written without the field is unconfirmed until re-measured.
    #[test]
    fn a_legacy_entry_is_not_confirmed() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path().join("measurements"));
        std::fs::create_dir_all(store.dir()).unwrap();
        std::fs::write(
            store.dir().join("m.json"),
            r#"{"type":"image","file":"/sd/u.gguf","measurements":{
                 "legacy": {"status":"ok","d_total":8.87,"abs_total":9.03,"load_s":12.0}}}"#,
        )
        .unwrap();

        let legacy = store.select("m", "legacy").unwrap().unwrap();
        assert!(legacy.is_ok(), "it is still a recorded measurement");
        assert!(!legacy.is_confirmed(), "…but nothing confirms the allocation finished");
        assert_eq!(legacy.allocation_confirmed, None);
        // No weights recorded, so no floor check is possible (never a false alarm).
        assert_eq!(legacy.weight_ratio(), None);
        assert!(!legacy.below_weight_floor());
    }

    /// The new evidence fields round-trip, and are omitted rather than written as
    /// `false`/`0` when unknown (same rule as the per-pool split).
    #[test]
    fn confirmation_fields_round_trip_and_omit_when_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path().join("measurements"));
        let mut measurements = IndexMap::new();
        measurements.insert(
            "confirmed".to_string(),
            Measurement {
                status: "ok".into(),
                d_total: 16.10,
                abs_total: 16.26,
                load_s: 14.0,
                allocation_confirmed: Some(true),
                serving_verified: Some(false),
                peak_total: Some(17.40),
                weights_gb: Some(16.55),
                ..Default::default()
            },
        );
        measurements.insert(
            "unknown".to_string(),
            Measurement { status: "ok".into(), d_total: 8.87, ..Default::default() },
        );
        store
            .write_model(
                "img",
                &ModelStore {
                    model_type: "image".into(),
                    file: Some("/sd/u.gguf".into()),
                    measurements,
                },
            )
            .unwrap();

        let confirmed = store.select("img", "confirmed").unwrap().unwrap();
        assert!(confirmed.is_confirmed());
        assert_eq!(confirmed.peak_total, Some(17.40));
        // 16.10 / 16.55 = 0.97 → above the floor.
        assert!(!confirmed.below_weight_floor());

        let json = std::fs::read_to_string(store.dir().join("img.json")).unwrap();
        assert!(json.contains("\"allocation_confirmed\": true"));
        assert!(json.contains("\"serving_verified\": false"), "a real false must persist");
        assert_eq!(
            json.matches("allocation_confirmed").count(),
            1,
            "the unknown entry must omit the field, not write false"
        );
    }

    /// The floor catches the reported failure: 8.87 GB recorded for a model whose
    /// weight files total 16.55 GB is 0.54 of its own weights.
    #[test]
    fn the_weights_floor_flags_a_half_measured_load() {
        let under = Measurement {
            status: "ok".into(),
            d_total: 8.87,
            weights_gb: Some(16.55),
            ..Default::default()
        };
        assert!(under.below_weight_floor());
        assert!((under.weight_ratio().unwrap() - 0.5359).abs() < 1e-3);

        // The two legitimately-just-under entries must stay clear of it.
        for (footprint, weights) in [(21.04, 21.50), (6.30, 6.46)] {
            let fine = Measurement {
                status: "ok".into(),
                d_total: footprint,
                weights_gb: Some(weights),
                ..Default::default()
            };
            assert!(
                !fine.below_weight_floor(),
                "{footprint} of {weights} GB is normal partial residency, not a bad measure"
            );
        }
    }

    #[test]
    fn version_stamp_is_monotonic_and_numeric() {
        assert!(version_is_newer("1.10.0", "1.9.0")); // numeric compare, not lexical
        assert!(!version_is_newer("1.0.0", "1.0.0"));
        assert!(!version_is_newer("bogus", "1.0.0")); // unparseable is never "newer"

        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path().join("measurements"));

        // A fresh write stamps the running version.
        store.write_box(&BoxMeta::default()).unwrap();
        assert_eq!(store.read_box().unwrap().written_by.as_deref(), Some(WRITER_VERSION));

        // Seed a stamp from a hypothetical newer build, then a normal write must not
        // stamp it back down (monotonic), while still updating the other fields.
        std::fs::write(
            store.dir().join(BOX_FILE),
            serde_json::to_string(&BoxMeta {
                written_by: Some("999.0.0".into()),
                ..Default::default()
            })
            .unwrap(),
        )
        .unwrap();
        store.write_box(&BoxMeta { baseline: 1.5, ..Default::default() }).unwrap();
        let after = store.read_box().unwrap();
        assert_eq!(after.written_by.as_deref(), Some("999.0.0"));
        assert_eq!(after.baseline, 1.5);
    }

    #[test]
    fn migrates_a_legacy_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("measurements.json");
        std::fs::write(
            &legacy,
            r#"{
              "baseline": 0.16,
              "budget": 111.5,
              "date": "2026-01-01",
              "models": {
                "embed": {"type":"embed","file":"/e.gguf","measurements":{"h":{"status":"ok","d_total":7.0,"load_s":6.0}}},
                "chat":  {"type":"llm","file":"/c.gguf","measurements":{"g":{"status":"ok","d_total":30.0,"load_s":20.0}}}
              },
              "additivity_check": {"combo":["embed","chat"],"predicted":37.16,"measured":37.16,"error":0.0}
            }"#,
        )
        .unwrap();

        let store = Store::new(dir.path().join("measurements"));
        let migrated = store.migrate_legacy(&legacy).unwrap();
        assert_eq!(migrated, 2);

        let box_meta = store.read_box().unwrap();
        assert_eq!(box_meta.baseline, 0.16);
        assert_eq!(box_meta.detected_total, Some(111.5));
        assert_eq!(store.select("embed", "h").unwrap().unwrap().d_total, 7.0);
        assert_eq!(store.select("chat", "g").unwrap().unwrap().d_total, 30.0);

        // idempotent: a second call is a no-op (store already populated)
        assert_eq!(store.migrate_legacy(&legacy).unwrap(), 0);
    }
}
