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

/// The reserved box-level file name inside `measurements/`.
pub const BOX_FILE: &str = "_box.json";

/// Box-level values that have no per-model home.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BoxMeta {
    /// Empty-pool occupancy in GB.
    #[serde(default)]
    pub baseline: f64,
    /// Physical pool total in GB at sweep time (build may override via budget).
    #[serde(default)]
    pub detected_total: Option<f64>,
    /// Last sweep date (YYYY-MM-DD).
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub additivity_check: Option<AdditivityCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdditivityCheck {
    pub combo: Vec<String>,
    pub predicted: f64,
    pub measured: f64,
    pub error: f64,
}

/// One measured footprint, keyed in a model's file by its param-hash.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Measurement {
    /// "ok" | "FAILED".
    pub status: String,
    #[serde(default)]
    pub d_total: f64,
    #[serde(default)]
    pub d_vram: f64,
    #[serde(default)]
    pub d_gtt: f64,
    #[serde(default)]
    pub abs_total: f64,
    #[serde(default)]
    pub abs_vram: f64,
    #[serde(default)]
    pub abs_gtt: f64,
    #[serde(default)]
    pub load_s: f64,
    /// The hashed (memory) command, human-readable.
    #[serde(default)]
    pub params: String,
    #[serde(default)]
    pub measured_at: String,
}

impl Measurement {
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
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
        let json = serde_json::to_string_pretty(meta)?;
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
        let store =
            serde_json::from_str(&json).with_context(|| format!("parsing {}", path.display()))?;
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
        })?;
        for (id, model_store) in &legacy.models {
            self.write_model(id, model_store)?;
        }
        Ok(legacy.models.len())
    }

    /// The `ok` measurement for `(id, param_hash)`. Falls back to a model's sole
    /// `ok` measurement (hand-set proxy entries not tied to a live config hash).
    pub fn select(&self, id: &str, param_hash: &str) -> Result<Option<Measurement>> {
        let Some(store) = self.read_model(id)? else {
            return Ok(None);
        };
        if let Some(measurement) = store.measurements.get(param_hash) {
            if measurement.is_ok() {
                return Ok(Some(measurement.clone()));
            }
        }
        let ok_measurements: Vec<&Measurement> =
            store.measurements.values().filter(|entry| entry.is_ok()).collect();
        if ok_measurements.len() == 1 {
            return Ok(Some(ok_measurements[0].clone()));
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
            })
            .unwrap();
        assert_eq!(store.read_box().unwrap().detected_total, Some(111.5));

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
        // sole-ok fallback on a hash miss
        assert_eq!(store.select("coder-70b", "other").unwrap().unwrap().d_total, 49.0);
        assert_eq!(store.list_ids(), vec!["coder-70b".to_string()]);
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
