//! Typed `--json` report shapes: the single source of truth for every verb's
//! machine-readable document. A verb fills one of these and the CLI emits it with
//! `serde_json::to_string`, while the human view is rendered from the same typed
//! data - so the two can never drift (the collect/render split, see
//! ../../CLI-PATTERNS.md and DECISIONS.md D16). This replaces per-verb hand-built
//! `serde_json::json!({...})` documents, which shared no type and nothing guarded.

use serde::Serialize;

use crate::build::MatrixPlan;

/// A bare `{ "status": ... }` document (e.g. `drift` / `measure` with nothing yet).
#[derive(Serialize)]
pub struct Status {
    pub status: &'static str,
}

/// `build` with no `--apply`/`--out`: the plan a bare `build` would print.
#[derive(Serialize)]
pub struct BuildPreview {
    pub budget: f64,
    pub ceiling: f64,
    pub packs: usize,
    pub heavies: usize,
    pub sets: usize,
    pub excluded: Vec<String>,
    /// Models packed from a footprint whose allocation was never confirmed, so the
    /// sets containing them may not fit (SPEC §7.2). Machine-readable because the
    /// default `on_unconfirmed = "warn"` still emits them.
    pub unconfirmed: Vec<String>,
    /// The host-RAM ceiling each set was checked against, when the box could report
    /// one; `null` means the host dimension was not checked (SPEC §7.4).
    pub host_ceiling: Option<f64>,
    /// Sets whose host cost is over that ceiling, as `[name, gb]`.
    pub host_over: Vec<(String, f64)>,
    pub warnings: Vec<String>,
}

impl BuildPreview {
    /// The `--json` view of a freshly built plan.
    pub fn of(plan: &MatrixPlan) -> Self {
        Self {
            budget: plan.budget,
            ceiling: plan.ceiling,
            packs: plan.n_packs,
            heavies: plan.n_heavies,
            sets: plan.sets.len(),
            excluded: plan.excluded.clone(),
            unconfirmed: plan.unconfirmed.clone(),
            host_ceiling: plan.host_ceiling,
            host_over: plan.host_over.clone(),
            warnings: plan.warnings.clone(),
        }
    }
}

/// `build --apply`: what the splice did.
#[derive(Serialize)]
pub struct BuildApplied {
    pub applied: bool,
    pub backup: String,
    pub verified: bool,
    pub note: String,
    pub packs: usize,
    pub heavies: usize,
    pub sets: usize,
    /// As `BuildPreview::unconfirmed` - carried here too, so applying a matrix built
    /// on unconfirmed footprints is visible in the machine-readable record of it.
    pub unconfirmed: Vec<String>,
}

/// `build --out FILE`: where the block was written.
#[derive(Serialize)]
pub struct BuildWrote {
    pub wrote: String,
    pub sets: usize,
}

/// `drift`: the live matrix block vs a fresh build.
#[derive(Serialize)]
pub struct Drift {
    pub in_sync: bool,
    pub has_block: bool,
    pub would_generate_sets: usize,
    pub packs: usize,
    pub heavies: usize,
    pub excluded: Vec<String>,
    pub unconfirmed: Vec<String>,
}

/// `prune`: what was removed, or (with `status`) why there was nothing to do.
#[derive(Serialize)]
pub struct Prune {
    pub removed: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<&'static str>,
}

/// `prune` without `--yes`: what a real prune would remove.
#[derive(Serialize)]
pub struct PrunePreview {
    pub would_remove: Vec<String>,
}

/// `setup` when `llama-matrix.toml` already exists (not overwritten).
#[derive(Serialize)]
pub struct SetupExists {
    pub status: &'static str,
    pub path: String,
}

/// `setup` when a starter `llama-matrix.toml` was written.
#[derive(Serialize)]
pub struct SetupWritten {
    pub written: String,
    pub config: Option<String>,
    pub endpoint: String,
    pub budget: Option<f64>,
    pub gpu: Option<String>,
}

/// `configure get`/`set`: one key's effective value.
#[derive(Serialize)]
pub struct ConfigValue {
    pub key: String,
    pub value: String,
}

/// `configure unset`.
#[derive(Serialize)]
pub struct ConfigUnset {
    pub unset: String,
}

/// One row of `configure keys`.
#[derive(Serialize)]
pub struct SettingInfo {
    pub key: &'static str,
    pub desc: &'static str,
    pub default: &'static str,
}

/// A fatal error, rendered as `{ "error": ... }` on the `--json` failure path so a
/// machine consumer gets a structured document instead of a bare stderr line.
#[derive(Serialize)]
pub struct ErrorReport {
    pub error: String,
}
