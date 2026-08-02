//! llama-matrix-core — the library behind the `llama-matrix` CLI.
//!
//! `llama-matrix` measures each of a llama-swap server's models' real GPU memory
//! footprint, then builds a co-residency `matrix:` block declaring only the model
//! combinations that provably fit under a budget. llama-swap's solver has no
//! memory awareness — it trusts the declared combinations — so the generator only
//! ever emits fitting ones (under-declaring is safe; over-declaring OOMs).
//!
//! Every capability lives here as a typed function; the `llama-matrix` binary is a
//! thin `--json`-emitting CLI on top. See ../../ARCHITECTURE.md and PRINCIPLES.md.
//!
//! Design rules that hold across the crate:
//! - Measure reality; never guess a footprint (an unmeasurable model is excluded).
//! - Two phases, two side-effect profiles: `measure` touches the GPU; `build` is pure.
//! - The live config is written in exactly one place (`apply`).
//! - Fail loud, never silent (undetected budget, failed load, combo-cap overflow).

pub mod build; // variant-collapse, roles, the knapsack, heavy classification -> a plan
pub mod cache; // the per-model measurement store (measurements/<id>.json + _box.json)
pub mod config; // parse llama-swap config.yaml (+ macro expansion) into model records
pub mod matrix; // render a plan to the `matrix:` DSL block (+ the generated marker)
pub mod measure; // phase 1: trigger -> ready -> stabilize; lockfile; failure classes
pub mod model; // per-model record: id, cmd, type, primary_file, param_hash
pub mod param_hash; // the footprint key: a hash of only the memory-affecting flags
pub mod platform; // GpuMemory trait + AMD sysfs / NVIDIA backends
pub mod policy; // llama-matrix.toml: budget/margin/strategy/roles/groups/paths
pub mod settings; // the `configure` get/set/unset/list/keys surface (scalars)
pub mod ui; // stdout/stderr discipline + colour

// Implemented in a later milestone (kept out of the module tree until it lands):
//   apply       backup -> splice -> reload wait -> verify -> rollback
