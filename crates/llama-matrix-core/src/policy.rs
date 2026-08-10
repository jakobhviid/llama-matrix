//! `llama-matrix.toml` — the operator's policy, separate from llama-swap's
//! `config.yaml`. All keys are optional; omission takes the documented default.
//! Scalars are managed by the `configure` surface (see `settings`); the
//! structured tables (`[paths]`, `[roles]`, `[groups]`) are hand-edited.

use std::path::Path;

use anyhow::{Context, Result};
use indexmap::IndexMap;
use serde::Deserialize;

/// The default image-probe resolution. 1024x1024 rather than a token 256x256: the
/// footprint recorded is whatever that generation allocates, so probing tiny would
/// under-measure every operator who serves at a real resolution (Principle 1).
pub fn default_probe_image_size() -> String {
    "1024x1024".to_string()
}

/// The packing strategy: how models become knapsack units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Strategy {
    /// No grouping — every model is independent; declare everything that fits.
    #[default]
    Flat,
    /// Collapse the `[groups]` declarations into single mutually-exclusive units.
    Family,
}

/// What to do when a generated set would exceed llama-swap's 1000-combination cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnOverflow {
    /// Omit any set that exceeds the 1000-combination cap (a safe
    /// under-declaration — dropping a combo never OOMs) and warn loudly.
    #[default]
    Group,
    /// Refuse to emit; the operator groups by hand.
    Error,
}

/// What `build` does with a footprint whose allocation was never confirmed (a
/// measurement taken before the model finished allocating may be a mid-load plateau,
/// which under-counts the matrix - see `cache::Measurement::is_confirmed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnUnconfirmed {
    /// Use it, but warn and name both the models and the sets that depend on them.
    #[default]
    Warn,
    /// Leave the model out of the matrix entirely (a safe under-declaration).
    Exclude,
    /// Refuse to build until the store has been re-measured.
    Error,
}

/// Role overrides (default roles are derived from model type).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Roles {
    /// Ride-along models reserved in every set (default: embed/rerank/stt/tts).
    pub aux: Vec<String>,
    /// The co-resident image pool (default: image-type models).
    pub images: Vec<String>,
}

/// The parsed `llama-matrix.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Policy {
    /// Path to the llama-swap `config.yaml` (default: `config.yaml` beside this
    /// file). A `--config` flag overrides it.
    pub config: Option<String>,
    /// llama-swap base URL.
    pub endpoint: String,
    /// GB llama-matrix may plan against; `None` = auto-detect the physical total.
    pub budget: Option<f64>,
    /// GB safety slack inside the budget (`ceiling = budget - margin`).
    pub margin: f64,
    pub strategy: Strategy,
    pub on_overflow: OnOverflow,
    pub on_unconfirmed: OnUnconfirmed,
    /// `WxH` for the image load-trigger. What a diffusion model allocates scales
    /// with the resolution it generates at, so this decides what an image
    /// footprint *means*: measure at the size you actually serve.
    pub probe_image_size: String,
    /// Container → host weight-dir mapping (empty for native deployments).
    pub paths: IndexMap<String, String>,
    pub roles: Roles,
    /// Named groups of distinct model ids, consulted only by reduction strategies.
    pub groups: IndexMap<String, Vec<String>>,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            config: None,
            endpoint: "http://localhost:8080".to_string(),
            budget: None,
            margin: 4.0,
            strategy: Strategy::Flat,
            on_overflow: OnOverflow::Group,
            on_unconfirmed: OnUnconfirmed::Warn,
            probe_image_size: default_probe_image_size(),
            paths: IndexMap::new(),
            roles: Roles::default(),
            groups: IndexMap::new(),
        }
    }
}

impl Policy {
    /// Load from a path, or return the defaults if the file doesn't exist.
    pub fn load(path: impl AsRef<Path>) -> Result<Policy> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Policy::default());
        }
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&s).with_context(|| format!("parsing {}", path.display()))
    }

    /// Map a container path to a host path via `[paths]`; unmapped paths pass
    /// through unchanged (native deployments).
    pub fn to_host(&self, container_path: &str) -> String {
        for (prefix, host) in &self.paths {
            if let Some(rest) = container_path.strip_prefix(prefix) {
                return format!("{host}{rest}");
            }
        }
        container_path.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_when_absent() {
        let p = Policy::load("/no/such/llama-matrix.toml").unwrap();
        assert_eq!(p.endpoint, "http://localhost:8080");
        assert_eq!(p.margin, 4.0);
        assert_eq!(p.strategy, Strategy::Flat);
        assert!(p.budget.is_none());
    }

    #[test]
    fn parses_scalars_and_tables() {
        let src = r#"
budget = 50.0
margin = 6.0
strategy = "family"
on_overflow = "error"
[paths]
"/models" = "/srv/w"
[roles]
aux = ["e", "r"]
[groups]
gemma = ["gemma-q4", "gemma-q4-nothink"]
"#;
        let p: Policy = toml::from_str(src).unwrap();
        assert_eq!(p.budget, Some(50.0));
        assert_eq!(p.margin, 6.0);
        assert_eq!(p.strategy, Strategy::Family);
        assert_eq!(p.on_overflow, OnOverflow::Error);
        assert_eq!(p.to_host("/models/a.gguf"), "/srv/w/a.gguf");
        assert_eq!(p.roles.aux, vec!["e", "r"]);
        assert_eq!(p.groups["gemma"].len(), 2);
    }
}
