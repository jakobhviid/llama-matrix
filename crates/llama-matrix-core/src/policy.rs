//! `llama-matrix.toml` — the operator's policy, separate from llama-swap's
//! `config.yaml`. All keys are optional; omission takes the documented default.
//! Scalars are managed by the `configure` surface (see `settings`); the
//! structured tables (`[paths]`, `[roles]`, `[groups]`, `[evict_costs]`) are
//! hand-edited.

use std::path::Path;

use anyhow::{bail, Context, Result};
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

/// Built-in eviction cost of an image model: the cheapest tier. A diffusion server
/// reloads in seconds and is used in bursts, so it is the natural eviction victim.
pub const IMAGE_EVICT_COST: u32 = 1;

/// Built-in eviction cost of an aux ride-along. Aux is reserved in nearly every set,
/// so it is rarely a candidate for eviction at all; the value only decides the few
/// cases where a `[roles]` override leaves an aux model out of some sets.
pub const AUX_EVICT_COST: u32 = 5;

/// Floor for the derived `llm` tier. Above `AUX_EVICT_COST`, round, and stable for
/// small rosters, leaving numeric room to hand-tune a single model between tiers.
pub const MIN_LLM_EVICT_COST: u32 = 10;

/// Largest cost accepted. Well beyond any real tier, and low enough that llama-swap
/// can sum a whole roster of them without overflowing.
pub const MAX_EVICT_COST: u32 = 1_000_000;

/// The tier a model's eviction cost is drawn from, mirroring the `[roles]` split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostRole {
    Llm,
    Image,
    Aux,
}

/// `[evict_costs]`: how much the solver must pay to evict each model.
///
/// llama-swap answers a request by picking the declared set that minimizes the summed
/// cost of the running models it would have to evict, so a cost is a *keep* weight:
/// higher = costlier to evict = prefer to keep. Left uniform, the solver compares body
/// counts, and a pile of idle image servers outvotes the model in active use, so the
/// costs rank by role, with per-id overrides on top.
///
/// Per-model-id overrides live in their own `[evict_costs.models]` sub-table rather
/// than beside the role keys: one flat namespace would make a model literally named
/// `llm` ambiguous, and would read `imgae = 1` as an override for a model that doesn't
/// exist instead of rejecting the typo.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvictCosts {
    /// Conversational models. Unset = derived per roster so that keeping one always
    /// beats keeping the entire idle image pool (see [`EvictCosts::derived_llm`]).
    pub llm: Option<u32>,
    /// The image pool.
    pub image: Option<u32>,
    /// The aux ride-alongs.
    pub aux: Option<u32>,
    /// Per-model-id overrides, which win over the role tier.
    pub models: IndexMap<String, u32>,
}

impl EvictCosts {
    /// The built-in `llm` tier for a roster whose image pool costs `image_pool` in
    /// total.
    ///
    /// The `+ 1` is the whole point: the guarantee wanted is "keeping a second
    /// conversational model beats keeping the entire idle image pool", and the pool is
    /// what it has to outweigh, so the tier scales with the pool rather than with a
    /// magic number.
    pub fn derived_llm(image_pool: u64) -> u32 {
        image_pool
            .saturating_add(1)
            .clamp(MIN_LLM_EVICT_COST as u64, MAX_EVICT_COST as u64) as u32
    }

    /// The cost of one model: its per-id override, else its role tier, else the
    /// built-in for that role. `image_pool` is consulted only for [`CostRole::Llm`].
    pub fn of(&self, id: &str, role: CostRole, image_pool: u64) -> u32 {
        if let Some(&cost) = self.models.get(id) {
            return cost;
        }
        match role {
            CostRole::Llm => self.llm.unwrap_or_else(|| Self::derived_llm(image_pool)),
            CostRole::Image => self.image.unwrap_or(IMAGE_EVICT_COST),
            CostRole::Aux => self.aux.unwrap_or(AUX_EVICT_COST),
        }
    }

    /// Every configured cost must be one llama-swap will accept: a positive integer
    /// (`0` disables nothing, it makes the model free to evict, which the schema
    /// rejects) that a summed roster of them cannot overflow.
    fn validate(&self) -> Result<()> {
        let roles = [("llm", self.llm), ("image", self.image), ("aux", self.aux)];
        for (key, configured) in roles {
            if let Some(cost) = configured {
                check_cost(&format!("[evict_costs] {key}"), cost)?;
            }
        }
        for (id, &cost) in &self.models {
            check_cost(&format!("[evict_costs.models] \"{id}\""), cost)?;
        }
        Ok(())
    }
}

fn check_cost(where_: &str, cost: u32) -> Result<()> {
    if cost == 0 {
        bail!("`{where_} = 0` is invalid: llama-swap eviction costs are positive integers (1 or more)");
    }
    if cost > MAX_EVICT_COST {
        bail!("`{where_} = {cost}` is above the {MAX_EVICT_COST} ceiling llama-matrix emits");
    }
    Ok(())
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
    /// Per-role and per-id eviction costs (which model the solver keeps under pressure).
    pub evict_costs: EvictCosts,
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
            evict_costs: EvictCosts::default(),
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
        let policy: Policy =
            toml::from_str(&s).with_context(|| format!("parsing {}", path.display()))?;
        policy
            .evict_costs
            .validate()
            .with_context(|| format!("parsing {}", path.display()))?;
        Ok(policy)
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
[evict_costs]
llm = 12
image = 2
[evict_costs.models]
"pinned-chat" = 40
"#;
        let p: Policy = toml::from_str(src).unwrap();
        assert_eq!(p.budget, Some(50.0));
        assert_eq!(p.margin, 6.0);
        assert_eq!(p.strategy, Strategy::Family);
        assert_eq!(p.on_overflow, OnOverflow::Error);
        assert_eq!(p.to_host("/models/a.gguf"), "/srv/w/a.gguf");
        assert_eq!(p.roles.aux, vec!["e", "r"]);
        assert_eq!(p.groups["gemma"].len(), 2);
        assert_eq!(p.evict_costs.llm, Some(12));
        assert_eq!(p.evict_costs.models["pinned-chat"], 40);
    }

    /// Per-id override beats the role tier, which beats the built-in.
    #[test]
    fn evict_cost_resolution_order() {
        let src = r#"
[evict_costs]
image = 3
[evict_costs.models]
"pinned-chat" = 40
"#;
        let costs: EvictCosts = toml::from_str::<Policy>(src).unwrap().evict_costs;

        // per-id override wins whatever the role
        assert_eq!(costs.of("pinned-chat", CostRole::Llm, 0), 40);
        // configured role tier beats the built-in
        assert_eq!(costs.of("some-image", CostRole::Image, 0), 3);
        // unconfigured role falls through to the built-in
        assert_eq!(costs.of("embed", CostRole::Aux, 0), AUX_EVICT_COST);
        // an unconfigured llm is derived from the image pool it must outweigh
        assert_eq!(costs.of("chat", CostRole::Llm, 4), MIN_LLM_EVICT_COST);
        assert_eq!(costs.of("chat", CostRole::Llm, 30), 31);
    }

    #[test]
    fn a_zero_or_absurd_cost_is_rejected_and_an_unknown_role_key_too() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("llama-matrix.toml");

        for bad in [
            "[evict_costs]\nimage = 0\n",
            "[evict_costs.models]\n\"chat\" = 0\n",
            "[evict_costs]\nllm = 2000000\n",
            // a typo in a role key must fail loudly, not read as a model override
            "[evict_costs]\nimgae = 1\n",
            // …including the plural that mirrors `[roles] images`
            "[evict_costs]\nimages = 1\n",
        ] {
            std::fs::write(&file, bad).unwrap();
            let error = Policy::load(&file).unwrap_err();
            assert!(
                format!("{error:#}").contains("evict_costs") || format!("{error:#}").contains("imgae")
                    || format!("{error:#}").contains("images"),
                "unhelpful error for {bad:?}: {error:#}"
            );
        }
    }
}
