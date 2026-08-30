//! `llama-matrix.toml` - the operator's policy, separate from llama-swap's
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

/// What to do when a generated set would exceed llama-swap's 1000-combination cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnOverflow {
    /// Omit any set that exceeds the 1000-combination cap (a safe
    /// under-declaration - dropping a combo never OOMs) and warn loudly.
    #[default]
    Group,
    /// Refuse to emit; the operator groups by hand.
    Error,
}

/// What `build` does with a declared set whose host-RAM cost exceeds the host
/// ceiling. The GPU fit is unaffected; what is at stake is the box's OOM killer
/// picking the largest RSS, which is a llama-server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnHostOverflow {
    /// Emit it, and name every set that is over with the arithmetic.
    ///
    /// The default is `warn` rather than `exclude` because the host cost is part
    /// measurement and part *declaration*: `d_host` is measured, but the host-side
    /// prompt cache is bounded by `-cram`, which the command may leave unstated, and
    /// then [`Policy::host_cache_gb`] stands in for it. Silently deleting packs on
    /// the strength of a stand-in would be the wrong trade; naming them is not.
    #[default]
    Warn,
    /// Leave the set out of the matrix (a safe under-declaration).
    Exclude,
    /// Refuse to emit until the roster or the budget changes.
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
pub const IMAGE_EVICT_COST: u32 = 5;

/// Built-in eviction costs of the small service backends, ordered by what it costs to
/// get them serving again.
///
/// These rank by **model type**, not by the `[roles]` pool a model happens to sit in.
/// The two used to be the same thing, so a `[roles]` list that demoted an embedding
/// model out of `aux` also promoted it into the *llm* tier: a 2-second reload priced
/// like a 100-second one, purely because of where it was not. What a model costs to
/// reload is a property of the model.
///
/// The ordering below is the measured one. Reload to `ready` on the reference box: a
/// proxy is instant (it fronts a service that is already running), whisper and an
/// embedder about 2 s, a reranker about 8 s, and an LLM 10 to 100 s. An image backend
/// reads as 2 s but is not cheap: `ready` for a diffusion server arrives long before
/// it allocates, and the first generation after a reload pays for it, so it sits
/// above the service tiers rather than with them.
pub const TTS_PROXY_EVICT_COST: u32 = 1;
pub const STT_EVICT_COST: u32 = 2;
pub const EMBED_EVICT_COST: u32 = 3;
pub const RERANK_EVICT_COST: u32 = 4;

/// Floor for the derived `llm` tier. Above every service tier, round, and stable for
/// small rosters, leaving numeric room to hand-tune a single model between tiers.
pub const MIN_LLM_EVICT_COST: u32 = 20;

/// Largest cost accepted. Well beyond any real tier, and low enough that llama-swap
/// can sum a whole roster of them without overflowing.
pub const MAX_EVICT_COST: u32 = 1_000_000;

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
    /// Diffusion backends.
    pub image: Option<u32>,
    pub embed: Option<u32>,
    pub rerank: Option<u32>,
    pub stt: Option<u32>,
    #[serde(rename = "tts-proxy", alias = "tts_proxy")]
    pub tts_proxy: Option<u32>,
    /// Shorthand for all four service tiers at once (`embed`, `rerank`, `stt`,
    /// `tts-proxy`). A tier set individually wins over it.
    ///
    /// Kept because it is the key existing configs already use, and because "price
    /// the small services together" is a thing people reasonably want to say in one
    /// line. It is no longer a *category* though: what a model costs to reload
    /// follows from what it is, not from which pool it rides in.
    pub aux: Option<u32>,
    /// Per-model-id overrides, which win over every tier.
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

    /// The cost of one model: its per-id override, else its **type** tier, else the
    /// `aux` shorthand where it applies, else the built-in. `image_pool` is consulted
    /// only for an LLM.
    pub fn of(&self, id: &str, model_type: crate::model::ModelType, image_pool: u64) -> u32 {
        use crate::model::ModelType::*;
        if let Some(&cost) = self.models.get(id) {
            return cost;
        }
        match model_type {
            Llm => self.llm.unwrap_or_else(|| Self::derived_llm(image_pool)),
            Image => self.image.unwrap_or(IMAGE_EVICT_COST),
            Embed => self.embed.or(self.aux).unwrap_or(EMBED_EVICT_COST),
            Rerank => self.rerank.or(self.aux).unwrap_or(RERANK_EVICT_COST),
            Stt => self.stt.or(self.aux).unwrap_or(STT_EVICT_COST),
            TtsProxy => self.tts_proxy.or(self.aux).unwrap_or(TTS_PROXY_EVICT_COST),
        }
    }

    /// Every configured cost must be one llama-swap will accept: a positive integer
    /// (`0` disables nothing, it makes the model free to evict, which the schema
    /// rejects) that a summed roster of them cannot overflow.
    fn validate(&self) -> Result<()> {
        let roles = [
            ("llm", self.llm),
            ("image", self.image),
            ("embed", self.embed),
            ("rerank", self.rerank),
            ("stt", self.stt),
            ("tts-proxy", self.tts_proxy),
            ("aux", self.aux),
        ];
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
    /// GB of **host** RAM llama-matrix may plan against; `None` = the total the
    /// store recorded at sweep time. Host RAM is a second budget: a pack that fits
    /// the GPU can still exhaust the box, and that failure presents as an
    /// unexplained upstream death rather than as anything the matrix reports.
    pub host_budget: Option<f64>,
    /// GB safety slack inside the host budget. Mirrors `margin`, for the same
    /// reason: the measured host baseline is a snapshot, and everything on the box
    /// grows.
    pub host_margin: f64,
    /// GB of host RAM to assume a llama-server holds for its prompt cache when its
    /// command does not say.
    ///
    /// llama.cpp's `-cram` / `--cache-ram` defaults to 8192 MiB **per process** on
    /// builds that have it (upstream PR 16391), and the memory is anonymous and
    /// private-dirty: the kernel cannot reclaim it, and llama.cpp evicts only
    /// against its own cap, never against host pressure. Nothing in a llama-swap
    /// `cmd` has to mention the flag for the process to take it.
    ///
    /// This is the one number in the host arithmetic that is assumed rather than
    /// read, which is why it is a setting: a command that states `-cram` uses its
    /// own value, and a build with no such cache takes `0`.
    pub host_cache_gb: f64,
    pub on_host_overflow: OnHostOverflow,
    /// Cap on how many models a declared set may hold at once; `None` = no cap.
    ///
    /// A count is the one axis no budget in gigabytes can express. `budget`/`margin`
    /// already cap a set's VRAM and `host_budget`/`host_margin` its host RAM; this
    /// caps the *number* of servers, which is what bounds process count, swap churn
    /// and anything that scales per-process rather than per-gigabyte.
    ///
    /// Proxy entries do not count (see [`crate::build::ModelFootprint::occupies_slot`]),
    /// and an alternation costs one whatever it names, so grouping models is also how
    /// you spend fewer slots.
    pub max_models_per_set: Option<usize>,
    /// Cap on how many **llama.cpp servers** a declared set may hold at once; `None`
    /// = no cap.
    ///
    /// The sharper of the two, because a llama.cpp server holds a host prompt cache
    /// (8192 MiB by default, §7.4) and nothing else on a typical roster does: an
    /// image, STT or proxy entry costs a model slot but no cache slot. Where
    /// `host_budget` is unusable because the store has no `d_host` yet, this bounds
    /// the same resource by counting instead of measuring.
    pub max_cache_holders_per_set: Option<usize>,
    pub on_overflow: OnOverflow,
    pub on_unconfirmed: OnUnconfirmed,
    /// `WxH` for the image load-trigger. What a diffusion model allocates scales
    /// with the resolution it generates at, so this decides what an image
    /// footprint *means*: measure at the size you actually serve.
    pub probe_image_size: String,
    /// Container → host weight-dir mapping (empty for native deployments).
    pub paths: IndexMap<String, String>,
    pub roles: Roles,
    /// Named groups of distinct model ids, each collapsed into one mutually
    /// exclusive unit (`(a | b | c)`, sized by the largest member).
    ///
    /// Applies to **any** model type, images included, and applies whenever it is
    /// declared. There is no separate switch to turn it on: a table an operator wrote
    /// by hand that quietly does nothing is the failure mode this tool has already
    /// been bitten by once.
    pub groups: IndexMap<String, Vec<String>>,
    /// Per-id model-type overrides, e.g. `"my-sd-fork" = "image"`.
    ///
    /// Type is normally derived from the launch command (binary and flags), which
    /// covers llama.cpp, stable-diffusion.cpp and whisper.cpp and falls back to `llm`
    /// for anything else. That fallback is not harmless: type picks the **load
    /// trigger**, so an unrecognised image backend gets sent a chat completion, fails
    /// to load, and is excluded from the matrix with a misleading reason. This is the
    /// escape hatch, and it is a *declaration* about a backend rather than a
    /// measurement, which is why it is hand-edited rather than measured.
    pub types: IndexMap<String, String>,
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
            host_budget: None,
            host_margin: 4.0,
            host_cache_gb: 8.0,
            on_host_overflow: OnHostOverflow::Warn,
            max_models_per_set: None,
            max_cache_holders_per_set: None,
            on_overflow: OnOverflow::Group,
            on_unconfirmed: OnUnconfirmed::Warn,
            probe_image_size: default_probe_image_size(),
            paths: IndexMap::new(),
            roles: Roles::default(),
            groups: IndexMap::new(),
            types: IndexMap::new(),
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
        policy.validate_types().with_context(|| format!("parsing {}", path.display()))?;
        Ok(policy)
    }

    /// Every `[types]` value has to be a type the tool knows, or the override would
    /// silently do nothing and the model would keep the wrong load trigger, which is
    /// the exact failure the table exists to fix.
    fn validate_types(&self) -> Result<()> {
        for (id, name) in &self.types {
            if crate::model::ModelType::from_name(name).is_none() {
                bail!(
                    "`[types] \"{id}\" = \"{name}\"` is not a model type; expected one of {:?}",
                    crate::model::ModelType::NAMES
                );
            }
        }
        Ok(())
    }

    /// The type an operator declared for `id`, if any.
    pub fn declared_type(&self, id: &str) -> Option<crate::model::ModelType> {
        self.types.get(id).and_then(|name| crate::model::ModelType::from_name(name))
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
        assert!(p.budget.is_none());
    }

    #[test]
    fn parses_scalars_and_tables() {
        let src = r#"
budget = 50.0
margin = 6.0
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
        assert_eq!(p.on_overflow, OnOverflow::Error);
        assert_eq!(p.to_host("/models/a.gguf"), "/srv/w/a.gguf");
        assert_eq!(p.roles.aux, vec!["e", "r"]);
        assert_eq!(p.groups["gemma"].len(), 2);
        assert_eq!(p.evict_costs.llm, Some(12));
        assert_eq!(p.evict_costs.models["pinned-chat"], 40);
    }

    /// A type nobody recognises must fail loudly. Accepting it would leave the model
    /// with the wrong load trigger, which is the exact failure `[types]` exists to
    /// fix, and the operator would have no way to tell it had not worked.
    #[test]
    fn an_unknown_declared_type_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("llama-matrix.toml");

        std::fs::write(&file, "[types]\n\"m\" = \"image\"\n").unwrap();
        assert_eq!(Policy::load(&file).unwrap().declared_type("m"), Some(crate::model::ModelType::Image));

        for bad in ["diffusion", "LLM", "tts_proxy", ""] {
            std::fs::write(&file, format!("[types]\n\"m\" = \"{bad}\"\n")).unwrap();
            let error = Policy::load(&file).unwrap_err();
            assert!(format!("{error:#}").contains("is not a model type"), "accepted `{bad}`");
        }
    }

    /// Per-id override beats an explicit tier, which beats the `aux` shorthand, which
    /// beats the built-in. The tier is chosen by model TYPE, so a `[roles]` list that
    /// demotes an embedder out of `aux` does not reprice it as an LLM.
    #[test]
    fn evict_cost_resolution_order() {
        let src = r#"
[evict_costs]
image = 3
[evict_costs.models]
"pinned-chat" = 40
"#;
        let costs: EvictCosts = toml::from_str::<Policy>(src).unwrap().evict_costs;

        use crate::model::ModelType;
        // per-id override wins whatever the type
        assert_eq!(costs.of("pinned-chat", ModelType::Llm, 0), 40);
        // configured tier beats the built-in
        assert_eq!(costs.of("some-image", ModelType::Image, 0), 3);
        // unconfigured type falls through to its own built-in, in reload order
        assert_eq!(costs.of("tts", ModelType::TtsProxy, 0), TTS_PROXY_EVICT_COST);
        assert_eq!(costs.of("whisper", ModelType::Stt, 0), STT_EVICT_COST);
        assert_eq!(costs.of("embed", ModelType::Embed, 0), EMBED_EVICT_COST);
        assert_eq!(costs.of("rerank", ModelType::Rerank, 0), RERANK_EVICT_COST);
        // The built-in tiers must stay in reload order, or the whole point is lost.
        let tiers = [
            TTS_PROXY_EVICT_COST,
            STT_EVICT_COST,
            EMBED_EVICT_COST,
            RERANK_EVICT_COST,
            IMAGE_EVICT_COST,
            MIN_LLM_EVICT_COST,
        ];
        assert!(tiers.windows(2).all(|pair| pair[0] < pair[1]), "{tiers:?} is not ascending");
        // an unconfigured llm is derived from the image pool it must outweigh
        assert_eq!(costs.of("chat", ModelType::Llm, 4), MIN_LLM_EVICT_COST);
        assert_eq!(costs.of("chat", ModelType::Llm, 30), 31);

        // `aux` still prices the service tiers together, and an explicit tier wins.
        let shorthand: EvictCosts =
            toml::from_str::<Policy>("[evict_costs]\naux = 7\nstt = 2\n").unwrap().evict_costs;
        assert_eq!(shorthand.of("embed", ModelType::Embed, 0), 7);
        assert_eq!(shorthand.of("whisper", ModelType::Stt, 0), 2);
        assert_eq!(shorthand.of("chat", ModelType::Llm, 0), MIN_LLM_EVICT_COST);
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
