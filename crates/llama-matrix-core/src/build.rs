//! Phase 2 — build the co-residency plan from measured footprints (pure).
//!
//! Policy = maximum flexibility, never OOM. Emit every maximal combination of
//! models that genuinely fits under `ceiling = budget - margin`, and never one
//! that doesn't (under-declaring is safe; over-declaring OOMs — Principle #1).
//!
//! Pipeline: collapse interchangeable variants into logical units → assign roles
//! (aux ride-alongs, image pool, llm knapsack subjects) → classify heavies →
//! knapsack the light units into maximal fitting packs → emit sets. Every emitted
//! set is checked against the fit invariant; a violation fails the build.

use std::collections::BTreeSet;

use anyhow::{bail, Result};

use crate::cache::Store;
use crate::config;
use crate::model::ModelType;
use crate::policy::{CostRole, OnHostOverflow, OnOverflow, OnUnconfirmed, Policy, Strategy};

/// Work budget for maximal-pack enumeration. Enumerating maximal fitting packs is
/// worst-case exponential in the light-unit count, so the recursion runs under a
/// node ceiling; past this many visited nodes we stop and fail over (never hang).
/// Sized so any physically plausible single-box roster completes with headroom
/// (a ~20-unit powerset is ~1M nodes) while a pathological roster stops in ~a
/// second. The "whole roster fits" common case is short-circuited before we ever
/// recurse, so this only bites the genuinely combinatorial regime.
const ENUM_NODE_BUDGET: usize = 5_000_000;

/// Ceiling on emitted maximal packs. A real matrix has tens; a roster of many
/// distinct pairwise-fitting units can yield C(n,k) maximal packs (thousands+),
/// which is an unusable block regardless. Past this many we stop and fail over.
const MAX_PACKS: usize = 1024;

/// A measured model handed to the builder.
#[derive(Debug, Clone)]
pub struct ModelFootprint {
    pub id: String,
    pub model_type: ModelType,
    /// Weight path as written in the command (used to collapse same-file variants).
    pub primary_file: Option<String>,
    /// GB delta over baseline — the footprint.
    pub d_total: f64,
    /// GB of host RAM this model costs while resident: what it was measured to add
    /// (`d_host`), plus the prompt-cache cap its command declares or the policy
    /// assumes. `None` where the store holds no host measurement, which disables the
    /// host check rather than guessing at it.
    pub host_gb: Option<f64>,
}

/// A logical model: one or more interchangeable variants, sized by the largest.
#[derive(Debug, Clone)]
struct Unit {
    /// Display key (DSL-safe), e.g. the base id or a group name.
    key: String,
    /// Member ids (quant/`-nothink` variants) — a `|` alternative group in the DSL.
    ids: Vec<String>,
    /// Footprint = the largest member's `d_total` (so any quant mix fits).
    size: f64,
    /// Host cost = the largest member's, for the same reason.
    host_size: Option<f64>,
}

impl Unit {
    /// The DSL reference for this unit: a single id, or a `(a | b)` alternative.
    fn expr(&self) -> String {
        if self.ids.len() == 1 {
            self.ids[0].clone()
        } else {
            format!("({})", self.ids.join(" | "))
        }
    }
    /// Fan-out contribution to the 1000-combination cap (its `|`-group size).
    fn fanout(&self) -> usize {
        self.ids.len().max(1)
    }
}

/// One emitted `sets:` entry, with the bookkeeping to prove it's safe.
#[derive(Debug, Clone)]
pub struct EmittedSet {
    pub name: String,
    pub expr: String,
    pub comment: String,
    /// baseline + Σ members (at max quant) + aux_cost — must be ≤ ceiling.
    pub footprint: f64,
    /// The same sum in host RAM: `host_baseline + Σ members + aux`. `None` when the
    /// host dimension is not being checked (see `MatrixPlan::host_ceiling`).
    pub host_footprint: Option<f64>,
    /// Product of the expression's `|`-group sizes — must be ≤ 1000.
    pub fanout: usize,
}

/// The full generated plan, ready to render (see `matrix`).
#[derive(Debug, Clone)]
pub struct MatrixPlan {
    pub vars: Vec<(String, String)>,
    pub evict_costs: Vec<(String, u32)>,
    pub sets: Vec<EmittedSet>,
    pub warnings: Vec<String>,
    pub excluded: Vec<String>,
    /// Models whose footprint was recorded without confirming the allocation
    /// finished, so it may be under-measured (SPEC §7.2). Named here whatever
    /// `on_unconfirmed` did with them, so a `--json` consumer can see them even under
    /// the default `warn`, where they are still packed.
    pub unconfirmed: Vec<String>,
    pub baseline: f64,
    pub budget: f64,
    pub margin: f64,
    pub ceiling: f64,
    pub aux_cost: f64,
    pub n_packs: usize,
    pub n_heavies: usize,
    /// The host-RAM ceiling each set was checked against, when it was checked at
    /// all. `None` means the check did not run, and `host_skipped` says why.
    pub host_ceiling: Option<f64>,
    /// Why the host check did not run, when it did not.
    pub host_skipped: Option<String>,
    /// Sets whose host cost exceeds `host_ceiling`, with what they need.
    pub host_over: Vec<(String, f64)>,
}

/// Inputs to a build. `budget` is already resolved (policy override, else the
/// detected total); `baseline` is the empty-pool occupancy.
pub struct BuildInput<'a> {
    pub models: &'a [ModelFootprint],
    pub policy: &'a Policy,
    pub baseline: f64,
    pub budget: f64,
    /// The host-RAM budget to check each set against. `None` skips the host check
    /// entirely, which is what a box that cannot report host RAM gets.
    pub host: Option<HostBudget>,
}

/// The host-RAM side of the fit: what the box has, and what it holds with nothing
/// loaded. Both measured (SPEC §7.4); `Policy::host_budget` may override the total.
#[derive(Debug, Clone, Copy)]
pub struct HostBudget {
    /// Host RAM held with no model loaded: the OS and everything else the box runs.
    pub baseline: f64,
    /// Host RAM to plan against.
    pub total: f64,
}

/// Load the llama-swap config + measurement store and compute the plan. Shared by
/// the `build` and `drift` verbs (so a second frontend never reimplements it).
/// `budget_override` is the `--budget` flag, if any; budget resolution never
/// guesses: override, else the configured budget, else the detected pool.
pub fn resolve_plan(
    config_path: &str,
    policy: &Policy,
    budget_override: Option<f64>,
    store: &Store,
) -> Result<MatrixPlan> {
    let parsed = config::parse_file(config_path)?;
    let box_meta = store.read_box()?;
    let budget = budget_override
        .or(policy.budget)
        .or(box_meta.detected_total)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no budget set - pass --budget <GB>, set `budget` in llama-matrix.toml, or run \
                 `measure` to detect the pool"
            )
        })?;

    let mut footprints = Vec::new();
    let mut unmeasured = Vec::new();
    // A footprint whose allocation was never confirmed may be a mid-load plateau
    // rather than the real number, which is the one error direction that OOMs
    // (Principle 1). `on_unconfirmed` decides what to do about it; either way the ids
    // are named in the plan so no consumer has to guess which numbers are evidence.
    let mut unconfirmed: Vec<String> = Vec::new();
    let mut dropped_unconfirmed: Vec<String> = Vec::new();
    let mut suspect: Vec<String> = Vec::new();
    // Footprints measured while something else was in the pool (SPEC §7.3). Named
    // because they explain a thinner matrix than the box can actually hold: they can
    // only be too high, so they cost packs rather than risking a pack that does not
    // fit, and re-measuring on a quiet box is what recovers them.
    let mut contended: Vec<String> = Vec::new();
    // Models with no recorded host footprint. One of these disables the host check
    // for the whole plan: a partial host sum is not a smaller answer, it is a wrong
    // one, and reporting it as a budget would be worse than reporting nothing.
    let mut host_unmeasured: Vec<String> = Vec::new();
    for record in &parsed.models {
        let Some(measurement) = store.select(&record.id, &record.param_hash)? else {
            unmeasured.push(record.id.clone());
            continue;
        };
        if !measurement.is_confirmed() {
            unconfirmed.push(record.id.clone());
            match policy.on_unconfirmed {
                OnUnconfirmed::Error => bail!(
                    "`{}`'s footprint was recorded without confirming that the model finished \
                     allocating, so it may be under-measured and a matrix built from it may not \
                     fit; re-run `llama-matrix measure`, or set `on_unconfirmed` to \"warn\" or \
                     \"exclude\"",
                    record.id
                ),
                OnUnconfirmed::Exclude => {
                    dropped_unconfirmed.push(record.id.clone());
                    continue;
                }
                OnUnconfirmed::Warn => {}
            }
        }
        if let (true, Some(ratio), Some(weights)) = (
            measurement.below_weight_floor(),
            measurement.weight_ratio(),
            measurement.weights_gb,
        ) {
            suspect.push(format!(
                "`{}` measured {:.2} GB, only {:.0}% of the {weights:.2} GB of weight files its \
                 command names - a fully offloaded model cannot hold much less than its weights, \
                 so this footprint may be under-measured (partial offload with -ngl/-ot/--cpu-moe \
                 is a legitimate reason to sit lower)",
                record.id,
                measurement.d_total,
                ratio * 100.0
            ));
        }
        if measurement.contended == Some(true) {
            contended.push(record.id.clone());
        }
        // Host cost = what the load was measured to add, plus the prompt cache the
        // process will fill on its own. The cap is read from the LIVE command, not
        // from the measurement: `-cram` moves host RAM only, so a footprint taken at
        // one value describes the GPU at any other, and changing it must not cost a
        // re-measure. A backend that is not llama.cpp has no such cache.
        let host_gb = measurement.d_host.map(|measured| {
            let cache = match record.model_type {
                ModelType::Llm | ModelType::Embed | ModelType::Rerank => {
                    crate::model::declared_cache_ram_gb(&record.cmd)
                        .unwrap_or(policy.host_cache_gb)
                }
                _ => 0.0,
            };
            measured + cache
        });
        if host_gb.is_none() {
            host_unmeasured.push(record.id.clone());
        }
        footprints.push(ModelFootprint {
            id: record.id.clone(),
            model_type: record.model_type,
            primary_file: record.primary_file.clone(),
            d_total: measurement.d_total,
            host_gb,
        });
    }

    let host = (box_meta.host_baseline, policy.host_budget.or(box_meta.host_total));
    let host_budget = match host {
        _ if !host_unmeasured.is_empty() => Err(format!(
            "{} model(s) have no recorded host-RAM footprint, so a pack's host cost cannot be \
             totalled: {}. Re-run `llama-matrix measure --force` to record it",
            host_unmeasured.len(),
            name_list(&host_unmeasured)
        )),
        (Some(baseline), Some(total)) => Ok(HostBudget { baseline, total }),
        _ => Err(
            "this box records no host-RAM total or baseline, so packs are checked against the \
             GPU only. Re-run `llama-matrix measure` on a platform that can report host RAM, or \
             set `host_budget`"
                .to_string(),
        ),
    };

    let mut plan = build(&BuildInput {
        models: &footprints,
        policy,
        baseline: box_meta.baseline,
        budget,
        host: host_budget.as_ref().ok().copied(),
    })?;
    if let Err(reason) = host_budget {
        plan.host_skipped = Some(reason);
    }
    plan.excluded.extend(unmeasured);
    plan.warnings.extend(suspect);
    if !contended.is_empty() {
        plan.warnings.push(format!(
            "{} footprint(s) were measured while something else was resident, so they are \
             probably too high and this matrix is probably smaller than the box can hold: {}. \
             Quiesce anything that requests models - health probes and pollers especially - and \
             re-measure with `llama-matrix measure --force`",
            contended.len(),
            name_list(&contended)
        ));
    }
    if !dropped_unconfirmed.is_empty() {
        plan.warnings.push(format!(
            "{} footprint(s) were recorded without confirming that the model finished allocating, \
             and are excluded from the matrix (a safe under-declaration): {} - re-run \
             `llama-matrix measure` to confirm them",
            dropped_unconfirmed.len(),
            name_list(&dropped_unconfirmed)
        ));
        plan.excluded.extend(dropped_unconfirmed);
    } else if !unconfirmed.is_empty() {
        // Naming the *sets* is the point: the models alone read as a housekeeping
        // note, while "these declared combinations may not fit" is the actual risk,
        // and one unconfirmed aux model puts every set on that list.
        let dependents = sets_naming(&plan.sets, &unconfirmed);
        plan.warnings.push(format!(
            "{} footprint(s) recorded without confirming the model finished allocating, so \
             possibly under-measured: {}. Declared sets that depend on them, which may therefore \
             not fit: {}. Re-run `llama-matrix measure` to confirm them, or set `on_unconfirmed` \
             to \"exclude\" to leave them out of the matrix",
            unconfirmed.len(),
            name_list(&unconfirmed),
            // Past a handful, the ratio is the actionable fact and the names are
            // not: one unconfirmed aux model rides along in nearly every set, and
            // "223 of 224" says that where eight pack names followed by a count
            // does not.
            if dependents.len() > NAMES_SHOWN {
                format!("{} of the {} declared sets", dependents.len(), plan.sets.len())
            } else {
                name_list(&dependents)
            }
        ));
    }
    plan.unconfirmed = unconfirmed;
    Ok(plan)
}

/// The model ids and `+helper` references a set expression names.
fn expr_tokens(expr: &str) -> impl Iterator<Item = &str> {
    expr.split(|character: char| {
        character.is_whitespace() || matches!(character, '&' | '|' | '(' | ')')
    })
    .filter(|token| !token.is_empty())
}

/// Names of the emitted sets that depend on any of `ids`, directly or through a
/// helper set (`+g_*`, `+aux`) that names one.
///
/// The indirection is the interesting half: a variant-collapsed model appears in a
/// pack only as `+g_<key>`, and an aux model appears only as `+aux` - so a single
/// under-measured aux model taints *every* set, which a direct-only search would
/// report as tainting none.
fn sets_naming(sets: &[EmittedSet], ids: &[String]) -> Vec<String> {
    let names_an_id =
        |expr: &str| expr_tokens(expr).any(|token| ids.iter().any(|id| id == token));
    let helper_refs: Vec<String> = sets
        .iter()
        .filter(|set| names_an_id(&set.expr))
        .map(|set| format!("+{}", set.name))
        .collect();
    sets.iter()
        .filter(|set| {
            names_an_id(&set.expr)
                || expr_tokens(&set.expr)
                    .any(|token| helper_refs.iter().any(|reference| reference == token))
        })
        .map(|set| set.name.clone())
        .collect()
}

/// How many names a warning spells out before it starts counting instead.
///
/// A 25-model roster produces a couple of hundred packs, and one unconfirmed aux
/// model taints every one of them. Naming them all is not information: it is
/// unreadable in a terminal, and `matrix::render` copies the warning into
/// `config.yaml` as a single multi-kilobyte comment line. Eight is enough to
/// recognise the shape of the list and act on it.
const NAMES_SHOWN: usize = 8;

/// Render names for a human-readable warning, bounded by [`NAMES_SHOWN`].
fn name_list<S: AsRef<str>>(names: &[S]) -> String {
    fn render<S: AsRef<str>>(slice: &[S]) -> String {
        slice.iter().map(AsRef::as_ref).collect::<Vec<_>>().join(", ")
    }
    match names.len() {
        0 => "none".to_string(),
        count if count <= NAMES_SHOWN => render(names),
        count => format!("{}, and {} more", render(&names[..NAMES_SHOWN]), count - NAMES_SHOWN),
    }
}

/// Sanitize an id into a DSL-safe set-name fragment.
fn safe_key(source: &str) -> String {
    source
        .chars()
        .map(|character| if character.is_ascii_alphanumeric() { character } else { '_' })
        .collect()
}

/// Collapse llm models into logical units. Flat: group by shared weight file
/// (same file ⇒ same model, e.g. `-nothink` twins), else the id is its own unit.
/// Family: additionally merge any units whose ids appear in a declared `[groups]`.
fn collapse_units(llm_models: &[&ModelFootprint], policy: &Policy) -> Vec<Unit> {
    // Flat base: bucket by primary_file (fallback: the id itself), file order kept.
    let mut file_order: Vec<String> = Vec::new();
    let mut by_file: std::collections::HashMap<String, Vec<&ModelFootprint>> =
        std::collections::HashMap::new();
    for model in llm_models {
        let file_key = model.primary_file.clone().unwrap_or_else(|| model.id.clone());
        by_file.entry(file_key.clone()).or_default().push(model);
        if !file_order.contains(&file_key) {
            file_order.push(file_key);
        }
    }
    let mut units: Vec<Unit> = file_order
        .iter()
        .map(|file_key| {
            let members = &by_file[file_key];
            let ids: Vec<String> = members.iter().map(|model| model.id.clone()).collect();
            let size = members.iter().map(|model| model.d_total).fold(0.0_f64, f64::max);
            // The largest member decides the host cost too: a set naming a `|` group
            // has to hold whichever member is loaded.
            let host_size = members
                .iter()
                .map(|model| model.host_gb)
                .try_fold(0.0_f64, |largest, host| Some(largest.max(host?)));
            Unit {
                key: safe_key(&ids[0]),
                ids,
                size,
                host_size,
            }
        })
        .collect();

    if policy.strategy == Strategy::Family && !policy.groups.is_empty() {
        units = apply_groups(units, policy);
    }
    units
}

/// Merge units whose member ids fall in a declared group into one unit per group.
fn apply_groups(units: Vec<Unit>, policy: &Policy) -> Vec<Unit> {
    let mut grouped: Vec<Unit> = Vec::new();
    let mut consumed: BTreeSet<usize> = BTreeSet::new();
    for (group_name, group_ids) in &policy.groups {
        let group_set: BTreeSet<&str> = group_ids.iter().map(String::as_str).collect();
        let mut ids: Vec<String> = Vec::new();
        let mut size = 0.0_f64;
        let mut host_size = Some(0.0_f64);
        for (index, unit) in units.iter().enumerate() {
            if consumed.contains(&index) {
                continue;
            }
            if unit.ids.iter().any(|id| group_set.contains(id.as_str())) {
                consumed.insert(index);
                ids.extend(unit.ids.iter().cloned());
                size = size.max(unit.size);
                host_size = match (host_size, unit.host_size) {
                    (Some(largest), Some(host)) => Some(largest.max(host)),
                    _ => None,
                };
            }
        }
        if !ids.is_empty() {
            grouped.push(Unit {
                key: safe_key(group_name),
                ids,
                size,
                host_size,
            });
        }
    }
    for (index, unit) in units.into_iter().enumerate() {
        if !consumed.contains(&index) {
            grouped.push(unit);
        }
    }
    grouped
}

/// Enumerate the **maximal** fitting packs of `sizes` directly: every subset whose
/// total ≤ `limit` that no further unit can be added to without exceeding it.
/// Emitting only maximal packs is sufficient — llama-swap treats any subset of a
/// declared set as valid (ARCHITECTURE §4.3) — and recording maximality inline (an
/// O(n) test per node) avoids the previous enumerate-all-subsets-then-filter pass,
/// whose maximal filter was O(subsets²) and hung on a large light-unit roster.
///
/// `node_budget` bounds the recursion and `results` is capped at `MAX_PACKS`:
/// enumerating maximal packs is worst-case exponential, so if either limit is hit
/// the walk stops and returns `false` (truncated). The packs collected so far are
/// still valid and safe — a smaller declaration never OOMs (Principle #1) — and the
/// caller warns and applies `on_overflow`. Returns `true` iff the walk completed.
fn enumerate_maximal_packs(
    start: usize,
    chosen: &mut Vec<usize>,
    running_total: f64,
    sizes: &[f64],
    limit: f64,
    node_budget: &mut usize,
    results: &mut Vec<Vec<usize>>,
) -> bool {
    if *node_budget == 0 || results.len() >= MAX_PACKS {
        return false;
    }
    *node_budget -= 1;

    // Maximal ⟺ no unit outside `chosen` still fits in the headroom. (A fitting
    // superset would add exactly such a unit, so "nothing addable" is precisely
    // "no fitting superset" — the predicate the old O(subsets²) filter computed.)
    let nothing_addable = (0..sizes.len())
        .filter(|index| !chosen.contains(index))
        .all(|index| running_total + sizes[index] > limit);
    if nothing_addable && !chosen.is_empty() {
        results.push(chosen.clone());
    }

    for index in start..sizes.len() {
        if running_total + sizes[index] <= limit {
            chosen.push(index);
            let completed = enumerate_maximal_packs(
                index + 1,
                chosen,
                running_total + sizes[index],
                sizes,
                limit,
                node_budget,
                results,
            );
            chosen.pop();
            if !completed {
                return false;
            }
        }
    }
    true
}

/// Build the plan. Returns an error only if the fit invariant is violated (a bug)
/// or an overflow can't be resolved under `on_overflow = error`.
pub fn build(input: &BuildInput) -> Result<MatrixPlan> {
    let BuildInput {
        models,
        policy,
        baseline,
        budget,
        host,
    } = *input;
    let ceiling = budget - policy.margin;
    // The host ceiling, when the box can report one. Same shape as the GPU ceiling
    // so the two arithmetics read alike: a floor the box holds anyway, plus what
    // each resident model costs, against a total less a margin.
    let host_ceiling = host.map(|host| host.total - policy.host_margin);
    let host_of = |model: &ModelFootprint| model.host_gb.unwrap_or(0.0);

    // ---- roles (type-derived; a NON-EMPTY `[roles]` list is AUTHORITATIVE) ----
    // An explicit list REPLACES the derivation for that role instead of adding to
    // it, which is the only way an operator can take a type-derived model OUT of a
    // pool. That is the case that matters: `aux` is reserved in every emitted set,
    // so a large-but-rarely-used embed/rerank model taxes every combination, and the
    // operator needs to be able to demote it to an ordinary evictable unit.
    //
    // This was `override || derived` until 2026-08-28 — purely additive, so it could
    // only ever ADD a model to a pool and silently ignored any attempt to narrow one.
    // The docs advertise `[roles]` as an override, and AUX_EVICT_COST even reasons
    // about "the few cases where a `[roles]` override leaves an aux model out of some
    // sets" — an outcome the additive form could never produce. Parsing was covered
    // by a test; the *effect* was not, which is how it shipped. See the tests below.
    //
    // Trade-off of authoritative-when-non-empty: a list written to PROMOTE one extra
    // model into a pool must now also name the models that would otherwise be
    // derived, or they leave it. That is the honest reading of a table documented as
    // an override, and it fails VISIBLY (the emitted set changes) rather than
    // silently, which is what the additive form did.
    let is_aux = |model: &ModelFootprint| {
        if policy.roles.aux.is_empty() {
            matches!(
                model.model_type,
                ModelType::Embed | ModelType::Rerank | ModelType::Stt | ModelType::TtsProxy
            )
        } else {
            policy.roles.aux.contains(&model.id)
        }
    };
    let is_image = |model: &ModelFootprint| {
        if policy.roles.images.is_empty() {
            model.model_type == ModelType::Image
        } else {
            policy.roles.images.contains(&model.id)
        }
    };

    let aux_models: Vec<&ModelFootprint> = models.iter().filter(|model| is_aux(model)).collect();
    let image_models: Vec<&ModelFootprint> = models
        .iter()
        .filter(|model| !is_aux(model) && is_image(model))
        .collect();
    let llm_models: Vec<&ModelFootprint> = models
        .iter()
        .filter(|model| !is_aux(model) && !is_image(model))
        .collect();

    let aux_cost: f64 = aux_models.iter().map(|model| model.d_total).sum();
    let aux_host: f64 = aux_models.iter().map(|model| host_of(model)).sum();
    let has_aux = !aux_models.is_empty();
    // The ` & +aux` suffix is only valid when an `aux` set is actually emitted;
    // with no aux models it must be omitted, or the block would reference an
    // undefined `+aux` — an invalid config (Principle #7).
    let aux_ref = if has_aux { " & +aux" } else { "" };

    // ---- logical units + heavy classification ----
    let units = collapse_units(&llm_models, policy);
    let smallest_other_unit = |index: usize| -> f64 {
        units
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, unit)| unit.size)
            .fold(f64::INFINITY, f64::min)
    };
    let is_heavy = |index: usize, unit: &Unit| -> bool {
        let other = smallest_other_unit(index);
        let other = if other.is_finite() { other } else { 0.0 };
        baseline + unit.size + other + aux_cost > ceiling
    };
    let (heavy_indices, light_indices): (Vec<usize>, Vec<usize>) =
        (0..units.len()).partition(|&index| is_heavy(index, &units[index]));

    // ---- knapsack over light units → maximal fitting packs ----
    // Enumerate maximal fitting packs directly (each comes out maximal, so no
    // O(subsets²) superset-filter pass). Two shapes are handled cheaply:
    //   • whole light roster fits at once → the single maximal pack is all of it
    //     (short-circuits the powerset for the common "everything co-resides" case);
    //   • otherwise a bounded recursive walk, which fails over (below) if a roster
    //     of many pairwise-fitting distinct units would produce too many packs.
    let light_sizes: Vec<f64> = light_indices.iter().map(|&index| units[index].size).collect();
    let members_limit = ceiling - baseline - aux_cost;
    let mut raw_packs: Vec<Vec<usize>> = Vec::new();
    let completed = if light_sizes.iter().sum::<f64>() <= members_limit {
        if !light_sizes.is_empty() {
            raw_packs.push((0..light_sizes.len()).collect());
        }
        true
    } else {
        let mut node_budget = ENUM_NODE_BUDGET;
        enumerate_maximal_packs(0, &mut Vec::new(), 0.0, &light_sizes, members_limit, &mut node_budget, &mut raw_packs)
    };
    let enumeration_truncated = !completed || raw_packs.len() > MAX_PACKS;
    raw_packs.truncate(MAX_PACKS);
    let mut packs: Vec<BTreeSet<usize>> =
        raw_packs.iter().map(|pack| pack.iter().copied().collect()).collect();
    // Deterministic order: larger packs first, then by total size.
    packs.sort_by(|left, right| {
        let left_size: f64 = left.iter().map(|&position| light_sizes[position]).sum();
        let right_size: f64 = right.iter().map(|&position| light_sizes[position]).sum();
        right
            .len()
            .cmp(&left.len())
            .then(left_size.partial_cmp(&right_size).unwrap_or(std::cmp::Ordering::Equal))
    });

    // ---- images that fit beside a unit (largest subset, smallest-first) ----
    let mut images_by_size: Vec<&ModelFootprint> = image_models.clone();
    images_by_size
        .sort_by(|left, right| left.d_total.partial_cmp(&right.d_total).unwrap_or(std::cmp::Ordering::Equal));
    let images_fitting = |free: f64| -> Vec<&ModelFootprint> {
        let mut chosen = Vec::new();
        let mut total = 0.0;
        for image in &images_by_size {
            if total + image.d_total <= free {
                chosen.push(*image);
                total += image.d_total;
            }
        }
        chosen
    };

    // A set's host cost, or `None` when the host is not being checked: the floor the
    // box holds anyway, plus each resident model's own host cost.
    let host_footprint = |members: f64| -> Option<f64> {
        host.map(|host| host.baseline + members)
    };

    // ---- emit ----
    let mut sets: Vec<EmittedSet> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Enumeration overflow → fail over via the SAME `on_overflow` knob as the
    // 1000-combination cap: `group` (default) keeps the bounded packs and warns
    // loudly to group the roster (a safe under-declaration — a smaller matrix never
    // OOMs, Principle #1); `error` refuses. Symmetric with guard 2 below.
    if enumeration_truncated {
        match policy.on_overflow {
            OnOverflow::Group => warnings.push(format!(
                "the light-unit roster ({} units) produces too many co-residency combinations to \
                 enumerate exhaustively — emitted {} maximal packs and stopped (a safe \
                 under-declaration; a smaller matrix never OOMs). Reduce it with \
                 `strategy = \"family\"` + `[groups]`, or set `on_overflow = \"error\"` to refuse.",
                light_indices.len(),
                packs.len()
            )),
            OnOverflow::Error => bail!(
                "the light-unit roster ({} units) produces too many co-residency combinations to \
                 enumerate; reduce it with `strategy = \"family\"` + `[groups]` (or allow a \
                 truncated matrix with `on_overflow = \"group\"`)",
                light_indices.len()
            ),
        }
    }

    // aux ride-along pool
    if !aux_models.is_empty() {
        let expr = aux_models
            .iter()
            .map(|model| model.id.clone())
            .collect::<Vec<_>>()
            .join(" & ");
        sets.push(EmittedSet {
            name: "aux".to_string(),
            expr,
            comment: format!("ride-along ({aux_cost:.1} GB)"),
            footprint: baseline + aux_cost,
            host_footprint: host_footprint(aux_host),
            fanout: 1,
        });
    }

    // helper set per light unit with >1 variant
    for &index in &light_indices {
        let unit = &units[index];
        if unit.ids.len() > 1 {
            sets.push(EmittedSet {
                name: format!("g_{}", unit.key),
                expr: unit.expr(),
                comment: format!("{:.1} GB (max variant)", unit.size),
                footprint: baseline + unit.size + aux_cost,
                host_footprint: host_footprint(unit.host_size.unwrap_or(0.0) + aux_host),
                fanout: unit.fanout(),
            });
        }
    }

    // images pool
    if !image_models.is_empty() {
        let images_joined = image_models
            .iter()
            .map(|model| model.id.clone())
            .collect::<Vec<_>>()
            .join(" & ");
        let expr = format!("{images_joined}{aux_ref}");
        let footprint: f64 =
            baseline + aux_cost + image_models.iter().map(|model| model.d_total).sum::<f64>();
        sets.push(EmittedSet {
            name: "images".to_string(),
            expr,
            comment: "any image subset + aux".to_string(),
            footprint,
            host_footprint: host_footprint(
                aux_host + image_models.iter().map(|model| host_of(model)).sum::<f64>(),
            ),
            fanout: 1,
        });
    }

    // helper-or-id reference for a light unit
    let unit_reference = |index: usize| -> String {
        let unit = &units[index];
        if unit.ids.len() > 1 {
            format!("+g_{}", unit.key)
        } else {
            unit.ids[0].clone()
        }
    };

    // packs
    for (position, pack) in packs.iter().enumerate() {
        let unit_indices: Vec<usize> = pack.iter().map(|&light_position| light_indices[light_position]).collect();
        let references: Vec<String> = unit_indices.iter().map(|&index| unit_reference(index)).collect();
        let expr = format!("{}{aux_ref}", references.join(" & "));
        let members_total: f64 = unit_indices.iter().map(|&index| units[index].size).sum();
        let members_host: f64 =
            unit_indices.iter().map(|&index| units[index].host_size.unwrap_or(0.0)).sum();
        let fanout: usize = unit_indices.iter().map(|&index| units[index].fanout()).product();
        let names: Vec<&str> = unit_indices.iter().map(|&index| units[index].key.as_str()).collect();
        sets.push(EmittedSet {
            name: format!("pack{}", position + 1),
            expr,
            comment: format!("{:.1} GB: {}", baseline + members_total + aux_cost, names.join("+")),
            footprint: baseline + members_total + aux_cost,
            host_footprint: host_footprint(members_host + aux_host),
            fanout,
        });
    }

    // one light unit + the images that fit beside it
    for &index in &light_indices {
        let unit = &units[index];
        let free = ceiling - baseline - unit.size - aux_cost;
        let fitting = images_fitting(free);
        if !fitting.is_empty() {
            let images_expr = fitting
                .iter()
                .map(|model| model.id.clone())
                .collect::<Vec<_>>()
                .join(" & ");
            let expr = format!("{} & {}{aux_ref}", unit_reference(index), images_expr);
            let footprint =
                baseline + unit.size + aux_cost + fitting.iter().map(|model| model.d_total).sum::<f64>();
            sets.push(EmittedSet {
                name: format!("llmimg_{}", unit.key),
                expr,
                comment: format!("{} + {} imgs", unit.key, fitting.len()),
                footprint,
                host_footprint: host_footprint(
                    unit.host_size.unwrap_or(0.0)
                        + aux_host
                        + fitting.iter().map(|model| host_of(model)).sum::<f64>(),
                ),
                fanout: unit.fanout(),
            });
        }
    }

    // heavies: a heavy runs WITH aux if it fits, else ALONE (it evicts aux too);
    // a model larger than the whole budget can't run at all and is excluded.
    let mut excluded: Vec<String> = Vec::new();
    for &index in &heavy_indices {
        let unit = &units[index];
        if baseline + unit.size > ceiling {
            excluded.extend(unit.ids.iter().cloned());
            warnings.push(format!(
                "`{}` ({:.1} GB) exceeds the ceiling {:.1} GB — excluded (can't run)",
                unit.key, unit.size, ceiling
            ));
            continue;
        }
        let with_aux = has_aux && baseline + unit.size + aux_cost <= ceiling;
        let reserved = if with_aux { aux_cost } else { 0.0 };
        let fitting = images_fitting(ceiling - baseline - unit.size - reserved);
        let images_suffix = if fitting.is_empty() {
            String::new()
        } else {
            format!(
                " & {}",
                fitting.iter().map(|model| model.id.clone()).collect::<Vec<_>>().join(" & ")
            )
        };
        let aux_suffix = if with_aux { " & +aux" } else { "" };
        // Only blame aux when aux actually exists but this heavy can't fit with it.
        if has_aux && !with_aux {
            warnings.push(format!(
                "heavy `{}` is too large to co-reside with aux ({aux_cost:.1} GB) — it runs alone",
                unit.key
            ));
        }
        let footprint =
            baseline + unit.size + reserved + fitting.iter().map(|model| model.d_total).sum::<f64>();
        let aux_note = if has_aux && !with_aux { ", no aux" } else { "" };
        sets.push(EmittedSet {
            name: format!("heavy_{}", unit.key),
            expr: format!("{}{}{}", unit.expr(), images_suffix, aux_suffix),
            comment: format!("{:.1} GB + {} imgs{aux_note}", unit.size, fitting.len()),
            footprint,
            host_footprint: host_footprint(
                unit.host_size.unwrap_or(0.0)
                    + if with_aux { aux_host } else { 0.0 }
                    + fitting.iter().map(|model| host_of(model)).sum::<f64>(),
            ),
            fanout: unit.fanout(),
        });
    }

    // ---- evict_costs: which model the solver keeps when it has to choose ----
    // llama-swap answers a request by picking the declared set that minimizes the
    // summed cost of the running models it would evict. Leave the costs uniform and
    // that comparison is a body count, so a pile of idle image servers outvotes the
    // model in active use. The tiers therefore rank by role (image < aux < llm), with
    // the llm tier derived from the image pool it has to outweigh (ARCHITECTURE §4.7).
    // Every model carries one, not just the non-default tiers, so the block states what
    // the solver will do rather than leaving it to be re-derived.
    let costs = &policy.evict_costs;
    let image_pool: u64 = image_models
        .iter()
        .map(|model| costs.of(&model.id, CostRole::Image, 0) as u64)
        .sum();
    let mut evict_costs: Vec<(String, u32)> = Vec::new();
    for model in models {
        // Nothing can evict what cannot load.
        if excluded.contains(&model.id) {
            continue;
        }
        let role = if is_aux(model) {
            CostRole::Aux
        } else if is_image(model) {
            CostRole::Image
        } else {
            CostRole::Llm
        };
        evict_costs.push((model.id.clone(), costs.of(&model.id, role, image_pool)));
    }
    let unknown: Vec<&str> = costs
        .models
        .keys()
        .map(String::as_str)
        .filter(|id| !models.iter().any(|model| model.id == *id))
        .collect();
    if !unknown.is_empty() {
        warnings.push(format!(
            "`[evict_costs.models]` names {} model id(s) the matrix has no cost for: {}. Check \
             the spelling, or whether the model is unmeasured or excluded",
            unknown.len(),
            name_list(&unknown)
        ));
    }

    // ---- guard 1: the fit invariant (Principle #1) — always fatal ----
    for set in &sets {
        if set.footprint > ceiling + 1e-6 {
            bail!(
                "internal error: emitted set `{}` is {:.2} GB > ceiling {:.2} GB — refusing to \
                 emit an unsafe matrix",
                set.name,
                set.footprint,
                ceiling
            );
        }
    }

    // ---- guard 2: the 1000-combination cap (Principle #7 — never emit invalid) ----
    // `error` refuses; `group` OMITS the over-cap set (a safe under-declaration —
    // dropping a combo never OOMs, it just declares less) and warns. Either way the
    // emitted block is always valid.
    let mut over_cap: Vec<String> = Vec::new();
    for set in &sets {
        if set.fanout > 1000 {
            match policy.on_overflow {
                OnOverflow::Error => bail!(
                    "set `{}` expands to {} combinations (> the 1000 cap); reduce it via `[groups]`, \
                     or set `on_overflow = \"group\"` to omit it",
                    set.name,
                    set.fanout
                ),
                OnOverflow::Group => {
                    warnings.push(format!(
                        "set `{}` expands to {} combinations (> the 1000 cap) — omitted (a safe \
                         under-declaration); split it via `[groups]` to cover those combinations",
                        set.name, set.fanout
                    ));
                    over_cap.push(set.name.clone());
                }
            }
        }
    }
    sets.retain(|set| !over_cap.contains(&set.name));

    // ---- guard 3: the host-RAM budget ----
    // A second, independent budget. The GPU fit is untouched by what happens here:
    // what is at stake is the box's OOM killer picking the largest RSS, which is a
    // llama-server, and presenting as an unexplained upstream death rather than as
    // anything the matrix reports. `warn` is the default because one term of the
    // host sum is a declared cap rather than a measurement (see `OnHostOverflow`).
    let mut host_over: Vec<(String, f64)> = Vec::new();
    if let Some(host_ceiling) = host_ceiling {
        for set in &sets {
            if let Some(needed) = set.host_footprint {
                if needed > host_ceiling {
                    host_over.push((set.name.clone(), needed));
                }
            }
        }
        if !host_over.is_empty() {
            let listed = name_list(
                &host_over
                    .iter()
                    .map(|(name, needed)| format!("{name} needs {needed:.1} GB"))
                    .collect::<Vec<_>>(),
            );
            let advice = format!(
                "{} of the {} declared sets cost more HOST RAM than the {host_ceiling:.1} GB \
                 ceiling: \
                 {listed}. The GPU fit is unaffected; the risk is the host OOM killer picking a \
                 llama-server, which presents as an unexplained upstream death. Bound it with \
                 `-cram <MiB>` on the llama-server entries (its default is 8192 MiB per process, \
                 taken whether or not the flag appears), raise `host_budget`, or set \
                 `on_host_overflow = \"exclude\"` to leave those sets out",
                host_over.len(),
                sets.len()
            );
            match policy.on_host_overflow {
                OnHostOverflow::Error => bail!("{advice}"),
                OnHostOverflow::Exclude => {
                    let names: Vec<String> =
                        host_over.iter().map(|(name, _)| name.clone()).collect();
                    sets.retain(|set| !names.contains(&set.name));
                    warnings.push(format!("{advice} (excluded)"));
                }
                OnHostOverflow::Warn => warnings.push(advice),
            }
        }
    }

    let n_packs = sets.iter().filter(|set| set.name.starts_with("pack")).count();
    let n_heavies = sets.iter().filter(|set| set.name.starts_with("heavy_")).count();
    Ok(MatrixPlan {
        vars: Vec::new(),
        evict_costs,
        sets,
        warnings,
        excluded,
        // Filled by `resolve_plan`, which is the layer that reads the store: the pure
        // builder is handed footprints and has no notion of their provenance.
        unconfirmed: Vec::new(),
        baseline,
        budget,
        margin: policy.margin,
        ceiling,
        aux_cost,
        n_packs,
        n_heavies,
        host_ceiling,
        // Filled by `resolve_plan` when it is the reason the check did not run.
        host_skipped: None,
        host_over,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cache::{BoxMeta, Measurement, ModelStore};
    use crate::param_hash::param_hash;

    const EMBED_CMD: &str = "/app/llama-server -m /models/e.gguf --embedding -c 8192";
    const CHAT_CMD: &str = "/app/llama-server -m /models/chat.gguf -ngl 99 -c 4096";
    const IMG_CMD: &str = "/opt/sdcpp/bin/sd-server --diffusion-model /sd/u.gguf";

    /// A working directory holding a llama-swap config plus a measurement store, where
    /// `unconfirmed` names the ids whose entries carry no allocation confirmation.
    fn store_and_config(dir: &std::path::Path, unconfirmed: &[&str]) -> (String, Store) {
        let config_path = dir.join("config.yaml");
        std::fs::write(
            &config_path,
            format!(
                "models:\n  \"embed\":\n    cmd: \"{EMBED_CMD}\"\n  \"chat\":\n    cmd: \
                 \"{CHAT_CMD}\"\n  \"img\":\n    cmd: \"{IMG_CMD}\"\n"
            ),
        )
        .unwrap();

        let store = Store::new(dir.join("measurements"));
        store
            .write_box(&BoxMeta {
                baseline: 0.16,
                detected_total: Some(100.0),
                ..Default::default()
            })
            .unwrap();
        for (id, cmd, model_type, d_total) in [
            ("embed", EMBED_CMD, "embed", 7.0),
            ("chat", CHAT_CMD, "llm", 30.0),
            ("img", IMG_CMD, "image", 8.87),
        ] {
            let mut measurements = indexmap::IndexMap::new();
            measurements.insert(
                param_hash(cmd),
                Measurement {
                    status: "ok".to_string(),
                    d_total,
                    load_s: 10.0,
                    allocation_confirmed: Some(!unconfirmed.contains(&id)),
                    ..Default::default()
                },
            );
            store
                .write_model(
                    id,
                    &ModelStore {
                        model_type: model_type.to_string(),
                        file: None,
                        measurements,
                    },
                )
                .unwrap();
        }
        (config_path.display().to_string(), store)
    }

    fn budgeted(on_unconfirmed: OnUnconfirmed) -> Policy {
        Policy {
            budget: Some(100.0),
            on_unconfirmed,
            ..Policy::default()
        }
    }

    /// A non-empty `[roles] aux` REPLACES the type derivation. Without this, the
    /// only expressible change is adding a model to the pool, and the case operators
    /// actually need — dropping a big, rarely-used embed/rerank model out of `aux` so
    /// its footprint stops being reserved in every single set — is silently ignored.
    #[test]
    fn a_non_empty_roles_aux_list_can_remove_a_type_derived_model() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, store) = store_and_config(dir.path(), &[]);

        let mut policy = budgeted(OnUnconfirmed::Warn);
        policy.roles.aux = vec!["chat".to_string()]; // deliberately NOT the embed model

        let plan = resolve_plan(&config_path, &policy, None, &store).unwrap();
        let aux = plan
            .sets
            .iter()
            .find(|set| set.name == "aux")
            .expect("an aux set is still emitted");

        assert!(
            aux.expr.contains("chat"),
            "the explicit list promotes `chat` into aux: {}",
            aux.expr
        );
        assert!(
            !aux.expr.contains("embed"),
            "and it must DEMOTE the type-derived `embed`, which the old additive \
             `override || derived` form could never do: {}",
            aux.expr
        );
    }

    /// The default path must be untouched: an absent/empty table still derives roles
    /// from model type, so existing configs keep their behaviour.
    #[test]
    fn an_empty_roles_table_still_derives_aux_from_model_type() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, store) = store_and_config(dir.path(), &[]);

        let plan =
            resolve_plan(&config_path, &budgeted(OnUnconfirmed::Warn), None, &store).unwrap();
        let aux = plan
            .sets
            .iter()
            .find(|set| set.name == "aux")
            .expect("aux is derived with no override");

        assert!(aux.expr.contains("embed"), "derived from type: {}", aux.expr);
        assert!(!aux.expr.contains("chat"), "an llm is not aux: {}", aux.expr);
    }

    /// Under the default `warn` the footprint is still packed, so the warning has to
    /// carry the risk: which models are unconfirmed *and* which declared sets they
    /// put in doubt.
    #[test]
    fn an_unconfirmed_footprint_is_named_along_with_the_sets_it_taints() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, store) = store_and_config(dir.path(), &["img"]);

        let plan = resolve_plan(&config_path, &budgeted(OnUnconfirmed::Warn), None, &store).unwrap();

        assert_eq!(plan.unconfirmed, vec!["img".to_string()]);
        assert!(
            plan.sets.iter().any(|set| set.expr.contains("img")),
            "`warn` keeps packing it"
        );
        let warning = plan
            .warnings
            .iter()
            .find(|warning| warning.contains("without confirming"))
            .expect("an unconfirmed footprint must warn");
        assert!(warning.contains("img"), "{warning}");
        let dependents = sets_naming(&plan.sets, &plan.unconfirmed);
        assert!(!dependents.is_empty(), "the image sets depend on it");
        for set in dependents {
            assert!(warning.contains(&set), "set {set} missing from: {warning}");
        }
    }

    /// aux rides along in every set, so one unconfirmed aux model puts the whole
    /// matrix in doubt - reached only through the `+aux` indirection.
    #[test]
    fn an_unconfirmed_aux_footprint_taints_every_set() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, store) = store_and_config(dir.path(), &["embed"]);

        let plan = resolve_plan(&config_path, &budgeted(OnUnconfirmed::Warn), None, &store).unwrap();
        let warning = plan
            .warnings
            .iter()
            .find(|warning| warning.contains("without confirming"))
            .expect("an unconfirmed footprint must warn");
        // Aux rides along in everything, so the honest summary is that every set is
        // affected, said once. Spelling out a couple of hundred pack names is not
        // information: it is unreadable, and `matrix::render` copies the warning into
        // config.yaml as one comment line.
        // A short list is spelled out, and every set is on it: aux rides along in
        // all of them, so the indirect `+aux` dependency has to be followed. (Past
        // NAMES_SHOWN the same warning switches to a ratio, which the roster here is
        // too small to reach; see `a_warning_names_a_few_and_counts_the_rest`.)
        for set in &plan.sets {
            assert!(warning.contains(&set.name), "set {} missing from: {warning}", set.name);
        }
    }

    /// A warning names a few and counts the rest. The bound is the point: the
    /// unbounded form put every pack name into a config.yaml comment.
    #[test]
    fn a_warning_names_a_few_and_counts_the_rest() {
        let names: Vec<String> = (1..=20).map(|index| format!("m{index}")).collect();
        let rendered = name_list(&names);
        assert!(rendered.starts_with("m1, m2, m3, m4, m5, m6, m7, m8, and 12 more"), "{rendered}");
        assert!(!rendered.contains("m9"), "{rendered}");
        // Short lists are spelled out in full, and an empty one says so.
        assert_eq!(name_list(&names[..3]), "m1, m2, m3");
        assert_eq!(name_list::<String>(&[]), "none");
    }

    #[test]
    fn on_unconfirmed_exclude_drops_it_and_error_refuses() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, store) = store_and_config(dir.path(), &["img"]);

        let plan =
            resolve_plan(&config_path, &budgeted(OnUnconfirmed::Exclude), None, &store).unwrap();
        assert!(plan.excluded.contains(&"img".to_string()));
        for set in &plan.sets {
            assert!(!set.expr.contains("img"), "excluded model appears in {}", set.name);
        }
        // The other two models still build normally.
        assert!(plan.sets.iter().any(|set| set.expr.contains("chat")));

        assert!(
            resolve_plan(&config_path, &budgeted(OnUnconfirmed::Error), None, &store).is_err(),
            "`error` must refuse rather than emit"
        );
    }

    /// The cheap invariant, at build time: a stored footprint below the weights it
    /// loads is surfaced even though it is confirmed and packed.
    #[test]
    fn a_footprint_below_its_weights_on_disk_is_flagged() {
        let dir = tempfile::tempdir().unwrap();
        let (config_path, store) = store_and_config(dir.path(), &[]);
        // Same shape as the reported failure: 8.87 GB recorded against 16.55 GB of
        // weight files (0.54), but confirmed, so only the floor can catch it.
        let mut measurements = indexmap::IndexMap::new();
        measurements.insert(
            param_hash(IMG_CMD),
            Measurement {
                status: "ok".to_string(),
                d_total: 8.87,
                load_s: 12.0,
                allocation_confirmed: Some(true),
                weights_gb: Some(16.55),
                ..Default::default()
            },
        );
        store
            .write_model(
                "img",
                &ModelStore {
                    model_type: "image".to_string(),
                    file: None,
                    measurements,
                },
            )
            .unwrap();

        let plan = resolve_plan(&config_path, &budgeted(OnUnconfirmed::Warn), None, &store).unwrap();
        assert!(plan.unconfirmed.is_empty(), "this entry is confirmed");
        assert!(
            plan.warnings.iter().any(|warning| warning.contains("`img` measured 8.87 GB, only 54%")),
            "expected a weights-floor warning, got {:?}",
            plan.warnings
        );
    }

    fn footprint(
        id: &str,
        model_type: ModelType,
        file: Option<&str>,
        d_total: f64,
    ) -> ModelFootprint {
        ModelFootprint {
            host_gb: None,
            id: id.to_string(),
            model_type,
            primary_file: file.map(String::from),
            d_total,
        }
    }

    /// Synthetic roster exercising heavy classification, packs, images, aux, and
    /// a same-file `-nothink` twin. Asserts the core fit invariant on every set.
    fn scenario() -> Vec<ModelFootprint> {
        vec![
            footprint("embed", ModelType::Embed, Some("/e.gguf"), 7.0),
            footprint("rerank", ModelType::Rerank, Some("/r.gguf"), 7.0),
            footprint("img-a", ModelType::Image, Some("/ia.gguf"), 4.0),
            footprint("img-b", ModelType::Image, Some("/ib.gguf"), 9.0),
            footprint("small-a", ModelType::Llm, Some("/sa.gguf"), 20.0),
            footprint("small-b", ModelType::Llm, Some("/sb.gguf"), 25.0),
            footprint("mid", ModelType::Llm, Some("/mid.gguf"), 40.0),
            // twin: two ids share one weight file -> one logical unit
            footprint("twin", ModelType::Llm, Some("/twin.gguf"), 20.0),
            footprint("twin-nothink", ModelType::Llm, Some("/twin.gguf"), 20.0),
            // heavy: cannot co-reside with even the smallest other unit
            footprint("huge", ModelType::Llm, Some("/huge.gguf"), 85.0),
        ]
    }

    /// The host budget is a second, independent fit. Two 10 GB-host LLMs plus a
    /// 4 GB baseline is 24 GB against a 31.7 GB box less a 4 GB margin: fine. Three
    /// is 34 GB: not. The GPU says all three fit, and on the box that motivated this
    /// the OOM killer disagreed.
    #[test]
    fn a_pack_over_the_host_ceiling_is_named_and_can_be_excluded() {
        let with_host = |id: &str, file: &str, gpu: f64, host: f64| ModelFootprint {
            host_gb: Some(host),
            id: id.to_string(),
            model_type: ModelType::Llm,
            primary_file: Some(file.to_string()),
            d_total: gpu,
        };
        let models = vec![
            with_host("a", "/a.gguf", 20.0, 10.0),
            with_host("b", "/b.gguf", 20.0, 10.0),
            with_host("c", "/c.gguf", 20.0, 10.0),
        ];
        let host = Some(HostBudget { baseline: 4.0, total: 31.7 });
        let mut policy = budgeted(OnUnconfirmed::Warn);
        policy.budget = Some(111.5);

        // warn (the default): the set is emitted, and named with the arithmetic.
        let plan = build(&BuildInput {
            models: &models,
            policy: &policy,
            baseline: 0.16,
            budget: 111.5,
            host,
        })
        .unwrap();
        assert_eq!(plan.host_ceiling, Some(27.7));
        // All three co-reside on the GPU, so there is one maximal pack, and it needs
        // 4 + 30 = 34 GB of host.
        assert_eq!(plan.n_packs, 1);
        let (over, needed) = plan.host_over.first().expect("the pack is over the host ceiling");
        assert_eq!(over, "pack1");
        assert!((needed - 34.0).abs() < 0.01, "{needed}");
        assert!(plan.sets.iter().any(|set| set.name == "pack1"), "warn still emits it");
        assert!(plan.warnings.iter().any(|warning| warning.contains("HOST RAM")));

        // exclude: the same set is left out, which is a safe under-declaration.
        policy.on_host_overflow = OnHostOverflow::Exclude;
        let excluded = build(&BuildInput {
            models: &models,
            policy: &policy,
            baseline: 0.16,
            budget: 111.5,
            host,
        })
        .unwrap();
        assert!(!excluded.sets.iter().any(|set| set.name == "pack1"));

        // error: refuses outright.
        policy.on_host_overflow = OnHostOverflow::Error;
        assert!(build(&BuildInput {
            models: &models,
            policy: &policy,
            baseline: 0.16,
            budget: 111.5,
            host,
        })
        .is_err());
    }

    /// With no host budget the plan is GPU-only and says so by leaving the host
    /// fields empty, rather than reporting a ceiling nothing was checked against.
    #[test]
    fn without_a_host_budget_nothing_is_host_checked() {
        let plan = build(&BuildInput {
            models: &scenario(),
            policy: &budgeted(OnUnconfirmed::Warn),
            baseline: 0.16,
            budget: 111.5,
            host: None,
        })
        .unwrap();
        assert_eq!(plan.host_ceiling, None);
        assert!(plan.host_over.is_empty());
        assert!(plan.sets.iter().all(|set| set.host_footprint.is_none()));
    }

    #[test]
    fn fit_invariant_holds_for_every_set() {
        let models = scenario();
        let policy = Policy::default(); // margin 4.0, flat
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &policy,
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();
        // ceiling = 96; every emitted set must fit (Principle #1).
        for set in &plan.sets {
            assert!(
                set.footprint <= plan.ceiling + 1e-6,
                "set {} = {:.2} > ceiling {:.2}",
                set.name,
                set.footprint,
                plan.ceiling
            );
            assert!(set.fanout <= 1000);
        }
        assert!((plan.ceiling - 96.0).abs() < 1e-9);
    }

    #[test]
    fn twins_collapse_and_huge_is_heavy() {
        let models = scenario();
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();

        // "huge" is a heavy → appears only in a heavy_ set, never in a pack.
        assert_eq!(plan.n_heavies, 1);
        let heavy = plan.sets.iter().find(|set| set.name.starts_with("heavy_")).unwrap();
        assert!(heavy.expr.contains("huge"));
        for set in plan.sets.iter().filter(|set| set.name.starts_with("pack")) {
            assert!(!set.expr.contains("huge"), "pack must not contain a heavy");
        }

        // twin + twin-nothink collapse into one `(a | b)` helper.
        let helper = plan.sets.iter().find(|set| set.name.starts_with("g_twin")).unwrap();
        assert!(
            helper.expr.contains("twin")
                && helper.expr.contains("twin-nothink")
                && helper.expr.contains('|')
        );

        // aux rides along (embed+rerank = 14).
        assert!((plan.aux_cost - 14.0).abs() < 1e-9);
    }

    #[test]
    fn family_strategy_collapses_declared_groups() {
        use crate::policy::Strategy;
        // Two DISTINCT gemma models (different files) that `flat` would keep
        // separate. Under `family` with a [groups] declaration they collapse into
        // one mutually-exclusive unit, sized by the larger (23).
        let models = vec![
            footprint("embed", ModelType::Embed, Some("/e.gguf"), 5.0),
            footprint("gemma-q4", ModelType::Llm, Some("/g4.gguf"), 20.0),
            footprint("gemma-abliterated", ModelType::Llm, Some("/gab.gguf"), 23.0),
            footprint("other", ModelType::Llm, Some("/o.gguf"), 25.0),
        ];
        let mut policy = Policy {
            strategy: Strategy::Family,
            ..Policy::default()
        };
        policy.groups.insert(
            "gemma".to_string(),
            vec!["gemma-q4".to_string(), "gemma-abliterated".to_string()],
        );

        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &policy,
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();

        // one helper for the collapsed group: (gemma-q4 | gemma-abliterated)
        let helper = plan.sets.iter().find(|set| set.name == "g_gemma").expect("g_gemma helper");
        assert!(
            helper.expr.contains("gemma-q4")
                && helper.expr.contains("gemma-abliterated")
                && helper.expr.contains('|')
        );

        // Both variants appear together only in the `g_gemma` helper, as `|`
        // alternatives (exactly one loads). No OTHER set may name both — they must
        // be referenced via `+g_gemma`, never co-resident.
        for set in plan.sets.iter().filter(|set| set.name != "g_gemma") {
            assert!(
                !(set.expr.contains("gemma-q4") && set.expr.contains("gemma-abliterated")),
                "set {} co-locates both gemma variants: {}",
                set.name,
                set.expr
            );
        }

        // the group pairs with `other` via its +g_gemma reference
        assert!(
            plan.sets
                .iter()
                .any(|set| set.name.starts_with("pack") && set.expr.contains("+g_gemma") && set.expr.contains("other")),
            "expected a pack pairing the gemma group with `other`"
        );
    }

    #[test]
    fn overflow_group_omits_the_over_cap_set_and_error_refuses() {
        use crate::policy::{OnOverflow, Strategy};
        // Two family groups of 32 tiny variants each → a pack pairing them has a
        // fan-out of 32×32 = 1024 (> the 1000 cap).
        let mut models = Vec::new();
        for group in ["gA", "gB"] {
            for index in 0..32 {
                models.push(footprint(
                    &format!("{group}-{index}"),
                    ModelType::Llm,
                    Some(&format!("/{group}-{index}.gguf")),
                    1.0,
                ));
            }
        }
        let mut group_policy = Policy {
            strategy: Strategy::Family,
            ..Policy::default()
        };
        group_policy
            .groups
            .insert("gA".to_string(), (0..32).map(|index| format!("gA-{index}")).collect());
        group_policy
            .groups
            .insert("gB".to_string(), (0..32).map(|index| format!("gB-{index}")).collect());

        // group (default): the 1024-combo pack is omitted, and every surviving set
        // is within the cap — the emitted block is always valid.
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &group_policy,
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();
        assert!(plan.sets.iter().all(|set| set.fanout <= 1000), "an over-cap set survived");
        assert!(
            plan.warnings.iter().any(|warning| warning.contains("omitted")),
            "expected an overflow-omission warning"
        );

        // error: refuse outright rather than emit
        let error_policy = Policy {
            on_overflow: OnOverflow::Error,
            ..group_policy.clone()
        };
        assert!(build(&BuildInput {
            host: None,
            models: &models,
            policy: &error_policy,
            baseline: 0.16,
            budget: 100.0,
        })
        .is_err());
    }

    #[test]
    fn a_fitting_pair_is_declared_and_a_too_big_one_is_not() {
        let models = scenario();
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();
        // small-a(20)+small-b(25)+base(0.16)+aux(14) = 59.16 ≤ 96 → some pack covers both.
        let covers_pair = plan.sets.iter().any(|set| {
            set.name.starts_with("pack") && set.expr.contains("small-a") && set.expr.contains("small-b")
        });
        assert!(covers_pair, "expected a pack co-locating small-a + small-b");
    }

    #[test]
    fn no_aux_models_means_no_dangling_aux_reference() {
        // A roster with no embed/rerank/stt/tts must never emit a `+aux` reference
        // (there's no `aux` set to point at) — that would be an invalid block.
        let models = vec![
            footprint("model-a", ModelType::Llm, Some("/a.gguf"), 20.0),
            footprint("model-b", ModelType::Llm, Some("/b.gguf"), 25.0),
        ];
        // roomy budget: both are light, packs get emitted
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();
        assert!(!plan.sets.iter().any(|set| set.name == "aux"));
        for set in &plan.sets {
            assert!(!set.expr.contains("+aux"), "dangling +aux in {}: {}", set.name, set.expr);
        }
        // tight budget: both become heavies (also must not reference +aux)
        let tight = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 30.0,
        })
        .unwrap();
        for set in &tight.sets {
            assert!(!set.expr.contains("+aux"), "dangling +aux in {}: {}", set.name, set.expr);
        }
    }

    #[test]
    fn whole_light_roster_that_fits_is_one_maximal_pack() {
        // Many small models that ALL co-reside: the single maximal pack is all of
        // them. This is the roster the old enumerate-all-then-filter pass hung on
        // (2^n subsets + O(subsets²) filter); it must now resolve to exactly one
        // pack, instantly, via the short-circuit.
        let mut models = Vec::new();
        for index in 0..20 {
            models.push(footprint(
                &format!("m{index}"),
                ModelType::Llm,
                Some(&format!("/m{index}.gguf")),
                4.0, // 20 × 4 = 80 GB, all fit under a 96 GB ceiling
            ));
        }
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();
        let packs: Vec<_> = plan.sets.iter().filter(|set| set.name.starts_with("pack")).collect();
        assert_eq!(packs.len(), 1, "the all-fit roster must yield exactly one maximal pack");
        for index in 0..20 {
            assert!(packs[0].expr.contains(&format!("m{index}")), "pack must contain every unit");
        }
        assert!(plan.warnings.is_empty(), "an all-fit roster is not an overflow");
        for set in &plan.sets {
            assert!(set.footprint <= plan.ceiling + 1e-6);
        }
    }

    #[test]
    fn enumeration_overflow_truncates_and_warns_under_group() {
        // Many DISTINCT units that pairwise fit but can't all co-reside → the
        // maximal-pack count explodes (≈ C(n,k)). Under `group` (default) the build
        // must still terminate: keep a bounded, valid set of packs and warn.
        let mut models = Vec::new();
        for index in 0..40 {
            models.push(footprint(
                &format!("m{index}"),
                ModelType::Llm,
                Some(&format!("/m{index}.gguf")),
                8.0, // 40 × 8 = 320 GB ≫ ceiling, but any ~11 co-reside → huge C(40,11)
            ));
        }
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();
        let packs = plan.sets.iter().filter(|set| set.name.starts_with("pack")).count();
        assert!(packs <= MAX_PACKS, "packs must be bounded by MAX_PACKS");
        assert!(
            plan.warnings.iter().any(|warning| warning.contains("too many co-residency combinations")),
            "expected an enumeration-overflow warning"
        );
        // every emitted pack still fits — truncation never emits an unsafe set
        for set in &plan.sets {
            assert!(set.footprint <= plan.ceiling + 1e-6, "{} exceeds ceiling", set.name);
        }
    }

    #[test]
    fn enumeration_overflow_refuses_under_error() {
        use crate::policy::OnOverflow;
        let mut models = Vec::new();
        for index in 0..40 {
            models.push(footprint(
                &format!("m{index}"),
                ModelType::Llm,
                Some(&format!("/m{index}.gguf")),
                8.0,
            ));
        }
        let policy = Policy {
            on_overflow: OnOverflow::Error,
            ..Policy::default()
        };
        assert!(
            build(&BuildInput {
            host: None,
                models: &models,
                policy: &policy,
                baseline: 0.16,
                budget: 100.0,
            })
            .is_err(),
            "an enumeration overflow under `error` must refuse the build"
        );
    }

    /// The cost of a model in a built plan, or `None` if none was emitted for it.
    fn cost_of(plan: &MatrixPlan, id: &str) -> Option<u32> {
        plan.evict_costs
            .iter()
            .find(|(model, _)| model == id)
            .map(|(_, cost)| *cost)
    }

    /// Every model that can run carries a cost, ranked image < aux < llm, so the
    /// solver never has to compare body counts. A heavy is an llm: it sits in exactly
    /// one declared set, so a tier of its own would only change the answer when an
    /// image is requested beside it, and there the llm tier is already the right one.
    #[test]
    fn every_runnable_model_gets_a_role_ranked_cost() {
        let plan = build(&BuildInput {
            host: None,
            models: &scenario(),
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();

        for image in ["img-a", "img-b"] {
            assert_eq!(cost_of(&plan, image), Some(1), "{image}");
        }
        for aux in ["embed", "rerank"] {
            assert_eq!(cost_of(&plan, aux), Some(5), "{aux}");
        }
        // …including both halves of a collapsed twin, and the heavy.
        for llm in ["small-a", "small-b", "mid", "twin", "twin-nothink", "huge"] {
            assert_eq!(cost_of(&plan, llm), Some(10), "{llm}");
        }
        assert_eq!(plan.evict_costs.len(), scenario().len(), "every model, exactly once");
    }

    /// The reported failure: an idle image pool outvoting the model in use. With
    /// uniform costs, keeping four images (4) beat keeping one LLM (1), so the solver
    /// evicted the LLM and the pair thrashed. Keeping either LLM must now cost strictly
    /// more than keeping the whole pool.
    #[test]
    fn one_llm_outweighs_the_whole_idle_image_pool() {
        let mut models = vec![
            footprint("chat-a", ModelType::Llm, Some("/a.gguf"), 28.0),
            footprint("chat-b", ModelType::Llm, Some("/b.gguf"), 26.0),
            footprint("embed", ModelType::Embed, Some("/e.gguf"), 4.0),
        ];
        for index in 0..4 {
            models.push(footprint(
                &format!("img-{index}"),
                ModelType::Image,
                Some(&format!("/i{index}.gguf")),
                8.0,
            ));
        }
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 111.5,
        })
        .unwrap();

        let pool: u32 = (0..4).map(|index| cost_of(&plan, &format!("img-{index}")).unwrap()).sum();
        for llm in ["chat-a", "chat-b"] {
            assert!(
                cost_of(&plan, llm).unwrap() > pool,
                "keeping {llm} ({:?}) must beat keeping the {pool}-cost image pool",
                cost_of(&plan, llm)
            );
        }
        // …and it scales with the pool rather than resting on a magic number: a
        // dearer image tier lifts the llm tier with it.
        let dear_images = Policy {
            evict_costs: crate::policy::EvictCosts {
                image: Some(6),
                ..Default::default()
            },
            ..Policy::default()
        };
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &dear_images,
            baseline: 0.16,
            budget: 111.5,
        })
        .unwrap();
        assert_eq!(cost_of(&plan, "img-0"), Some(6));
        assert_eq!(cost_of(&plan, "chat-a"), Some(25), "4 x 6 + 1");
    }

    /// Per-id override beats the role tier; an override naming nothing in the matrix
    /// is a typo the operator has to be told about, not a silent no-op.
    #[test]
    fn a_per_id_override_wins_and_an_unknown_id_warns() {
        let mut costs = crate::policy::EvictCosts {
            aux: Some(2),
            ..Default::default()
        };
        costs.models.insert("small-a".to_string(), 99);
        costs.models.insert("smal-a".to_string(), 99);
        let policy = Policy {
            evict_costs: costs,
            ..Policy::default()
        };

        let plan = build(&BuildInput {
            host: None,
            models: &scenario(),
            policy: &policy,
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();

        assert_eq!(cost_of(&plan, "small-a"), Some(99), "the override wins");
        assert_eq!(cost_of(&plan, "small-b"), Some(10), "its neighbour keeps the tier");
        assert_eq!(cost_of(&plan, "embed"), Some(2), "the configured aux tier wins");
        assert!(cost_of(&plan, "smal-a").is_none(), "an unknown id emits nothing");
        assert!(
            plan.warnings.iter().any(|warning| warning.contains("smal-a")),
            "expected a typo warning, got {:?}",
            plan.warnings
        );
    }

    #[test]
    fn a_model_too_big_to_run_gets_no_cost() {
        let models = vec![
            footprint("chat", ModelType::Llm, Some("/c.gguf"), 20.0),
            footprint("giant", ModelType::Llm, Some("/g.gguf"), 200.0),
        ];
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();

        assert!(plan.excluded.contains(&"giant".to_string()));
        assert!(cost_of(&plan, "giant").is_none(), "nothing can evict what cannot load");
        assert_eq!(cost_of(&plan, "chat"), Some(10));
    }

    #[test]
    fn heavy_that_fits_with_aux_keeps_aux() {
        // ceiling 92. heavy(75): base+75+small(15)+aux(5) = 95.16 > 92 → heavy,
        // but base+75+aux(5) = 80.16 ≤ 92 → it keeps `+aux`.
        let models = vec![
            footprint("embed", ModelType::Embed, Some("/e.gguf"), 5.0),
            footprint("small", ModelType::Llm, Some("/s.gguf"), 15.0),
            footprint("heavy", ModelType::Llm, Some("/h.gguf"), 75.0),
        ];
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 96.0,
        })
        .unwrap();
        let heavy = plan.sets.iter().find(|set| set.name == "heavy_heavy").unwrap();
        assert!(heavy.expr.contains("+aux"), "a heavy that fits with aux must keep it");
        assert!(heavy.footprint <= plan.ceiling + 1e-6);
        assert!(plan.excluded.is_empty());
    }
}
