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

use crate::model::ModelType;
use crate::policy::{OnOverflow, Policy, Strategy};

/// A measured model handed to the builder.
#[derive(Debug, Clone)]
pub struct ModelFootprint {
    pub id: String,
    pub model_type: ModelType,
    /// Weight path as written in the command (used to collapse same-file variants).
    pub primary_file: Option<String>,
    /// GB delta over baseline — the footprint.
    pub d_total: f64,
    /// Seconds to load (feeds evict_costs).
    pub load_s: f64,
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
    /// Largest member load time (for evict_costs).
    max_load_s: f64,
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
    pub baseline: f64,
    pub budget: f64,
    pub margin: f64,
    pub ceiling: f64,
    pub aux_cost: f64,
    pub n_packs: usize,
    pub n_heavies: usize,
}

/// Inputs to a build. `budget` is already resolved (policy override, else the
/// detected total); `baseline` is the empty-pool occupancy.
pub struct BuildInput<'a> {
    pub models: &'a [ModelFootprint],
    pub policy: &'a Policy,
    pub baseline: f64,
    pub budget: f64,
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
            let max_load_s = members.iter().map(|model| model.load_s).fold(0.0_f64, f64::max);
            Unit {
                key: safe_key(&ids[0]),
                ids,
                size,
                max_load_s,
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
        let mut max_load_s = 0.0_f64;
        for (index, unit) in units.iter().enumerate() {
            if consumed.contains(&index) {
                continue;
            }
            if unit.ids.iter().any(|id| group_set.contains(id.as_str())) {
                consumed.insert(index);
                ids.extend(unit.ids.iter().cloned());
                size = size.max(unit.size);
                max_load_s = max_load_s.max(unit.max_load_s);
            }
        }
        if !ids.is_empty() {
            grouped.push(Unit {
                key: safe_key(group_name),
                ids,
                size,
                max_load_s,
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

/// Enumerate every subset of `sizes` whose total ≤ `limit` (indices strictly
/// increasing, so each combination appears once).
fn enumerate_fitting_subsets(
    start: usize,
    chosen: &mut Vec<usize>,
    running_total: f64,
    sizes: &[f64],
    limit: f64,
    results: &mut Vec<Vec<usize>>,
) {
    for index in start..sizes.len() {
        if running_total + sizes[index] <= limit {
            chosen.push(index);
            enumerate_fitting_subsets(index + 1, chosen, running_total + sizes[index], sizes, limit, results);
            chosen.pop();
        }
    }
    if !chosen.is_empty() {
        results.push(chosen.clone());
    }
}

/// Build the plan. Returns an error only if the fit invariant is violated (a bug)
/// or an overflow can't be resolved under `on_overflow = error`.
pub fn build(input: &BuildInput) -> Result<MatrixPlan> {
    let BuildInput {
        models,
        policy,
        baseline,
        budget,
    } = *input;
    let ceiling = budget - policy.margin;

    // ---- roles (type-derived, with policy overrides) ----
    let is_aux = |model: &ModelFootprint| {
        policy.roles.aux.contains(&model.id)
            || matches!(
                model.model_type,
                ModelType::Embed | ModelType::Rerank | ModelType::Stt | ModelType::TtsProxy
            )
    };
    let is_image = |model: &ModelFootprint| {
        policy.roles.images.contains(&model.id) || model.model_type == ModelType::Image
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
    let light_sizes: Vec<f64> = light_indices.iter().map(|&index| units[index].size).collect();
    let members_limit = ceiling - baseline - aux_cost;
    let mut raw_subsets: Vec<Vec<usize>> = Vec::new();
    enumerate_fitting_subsets(0, &mut Vec::new(), 0.0, &light_sizes, members_limit, &mut raw_subsets);
    let subsets: Vec<BTreeSet<usize>> = raw_subsets
        .iter()
        .map(|subset| subset.iter().copied().collect())
        .collect();
    let mut packs: Vec<BTreeSet<usize>> = subsets
        .iter()
        .filter(|candidate| {
            !subsets
                .iter()
                .any(|other| other.len() > candidate.len() && candidate.is_subset(other))
        })
        .cloned()
        .collect();
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

    // ---- emit ----
    let mut sets: Vec<EmittedSet> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

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
        let fanout: usize = unit_indices.iter().map(|&index| units[index].fanout()).product();
        let names: Vec<&str> = unit_indices.iter().map(|&index| units[index].key.as_str()).collect();
        sets.push(EmittedSet {
            name: format!("pack{}", position + 1),
            expr,
            comment: format!("{:.1} GB: {}", baseline + members_total + aux_cost, names.join("+")),
            footprint: baseline + members_total + aux_cost,
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
        if !with_aux {
            warnings.push(format!(
                "heavy `{}` is too large to co-reside with aux ({aux_cost:.1} GB) — it runs alone",
                unit.key
            ));
        }
        let footprint =
            baseline + unit.size + reserved + fitting.iter().map(|model| model.d_total).sum::<f64>();
        let aux_note = if with_aux { "" } else { ", no aux" };
        sets.push(EmittedSet {
            name: format!("heavy_{}", unit.key),
            expr: format!("{}{}{}", unit.expr(), images_suffix, aux_suffix),
            comment: format!("{:.1} GB + {} imgs{aux_note}", unit.size, fitting.len()),
            footprint,
            fanout: unit.fanout(),
        });
    }

    // ---- evict_costs: protect expensive-to-reload tiers ----
    let mut evict_costs: Vec<(String, u32)> = Vec::new();
    for model in &aux_models {
        evict_costs.push((model.id.clone(), 5));
    }
    for &index in &heavy_indices {
        if baseline + units[index].size > ceiling {
            continue; // excluded (can't run) — no evict cost
        }
        let cost = ((units[index].max_load_s / 4.0).round() as i64).clamp(15, 50) as u32;
        for id in &units[index].ids {
            evict_costs.push((id.clone(), cost));
        }
    }

    // ---- guards: the fit invariant (Principle #1) and the 1000-combo cap ----
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
        if set.fanout > 1000 {
            match policy.on_overflow {
                OnOverflow::Error => bail!(
                    "set `{}` expands to {} combinations (> 1000 cap); set `on_overflow` or add \
                     `[groups]` to reduce it",
                    set.name,
                    set.fanout
                ),
                OnOverflow::Group => warnings.push(format!(
                    "set `{}` expands to {} combinations (> 1000 cap); auto-reduction is not yet \
                     implemented — group it in `[groups]`",
                    set.name, set.fanout
                )),
            }
        }
    }

    let n_packs = packs.len();
    let n_heavies = sets.iter().filter(|set| set.name.starts_with("heavy_")).count();
    Ok(MatrixPlan {
        vars: Vec::new(),
        evict_costs,
        sets,
        warnings,
        excluded,
        baseline,
        budget,
        margin: policy.margin,
        ceiling,
        aux_cost,
        n_packs,
        n_heavies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn footprint(
        id: &str,
        model_type: ModelType,
        file: Option<&str>,
        d_total: f64,
        load_s: f64,
    ) -> ModelFootprint {
        ModelFootprint {
            id: id.to_string(),
            model_type,
            primary_file: file.map(String::from),
            d_total,
            load_s,
        }
    }

    /// Synthetic roster exercising heavy classification, packs, images, aux, and
    /// a same-file `-nothink` twin. Asserts the core fit invariant on every set.
    fn scenario() -> Vec<ModelFootprint> {
        vec![
            footprint("embed", ModelType::Embed, Some("/e.gguf"), 7.0, 6.0),
            footprint("rerank", ModelType::Rerank, Some("/r.gguf"), 7.0, 6.0),
            footprint("img-a", ModelType::Image, Some("/ia.gguf"), 4.0, 2.0),
            footprint("img-b", ModelType::Image, Some("/ib.gguf"), 9.0, 2.0),
            footprint("small-a", ModelType::Llm, Some("/sa.gguf"), 20.0, 10.0),
            footprint("small-b", ModelType::Llm, Some("/sb.gguf"), 25.0, 12.0),
            footprint("mid", ModelType::Llm, Some("/mid.gguf"), 40.0, 30.0),
            // twin: two ids share one weight file -> one logical unit
            footprint("twin", ModelType::Llm, Some("/twin.gguf"), 20.0, 15.0),
            footprint("twin-nothink", ModelType::Llm, Some("/twin.gguf"), 20.0, 15.0),
            // heavy: cannot co-reside with even the smallest other unit
            footprint("huge", ModelType::Llm, Some("/huge.gguf"), 85.0, 70.0),
        ]
    }

    #[test]
    fn fit_invariant_holds_for_every_set() {
        let models = scenario();
        let policy = Policy::default(); // margin 4.0, flat
        let plan = build(&BuildInput {
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
            footprint("embed", ModelType::Embed, Some("/e.gguf"), 5.0, 6.0),
            footprint("gemma-q4", ModelType::Llm, Some("/g4.gguf"), 20.0, 10.0),
            footprint("gemma-abliterated", ModelType::Llm, Some("/gab.gguf"), 23.0, 12.0),
            footprint("other", ModelType::Llm, Some("/o.gguf"), 25.0, 15.0),
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
    fn a_fitting_pair_is_declared_and_a_too_big_one_is_not() {
        let models = scenario();
        let plan = build(&BuildInput {
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
            footprint("model-a", ModelType::Llm, Some("/a.gguf"), 20.0, 10.0),
            footprint("model-b", ModelType::Llm, Some("/b.gguf"), 25.0, 12.0),
        ];
        // roomy budget: both are light, packs get emitted
        let plan = build(&BuildInput {
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
    fn heavy_that_fits_with_aux_keeps_aux() {
        // ceiling 92. heavy(75): base+75+small(15)+aux(5) = 95.16 > 92 → heavy,
        // but base+75+aux(5) = 80.16 ≤ 92 → it keeps `+aux`.
        let models = vec![
            footprint("embed", ModelType::Embed, Some("/e.gguf"), 5.0, 6.0),
            footprint("small", ModelType::Llm, Some("/s.gguf"), 15.0, 10.0),
            footprint("heavy", ModelType::Llm, Some("/h.gguf"), 75.0, 60.0),
        ];
        let plan = build(&BuildInput {
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
