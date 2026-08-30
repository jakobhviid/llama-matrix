//! Render a [`MatrixPlan`](crate::build::MatrixPlan) to the llama-swap `matrix:`
//! DSL block, prefixed with the generated marker that `apply` anchors its splice
//! on (so a regeneration replaces the previous block cleanly, and the first
//! cutover from `groups:` uses the same code path).

use crate::build::MatrixPlan;

/// The marker line every generated block starts with. `apply` cuts on this.
pub const MARKER: &str = "# ==== GENERATED matrix block (llama-matrix) ====";

/// Render the plan to the full `matrix:` block text (ends with a newline).
pub fn render(plan: &MatrixPlan) -> String {
    let mut lines: Vec<String> = Vec::new();

    lines.push(MARKER.to_string());
    lines.push(format!(
        "# budget {:.1} GB | baseline {:.2} | margin {:.1} | ceiling {:.1}",
        plan.budget, plan.baseline, plan.margin, plan.ceiling
    ));
    // Normalize a possible negative zero so an aux-less roster reads "0.0", not "-0.0".
    let aux_cost = if plan.aux_cost == 0.0 { 0.0 } else { plan.aux_cost };
    lines.push(format!(
        "# policy: max flexibility, never OOM. {} packs, {} heavies, aux rides along ({aux_cost:.1} GB).",
        plan.n_packs, plan.n_heavies
    ));
    for warning in &plan.warnings {
        lines.push(format!("# WARNING: {warning}"));
    }
    if !plan.excluded.is_empty() {
        lines.push(format!("# NOT measured, excluded: {}", plan.excluded.join(", ")));
    }

    lines.push("matrix:".to_string());

    if !plan.vars.is_empty() {
        lines.push("  vars:".to_string());
        for (alias, id) in &plan.vars {
            lines.push(format!("    {alias}: {id}"));
        }
    }

    if !plan.evict_costs.is_empty() {
        lines.push(
            "  evict_costs:   # higher = costlier to evict = prefer to keep; set in \
             [evict_costs] (llama-matrix.toml)"
                .to_string(),
        );
        for (id, cost) in &plan.evict_costs {
            lines.push(format!("    {id}: {cost}"));
        }
    }

    lines.push("  sets:".to_string());
    for set in &plan.sets {
        lines.push(format!("    {}: \"{}\"   # {}", set.name, set.expr, set.comment));
    }

    let mut text = lines.join("\n");
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{build, BuildInput, ModelFootprint};
    use crate::model::ModelType;
    use crate::policy::Policy;

    #[test]
    fn renders_a_well_formed_block() {
        let models = vec![
            ModelFootprint {
                host_gb: None,
                cache_gb: 0.0,
                holds_cache: false,
                occupies_slot: true,
                id: "embed".into(),
                model_type: ModelType::Embed,
                primary_file: Some("/e.gguf".into()),
                d_total: 7.0,
            },
            ModelFootprint {
                host_gb: None,
                cache_gb: 0.0,
                holds_cache: false,
                occupies_slot: true,
                id: "chat".into(),
                model_type: ModelType::Llm,
                primary_file: Some("/c.gguf".into()),
                d_total: 30.0,
            },
        ];
        let plan = build(&BuildInput {
            host: None,
            models: &models,
            policy: &Policy::default(),
            baseline: 0.16,
            budget: 100.0,
        })
        .unwrap();
        let text = render(&plan);

        assert!(text.starts_with(MARKER));
        assert!(text.contains("\nmatrix:\n"));
        assert!(text.contains("  sets:"));
        assert!(text.contains("aux:"));
        assert!(text.ends_with('\n'));
        // Both models carry a cost, and the block says where the numbers come from.
        assert!(text.contains("  evict_costs:"));
        assert!(text.contains("[evict_costs]"));
        for (id, cost) in &plan.evict_costs {
            assert!(text.contains(&format!("    {id}: {cost}")), "{id} missing a cost line");
        }
        // Every rendered set line stays inside a quoted expression.
        for set in &plan.sets {
            assert!(text.contains(&format!("{}: \"", set.name)));
        }
    }
}

