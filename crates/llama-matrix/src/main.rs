//! llama-matrix — measure llama-swap model memory footprints and generate a
//! co-residency `matrix:` block so as many models run concurrently as physically
//! fit, without exceeding VRAM.
//!
//! This is the thin CLI layer; all logic lives in `llama-matrix-core`. Each verb
//! resolves config + policy, calls into core, and renders a human summary or
//! (`--json`) a machine-readable document.
//!
//! Status: scaffolding. The parsing core (config + macro expansion, model
//! classification, param-hash) and the emit surface (`completions`, `--man`,
//! `--llm`) are live; `measure`/`build`/`apply`/`configure`/`setup` are being
//! implemented (see ../../ROADMAP.md).

use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use llama_matrix_core::{apply, build, cache, config as ls_config, matrix, policy::Policy, settings, ui};

mod completions;

const REPO_URL: &str = "https://github.com/jakobhviid/llama-matrix";
const AFTER_HELP: &str = concat!(
    "Repository: https://github.com/jakobhviid/llama-matrix\n",
    "LLM guide: `llama-matrix --llm` prints the full machine-readable reference ",
    "(every command + the design)."
);

#[derive(Parser)]
#[command(
    name = "llama-matrix",
    version,
    about = "Measure llama-swap model footprints and build a co-residency matrix (never exceed VRAM).",
    after_help = AFTER_HELP,
    after_long_help = AFTER_HELP,
    disable_help_subcommand = true
)]
struct Cli {
    /// Emit machine-readable JSON instead of human output.
    #[arg(long, global = true)]
    json: bool,

    /// Print the full LLM-readable guide (every command + the design) and exit.
    #[arg(long, global = true)]
    llm: bool,

    /// Show extra detail where available.
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Measure each model's real footprint (loads each model alone; GPU-touching).
    ///
    /// Sweeps the config worklist, records each model's stabilized VRAM/GTT
    /// footprint into the per-model measurement store, keyed by param-hash.
    /// Incremental: an unchanged model is a cache hit.
    Measure {
        /// llama-swap config.yaml (default: from llama-matrix.toml / discovery).
        #[arg(long)]
        config: Option<String>,
        /// llama-swap base URL (default: from config, else http://localhost:8080).
        #[arg(long)]
        endpoint: Option<String>,
        /// Re-measure even on a cache hit.
        #[arg(long)]
        force: bool,
        /// Restrict to these model ids (comma-separated).
        #[arg(long)]
        only: Option<String>,
    },
    /// Build the `matrix:` block from measured footprints (pure; safe anytime).
    ///
    /// Collapses variants, runs the co-residency knapsack, and emits the block.
    /// Prints by default; `--apply` splices it into config.yaml and verifies.
    Build {
        /// llama-swap config.yaml (default: from llama-matrix.toml, else ./config.yaml).
        #[arg(long)]
        config: Option<String>,
        /// VRAM+GTT pool to plan against, in GB (overrides the configured budget).
        #[arg(long, visible_alias = "vram")]
        budget: Option<f64>,
        /// Safety margin in GB (overrides the configured margin).
        #[arg(long)]
        margin: Option<f64>,
        /// Splice the generated block into config.yaml (backup + verify + rollback).
        #[arg(long)]
        apply: bool,
        /// Write the generated block to a file instead of stdout.
        #[arg(long)]
        out: Option<String>,
    },
    /// Show what's out of sync: current config's matrix vs what build would emit.
    Drift,
    /// Provision llama-matrix.toml (find config + endpoint, detect the budget).
    Setup {
        /// llama-swap config.yaml to use (skips discovery).
        #[arg(long)]
        config: Option<String>,
        /// llama-swap base URL.
        #[arg(long)]
        endpoint: Option<String>,
    },
    /// Get/set llama-matrix.toml scalar settings (budget, margin, strategy, …).
    Configure {
        #[command(subcommand)]
        action: ConfigureAction,
    },
    /// Remove measurement entries whose weight files are gone from disk.
    Prune {
        /// Skip the confirmation prompt.
        #[arg(long)]
        yes: bool,
    },
    /// Print a shell completion script.
    Completions {
        /// The shell to generate for.
        shell: Shell,
    },
}

#[derive(Subcommand)]
enum ConfigureAction {
    /// Set a setting's value (writes llama-matrix.toml, comment-preserving).
    Set { key: String, value: String },
    /// Remove a setting's override (revert to its default).
    Unset { key: String },
    /// Print one setting's effective value.
    Get { key: String },
    /// List every setting with its effective value.
    List,
    /// List the settable keys.
    Keys,
}

fn main() -> ExitCode {
    // `--llm` and `--man` are documentation flags like `--help`: they work from
    // anywhere and need no subcommand, so intercept them before clap enforces one.
    // `--man` is intentionally not a declared flag so it never leaks into the
    // shell completion list.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--llm") {
        print!("{}", llm_guide());
        return ExitCode::SUCCESS;
    }
    if args.iter().any(|a| a == "--man") {
        return match completions::print_man() {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }

    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(cli: Cli) -> Result<()> {
    // json/verbose are consumed by the verbs as they land; llm is handled in main.
    let Cli {
        json,
        verbose,
        llm: _,
        cmd,
    } = cli;
    let _ = verbose;
    match cmd {
        None => {
            Cli::command().print_help()?;
            println!();
        }
        Some(Cmd::Completions { shell }) => completions::print_completions(shell),
        Some(Cmd::Build {
            config,
            budget,
            margin,
            apply,
            out,
        }) => cmd_build(config, budget, margin, apply, out, json)?,
        Some(Cmd::Configure { action }) => cmd_configure(action, json)?,
        Some(Cmd::Measure {
            config,
            endpoint,
            force,
            only,
        }) => cmd_measure(config, endpoint, force, only, json)?,
        Some(other) => not_yet(&verb_name(&other))?,
    }
    Ok(())
}

/// `measure` — the GPU-touching solo-footprint sweep.
fn cmd_measure(
    config: Option<String>,
    endpoint: Option<String>,
    force: bool,
    only: Option<String>,
    json: bool,
) -> Result<()> {
    use llama_matrix_core::measure::{sweep, MeasureOptions};
    use std::time::Duration;

    let policy = Policy::load(PathBuf::from("llama-matrix.toml"))?;
    let config_dir = std::env::current_dir()?;
    let config_path = config
        .or_else(|| policy.config.clone())
        .unwrap_or_else(|| "config.yaml".to_string());
    let parsed = ls_config::parse_file(&config_path)?;
    let store = cache::Store::new(config_dir.join("measurements"));

    let endpoint = endpoint.unwrap_or_else(|| policy.endpoint.clone());
    let only = only.map(|list| list.split(',').map(|id| id.trim().to_string()).collect::<Vec<_>>());
    let options = MeasureOptions {
        endpoint,
        force,
        only,
        load_timeout: Duration::from_secs(300),
    };

    if !json {
        ui::info(&format!(
            "measuring {} models against {} — this loads each model in turn",
            parsed.models.len(),
            options.endpoint
        ));
    }
    let summary = sweep(&parsed.models, &store, &policy, &options)?;

    if json {
        println!(
            "{}",
            serde_json::json!({
                "baseline": summary.baseline,
                "detected_total": summary.detected_total,
                "measured": summary.measured,
                "cached": summary.cached,
                "failed": summary.failed.iter()
                    .map(|(id, reason)| serde_json::json!({ "id": id, "reason": reason }))
                    .collect::<Vec<_>>(),
                "skipped_missing": summary.skipped_missing,
            })
        );
    } else {
        ui::ok(&format!(
            "baseline {:.2} GB · pool {:.1} GB · measured {} · cached {} · failed {} · missing {}",
            summary.baseline,
            summary.detected_total,
            summary.measured.len(),
            summary.cached.len(),
            summary.failed.len(),
            summary.skipped_missing.len()
        ));
        for (id, reason) in &summary.failed {
            ui::warn(&format!("{id}: {reason}"));
        }
        for id in &summary.skipped_missing {
            ui::warn(&format!("{id}: weight file missing — skipped"));
        }
    }
    Ok(())
}

/// `configure` — the validated scalar-settings surface over llama-matrix.toml.
fn cmd_configure(action: ConfigureAction, json: bool) -> Result<()> {
    let file = PathBuf::from("llama-matrix.toml");
    match action {
        ConfigureAction::Set { key, value } => {
            let display = settings::set(&file, &key, &value)?;
            if json {
                println!("{}", serde_json::json!({ "key": key, "value": display }));
            } else {
                ui::ok(&format!("{key} = {display}"));
            }
        }
        ConfigureAction::Unset { key } => {
            settings::unset(&file, &key)?;
            if json {
                println!("{}", serde_json::json!({ "unset": key }));
            } else {
                ui::ok(&format!("unset {key} (reverted to default)"));
            }
        }
        ConfigureAction::Get { key } => {
            let value = settings::get(&file, &key)?;
            if json {
                println!("{}", serde_json::json!({ "key": key, "value": value }));
            } else {
                println!("{value}");
            }
        }
        ConfigureAction::List => {
            let effective = settings::list(&file);
            if json {
                let map: serde_json::Map<String, serde_json::Value> = effective
                    .iter()
                    .map(|(key, value)| ((*key).to_string(), serde_json::Value::String(value.clone())))
                    .collect();
                println!("{}", serde_json::Value::Object(map));
            } else {
                for (key, value) in effective {
                    println!("{key} = {value}");
                }
            }
        }
        ConfigureAction::Keys => {
            if json {
                let entries: Vec<serde_json::Value> = settings::SETTINGS
                    .iter()
                    .map(|setting| {
                        serde_json::json!({
                            "key": setting.key,
                            "desc": setting.desc,
                            "default": setting.default,
                        })
                    })
                    .collect();
                println!("{}", serde_json::Value::Array(entries));
            } else {
                for setting in settings::SETTINGS {
                    println!("{:<12} {}  (default: {})", setting.key, setting.desc, setting.default);
                }
            }
        }
    }
    Ok(())
}

/// `build` — generate the matrix block from the measurement store + config.
fn cmd_build(
    config: Option<String>,
    budget: Option<f64>,
    margin: Option<f64>,
    apply: bool,
    out: Option<String>,
    json: bool,
) -> Result<()> {
    // Policy lives in the working directory; measurements/ sits beside it.
    let mut policy = Policy::load(PathBuf::from("llama-matrix.toml"))?;
    if let Some(margin) = margin {
        policy.margin = margin;
    }
    let config_dir = std::env::current_dir()?;

    // Resolve the llama-swap config path: --config > policy.config > ./config.yaml.
    let config_path = config
        .or_else(|| policy.config.clone())
        .unwrap_or_else(|| "config.yaml".to_string());
    let parsed = ls_config::parse_file(&config_path)?;

    let store = cache::Store::new(config_dir.join("measurements"));
    let box_meta = store.read_box()?;
    if store.list_ids().is_empty() {
        anyhow::bail!(
            "no measurements in {} — run `llama-matrix measure` first",
            store.dir().display()
        );
    }

    // Budget resolution (never a silent guess): --budget > policy > detected total.
    let budget = budget
        .or(policy.budget)
        .or(box_meta.detected_total)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no budget set — pass --budget <GB>, set `budget` in llama-matrix.toml, or run \
                 `measure` to detect the pool"
            )
        })?;

    // Build measured footprints (config × store), noting unmeasured models.
    let mut footprints = Vec::new();
    let mut unmeasured = Vec::new();
    for record in &parsed.models {
        match store.select(&record.id, &record.param_hash)? {
            Some(measurement) => footprints.push(build::ModelFootprint {
                id: record.id.clone(),
                model_type: record.model_type,
                primary_file: record.primary_file.clone(),
                d_total: measurement.d_total,
                load_s: measurement.load_s,
            }),
            None => unmeasured.push(record.id.clone()),
        }
    }

    let mut plan = build::build(&build::BuildInput {
        models: &footprints,
        policy: &policy,
        baseline: box_meta.baseline,
        budget,
    })?;
    plan.excluded.extend(unmeasured);
    let block = matrix::render(&plan);

    if !json {
        for warning in &plan.warnings {
            ui::warn(warning);
        }
    }

    if apply {
        let result = apply::apply(Path::new(&config_path), &block, &policy.endpoint)?;
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "applied": true,
                    "backup": result.backup.display().to_string(),
                    "verified": result.verified,
                    "note": result.note,
                    "packs": plan.n_packs,
                    "heavies": plan.n_heavies,
                    "sets": plan.sets.len(),
                })
            );
        } else {
            ui::ok(&format!("applied to {config_path} — {}", result.note));
            ui::info(&format!("backup: {}", result.backup.display()));
            if !result.verified {
                ui::warn("could not fully verify the reload — check llama-swap's logs");
            }
        }
        return Ok(());
    }

    if let Some(out_path) = out {
        std::fs::write(&out_path, &block)?;
        if json {
            println!("{}", serde_json::json!({ "wrote": out_path, "sets": plan.sets.len() }));
        } else {
            ui::ok(&format!("wrote {out_path} ({} sets)", plan.sets.len()));
        }
        return Ok(());
    }

    if json {
        println!(
            "{}",
            serde_json::json!({
                "budget": plan.budget,
                "ceiling": plan.ceiling,
                "packs": plan.n_packs,
                "heavies": plan.n_heavies,
                "sets": plan.sets.len(),
                "excluded": plan.excluded,
                "warnings": plan.warnings,
            })
        );
    } else {
        print!("{block}");
    }
    Ok(())
}

/// Placeholder for verbs still being implemented, so the CLI surface (and its
/// help, completions, and `--llm` guide) is complete while the internals land.
fn not_yet(verb: &str) -> Result<()> {
    anyhow::bail!(
        "`{verb}` is not implemented in this build yet — the parsing core and the \
         emit surface (completions/--man/--llm) are live; see {REPO_URL} and \
         `llama-matrix --llm` for the design."
    )
}

fn verb_name(cmd: &Cmd) -> String {
    match cmd {
        Cmd::Measure { .. } => "measure",
        Cmd::Build { .. } => "build",
        Cmd::Drift => "drift",
        Cmd::Setup { .. } => "setup",
        Cmd::Configure { .. } => "configure",
        Cmd::Prune { .. } => "prune",
        Cmd::Completions { .. } => "completions",
    }
    .to_string()
}

/// The `--llm` guide: the design docs, embedded at compile time so they never
/// drift from the shipped binary (a doc change ships in the same commit).
fn llm_guide() -> String {
    let mut out = String::new();
    let interactive = io::stdout().is_terminal();
    if interactive {
        out.push_str("# llama-matrix — LLM guide\n\n");
        out.push_str(
            "The full machine-readable reference: what the tool is, how to operate it, \
             and the schemas of record. Sections below are the repository docs, embedded \
             at build time.\n\n",
        );
    }
    let sections: &[(&str, &str)] = &[
        ("README", include_str!("../../../README.md")),
        ("WORKFLOWS", include_str!("../../../WORKFLOWS.md")),
        ("SPEC", include_str!("../../../SPEC.md")),
        ("ARCHITECTURE", include_str!("../../../ARCHITECTURE.md")),
        ("PRINCIPLES", include_str!("../../../PRINCIPLES.md")),
        ("ROADMAP", include_str!("../../../ROADMAP.md")),
    ];
    for (name, body) in sections {
        out.push_str(&format!("\n\n===== {name} =====\n\n"));
        out.push_str(body);
    }
    out
}
