//! llama-matrix — measure llama-swap model memory footprints and generate a
//! co-residency `matrix:` block so as many models run concurrently as physically
//! fit, without exceeding VRAM.
//!
//! This is the thin CLI layer; all logic lives in `llama-matrix-core`. Each verb
//! resolves config + policy, calls one core function, and renders either a human
//! summary or (`--json`) a machine-readable document built from a typed report
//! (`llama_matrix_core::report`), so the two views can't drift (ARCHITECTURE.md).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use llama_matrix_core::{
    apply, build, cache, config as ls_config, matrix, policy::Policy, report, settings, ui,
};

mod completions;

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
    ///
    /// A whole-program documentation flag like `--help`/`--version`, so it is
    /// root-only (never `global`): `llama-matrix --llm`, not per-subcommand.
    #[arg(long)]
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
        /// Splice the generated block into config.yaml (backup + liveness check + rollback).
        #[arg(long)]
        apply: bool,
        /// With --apply, skip the post-write liveness check (pure backup + splice;
        /// no network round-trip). llama-swap hot-reloads on its own.
        #[arg(long)]
        no_verify: bool,
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
    ///
    /// A bare `prune` only previews what would be removed; pass `--yes` to delete.
    Prune {
        /// Actually delete the entries (a bare `prune` only previews).
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
    Set {
        /// The setting to change (completes from `configure keys`).
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(settings::keys()))]
        key: String,
        value: String,
    },
    /// Remove a setting's override (revert to its default).
    Unset {
        /// The setting to revert (completes from `configure keys`).
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(settings::keys()))]
        key: String,
    },
    /// Print one setting's effective value.
    Get {
        /// The setting to read (completes from `configure keys`).
        #[arg(value_parser = clap::builder::PossibleValuesParser::new(settings::keys()))]
        key: String,
    },
    /// List every setting with its effective value.
    List,
    /// List the settable keys.
    Keys,
}

fn main() -> ExitCode {
    // Restore the default SIGPIPE disposition. Rust ignores SIGPIPE at startup, so
    // `llama-matrix build | head` would otherwise panic ("failed printing to
    // stdout") when the reader closes the pipe; the default (terminate quietly) is
    // the expected shell behaviour.
    // SAFETY: `signal` with a standard handler is async-signal-safe and this runs
    // once before any threads are spawned.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    // `--llm` and `--man` are whole-program documentation flags like `--help`:
    // they work from anywhere and need no subcommand, so intercept them before clap
    // enforces one. Scan only the *leading* options (up to the first subcommand
    // token), never the whole arg vector, so a future value that happens to equal
    // `--llm` can't hijack them. `--man` is intentionally not a declared flag so it
    // never leaks into the shell completion list.
    let leading: Vec<String> = std::env::args()
        .skip(1)
        .take_while(|arg| arg.starts_with('-'))
        .collect();
    if leading.iter().any(|arg| arg == "--llm") {
        print!("{}", llm_guide());
        return ExitCode::SUCCESS;
    }
    if leading.iter().any(|arg| arg == "--man") {
        return match completions::print_man() {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }

    let cli = Cli::parse();
    let json = cli.json;
    if let Err(e) = run(cli) {
        if json {
            // The error is this run's result document, so it goes to stdout.
            let doc = report::ErrorReport { error: format!("{e:#}") };
            println!("{}", serde_json::to_string(&doc).unwrap_or_default());
        } else {
            eprintln!("error: {e:#}");
        }
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
            no_verify,
            out,
        }) => cmd_build(config, budget, margin, apply, no_verify, out, json)?,
        Some(Cmd::Configure { action }) => cmd_configure(action, json)?,
        Some(Cmd::Measure {
            config,
            endpoint,
            force,
            only,
        }) => cmd_measure(config, endpoint, force, only, json)?,
        Some(Cmd::Setup { config, endpoint }) => cmd_setup(config, endpoint, json)?,
        Some(Cmd::Drift) => cmd_drift(json)?,
        Some(Cmd::Prune { yes }) => cmd_prune(yes, json)?,
    }
    Ok(())
}

/// `prune` — remove measurement files whose weight file is gone from disk
/// (explicit only; the store is otherwise retained indefinitely).
fn cmd_prune(yes: bool, json: bool) -> Result<()> {
    let policy = Policy::load(PathBuf::from("llama-matrix.toml"))?;
    let config_dir = std::env::current_dir()?;
    let store = open_store(&config_dir, json)?;

    let mut removable = Vec::new();
    for id in store.list_ids() {
        if let Some(model_store) = store.read_model(&id)? {
            if let Some(container_file) = &model_store.file {
                let host_path = policy.to_host(container_file);
                if !Path::new(&host_path).exists() {
                    removable.push(id);
                }
            }
        }
    }

    if removable.is_empty() {
        if json {
            let doc = report::Prune {
                removed: Vec::new(),
                status: Some("nothing-to-prune"),
            };
            println!("{}", serde_json::to_string(&doc)?);
        } else {
            ui::ok("nothing to prune — every measured model's weight file is still present");
        }
        return Ok(());
    }

    if !yes {
        if json {
            let doc = report::PrunePreview { would_remove: removable.clone() };
            println!("{}", serde_json::to_string(&doc)?);
        } else {
            ui::warn(&format!(
                "would remove {} measurement file(s) (weights gone): {}",
                removable.len(),
                removable.join(", ")
            ));
            ui::info("re-run with --yes to remove them");
        }
        return Ok(());
    }

    for id in &removable {
        store.remove_model(id)?;
    }
    if json {
        let doc = report::Prune { removed: removable.clone(), status: None };
        println!("{}", serde_json::to_string(&doc)?);
    } else {
        ui::ok(&format!("pruned {} measurement file(s)", removable.len()));
    }
    Ok(())
}

/// Open the measurement store, first migrating a legacy single-file
/// `measurements.json` in the config dir if the per-model store is still empty.
fn open_store(config_dir: &Path, json: bool) -> Result<cache::Store> {
    let store = cache::Store::new(config_dir.join("measurements"));
    let migrated = store.migrate_legacy(&config_dir.join("measurements.json"))?;
    if migrated > 0 && !json {
        ui::info(&format!("migrated {migrated} model(s) from a legacy measurements.json"));
    }
    Ok(store)
}

/// `drift` — compare the live config's matrix block to a fresh build (read-only).
fn cmd_drift(json: bool) -> Result<()> {
    let policy = Policy::load(PathBuf::from("llama-matrix.toml"))?;
    let config_dir = std::env::current_dir()?;
    let config_path = policy.config.clone().unwrap_or_else(|| "config.yaml".to_string());
    let store = open_store(&config_dir, json)?;

    if store.list_ids().is_empty() {
        if json {
            let doc = report::Status { status: "no-measurements" };
            println!("{}", serde_json::to_string(&doc)?);
        } else {
            ui::warn("no measurements yet — run `llama-matrix measure`");
        }
        return Ok(());
    }

    let plan = build::resolve_plan(&config_path, &policy, None, &store)?;
    let block = matrix::render(&plan);
    let config_text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {config_path}"))?;
    let existing = apply::existing_block(&config_text);
    let in_sync = existing
        .as_deref()
        .map(|current| current.trim() == block.trim())
        .unwrap_or(false);

    if json {
        let doc = report::Drift {
            in_sync,
            has_block: existing.is_some(),
            would_generate_sets: plan.sets.len(),
            packs: plan.n_packs,
            heavies: plan.n_heavies,
            excluded: plan.excluded.clone(),
            unconfirmed: plan.unconfirmed.clone(),
        };
        println!("{}", serde_json::to_string(&doc)?);
        return Ok(());
    }

    if in_sync {
        ui::ok("in sync — the live matrix block matches a fresh build");
    } else if existing.is_none() {
        ui::warn(&format!(
            "no matrix block in {config_path} — run `llama-matrix build --apply` to add one ({} sets)",
            plan.sets.len()
        ));
    } else {
        ui::warn(&format!(
            "drift — the live block differs from a fresh build ({} sets); run `llama-matrix build --apply`",
            plan.sets.len()
        ));
    }
    if !plan.excluded.is_empty() {
        ui::info(&format!("{} model(s) unmeasured/excluded", plan.excluded.len()));
    }
    if !plan.unconfirmed.is_empty() {
        ui::warn(&format!(
            "{} footprint(s) unconfirmed (may be under-measured): run `llama-matrix build` to see \
             the affected sets, or re-run `measure`",
            plan.unconfirmed.len()
        ));
    }
    Ok(())
}

/// `setup` — provision llama-matrix.toml (discover config + endpoint, probe budget).
fn cmd_setup(config: Option<String>, endpoint: Option<String>, json: bool) -> Result<()> {
    let file = PathBuf::from("llama-matrix.toml");
    let config_path = config.or_else(|| {
        ["config.yaml", "config/config.yaml"]
            .iter()
            .find(|candidate| Path::new(candidate).exists())
            .map(|candidate| (*candidate).to_string())
    });
    let endpoint = endpoint.unwrap_or_else(|| "http://localhost:8080".to_string());
    let (budget, gpu_label) = match llama_matrix_core::platform::detect() {
        Ok(gpu) => (gpu.total_gb().ok(), Some(gpu.label())),
        Err(_) => (None, None),
    };

    let mut lines = Vec::new();
    lines.push(format!("endpoint = \"{endpoint}\""));
    if let Some(path) = &config_path {
        lines.push(format!("config = \"{path}\""));
    }
    match budget {
        Some(total) => {
            lines.push(format!(
                "# detected ~{total:.1} GB pool; lower `budget` to reserve room for other apps"
            ));
            lines.push(format!("budget = {total:.1}"));
        }
        None => {
            lines.push("# no GPU sensor detected — set the pool to plan against:".to_string());
            lines.push("# budget = 96.0".to_string());
        }
    }
    lines.push("margin = 4.0".to_string());
    let content = lines.join("\n") + "\n";

    if file.exists() {
        if json {
            let doc = report::SetupExists {
                status: "exists",
                path: file.display().to_string(),
            };
            println!("{}", serde_json::to_string(&doc)?);
        } else {
            ui::warn("llama-matrix.toml already exists — not overwriting; setup would write:");
            print!("{content}");
        }
        return Ok(());
    }

    std::fs::write(&file, &content).with_context(|| format!("writing {}", file.display()))?;
    if json {
        let doc = report::SetupWritten {
            written: file.display().to_string(),
            config: config_path.clone(),
            endpoint: endpoint.clone(),
            budget,
            gpu: gpu_label.clone(),
        };
        println!("{}", serde_json::to_string(&doc)?);
    } else {
        ui::ok(&format!("wrote {}", file.display()));
        if let Some(label) = &gpu_label {
            ui::info(&format!(
                "detected {label} (~{} GB)",
                budget.map(|total| format!("{total:.1}")).unwrap_or_default()
            ));
        }
        if config_path.is_none() {
            ui::warn("no config.yaml found — set `config` in llama-matrix.toml or pass --config");
        }
        ui::info("next: `llama-matrix measure`  →  `llama-matrix build`");
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
    use llama_matrix_core::measure::{
        sweep, MeasureOptions, DEFAULT_LOAD_TIMEOUT, DEFAULT_TRIGGER_TIMEOUT,
    };

    let policy = Policy::load(PathBuf::from("llama-matrix.toml"))?;
    let config_dir = std::env::current_dir()?;
    let config_path = config
        .or_else(|| policy.config.clone())
        .unwrap_or_else(|| "config.yaml".to_string());
    let parsed = ls_config::parse_file(&config_path)?;
    let store = open_store(&config_dir, json)?;

    let endpoint = endpoint.unwrap_or_else(|| policy.endpoint.clone());
    let only = only.map(|list| list.split(',').map(|id| id.trim().to_string()).collect::<Vec<_>>());
    let options = MeasureOptions {
        endpoint,
        force,
        only,
        load_timeout: DEFAULT_LOAD_TIMEOUT,
        trigger_timeout: DEFAULT_TRIGGER_TIMEOUT,
        probe_image_size: policy.probe_image_size.clone(),
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
        // The summary *is* the report (collect/render split, D16): serialize it.
        println!("{}", serde_json::to_string(&summary)?);
    } else {
        let headline = format!(
            "baseline {:.2} GB · pool {:.1} GB · measured {} · cached {} · failed {} · missing {}",
            summary.baseline,
            summary.detected_total,
            summary.measured.len(),
            summary.cached.len(),
            summary.failed.len(),
            summary.skipped_missing.len()
        );
        // Failure-aware headline (D3): a green ✓ must not sit atop a run that
        // failed. When any model failed or was skipped, flag the headline with ⚠
        // (glyph + the counts in the text), since the exit status stays 0.
        // An unconfirmable serving check does *not* escalate the headline: having no
        // /props is a permanent property of a backend (image, STT), so every sweep on
        // such a roster would carry a ⚠ and train the operator to ignore it. It gets
        // its own warning line below instead.
        //
        // An *unconfirmed allocation* does escalate: unlike the serving check it is
        // clearable, it means a recorded footprint may be short, and `build` will pack
        // it by default, so it is exactly the case that must not look clean.
        if summary.failed.is_empty()
            && summary.skipped_missing.is_empty()
            && summary.unconfirmed_allocation.is_empty()
        {
            ui::ok(&headline);
        } else {
            ui::alert(&headline);
        }
        for failure in &summary.failed {
            ui::warn(&format!("{}: {}", failure.id, failure.reason));
        }
        for id in &summary.skipped_missing {
            ui::warn(&format!("{id}: weight file missing — skipped"));
        }
        for unconfirmed in &summary.unconfirmed_allocation {
            ui::warn(&format!(
                "{}: recorded WITHOUT confirming the allocation finished - {}",
                unconfirmed.id, unconfirmed.reason
            ));
        }
        for suspect in &summary.below_weight_floor {
            ui::warn(&format!("{}: {}", suspect.id, suspect.reason));
        }
        if !summary.unverified_serving.is_empty() {
            ui::warn(&format!(
                "{} model(s) recorded without confirming llama-swap loaded the measured \
                 command (no /props on that backend): {}",
                summary.unverified_serving.len(),
                summary.unverified_serving.join(", ")
            ));
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
                let doc = report::ConfigValue { key, value: display };
                println!("{}", serde_json::to_string(&doc)?);
            } else {
                ui::ok(&format!("{key} = {display}"));
            }
        }
        ConfigureAction::Unset { key } => {
            settings::unset(&file, &key)?;
            if json {
                let doc = report::ConfigUnset { unset: key };
                println!("{}", serde_json::to_string(&doc)?);
            } else {
                ui::ok(&format!("unset {key} (reverted to default)"));
            }
        }
        ConfigureAction::Get { key } => {
            let value = settings::get(&file, &key)?;
            if json {
                let doc = report::ConfigValue { key, value };
                println!("{}", serde_json::to_string(&doc)?);
            } else {
                println!("{value}");
            }
        }
        ConfigureAction::List => {
            let effective = settings::list(&file);
            if json {
                // A key -> effective-value map (BTreeMap: deterministic, sorted).
                let map: std::collections::BTreeMap<&str, String> = effective
                    .iter()
                    .map(|(key, value)| (*key, value.clone()))
                    .collect();
                println!("{}", serde_json::to_string(&map)?);
            } else {
                for (key, value) in effective {
                    println!("{key} = {value}");
                }
            }
        }
        ConfigureAction::Keys => {
            if json {
                let entries: Vec<report::SettingInfo> = settings::SETTINGS
                    .iter()
                    .map(|setting| report::SettingInfo {
                        key: setting.key,
                        desc: setting.desc,
                        default: setting.default,
                    })
                    .collect();
                println!("{}", serde_json::to_string(&entries)?);
            } else {
                for setting in settings::SETTINGS {
                    println!("{:<16} {}  (default: {})", setting.key, setting.desc, setting.default);
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
    no_verify: bool,
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

    let store = open_store(&config_dir, json)?;
    if store.list_ids().is_empty() {
        anyhow::bail!(
            "no measurements in {} — run `llama-matrix measure` first",
            store.dir().display()
        );
    }

    let plan = build::resolve_plan(&config_path, &policy, budget, &store)?;
    let block = matrix::render(&plan);

    if !json {
        for warning in &plan.warnings {
            ui::warn(warning);
        }
    }

    if apply {
        let result = apply::apply(Path::new(&config_path), &block, &policy.endpoint, !no_verify)?;
        if json {
            let doc = report::BuildApplied {
                applied: true,
                backup: result.backup.display().to_string(),
                verified: result.verified,
                note: result.note.clone(),
                packs: plan.n_packs,
                heavies: plan.n_heavies,
                sets: plan.sets.len(),
                unconfirmed: plan.unconfirmed.clone(),
            };
            println!("{}", serde_json::to_string(&doc)?);
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
            let doc = report::BuildWrote { wrote: out_path.clone(), sets: plan.sets.len() };
            println!("{}", serde_json::to_string(&doc)?);
        } else {
            ui::ok(&format!("wrote {out_path} ({} sets)", plan.sets.len()));
        }
        return Ok(());
    }

    if json {
        let doc = report::BuildPreview::of(&plan);
        println!("{}", serde_json::to_string(&doc)?);
    } else {
        print!("{block}");
    }
    Ok(())
}

/// The `--llm` guide: the auto-generated command reference (from the one clap
/// definition, so the command surface can never drift from the real binary)
/// followed by the design docs, embedded at compile time so they never drift from
/// the shipped binary (a doc change ships in the same commit). See DECISIONS.md D1.
fn llm_guide() -> String {
    let mut out = String::new();
    out.push_str("# llama-matrix LLM guide\n\n");
    out.push_str(
        "The full machine-readable reference: the command surface (auto-generated from \
         the CLI, so it always matches the real binary) followed by the repository \
         design docs, embedded at build time.\n\n",
    );

    // The command reference, rendered from the single clap definition: the root
    // long help plus each visible subcommand's, so `--llm` doubles as a `man` page
    // and the set of commands/flags can never drift from the actual CLI.
    let mut command = Cli::command();
    command.build();
    out.push_str("===== COMMANDS =====\n\n");
    out.push_str(&command.render_long_help().to_string());
    for sub in command.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }
        out.push_str(&format!("\n\n----- {} -----\n\n", sub.get_name()));
        out.push_str(&sub.clone().render_long_help().to_string());
    }

    let sections: &[(&str, &str)] = &[
        ("README", include_str!("../../../README.md")),
        ("WORKFLOWS", include_str!("../../../WORKFLOWS.md")),
        ("SPEC", include_str!("../../../SPEC.md")),
        ("ARCHITECTURE", include_str!("../../../ARCHITECTURE.md")),
        ("PRINCIPLES", include_str!("../../../PRINCIPLES.md")),
        ("ROADMAP", include_str!("../../../ROADMAP.md")),
        ("ADOPT", include_str!("../../../ADOPT.md")),
    ];
    for (name, body) in sections {
        out.push_str(&format!("\n\n===== {name} =====\n\n"));
        out.push_str(body);
    }
    out
}

#[cfg(test)]
mod doc_tests {
    //! Docs-can't-drift guards. The design docs are embedded into `--llm` and are
    //! load-bearing, so a rename or new capability that skips them is a defect. If
    //! a settable key isn't in SPEC.md, or a command isn't in the README table,
    //! the build fails here rather than shipping a stale guide (the amdl/dotsync
    //! `workflows_doc` pattern).
    use super::Cli;
    use clap::CommandFactory;
    use llama_matrix_core::settings;

    const SPEC: &str = include_str!("../../../SPEC.md");
    const README: &str = include_str!("../../../README.md");

    #[test]
    fn every_setting_is_documented_in_spec() {
        for setting in settings::SETTINGS {
            assert!(
                SPEC.contains(setting.key),
                "SPEC.md never mentions the `{}` setting - document it or drop the key",
                setting.key
            );
        }
    }

    /// The store schema is a published contract (SPEC §2 shows it field by field),
    /// and a field nobody documents is how the per-pool split came to be described
    /// as recorded while nothing wrote it. Serialize a fully populated measurement
    /// and require every key to appear in SPEC.md.
    #[test]
    fn every_measurement_field_is_documented_in_spec() {
        let populated = llama_matrix_core::cache::Measurement {
            status: "ok".into(),
            d_total: 49.05,
            d_vram: Some(48.77),
            d_gtt: Some(0.27),
            abs_total: 49.21,
            abs_vram: Some(48.92),
            abs_gtt: Some(0.29),
            load_s: 42.0,
            allocation_confirmed: Some(true),
            serving_verified: Some(true),
            peak_total: Some(49.60),
            weights_gb: Some(49.90),
            params: "/app/llama-server -m /m.gguf -c 4096".into(),
            measured_at: "2026-01-01".into(),
        };
        let serde_json::Value::Object(fields) = serde_json::to_value(&populated).unwrap() else {
            panic!("a measurement serializes to a JSON object");
        };
        for key in fields.keys() {
            assert!(
                SPEC.contains(&format!("\"{key}\"")),
                "SPEC.md §2 never shows the `{key}` measurement field - document it or drop it"
            );
        }
    }

    #[test]
    fn every_command_is_documented_in_readme() {
        for sub in Cli::command().get_subcommands() {
            let name = sub.get_name();
            // `completions` is plumbing (Homebrew calls it), not a user-facing verb.
            if name == "completions" {
                continue;
            }
            assert!(
                README.contains(name),
                "README.md never mentions the `{name}` command - add it to the command table"
            );
        }
    }
}
