//! Parse a llama-swap `config.yaml` into classified model records.
//!
//! Order of operations matters (see SPEC.md §5): the `macros:` section and the
//! reserved `${…}` substitutions are expanded **before** deriving type, primary
//! file, or param-hash — hashing or stat-ing an unexpanded `${…}` placeholder is
//! a bug. A macro-free config expands to itself, so this is always safe.
//!
//! Unknown keys are tolerated (serde ignores them): llama-swap has many
//! model-level keys llama-matrix doesn't consume. Entries without a runnable
//! `cmd` (selectors / virtual model ids) are skipped from the worklist.

use std::path::Path;

use anyhow::{Context, Result};
use indexmap::IndexMap;
use regex::Regex;
use serde::Deserialize;

use crate::model::ModelRecord;

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    macros: IndexMap<String, String>,
    #[serde(default)]
    models: IndexMap<String, RawModel>,
}

#[derive(Debug, Deserialize)]
struct RawModel {
    #[serde(default)]
    cmd: Option<String>,
}

/// The parsed roster plus the raw macro table (kept for reference/debugging).
pub struct ParsedConfig {
    pub models: Vec<ModelRecord>,
    pub macros: IndexMap<String, String>,
}

/// Parse a config from a YAML string.
pub fn parse_str(yaml: &str) -> Result<ParsedConfig> {
    let raw: RawConfig = serde_yaml::from_str(yaml).context("parsing llama-swap config YAML")?;
    let mut models = Vec::new();
    for (id, m) in &raw.models {
        let cmd = match &m.cmd {
            Some(c) if !c.trim().is_empty() => c,
            // No runnable command → a selector / virtual id → not in the worklist.
            _ => continue,
        };
        let normalized = normalize_ws(cmd);
        let expanded = expand_macros(&normalized, &raw.macros, id);
        models.push(ModelRecord::from_expanded(id.clone(), expanded));
    }
    Ok(ParsedConfig {
        models,
        macros: raw.macros,
    })
}

/// Parse a config from a file path.
pub fn parse_file(path: impl AsRef<Path>) -> Result<ParsedConfig> {
    let path = path.as_ref();
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("reading llama-swap config {}", path.display()))?;
    parse_str(&s)
}

/// Collapse all whitespace (folded/literal YAML scalars, newlines) to single
/// spaces and trim — the command becomes one shell line.
fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Expand llama-swap macros and reserved substitutions in a command string.
///
/// Handles: user `${macro}` references (from the `macros:` map, recursively),
/// `${env.VAR}` (from the process environment), and `${MODEL_ID}`. The runtime-
/// assigned `${PORT}`/`${PID}` are left as-is — they only ever appear in
/// footprint-irrelevant positions (stripped by the param-hash) and their real
/// values aren't known outside a live launch.
fn expand_macros(s: &str, macros: &IndexMap<String, String>, model_id: &str) -> String {
    let mut cur = s.to_string();
    // Multi-pass so a macro whose value contains another macro resolves; bounded
    // to avoid an infinite loop on a self-referential macro.
    for _ in 0..16 {
        let next = substitute_once(&cur, macros, model_id);
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

fn substitute_once(s: &str, macros: &IndexMap<String, String>, model_id: &str) -> String {
    let re = Regex::new(r"\$\{([^}]+)\}").expect("static regex");
    re.replace_all(s, |caps: &regex::Captures| {
        let key = &caps[1];
        if let Some(var) = key.strip_prefix("env.") {
            std::env::var(var).unwrap_or_default()
        } else if key == "MODEL_ID" {
            model_id.to_string()
        } else if let Some(v) = macros.get(key) {
            v.clone()
        } else {
            // PORT / PID / unknown — leave the placeholder untouched.
            caps[0].to_string()
        }
    })
    .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelType;

    const CONFIG: &str = r#"
macros:
  server: "/app/llama-server -ngl 99 -fa on"
  mdir: "/models"
models:
  "chat":
    proxy: "http://127.0.0.1:9001"
    cmd: >
      ${server} -m ${mdir}/chat.gguf
      --host 127.0.0.1 --port 9001 -c 4096 --jinja
    ttl: 0
  "embed":
    cmd: "/app/llama-server -m /models/e.gguf --embedding --pooling last -c 8192"
  "router":
    proxy: "http://127.0.0.1:9999"
"#;

    #[test]
    fn parses_expands_and_classifies() {
        let p = parse_str(CONFIG).unwrap();
        // "router" has no cmd → excluded (selector-like).
        assert_eq!(p.models.len(), 2);

        let chat = &p.models[0];
        assert_eq!(chat.id, "chat");
        // Macros expanded, whitespace normalized to one line.
        assert_eq!(
            chat.cmd,
            "/app/llama-server -ngl 99 -fa on -m /models/chat.gguf --host 127.0.0.1 --port 9001 -c 4096 --jinja"
        );
        assert_eq!(chat.model_type, ModelType::Llm);
        assert_eq!(chat.primary_file, Some("/models/chat.gguf".to_string()));

        let embed = &p.models[1];
        assert_eq!(embed.model_type, ModelType::Embed);
    }

    #[test]
    fn env_and_model_id_substitution() {
        std::env::set_var("LM_TEST_MODELS", "/srv/w");
        let cfg = r#"
models:
  "m1":
    cmd: "/app/llama-server -m ${env.LM_TEST_MODELS}/${MODEL_ID}.gguf -c 2048"
"#;
        let p = parse_str(cfg).unwrap();
        assert_eq!(
            p.models[0].primary_file,
            Some("/srv/w/m1.gguf".to_string())
        );
    }
}
