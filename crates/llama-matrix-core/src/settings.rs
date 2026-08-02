//! The `llama-matrix configure` surface — a small, validated set of *scalar*
//! settings in `llama-matrix.toml`, exposed as get/set/unset/list/keys so they're
//! discoverable (and shell-completable) instead of hand-edited guesswork.
//!
//! Only scalars live here; structured tables (`[paths]`, `[roles]`, `[groups]`)
//! stay hand-edited. `SETTINGS` is the single source of truth — it drives
//! validation, `list`, `keys`, and completion. Writes are comment-preserving
//! (`toml_edit`), so hand-written comments and structured tables survive.

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

/// The value domain of a setting — decides how a value is parsed/validated.
pub enum Kind {
    /// A free string (e.g. a URL).
    Str,
    /// A floating-point number of GB.
    Float,
    /// One of a fixed set of lowercase words.
    Enum(&'static [&'static str]),
}

/// One settable scalar: its key (mirrors the `llama-matrix.toml` key), value
/// domain, one-line description, and the display shown when it's unset.
pub struct Setting {
    pub key: &'static str,
    pub kind: Kind,
    pub desc: &'static str,
    pub default: &'static str,
}

/// Every key `llama-matrix configure` knows.
pub const SETTINGS: &[Setting] = &[
    Setting {
        key: "config",
        kind: Kind::Str,
        desc: "path to the llama-swap config.yaml (also written by `setup`)",
        default: "(./config.yaml)",
    },
    Setting {
        key: "endpoint",
        kind: Kind::Str,
        desc: "llama-swap base URL",
        default: "http://localhost:8080",
    },
    Setting {
        key: "budget",
        kind: Kind::Float,
        desc: "GB to plan against (unset = auto-detect the pool)",
        default: "(auto-detect)",
    },
    Setting {
        key: "margin",
        kind: Kind::Float,
        desc: "GB safety slack inside the budget",
        default: "4.0",
    },
    Setting {
        key: "strategy",
        kind: Kind::Enum(&["flat", "family"]),
        desc: "packing strategy",
        default: "flat",
    },
    Setting {
        key: "on_overflow",
        kind: Kind::Enum(&["group", "error"]),
        desc: "1000-combination cap handling",
        default: "group",
    },
];

/// The settable keys (also the source for shell completion).
pub fn keys() -> Vec<&'static str> {
    SETTINGS.iter().map(|setting| setting.key).collect()
}

fn find(key: &str) -> Result<&'static Setting> {
    SETTINGS
        .iter()
        .find(|setting| setting.key == key)
        .ok_or_else(|| anyhow!("unknown setting `{key}` — run `llama-matrix configure keys`"))
}

/// Validate + normalize a value, returning its display form and the toml item.
fn normalize(setting: &Setting, value: &str) -> Result<(String, toml_edit::Item)> {
    match &setting.kind {
        Kind::Str => Ok((value.to_string(), toml_edit::value(value))),
        Kind::Float => {
            let number: f64 = value
                .trim()
                .parse()
                .with_context(|| format!("`{}` expects a number, got `{value}`", setting.key))?;
            Ok((format!("{number}"), toml_edit::value(number)))
        }
        Kind::Enum(allowed) => {
            let chosen = value.trim().to_lowercase();
            if !allowed.contains(&chosen.as_str()) {
                bail!("`{}` expects one of {:?}, got `{value}`", setting.key, allowed);
            }
            Ok((chosen.clone(), toml_edit::value(chosen)))
        }
    }
}

/// Set `key` to `value` in `llama-matrix.toml` (comment-preserving), returning the
/// normalized display value.
pub fn set(file: &Path, key: &str, value: &str) -> Result<String> {
    let setting = find(key)?;
    let (display, item) = normalize(setting, value)?;
    let text = std::fs::read_to_string(file).unwrap_or_default();
    let mut document: toml_edit::DocumentMut = text
        .parse()
        .with_context(|| format!("parsing {}", file.display()))?;
    document.as_table_mut().insert(key, item);
    std::fs::write(file, document.to_string())
        .with_context(|| format!("writing {}", file.display()))?;
    Ok(display)
}

/// Remove `key`'s override (revert to its default). A no-op if it isn't set.
pub fn unset(file: &Path, key: &str) -> Result<()> {
    find(key)?;
    let text = std::fs::read_to_string(file).unwrap_or_default();
    let mut document: toml_edit::DocumentMut = text
        .parse()
        .with_context(|| format!("parsing {}", file.display()))?;
    document.as_table_mut().remove(key);
    std::fs::write(file, document.to_string())
        .with_context(|| format!("writing {}", file.display()))?;
    Ok(())
}

/// The effective (file value, else default) display value of one setting.
pub fn get(file: &Path, key: &str) -> Result<String> {
    let setting = find(key)?;
    let text = std::fs::read_to_string(file).unwrap_or_default();
    Ok(effective(&text, setting))
}

/// Every setting with its effective value — backs `list`.
pub fn list(file: &Path) -> Vec<(&'static str, String)> {
    let text = std::fs::read_to_string(file).unwrap_or_default();
    SETTINGS
        .iter()
        .map(|setting| (setting.key, effective(&text, setting)))
        .collect()
}

fn effective(text: &str, setting: &Setting) -> String {
    let parsed = text.parse::<toml::Value>().ok();
    let raw = parsed.as_ref().and_then(|table| table.get(setting.key));
    match &setting.kind {
        Kind::Float => raw
            .and_then(|value| value.as_float())
            .map(|number| format!("{number}"))
            .unwrap_or_else(|| setting.default.to_string()),
        _ => raw
            .and_then(|value| value.as_str().map(str::to_string))
            .unwrap_or_else(|| setting.default.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_unset_roundtrip_preserving_comments() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("llama-matrix.toml");
        std::fs::write(&file, "# hand-written header\nmargin = 4.0\n").unwrap();

        assert_eq!(get(&file, "budget").unwrap(), "(auto-detect)");
        set(&file, "budget", "50").unwrap();
        assert_eq!(get(&file, "budget").unwrap(), "50");
        set(&file, "strategy", "FAMILY").unwrap(); // case-insensitive
        assert_eq!(get(&file, "strategy").unwrap(), "family");

        // the hand-written comment survives a comment-preserving edit
        assert!(std::fs::read_to_string(&file).unwrap().contains("# hand-written header"));

        unset(&file, "budget").unwrap();
        assert_eq!(get(&file, "budget").unwrap(), "(auto-detect)");
    }

    #[test]
    fn rejects_bad_key_and_values() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("llama-matrix.toml");
        assert!(set(&file, "nope", "x").is_err());
        assert!(set(&file, "strategy", "bogus").is_err());
        assert!(set(&file, "margin", "lots").is_err());
    }
}
