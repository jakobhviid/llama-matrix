//! Splice the generated `matrix:` block into the live llama-swap `config.yaml`,
//! then let `-watch-config` hot-reload it. The live config is written in exactly
//! one place — here (Principle #5): always after a backup, always anchored on the
//! generated marker, always with a basic post-write liveness check.
//!
//! llama-swap rejects an invalid config and keeps the old one, so a splice can't
//! silently break the service; the guarantee we own is that `build` only emits a
//! *fitting* block, and that the splice is textually clean.

use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use anyhow::{bail, Context, Result};

/// Where the block-replacing splice was anchored, and whether the reload verified.
pub struct ApplyResult {
    pub backup: PathBuf,
    pub verified: bool,
    pub note: String,
}

/// The byte offset in `config_text` where the generated block should start,
/// replacing everything from there to end-of-file.
///
/// Anchor priority: a previously-generated marker (matches both this tool's and
/// the reference tool's `# ==== GENERATED matrix block …` line) → a top-level
/// `matrix:` block → a top-level `groups:` block → append at the end.
fn find_anchor(config_text: &str) -> usize {
    if let Some(offset) = line_offset(config_text, |line| {
        line.trim_start().starts_with("# ==== GENERATED matrix block")
    }) {
        return offset;
    }
    if let Some(offset) = line_offset(config_text, |line| is_top_level_key(line, "matrix")) {
        return offset;
    }
    if let Some(offset) = line_offset(config_text, |line| is_top_level_key(line, "groups")) {
        return offset;
    }
    config_text.len()
}

/// The byte offset of the first line satisfying `predicate` (line without its `\n`).
fn line_offset(text: &str, predicate: impl Fn(&str) -> bool) -> Option<usize> {
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if predicate(line.trim_end_matches('\n')) {
            return Some(offset);
        }
        offset += line.len();
    }
    None
}

/// A top-level `key:` line has no leading indentation.
fn is_top_level_key(line: &str, key: &str) -> bool {
    !line.starts_with([' ', '\t']) && (line == format!("{key}:") || line.starts_with(&format!("{key}:")))
}

/// Produce the new config text: everything up to the anchor, then the block. The
/// `matrix:` block must be the last top-level block, so nothing follows it.
pub fn splice(config_text: &str, block: &str) -> String {
    let anchor = find_anchor(config_text);
    let head = config_text[..anchor].trim_end();
    if head.is_empty() {
        block.to_string()
    } else {
        format!("{head}\n\n{block}")
    }
}

/// Back up the config, splice in the block, and check llama-swap stays live. On a
/// dead endpoint after the write, roll back to the backup.
pub fn apply(config_path: &Path, block: &str, endpoint: &str) -> Result<ApplyResult> {
    let original = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let backup = PathBuf::from(format!("{}.pre-matrix.bak", config_path.display()));
    std::fs::write(&backup, &original)
        .with_context(|| format!("writing backup {}", backup.display()))?;

    let spliced = splice(&original, block);
    std::fs::write(config_path, &spliced)
        .with_context(|| format!("writing {}", config_path.display()))?;

    let agent = ureq::builder().timeout(Duration::from_secs(8)).build();
    let models_url = format!("{endpoint}/v1/models");
    if agent.get(&models_url).call().is_err() {
        return Ok(ApplyResult {
            backup,
            verified: false,
            note: "wrote config; endpoint unreachable — verify the reload manually".to_string(),
        });
    }

    // Give -watch-config time to poll + reload, then confirm it still serves. A
    // rejected config leaves the old one live (still 200), so this proves the
    // service survived, not that the new block parsed — check the logs to be sure.
    thread::sleep(Duration::from_secs(3));
    match agent.get(&models_url).call() {
        Ok(response) if response.status() == 200 => Ok(ApplyResult {
            backup,
            verified: true,
            note: "spliced; llama-swap reloaded and is serving".to_string(),
        }),
        _ => {
            std::fs::write(config_path, &original).with_context(|| {
                format!("rolling back {}", config_path.display())
            })?;
            bail!("llama-swap stopped serving after the write — rolled back to the backup");
        }
    }
}

/// The current generated / `matrix:` block in a config (from the marker or a
/// top-level `matrix:` line to end-of-file), or None if there is no matrix block
/// yet. Used by `drift` to compare the live block against a fresh build.
pub fn existing_block(config_text: &str) -> Option<String> {
    let has_marker = config_text
        .lines()
        .any(|line| line.trim_start().starts_with("# ==== GENERATED matrix block"));
    let has_matrix = config_text.lines().any(|line| is_top_level_key(line, "matrix"));
    if !has_marker && !has_matrix {
        return None;
    }
    let anchor = find_anchor(config_text);
    if anchor >= config_text.len() {
        return None;
    }
    Some(config_text[anchor..].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: &str = "# ==== GENERATED matrix block (llama-matrix) ====\nmatrix:\n  sets:\n    aux: \"e\"\n";

    #[test]
    fn replaces_a_previous_generated_block() {
        let config = "models:\n  \"a\":\n    cmd: x\n\n# ==== GENERATED matrix block (replaces the entire `groups:` block) ====\n# old header\nmatrix:\n  sets:\n    old: \"z\"\n";
        let out = splice(config, BLOCK);
        assert!(out.contains("models:"));
        assert!(!out.contains("old header"), "stale block must be gone");
        assert!(!out.contains("old: \"z\""));
        assert!(out.trim_end().ends_with("aux: \"e\""));
        // exactly one generated marker
        assert_eq!(out.matches("# ==== GENERATED matrix block").count(), 1);
    }

    #[test]
    fn replaces_a_top_level_matrix_block_without_marker() {
        let config = "models:\n  \"a\":\n    cmd: x\nmatrix:\n  sets:\n    old: \"z\"\n";
        let out = splice(config, BLOCK);
        assert!(out.contains("models:"));
        assert!(!out.contains("old: \"z\""));
        assert!(out.contains("aux: \"e\""));
    }

    #[test]
    fn replaces_a_groups_block() {
        let config = "models:\n  \"a\":\n    cmd: x\ngroups:\n  g1:\n    exclusive: true\n";
        let out = splice(config, BLOCK);
        assert!(!out.contains("exclusive"));
        assert!(out.contains("matrix:"));
    }

    #[test]
    fn appends_when_no_anchor_present() {
        let config = "models:\n  \"a\":\n    cmd: x\n";
        let out = splice(config, BLOCK);
        assert!(out.starts_with("models:"));
        assert!(out.contains("\n\n# ==== GENERATED matrix block"));
        assert!(out.contains("aux: \"e\""));
    }

    #[test]
    fn does_not_match_an_indented_matrix_key() {
        // a `matrix:` nested under a model must NOT be treated as the anchor
        let config = "models:\n  \"a\":\n    matrix: something\n";
        let out = splice(config, BLOCK);
        assert!(out.contains("matrix: something")); // preserved
        assert!(out.trim_end().ends_with("aux: \"e\""));
    }
}
