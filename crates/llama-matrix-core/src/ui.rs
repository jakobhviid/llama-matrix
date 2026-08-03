//! Output discipline so `--json` stays pipe-clean: human output → stdout,
//! progress + status → stderr. Plus a tiny ANSI palette for human renderers.
//!
//! Colour is emitted only when stderr is a real terminal AND `NO_COLOR` is unset
//! — so a redirect / pipe (including the `--json` path, which never calls the
//! `info`/`warn`/`err` helpers) stays clean. The decision is computed once.

use std::io::IsTerminal;
use std::sync::OnceLock;

fn color_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal())
}

fn paint(code: &str, s: &str) -> String {
    if color_enabled() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn green(s: &str) -> String {
    paint("1;32", s)
}
pub fn red(s: &str) -> String {
    paint("1;31", s)
}
pub fn yellow(s: &str) -> String {
    paint("1;33", s)
}
pub fn cyan(s: &str) -> String {
    paint("1;36", s)
}
pub fn bold(s: &str) -> String {
    paint("1", s)
}
pub fn dim(s: &str) -> String {
    paint("2", s)
}

/// Informational progress line (stderr).
pub fn info(msg: &str) {
    eprintln!("{msg}");
}

/// A warning (stderr), yellow-flagged.
pub fn warn(msg: &str) {
    eprintln!("{} {msg}", yellow("warning:"));
}

/// An error line (stderr), red-flagged. (Callers still return the error too.)
pub fn err(msg: &str) {
    eprintln!("{} {msg}", red("error:"));
}

/// A success line (stderr), green check.
pub fn ok(msg: &str) {
    eprintln!("{} {msg}", green("✓"));
}

/// A result headline flagging a problem (stderr): yellow `⚠` + message. Use this
/// instead of `ok` for a per-item-outcome command's headline whenever any item
/// failed, so the failure is salient in the glyph and text (not colour alone) even
/// though the exit status stays 0 (see ../../PRINCIPLES.md and DECISIONS.md D3).
pub fn alert(msg: &str) {
    eprintln!("{} {msg}", yellow("⚠"));
}
