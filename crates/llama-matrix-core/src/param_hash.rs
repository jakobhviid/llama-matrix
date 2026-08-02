//! The memory **param-hash**: a stable key over only the footprint-affecting
//! tokens of a launch command.
//!
//! A model can carry several footprints over its life (a re-quant, a context or
//! parallelism change). Each measurement is keyed by this hash, so flipping a
//! flag that does *not* affect memory (host/port, reasoning toggle, sampler knobs)
//! never invalidates a cached measurement, while a real change (`-c`/`-np`/quant)
//! produces a new key and a new measurement alongside the old one.
//!
//! The strip-list is a **conservative allowlist of flags known not to affect the
//! footprint**. Everything else stays in the hash. The risk direction is fixed by
//! design: an unlisted-but-irrelevant flag costs only a harmless extra measure —
//! never a wrong cache hit (which would under-count the matrix and could OOM).
//! When unsure whether a flag affects memory, do not add it here.

use sha1::{Digest, Sha1};

/// Flags whose following value token is also dropped (neither affects VRAM).
const STRIP_WITH_VALUE: &[&str] = &[
    "--host",
    "--port",
    "--listen-ip",
    "--listen-port",
    "--inference-path",
    "--reasoning",
    "--chat-template-file",
    "--cache-reuse",
    // stable-diffusion.cpp runtime knobs (do not change the resident footprint):
    "--steps",
    "--cfg-scale",
    "--guidance",
    "--cache-mode",
    "--cache-option",
];

/// Bare flags (no value) that do not affect VRAM.
const STRIP_BARE: &[&str] = &["--jinja"];

/// Reduce a launch command to only its footprint-affecting tokens.
pub fn memory_cmd(cmd: &str) -> String {
    let toks: Vec<&str> = cmd.split_whitespace().collect();
    let mut out: Vec<&str> = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        if STRIP_WITH_VALUE.contains(&t) {
            i += 2; // flag consumes its value
            continue;
        }
        if STRIP_BARE.contains(&t) {
            i += 1;
            continue;
        }
        out.push(t);
        i += 1;
    }
    out.join(" ")
}

/// A 12-hex key identifying a distinct memory footprint (sha1 of `memory_cmd`,
/// truncated to 12 hex chars — matches the reference tooling's key format).
pub fn param_hash(cmd: &str) -> String {
    let mut h = Sha1::new();
    h.update(memory_cmd(cmd).as_bytes());
    let digest = h.finalize();
    let mut hex = String::with_capacity(12);
    for b in digest.iter().take(6) {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_port_and_reasoning() {
        // A "-nothink" twin differs only by `--reasoning off` (+ a different port);
        // both are stripped, so it hashes equal to its base — no separate measure.
        let base = "/app/llama-server -m /models/g.gguf --host 127.0.0.1 --port 9001 \
                    -ngl 99 -c 262144 -np 2 -fa on --jinja";
        let nothink = "/app/llama-server -m /models/g.gguf --host 127.0.0.1 --port 9006 \
                       -ngl 99 -c 262144 -np 2 -fa on --reasoning off --jinja";
        assert_eq!(param_hash(base), param_hash(nothink));
        assert_eq!(
            memory_cmd(base),
            "/app/llama-server -m /models/g.gguf -ngl 99 -c 262144 -np 2 -fa on"
        );
    }

    #[test]
    fn distinguishes_np_and_context() {
        // Same weights file, different `-np`/`-c` -> different footprint -> different hash.
        let next = "/app/llama-server -m /models/coder.gguf -ngl 99 -c 262144 -np 2 -fa on";
        let ultra = "/app/llama-server -m /models/coder.gguf -ngl 99 -c 1572864 -np 6 -fa on";
        assert_ne!(param_hash(next), param_hash(ultra));
    }

    #[test]
    fn hash_is_twelve_hex() {
        let h = param_hash("/app/llama-server -m /x.gguf -c 4096");
        assert_eq!(h.len(), 12);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn quant_change_changes_hash() {
        let q4 = "/app/llama-server -m /models/Cydonia-Q4_K_L.gguf -c 0 -fa on";
        let q6 = "/app/llama-server -m /models/Cydonia-Q6_K.gguf -c 0 -fa on";
        assert_ne!(param_hash(q4), param_hash(q6));
    }
}
