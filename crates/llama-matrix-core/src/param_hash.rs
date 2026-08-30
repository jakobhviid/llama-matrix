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
//! design: an unlisted-but-irrelevant flag costs only a harmless extra measure -
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
    // llama.cpp's host-side prompt cache cap. It moves HOST RAM, never the GPU, and
    // `build` reads the cap straight from the live command rather than from a
    // measurement - so a footprint taken at one `-cram` describes the GPU at any
    // other, and re-measuring on a change would record the identical number. This is
    // the one entry justified by the flag being memory-affecting on an axis the
    // measurement does not carry, rather than by it being memory-neutral.
    "-cram",
    "--cache-ram",
    // stable-diffusion.cpp sampling knobs: they change the work done per step, not
    // what stays resident.
    "--steps",
    "--cfg-scale",
    "--guidance",
    // Deliberately NOT stripped, though they look like siblings of the above:
    // `--cache-mode` / `--cache-option` (sd-server's easycache). A step cache holds
    // intermediate tensors on the compute device, so toggling it - or changing its
    // threshold, which changes how much is cached - plausibly changes the resident
    // footprint. Under the rule in the module docs an uncertain flag stays in the
    // hash: the cost is one extra measurement, never a wrong reuse.
];

/// Bare flags (no value) that do not affect VRAM.
const STRIP_BARE: &[&str] = &["--jinja"];

/// Reduce a launch command to only its footprint-affecting tokens.
pub fn memory_cmd(cmd: &str) -> String {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let mut kept: Vec<&str> = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        let token = tokens[index];
        if STRIP_WITH_VALUE.contains(&token) {
            index += 2; // flag consumes its value
            continue;
        }
        if STRIP_BARE.contains(&token) {
            index += 1;
            continue;
        }
        kept.push(token);
        index += 1;
    }
    kept.join(" ")
}

/// The tokens of `left` that `right` does not have, rendered for a message.
///
/// A whole 20-token command on each side of a difference buries the one flag that
/// moved; what changed is the part anyone can act on. `(nothing)` when the two sides
/// differ only by the *other* direction's extra tokens.
///
/// **A differing value carries its flag.** Two commands that differ only in context
/// share the token `-c`, so a plain token diff reports `131072` with no way to tell
/// what it sets. Where a differing token follows a flag, the flag comes with it; where
/// a run of tokens differs, they are kept together as one phrase.
pub fn token_difference(left: &str, right: &str) -> String {
    let left_tokens: Vec<&str> = left.split_whitespace().collect();
    let mut remaining: Vec<&str> = right.split_whitespace().collect();
    let mut only_in_left: Vec<String> = Vec::new();
    let mut previous_differed = false;

    for (index, token) in left_tokens.iter().enumerate() {
        if let Some(at) = remaining.iter().position(|other| other == token) {
            remaining.remove(at);
            previous_differed = false;
            continue;
        }
        let preceding_flag = index
            .checked_sub(1)
            .and_then(|before| left_tokens.get(before))
            .filter(|before| is_flag(before));
        match (previous_differed, preceding_flag) {
            // A run of differing tokens is one phrase (`--spec-type draft-mtp`).
            (true, _) => {
                if let Some(last) = only_in_left.last_mut() {
                    last.push(' ');
                    last.push_str(token);
                }
            }
            (false, Some(flag)) => only_in_left.push(format!("{flag} {token}")),
            (false, None) => only_in_left.push((*token).to_string()),
        }
        previous_differed = true;
    }

    if only_in_left.is_empty() {
        "(nothing)".to_string()
    } else {
        only_in_left.join(" ")
    }
}

/// Is this token a flag rather than a value? `-1` and `0.5` are values that happen to
/// start with a dash or a digit; a flag's second character is a letter.
fn is_flag(token: &str) -> bool {
    token
        .strip_prefix('-')
        .map(|rest| rest.trim_start_matches('-'))
        .is_some_and(|rest| rest.chars().next().is_some_and(|first| first.is_ascii_alphabetic()))
}

/// A 12-hex key identifying a distinct memory footprint (sha1 of `memory_cmd`,
/// truncated to 12 hex chars - matches the reference tooling's key format).
pub fn param_hash(cmd: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(memory_cmd(cmd).as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(12);
    for byte in digest.iter().take(6) {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_port_and_reasoning() {
        // A "-nothink" twin differs only by `--reasoning off` (+ a different port);
        // both are stripped, so it hashes equal to its base - no separate measure.
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

    /// A sampling knob is footprint-neutral, but a device-side step cache is not:
    /// `--cache-mode`/`--cache-option` stay in the hash, so flipping easycache buys
    /// a fresh measurement instead of reusing a footprint taken without it.
    #[test]
    fn sampling_knobs_are_stripped_but_the_step_cache_is_not() {
        let base = "/opt/sdcpp/bin/sd-server --diffusion-model /sd/u.gguf --steps 16 --cfg-scale 4.0";
        let restepped =
            "/opt/sdcpp/bin/sd-server --diffusion-model /sd/u.gguf --steps 28 --cfg-scale 6.0";
        assert_eq!(param_hash(base), param_hash(restepped));

        let cached = "/opt/sdcpp/bin/sd-server --diffusion-model /sd/u.gguf --steps 16 \
                      --cfg-scale 4.0 --cache-mode easycache --cache-option threshold=0.05";
        assert_ne!(param_hash(base), param_hash(cached));
        // …and the threshold itself is part of the key.
        let looser = cached.replace("threshold=0.05", "threshold=0.2");
        assert_ne!(param_hash(&looser), param_hash(cached));
    }

    /// The mismatch message names the tokens that moved, not both whole commands:
    /// a 20-token command on each side buries the one flag that changed.
    #[test]
    fn token_difference_names_only_what_moved() {
        // A differing value carries its flag: `-c` is on both sides, so a plain token
        // diff would report "8192" with no way to tell what it sets.
        assert_eq!(token_difference("a -c 8192 b", "a -c 262144 b"), "-c 8192");
        assert_eq!(token_difference("a b", "a b c"), "(nothing)");
        // A repeated token is matched by count, not by presence.
        assert_eq!(token_difference("a a b", "a b"), "a");
        // A run of differing tokens stays one phrase rather than being split.
        assert_eq!(
            token_difference("s -m /m.gguf --spec-type draft-mtp", "s -m /m.gguf"),
            "--spec-type draft-mtp"
        );
        // A negative value is a value, not a flag to hang the next token off.
        assert_eq!(token_difference("s --gpu -1 -c 4096", "s --gpu -1 -c 8192"), "-c 4096");
    }    #[test]
    fn quant_change_changes_hash() {
        let q4 = "/app/llama-server -m /models/Cydonia-Q4_K_L.gguf -c 0 -fa on";
        let q6 = "/app/llama-server -m /models/Cydonia-Q6_K.gguf -c 0 -fa on";
        assert_ne!(param_hash(q4), param_hash(q6));
    }
}
