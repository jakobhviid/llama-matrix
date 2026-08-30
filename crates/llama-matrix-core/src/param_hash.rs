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

    #[test]
    fn quant_change_changes_hash() {
        let q4 = "/app/llama-server -m /models/Cydonia-Q4_K_L.gguf -c 0 -fa on";
        let q6 = "/app/llama-server -m /models/Cydonia-Q6_K.gguf -c 0 -fa on";
        assert_ne!(param_hash(q4), param_hash(q6));
    }
}
