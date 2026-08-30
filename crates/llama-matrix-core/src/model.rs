//! The per-model record derived from a llama-swap config entry, and the pure
//! helpers that classify it. Everything here operates on an **already
//! macro-expanded** command string (see `config`).

use crate::param_hash::param_hash;

/// A model's role, derived from its launch command - never a hardcoded id-set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Llm,
    Embed,
    Rerank,
    Stt,
    Image,
    /// A fronted service with a placeholder `cmd` (e.g. a TTS proxy) - no GPU of
    /// its own; excluded from the measure worklist, footprint hand-set.
    TtsProxy,
}

impl ModelType {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelType::Llm => "llm",
            ModelType::Embed => "embed",
            ModelType::Rerank => "rerank",
            ModelType::Stt => "stt",
            ModelType::Image => "image",
            ModelType::TtsProxy => "tts-proxy",
        }
    }
}

/// Classify a model by the binary / flags in its command.
pub fn type_from_cmd(cmd: &str) -> ModelType {
    if cmd.contains("sd-server") {
        ModelType::Image
    } else if cmd.contains("whisper-server") {
        ModelType::Stt
    } else if cmd.contains("--reranking") {
        ModelType::Rerank
    } else if cmd.contains("--embedding") {
        ModelType::Embed
    } else {
        ModelType::Llm
    }
}

/// The model's own weight file (as it appears in the command - an in-container
/// path until the `[paths]` map resolves it), for existence/prune checks. Flags
/// are tried in priority order so a diffusion model wins over a stray `-m`.
pub fn primary_file(cmd: &str) -> Option<String> {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    for flag in ["--diffusion-model", "-m", "--model", "--llm"] {
        for (index, token) in tokens.iter().enumerate() {
            if *token == flag {
                if let Some(value) = tokens.get(index + 1) {
                    return Some((*value).to_string());
                }
            }
        }
    }
    None
}

/// Extensions that name a weight file a backend loads into memory.
///
/// Used to total a model's weights on disk, which is a floor on the footprint of a
/// fully offloaded model (see `measure`). Deliberately extension-driven rather than
/// a flag allowlist: one command names its weights across many flags (`-m`,
/// `--diffusion-model`, `--vae`, `--t5xxl`, `--llm`, `--mmproj`, ...) and the next
/// backend invents more, so an allowlist silently misses one while the extension
/// keeps holding.
const WEIGHT_EXTENSIONS: &[&str] = &[".gguf", ".safetensors", ".bin", ".ckpt", ".pt", ".pth"];

/// Every weight file the command names, in order, deduplicated.
///
/// Callers stat these, so a token that merely looks like a path costs nothing and a
/// path that isn't readable (an unmapped container path) simply doesn't count.
pub fn weight_files(cmd: &str) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    for token in cmd.split_whitespace() {
        let lowered = token.to_ascii_lowercase();
        if WEIGHT_EXTENSIONS.iter().any(|extension| lowered.ends_with(extension))
            && !files.iter().any(|seen| seen == token)
        {
            files.push(token.to_string());
        }
    }
    files
}

/// GB of host RAM a llama-server holds for its prompt cache, per the command.
///
/// `-cram` / `--cache-ram` takes MiB and caps llama.cpp's host-side prompt cache.
/// The memory is anonymous and private-dirty: the kernel cannot reclaim it, and
/// llama.cpp evicts only against this cap, never against host pressure. `-cram 0`
/// disables the cache.
///
/// `None` when the command does not say, which is the common case and is not the
/// same as zero: recent builds default to 8192 MiB without the flag appearing
/// anywhere. The caller supplies what to assume then (`host_cache_gb`).
pub fn declared_cache_ram_gb(cmd: &str) -> Option<f64> {
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let index = tokens.iter().position(|token| matches!(*token, "-cram" | "--cache-ram"))?;
    let mib: f64 = tokens.get(index + 1)?.parse().ok()?;
    Some(mib * 1024.0 * 1024.0 / crate::platform::BYTES_PER_GIB)
}

/// Does this command look like a placeholder proxy rather than a real server
/// (e.g. `sleep infinity`)? Such entries allocate no GPU and are excluded from
/// the measure worklist.
pub fn looks_like_proxy(cmd: &str) -> bool {
    let first = cmd.split_whitespace().next().unwrap_or("");
    first == "sleep" || (primary_file(cmd).is_none() && !cmd.contains("-server"))
}

/// A parsed, classified model ready for measurement or matrix math.
#[derive(Debug, Clone)]
pub struct ModelRecord {
    pub id: String,
    /// The launch command, macro-expanded and normalized to one line.
    pub cmd: String,
    pub model_type: ModelType,
    /// Weight path as written in the command (container path until mapped).
    pub primary_file: Option<String>,
    pub param_hash: String,
}

impl ModelRecord {
    /// Build a record from an id and an already-expanded, normalized command.
    pub fn from_expanded(id: impl Into<String>, cmd: impl Into<String>) -> Self {
        let id = id.into();
        let cmd = cmd.into();
        let mut model_type = type_from_cmd(&cmd);
        if model_type == ModelType::Llm && looks_like_proxy(&cmd) {
            model_type = ModelType::TtsProxy;
        }
        let primary_file = primary_file(&cmd);
        let param_hash = param_hash(&cmd);
        ModelRecord {
            id,
            cmd,
            model_type,
            primary_file,
            param_hash,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_derivation() {
        assert_eq!(
            type_from_cmd("/opt/sdcpp/bin/sd-server --diffusion-model /x.gguf"),
            ModelType::Image
        );
        assert_eq!(
            type_from_cmd("/opt/whisper/bin/whisper-server -m /x.bin"),
            ModelType::Stt
        );
        assert_eq!(
            type_from_cmd("/app/llama-server -m /x.gguf --reranking --pooling rank --embedding"),
            ModelType::Rerank
        );
        assert_eq!(
            type_from_cmd("/app/llama-server -m /x.gguf --embedding --pooling last"),
            ModelType::Embed
        );
        assert_eq!(type_from_cmd("/app/llama-server -m /x.gguf -c 4096"), ModelType::Llm);
    }

    #[test]
    fn primary_file_priority() {
        assert_eq!(
            primary_file("/opt/sdcpp/bin/sd-server --diffusion-model /sd/u.gguf --vae /sd/vae"),
            Some("/sd/u.gguf".to_string())
        );
        assert_eq!(
            primary_file("/app/llama-server -m /models/a.gguf -ngl 99"),
            Some("/models/a.gguf".to_string())
        );
        assert_eq!(primary_file("/app/llama-server -ngl 99"), None);
    }

    #[test]
    fn weight_files_collects_every_named_file() {
        // A diffusion command names its weights across four different flags; the
        // footprint floor needs all of them, not just the primary file.
        let sd = "/opt/sdcpp/bin/sd-server --diffusion-model /sd/unet/z.Q6_K.gguf \
                  --llm /sd/text_encoders/qwen3.Q4_K_M.gguf --vae /sd/vae/ae.safetensors \
                  --diffusion-fa --vae-tiling --steps 8";
        assert_eq!(
            weight_files(sd),
            vec![
                "/sd/unet/z.Q6_K.gguf".to_string(),
                "/sd/text_encoders/qwen3.Q4_K_M.gguf".to_string(),
                "/sd/vae/ae.safetensors".to_string(),
            ]
        );
        // whisper's `.bin`, and a repeated path counted once.
        assert_eq!(
            weight_files("/opt/whisper/bin/whisper-server -m /m/ggml.bin --model /m/ggml.bin"),
            vec!["/m/ggml.bin".to_string()]
        );
        // Non-weight tokens (an output path, a template) are not weights.
        assert!(weight_files("/app/llama-server -ngl 99 -o /tmp/out.png").is_empty());
    }

    /// `-cram` is in MiB and caps host RAM, not VRAM. An absent flag is not zero:
    /// recent llama.cpp takes 8192 MiB without the flag appearing anywhere, so the
    /// caller has to decide what to assume rather than read a 0 here.
    #[test]
    fn cache_ram_is_read_in_mib_and_absence_is_not_zero() {
        assert_eq!(
            declared_cache_ram_gb("/app/llama-server -m /m.gguf -cram 4096"),
            Some(4.0)
        );
        assert_eq!(
            declared_cache_ram_gb("/app/llama-server -m /m.gguf --cache-ram 8192 -c 4096"),
            Some(8.0)
        );
        // Explicitly disabled is a real, declared zero.
        assert_eq!(declared_cache_ram_gb("/app/llama-server -m /m.gguf -cram 0"), Some(0.0));
        // Unstated, and a flag with nothing after it.
        assert_eq!(declared_cache_ram_gb("/app/llama-server -m /m.gguf -c 4096"), None);
        assert_eq!(declared_cache_ram_gb("/app/llama-server -m /m.gguf -cram"), None);
    }

    #[test]
    fn proxy_detection() {
        let r = ModelRecord::from_expanded("tts-1", "sleep infinity");
        assert_eq!(r.model_type, ModelType::TtsProxy);
        assert_eq!(r.primary_file, None);
    }
}
