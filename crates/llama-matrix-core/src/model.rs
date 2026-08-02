//! The per-model record derived from a llama-swap config entry, and the pure
//! helpers that classify it. Everything here operates on an **already
//! macro-expanded** command string (see `config`).

use crate::param_hash::param_hash;

/// A model's role, derived from its launch command — never a hardcoded id-set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Llm,
    Embed,
    Rerank,
    Stt,
    Image,
    /// A fronted service with a placeholder `cmd` (e.g. a TTS proxy) — no GPU of
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

/// The model's own weight file (as it appears in the command — an in-container
/// path until the `[paths]` map resolves it), for existence/prune checks. Flags
/// are tried in priority order so a diffusion model wins over a stray `-m`.
pub fn primary_file(cmd: &str) -> Option<String> {
    let toks: Vec<&str> = cmd.split_whitespace().collect();
    for flag in ["--diffusion-model", "-m", "--model", "--llm"] {
        for (i, t) in toks.iter().enumerate() {
            if *t == flag {
                if let Some(v) = toks.get(i + 1) {
                    return Some((*v).to_string());
                }
            }
        }
    }
    None
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
    fn proxy_detection() {
        let r = ModelRecord::from_expanded("tts-1", "sleep infinity");
        assert_eq!(r.model_type, ModelType::TtsProxy);
        assert_eq!(r.primary_file, None);
    }
}
