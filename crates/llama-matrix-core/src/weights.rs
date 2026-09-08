//! What a launch command loads off disk, and the shard chains it takes to find it.
//!
//! A fully offloaded model cannot hold much less than its weights occupy on disk,
//! which makes this total the one cross-check on a footprint that needs no GPU and no
//! cooperation from the backend (SPEC §7.2). It totals what the command **loads**,
//! which is not what it **names**: llama.cpp is handed the first file of a split GGUF
//! and resolves the siblings itself, and under the common split conventions the file
//! it is handed holds no tensors at all.
//!
//! **A total is claimed only when every file was accounted for.** A sum over whichever
//! subset happened to be readable is a floor far below the real one, so it cannot trip,
//! and a check that cannot trip reads exactly like a check that passed (Principles 1
//! and 7). A command naming no weight file at all likewise has no total rather than a
//! total of zero: a fronted proxy loads nothing, and zero is the absence of a floor
//! rather than a floor at zero. `None` says no floor is available, which is the honest
//! answer and one the store already carries.

use std::io::Read;
use std::path::Path;

use crate::model::weight_files;
use crate::platform::BYTES_PER_GIB;
use crate::policy::Policy;

/// Shard counts above this read as a malformed name rather than as a model.
///
/// The count is a five-digit field, so a mangled name can claim 99999 shards and each
/// claimed shard costs a `stat`. Nothing real approaches it: a 77 GiB model splits
/// into three.
const MAX_SHARDS: u32 = 999;

/// The weights a command loads, totalled on disk.
#[derive(Debug, Clone, Default)]
pub struct OnDisk {
    /// GB across every file the command loads - and `None` unless every one of them
    /// was accounted for.
    pub total: Option<f64>,
    /// The files that stopped a total being claimed, for the operator to act on.
    pub unresolved: Vec<String>,
}

/// Every file in `path`'s split chain, in index order; just `path` when it names no
/// chain.
///
/// This mirrors llama.cpp's own resolution, which is where its authority comes from.
/// `llama_model_loader` takes the shard count from the first file's `split.count`
/// metadata, but builds the sibling *paths* by stripping `-%05d-of-%05d.gguf` off the
/// name it was given and reformatting it per index (`llama_split_prefix` /
/// `llama_split_path`). A chain whose names do not follow that pattern cannot be
/// loaded at all, so the pattern is not a heuristic: it is half of the contract, and
/// it carries the same count the metadata does because one splitter wrote both.
/// Reading the count back out of the GGUF would mean walking dozens of
/// length-prefixed metadata entries to re-derive a number already printed in the
/// name.
///
/// Which index the command names does not matter, so a command pointing at shard 2
/// still resolves the whole chain. Reconstructing the paths rather than globbing for
/// them is what makes a *missing* shard detectable: a glob reports what exists, and
/// only the count says what should.
///
/// The same `-00001-of-00002` convention names HuggingFace `.safetensors` shards,
/// which a GGUF metadata read could not reach.
pub fn shard_chain(path: &str) -> Vec<String> {
    match split_name(path) {
        Some((stem, count, extension)) => (1..=count)
            .map(|index| format!("{stem}-{index:05}-of-{count:05}{extension}"))
            .collect(),
        None => vec![path.to_string()],
    }
}

/// `("/m/w", 3, ".gguf")` for `/m/w-00001-of-00003.gguf`; `None` when the name is not
/// a shard of a chain.
fn split_name(path: &str) -> Option<(&str, u32, &str)> {
    let dot = path.rfind('.')?;
    let (body, extension) = path.split_at(dot);
    // A dot inside a directory name is not an extension (`/models/v1.5/weights`).
    if extension.len() < 2 || !extension[1..].bytes().all(|byte| byte.is_ascii_alphabetic()) {
        return None;
    }
    let (head, count) = body.rsplit_once("-of-")?;
    let (stem, index) = head.rsplit_once('-')?;
    if !is_five_digits(index) || !is_five_digits(count) {
        return None;
    }
    let count: u32 = count.parse().ok()?;
    (1..=MAX_SHARDS).contains(&count).then_some((stem, count, extension))
}

fn is_five_digits(field: &str) -> bool {
    field.len() == 5 && field.bytes().all(|byte| byte.is_ascii_digit())
}

/// The tensor count from a GGUF's fixed header; `None` if `path` is not a readable
/// GGUF.
///
/// Only the 24-byte header is read - magic, version, tensor count, metadata count,
/// all at fixed offsets. Deliberately not a metadata parser: the key-value block is
/// length-prefixed strings and nested arrays whose lengths come off disk, and nothing
/// here needs a field out of it.
pub fn tensor_count(path: &Path) -> Option<u64> {
    let mut header = [0u8; 24];
    std::fs::File::open(path).ok()?.read_exact(&mut header).ok()?;
    if &header[..4] != b"GGUF" {
        return None;
    }
    let tensors: [u8; 8] = header[8..16].try_into().ok()?;
    Some(u64::from_le_bytes(tensors))
}

/// What the weight files `cmd` loads total on disk, through the container→host path
/// map.
pub fn on_disk(cmd: &str, policy: &Policy) -> OnDisk {
    let mut total = 0.0;
    let mut counted: Vec<String> = Vec::new();
    let mut unresolved: Vec<String> = Vec::new();
    for named in weight_files(cmd) {
        // A token that merely looks like a weight path could never have resolved to
        // one, so it is neither a file this command loads nor a hole in the total.
        // Collection is extension-driven on purpose (see `model::weight_files`), and
        // that looseness must not cost a model its floor.
        if !named.starts_with('/') {
            continue;
        }
        let chain = shard_chain(&named);
        let names_a_chain = chain.len() > 1;
        for file in chain {
            let host = policy.to_host(&file);
            // Two flags naming different shards of one chain resolve to the same set.
            if counted.contains(&host) {
                continue;
            }
            counted.push(host.clone());
            match std::fs::metadata(&host) {
                Ok(metadata) => total += metadata.len() as f64 / BYTES_PER_GIB,
                Err(_) => unresolved.push(file),
            }
        }
        // A GGUF holding no tensors is not a whole model, and one that names no chain
        // has nothing left to resolve into: it is a stub of a split convention this
        // build cannot follow, and its few MB are not the total. The tripwire matters
        // more than the case it was written for - without it, the next convention
        // reintroduces a floor that silently cannot trip.
        if !names_a_chain
            && named.ends_with(".gguf")
            && tensor_count(Path::new(&policy.to_host(&named))) == Some(0)
        {
            unresolved.push(format!("{named} (declares no tensors of its own)"));
        }
    }
    // Nothing to total is not a total of nothing. A placeholder `cmd` (a fronted TTS
    // proxy) names no weights, and writing it a 0.0 floor would put a figure in the
    // store that describes no file and bounds no footprint.
    let complete = unresolved.is_empty() && !counted.is_empty();
    OnDisk { total: complete.then_some(total), unresolved }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_chains_are_resolved_from_the_split_name() {
        let chain = vec![
            "/m/w-00001-of-00003.gguf".to_string(),
            "/m/w-00002-of-00003.gguf".to_string(),
            "/m/w-00003-of-00003.gguf".to_string(),
        ];
        assert_eq!(shard_chain("/m/w-00001-of-00003.gguf"), chain);
        // Whichever shard the command names, the chain is the same.
        assert_eq!(shard_chain("/m/w-00002-of-00003.gguf"), chain);

        // A single-file model, and a chain of one, are themselves.
        assert_eq!(shard_chain("/m/w.gguf"), vec!["/m/w.gguf".to_string()]);
        assert_eq!(
            shard_chain("/m/w-00001-of-00001.gguf"),
            vec!["/m/w-00001-of-00001.gguf".to_string()]
        );

        // The same convention names HuggingFace safetensors shards.
        assert_eq!(shard_chain("/m/model-00001-of-00002.safetensors").len(), 2);

        // Nothing that isn't a five-digit-of-five-digit pair is a chain, and a count
        // no real split reaches is a mangled name rather than 99999 stats.
        for not_a_chain in [
            "/m/w-1-of-3.gguf",
            "/m/w-00001-of-3.gguf",
            "/m/w-of-00003.gguf",
            "/m/w-00000-of-00000.gguf",
            "/m/w-00001-of-99999.gguf",
            "/m/v1.5/weights",
        ] {
            assert_eq!(shard_chain(not_a_chain), vec![not_a_chain.to_string()], "{not_a_chain}");
        }
    }

    /// The bug this module exists for. `measure` stat'ed only the file `-m` named, and
    /// the common GGUF split conventions emit a **metadata-only first shard with zero
    /// tensors**: the observation was a 77.18 GiB model reporting `weights_gb: 0.86`,
    /// which is its mmproj plus an 11 MB stub, so its floor sat 83x clear of tripping
    /// and certified nothing.
    #[test]
    fn a_metadata_only_first_shard_does_not_stand_in_for_the_whole_model() {
        let dir = tempfile::tempdir().unwrap();
        let mib = |count: usize| vec![0u8; count * 1024 * 1024];
        // Shard 1 as the splitter writes it: a GGUF header declaring no tensors.
        let mut stub = b"GGUF".to_vec();
        stub.extend_from_slice(&3u32.to_le_bytes());
        stub.extend_from_slice(&0u64.to_le_bytes());
        stub.extend_from_slice(&67u64.to_le_bytes());
        stub.resize(1024 * 1024, 0);
        std::fs::write(dir.path().join("w-00001-of-00003.gguf"), &stub).unwrap();
        std::fs::write(dir.path().join("w-00002-of-00003.gguf"), mib(40)).unwrap();
        std::fs::write(dir.path().join("w-00003-of-00003.gguf"), mib(20)).unwrap();
        std::fs::write(dir.path().join("mmproj.gguf"), mib(3)).unwrap();

        let mut policy = Policy::default();
        policy.paths.insert("/models".to_string(), dir.path().display().to_string());
        let cmd = "/app/llama-server -m /models/w-00001-of-00003.gguf -ngl 99 \
                   --mmproj /models/mmproj.gguf";

        let weights = on_disk(cmd, &policy);
        assert!(weights.unresolved.is_empty(), "{:?}", weights.unresolved);
        let total = weights.total.expect("a complete chain totals");
        assert!((total - 64.0 / 1024.0).abs() < 1e-9, "total {total}");

        // A chain missing a shard yields no total at all. Summing what is left is the
        // failure being fixed, quieter: it would report 24 of 64 MiB as *the* total.
        std::fs::remove_file(dir.path().join("w-00002-of-00003.gguf")).unwrap();
        let partial = on_disk(cmd, &policy);
        assert_eq!(partial.total, None);
        assert_eq!(partial.unresolved, vec!["/models/w-00002-of-00003.gguf".to_string()]);
    }

    /// The tripwire for a split convention this build cannot follow: a GGUF that
    /// declares no tensors and names no chain is a stub, not a model, and its size is
    /// not a floor on anything.
    #[test]
    fn a_zero_tensor_gguf_naming_no_chain_claims_no_total() {
        let dir = tempfile::tempdir().unwrap();
        let mut stub = b"GGUF".to_vec();
        stub.extend_from_slice(&3u32.to_le_bytes());
        stub.extend_from_slice(&0u64.to_le_bytes());
        stub.extend_from_slice(&12u64.to_le_bytes());
        stub.resize(4096, 0);
        std::fs::write(dir.path().join("stub.gguf"), &stub).unwrap();
        let mut policy = Policy::default();
        policy.paths.insert("/models".to_string(), dir.path().display().to_string());

        let weights = on_disk("/app/llama-server -m /models/stub.gguf", &policy);
        assert_eq!(weights.total, None);
        assert!(weights.unresolved[0].contains("declares no tensors"), "{:?}", weights.unresolved);
    }

    #[test]
    fn a_gguf_header_is_read_only_where_it_is_fixed() {
        let dir = tempfile::tempdir().unwrap();
        let mut real = b"GGUF".to_vec();
        real.extend_from_slice(&3u32.to_le_bytes());
        real.extend_from_slice(&468u64.to_le_bytes());
        real.extend_from_slice(&3u64.to_le_bytes());
        let path = dir.path().join("shard.gguf");
        std::fs::write(&path, &real).unwrap();
        assert_eq!(tensor_count(&path), Some(468));

        // Too short to hold a header, not a GGUF at all, and absent: all unknown,
        // and unknown is not zero (only a declared zero is a stub).
        std::fs::write(dir.path().join("short.gguf"), b"GGUF").unwrap();
        assert_eq!(tensor_count(&dir.path().join("short.gguf")), None);
        std::fs::write(dir.path().join("other.safetensors"), vec![0u8; 64]).unwrap();
        assert_eq!(tensor_count(&dir.path().join("other.safetensors")), None);
        assert_eq!(tensor_count(&dir.path().join("absent.gguf")), None);
    }

    /// Every weight file the command loads counts, across every flag that names one -
    /// and a file that cannot be read means no total, not a shorter one. An
    /// `--mmproj` under a prefix `[paths]` does not map would otherwise shave a
    /// gigabyte off the floor with nothing saying so.
    #[test]
    fn an_unaccountable_file_claims_no_total() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("unet.gguf"), vec![0u8; 3 * 1024 * 1024]).unwrap();
        std::fs::write(dir.path().join("vae.safetensors"), vec![0u8; 1024 * 1024]).unwrap();
        let mut policy = Policy::default();
        policy.paths.insert("/sd".to_string(), dir.path().display().to_string());

        let complete = on_disk(
            "/opt/sdcpp/bin/sd-server --diffusion-model /sd/unet.gguf --vae /sd/vae.safetensors",
            &policy,
        );
        let total = complete.total.expect("readable weights");
        assert!((total - 4.0 / 1024.0).abs() < 1e-9, "total {total}");

        let short = on_disk(
            "/opt/sdcpp/bin/sd-server --diffusion-model /sd/unet.gguf \
             --t5xxl /elsewhere/t5.safetensors",
            &policy,
        );
        assert_eq!(short.total, None);
        assert_eq!(short.unresolved, vec!["/elsewhere/t5.safetensors".to_string()]);

        // A relative token could never have named a file this command loads, so it
        // neither counts nor withholds the total.
        let loose = on_disk(
            "/opt/sdcpp/bin/sd-server --diffusion-model /sd/unet.gguf --lora-model-dir x.safetensors",
            &policy,
        );
        assert!((loose.total.expect("still totalled") - 3.0 / 1024.0).abs() < 1e-9);
    }

    /// A hand-set proxy entry names no weights: its footprint is the operator's
    /// declaration, and a 0.0 total would read as a floor while bounding nothing.
    #[test]
    fn a_command_naming_no_weights_has_no_total() {
        let policy = Policy::default();
        assert_eq!(on_disk("sleep infinity", &policy).total, None);
        assert_eq!(on_disk("(migrated)", &policy).total, None);
        assert!(on_disk("sleep infinity", &policy).unresolved.is_empty());
    }

    /// Two flags naming shards of one chain describe one set of files, not two.
    #[test]
    fn one_chain_is_counted_once() {
        let dir = tempfile::tempdir().unwrap();
        for index in 1..=2 {
            std::fs::write(
                dir.path().join(format!("w-{index:05}-of-00002.gguf")),
                vec![0u8; 1024 * 1024],
            )
            .unwrap();
        }
        let mut policy = Policy::default();
        policy.paths.insert("/models".to_string(), dir.path().display().to_string());
        let weights = on_disk(
            "/app/llama-server -m /models/w-00001-of-00002.gguf \
             --model /models/w-00002-of-00002.gguf",
            &policy,
        );
        let total = weights.total.expect("readable weights");
        assert!((total - 2.0 / 1024.0).abs() < 1e-9, "total {total}");
    }
}
