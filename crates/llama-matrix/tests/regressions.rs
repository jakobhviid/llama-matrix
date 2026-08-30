//! One reproducing test per confirmed bug, at the level the bug was observed.
//!
//! Separate from `cli.rs` on purpose. `cli.rs` says what the CLI is *supposed* to do;
//! this file says what it once did *wrong*, and each test carries the observation that
//! produced it. A test here failing means a fixed bug came back, which is a different
//! and more alarming thing than a feature test failing.
//!
//! A bug that is only visible inside one function gets its reproducing test next to
//! that function instead; these are the ones an operator could see.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use llama_matrix_core::param_hash::param_hash;

/// A roster of `count` equal-sized LLMs that cannot all co-reside, plus one aux model,
/// so the builder produces many maximal packs and every one of them rides along with
/// aux. Returns the working directory's path, already populated.
fn crowded_roster(dir: &Path, count: usize, unconfirmed_aux: bool) {
    let llm_cmd = |index: usize| format!("/app/llama-server -m /models/m{index}.gguf -c 4096");
    let aux_cmd = "/app/llama-server -m /models/e.gguf --embedding --pooling last -c 8192";

    let mut config = String::from("models:\n");
    for index in 0..count {
        config.push_str(&format!("  \"m{index}\":\n    cmd: \"{}\"\n", llm_cmd(index)));
    }
    config.push_str(&format!("  \"embed\":\n    cmd: \"{aux_cmd}\"\n"));
    fs::write(dir.join("config.yaml"), config).unwrap();
    fs::write(dir.join("llama-matrix.toml"), "budget = 100.0\nmargin = 4.0\n").unwrap();

    let measurements = dir.join("measurements");
    fs::create_dir_all(&measurements).unwrap();
    fs::write(measurements.join("_box.json"), r#"{"baseline":0.16,"detected_total":100.0}"#)
        .unwrap();
    for index in 0..count {
        // 20 GB each: four fit under the 96 GB ceiling, five do not, so the pack count
        // is combinatorial rather than one.
        fs::write(
            measurements.join(format!("m{index}.json")),
            format!(
                r#"{{"type":"llm","file":"/models/m{index}.gguf","measurements":{{"{}":{{"status":"ok","d_total":20.0,"allocation_confirmed":true}}}}}}"#,
                param_hash(&llm_cmd(index))
            ),
        )
        .unwrap();
    }
    let confirmed = if unconfirmed_aux { "" } else { r#","allocation_confirmed":true"# };
    fs::write(
        measurements.join("embed.json"),
        format!(
            r#"{{"type":"embed","file":"/models/e.gguf","measurements":{{"{}":{{"status":"ok","d_total":5.0{confirmed}}}}}}}"#,
            param_hash(aux_cmd)
        ),
    )
    .unwrap();
}

/// A build warning named every set that depended on an unconfirmed footprint. One
/// unconfirmed **aux** model rides along in all of them, so on a 25-model roster the
/// warning listed 224 set names: 3.5 KB, unreadable in a terminal, and copied verbatim
/// into `config.yaml` as a single comment line by `matrix::render`.
///
/// Fixed in 9523042. Past eight names the warning reports the ratio instead.
#[test]
fn a_warning_does_not_list_every_pack_by_name() {
    let dir = tempfile::tempdir().unwrap();
    crowded_roster(dir.path(), 8, true);

    let output = Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .arg("build")
        .assert()
        .success();
    let block = String::from_utf8(output.get_output().stdout.clone()).unwrap();

    let warning = block
        .lines()
        .find(|line| line.contains("without confirming"))
        .expect("an unconfirmed aux footprint must warn");
    // The test is only meaningful if the roster is combinatorial: with one pack there
    // is no roll-call to avoid. Eight 20 GB units under a 96 GB ceiling pack four at a
    // time, so this is C(8,4) = 70 maximal packs.
    let packs = block.lines().filter(|line| line.trim_start().starts_with("pack")).count();
    assert!(packs > 8, "the fixture must produce many packs to be a regression, got {packs}");
    assert!(
        warning.contains("declared sets"),
        "the warning should report a ratio, not a roll-call: {warning}"
    );
    // The roll-call was the defect. Nine names is already past the bound.
    let named = (0..70).filter(|index| warning.contains(&format!("pack{index}"))).count();
    assert!(named <= 8, "warning names {named} packs individually: {warning}");
    assert!(warning.len() < 400, "warning is {} chars: {warning}", warning.len());
}

/// `strategy` was retired when `[groups]` became authoritative, but three build
/// warnings and the `configure` help still told the reader to set it. Advice from the
/// tool IS documentation, and it outranks the docs: an agent reading
/// "reduce it with `strategy = \"family\"`" put the dead key back into a live config
/// twice, correctly citing the tool as its source.
///
/// Fixed in the commit adding this test. A retired setting has to leave every surface
/// that names it, not only the prose.
#[test]
fn no_surface_tells_you_to_set_a_retired_setting() {
    let surfaces = [vec!["--llm"], vec!["configure", "--help"], vec!["--help"], vec!["build", "--help"]];
    for args in surfaces {
        let output = Command::cargo_bin("llama-matrix")
            .unwrap()
            .args(&args)
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        for dead in ["strategy = \"family\"", "strategy = \"flat\"", "configure set strategy"] {
            assert!(
                !text.contains(dead),
                "`llama-matrix {}` still tells the reader to set `{dead}`",
                args.join(" ")
            );
        }
    }

    // …and it is not settable, so following such advice fails loudly rather than
    // writing a key nothing reads.
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .args(["configure", "set", "strategy", "family"])
        .assert()
        .failure();
}

/// `allocation_confirmed` asks whether the load-trigger finished. A hand-set proxy
/// entry never loads, so it could never answer, and the build carried an unconfirmed
/// warning no amount of measuring could clear. Because such a model is usually aux, it
/// rode along in every set: 224 of 229 named, every build, forever.
///
/// Fixed in 5d106b6. A hand-set footprint is reported as a declaration instead.
#[test]
fn a_hand_set_proxy_does_not_warn_as_unconfirmed_forever() {
    let dir = tempfile::tempdir().unwrap();
    crowded_roster(dir.path(), 4, false);

    // A fronted service with a placeholder `cmd`: no GPU of its own, footprint written
    // by hand under a key of the operator's choosing rather than a param-hash.
    let config = fs::read_to_string(dir.path().join("config.yaml")).unwrap();
    fs::write(
        dir.path().join("config.yaml"),
        format!("{config}  \"tts-1\":\n    cmd: \"sleep infinity\"\n"),
    )
    .unwrap();
    fs::write(
        dir.path().join("measurements/tts-1.json"),
        r#"{"type":"tts-proxy","measurements":{"manual":{"status":"ok","d_total":0.1}}}"#,
    )
    .unwrap();

    let output = Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .arg("build")
        .assert()
        .success();
    let block = String::from_utf8(output.get_output().stdout.clone()).unwrap();

    assert!(
        !block.contains("without confirming"),
        "a hand-set proxy must not read as an unconfirmed measurement:\n{block}"
    );
    assert!(block.contains("tts-1"), "the proxy is still in the matrix:\n{block}");
}
