//! End-to-end CLI tests: drive the `build` verb against a synthetic working
//! directory (llama-swap config + llama-matrix.toml + a measurement store) and
//! check the emitted matrix block.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use llama_matrix_core::param_hash::param_hash;

const CHAT_CMD: &str = "/app/llama-server -m /models/chat.gguf -ngl 99 -c 4096 -fa on";
const EMBED_CMD: &str = "/app/llama-server -m /models/e.gguf --embedding --pooling last -c 8192";

fn write_working_dir(dir: &Path) {
    fs::write(
        dir.join("config.yaml"),
        format!(
            r#"
models:
  "chat":
    cmd: "{CHAT_CMD}"
  "embed":
    cmd: "{EMBED_CMD}"
"#
        ),
    )
    .unwrap();
    fs::write(dir.join("llama-matrix.toml"), "budget = 100.0\n").unwrap();

    let measurements = dir.join("measurements");
    fs::create_dir_all(&measurements).unwrap();
    fs::write(
        measurements.join("_box.json"),
        r#"{"baseline":0.16,"detected_total":100.0}"#,
    )
    .unwrap();
    // Key each entry under the hash of its own command, exactly as `measure` does:
    // `build` selects by the *current* config's param-hash, and a hash miss is a
    // miss, so a placeholder key here would leave both models unmeasured.
    fs::write(
        measurements.join("chat.json"),
        format!(
            r#"{{"type":"llm","file":"/models/chat.gguf","measurements":{{"{}":{{"status":"ok","d_total":30.0,"load_s":20.0}}}}}}"#,
            param_hash(CHAT_CMD)
        ),
    )
    .unwrap();
    fs::write(
        measurements.join("embed.json"),
        format!(
            r#"{{"type":"embed","file":"/models/e.gguf","measurements":{{"{}":{{"status":"ok","d_total":7.0,"load_s":6.0}}}}}}"#,
            param_hash(EMBED_CMD)
        ),
    )
    .unwrap();
}

#[test]
fn build_emits_a_matrix_block() {
    let dir = tempfile::tempdir().unwrap();
    write_working_dir(dir.path());
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .arg("build")
        .assert()
        .success()
        .stdout(predicates::str::contains("matrix:"))
        .stdout(predicates::str::contains("aux:"))
        .stdout(predicates::str::contains("chat"));
}

#[test]
fn build_json_reports_counts() {
    let dir = tempfile::tempdir().unwrap();
    write_working_dir(dir.path());
    let output = Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .args(["build", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("\"packs\""), "json missing packs: {text}");
    assert!(text.contains("\"ceiling\""), "json missing ceiling: {text}");
}

#[test]
fn build_without_measurements_errors_clearly() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.yaml"),
        "models:\n  \"chat\":\n    cmd: \"/app/llama-server -m /m.gguf -c 4096\"\n",
    )
    .unwrap();
    fs::write(dir.path().join("llama-matrix.toml"), "budget = 100.0\n").unwrap();
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .arg("build")
        .assert()
        .failure()
        .stderr(predicates::str::contains("no measurements"));
}

#[test]
fn help_and_llm_work() {
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("co-residency matrix"));
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .arg("--llm")
        .assert()
        .success()
        .stdout(predicates::str::contains("SPEC"));
}

#[test]
fn build_apply_splices_the_block_and_backs_up() {
    let dir = tempfile::tempdir().unwrap();
    write_working_dir(dir.path());
    // Point at an unreachable endpoint: apply writes + backs up, skips verify, no
    // rollback (rollback only fires when the endpoint was reachable then dies).
    fs::write(
        dir.path().join("llama-matrix.toml"),
        "budget = 100.0\nendpoint = \"http://127.0.0.1:59999\"\n",
    )
    .unwrap();

    Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .args(["build", "--apply"])
        .assert()
        .success();

    let config = fs::read_to_string(dir.path().join("config.yaml")).unwrap();
    assert!(config.contains("matrix:"), "config should now contain the matrix block");
    assert!(config.contains("# ==== GENERATED matrix block"));
    assert!(config.contains("models:"), "original models must be preserved");
    assert!(
        dir.path().join("config.yaml.pre-matrix.bak").exists(),
        "a backup must be written"
    );
}

#[test]
fn build_apply_no_verify_splices_without_a_network_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    write_working_dir(dir.path());
    fs::write(
        dir.path().join("llama-matrix.toml"),
        "budget = 100.0\nendpoint = \"http://127.0.0.1:59999\"\n",
    )
    .unwrap();
    // --no-verify: pure backup + splice, no liveness check (so an unreachable
    // endpoint is irrelevant); still succeeds and writes the block + backup.
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .args(["build", "--apply", "--no-verify"])
        .assert()
        .success();
    assert!(fs::read_to_string(dir.path().join("config.yaml")).unwrap().contains("matrix:"));
    assert!(dir.path().join("config.yaml.pre-matrix.bak").exists());
}

#[test]
fn setup_writes_a_starter_config() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.yaml"),
        "models:\n  \"a\":\n    cmd: \"/app/llama-server -m /m.gguf -c 4096\"\n",
    )
    .unwrap();
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .arg("setup")
        .assert()
        .success();
    let toml = fs::read_to_string(dir.path().join("llama-matrix.toml")).unwrap();
    assert!(toml.contains("endpoint"));
    assert!(toml.contains("config = \"config.yaml\""));
}

#[test]
fn drift_detects_missing_then_synced_block() {
    let dir = tempfile::tempdir().unwrap();
    write_working_dir(dir.path());
    fs::write(
        dir.path().join("llama-matrix.toml"),
        "budget = 100.0\nconfig = \"config.yaml\"\nendpoint = \"http://127.0.0.1:59999\"\n",
    )
    .unwrap();

    // no block yet
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .args(["drift", "--json"])
        .assert()
        .success()
        .stdout(predicates::str::contains("\"has_block\":false"));

    // apply, then the live block matches a fresh build
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .args(["build", "--apply"])
        .assert()
        .success();
    Command::cargo_bin("llama-matrix")
        .unwrap()
        .current_dir(dir.path())
        .args(["drift", "--json"])
        .assert()
        .success()
        .stdout(predicates::str::contains("\"in_sync\":true"));
}
