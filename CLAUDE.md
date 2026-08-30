See @AGENTS.md for this repository's agent guidelines - notably:

- **never** add AI attribution (no `Co-Authored-By` / "Generated with …") to commits or PRs;
- **never** use an em-dash (`—`) in commits, PRs, comments, or docs (use a hyphen, comma, colon, or parentheses);
- **docs describe the tool as it is, not what changed.** `ROADMAP.md` holds only
  unbuilt work and an item that ships is deleted from it; `ADOPT.md` is temporary and
  **must be deleted once the implementation is complete**, with anything durable moved
  into README/WORKFLOWS/SPEC/ARCHITECTURE. What changed between versions is the
  commit message's job;
- use **Conventional-Commit** subject prefixes (`feat:` → minor, `fix:` → patch,
  `feat!:`/breaking → major) so CI **auto-derives** the release version - never
  bump `version` in `Cargo.toml` by hand to release.
