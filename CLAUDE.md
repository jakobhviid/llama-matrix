See @AGENTS.md for this repository's agent guidelines — notably:

- **never** add AI attribution (no `Co-Authored-By` / "Generated with …") to commits or PRs;
- use **Conventional-Commit** subject prefixes (`feat:` → minor, `fix:` → patch,
  `feat!:`/breaking → major) so CI **auto-derives** the release version — never
  bump `version` in `Cargo.toml` by hand to release.
