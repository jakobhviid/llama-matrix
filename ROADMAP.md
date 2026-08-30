# ROADMAP.md — where llama-matrix is going

What v1.0 ships and the deliberately-deferred work after it. The v1 build is
intentionally conservative: it never OOMs, at the cost of leaving some headroom on
the table. Most items below trade that conservatism for tighter packing. Compiled
into `--llm`.

## v1.0 scope

- **measure** — solo-footprint sweep into the per-model, per-box measurement store,
  incremental by param-hash, lockfile-guarded, with failure classification.
- **build** — variant-collapse, the `flat` (max-flexibility) strategy, the knapsack,
  heavy classification, DSL emission, and the 1000-combination guard.
- **apply** — backup → marker-anchored splice → hot-reload wait → verify → rollback.
- **platforms** — AMD sysfs and NVIDIA memory backends; `--budget` makes `build`
  work with no sensor.
- **config** — `llama-matrix.toml` with the `configure` scalar surface, `setup`
  provisioning, and the `--json` / `--llm` / completions / man surfaces.
- **regression** — reproduce the reference fixtures (a known measurements set →
  its expected matrix block, and a tighter second budget).

## After v1.0

Roughly in order of value-to-effort:

1. **Actual-quant sizing.** Size each quant by its own measured footprint instead of
   the logical model's largest, unlocking combos a max-quant unit conservatively
   forbids. Cost: more sets — mind the combo cap.
2. **Richer combos.** Enumerate maximal groups over the *union* of LLM units and
   images (e.g. "2 LLMs + an image"), not just LLM-packs or single-LLM-plus-images.
3. **Configurable model-type detection.** Today `type` is inferred from the launch
   command (binary + flags: `sd-server` → image, `whisper-server` → stt,
   `--embedding`/`--reranking` → embed/rerank, else llm). Let operators **override
   it in settings** — a per-id `type` map (and/or custom binary→type rules) — so an
   unusual image/STT/rerank backend classifies correctly instead of falling back to
   `llm`. Deferred; not worked on now.
4. **Auto-group detection.** Derive candidate `[groups]` by normalizing ids
   (strip quant/mode) + a confirmation heuristic, so a reduction strategy needs zero
   hand-declaration for common rosters.
5. **Context-parametric footprints.** Measure a model at several `-c` values to get
   a KV slope, then re-target the matrix for a different serving context without
   re-measuring, and answer "does pack Y still fit if I bump X to 128k?"
6. **KV-quant sensitivity.** Record footprint under q8 vs f16 KV so the tool can
   advise "switch this model to q8 KV to unlock pack Y."
7. **evict_cost from telemetry.** The `[evict_costs]` table ranks models by *role*,
   which is the axis static config can express and is enough to stop an idle image
   pool outvoting the model in use. It cannot express **recency**: two same-tier models
   that do not fit together still alternate. Derive keep/evict weights from real
   llama-swap request frequency/recency, layered over the static tiers rather than
   replacing them (the operator's declared priorities should still win).
8. **Dynamic margin.** Scale the safety margin per-combo (more co-resident models →
   more compute-buffer slack) or from measured additivity variance, instead of a
   flat value.
9. **Generation-peak budgeting.** Image servers transiently allocate more during a
   diffusion step than at idle-load; budget the transient peak when co-running image
   generation near the ceiling. `measure` samples occupancy throughout the load-trigger
   and records the highest delta as `peak_total`, so this item is a *consumer* for that
   field (fit against `max(d_total, peak_total)` under a policy knob), not a new
   measurement.
10. **More platform backends.** `rocm-smi` and other AMD paths where sysfs isn't
    available. (Apple Silicon via Metal unified memory has shipped: `total` from
    `hw.memsize`, `used` from the `ioreg` IOAccelerator counter.)
11. **Multi-GPU / multi-node.** A single unified pool today; discrete or multi-GPU
    makes the knapsack multi-dimensional (per-device budgets) — the Multi-Choice
    Multi-Dimensional Knapsack. A larger change to the fit predicate and emission.
12. **Readable output (short vars / set names).** Mint the reserved `vars` aliases
    (and/or shorter set names) so the block reads `+g_gemma` rather than
    `+g_gemma_4_26b_a4b_q4qat`. Cosmetic — full ids work as-is on llama-swap v243+.
13. **Per-pool VRAM/GTT split beyond AMD.** Shipped for AMD `amdgpu` sysfs
    (`GpuMemory::used_split_gb`, which reads both counters it already sums); NVIDIA
    and Apple Silicon report no split, and omit the fields rather than writing zeros.
    Remaining: a per-pool read for the other backends where the concept applies, and
    a consumer for it (`build` still uses only `d_total`, so the split is recorded
    for insight and to feed item 11's per-device budgets).

14. **Recover a renamed model's footprint.** A measurement file is opened by model
    id, so renaming an id in the config orphans its file and re-measures under the
    new name. Scan the store for a file holding a matching param-hash before
    measuring, and adopt it (a rename is not a new footprint).

## House-style conformance backlog

Alignment work against the house guidelines (rust-cli-guidelines) that is deferred,
not skipped. These are style and ops items, separate from the product roadmap
above; each notes why it is not yet done.

- **Purge em-dashes from the docs and comments.** The "No em-dashes" rule
  (AGENTS.md, CLAUDE.md) is now stated, but text written before it still carries
  roughly 260 em-dashes across the docs and Rust comments. Do one careful,
  judgment-based pass (each em-dash becomes a hyphen, comma, colon, parentheses, or
  a rewrite, never a blanket replace that harms readability). En-dashes in ranges
  stay.
- **Supply-chain gate (`deny.toml` + a CI `deny` job).** Add a cargo-deny check for
  licenses and sources (crates.io only), following dotsync. Note the difference:
  llama-matrix does network (HTTP to llama-swap), so it does not ban the TLS/crypto
  stacks dotsync bans. Advisories deliberately excluded, since a new CVE must never
  block an unrelated release; review them out of band.
- **PR-triggered CI + build caching.** The release workflow only runs the green
  gate on push to main (after merge). Add a `pull_request`-triggered clippy+test
  workflow with `Swatinem/rust-cache`, so the gate runs before merge and CI is
  faster. The whole fleet shares this gap.
- **`regressions.rs`.** Adopt the temper/dotsync bug-to-regression-test discipline:
  a dedicated integration-test file where every confirmed bug gets a reproducing
  test before its fix.
- **`[workspace.lints.clippy]` in `Cargo.toml`.** Declare the clippy policy in the
  manifest (dotsync does), so a bare local `cargo clippy` enforces the same gate CI
  runs, not only the `-D warnings` flag in release.yml.
- **Finish moving CLI orchestration into core (guidelines D6).** `resolve_plan` now
  lives in core; the remaining provisioning glue in `main.rs` (the legacy-store
  migration decision in `open_store`, and `prune` / `setup` discovery) could follow,
  so a second frontend never has to reimplement it.

## Premise risk (track this)

llama-swap is very actively developed — the matrix engine changed across multiple
releases in a single week around our reference point. The tool's premise is that the
solver has **no memory awareness**; that holds today (the config schema has no
memory/budget keys). If upstream ever adds VRAM auto-detection or a memory-aware
solver, llama-matrix's **build** half could become redundant — but the **measure**
half (real per-model footprints + a fit-proof) keeps standalone value regardless.
Mitigation: pin and test against a known llama-swap version range, and watch
releases + the Groups-V2 discussion for memory-awareness landing.

## Non-goals

- Managing the llama-swap install, quadlets, image builds, or model downloads.
- Managing idle-TTL policy (residency is demand-driven via the matrix).
- Emitting the legacy `groups:` engine (llama-matrix targets the memory-aware
  `matrix:` engine only).
- Inventing footprints for unmeasurable models (Principle #2).
