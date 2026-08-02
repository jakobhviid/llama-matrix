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
7. **evict_cost from telemetry.** Derive keep/evict weights from real llama-swap
   request frequency/recency instead of load-time heuristics.
8. **Dynamic margin.** Scale the safety margin per-combo (more co-resident models →
   more compute-buffer slack) or from measured additivity variance, instead of a
   flat value.
9. **Generation-peak budgeting.** Image servers transiently allocate more during a
   diffusion step than at idle-load; budget the transient peak when co-running image
   generation near the ceiling.
10. **More platform backends.** Apple Silicon (unified memory via Metal), and
    `rocm-smi`/other AMD paths where sysfs isn't available.
11. **Multi-GPU / multi-node.** A single unified pool today; discrete or multi-GPU
    makes the knapsack multi-dimensional (per-device budgets) — the Multi-Choice
    Multi-Dimensional Knapsack. A larger change to the fit predicate and emission.
12. **Readable output (short vars / set names).** Mint the reserved `vars` aliases
    (and/or shorter set names) so the block reads `+g_gemma` rather than
    `+g_gemma_4_26b_a4b_q4qat`. Cosmetic — full ids work as-is on llama-swap v243+.
13. **Per-pool VRAM/GTT split.** The `GpuMemory` trait reports summed occupancy, so
    `d_vram`/`d_gtt` are recorded as 0. Expose the split (the AMD sysfs backend
    already reads both pools) for per-pool insight — `build` uses only `d_total`.

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
