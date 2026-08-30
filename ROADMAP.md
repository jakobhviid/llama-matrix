# ROADMAP.md - where llama-matrix is going

What v1.0 ships and the deliberately-deferred work after it. The v1 build is
intentionally conservative: it never OOMs, at the cost of leaving some headroom on
the table. Most items below trade that conservatism for tighter packing. Compiled
into `--llm`.

## v1.0 scope

- **measure** - solo-footprint sweep into the per-model, per-box measurement store,
  incremental by param-hash, lockfile-guarded, with failure classification.
- **build** - variant-collapse, the `flat` (max-flexibility) strategy, the knapsack,
  heavy classification, DSL emission, and the 1000-combination guard.
- **apply** - backup → marker-anchored splice → hot-reload wait → verify → rollback.
- **platforms** - AMD sysfs and NVIDIA memory backends; `--budget` makes `build`
  work with no sensor.
- **config** - `llama-matrix.toml` with the `configure` scalar surface, `setup`
  provisioning, and the `--json` / `--llm` / completions / man surfaces.
- **regression** - reproduce the reference fixtures (a known measurements set →
  its expected matrix block, and a tighter second budget).

## After v1.0

Roughly in order of value-to-effort:

1. **Actual-quant sizing.** Size each quant by its own measured footprint instead of
   the logical model's largest, unlocking combos a max-quant unit conservatively
   forbids. Cost: more sets - mind the combo cap.
2. **Richer combos: knapsack LLM units and images together.** A pack now carries the
   images that fit in the headroom its LLM units left, which covers "2 LLMs + an
   image" at no cost in sets or fan-out. Checked on the device, not just in the plan:
   `validate` on a set of one LLM plus **five** diffusion servers plus whisper (7
   models) predicted 91.97 GB and measured 91.87 GB, so diffusion backends are as
   additive as llama.cpp ones and the ride-along is sound. What is still missing is the *joint*
   enumeration: the images take what the LLM knapsack left rather than competing for
   it, so the builder will never trade an LLM away to fit two image servers. That
   needs maximal groups over the union of both, and a careful look at the combination
   count it produces.
3. **Custom binary→type rules.** The per-id `[types]` map has shipped, which covers an
   unusual backend by naming it. What is left is the general form: a rule mapping a
   binary or flag pattern to a type, so a roster with twenty entries of one
   unrecognised backend does not need twenty lines. Worth doing once someone has such
   a roster; twenty explicit lines are not obviously worse than one rule nobody can
   read.
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

   **The data source exists.** `GET /api/metrics/activity` (llama-swap v251) returns
   per-request records carrying exactly what this needs:

   ```json
   {"data": [{"id": 4235, "timestamp": "2026-08-30T19:48:32Z",
              "model": "qwen3-embedding-4b-q8", "req_path": "/v1/embeddings",
              "resp_status_code": 200, "duration_ms": 31, "tokens": {…}}]}
   ```

   So the open questions are design, not discovery: how far back to look, how to decay
   recency into a weight, and how to combine it with a declared tier without letting a
   busy hour override a deliberate `[evict_costs.models]` pin. Note the emitted costs
   would then change whenever traffic does, which turns `build` from pure-over-the-
   store into pure-over-the-store-and-a-time-window; `drift` would report a difference
   after nothing but usage. That interaction needs settling before the arithmetic does.
8. **Dynamic margin.** Scale the safety margin per-combo (more co-resident models →
   more compute-buffer slack) or from measured additivity variance, instead of a
   flat value.

   `validate` now supplies the measurement this would be scaled from, and the first
   two readings are strikingly tight: -0.09 GB on 6 models at 107.5 GB, and -0.10 GB
   on 7 models (five of them diffusion servers) at 92 GB. Both *negative*, i.e. the
   models share rather than compete.

   That is **not** a licence to shrink the default margin, and the item should not be
   read as one. Two samples on one box, one backend mix, one driver, with the pool
   loaded in one order. The margin also absorbs the transient generation peak (item 9,
   up to 0.73 GB), footprint drift between measure and use, and whatever a different
   load order fragments. What the numbers do justify is the shape of the feature:
   derive the margin from *this box's* measured error rather than from a constant, and
   require several validations before trusting it.
9. **Generation-peak budgeting.** Image servers transiently allocate more during a
   diffusion step than they leave resident; budget that peak when co-running image
   generation near the ceiling. `measure` already records the highest delta seen while
   allocating as `peak_total`, so this is a *consumer* for that field (fit against
   `max(d_total, peak_total)` under a policy knob), not a new measurement.

   **The size of the effect is measured, and it is small.** Across five diffusion
   models at `1024x1024`, the peak sits 0.47-0.73 GB above the resident footprint
   (3-7%), and it reproduces to the hundredth of a GB across sweeps twenty days apart:

   | model | resident | peak | over |
   |---|---|---|---|
   | `chroma1-hd-q6k` | 16.12 | 16.85 | +0.73 |
   | `flux-kontext-edit` | 21.09 | 21.76 | +0.67 |
   | `chroma1-hd-flash-q4km` | 14.18 | 14.91 | +0.73 |
   | `z-image-turbo-q6k` | 8.44 | 9.02 | +0.58 |
   | `one-obsession-v22-fp16` | 6.43 | 6.90 | +0.47 |

   The default `margin` of 4.0 GB already covers that several times over, and only one
   image pool is ever co-resident, so this stays deferred. It becomes worth doing if
   `margin` is tuned close to zero, or if a backend turns up whose peak is a larger
   fraction of its footprint. Re-measure before assuming the numbers above hold at a
   different `probe_image_size`: the peak scales with the resolution, as the footprint
   does.
10. **More platform backends.** `rocm-smi` and other AMD paths where sysfs isn't
    available. (Apple Silicon via Metal unified memory has shipped: `total` from
    `hw.memsize`, `used` from the `ioreg` IOAccelerator counter.)
11. **Multi-GPU / multi-node.** A single unified pool today; discrete or multi-GPU
    makes the knapsack multi-dimensional (per-device budgets) - the Multi-Choice
    Multi-Dimensional Knapsack. A larger change to the fit predicate and emission.
12. **Readable output (short vars / set names).** Mint the reserved `vars` aliases
    (and/or shorter set names) so the block reads `+g_gemma` rather than
    `+g_gemma_4_26b_a4b_q4qat`. Cosmetic; full ids work as-is on llama-swap v243+.
    Blocked on documentation rather than on effort: the `vars:` syntax is not in the
    README that ships in the llama-swap image (checked against v251), and guessing at
    it would risk emitting a block a live server rejects, which is the one thing
    `apply` exists to avoid. Confirm the syntax upstream first, then this is small.
    Shorter *set names* need no upstream knowledge and could be done independently.
13. **Per-pool VRAM/GTT split beyond AMD.** Shipped for AMD `amdgpu` sysfs
    (`GpuMemory::used_split_gb`, which reads both counters it already sums); NVIDIA
    and Apple Silicon report no split, and omit the fields rather than writing zeros.
    Remaining: a per-pool read for the other backends where the concept applies, and
    a consumer for it (`build` still uses only `d_total`, so the split is recorded
    for insight and to feed item 11's per-device budgets).


## House-style conformance backlog

Alignment work against the house guidelines (rust-cli-guidelines) that is deferred,
not skipped. These are style and ops items, separate from the product roadmap
above; each notes why it is not yet done.

- **Finish moving CLI orchestration into core (guidelines D6).** `resolve_plan` now
  lives in core; the remaining provisioning glue in `main.rs` (the legacy-store
  migration decision in `open_store`, and `prune` / `setup` discovery) could follow,
  so a second frontend never has to reimplement it.

## Premise risk (track this)

llama-swap is very actively developed - the matrix engine changed across multiple
releases in a single week around our reference point. The tool's premise is that the
solver has **no memory awareness**; that holds today (the config schema has no
memory/budget keys). If upstream ever adds VRAM auto-detection or a memory-aware
solver, llama-matrix's **build** half could become redundant - but the **measure**
half (real per-model footprints + a fit-proof) keeps standalone value regardless.
Mitigation: pin and test against a known llama-swap version range, and watch
releases + the Groups-V2 discussion for memory-awareness landing.

## Non-goals

- Managing the llama-swap install, quadlets, image builds, or model downloads.
- Managing idle-TTL policy (residency is demand-driven via the matrix).
- Emitting the legacy `groups:` engine (llama-matrix targets the memory-aware
  `matrix:` engine only).
- Inventing footprints for unmeasurable models (Principle #2).
