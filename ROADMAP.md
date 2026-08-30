# ROADMAP.md - what llama-matrix does not do yet

Deliberately deferred work, and the reasons. **Nothing here describes shipped
behaviour** - that lives in `README.md`, `WORKFLOWS.md`, `SPEC.md`, `ARCHITECTURE.md`
and `PRINCIPLES.md`. An item that gets built leaves this file.

The v1 build is intentionally conservative: it never OOMs, at the cost of leaving some
headroom on the table. Most items below trade that conservatism for tighter packing.
Compiled into `--llm`.

## Deferred

Roughly in order of value-to-effort:

1. **Per-variant packs.** A unit that alternates (`(a | b)`) is reserved at its
   largest member's footprint in every pack it joins, because the matrix must be safe
   for whichever one llama-swap loads. Where the members differ in size that leaves
   headroom on the table: a pack holding the smaller variant could have taken another
   model. Enumerating packs **per variant** rather than per unit recovers it, at the
   cost of multiplying the set count by the alternation width - so it trades directly
   against the 1000-combination cap and `MAX_PACKS`, and is only worth it where
   variants differ enough to unlock a real combination.

2. **Knapsack LLM units and images *together*.** Images currently take the headroom
   the LLM knapsack left rather than competing for it, so the builder will never trade
   an LLM away to fit two image servers. The general form needs maximal groups over
   the union of both, and a careful look at the combination count that produces.

3. **Custom binary→type rules.** A rule mapping a binary or flag pattern to a model
   type, so a roster with twenty entries of one unrecognised backend does not need
   twenty `[types]` lines. Worth doing once someone has such a roster; twenty explicit
   lines are not obviously worse than one rule nobody can read.

4. **Auto-group detection.** Derive candidate `[groups]` by normalizing ids (strip
   quant/mode) plus a confirmation heuristic, so a reduction strategy needs zero
   hand-declaration for common rosters.

5. **Context-parametric footprints.** Measure a model at several `-c` values to get a
   KV slope, then re-target the matrix for a different serving context without
   re-measuring, and answer "does pack Y still fit if I bump X to 128k?"

6. **Advise which change unlocks a *specific* pack.** Reporting every cheaper measured
   configuration is one thing; naming the one that matters is another. *"Switching
   this model to its measured 26.96 GB configuration would let pack Y hold one more
   model"* means re-running the knapsack per candidate swap and reporting only swaps
   that change the pack set. More expensive, and only useful to an operator already at
   the ceiling.

7. **evict_cost from telemetry.** The role tiers are the axis static config can
   express, and they cannot express **recency**: two same-tier models that do not fit
   together still alternate. Derive keep/evict weights from real request
   frequency/recency, layered over the static tiers rather than replacing them, so the
   operator's declared priorities still win.

   The data source exists. `GET /api/metrics/activity` (llama-swap v251) returns
   per-request records carrying what this needs:

   ```json
   {"data": [{"id": 4235, "timestamp": "2026-08-30T19:48:32Z",
              "model": "qwen3-embedding-4b-q8", "req_path": "/v1/embeddings",
              "resp_status_code": 200, "duration_ms": 31, "tokens": {…}}]}
   ```

   So the open questions are design, not discovery: how far back to look, how to decay
   recency into a weight, and how to combine it with a declared tier without letting a
   busy hour override a deliberate `[evict_costs.models]` pin. The one that has to be
   settled first: derived costs change whenever traffic does, which turns `build` from
   pure-over-the-store into pure-over-the-store-and-a-time-window, so `drift` would
   report a difference after nothing but usage.

8. **Dynamic margin.** Scale the safety margin per-combo (more co-resident models →
   more compute-buffer slack), or from this box's measured additivity error, instead
   of a flat value.

   `validate` supplies the measurement it would scale from (SPEC §7.5), and the first
   readings on the reference box are tight and *negative*. That is **not** a licence to
   shrink the default margin and the item must not be read as one: two samples, one
   box, one backend mix, one load order, and the margin also absorbs the transient
   generation peak, footprint drift between measure and use, and whatever a different
   load order fragments. What it justifies is the feature's shape - derive from the
   box's own error rather than a constant, and require several validations first.

9. **Generation-peak budgeting.** Fit against `max(d_total, peak_total)` under a policy
   knob, so a diffusion step's transient allocation is budgeted when co-running near
   the ceiling. A consumer for a field `measure` already records, not a new
   measurement.

   Deferred on the numbers: the peak sits 3-7% above the resident footprint (SPEC §2),
   several times inside the default 4.0 GB `margin`, and only one image pool is ever
   co-resident. It becomes worth doing if `margin` is tuned close to zero, or if a
   backend turns up whose peak is a larger fraction of its footprint.

10. **More platform backends.** `rocm-smi` and other AMD paths where sysfs isn't
    available.

11. **Multi-GPU / multi-node.** A single unified pool today; discrete or multi-GPU
    makes the knapsack multi-dimensional (per-device budgets), the Multi-Choice
    Multi-Dimensional Knapsack. A larger change to the fit predicate and emission.

12. **Readable output (short vars / set names).** Mint the reserved `vars` aliases
    (and/or shorter set names) so the block reads `+g_gemma` rather than
    `+g_gemma_4_26b_a4b_q4qat`. Cosmetic; full ids work as-is on llama-swap v243+.
    Blocked on documentation rather than effort: the `vars:` syntax is not in the
    README shipped in the llama-swap image (checked against v251), and guessing at it
    would risk emitting a block a live server rejects, which is the one thing `apply`
    exists to avoid. Confirm the syntax upstream first, then this is small. Shorter
    *set names* need no upstream knowledge and are separable.

13. **Per-pool VRAM/GTT split beyond AMD.** A per-pool read for the backends that
    report no split today, and a consumer for it - `build` uses only `d_total`, so the
    split is recorded for insight and to feed item 11's per-device budgets.

14. **Finish moving CLI orchestration into core** (house guidelines D6). The remaining
    provisioning glue in `main.rs` - the legacy-store migration decision in
    `open_store`, and `prune` / `setup` discovery - could follow `resolve_plan` into
    the library, so a second frontend never has to reimplement it.

## Premise risk (track this)

llama-swap is very actively developed - the matrix engine changed across multiple
releases in a single week around our reference point. The tool's premise is that the
solver has **no memory awareness**; that holds today (the config schema has no
memory/budget keys). If upstream ever adds VRAM auto-detection or a memory-aware
solver, llama-matrix's **build** half could become redundant - but the **measure**
half (real per-model footprints + a fit-proof) keeps standalone value regardless.
Mitigation: pin and test against a known llama-swap version range, and watch releases
plus the Groups-V2 discussion for memory-awareness landing.

## Non-goals

- Managing the llama-swap install, quadlets, image builds, or model downloads.
- Managing idle-TTL policy (residency is demand-driven via the matrix).
- Emitting the legacy `groups:` engine (llama-matrix targets the memory-aware
  `matrix:` engine only).
- Inventing footprints for unmeasurable models (Principle #2).
