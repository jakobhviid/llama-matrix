# ROADMAP.md - what llama-matrix does not do yet

Deliberately deferred work, and the reasons. **Nothing here describes shipped
behaviour** - that lives in `README.md`, `WORKFLOWS.md`, `SPEC.md`, `ARCHITECTURE.md`
and `PRINCIPLES.md`. An item that gets built leaves this file.

The v1 build is intentionally conservative: it never OOMs, at the cost of leaving some
headroom on the table. Most items below trade that conservatism for tighter packing.
Compiled into `--llm`.

## Deferred

Roughly in order of value-to-effort:

### Per-variant packs

A unit that alternates (`(a | b)`) is reserved at its largest member's footprint in
every pack it joins, because the matrix must be safe for whichever one llama-swap loads.
Where the members differ in size that leaves headroom on the table: a pack holding the
smaller variant could have taken another model. Enumerating packs **per variant** rather
than per unit recovers it, at the cost of multiplying the set count by the alternation
width - so it trades directly against the 1000-combination cap and `MAX_PACKS`, and is
only worth it where variants differ enough to unlock a real combination.

### Knapsack LLM units and images *together*

Images currently take the headroom the LLM knapsack left rather than competing for it,
so the builder will never trade an LLM away to fit two image servers. The general form
needs maximal groups over the union of both, and a careful look at the combination count
that produces.

### Check the host budget per set, not all-or-nothing

One model without a `d_host` disables the host dimension for the whole plan, on the
reasoning that a partial sum is not a smaller answer but a wrong one. That reasoning is
right about a *set* and wrong about a *plan*: a set whose members all have `d_host` can
be totalled exactly, whatever some other set is missing. Check those, skip the rest, and
say how many were skipped and why.

Not hypothetical. On the reference box two models were re-measured under new
param-hashes, and the host check - the whole feature - silently stopped applying to the
other twenty-three. The all-or-nothing rule turns a two-model gap into a total outage of
the thing most likely to catch an OOM.

### Size `-cram` per model from observed prompts

The prescription solves one equation with one unknown, so it names a uniform floor. The
real answer is heterogeneous by a wide margin: on the reference box an embedder and a
reranker with largest observed prompts of 2,263 and 9,275 tokens were each reserving the
8 GiB default, and pinning those two to 512 MiB freed 15 GiB, which is what let the LLMs
run at 2560 rather than 1792.

The data exists and its shape is known. `GET /api/metrics/activity` returns
`tokens.input_tokens` per request with `tokens.cache_tokens` beside it, so a per-model
prompt distribution is one paginated read away. Two things the reference data showed,
which any implementation has to handle or it will produce nonsense: a periodic health
probe with 2-token prompts dominates the percentiles unless you segment by model, and
the long prompts that would have overflowed the cache arrived with `cache_tokens = 0`,
meaning they were not being served from it and sizing to hold them buys nothing.

What is missing beyond the read is the conversion from tokens to gigabytes, which needs
KV bytes per token and therefore something model-specific the tool does not have today.
Until then it is an *advisory* feature: report the observed distribution per model
beside the cache each is holding, and let the operator do the last step.

### Record `allocate_s`

`load_s` is time to `ready`, and for a lazily-allocating backend that is not the time to
get it serving: a diffusion server is ready in about two seconds and finishes allocating
minutes later. `await_allocation` already knows the trigger's duration and throws it
away.

Recording it would put two existing decisions on data instead of an anecdote. The
`image` eviction tier sits above the service tiers on the argument that `ready`
understates it, and generation-peak budgeting is justified by a gap the store measures
only indirectly. Both would then cite a number the tool collected.

### Report the trade curve, not one point of it

Two knobs are chosen by walking a curve with a cliff in it, and the tool reports a single
point on each. Measured on the reference box:

| `-cram` | packs | excluded for host |
|---|---|---|
| 2560 | 294 | 0 |
| 3072 | 284 | 10 |
| 3584 | 154 | 140 |
| 4096 | 64 | 230 |

and separately a cardinality cap does not move the clean threshold but changes how many
sets exceed it (cap 5 at 3072 excludes 10; cap 6 excludes 29), so comparing thresholds
shows nothing and only the grid does. Both are documented as manual sweeps (SPEC §1.4,
WORKFLOWS Loop 6), which is fine as far as it goes and is exactly the kind of thing a
pure, fast `build` could do for itself: re-solve at two or three points and print the
shape. The cost is running the knapsack several times, which is cheap, and the risk is
turning one number into a table nobody reads, which is why it is deferred rather than
obvious.

### Custom binary→type rules

A rule mapping a binary or flag pattern to a model type, so a roster with twenty entries
of one unrecognised backend does not need twenty `[types]` lines. Worth doing once
someone has such a roster; twenty explicit lines are not obviously worse than one rule
nobody can read.

### Auto-group detection

Derive candidate `[groups]` by normalizing ids (strip quant/mode) plus a confirmation
heuristic, so a reduction strategy needs zero hand-declaration for common rosters.

### Context-parametric footprints

Measure a model at several `-c` values to get a KV slope, then re-target the matrix for
a different serving context without re-measuring, and answer "does pack Y still fit if I
bump X to 128k?"

### Advise which change unlocks a *specific* pack

Reporting every cheaper measured configuration is one thing; naming the one that matters
is another. *"Switching this model to its measured 26.96 GB configuration would let pack
Y hold one more model"* means re-running the knapsack per candidate swap and reporting
only swaps that change the pack set. More expensive, and only useful to an operator
already at the ceiling.

### evict_cost from recency

The type tiers are the axis static config can express, and they cannot express
**recency**: a model untouched for hours is priced exactly as it was when it was hot,
and two same-tier models that do not fit together still alternate. Derive keep weights
from real request recency, layered over the static tiers rather than replacing them, so
the operator's declared priorities still win.

**Recency is the load-bearing half, and reload latency alone is a trap.** The store
already holds `load_s`, so pricing each model by what it costs to reload looks like
the obvious move. It is not: llama-swap evicts whatever is cheapest to evict, so a
large slow model becomes permanently unevictable. Concretely, with
`glm-4.5-air` (74 GB, 100 s) and `gemma-4-26b` (24 GB, 22 s) resident and a third
model requested, the solver always drops gemma, and it will keep doing so however
long glm has sat idle. The flat tier that exists today does not have this problem,
so latency-derived costs would be a regression.

The right shape is the product:

```
keep(M) ≈ decay(time since M was last requested) × reload_seconds(M)
```

probability of being wrong × cost of being wrong, floored at 1. Latency alone
assumes the probability is uniform; recency alone assumes the latency is, and on
the reference box that is wrong by 50x (1 s to 100 s). The product also retires
the `Σ image costs + 1` derivation, because an idle image pool decays to 1 on its
own.

Two things to settle before any of it:

- **Config churn.** Costs that track traffic have to be rewritten on a timer, and
  each rewrite hot-reloads llama-swap. Whether a reload preserves residency decides
  whether this is viable at all: if it evicts, the cure thrashes worse than the
  disease. Not documented in the llama-swap README shipped in the image (v251);
  verify on a quiet box first.
- **`drift` stops meaning "someone changed something".** Worth fixing regardless:
  split it into *the declared sets differ* (structural, actionable) and *only the
  weights differ* (expected). Then a cost refresh is invisible to drift and a real
  config change still shows.

Plus the obvious guard: an explicit `[evict_costs.models]` pin must still win, or a
busy hour overrides a deliberate decision.

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

### Dynamic margin

Scale the safety margin per-combo (more co-resident models → more compute-buffer slack),
or from this box's measured additivity error, instead of a flat value.

`validate` supplies the measurement it would scale from (SPEC §7.5), and the first
readings on the reference box are tight and *negative*. That is **not** a licence to
shrink the default margin and the item must not be read as one: two samples, one
box, one backend mix, one load order, and the margin also absorbs the transient
generation peak, footprint drift between measure and use, and whatever a different
load order fragments. What it justifies is the feature's shape - derive from the
box's own error rather than a constant, and require several validations first.

### Generation-peak budgeting

Fit against `max(d_total, peak_total)` under a policy knob, so a diffusion step's
transient allocation is budgeted when co-running near the ceiling. `peak_total` is in
the store (SPEC §2), so the work is in the fit predicate and a policy knob, not in
measurement.

Deferred on the numbers: the peak sits 3-7% above the resident footprint (SPEC §2),
several times inside the default 4.0 GB `margin`, and only one image pool is ever
co-resident. It becomes worth doing if `margin` is tuned close to zero, or if a
backend turns up whose peak is a larger fraction of its footprint.

### More platform backends

`rocm-smi` and other AMD paths where sysfs isn't available.

### Multi-GPU / multi-node

A single unified pool today; discrete or multi-GPU makes the knapsack multi-dimensional
(per-device budgets), the Multi-Choice Multi-Dimensional Knapsack. A larger change to
the fit predicate and emission.

### Readable output (short vars / set names)

Mint the reserved `vars` aliases (and/or shorter set names) so the block reads
`+g_gemma` rather than `+g_gemma_4_26b_a4b_q4qat`. Cosmetic; full ids work as-is on
llama-swap v243+. Blocked on documentation rather than effort: the `vars:` syntax is not
in the README shipped in the llama-swap image (checked against v251), and guessing at it
would risk emitting a block a live server rejects, which is the one thing `apply` exists
to avoid. Confirm the syntax upstream first, then this is small. Shorter
*set names* need no upstream knowledge and are separable.

### Per-pool VRAM/GTT split beyond AMD

A per-pool read for the backends that report no split today, and a consumer for it -
`build` uses only `d_total`, so the split is recorded for insight and to feed
*multi-GPU / multi-node*'s per-device budgets.

### Finish moving CLI orchestration into core

(house guidelines D6). The remaining provisioning glue in `main.rs` - the legacy-store
migration decision in `open_store`, and `prune` / `setup` discovery - could follow
`resolve_plan` into the library, so a second frontend never has to reimplement it.

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
