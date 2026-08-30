# ARCHITECTURE.md - how llama-matrix is built

The model behind the tool: what it reads, what it computes, what it writes, and
how the pieces fit. Read `PRINCIPLES.md` first for *why*; this is *how*. Compiled
into `--llm`.

---

## 0. The one-paragraph version

llama-swap can keep several models resident at once but has **no memory
awareness** - it only trusts a `matrix:` block that declares which model
combinations may co-reside. llama-matrix **measures** each model's real memory
footprint on your box, then **builds** the largest set of combinations that
provably fit under a budget, and **splices** that block into your llama-swap
`config.yaml`. Two phases, two subcommands: `measure` (GPU-touching, stateful) and
`build` (pure). A small config file (`llama-matrix.toml`) holds policy; llama-swap's
`config.yaml` stays untouched except for the generated block.

---

## 1. The memory model

### 1.1 Pools and the budget triple

A GPU exposes one or more memory pools. Two common shapes:

- **Discrete GPU** - a single VRAM pool (NVIDIA via NVML/`nvidia-smi`; AMD via
  `amdgpu` sysfs or `rocm-smi`).
- **Unified-memory APU** - a VRAM carve-out plus a GPU-accessible system-RAM pool
  (on AMD `amdgpu`: `mem_info_vram_*` + `mem_info_gtt_*`). Models spill from the
  first into the second, so occupancy is tracked as **one number** (the sum).
- **Apple Silicon**: one unified-memory pool shared by CPU and GPU (Metal). A model
  (llama.cpp Metal or MLX) allocates from it, so occupancy is the GPU's in-use
  unified memory (the `ioreg` `IOAccelerator` "In use system memory" counter) and
  total is `hw.memsize`. The GPU shares the pool with the OS, so reserve headroom
  with `budget`/`margin` rather than planning against the full total.

Three distinct quantities drive every fit decision:

| value | meaning | source |
|---|---|---|
| **detected total** | the physical pool the GPU exposes | runtime probe (informational) |
| **budget** | how much of the pool llama-matrix may plan against - a hard user reservation | config / CLI / defaults to detected total |
| **margin** | fragmentation + compute-buffer slack *inside* the budget | config (default 4 GB) |

`ceiling = budget − margin`. A 96 GB card where you want to keep 46 GB for other
apps is simply `budget = 50` → `ceiling = 46`.

**Budget resolution order:** `--budget/--vram N` → `budget` in `llama-matrix.toml`
→ auto-detected total → **hard error** (never a silent guess).

### 1.2 The platform abstraction

A `GpuMemory` trait exposes `total()` and `used()` (in bytes, summed across pools)
plus a human label. Backends implement it per platform; the rest of the tool is
platform-agnostic. It ships **AMD sysfs**, **NVIDIA**, and **Apple Silicon** (Metal
unified memory) backends, auto-selected by probing for each in turn. When no backend is available, `measure` cannot run, but
`build` still works entirely from a supplied budget + an existing measurements
file - the pure half never needs a sensor.

---

## 2. The two phases

```
                 llama-swap config.yaml (roster + cmds)
                              │
              ┌───────────────┴────────────────┐
              ▼                                 ▼
        measure  (GPU)                     build  (pure)
   load each model alone,            select each model's current
   read stabilized footprint    ──▶  footprint, knapsack the fitting
              │                       combinations, emit the block
              ▼                                 │
        measurements/  ◀────────────────────────┘
        (per-model, per-box store)               ▼
                                          matrix: block  ──apply──▶  config.yaml
```

### 2.1 Phase 1 - measure

A full sweep loads every model in turn and can run for the better part of an hour,
most of it inside one wait for `ready`. `sweep` therefore takes a progress callback
and reports each model as it starts and finishes; the core never prints (the CLI puts
those lines on stderr, so a `--json` pipe stays clean - Principle 9).

A miss under the model's own id is not always work: a rename orphans a measurement
file, and the store is checked for the same param-hash under an id the config has
dropped before anything is loaded (SPEC §2).

For each model in the config worklist:

1. Unload everything (`POST /api/models/unload`; `GET /unload` is a legacy
   fallback), wait for `/running` to go empty, then wait for occupancy itself to
   settle - the proxy's bookkeeping is not the device's occupancy. The settled
   reading is **this model's** baseline, so a pool that failed to clear shortens one
   delta visibly instead of every delta silently (SPEC §7.3).
2. Trigger the model to load via its type's endpoint, on its own thread so the load
   is in flight while `/running` is polled.
3. Poll `/running` until the model's state is `ready` - the only go signal.
   `starting` means wait; `stopping`/`stopped`/`shutdown` mean do-not-sample (a
   tearing-down model's reading is meaningless).
4. **Cross-check** that the server which just loaded is running the command we
   hashed: llama-swap serves whatever config *it* last reloaded, which is not
   necessarily the file we parsed. `/running` carries the command llama-swap
   launched, and that is compared on its memory tokens; where it is absent, the
   served command is inferred from the loaded server's own `/props` through the
   `/upstream/<id>/` route. A mismatch records nothing and fails the model; where
   neither source can decide, the model is measured and reported as unverified
   (SPEC §7.1).
5. **Await the trigger.** `ready` means the upstream answers HTTP, which for a
   lazily-allocating backend is long before its weights are resident: sd-server
   allocates when a generation actually runs, so the trigger's own completion is the
   allocation signal. A 2xx means the allocating work finished; a non-2xx or a
   transport failure records `FAILED` naming the status (the model may be half
   loaded); overrunning the budget records the reading as **unconfirmed**. Occupancy
   is sampled throughout for its peak (SPEC §7.2).
6. **Stabilize**: sample occupancy until three consecutive readings are within a
   small epsilon (KV and compute buffers finish allocating *after* `ready`). Still
   moving when sampling stops is also recorded as unconfirmed, rather than returned
   as though it had settled.
7. Record the delta over baseline and the load time, plus the VRAM/GTT split when
   the backend separates pools (AMD sysfs; a unified or single-pool device omits it
   rather than recording zeros), the allocation peak, whether allocation and serving
   were confirmed, and the total size of the weight files the command names.
8. Unload (all, or `POST /api/models/unload/:model_id` for just this one).

Then an **additivity check**: load a real co-resident combo, compare the measured
total to `baseline + Σ(solo deltas)`. Footprints are additive to within a small
error empirically, which is what makes the knapsack valid; the residual sizes the
safety margin.

Guards: a **pid lockfile** (two concurrent sweeps share the unload primitive and
corrupt each other's readings); a **pre-check** that the weight file exists (a
missing file exits the loader instantly - skip it with a clear message rather than
burning the load timeout); **failure classification** (missing-file vs timeout
vs premature-exit) so a broken model is recorded, not retried forever; **no request
outliving its model** (the trigger is awaited, or waited out, before the sweep moves
on, so one model's generation can never allocate during the next model's sampling
window); and the **weights-on-disk floor** (a fully offloaded model cannot hold much
less than its weights, so a footprint below 0.90 of the total size of the weight files
its command names is flagged - a warning, since partial offload legitimately sits
lower).

An **unconfirmed footprint is a cache miss**: `measure` re-measures it rather than
reporting `cached`. That is what makes the incremental cache safe to trust, and it needs
no operator judgement about which entries to distrust. A store holding no confirmations
re-measures in full; a store holding them re-measures only what is suspect.

### 2.2 Phase 2 - build

Pure. Reads the `measurements/` store + the config + policy, computes the matrix, and
prints it (stdout), writes it (`--out FILE`), or splices it into the config
(`--apply`). No GPU, safe to run anytime. See §4.

---

## 3. Reading the roster (the anti-drift core)

Both phases derive the model worklist **from the config itself** - never a
hand-kept list, which silently drifts. A shared parsing core (mirrors the
reference `mlib`) is the single source of truth:

- **Parse** the llama-swap `config.yaml` (`models:` map) into per-model records.
- **Expand macros first.** llama-swap supports a global `macros:` map, `${macro}`
  references, and reserved `${PORT}`/`${PID}`/`${MODEL_ID}`/`${env.VAR}`
  substitutions inside `cmd`. Everything below runs on the **expanded** command -
  deriving type/file/hash from an unexpanded `${…}` is a bug (a macro-free config
  expands to itself, so this is always safe).
- **`cmd`** - the launch command (folded/literal YAML scalar → one line), expanded.
- **type** - derived from the command, not a hardcoded id-set: an image server
  binary → `image`, a whisper binary → `stt`, `--reranking` → `rerank`,
  `--embedding` → `embed`, else `llm`. (See `SPEC.md` for the exact table.)
- **primary file** - the weight path (`-m` / `--model` / `--diffusion-model` / …),
  for existence checks and pruning.
- **param-hash** - the footprint key (§3.1).
- **path mapping** - when llama-swap runs in a container, the paths in `cmd` are
  container paths; a configurable `[paths]` map resolves them to host paths (an
  identity map when llama-swap runs natively).
- **Exclusions & tolerance** - skip hand-set proxy entries and llama-swap
  **selectors / virtual model ids** (per-request routing entries, not loadable
  servers) from the measure worklist; ignore unknown model-level keys (`cmdStop`,
  `unloadTimeout`, `capabilities`, …) rather than failing on them.

### 3.1 The param-hash & the multi-measurement store

A model can have several footprints over its life (a re-quant, a context or
parallelism change). Each is stored under a **param-hash** = a hash of the launch
command reduced to only its footprint-affecting tokens (a conservative allowlist
of known-irrelevant flags - host/port, reasoning toggle, chat template, sampler
knobs, etc. - is stripped; everything else is hashed). Consequences:

- Flip a non-memory flag (e.g. reasoning off) → same hash → instant cache hit.
- Change `-c`/`-np`/quant → new hash → a new measurement is **added** alongside the
  old one. Revert later → the old hash hits instantly. Nothing is thrown away.
- The strip-list is deliberately conservative: an unlisted-but-irrelevant flag
  causes a harmless extra measure, never a wrong reuse (Principle #6).
- Lookup is by **exact hash, and a miss is a miss**: an entry stored under another
  hash was measured under different memory flags, so reusing it would be the wrong
  cache hit Principle #6 forbids. The one carve-out is a `tts-proxy` entry, which is
  keyed by hand because it is never measured (SPEC §2 consumer rule).

**Storage.** The cache is a `measurements/` directory beside `llama-matrix.toml` -
**one JSON file per model** (`<model-id>.json`, holding that model's param-hash-keyed
measurement map) plus a reserved `_box.json` for the box-level values (baseline,
detected total, additivity check) that have no per-model home. Per-model files
avoid a single hand-edited blob, are cheap, and are **retained indefinitely** - a
model removed from the config keeps its file, so re-adding it hits the cache.
Pruning is **explicit only** (`llama-matrix prune`).

**Why here, not beside the weights.** A footprint is a property of *(model, box)*,
not of the model alone - the same weights on a different GPU measure differently. A
sidecar next to the weights would carry a box-specific number across box boundaries
(and often the weights dir is a read-only mount). Keeping the store in the
config folder scopes it correctly to the box that measured it, alongside the
box-level baseline/budget it belongs with.

---

## 4. Building the matrix

### 4.1 Units: variant-collapse → logical models

Before any knapsack, interchangeable variants of one model (same weights across
quants, and `-nothink` twins) collapse into a **logical model**, sized by its
largest measured quant. Under the default `flat` strategy each logical model is an
independent knapsack unit. Under an opt-in reduction strategy (`family`), a
user-declared group of distinct models becomes one unit instead (see `SPEC.md`).

### 4.2 Roles

- **aux** - small, always-useful service models (embeddings, rerank, STT, TTS
  proxy). They **ride along** with everything: their cost is reserved in every
  combination so a request for one never evicts an LLM. Type-derived by default,
  overridable in `[roles]`.
- **images** - image models are small and all fit together, so they form a single
  co-resident pool joined with `&` (any subset valid).
- **llm** - the logical models the knapsack combines.

### 4.3 The fit predicate and the knapsack

```
fits(units) := baseline + Σ solo[u] + aux_cost ≤ ceiling      # aux always reserved
```

A logical model is **heavy** if it can't co-reside with even the smallest other
unit (`baseline + size + min_other + aux_cost > ceiling`) - a footprint fact, not a
config flag. Heavies are emitted alone (+ aux + any images that still fit).

For the non-heavy units, enumerate the **maximal** fitting subsets (a recursive
knapsack). Because llama-swap treats *any subset of a declared set as valid*,
emitting only maximal groups is sufficient - a declared `{a,b}` also licenses `{a}`
and `{b}` alone. This keeps the set count and the DSL fan-out small.

Maximal packs are recorded **inline** during the walk (a pack is maximal iff no
unit outside it still fits) rather than enumerating every fitting subset and
filtering - the filter was quadratic in the subset count and could hang on a large
light-unit roster. The common "whole light roster co-resides" case short-circuits
to a single pack without recursing. Enumerating maximal packs is nonetheless
worst-case exponential (many distinct pairwise-fitting units yield ~C(n,k) packs),
so the walk runs under a work budget; if it overruns, the packs found so far are
kept (a safe under-declaration - a smaller matrix never OOMs) and the build fails
over via `on_overflow` exactly as the 1000-combination cap does (§4.4).

The GPU is not the only budget. Each emitted set is totalled a second time against
host RAM:

```
host(units) := host_baseline + Σ host[u] ≤ host_budget − host_margin
host[u]     := d_host[u] + (declared -cram, else host_cache_gb)
```

The two are checked separately rather than knapsacked together, because they are not
the same kind of number: the GPU side is a proof from measurements, while `d_host` is
a floor (the host-side prompt cache fills with use, not at load) topped up by a
declared cap. So the host side reports rather than decides by default, under
`on_host_overflow`, and is skipped outright with a stated reason where the box or the
store cannot supply it (SPEC §7.4).

### 4.4 Emission & the 1000-combination guard

The block is a set of named DSL expressions (see `SPEC.md` §3 for the grammar):

- `aux` - the ride-along pool, referenced elsewhere as `+aux` (omitted when there
  are no aux models).
- one `g_<name>` helper per logical model with >1 variant - the quant alternatives
  (`|`), so the long OR-lists appear once and are referenced by `+g_<name>`. A
  single-variant model is referenced by its bare id (no helper).
- `images` - the image pool (`&`, + `+aux`).
- one `pack*` per maximal fitting combination of logical models (`&`, + `+aux`).
- one `llmimg_*` per logical model with the largest image subset that still fits.
- one `heavy_*` per heavy unit.
- `evict_costs` (one per model, ranked by role, §4.7). No `vars:` are emitted: sets
  use full model ids (see `SPEC.md` §3).

llama-swap caps expansion at **1000 combinations per expression** (the product of a
set's `|`-group sizes). After generation the tool counts every expression's fan-out
and the total set count; if any expression would exceed the cap it **never emits an
invalid block** - it warns (a `# WARNING:` in the block and a structured `--json`
warning) and applies the configured `on_overflow` strategy: `group` (default)
**omits** the over-cap set (a safe under-declaration - dropping a combination never
OOMs), `error` refuses the build. See `PRINCIPLES.md` #7.

The **same `on_overflow` knob** governs the other way a roster can be too large: a
maximal-pack enumeration that overruns its work budget (§4.3). There `group` keeps
the bounded packs found so far and warns; `error` refuses. Both are the identical
"the roster is too big - group it or accept less" decision, so they share one knob.

### 4.5 Footprints that are not evidence

A measurement carries whether the allocation it describes was confirmed to have
finished (§2.1, SPEC §7.2). `build` cannot re-derive that - the number looks the same
either way - so it is read from the store and turned into policy by `on_unconfirmed`:

- **`warn`** (default) plans with the footprint, then names both the models and the
  **declared sets that depend on them**, in the block's comment header and in `--json`.
  Naming the sets is the point: "these combinations may not fit" is the risk, and
  because aux rides along in every set, one unconfirmed aux model puts the whole matrix
  on that list.
- **`exclude`** drops the model from the matrix entirely, which is the same treatment an
  unmeasurable model gets (Principle #2) and a safe under-declaration.
- **`error`** refuses to build until the store has been re-measured.

The weights-on-disk floor is re-checked here too, from the stored `weights_gb`, so a
suspiciously small footprint is surfaced at the moment a matrix is generated from it and
not only in the sweep that recorded it.

### 4.6 The invariant every build asserts

For **every** emitted set: `baseline + Σ(members at max quant) + aux_cost ≤
ceiling`. A violation means the generator is unsafe - the build fails rather than
emit it.

### 4.7 Eviction costs: which model the solver keeps

Fitting decides what *may* co-reside; eviction cost decides what survives when two
declared sets both satisfy a request. llama-swap picks the set minimizing the summed
cost of the running models it would evict, so a cost is a **keep** weight. Costs are a
tie-break among sets that already fit; the fit predicate and the knapsack never see
them.

The load-bearing consequence is that a *uniform* cost turns that comparison into a
body count, and small models are numerous. On a roster of two chat models plus four
image servers, a set keeping the chat pair evicts four bodies while a set keeping the
image pool evicts one, so the solver drops the model in active use and the pair
thrashes on every alternating request, paying a full reload each time, while an image
pool nothing has touched for hours stays resident.

So the costs are **tiered by role**: `image` < `aux` < `llm`, with the `llm` tier
derived as `Σ image costs + 1` (floored at 10). Two properties are deliberate:

- **Role, not load time.** A load-proportional scale does not fix this: an image
  server loads in ~10 s and a 28 GB LLM in ~14 s, so four images still outweigh one
  LLM. The axis that separates them is what the model is *for*, not how long it takes
  to come back.
- **Scaled to the pool, not to a constant.** The guarantee wanted is "keeping a second
  conversational model beats keeping the entire idle image pool", which is a fact about
  the pool's size. A constant would hold for four image servers and fail for twelve.

A heavy takes the llm tier like any other conversational model. It appears in exactly
one declared set, so for every request that does not name it its cost is a constant
added to all candidates and cannot change the argmin; the one case where it *does*
decide something is a request for an image beside it, and there the llm tier is
already the value that keeps it.

Every model in the matrix is emitted with a cost, not only the non-default tiers, so
the block is readable as a statement of policy. A model excluded as unrunnable gets
none: nothing can evict what cannot load. The operator overrides both the tiers and
individual ids in `[evict_costs]` (`SPEC.md` §1.3).

What this **cannot** express is recency. Static weights rank roles; they cannot say
"the one I used 30 seconds ago", so two same-tier models that do not fit together still
alternate. That is `ROADMAP.md` #7.

---

## 5. Apply, verify, roll back

The apply step (invoked by `build --apply` - there is no standalone `apply` verb):

1. **Back up** the current `config.yaml`.
2. **Splice** - replace everything from the generated marker line to EOF with the
   new block. Anchoring on the marker (not on `\nmatrix:`) makes the first cutover
   and every regeneration one code path and avoids duplicating the comment header.
   The `matrix:` block must be the last top-level block.
3. **Liveness-check** - ping `/v1/models` before and after the write to confirm
   llama-swap is still serving. This does **not** load any model or touch the GPU;
   and since llama-swap keeps the old config when the new one is invalid, a pass
   means "the service survived", not "the new block parsed" - check the logs for
   certainty. `build --apply --no-verify` skips this step entirely (pure backup +
   splice, no network round-trip).
4. **Roll back** to the backup if the service stops serving after the write.

A *functional* check - loading a `pack`'s models to confirm co-residency and
eviction - costs GPU time, so it's an **optional manual step** (see WORKFLOWS
Loop 6), not part of `apply`.

`matrix:` and llama-swap's older `groups:` engine are mutually exclusive; the
generated block replaces `groups:` on first cutover.

---

## 6. Crate & module layout

A Cargo workspace (matches the house style):

```
crates/llama-matrix/            # thin CLI: clap, --json/--llm/-v, delegates to core
  src/main.rs
  src/completions.rs            # completions + man, generated from the clap def
  tests/                        # CLI + fixture-reproduction tests
crates/llama-matrix-core/
  src/lib.rs
  src/config.rs                 # parse llama-swap config.yaml (roster + cmds); macro expansion
  src/policy.rs                 # llama-matrix.toml: budget/margin/strategy/roles/groups/paths/evict_costs
  src/settings.rs               # `configure` get/set/unset/list/keys (SETTINGS table)
  src/model.rs                  # per-model record: id, cmd, type, file, mem_cmd, param_hash
  src/param_hash.rs             # strip-list → hash
  src/platform.rs               # GpuMemory trait + AMD sysfs / NVIDIA / Apple Silicon backends
  src/measure.rs                # phase 1: trigger→ready→stabilize; lockfile; failures
  src/cache.rs                  # measurements/ per-model store + retention + migrate
  src/build.rs                  # variant-collapse, roles, knapsack, heavy classification, evict costs
  src/matrix.rs                 # DSL emission + 1000-combo guard
  src/apply.rs                  # backup → splice → reload wait → verify → rollback
  src/ui.rs                     # stdout/stderr discipline + colour
```

The CLI is a thin `--json`-emitting layer; every capability is a typed function in
core. `measure` and `build` stay separate subcommands with separate side-effect
profiles (Principle #8).

---

## 7. Configuration surface

`llama-matrix.toml` holds policy, separate from llama-swap's `config.yaml`:

- **Scalars** (`config`, `endpoint`, `budget`, `margin`, `strategy`, `on_overflow`)
  are managed through `llama-matrix configure get/set/unset/list/keys` - a validated,
  shell-completable, comment-preserving surface (never hand-edit guesswork).
- **Structured tables** (`[paths]`, `[roles]`, `[groups]`, `[evict_costs]`) are
  hand-edited.

`llama-matrix setup` provisions the file on first run: it discovers the llama-swap
config, sets the endpoint, probes the GPU to auto-detect the total, and writes a
starter `llama-matrix.toml` with `budget` set to the full detected pool (plus a
comment on reserving some). To reserve room for other apps, lower it afterward with
`configure set budget <GB>`. See `SPEC.md` for the full schema and `WORKFLOWS.md`
for the operating loops.
