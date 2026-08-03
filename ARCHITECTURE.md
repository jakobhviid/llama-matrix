# ARCHITECTURE.md — how llama-matrix is built

The model behind the tool: what it reads, what it computes, what it writes, and
how the pieces fit. Read `PRINCIPLES.md` first for *why*; this is *how*. Compiled
into `--llm`.

---

## 0. The one-paragraph version

llama-swap can keep several models resident at once but has **no memory
awareness** — it only trusts a `matrix:` block that declares which model
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

- **Discrete GPU** — a single VRAM pool (NVIDIA via NVML/`nvidia-smi`; AMD via
  `amdgpu` sysfs or `rocm-smi`).
- **Unified-memory APU** — a VRAM carve-out plus a GPU-accessible system-RAM pool
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
| **budget** | how much of the pool llama-matrix may plan against — a hard user reservation | config / CLI / defaults to detected total |
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
file — the pure half never needs a sensor.

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

### 2.1 Phase 1 — measure

For each model in the config worklist:

1. Unload everything (`POST /api/models/unload`; `GET /unload` is a legacy
   fallback); settle to a clean baseline (read once per sweep).
2. Trigger the model to load via its type's endpoint (fire-and-forget; poll for
   ready rather than awaiting the response — an image generation blocks long after
   the model is already resident).
3. Poll `/running` until the model's state is `ready` — the only go signal.
   `starting` means wait; `stopping`/`stopped`/`shutdown` mean do-not-sample (a
   tearing-down model's reading is meaningless).
4. **Stabilize**: sample occupancy until two consecutive readings are within a
   small epsilon (KV and compute buffers finish allocating *after* `ready`).
5. Record the delta over baseline, the VRAM/GTT split, and the load time.
6. Unload (all, or `POST /api/models/unload/:model_id` for just this one).

Then an **additivity check**: load a real co-resident combo, compare the measured
total to `baseline + Σ(solo deltas)`. Footprints are additive to within a small
error empirically, which is what makes the knapsack valid; the residual sizes the
safety margin.

Guards: a **pid lockfile** (two concurrent sweeps share the unload primitive and
corrupt each other's readings); a **pre-check** that the weight file exists (a
missing file exits the loader instantly — skip it with a clear message rather than
burning the load timeout); and **failure classification** (missing-file vs timeout
vs premature-exit) so a broken model is recorded, not retried forever.

### 2.2 Phase 2 — build

Pure. Reads the `measurements/` store + the config + policy, computes the matrix, and
prints it (stdout), writes it (`--out FILE`), or splices it into the config
(`--apply`). No GPU, safe to run anytime. See §4.

---

## 3. Reading the roster (the anti-drift core)

Both phases derive the model worklist **from the config itself** — never a
hand-kept list, which silently drifts. A shared parsing core (mirrors the
reference `mlib`) is the single source of truth:

- **Parse** the llama-swap `config.yaml` (`models:` map) into per-model records.
- **Expand macros first.** llama-swap supports a global `macros:` map, `${macro}`
  references, and reserved `${PORT}`/`${PID}`/`${MODEL_ID}`/`${env.VAR}`
  substitutions inside `cmd`. Everything below runs on the **expanded** command —
  deriving type/file/hash from an unexpanded `${…}` is a bug (a macro-free config
  expands to itself, so this is always safe).
- **`cmd`** — the launch command (folded/literal YAML scalar → one line), expanded.
- **type** — derived from the command, not a hardcoded id-set: an image server
  binary → `image`, a whisper binary → `stt`, `--reranking` → `rerank`,
  `--embedding` → `embed`, else `llm`. (See `SPEC.md` for the exact table.)
- **primary file** — the weight path (`-m` / `--model` / `--diffusion-model` / …),
  for existence checks and pruning.
- **param-hash** — the footprint key (§3.1).
- **path mapping** — when llama-swap runs in a container, the paths in `cmd` are
  container paths; a configurable `[paths]` map resolves them to host paths (an
  identity map when llama-swap runs natively).
- **Exclusions & tolerance** — skip hand-set proxy entries and llama-swap
  **selectors / virtual model ids** (per-request routing entries, not loadable
  servers) from the measure worklist; ignore unknown model-level keys (`cmdStop`,
  `unloadTimeout`, `capabilities`, …) rather than failing on them.

### 3.1 The param-hash & the multi-measurement store

A model can have several footprints over its life (a re-quant, a context or
parallelism change). Each is stored under a **param-hash** = a hash of the launch
command reduced to only its footprint-affecting tokens (a conservative allowlist
of known-irrelevant flags — host/port, reasoning toggle, chat template, sampler
knobs, etc. — is stripped; everything else is hashed). Consequences:

- Flip a non-memory flag (e.g. reasoning off) → same hash → instant cache hit.
- Change `-c`/`-np`/quant → new hash → a new measurement is **added** alongside the
  old one. Revert later → the old hash hits instantly. Nothing is thrown away.
- The strip-list is deliberately conservative: an unlisted-but-irrelevant flag
  causes a harmless extra measure, never a wrong reuse (Principle #6).

**Storage.** The cache is a `measurements/` directory beside `llama-matrix.toml` —
**one JSON file per model** (`<model-id>.json`, holding that model's param-hash-keyed
measurement map) plus a reserved `_box.json` for the box-level values (baseline,
detected total, additivity check) that have no per-model home. Per-model files
avoid a single hand-edited blob, are cheap, and are **retained indefinitely** — a
model removed from the config keeps its file, so re-adding it hits the cache.
Pruning is **explicit only** (`llama-matrix prune`).

**Why here, not beside the weights.** A footprint is a property of *(model, box)*,
not of the model alone — the same weights on a different GPU measure differently. A
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

- **aux** — small, always-useful service models (embeddings, rerank, STT, TTS
  proxy). They **ride along** with everything: their cost is reserved in every
  combination so a request for one never evicts an LLM. Type-derived by default,
  overridable in `[roles]`.
- **images** — image models are small and all fit together, so they form a single
  co-resident pool joined with `&` (any subset valid).
- **llm** — the logical models the knapsack combines.

### 4.3 The fit predicate and the knapsack

```
fits(units) := baseline + Σ solo[u] + aux_cost ≤ ceiling      # aux always reserved
```

A logical model is **heavy** if it can't co-reside with even the smallest other
unit (`baseline + size + min_other + aux_cost > ceiling`) — a footprint fact, not a
config flag. Heavies are emitted alone (+ aux + any images that still fit).

For the non-heavy units, enumerate the **maximal** fitting subsets (a recursive
knapsack). Because llama-swap treats *any subset of a declared set as valid*,
emitting only maximal groups is sufficient — a declared `{a,b}` also licenses `{a}`
and `{b}` alone. This keeps the set count and the DSL fan-out small.

Maximal packs are recorded **inline** during the walk (a pack is maximal iff no
unit outside it still fits) rather than enumerating every fitting subset and
filtering — the filter was quadratic in the subset count and could hang on a large
light-unit roster. The common "whole light roster co-resides" case short-circuits
to a single pack without recursing. Enumerating maximal packs is nonetheless
worst-case exponential (many distinct pairwise-fitting units yield ~C(n,k) packs),
so the walk runs under a work budget; if it overruns, the packs found so far are
kept (a safe under-declaration — a smaller matrix never OOMs) and the build fails
over via `on_overflow` exactly as the 1000-combination cap does (§4.4).

### 4.4 Emission & the 1000-combination guard

The block is a set of named DSL expressions (see `SPEC.md` §3 for the grammar):

- `aux` — the ride-along pool, referenced elsewhere as `+aux` (omitted when there
  are no aux models).
- one `g_<name>` helper per logical model with >1 variant — the quant alternatives
  (`|`), so the long OR-lists appear once and are referenced by `+g_<name>`. A
  single-variant model is referenced by its bare id (no helper).
- `images` — the image pool (`&`, + `+aux`).
- one `pack*` per maximal fitting combination of logical models (`&`, + `+aux`).
- one `llmimg_*` per logical model with the largest image subset that still fits.
- one `heavy_*` per heavy unit.
- `evict_costs` (higher = costlier to evict = prefer to keep; derived from load
  time, tunable). No `vars:` are emitted — sets use full model ids (see `SPEC.md` §3).

llama-swap caps expansion at **1000 combinations per expression** (the product of a
set's `|`-group sizes). After generation the tool counts every expression's fan-out
and the total set count; if any expression would exceed the cap it **never emits an
invalid block** — it warns (a `# WARNING:` in the block and a structured `--json`
warning) and applies the configured `on_overflow` strategy: `group` (default)
**omits** the over-cap set (a safe under-declaration — dropping a combination never
OOMs), `error` refuses the build. See `PRINCIPLES.md` #7.

The **same `on_overflow` knob** governs the other way a roster can be too large: a
maximal-pack enumeration that overruns its work budget (§4.3). There `group` keeps
the bounded packs found so far and warns; `error` refuses. Both are the identical
"the roster is too big — group it or accept less" decision, so they share one knob.

### 4.5 The invariant every build asserts

For **every** emitted set: `baseline + Σ(members at max quant) + aux_cost ≤
ceiling`. A violation means the generator is unsafe — the build fails rather than
emit it.

---

## 5. Apply, verify, roll back

The apply step (invoked by `build --apply` — there is no standalone `apply` verb):

1. **Back up** the current `config.yaml`.
2. **Splice** — replace everything from the generated marker line to EOF with the
   new block. Anchoring on the marker (not on `\nmatrix:`) makes the first cutover
   and every regeneration one code path and avoids duplicating the comment header.
   The `matrix:` block must be the last top-level block.
3. **Liveness-check** — ping `/v1/models` before and after the write to confirm
   llama-swap is still serving. This does **not** load any model or touch the GPU;
   and since llama-swap keeps the old config when the new one is invalid, a pass
   means "the service survived", not "the new block parsed" — check the logs for
   certainty. `build --apply --no-verify` skips this step entirely (pure backup +
   splice, no network round-trip).
4. **Roll back** to the backup if the service stops serving after the write.

A *functional* check — loading a `pack`'s models to confirm co-residency and
eviction — costs GPU time, so it's an **optional manual step** (see WORKFLOWS
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
  src/policy.rs                 # llama-matrix.toml: budget/margin/strategy/roles/groups/paths
  src/settings.rs               # `configure` get/set/unset/list/keys (SETTINGS table)
  src/model.rs                  # per-model record: id, cmd, type, file, mem_cmd, param_hash
  src/param_hash.rs             # strip-list → hash
  src/platform.rs               # GpuMemory trait + AMD sysfs / NVIDIA / Apple Silicon backends
  src/measure.rs                # phase 1: trigger→ready→stabilize; lockfile; failures
  src/cache.rs                  # measurements/ per-model store + retention + migrate
  src/build.rs                  # variant-collapse, roles, knapsack, heavy classification
  src/matrix.rs                 # DSL emission + 1000-combo guard + evict_costs
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
  are managed through `llama-matrix configure get/set/unset/list/keys` — a validated,
  shell-completable, comment-preserving surface (never hand-edit guesswork).
- **Structured tables** (`[paths]`, `[roles]`, `[groups]`) are hand-edited.

`llama-matrix setup` provisions the file on first run: it discovers the llama-swap
config, sets the endpoint, probes the GPU to auto-detect the total, and writes a
starter `llama-matrix.toml` with `budget` set to the full detected pool (plus a
comment on reserving some). To reserve room for other apps, lower it afterward with
`configure set budget <GB>`. See `SPEC.md` for the full schema and `WORKFLOWS.md`
for the operating loops.
