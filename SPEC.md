# SPEC.md — schemas & contracts of record

The authoritative definitions. When this file and the code disagree, the **code
wins and this doc is the bug** — fix the doc (Principle #10). Covers: the
`llama-matrix.toml` policy schema, the per-model measurement-store schema, the
llama-swap config parsing contract, the param-hash, model-type derivation, load
triggers, and the matrix DSL. Compiled into `--llm`.

---

## 1. `llama-matrix.toml` — policy

llama-matrix's own config, separate from llama-swap's `config.yaml`. All keys are
optional; omission takes the documented default. Scalars are managed by
`llama-matrix configure`; structured tables are hand-edited.

```toml
# ---- scalars (managed by `llama-matrix configure set …`) ----
config      = "config.yaml"              # path to the llama-swap config.yaml
endpoint    = "http://localhost:8080"  # llama-swap base URL
budget      = 50.0                       # GB llama-matrix may plan against.
                                         #   Omit → auto-detect the physical total.
margin      = 4.0                        # GB safety slack inside the budget.
strategy    = "flat"                     # flat | family
on_overflow = "group"                    # group | error  (over-cap / too-many-combos handling)
on_unconfirmed  = "warn"                 # warn | exclude | error  (unconfirmed footprints, §7.2)
probe_image_size = "1024x1024"           # WxH the image load-trigger generates at (§7)

# ---- structured tables (hand-edited) ----

[paths]                    # container → host weight-dir mapping. Omit if native.
"/models"    = "/srv/llama/models"
"/sd-models" = "/srv/llama/sd-models"

[roles]                    # override the type-derived role assignment.
aux    = ["embed-id", "rerank-id", "whisper-id", "tts-id"]
images = ["image-a", "image-b"]

[groups]                   # only consulted by reduction strategies / on_overflow.
# A named group of DISTINCT model ids treated as one mutually-exclusive unit.
gemma = ["gemma-27b-q4", "gemma-27b-q4-nothink", "gemma-27b-abliterated-q5"]

[evict_costs]              # which model the solver keeps under pressure (§1.3).
llm   = 10                 # per-role tiers; higher = costlier to evict = prefer to keep
image = 1
aux   = 5

[evict_costs.models]       # per-model-id overrides, which win over the role tier.
"qwen3-coder-30b" = 40
```

### 1.1 Scalar semantics

| key | type | default | notes |
|---|---|---|---|
| `config` | string (path) | `config.yaml` | the llama-swap config.yaml (relative to the working dir); written by `setup`, overridable per-run with `--config` |
| `endpoint` | string (URL) | `http://localhost:8080` | llama-swap address; also `--endpoint` per run |
| `budget` | float (GB) | *auto-detected total* | hard cap; resolution: `--budget` > this > detected total > **error** |
| `margin` | float (GB) | `4.0` | `ceiling = budget − margin` |
| `strategy` | enum | `flat` | `flat` = no grouping (max flexibility); `family` = collapse `[groups]` |
| `on_overflow` | enum | `group` | applies to **both** the 1000-combination cap and an intractably large maximal-pack enumeration. `group` = drop the over-cap set / keep the bounded packs + warn (a safe under-declaration); `error` = refuse |
| `on_unconfirmed` | enum | `warn` | what `build` does with a footprint whose allocation was never confirmed (§7.2, and every entry written before that was recorded). `warn` = plan with it, but name it *and* the declared sets that depend on it; `exclude` = leave the model out of the matrix (a safe under-declaration); `error` = refuse to build |
| `probe_image_size` | string `WxH` | `1024x1024` | resolution the image load-trigger generates at. A diffusion model's allocation scales with it, so this decides what an image footprint **means** - probe at the size you actually serve, since a footprint measured at 256x256 is only a floor for anything larger |

### 1.2 The `configure` surface

`configure` manages **only scalars** (§1.1), via a single `SETTINGS` source of
truth (dotted key ↔ TOML path, value domain, description, default):

```
llama-matrix configure list            # every scalar + its effective value
llama-matrix configure keys            # settable keys (feeds shell completion)
llama-matrix configure get budget
llama-matrix configure set budget 50   # validates, writes comment-preserving
llama-matrix configure unset budget    # revert to default (auto-detect)
```

Values accept friendly forms (enums case-insensitive; floats plain). Writes use a
comment-preserving TOML editor, so hand-written comments and the `[paths]`/`[roles]`/
`[groups]`/`[evict_costs]` tables survive. Structured tables are **not** settable here;
edit them directly.

### 1.3 `[evict_costs]`: which model the solver keeps

llama-swap answers a request by picking the declared set that minimizes the summed
cost of the running models it would have to evict (§3), so a cost is a **keep**
weight: higher = costlier to evict = prefer to keep. Positive integers only.

Every model in the matrix is emitted with a cost, resolved in this order:

1. a `[evict_costs.models]` entry for that exact model id;
2. the tier for its role (`llm` / `image` / `aux`, matching the `[roles]` split);
3. the built-in for that role.

| role | built-in | why |
|---|---|---|
| `image` | `1` | a diffusion server reloads in seconds and is used in bursts, so it is the natural eviction victim |
| `aux` | `5` | reserved in nearly every set, so rarely a candidate for eviction at all |
| `llm` | `max(10, Σ image costs + 1)` | must outweigh the **whole** idle image pool, or a large enough pool wins on count alone |

The `llm` tier scales with the image pool rather than resting on a constant, because
the guarantee wanted is "keeping a second conversational model beats keeping the
entire idle image pool" - which is a fact about the pool's size, not about any one
number. A **heavy** is an llm and takes the llm tier: it occupies exactly one declared
set, so a tier of its own would change the solver's answer only when an image is
requested beside it, and there the llm tier is already the one that keeps it.

Costs are keyed by **model id**, not by logical unit, so an override on one quant of a
collapsed model leaves its twins on the role tier. A `[evict_costs.models]` key naming
no model in the matrix is a warning (a typo, or a model that is unmeasured or excluded);
a `0` or a value above 1000000 is rejected when the policy is read.

Costs are a tie-break among sets that **already fit** - they never affect the fit
predicate or the knapsack.

---

## 2. The measurement store — the measure↔build contract

Measurements live in a **`measurements/` directory** beside `llama-matrix.toml`,
**one JSON file per model** plus one reserved box-level file. Not a single blob:
per-model files are small, never hand-edited, retained indefinitely (even after a
model leaves the config), and cheap to keep — so old footprints stay cached and a
re-added model is an instant hit. The store is **per box** (footprints are a
property of `(model, box)`, so they must not travel with the weights).

```
<config-dir>/measurements/
  _box.json            # box-level values (no per-model home)
  <model-id>.json      # one per model; footprints stack here, keyed by param-hash
```

**`_box.json`:**

```json
{
  "baseline": 0.16,           // GB, empty pool occupancy
  "detected_total": 111.5,    // GB physical pool at sweep time (build may override via budget)
  "date": "2026-01-01",       // last sweep
  "additivity_check": { "combo": ["a","b","c"], "predicted": 73.15, "measured": 73.15, "error": 0.0 }
}
```

**`<model-id>.json`** — multi-measurement per model, keyed by the param-hash (§4).
Carries the model's `type`, its primary weight `file`, and a `measurements` map,
one entry per distinct footprint it has been measured at:

```json
{
  "type": "llm",              // llm | embed | rerank | stt | image | tts-proxy
  "file": "/models/Coder-70B/Coder-70B-Q4_K_M.gguf",  // in-container path
  "measurements": {
    "b7e718dc3aac": {         // param-hash of the footprint-affecting flags
      "status": "ok",         // ok | FAILED
      "d_total": 49.05,       // GB delta over baseline — the primary number
      "d_vram": 48.77, "d_gtt": 0.27,           // optional; omitted, never 0, when unknown
      "abs_total": 49.21, "abs_vram": 48.92, "abs_gtt": 0.29,
      "load_s": 42.0,         // seconds to ready → feeds evict_costs
      "allocation_confirmed": true,   // was the load-trigger seen to finish? (§7.2)
      "serving_verified": true,       // did /props confirm the served cmd? (§7.1)
      "peak_total": 49.60,    // highest delta seen while allocating (insight only)
      "weights_gb": 49.90,    // total size of the weight files the cmd names
      "params": "…the hashed (memory) cmd, human-readable…",
      "measured_at": "2026-01-01"
    }
  }
}
```

The filename is the model id (a legible 1:1 with config entries), and lookup opens
that file directly. Renaming a model id therefore orphans its file and re-measures
under the new name; recovering the old footprint by scanning the directory for a
matching param-hash is a roadmap item, not current behaviour.

> **The per-pool fields are optional.** `d_vram`/`d_gtt`/`abs_vram`/`abs_gtt` are
> written only by a backend that can separate pools (AMD `amdgpu` sysfs, which reads
> `mem_info_vram_used` + `mem_info_gtt_used`). A single-pool or unified-memory
> backend (NVIDIA, Apple Silicon) **omits them entirely** rather than writing `0`, so
> a recorded `0` always means a measured zero and never "not measured". `build`
> consumes only `d_total` either way.
>
> Entries written before this distinction existed carry a literal `0` in all four.
> A nonzero total cannot hold zero in *both* pools, so that combination is
> recognised as unpopulated and cleared to "unknown" on read (a per-entry schema
> version does not exist; `_box.json`'s `written_by` is box-level).

> **`allocation_confirmed` is evidence, not decoration.** `status: "ok"` says a number
> was recorded; this says the number is **complete** - the load-trigger was seen to
> finish, and occupancy then stopped moving (§7.2). A footprint recorded without it may
> be a mid-load plateau, which under-counts the matrix, the one error direction that
> OOMs. `false` and *absent* carry the same weight (absent = the writer recorded no
> confirmation), so every entry in a store written without the field is unconfirmed
> until re-measured; `measure` treats such an entry as a cache **miss** and `build`
> applies `on_unconfirmed`. `serving_verified` is the §7.1 sibling and is informational: it is
> permanently unobtainable on some backends, so nothing gates on it.
>
> `peak_total` is the highest delta over baseline seen while the model was allocating,
> recorded because a diffusion step can transiently allocate above what it leaves
> resident. Nothing consumes it yet (`build` plans against `d_total`); it exists so
> peak budgeting has data to work from. `weights_gb` totals the weight files the
> command names, when they were readable at measure time.

**Consumer rule (build):** for each model, compute its param-hash from the *current*
config, read `measurements/<id>.json`, and select `measurements[hash]`.

**A hash miss is a miss.** The param-hash covers every flag known to affect the
footprint, so an entry under a *different* hash was measured under different memory
flags and is not this model's footprint: reusing it would report a cache hit,
skip the re-measure, and plan the knapsack against a stale number (the exact
under-count §1 and §6 exist to prevent). The sole exception is a **hand-set proxy
entry**: a model typed `tts-proxy`, which is excluded from the measure worklist and
so is keyed by hand rather than by a config-derived hash (§6). Such an entry
resolves without a hash match, and only when it is the model's *only* `ok` entry.
The carve-out is on the model's **type**, never on "this model happens to have one
measurement".

Use `d_total` for fit math, `load_s` for eviction cost. `FAILED` entries carry no
footprint and are excluded (including at a matching hash). A model with no `ok`
measurement at the current hash is skipped with a warning: the build runs on
partial data; **missing is never treated as fits.**

**Retention & prune:** nothing is auto-deleted — a model removed from the config
keeps its file (re-adding hits the cache). A `FAILED` result likewise never overwrites
an existing `ok` footprint at the same hash: a bad load in one sweep (a rejected
trigger, a timeout during a `--force` re-measure) is no evidence against the stored
number, and clobbering it would silently drop the model from every future matrix. The
failure is reported in the sweep summary regardless. Pruning is **explicit only**
(`llama-matrix prune`), which may drop entries whose weight file is gone from disk.

**Migration:** a legacy single `measurements.json` (one blob, `models`/`baseline`/
`additivity_check` at top level) is read and split into the per-model layout on
first write; a legacy flat (one-measurement-per-model) entry is re-keyed under the
model's current param-hash. The reference tooling read both memory pools directly,
so **migrated entries usually carry a real VRAM/GTT split even where a
llama-matrix-written entry beside them has none**: a difference in provenance, not
in write path.

---

## 3. The matrix DSL (what `build` emits, what llama-swap consumes)

A `matrix:` block has three sub-keys:

- **`vars`** — short alias → model id, for readable expressions. A var name wins
  over an identical model id; if minted, keep aliases to **≤8 alphanumeric
  characters** (the schema-safe bound). Vars are optional as of llama-swap v243, and
  **llama-matrix currently emits none** — sets reference full model ids directly
  (valid on v243+). The `vars:` sub-key is reserved for a future readability pass.
- **`evict_costs`** — positive integers, default 1. Higher = costlier to evict =
  prefer to keep. llama-matrix emits one for **every** model in the matrix, so the
  block states what the solver will do rather than leaving it to be re-derived; the
  numbers come from `[evict_costs]` (§1.3).
- **`sets`** — named **DSL strings** (not lists).

**Operators:**

| op | meaning | expands |
|----|---------|---------|
| `&` | run together | `a & b` → `[a,b]` |
| `\|` | alternatives | `a \| b` → `[a]`, `[b]` |
| `()` | group | `(a\|b) & c` → `[a,c]`, `[b,c]` |
| `+ref` | inline another set | `+aux & x` |

`(a|b) & (c|d)` → `[a,c] [a,d] [b,c] [b,d]`. **Any subset of a declared set is a
valid co-resident group** — you need not load all of it; only the requested model
starts, and the set is the *maximal* group it may share space with.

**Solver on a request for X:** if X is running, forward; else among all declared
combinations containing X, pick the one minimizing the eviction cost of running
models not in it, evict the rest, start X.

**Constraints:** `matrix` XOR `groups` (mutually exclusive engines). **Expansion is
capped at 1000 combinations per expression** — the product of that expression's
`|`-group sizes.

### 3.1 Emitted set shapes

| set | form | meaning |
|---|---|---|
| `aux` | `embed & rerank & whisper` | ride-along pool (`&`) |
| `g_<name>` helper | `(q4 \| q6 \| q8)` | a logical model's quant alternatives (`\|`), referenced by `+g_<name>` — emitted **only** for a model with more than one variant |
| `images` | `img1 & img2 & +aux` | all image models co-resident (`&`), any subset valid |
| `pack<N>` | `single-a & +g_multi-b & +aux` | a maximal fitting combination of logical models |
| `llmimg_<id>` | `+g_a & img1 & img2 & +aux` | one logical model + the largest fitting image subset |
| `heavy_<id>` | `(q4 \| q6) & img1 & +aux` | a heavy unit alone + any images that still fit |

**Reference rules:** a logical model with a single variant is referenced by its
**bare id** (`chat-a`); one with multiple variants gets a `g_<name>` helper set
(the `(a | b | …)` alternation) and is referenced by `+g_<name>`. `aux` is
referenced by `+aux` (omitted entirely when there are no aux models). Quant slots
and mutually-exclusive units → `|`; co-resident pools (images, multi-unit packs) →
`&`. Every emitted set satisfies `baseline + Σ(members at max quant) + aux_cost ≤
ceiling` (Architecture §4.6) or the build fails. (llama-matrix emits no `vars:`, see §3
above; the `<name>` here are the model ids themselves.)

### 3.2 The generated marker

The block begins with a fixed marker comment line. `apply` anchors its splice on
that marker, so regeneration replaces the previous block cleanly (no duplicated
headers) and the very first cutover from `groups:` uses the same code path.

---

## 4. The param-hash

```
memory_cmd(cmd):
    drop each flag in STRIP_WITH_VALUE together with its following token
    drop each bare flag in STRIP_BARE
    keep everything else, in order
param_hash(cmd) := hash(memory_cmd(cmd))   # short hex key
```

`STRIP_WITH_VALUE` and `STRIP_BARE` are a **conservative allowlist of flags known
not to affect the footprint** (host/port/listen address, inference path, reasoning
toggle, chat-template file, cache-reuse, image sampler knobs like steps/cfg/guidance/
cache-mode, and the bare `--jinja`). Anything not on the list stays in the hash.
The risk direction is fixed by design: a stripped flag that *did* matter would be a
bug, so the list is short and audited; an unlisted flag that *doesn't* matter costs
only a harmless extra measure (Principle #6). **When adding a flag to the strip
list: if unsure whether it affects memory, don't.**

Examples of intended behavior: a `-nothink` twin (reasoning stripped) hashes equal
to its base → no separate measurement. The same weights at `-np 2 -c 262144` vs
`-np 6 -c 1572864` hash differently → two measurements under one model id.

---

## 5. llama-swap config parsing contract

llama-matrix reads a standard llama-swap `config.yaml`:

- **Macro expansion — do this FIRST, before anything below.** llama-swap configs
  may define a global `macros:` map and reference `${macro}` inside `cmd` strings,
  plus the reserved substitutions `${PORT}`, `${PID}`, `${MODEL_ID}`, and
  `${env.VAR}` (multi-pass expansion). Every downstream step (binary/type
  detection, primary-file existence check, param-hash) operates on the **expanded**
  command — hashing or stat-ing an unexpanded `${…}` placeholder is a bug. A
  macro-free config expands to itself, so this is always safe to run.
- **Worklist** = the `models:` map keys, minus (a) hand-set proxy entries and
  (b) **selectors / virtual model ids** (llama-swap's per-request routing entries
  with strategies like `warm`/`pin`/`spillover`) — those are not loadable servers,
  so `measure` must skip them. The set of ids to measure *is* the config — never a
  parallel hand-kept list.
- **`cmd`** — the launch command scalar (folded `>` or literal `|`), normalized to
  one line, then macro-expanded. First token is the binary.
- **type** — derived from `cmd` (§6), never a hardcoded id-set.
- **primary file** — the first match of `--diffusion-model`, `-m`, `--model`,
  `--llm`.
- **path mapping** — a container path is resolved to a host path via `[paths]`;
  unmapped paths pass through unchanged (native deployments).
- **Unknown keys are tolerated.** llama-swap has many model-level keys llama-matrix
  doesn't consume (`cmdStop`, `unloadTimeout`, `concurrencyLimit`, `capabilities`,
  `filters`, `metadata`, `name`, `description`, …). The parser must ignore
  unrecognized fields, never fail closed on them — and the marker-anchored splice
  preserves every non-matrix key untouched.

## 6. Model-type derivation (from `cmd`)

| detected in `cmd` | type |
|---|---|
| an image/diffusion server binary (e.g. `sd-server`) | `image` |
| a whisper server binary (e.g. `whisper-server`) | `stt` |
| `--reranking` | `rerank` |
| `--embedding` | `embed` |
| otherwise | `llm` |

Hand-set proxy entries (e.g. a fronted TTS service with a placeholder `cmd`) are
typed `tts-proxy` and excluded from the measure worklist; their footprint is set in
their `measurements/<id>.json` by hand (often ~0 GPU).

## 7. Load triggers (how `measure` forces a load)

Fire the request on its own thread (the load has to be **in flight** while `/running`
is polled for `ready`), then **wait for it to finish before sampling** - see §7.2 for
why awaiting it is load-bearing rather than tidy.

| type | endpoint | minimal body |
|---|---|---|
| llm | `POST /v1/chat/completions` | `{"model":M,"messages":[{"role":"user","content":"hi"}],"max_tokens":1}` |
| embed | `POST /v1/embeddings` | `{"model":M,"input":"x"}` |
| rerank | `POST /v1/rerank` | `{"model":M,"query":"x","documents":["a","b"]}` |
| stt | `POST /v1/audio/transcriptions` (multipart) | `model=M`, `file=@<tiny.wav>` |
| image | `POST /v1/images/generations` | `{"model":M,"prompt":"a cube","size":<probe_image_size>}` |

The image body's `size` comes from `probe_image_size` (§1.1, default `1024x1024`), not
a fixed token resolution: what a diffusion backend allocates scales with the
resolution, so the probe size is what an image footprint is a measurement *of*.

Adding a new type = add a row here (endpoint + minimal body) and, if it lives on a
different service/port, point the trigger there. The measurement math is
type-agnostic; only the load trigger differs.

### 7.1 Serving cross-check (the loaded cmd must be the hashed cmd)

`measure` derives the param-hash and `params` from the config **file**, but the load
runs through llama-swap, which serves whatever config **it** last hot-reloaded. When
those disagree (the file was edited underneath it, the reload hasn't landed, or
`--config` points at a copy), the footprint would be filed under the new hash while
describing a command that never ran: wrong data that never self-corrects, because
the hash then looks present.

No llama-swap endpoint reports a model's `cmd` (checked against v247), so the served
command is confirmed through the loaded server itself: `GET /upstream/<id>/props`
returns llama.cpp's `default_generation_settings.n_ctx` and `total_slots`.

**`-c` is not always the same quantity**, which the comparison has to respect.
Measured on one llama-swap v247 against one llama.cpp build:

| config | reported `n_ctx` | `total_slots` |
|---|---|---|
| `-c 262144 -np 2` | `131072` (`-c` divided by slots) | 2 |
| `-c 8192`, no `-np` | `8192` (`-c` itself) | 4 |

So the declared `-c` is accepted when it matches **either** the per-slot figure or
the reconstructed total (`n_ctx x total_slots`, allowing `slots - 1` tokens for the
integer division), and `total_slots` is compared **only** when the command states
`-np` (its default is neither 1 nor derivable from the command). That is still
decisive for the failure being guarded: any change to `-c` makes both readings wrong
at once.

- **Mismatch** → record nothing and report the model as failed with both numbers.
  Storing it is the one outcome that must not happen (§1 of `PRINCIPLES.md`).
- **Unconfirmable** → measure and record, but list the model in the summary's
  `unverified_serving`. There is no `/props` on an image or STT backend, and none to
  compare against when the command says `-c 0` (resolved from the model at load).
  Reported rather than passed off as verified (fail loud, never silent).

A model id missing from `GET /v1/models` is a **hint** only, used to explain a failed
load: an `unlisted` model is absent from that roster and still loadable (§8).

### 7.2 Allocation confirmation (`ready` is not `allocated`)

llama-swap reports a model `ready` when its upstream answers HTTP. For llama.cpp that
is after the allocation; for a **lazily-allocating backend it is not**. sd-server serves
immediately and allocates its weights and compute buffers when a generation actually
runs, so a footprint sampled at `ready` can be **under half** the truth, and is
non-deterministic (it captures whatever the loader happened to have allocated when
sampling started). Nothing about a plateau distinguishes it from a settled reading, so
no amount of stabilizing fixes this: the sampler cannot tell mid-load quiet from
post-load quiet.

The trigger's completion is the signal, because for such a backend **the trigger's work
is the allocation**:

1. Fire the trigger (§7), poll `/running` for `ready`, cross-check the served command
   (§7.1).
2. **Await the trigger.** A 2xx means the work that allocates has finished. A non-2xx or
   a transport failure means the model may be half-loaded or already tearing down, so
   nothing is recorded beyond a `FAILED` entry naming the status. Overrunning the wait
   budget (900 s, measured from when the trigger was **fired**, so it never stacks on
   top of the 300 s ready timeout) records the reading with
   `allocation_confirmed: false`.
3. Only then stabilize (§8), requiring three consecutive quiet samples. Occupancy still
   moving when sampling stops is also `allocation_confirmed: false`.
4. Record `allocation_confirmed` with the measurement (§2), so the distinction survives
   into `build` instead of dying in the sweep log.

Two consequences that are part of the contract:

- **No request outlives its model.** The sweep waits for the trigger (or waits it out)
  before moving on, including after a failed load, so a request fired for one model can
  never still be allocating during the next model's baseline read or sampling window.
- **An unconfirmed entry is a cache miss.** `measure` re-measures it rather than
  reporting `cached`, so no operator has to know which entries to distrust. A store
  holding no confirmations re-measures in full; one holding them re-measures only what
  is suspect.

**The weights-on-disk floor.** A fully offloaded model cannot hold much less than its
weights, so `measure` totals the weight files its command names (`weights_gb`) and flags
any footprint below **0.90** of that total, in the sweep output and again in `build`'s
warnings. A **warning, never a verdict**: partial offload (`-ngl` below all layers,
`-ot`, `--cpu-moe`) is a legitimate reason to sit lower, and two verified image
measurements sat at 0.97-0.98, which is why the floor is not 1.0. It needs no GPU and no
cooperation from the backend, which is what makes it the cheapest cross-check available.

## 8. Server control endpoints (llama-swap)

- **Unload** — `POST /api/models/unload` unloads all models (the clean-slate
  primitive between measurements); `POST /api/models/unload/:model_id` unloads one
  (useful for tightening incremental sweeps — a baseline reset still needs
  unload-all). The bare `GET /unload` still works as a **legacy fallback** but is
  no longer the documented surface; prefer the `POST /api/models/…` forms and fall
  back to `GET /unload` only if they 404 (older builds).
- `GET /running` → `{"running":[{"model","state",…}]}`. The state set is
  `ready`, `starting`, `stopping`, `stopped`, `shutdown`. **Only `ready` is a go
  signal** — poll until then; treat `starting` as wait and `stopping`/`stopped`/
  `shutdown` as "do not sample" (a tearing-down model's memory reading is
  meaningless). Reading memory right at `ready` is still too early (KV/compute
  buffers allocating; see stabilize).
- `GET /v1/models`, `GET /health` — sanity/verify. (`unlisted` models are hidden
  from `/v1/models` but still requestable — which is why the worklist comes from the
  `models:` map, not from `/v1/models`.)

## 8a. Version & compatibility

- **Requires a llama-swap build with the matrix engine** (Groups V2 / Swap Matrix,
  merged upstream via PR #646 — not experimental). Probe by loading a config with a
  `matrix:` block and confirming a clean reload (no "must use either groups or
  matrix" / unknown-key error).
- **Full model ids in sets require v243+**; older matrix builds may need `vars`.
  llama-matrix targets current llama-swap — pin and test against a known version, and
  re-verify the 1000-combination expansion cap (§3) against the build you pin (the
  solver became symbolic in v244; the cap still lives in `matrix_dsl.go`).
- Because upstream iterates quickly on the matrix engine, treat the tested version
  range as part of the contract and watch releases for memory-awareness landing
  (which would change the `build` half's value — see `ROADMAP.md`).

## 9. Not in the schema (explicitly)

- No per-model hand-written footprints in `llama-matrix.toml` — footprints live
  only in the `measurements/` store, only from a real measurement.
- No `groups:` output — llama-matrix emits `matrix:` only (the memory-aware engine).
- No idle-TTL policy management — residency is demand-driven via the matrix; TTLs
  are the operator's concern in `config.yaml`.
