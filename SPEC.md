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
on_overflow = "group"                    # group | error  (1000-combo cap handling)

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
```

### 1.1 Scalar semantics

| key | type | default | notes |
|---|---|---|---|
| `config` | string (path) | `config.yaml` | the llama-swap config.yaml (relative to the working dir); written by `setup`, overridable per-run with `--config` |
| `endpoint` | string (URL) | `http://localhost:8080` | llama-swap address; also `--endpoint` per run |
| `budget` | float (GB) | *auto-detected total* | hard cap; resolution: `--budget` > this > detected total > **error** |
| `margin` | float (GB) | `4.0` | `ceiling = budget − margin` |
| `strategy` | enum | `flat` | `flat` = no grouping (max flexibility); `family` = collapse `[groups]` |
| `on_overflow` | enum | `group` | `group` = omit any over-cap set + warn (a safe under-declaration); `error` = refuse |

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
`[groups]` tables survive. Structured tables are **not** settable here — edit them
directly.

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
      "d_vram": 48.77, "d_gtt": 0.27,
      "abs_total": 49.21, "abs_vram": 48.92, "abs_gtt": 0.29,
      "load_s": 42.0,         // seconds to ready → feeds evict_costs
      "params": "…the hashed (memory) cmd, human-readable…",
      "measured_at": "2026-01-01"
    }
  }
}
```

The filename is the model id (a legible 1:1 with config entries). Lookup is by
param-hash regardless of filename, so a model whose id was renamed can be recovered
by scanning the directory for a matching param-hash before re-measuring.

> The VRAM/GTT split fields (`d_vram`/`d_gtt`/`abs_vram`/`abs_gtt`) are recorded as
> `0` in the current build — the platform layer reports summed occupancy, and
> `build` consumes only `d_total`. They are reserved for a future per-pool sensor.

**Consumer rule (build):** for each model, compute its param-hash from the *current*
config, read `measurements/<id>.json`, and select `measurements[hash]`. Hand-set
proxy entries not in the config worklist fall back to their sole `ok` measurement.
Use `d_total` for fit math, `load_s` for eviction cost. `FAILED` entries carry no
footprint and are excluded. A model with no `ok` measurement at the current hash is
skipped with a warning — the build runs on partial data; **missing is never treated
as fits.**

**Retention & prune:** nothing is auto-deleted — a model removed from the config
keeps its file (re-adding hits the cache). Pruning is **explicit only**
(`llama-matrix prune`), which may drop entries whose weight file is gone from disk.

**Migration:** a legacy single `measurements.json` (one blob, `models`/`baseline`/
`additivity_check` at top level) is read and split into the per-model layout on
first write; a legacy flat (one-measurement-per-model) entry is re-keyed under the
model's current param-hash.

---

## 3. The matrix DSL (what `build` emits, what llama-swap consumes)

A `matrix:` block has three sub-keys:

- **`vars`** — short alias → model id, for readable expressions. A var name wins
  over an identical model id; if minted, keep aliases to **≤8 alphanumeric
  characters** (the schema-safe bound). Vars are optional as of llama-swap v243, and
  **llama-matrix currently emits none** — sets reference full model ids directly
  (valid on v243+). The `vars:` sub-key is reserved for a future readability pass.
- **`evict_costs`** — positive integers, default 1. Higher = costlier to evict =
  prefer to keep.
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
ceiling` (Architecture §4.5) or the build fails. (llama-matrix emits no `vars:` — §3
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

Fire the request **detached** and poll `/running` for `ready` — do not await the
response (image generation blocks long after the model is resident). Even a request
that will 400 on params still triggers the load.

| type | endpoint | minimal body |
|---|---|---|
| llm | `POST /v1/chat/completions` | `{"model":M,"messages":[{"role":"user","content":"hi"}],"max_tokens":1}` |
| embed | `POST /v1/embeddings` | `{"model":M,"input":"x"}` |
| rerank | `POST /v1/rerank` | `{"model":M,"query":"x","documents":["a","b"]}` |
| stt | `POST /v1/audio/transcriptions` (multipart) | `model=M`, `file=@<tiny.wav>` |
| image | `POST /v1/images/generations` | `{"model":M,"prompt":"a cat","size":"256x256"}` |

Adding a new type = add a row here (endpoint + minimal body) and, if it lives on a
different service/port, point the trigger there. The measurement math is
type-agnostic; only the load trigger differs.

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
