# llama-matrix

Measure how much memory each of your [llama-swap](https://github.com/mostlygeek/llama-swap)
models really uses, then generate a co-residency `matrix:` block that lets as many
models run concurrently as physically fit - **without ever exceeding VRAM**.

llama-swap can keep several models resident at once, but its solver has **no
memory awareness**: it only trusts the combinations you declare in a `matrix:`
block, and evicts to satisfy each request without checking free memory. Declare a
combination that doesn't fit and you OOM; declare too few and you waste headroom.
llama-matrix closes that gap - it **measures** each model's real footprint on your
box, then **builds** the largest set of combinations that provably fit under a
budget you control, and splices the result into your `config.yaml`.

> **Not an importance matrix.** The `matrix:` in the name is
> [llama-swap](https://github.com/mostlygeek/llama-swap)'s co-residency block (upstream
> calls it Groups V2 / Swap Matrix), which declares *which models may be resident at the
> same time*. It is unrelated to llama.cpp's `imatrix` / importance matrix, which is a
> calibration input for quantisation. This tool is a companion to llama-swap and does
> nothing to your weights.

Two phases:

- **`measure`** - load each model alone, read real GPU memory occupancy after it
  stabilizes, and cache the footprint (keyed by the memory-affecting launch flags,
  so a non-memory config edit never forces a re-measure). Host RAM is read the same
  way, because a combination that fits the GPU can still exhaust the box.
- **`build`** - a pure knapsack over your models that emits only combinations whose
  measured footprints sum under the ceiling, then (with `--apply`) splices the
  `matrix:` block into your `config.yaml`.

**v1.0 is released.** Install below, then run `llama-matrix setup`.

## Requirements

- **A running llama-swap with the matrix engine** - upstream calls it Groups V2 /
  Swap Matrix (merged in llama-swap PR #646). Quick check: load a config with a
  `matrix:` block; it should hot-reload with no "must use either groups or matrix"
  error.
- **A GPU sensor for `measure`** - AMD (`amdgpu` sysfs), NVIDIA (`nvidia-smi`), or
  Apple Silicon (Metal unified memory, read via `ioreg`). `build` needs none: it
  works from an existing measurement store and a `--budget`.
- **No** root, no compiler, and no particular process manager or container runtime.

## Works with any llama-swap setup

llama-matrix is **general-purpose and deployment-agnostic** - for anyone running a
llama-swap server with more models than fit in memory at once. It talks HTTP to your
llama-swap at a configurable endpoint (default `http://localhost:8080`), reads your
`config.yaml` wherever it lives, and auto-detects your GPU budget (AMD `amdgpu`,
NVIDIA, or Apple Silicon unified memory; APU, discrete card, or Mac). It does
**not** manage your service -
it writes the config block and lets llama-swap hot-reload - and it assumes nothing
about your process manager, container runtime, model roster, or backend mix. Nothing
about any particular machine is baked in.

## Install

```sh
# Homebrew (macOS + Linux):
brew install jakobhviid/tap/llama-matrix

# …or a prebuilt binary - no compiler, Homebrew, or root required:
curl -fsSL https://raw.githubusercontent.com/jakobhviid/llama-matrix/main/install.sh | sh

# …or from source (needs a Rust toolchain):
cargo build --release   # binary at target/release/llama-matrix
```

## Quickstart

```sh
llama-matrix setup            # find your config.yaml + endpoint, detect the GPU budget
llama-matrix measure          # load each model, record its real footprint
llama-matrix build            # preview the generated matrix block (stdout)
llama-matrix build --out m.yaml   # …or write it to a file
llama-matrix build --apply    # …or splice it into config.yaml, wait for reload, verify
llama-matrix validate         # load the tightest declared combo; does it really fit?
```

> `measure` talks to your **running** llama-swap and loads each model in turn - it
> evicts your warm models, and a first full sweep can take minutes. `build --apply`
> itself only writes the config and does a **liveness check** (it never loads models
> or touches the GPU); add `--no-verify` for a pure backup-and-splice with no network
> round-trip.

Reserve part of the GPU for other apps, permanently or per-run:

```sh
llama-matrix configure set budget 50    # 96 GB card? plan against 50, keep the rest
llama-matrix build --budget 96          # or just for this run
```

## What it produces

A `matrix:` block that llama-swap consumes - the maximal set of model combinations
that fit under your budget. On a roster where a couple of coders + a general model +
small aux services fit together, but a 122B model can't:

```yaml
matrix:
  evict_costs:                               # what to keep when something has to go
    gemma: 10                                #   chat models outrank the image pool…
    z-image-turbo: 1                         #   …which reloads in seconds
  sets:
    aux:    "embed & rerank & whisper"       # small services; ride along with every set
    pack1:  "gemma & glm-flash & +aux"       # three models co-resident, under budget
    pack2:  "coder-30b & gemma & +aux"
    heavy_qwen122: "qwen122 & +aux"          # too big to share - runs alone (+ aux)
```

**Terms:** an **aux** model is a small always-useful service reserved in every
combination; a **pack** is a maximal group of models that co-reside safely; a
**heavy** is a model too large to share with any other. **budget** is the GB
llama-matrix may plan against and **margin** is safety slack (`ceiling = budget −
margin`). (Interchangeable quant/`-nothink` variants of one model collapse into a
`+g_<name>` "pick one" helper.) The payoff: instead of one model at a time,
llama-swap keeps each declared combination resident and evicts only when an
incompatible model is requested - never OOMing.

**Eviction costs** decide *which* model goes when something has to. The defaults rank
by role (an image model is the cheapest thing to drop; a chat model outranks the whole
image pool), and you can retune a tier or pin a single model in `[evict_costs]`
(`SPEC.md` §1.3).

## Commands

| command | what it does |
|---|---|
| `setup` | discover your config + endpoint, detect the budget, write `llama-matrix.toml` |
| `measure` | load each model, record its real footprint (GPU-touching, stateful) |
| `build` | generate the matrix block; `--out FILE` to write it, `--apply` to splice it |
| `drift` | show whether the live matrix block matches a fresh build (read-only) |
| `validate` | load the tightest declared combination and check it really fits (`--set <name>` for a specific one; GPU-touching) |
| `configure` | get/set the scalar settings (budget, margin, caps, …) |
| `prune` | drop measurements whose weight files are gone (`--yes` to delete) |

Every command takes `--json`, and `llama-matrix --llm` prints the full guide (every
command plus the design) - readable by a human *or* an LLM/agent. See
**`WORKFLOWS.md`** for the operating loops.

## How it works

`measure` loads each model alone and reads live GPU occupancy once the allocation
settles, caching the delta in a per-model store keyed so that flipping a non-memory
flag never re-measures. Two words there are load-bearing.

*Settles*: llama-swap reports a model ready when its server answers, which for an
image backend is *before* it has allocated anything (the generation **is** the
allocation), so `measure` waits for the load-trigger to finish, then for occupancy to
go quiet, and records whether it got that confirmation. *Alone*: a footprint is a
**solo** footprint, so the sweep waits for occupancy to settle after each unload
rather than trusting the proxy's bookkeeping, reads each model's baseline immediately
before that model loads, and refuses or flags a reading something else was in.

`build` collapses each model's interchangeable quant/`-nothink` variants into one
unit, then finds every *maximal* combination that fits under
`ceiling = budget − margin` - because llama-swap treats any subset of a declared set
as valid, declaring the maximal groups licenses all the smaller ones too. Each
combination is totalled against **host RAM** as well, since llama.cpp holds a
host-side prompt cache of 8192 MiB per server by default and four co-resident LLMs
can exhaust a 32 GB box while sitting comfortably inside VRAM. By default it declares
everything that fits (maximum flexibility); `[groups]` collapses models you want
mutually exclusive, and `max_models_per_set` / `max_cache_holders_per_set` cap how
many may be resident at once. `build --apply` backs up your config, splices on a generated marker, waits
for the hot-reload, verifies, and rolls back on any anomaly.

Because the store keeps one footprint per distinct set of memory flags, `build` can
also tell you what a model costs configured differently *on this box*: re-measure a
model after a `-c` change and both numbers stay, so a pack that will not fit comes
with a list of measured alternatives and what each would save. Reported, never acted
on - a smaller footprint is usually a smaller context, and that trade is yours.

Every footprint is measured **alone** and then summed, so `validate` is the step that
tests whether that sum holds on your box: it loads one declared combination for real
and compares the occupancy against the prediction. A positive error is the one that
matters (the models together hold more than predicted, so every declared combination
sits closer to the ceiling than the plan says), and it is reported against the margin
that is supposed to absorb it.

This targets llama-swap's `matrix:` engine, which replaces the legacy `groups:` one
(the two are mutually exclusive). `groups:` has no memory model - it can't know which
models physically fit together - and that's the gap llama-matrix fills.

## Documentation

- **`ARCHITECTURE.md`** - the memory model, the two phases, the crate/module map.
- **`SPEC.md`** - schemas of record: `llama-matrix.toml`, the measurement store, the
  matrix DSL, the param-hash, config parsing.
- **`WORKFLOWS.md`** - the operating loops (setup → measure → build → apply),
  written to be driven by a human or an agent.
- **`PRINCIPLES.md`** - the design rules (never OOM, measure reality, fail loud, …).
- **`ROADMAP.md`** - what the tool does *not* do yet, and why each is deferred.

All of the above are compiled into `llama-matrix --llm`.

## Known limitations

- v1 optimizes a **single** unified memory pool; multi-GPU / per-device budgets are
  on the roadmap.
- Two entries pointing at the **same weight file** (a `-nothink` twin, an alias with
  different sampler flags) are one logical model, emitted as `(a | b)` and reserved at
  the larger one's footprint, since the matrix must be safe for whichever loads. Two
  *different* quant files are two units at their own footprints; merging those is
  opt-in via `[groups]`. Where an alternation's members differ in size the reservation
  is pessimistic by that difference - see the roadmap for per-variant packs.
- `measure` needs a supported GPU sensor (AMD sysfs, NVIDIA, or Apple Silicon via
  Metal unified memory); `build` works anywhere from an existing measurement store
  and a supplied `--budget`.
- An **image footprint is measured at `probe_image_size`** (default `1024x1024`): what
  a diffusion model allocates scales with the resolution it generates at, so a
  footprint probed small is only a floor for anything larger. Set it to the size you
  actually serve (`llama-matrix configure set probe_image_size 1024x1024`).
- A footprint whose allocation could not be **confirmed** (the load-trigger never
  returned, or occupancy never went quiet) is still recorded, and by default still
  planned with, while naming the sets it puts in doubt. Tighten that with
  `configure set on_unconfirmed exclude` (leave those models out) or `error` (refuse
  to build). An unconfirmed footprint is re-measured on the next sweep rather than
  reused, so a store that holds none sweeps in full.
- The **host-RAM** check needs one assumed number: how much prompt cache a
  llama-server holds when its command does not state `-cram`. It defaults to 8 GB,
  llama.cpp's own per-process default, and sets that exceed the host ceiling are named
  rather than dropped (`configure set on_host_overflow exclude` to drop them). Set
  `-cram <MiB>` per entry and nothing is assumed; set `host_cache_gb 0` if your
  llama.cpp has no such cache. A store measured before host RAM was recorded gets no
  host check, and `build` says so.
- Model **type** is inferred from the launch command (`sd-server` → image,
  `whisper-server` → stt, `--embedding`/`--reranking` → embed/rerank, else llm). An
  unusual backend binary falls back to `llm`; if its load-trigger then doesn't fit
  it's recorded `FAILED` and excluded, never mis-measured. Name it in `[types]`
  (`"my-sd-fork" = "image"`) and it is measured correctly.

## AI disclosure

Parts of this project were written with the assistance of AI coding agents (Claude
Code, opencode, and others). All changes are reviewed by the maintainer. This is the
single place that fact is disclosed; it is deliberately kept out of the commit history.

## License

MIT - see [LICENSE](LICENSE).
