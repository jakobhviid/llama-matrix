# llama-matrix

Measure how much memory each of your [llama-swap](https://github.com/mostlygeek/llama-swap)
models really uses, then generate a co-residency `matrix:` block that lets as many
models run concurrently as physically fit — **without ever exceeding VRAM**.

llama-swap can keep several models resident at once, but its solver has **no
memory awareness**: it only trusts the combinations you declare in a `matrix:`
block, and evicts to satisfy each request without checking free memory. Declare a
combination that doesn't fit and you OOM; declare too few and you waste headroom.
llama-matrix closes that gap — it **measures** each model's real footprint on your
box, then **builds** the largest set of combinations that provably fit under a
budget you control, and splices the result into your `config.yaml`.

Two phases, two subcommands:

- **`measure`** — load each model alone, read real GPU memory occupancy after it
  stabilizes, and cache the footprint (keyed by the memory-affecting launch flags,
  so a non-memory config edit never forces a re-measure).
- **`build`** — a pure knapsack over your models that emits only combinations whose
  measured footprints sum under the ceiling, then (with `--apply`) splices the
  `matrix:` block into your `config.yaml`.

> **Status: early development.** The design is settled and a working reference
> exists; the Rust implementation is being built toward v1.0. Interfaces may change
> until the first tagged release.

## Scope

llama-matrix is **general-purpose** — for anyone running a llama-swap server with
more models than fit in memory at once. It reads a standard llama-swap
`config.yaml`, measures against whatever GPU or unified-memory box you have
(AMD and NVIDIA in v1), and writes a matrix the running llama-swap consumes. It is
not tied to any particular model roster, backend mix, or machine.

## Install

Prebuilt binaries and a Homebrew formula ship with each release. Until then, build
from source:

```sh
cargo build --release
# binary at target/release/llama-matrix
```

Once released:

```sh
# prebuilt binary — no compiler, Homebrew, or root required:
curl -fsSL https://raw.githubusercontent.com/jakobhviid/llama-matrix/main/install.sh | sh

# …or via Homebrew:
brew install jakobhviid/tap/llama-matrix
```

## Quickstart

```sh
llama-matrix setup            # find your config.yaml + endpoint, detect the GPU budget
llama-matrix measure          # load each model, record its real footprint
llama-matrix build            # preview the generated matrix block
llama-matrix build --apply    # splice it into config.yaml, wait for reload, verify
```

Reserve part of the GPU for other apps, permanently or per-run:

```sh
llama-matrix configure set budget 50    # 96 GB card? plan against 50, keep the rest
llama-matrix build --budget 96          # or just for this run
```

Every command takes `--json`, and `llama-matrix --llm` prints the full
machine-readable guide (every command plus the design) — so a human *or* an
LLM/agent can drive the whole lifecycle. See **`WORKFLOWS.md`**.

## How it works

`measure` loads each model alone and reads live GPU occupancy after allocation
settles, caching the delta over an empty baseline in a per-model measurement store
(keyed so that flipping a non-memory flag never re-measures). `build` collapses each model's
interchangeable quant/`-nothink` variants into one unit, then finds every *maximal*
combination that fits under `ceiling = budget − margin` — because llama-swap treats
any subset of a declared set as valid, declaring the maximal groups licenses all the
smaller ones too. The default strategy declares everything that fits (maximum
flexibility); grouping to shrink the matrix is opt-in. `apply` backs up your config,
splices on a generated marker, waits for the hot-reload, verifies, and rolls back on
any anomaly.

## Documentation

- **`ARCHITECTURE.md`** — the memory model, the two phases, the crate/module map.
- **`SPEC.md`** — schemas of record: `llama-matrix.toml`, the measurement store, the
  matrix DSL, the param-hash, config parsing.
- **`WORKFLOWS.md`** — the operating loops (setup → measure → build → apply),
  written to be driven by a human or an agent.
- **`PRINCIPLES.md`** — the design rules (never OOM, measure reality, fail loud, …).
- **`ROADMAP.md`** — v1.0 scope and deferred work.

All of the above are compiled into `llama-matrix --llm`.

## Known limitations

- v1 optimizes a **single** unified memory pool; multi-GPU / per-device budgets are
  on the roadmap.
- A family/logical model is sized by its largest quant (safe but slightly
  pessimistic) — see the roadmap for actual-quant sizing.
- `measure` needs a supported GPU sensor (AMD sysfs or NVIDIA); `build` works
  anywhere from an existing measurement store and a supplied `--budget`.

## AI disclosure

Parts of this project were written with AI assistance. This is the single place
that fact is disclosed; it is deliberately kept out of the commit history.

## License

MIT — see [LICENSE](LICENSE).
