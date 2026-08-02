# llama-matrix

Measure how much memory each of your [llama-swap](https://github.com/mostlygeek/llama-swap)
models really uses, then generate a co-residency `matrix:` block that lets as many
models run concurrently as physically fit — **without ever exceeding VRAM**.

llama-swap can keep several models resident at once, but its solver has **no
memory awareness**: it only trusts the combinations you declare in a `matrix:`
block. Over-declare and you OOM; under-declare and you waste headroom. llama-matrix
closes that gap — it **measures** each model's real footprint on your box, then
**builds** the largest set of combinations that provably fit under your budget.

Two phases, two subcommands:

- **measure** — load each model alone, read real GPU memory occupancy after it
  stabilizes, and cache the footprint (keyed by the memory-affecting launch flags,
  so a non-memory config edit never forces a re-measure).
- **build** — a pure knapsack over your models that emits only combinations whose
  measured footprints sum under the budget, then splices the `matrix:` block into
  your `config.yaml`.

> **Status: early development.** The design is settled and a working reference
> exists; the Rust implementation is being built. Interfaces will change until the
> first tagged release.

## Scope

llama-matrix is a **general-purpose** tool for anyone running a llama-swap server
with more models than fit in memory at once. It reads a standard llama-swap
`config.yaml`, measures against whatever GPU/APU the box exposes, and writes a
matrix the running llama-swap consumes. It is not tied to any particular model
roster, backend mix, or machine.

## Install

Prebuilt binaries and a Homebrew formula will be published with the first release.

## AI disclosure

Parts of this project were written with AI assistance. This is the single place
that fact is disclosed; it is deliberately kept out of the commit history.

## License

MIT — see [LICENSE](LICENSE).
