# PRINCIPLES.md — the rules llama-matrix holds to

The design commitments behind every command. If a change violates one of these,
the change is wrong (or the principle needs an explicit, documented amendment).
Compiled into `llama-matrix --llm` so an operator or agent can reason about *why*
the tool behaves as it does, not just *what* it does.

## 1. Never OOM. Under-declaring is safe; over-declaring is fatal.

llama-swap's solver has **no memory awareness** — it trusts the co-residency
combinations you declare and evicts to satisfy a request without ever checking
free memory. So the declared combinations must be *exactly* the ones that
physically fit.

- **Under-declare** (leave out a combo that would have fit) → safe. The solver
  just evicts more than strictly necessary; nothing breaks.
- **Over-declare** (declare a combo that doesn't fit) → an out-of-memory crash on
  load.

Therefore the generator emits **only** combinations whose measured footprints sum
under the ceiling, and **when in doubt, leaves a combo out.** The fit guarantee is
the product's core promise; everything else is optimization within it.

## 2. Measure reality. Never guess a footprint.

A model's footprint is read from the live GPU after it loads and its allocation
stabilizes — not estimated from parameter counts, quant, or context. An
unmeasurable model (missing weights, failed load) is **excluded** from the matrix
with a visible reason, never assigned an invented number. Guessing low risks
principle #1.

## 3. Flexibility first. Declare everything that fits.

The default strategy (`flat`) treats every model as an independent unit and
declares **every maximal combination that physically fits**. Any two models that
fit together may co-reside — nothing is artificially forbidden. The goal is: *any
model within the memory budget stays loaded until an incompatible request needs
its space.* Grouping models to shrink the matrix is an **opt-in** trade of
flexibility for a smaller declaration, never the default.

## 4. Collapse a model with itself, never distinct models (by default).

The same model appearing under several names — different quant files, or a
`-nothink` runtime twin of identical weights — is **one logical unit**, sized by
its largest quant (if the big one fits, any fits). This is physically necessary
deduplication, not policy. Grouping *distinct* models is a separate, opt-in
strategy (#3).

## 5. The live config is written in exactly one place.

`measure` touches the GPU but never writes `config.yaml`. `build` is pure and
writes only its output file. Only `apply` (or `build --apply`) mutates the live
llama-swap config — always after a backup, always anchored on a generated marker,
always followed by a verify with rollback on any anomaly. One writer, one
audited path.

## 6. Safe caching: extra work beats wrong reuse.

A measurement is keyed by a hash of only the launch flags **known** to affect the
footprint (a conservative strip-list of known-irrelevant flags is removed; the
rest is hashed). The failure direction is deliberate: an unlisted-but-irrelevant
flag change causes at most a harmless *re-measure* — never a *wrong cache hit*,
which would under-count the matrix and could OOM (#1). When unsure whether a flag
affects memory, it stays in the hash.

## 7. Fail loud. Never silent.

Every boundary condition surfaces:
- No GPU sensor and no configured budget → **error** telling you how to set one,
  never a guessed budget.
- A model that won't load → recorded as `FAILED` with a reason and excluded, with
  a `# NOT measured` note in the generated block.
- A generated matrix that would exceed llama-swap's 1000-combination expansion
  cap → the tool **never emits an invalid block**; it warns loudly and applies the
  configured overflow strategy (or refuses).
- A truncation, sampling, or reduction the tool performs → logged, in both human
  output and `--json`.

## 8. Two phases, two side-effect profiles.

`measure` is stateful, slow, and churns the live GPU (it evicts warm models).
`build` is pure, fast, and safe to run anytime. They are separate subcommands with
separate guarantees, and the incremental measurement cache means a full sweep is
rare. Never entangle them.

## 9. Machine-readable by default, human-readable by courtesy.

Every command supports `--json` for scripting and agents; human output goes to
stdout and progress/errors to stderr so a `--json` pipe stays clean. The tool is
built to be driven by a person *and* by an LLM/agent — see `WORKFLOWS.md`.

## 10. Docs are load-bearing.

`README`, `ARCHITECTURE`, `SPEC`, `WORKFLOWS`, and this file are compiled into
`--llm`. A behaviour change ships with its doc change in the same commit. When the
code and `SPEC.md` disagree, the code wins and the doc is the bug — fix the doc.
The test: *could a fresh operator or LLM run this tool correctly from `--llm`
alone?* If a change would make them guess or fail, the doc isn't done.

## 11. General, not personal.

llama-matrix targets *any* llama-swap deployment: any model roster, any backend
mix (llama.cpp / stable-diffusion.cpp / whisper.cpp / …), any GPU or unified-memory
box, containerized or native. Nothing about one operator's hardware, hostnames, or
model list is baked into the tool — it is discovered, measured, or configured.
