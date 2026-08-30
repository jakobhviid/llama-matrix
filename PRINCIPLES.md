# PRINCIPLES.md - the rules llama-matrix holds to

The design commitments behind every command. If a change violates one of these,
the change is wrong (or the principle needs an explicit, documented amendment).
Compiled into `llama-matrix --llm` so an operator or agent can reason about *why*
the tool behaves as it does, not just *what* it does.

## 1. Never OOM. Under-declaring is safe; over-declaring is fatal.

llama-swap's solver has **no memory awareness** - it trusts the co-residency
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
stabilizes - not estimated from parameter counts, quant, or context. An
unmeasurable model (missing weights, failed load) is **excluded** from the matrix
with a visible reason, never assigned an invented number. Guessing low risks
principle #1.

**A number sampled before the allocation finished is a guess wearing a measurement's
clothes.** "Loaded" is not a self-evident state: llama-swap reports a model `ready`
when its upstream answers HTTP, which for a lazily-allocating backend happens before
the weights are resident, and a mid-load plateau is indistinguishable from a settled
reading by inspection. So the tool obtains **positive evidence** that allocation
finished (it waits for the load-trigger to complete, then for occupancy to go quiet)
and **records whether it got that evidence** with the measurement, rather than
assuming it. What cannot be confirmed is labelled unconfirmed and surfaced everywhere
it is used - never quietly promoted to a footprint (SPEC §7.2).

## 3. Flexibility first. Declare everything that fits.

By default every model is an independent unit and the tool declares **every maximal
combination that physically fits**. Any two models that fit together may co-reside -
nothing is artificially forbidden. The goal is: *any model within the memory budget
stays loaded until an incompatible request needs its space.*

Narrowing that is always **opt-in**, never the default, and there are two doors:
`[groups]` makes models mutually exclusive, and `max_models_per_set` /
`max_cache_holders_per_set` cap how many may be resident at once. Both trade
flexibility for a smaller declaration, which is safe in the direction that matters
(#1) and is the operator's call, not the tool's.

## 4. Collapse a model with itself, never distinct models (by default).

Two config entries pointing at the **same weight file** are one model wearing two
names (a `-nothink` runtime twin, an alias with different sampler flags), so they
become **one logical unit**, emitted as a `(a | b)` alternation and sized by the
largest member: the matrix has to be safe for whichever one is loaded.

Two *different* quant files are two units, each at its own measured footprint. They
are different weights that happen to share a lineage, and nothing physical stops a
box holding both. Merging them is a judgement about how you want the box used, not a
fact about memory, so it is opt-in through `[groups]` - the same door that groups
genuinely distinct models (#3), and the same one that makes an image pool mutually
exclusive.

## 5. The live config is written in exactly one place.

`measure` touches the GPU but never writes `config.yaml`. `build` is pure and
writes only its output file. Only the apply step (via `build --apply`) mutates the live
llama-swap config - always after a backup, always anchored on a generated marker,
always followed by a verify with rollback on any anomaly. One writer, one
audited path.

## 6. Safe caching: extra work beats wrong reuse.

A measurement is keyed by a hash of only the launch flags **known** to affect the
footprint (a conservative strip-list of known-irrelevant flags is removed; the
rest is hashed). The failure direction is deliberate: an unlisted-but-irrelevant
flag change causes at most a harmless *re-measure* - never a *wrong cache hit*,
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

`measure` is stateful, slow, and churns the live GPU (it evicts warm models), and so
is `validate`, which shares its guarantees (lockfile, one loader at a time).
`build` is pure, fast, and safe to run anytime. They are separate subcommands with
separate guarantees, and the incremental measurement cache means a full sweep is
rare. Never entangle them.

## 9. Machine-readable by default, human-readable by courtesy.

Every command supports `--json` for scripting and agents; human output goes to
stdout and progress/errors to stderr so a `--json` pipe stays clean. The tool is
built to be driven by a person *and* by an LLM/agent - see `WORKFLOWS.md`.

## 10. Docs are load-bearing.

`README`, `ARCHITECTURE`, `SPEC`, `WORKFLOWS`, `ROADMAP`, and this file are compiled
into `--llm`. A behaviour change ships with its doc change in the same commit. When the
code and `SPEC.md` disagree, the code wins and the doc is the bug - fix the doc.
The test: *could a fresh operator or LLM run this tool correctly from `--llm`
alone?* If a change would make them guess or fail, the doc isn't done.

## 11. General, not personal.

llama-matrix targets *any* llama-swap deployment: any model roster, any backend
mix (llama.cpp / stable-diffusion.cpp / whisper.cpp / …), any GPU or unified-memory
box, containerized or native. Nothing about one operator's hardware, hostnames, or
model list is baked into the tool - it is discovered, measured, or configured.
