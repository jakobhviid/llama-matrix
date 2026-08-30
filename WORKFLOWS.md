# WORKFLOWS.md — how you operate llama-matrix

The day-to-day loops — *what you run and when*, not the schema (`SPEC.md` is *what
you write*; `ARCHITECTURE.md` is *how it works*). Written to be driven by a human
**or** an LLM/agent: every command takes `--json`, and every interactive step has a
flag path that skips the prompt, so the whole lifecycle runs non-interactively.
Compiled into `--llm`.

## Model (read this first)

- llama-swap keeps several models resident and evicts on demand, but **trusts a
  declared `matrix:` block** — it never checks free memory. llama-matrix generates
  that block from **real measured footprints** so every declared combination
  physically fits (never OOM) while allowing as many concurrent models as fit.
- **Two phases.** `measure` loads each model alone and records its footprint
  (GPU-touching, slow, cached). `build` is pure math over those footprints and
  emits/splices the block (fast, safe anytime).
- **What llama-matrix touches.** Your llama-swap `config.yaml` (the roster —
  llama-matrix reads it, and only ever rewrites the generated block). A
  `measurements/` directory (one small JSON per model, the per-box cache
  llama-matrix owns). `llama-matrix.toml` (your policy — budget, margin, strategy).
- **Golden rule.** Under-declaring a fitting combo is safe; over-declaring OOMs. So
  after any change to a model's *memory* settings (`-c`, `-np`, quant, add/remove),
  re-measure the affected model and regenerate.

---

## Loop 0 — First-time setup

```
llama-matrix setup
```

Discovers your llama-swap `config.yaml`, sets the `endpoint`, probes the GPU to
auto-detect the total pool, and writes a starter `llama-matrix.toml` with `budget`
set to the full detected pool (plus a comment on reserving some). To reserve room
for other apps, lower it: `llama-matrix configure set budget <GB>`. Forms:

```
llama-matrix setup --config /path/config.yaml --endpoint http://host:8080   # scriptable
llama-matrix setup --json     # reports what it discovered; does not prompt
```

If no GPU sensor is found, set the budget yourself before measuring:

```
llama-matrix configure set budget 50
```

---

## Loop 1 — The core lifecycle (measure → build → apply)

The loop you run whenever the roster or a model's memory settings change:

```
llama-matrix measure                 # sweep: load each changed model, record footprint
llama-matrix build                   # preview the generated matrix block (prints; no writes)
llama-matrix build --out matrix.yaml # …or write it to a file instead of stdout
llama-matrix build --apply           # …or splice into config.yaml (backup + liveness check + rollback)
llama-matrix build --apply --no-verify   # …or a pure backup-and-splice (no network round-trip)
```

- `measure` is **incremental** — a model whose footprint-affecting flags are
  unchanged is a cache hit and is skipped. A first/full sweep loads every model
  (minutes); subsequent runs usually load nothing but the additivity combo. One
  exception: an entry whose allocation was never **confirmed** is re-measured rather
  than reused. A store holding no confirmations therefore sweeps in full (budget the
  time for it); one holding them re-loads only what is suspect.
- A sweep reports each model as it starts and finishes (`[3/26] loading x …`) on
  stderr, so a long one is visibly alive and `--json` is unaffected.
- **Quiesce the box before a sweep.** A footprint is a *solo* footprint, and any
  client that asks llama-swap for a model during a sample window puts its memory in
  someone else's number. Go looking for the periodic callers specifically: health
  probes, RAG pollers, scheduled jobs. `measure` reports what it catches (a resident
  that arrives is flagged `contended`, one that leaves fails the model outright, and
  a moved empty-pool baseline flags the run), but a quiet box needs none of that.
- Read the sweep's warnings before building. Two of them mean different things:
  *recorded WITHOUT confirming the allocation finished* is actionable (the number may
  be short - re-measure, and check the model's trigger works), while *recorded without
  confirming llama-swap loaded the measured command* is informational, and on a
  llama-swap that does not report the command it launched it can be permanent for a
  backend with no `/props`.
- **Host RAM is a second budget.** `build` totals each declared set against the host
  as well as the GPU, and a set that is over is named with the arithmetic (it is still
  emitted; set `on_host_overflow = "exclude"` to leave it out). The dominant term is
  llama.cpp's host-side prompt cache, 8192 MiB per llama-server whether or not `-cram`
  appears in the command, so bounding it with `-cram <MiB>` is usually what turns a
  warning off. A store with no `d_host` gets no host check and says so; re-measure to
  enable it.
- `build` selects each model's *current-config* footprint, collapses variants,
  runs the knapsack, and emits the block. Always preview before `--apply`. If the
  header carries an *unconfirmed footprint* warning, the sets it names are the ones
  that may not fit; re-measure, or set `on_unconfirmed = "exclude"` to leave those
  models out until you have.
- `--apply` backs up `config.yaml`, splices on the generated marker, waits for the
  hot-reload, verifies, and rolls back on any anomaly.

Add `--json` to any step to capture structured output for an agent to inspect and
feed to the next.

---

## Loop 2 — Add / change / remove a model

1. Edit your llama-swap `config.yaml` (add the stanza / change `-c`/`-np`/quant /
   remove it). It hot-reloads on its own.
2. `llama-matrix measure` — measures the new/changed footprint (unchanged models
   are cache hits; a changed one gets a **new** measurement added alongside the old,
   so reverting later is instant). Scope it with `--only <id>[,<id>]` to touch just
   one model, or force a re-measure with `--force`.
3. `llama-matrix build --apply` — regenerate and splice.

A non-memory edit (port, reasoning toggle, comments, TTL) doesn't change the
param-hash → no re-measure and the matrix is identical, so no regeneration needed.

`measure` does not take the hot-reload on trust: before recording, it confirms that
the server llama-swap actually loaded is running the command it just hashed. If you
measure a config llama-swap hasn't picked up yet (or a copy of one), the model is
reported as failed with both context sizes rather than filed under a footprint it
never had. So there is no need to pass `--force` after a memory-flag change: a
changed flag is a new hash, and a new hash is measured.

Removing a model: drop it from `config.yaml`; its `measurements/<id>.json` is
**kept** (cheap, and re-adding is then an instant hit). Nothing is auto-deleted —
run `llama-matrix prune --yes` to clear entries whose weights are gone (a bare
`prune` only previews).

---

## Loop 3 — Plan against a different budget (no re-measure)

`build` is pure, so re-target the ceiling without touching the GPU:

```
llama-matrix build --budget 96       # optimize against 96 GB (reserve the rest)
llama-matrix build --margin 6        # more fragmentation headroom
llama-matrix configure set budget 50 # make the reservation permanent
```

Lower budget → fewer packs, more units become heavy → a tighter matrix. This is how
you carve out room for other apps on the box.

---

## Loop 4 — Change the packing strategy

Default is `flat` (no grouping: any models that fit may co-reside — maximum
flexibility). Opt into grouping only to curate or to relieve the combo cap:

```
llama-matrix configure set strategy family   # collapse [groups] into single units
```

Then declare your groups in `llama-matrix.toml` under `[groups]` (see `SPEC.md` §1).
A group of distinct models becomes one mutually-exclusive slot — smaller matrix,
less flexibility.

**If a build would exceed llama-swap's 1000-combination cap**, llama-matrix never
emits an invalid block. By default (`on_overflow = "group"`) it **omits** the
over-cap set and warns (a `# WARNING:` in the block and a `--json` warning) —
omitting a combination is safe (it just declares less, never OOMs). To cover those
combinations, split the offending family in `[groups]`. Set `on_overflow = "error"`
to make it refuse the whole build instead.

---

## Loop 5 — Inspect without changing anything

```
llama-matrix drift        # current config's matrix vs what build would generate now
llama-matrix build        # print the would-be block (no writes)
llama-matrix configure list   # effective budget / margin / strategy / endpoint
```

All read-only and safe to run anytime.

---

## Loop 6 — Verify & roll back

`build --apply` does a **liveness check** automatically — it pings `/v1/models` to
confirm llama-swap is still serving, and rolls back if not. It does **not** load
models or touch the GPU. For a *functional* check (or after a `--no-verify` splice),
do this by hand:

1. Confirm a clean reload (llama-swap accepted the config; no `error`/`invalid`).
2. **Co-residency:** request the models of one `pack` in turn; confirm they all
   stay resident and the summed occupancy is under budget.
3. **Eviction:** request a heavy model; confirm the pack is evicted, aux is kept,
   and occupancy stays under budget.
4. **Rollback** if anything is off — restore the backup `config.yaml` (it
   hot-reloads), or revert the file in version control.

---

## Loop 7 - The wrong model keeps getting evicted

Symptom: two models you use together alternate on every request, each swap paying a
full reload, while models you have not touched in hours stay resident. Read the
`matrix:` decision line in the llama-swap log:

```
matrix: model=<requested> set=<chosen> evict=[<what it dropped>] cost=<n>
```

`cost` is the summed eviction cost of what the chosen set drops, and llama-swap picks
the cheapest. If it dropped the model you were using, that model was priced too low
relative to what it kept. Retune the tier, or pin the one model, in
`llama-matrix.toml`:

```toml
[evict_costs]
image = 1                    # the whole image pool is cheap to drop
llm   = 20                   # …and any chat model outranks all of it

[evict_costs.models]
"qwen3-coder-30b" = 40       # this one outranks even the other chat models
```

Then `llama-matrix build --apply` and re-request the model: the decision line should
name a set that keeps it. Costs are a tie-break among combinations that **already
fit**, so retuning them can never make a set unsafe.

Two things this cannot fix, so check them first:

- **Capacity, not policy.** If two models plus aux exceed the ceiling, no cost
  assignment holds both - they share no declared set. `llama-matrix build` shows which
  sets exist; if the pair appears in none, you need a smaller quant or a bigger budget.
- **Recency.** Costs rank *roles*, not "the model I used 30 seconds ago". Two
  equally-priced models that do not fit together will still alternate; price one of
  them above the other to break it.

---

## Agent recipe (fully non-interactive)

```
llama-matrix setup --config "$CFG" --endpoint "$EP" --json
llama-matrix measure --json                 # inspect: any FAILED models?
llama-matrix build --json                    # inspect: overflow warnings? every set fits?
llama-matrix build --apply --json            # splice + verify; check the verify result
```

Each step's `--json` is designed to be inspected and gated on before the next: a
`FAILED` model means excluded-from-matrix (surface it); an overflow warning means a
strategy decision is pending; a failed verify means the change was rolled back.

---

## Cadence

- **Routine:** run Loop 1 after any memory-affecting roster change. The incremental
  cache keeps it cheap.
- **Rare:** a full `measure --force` sweep — only when you suspect a stored number
  is stale (a suspicious footprint is only as fresh as its last real load).
- **Anytime:** `build`, `drift`, `configure list` — pure and read-only.
