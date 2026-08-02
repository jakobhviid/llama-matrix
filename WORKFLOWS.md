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
  (minutes); subsequent runs usually load nothing but the additivity combo.
- `build` selects each model's *current-config* footprint, collapses variants,
  runs the knapsack, and emits the block. Always preview before `--apply`.
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
