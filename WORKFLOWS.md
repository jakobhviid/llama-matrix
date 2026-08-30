# WORKFLOWS.md - how you operate llama-matrix

The day-to-day loops - *what you run and when*, not the schema (`SPEC.md` is *what
you write*; `ARCHITECTURE.md` is *how it works*). Written to be driven by a human
**or** an LLM/agent: every command takes `--json`, and every interactive step has a
flag path that skips the prompt, so the whole lifecycle runs non-interactively.
Compiled into `--llm`.

## Model (read this first)

- llama-swap keeps several models resident and evicts on demand, but **trusts a
  declared `matrix:` block** - it never checks free memory. llama-matrix generates
  that block from **real measured footprints** so every declared combination
  physically fits (never OOM) while allowing as many concurrent models as fit.
- **Two phases.** `measure` loads each model alone and records its footprint
  (GPU-touching, slow, cached). `build` is pure math over those footprints and
  emits/splices the block (fast, safe anytime). `validate` belongs to the first
  phase: it loads a whole declared combination and checks that the footprints really
  sum, so it evicts your warm models the same way a sweep does.
- **What llama-matrix touches.** Your llama-swap `config.yaml` (the roster -
  llama-matrix reads it, and only ever rewrites the generated block). A
  `measurements/` directory (one small JSON per model, the per-box cache
  llama-matrix owns). `llama-matrix.toml` (your policy - budget, margin, strategy).
- **Golden rule.** Under-declaring a fitting combo is safe; over-declaring OOMs. So
  after any change to a model's *memory* settings (`-c`, `-np`, quant, add/remove),
  re-measure the affected model and regenerate.

---

## Loop 0 - First-time setup

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

## Loop 1 - The core lifecycle (measure → build → apply → validate)

The loop you run whenever the roster or a model's memory settings change:

```
llama-matrix measure                 # sweep: load each changed model, record footprint
llama-matrix build                   # preview the generated matrix block (prints; no writes)
llama-matrix build --out matrix.yaml # …or write it to a file instead of stdout
llama-matrix build --apply           # …or splice into config.yaml (backup + liveness check + rollback)
llama-matrix build --apply --no-verify   # …or a pure backup-and-splice (no network round-trip)
llama-matrix validate                # …then load the tightest declared combo: does the sum hold?
```

- `measure` is **incremental** - a model whose footprint-affecting flags are
  unchanged is a cache hit and is skipped. A first/full sweep loads every model
  (minutes); subsequent runs usually load nothing at all. One
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
  warning off - and the warning names the largest value that would, so there is
  nothing to compute. A store with no `d_host` gets no host check and says so;
  re-measure to enable it.
- **`build` tells you what a model costs configured differently**, when this box has
  already measured it: the store keeps one footprint per distinct set of memory flags,
  so a model re-measured after a `-c` change keeps both. Reported largest-saving-first
  with the tokens that differ. Not a recommendation, since a smaller footprint is
  usually a smaller context, but the price is measured rather than guessed, which is
  what you want when a pack will not fit.
- `build` selects each model's *current-config* footprint, collapses variants,
  runs the knapsack, and emits the block. Always preview before `--apply`. If the
  header carries an *unconfirmed footprint* warning, the sets it names are the ones
  that may not fit; re-measure, or set `on_unconfirmed = "exclude"` to leave those
  models out until you have.
- `--apply` backs up `config.yaml`, splices on the generated marker, waits for the
  hot-reload, verifies, and rolls back on any anomaly.
- **`validate` closes the loop.** Every footprint is measured alone and then summed;
  this loads the tightest declared combination for real and compares the occupancy
  against the prediction. Run it after `--apply`, because llama-swap will not hold
  models co-resident unless the live config declares them (if it will not, `validate`
  reports which ones never became ready and records nothing). A **positive** error is
  the one that matters: the models together hold more than their solo footprints
  predicted, so every declared combination is closer to the ceiling than the plan
  says. It is reported against `margin`, which is what has to absorb it. `--set
  <name>` tests a named set instead of the tightest, for when you have a specific
  worry rather than a general one. Budget the time on an image-heavy set: an image
  model's load-trigger is a full generation at `probe_image_size`, so validating one
  costs an image per diffusion server.

Add `--json` to any step to capture structured output for an agent to inspect and
feed to the next.

---

## Loop 2 - Add / change / remove a model

1. Edit your llama-swap `config.yaml` (add the stanza / change `-c`/`-np`/quant /
   remove it). It hot-reloads on its own.
2. `llama-matrix measure` - measures the new/changed footprint (unchanged models
   are cache hits; a changed one gets a **new** measurement added alongside the old,
   so reverting later is instant). Scope it with `--only <id>[,<id>]` to touch just
   one model, or force a re-measure with `--force`.
3. `llama-matrix build --apply` - regenerate and splice.

A non-memory edit (port, reasoning toggle, comments, TTL) doesn't change the
param-hash → no re-measure and the matrix is identical, so no regeneration needed.

`measure` does not take the hot-reload on trust: before recording, it confirms that
the server llama-swap actually loaded is running the command it just hashed. If you
measure a config llama-swap hasn't picked up yet (or a copy of one), the model is
reported as failed with both context sizes rather than filed under a footprint it
never had. So there is no need to pass `--force` after a memory-flag change: a
changed flag is a new hash, and a new hash is measured.

Removing a model: drop it from `config.yaml`; its `measurements/<id>.json` is
**kept** (cheap, and re-adding is then an instant hit). Nothing is auto-deleted -
run `llama-matrix prune --yes` to clear entries whose weights are gone (a bare
`prune` only previews).

---

## Loop 3 - Plan against a different budget (no re-measure)

`build` is pure, so re-target the ceiling without touching the GPU:

```
llama-matrix build --budget 96       # optimize against 96 GB (reserve the rest)
llama-matrix build --margin 6        # more fragmentation headroom
llama-matrix configure set budget 50 # make the reservation permanent
```

Lower budget → fewer packs, more units become heavy → a tighter matrix. This is how
you carve out room for other apps on the box.

---

## Loop 4 - Change the packing strategy

Default is `flat` (no grouping: any models that fit may co-reside - maximum
flexibility). Opt into grouping only to curate or to relieve the combo cap:

```
llama-matrix configure set strategy family   # collapse [groups] into single units
```

Then declare your groups in `llama-matrix.toml` under `[groups]` (see `SPEC.md` §1).
A group of distinct models becomes one mutually-exclusive slot - smaller matrix,
less flexibility.

**If a build would exceed llama-swap's 1000-combination cap**, llama-matrix never
emits an invalid block. By default (`on_overflow = "group"`) it **omits** the
over-cap set and warns (a `# WARNING:` in the block and a `--json` warning) -
omitting a combination is safe (it just declares less, never OOMs). To cover those
combinations, split the offending family in `[groups]`. Set `on_overflow = "error"`
to make it refuse the whole build instead.

---

## Loop 5 - Inspect without changing anything

```
llama-matrix drift        # current config's matrix vs what build would generate now
llama-matrix build        # print the would-be block (no writes)
llama-matrix configure list   # effective budget / margin / strategy / endpoint
```

All read-only and safe to run anytime.

---

## Loop 6 - Verify & roll back

`build --apply` does a **liveness check** automatically - it pings `/v1/models` to
confirm llama-swap is still serving, and rolls back if not. It does **not** load
models or touch the GPU.

For a *functional* check (or after a `--no-verify` splice):

1. Confirm a clean reload (llama-swap accepted the config; no `error`/`invalid`).
2. **Co-residency:** `llama-matrix validate`. It loads the tightest declared
   combination and reports what it actually occupies against the prediction.
   `--set <name>` checks a particular one instead.
3. **Eviction** is the part still worth doing by hand: request a heavy model and
   confirm the pack is evicted, aux is kept, and occupancy stays under budget.
   Nothing automates this, because it is a claim about llama-swap's solver rather
   than about memory.
4. **Rollback** if anything is off - restore the backup `config.yaml` (it
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
llama-matrix validate --json                 # inspect: `error` positive and > margin?
```

Each step's `--json` is designed to be inspected and gated on before the next: a
`FAILED` model means excluded-from-matrix (surface it); an overflow warning means a
strategy decision is pending; a failed verify means the change was rolled back.

### What to gate on

`measure --json` reports each outcome as its own key, so a driver can act on the one
it cares about instead of parsing prose. Split by what they mean:

| key | it means | act on it? |
|---|---|---|
| `measured`, `cached`, `adopted` | ids that now have a footprint (loaded, reused, or recovered from a renamed id) | no |
| `failed` | `{id, reason}`; excluded from the matrix | **yes**, surface it |
| `skipped_missing` | the weight file is not on this host | **yes**, the config or the mount is wrong |
| `unconfirmed_allocation` | `{id, reason}`; the footprint may be **short**, which is the direction that OOMs | **yes**, re-measure |
| `no_empty_pool` | the pool was never seen empty, so every footprint here is a delta over something else | **yes**, quiesce and re-measure |
| `baseline_was` | the empty-pool baseline moved; an increase is what a pool that only *looked* empty produces | **yes** if it went up |
| `contended` | `{id, reason}`; something else was resident, so the footprint is **too high** | no, but re-measure to recover the packs |
| `changed` | `{id, previous, current, previous_measured_at}`; same box, same flags, a different number | no, but at most one of them is right |
| `unverified_serving` | llama-swap could not confirm which command it ran | no, informational |
| `below_weight_floor` | the footprint is under 0.90 of the weights on disk; partial offload is a legitimate cause | no, a signal |
| `baseline`, `detected_total`, `host_baseline`, `host_total` | the box, as measured | no |

`build` also reads back what `validate` last recorded: an additivity error the
`margin` cannot absorb becomes a warning there too, so the evidence reaches the step
that depends on it rather than only the step that produced it.

`build --json` reports the plan: `budget`, `ceiling`, `packs`, `heavies`, `sets`,
`excluded`, `unconfirmed`, `warnings`, plus the host dimension and the alternatives it
found.

| key | it means | act on it? |
|---|---|---|
| `host_ceiling` | what each set was checked against; **`null` means not checked**, which is not the same as checked-and-fine | **yes** if null and you care about host RAM |
| `host_over` | `[name, gb]` per set over that ceiling | **yes**, see `host_cram_gb` |
| `host_cram_gb` | the largest uniform `-cram` that brings them all under; `null` *with* a non-empty `host_over` means no `-cram` can, because the overrun is in measured memory | it is the fix |
| `cheaper` | per model, the cheapest other footprint this box has measured, with the tokens that differ | no, but it is where headroom is |
| `excluded` | models left out of the matrix entirely | **yes**, surface it |

`validate --json` reports one co-residency reading: `set`, `combo`, `predicted`,
`measured`, `error`, `ceiling`, `margin`, and the two ways it can be void, `absent`
and `intruders`. **A non-empty `absent` or `intruders` means nothing was recorded** and
`error` is not a measurement of anything; otherwise gate on `error > margin`.

The split that matters when deciding whether to stop: `unconfirmed_allocation`,
`no_empty_pool` and a rising `baseline_was` can leave a matrix that does not fit.
`contended` and `changed` can only leave one that is smaller than it needs to be.

---

## Cadence

- **Routine:** run Loop 1 after any memory-affecting roster change. The incremental
  cache keeps it cheap.
- **Rare:** a full `measure --force` sweep - only when you suspect a stored number
  is stale (a suspicious footprint is only as fresh as its last real load).
- **Anytime:** `build`, `drift`, `configure list` - pure and read-only.
