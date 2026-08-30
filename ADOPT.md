# ADOPT.md - behaviour changes an existing config must know about

Upgrade notes for anyone already running llama-matrix. One entry per change that can
alter the matrix your config produces. `llama-matrix build` is pure, so every entry
here can be checked by regenerating and diffing before you apply anything.

---

## `[roles]` became authoritative instead of additive (after 1.6.0)

**If you do not set `[roles]`, nothing changes.** Roles are still derived from model
type and every existing config produces the same matrix.

### The problem

`[roles]` was documented as "override the type-derived role assignment" and
implemented as:

```rust
policy.roles.aux.contains(&model.id)          // the override
    || matches!(model.model_type, Embed | Rerank | Stt | TtsProxy)   // ...OR the derivation
```

The `||` makes the table **purely additive**. It can add a model to a pool; it can
never remove one. So any attempt to *narrow* a pool was ignored - silently, with no
warning, no error, and no diagnostic. The emitted block looked exactly as it did
before, which reads as "I misunderstood the syntax" rather than "the feature is
inert".

That made the case operators actually need inexpressible. `aux` is a **ride-along**:
its footprint is reserved in *every* declared combination so a request for an aux
model can never evict an LLM. Type derivation sweeps in anything with `--embedding`
or `--reranking`, which is right for a small always-on service and wrong for a big,
rarely-used one. On the box that surfaced this, two 4B RAG models were **14.1 GB of a
16 GB aux pool** and were used only for occasional retrieval, so 14 GB was reserved in
every pack to avoid a ~5 s cold load. There was no way to say "these are ordinary
evictable units".

The intent had been there all along - `AUX_EVICT_COST`'s own doc comment reasons
about "the few cases where a `[roles]` override leaves an aux model out of some
sets", an outcome the additive form could never produce.

**Why it shipped:** `parses_scalars_and_tables` asserted the table was *read*
(`p.roles.aux == ["e", "r"]`). Nothing asserted it had any *effect*. A parse test on
a policy knob proves deserialization, not behaviour - the gap between those two is
exactly where an inert feature hides.

### The change

A **non-empty** list for a role now **replaces** that role's type derivation. An
absent or empty list still derives.

```toml
[roles]
# The complete aux set. Type-derived embed/rerank models NOT named here become
# ordinary evictable units.
aux = ["whisper-turbo", "tts-1"]
```

### What you have to check

The trade is deliberate and it can bite: **a list written to promote one extra model
into a pool must now also name the models that would otherwise be derived**, or they
leave that pool.

```toml
# Was:  aux = derived {embed, rerank, stt, tts} PLUS chat-7b
# Now:  aux = exactly {chat-7b} - the four derived ones are demoted.
[roles]
aux = ["chat-7b"]
```

If that is not what you meant, name them all:

```toml
aux = ["chat-7b", "embed-4b", "rerank-4b", "whisper-turbo", "tts-1"]
```

This fails **visibly** (the `aux:` line and the pack count both change) where the
additive form failed silently. Regenerate and diff before applying:

```sh
llama-matrix build --out /tmp/new.yaml    # pure; touches nothing
```

### Direction of risk

Narrowing `aux` **increases** what the matrix declares co-resident, because reserved
GB become schedulable. That is a real memory commitment, so it is only safe on
*measured* footprints - which is the whole premise of the tool, but worth stating:
if you narrow `aux`, the models you demoted can now be evicted, and whatever moves
into the freed space is planned against `d_total`, not a guess.

Observed on the box that motivated the fix:

| | aux reserved | packs | max co-residency |
|---|---|---|---|
| additive (1.6.0) | 15.0 GB | 90 | 3 LLMs |
| authoritative | 1.9 GB | 205 | 5 LLMs |

Verified live afterwards, not just in the plan: three LLMs plus the demoted embedding
model resident *and* serving together at 98.05 GB against a 107.5 GB ceiling.

The cost of demotion is what you would expect and should accept knowingly: a request
for a demoted model can now evict an LLM, and vice versa, paying that model's cold
load. Only narrow `aux` for models whose reload you are willing to wait for.

### Mixed-version hazard

A config using `[roles]` to narrow a pool produces **a different matrix depending on
which binary generated it**. An older binary silently ignores the setting and emits
the wider-aux, fewer-packs block; `drift` then disagrees across versions on the same
config. It fails safe - the old behaviour is strictly more conservative and cannot
OOM - but it costs co-residency with no warning. If you pin llama-matrix anywhere
(a Homebrew formula, CI, a second host), move them together, or leave `[roles]`
unset until they are aligned.

---

## Host RAM is now a second budget (after 1.6.3)

**Nothing about the GPU fit changes.** This adds a check alongside it, and on an
existing store the check does not run until you re-measure.

### The problem

`measure` recorded `d_vram`, `d_gtt` and `d_total`: memory the GPU driver reports. It
recorded nothing about **host RAM**, so `build` packed against the GPU budget alone. A
pack that fits the ceiling can still exhaust the box it runs on, and that failure does
not present as anything the matrix reports: the host OOM killer picks the largest RSS,
which is a llama-server, and it reads as an unexplained upstream death.

Recent llama.cpp (upstream PR 16391) ships a host-RAM prompt cache, `-cram` /
`--cache-ram`, **on by default at 8192 MiB per llama-server process**. It is
anonymous, private-dirty memory: the kernel cannot reclaim it, and llama.cpp evicts
only against its own cap, never against host pressure. Nothing in a llama-swap `cmd`
has to mention the flag for the process to take the memory.

Measured on the box that surfaced this (unified memory, 96 GB carved out as VRAM,
leaving 31.7 GB of host RAM, no swap, no zram):

| | |
|---|---|
| host RAM total | 31.7 GB |
| two resident LLMs, anonymous RSS | 9.84 GB + 9.18 GB |
| available | 2.5 GB |

Both processes were sitting at the 8 GiB cap and thrashing it. The `[roles]` change
above sharpened it: narrowing `aux` took that box from 90 packs / 3 LLMs to 205 packs
/ 5 LLMs, which is sound on the GPU and was verified live at 98.05 GB against a
107.5 GB ceiling, while multiplying the *host* cost by the number of co-resident LLMs
with nothing bounding it.

### The change

`measure` reads host RAM the same way it reads the GPU pool, `total - available`
(reclaimable page cache is not memory anyone is holding). With nothing loaded that is
`host_baseline`; after a model loads the delta is its `d_host`. Both go in the store.

`d_host` is a **floor**, and the distinction is the crux: it is what the process had
dirtied by the time it was serving, and it cannot include the prompt cache, because
that fills as prompts are processed and the load-trigger processes one tiny prompt.
The cache is bounded by `-cram` instead, which `build` reads from the **live
command**, falling back to the new `host_cache_gb` (default 8.0) where the command
does not state it. So per model:

```
host_gb = d_host + (declared -cram, else host_cache_gb)
```

and `build` totals each emitted set as `host_baseline + Σ members`, against
`host_ceiling = host_budget - host_margin`. Four new scalars, all settable through
`configure`: `host_budget` (unset = the total `measure` detected), `host_margin`
(4.0), `host_cache_gb` (8.0), `on_host_overflow` (`warn`).

`-cram` is now on the param-hash strip list, so setting it costs no re-measure. It is
the one entry there justified by a flag affecting an axis the *measurement* does not
carry, rather than by being memory-neutral.

### What you have to check

- **The check does not run until you re-measure.** An existing store has no `d_host`,
  and a partial host sum is not a smaller answer, it is a wrong one, so `build` skips
  the host dimension entirely and names the models it is missing. Run
  `llama-matrix measure --force` to enable it.
- **Expect warnings the first time, and read them as news rather than as noise.** On
  the box above, essentially every pack is over the host ceiling, and that is
  correct: three co-resident LLMs need ~28.5 GB of a 31.7 GB box. The remedy is
  `-cram 4096` (or lower) on the llama-server entries, chosen as
  `(host RAM - OS - non-LLM services) / max co-resident LLMs`, less headroom.
  `-cram 0` disables the cache outright, which is usually the wrong trade: the cache
  is what avoids reprocessing a long prompt when a third conversation switches back
  in, and prompt reprocessing is exactly the cost the matrix exists to avoid.
- **`host_cache_gb` is the one assumed number.** If your llama.cpp predates the
  host-side cache, set it to `0` and the arithmetic becomes pure measurement. If you
  set `-cram` on an entry, that entry stops assuming anything.
- **It applies to your embed and rerank entries too, and that is not conservatism.**
  Measured on a 4B embedding server with `-cram` unstated: anonymous RSS 1.64 GB after
  loading, 5.88 GB after 25 embedding requests with distinct inputs, weights on the
  GPU throughout. An embed server is a llama-server. Two aux entries on a 32 GB box
  will eventually hold ~16 GB between them, before any LLM loads.
- **Nothing is excluded by default.** `on_host_overflow = "warn"` emits the set and
  names it, because one term of the host sum is a declared cap rather than a
  measurement and silently deleting packs on that basis would be the wrong trade. Set
  it to `"exclude"` once you trust your numbers.

### Direction of risk

The host check can only ever *remove* declarations, never add any, so it cannot make
a matrix less safe. Under `warn` it changes nothing at all about what is emitted. The
assumed cache term is deliberately the conservative direction: a box whose llama.cpp
has no such cache is over-warned, which costs a sentence, while the reverse would
cost an OOM kill.

## The served command is verified against llama-swap's own record (after 1.6.3)

**A measurement that was correct before is still correct.** This changes what
`measure` will *refuse* to record, and clears a warning that used to be permanent.

### The change

llama-swap v251 reports the command it launched in `GET /running`. `measure` now reads
it at the moment a model goes `ready` and compares it against the config's, on the
memory command (the exact token set the param-hash is built from). Where llama-swap
reports no command, the older `/props` context comparison still runs.

Two consequences:

- **Every backend can now be verified.** An image or STT server answers no `/props`,
  so it used to sit in `unverified_serving` with no way to ever clear it. Re-measure
  and it comes back verified.
- **More real mismatches are caught.** The `/props` path only ever compared `-c` and
  `-np`, and gave up entirely on `-c 0`. A quant swap, a `-b`/`-ub` change or a KV-type
  change between the config on disk and the config llama-swap loaded now fails the
  model instead of filing a footprint under a hash that never ran.

### What you have to check

If `measure` starts failing models with "llama-swap is serving a different command",
llama-swap is serving a config other than the one being measured. That was true before
too; it was just invisible. Reload llama-swap, or point `--config` at the file it
actually loaded. The message names the tokens that differ.

Nothing in the store is invalidated, and `build` is unaffected: this changes only what
a new sweep will record.

## Solo residency is now checked, and the baseline is per model (after 1.6.3)

**If your box is quiet while you sweep, nothing changes** beyond a few extra seconds
per model and two new fields in the store. Existing footprints are untouched; this
only affects what a *new* sweep does.

### The problem

`measure` unloaded everything, read one baseline for the whole sweep, then loaded each
model alone and recorded the delta. That is correct exactly while nothing else can put
a model into the pool, and nothing enforced it. Any client that requested a model
during a sample window made llama-swap load it, and its footprint landed in the delta
of the model under test. The contaminated reading was recorded with
`allocation_confirmed: true`, indistinguishable from a clean one.

The case that surfaced it was a **container health check in an unrelated service**,
probing a readiness endpoint that embedded a short string on every call. Interval
30 s, sample windows 25-30 s, so contamination hit roughly six times in seven, and
the inflation was exactly the embedding model's own footprint: 6.52 GB, turning a
32.16 GB entry into 38.68 GB.

**The direction of risk is not symmetric.** Contamination during a *model's* window
over-measures: wasted packs, never an overcommit. Contamination during the
once-per-sweep *baseline* read is the dangerous one, because the baseline is then too
high and **every** delta in that sweep is short by that amount, so the emitted matrix
declares combinations that do not fit. That is the one failure the tool exists to
prevent, and it was the quieter of the two.

### The change

Four mechanisms, sized to the asymmetry (SPEC §7.3).

- **The pool is cleared, not assumed cleared.** Unload, wait for `/running` to empty,
  then wait for occupancy itself to settle, the same positive evidence §7.2 already
  demanded after a load.
- **Each model's baseline is read immediately before it loads**, and stored as
  `pool_baseline`, so `abs_total - pool_baseline = d_total` is checkable afterwards.
  This removes the dangerous shape rather than detecting it: there is no longer one
  baseline whose contamination can shorten every delta in the sweep.
- **A resident that *leaves* mid-window fails the model.** It was subtracted from the
  reading and is not in it, so the delta would be short. Nothing is recorded.
- **A resident that *arrives* mid-window sets `contended: true`** and is reported by
  `measure` and again by `build`. It gates nothing in `build`, because arriving
  memory can only make a reading too high, but it does stop that reading from
  overwriting one already recorded as clean. Newer is not better when the newer
  number is known to be contaminated.

Two more numbers get checked because they were already on disk:

- **A re-measure that disagrees** with the stored footprint for the same
  `(model, param-hash)` by more than `max(0.25 GB, 2%)` names both values and the date
  of the old one. The new value is still written; it is just no longer written
  silently.
- **A moved empty-pool baseline is reported**, and an upward move flags the run. This
  is the check that catches what `/running` cannot: llama-swap can report nothing
  resident while the device still holds a model it has stopped accounting for, and
  such a reading passes every other test. Comparison against what the same box read
  last time is the only thing that sees it.

`_box.json`'s `baseline` is now the **lowest** reading taken with the pool verifiably
empty, and a sweep that never saw one keeps the stored value and says so.

### What you have to check

- **A sweep on a busy box now says so, loudly.** Warnings you have not seen before do
  not mean something new is broken; they mean it was always there. Read them before
  building.
- **A model may now fail where it previously produced a number**, if something left
  the pool during its window. That is a refusal to record a footprint known to be
  short, and it is the correct outcome. Quiesce the box and re-measure.
- **Existing entries carry no `contended` field**, which reads as "the writer ran no
  such check" and gates nothing. If you have an entry you suspect (the arithmetic
  test still applies: be suspicious of any delta that equals another entry's recorded
  footprint), `--force` it on a quiet box and let the disagreement report tell you.
- **Budget a little more sweep time.** Waiting for occupancy to settle after each
  unload costs a few seconds per model, in exchange for the baseline being a
  measurement rather than an assumption.

### Direction of risk

Every part of this either removes a way for a footprint to come out **too low** or
reports one that is **too high**. A matrix built after a clean sweep can only be the
same or slightly larger (contaminated over-measurements are now visible and
correctable); one built after a contaminated sweep is now labelled as such instead of
looking clean.

## Two ids sharing a param-hash: what actually happens

**No change, and no gap.** This entry exists to correct an earlier reading of the
store that was wrong, and to keep the real, smaller surprise that sits next to it.

### Each id is measured, not one of them

The param-hash strips flags believed memory-neutral, so two entries differing only in
such a flag share one hash. It is easy to read that as "one measurement serves both,
and the unmeasured member inherits a number nobody took". It does not: the store is
**one file per model id**, and the hash keys entries *within* a file. `measure`'s
cache lookup is `(id, param-hash)`, so a second id sharing a hash has no file of its
own, misses, and is loaded and measured like any other model.

The store on the box that raised this shows it directly. Seven `-nothink` twins share
a hash with their base id, and every one is recorded under both ids, from two separate
loads, agreeing to within 0.02 GB:

| hash | ids | footprints |
|---|---|---|
| `99c113c1f967` | `gemma-4-26b-a4b-q4qat`, `…-nothink` | 19.41, 19.42 |
| `2940c07d184a` | `qwen3.8-27b-q4kxl`, `…-nothink` | 26.97, 26.95 |
| `0a92a545d32d` | `qwen3.5-122b-a10b-q4kxl`, `…-nothink` | 80.00, 80.00 |

`build` then collapses ids that share a **weight file** into one unit sized by the
largest member, which is conservative by construction.

### The real surprise: the strip list is not symmetric

`--reasoning` is stripped; `--reasoning-budget` is not. Adding a budget to one member
of a pair therefore splits a collapsed unit in two and changes the pack set, with the
emitted diff as the only warning. That behaviour is **correct** (an unrecognised flag
must be assumed to matter, or the tool risks a wrong cache hit), but "these two flags
are treated differently" is not discoverable before the fact.

Splitting the pair gave the new hash its own measurement, and on the box that raised
this it came back **identical** to its sibling's, 32.16 GB both ways, which is the
evidence that `--reasoning-budget` is footprint-neutral. It is left in the hash
anyway: one extra measurement is the price of the rule that keeps the cache safe.

### What you have to check

- After changing a flag on one member of a pair, **diff the emitted block**. If the
  unit count moved, the pair split, and the new hash costs one measurement.
- When a newly-split hash disagrees with its sibling, suspect the measurement before
  crediting the flag. `measure` now reports a re-measure that disagrees with the
  stored value, which is the check that would have shortened the original
  investigation from hours to a line of output.
