# ADOPT.md — behaviour changes an existing config must know about

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
never remove one. So any attempt to *narrow* a pool was ignored — silently, with no
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

The intent had been there all along — `AUX_EVICT_COST`'s own doc comment reasons
about "the few cases where a `[roles]` override leaves an aux model out of some
sets", an outcome the additive form could never produce.

**Why it shipped:** `parses_scalars_and_tables` asserted the table was *read*
(`p.roles.aux == ["e", "r"]`). Nothing asserted it had any *effect*. A parse test on
a policy knob proves deserialization, not behaviour — the gap between those two is
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
# Now:  aux = exactly {chat-7b} — the four derived ones are demoted.
[roles]
aux = ["chat-7b"]
```

If that is not what you meant, name them all:

```toml
aux = ["chat-7b", "embed-4b", "rerank-4b", "whisper-turbo", "tts-1"]
```

This fails **visibly** — the `aux:` line and the pack count both change — where the
additive form failed silently. Regenerate and diff before applying:

```sh
llama-matrix build --out /tmp/new.yaml    # pure; touches nothing
```

### Direction of risk

Narrowing `aux` **increases** what the matrix declares co-resident, because reserved
GB become schedulable. That is a real memory commitment, so it is only safe on
*measured* footprints — which is the whole premise of the tool, but worth stating:
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
config. It fails safe — the old behaviour is strictly more conservative and cannot
OOM — but it costs co-residency with no warning. If you pin llama-matrix anywhere
(a Homebrew formula, CI, a second host), move them together, or leave `[roles]`
unset until they are aligned.

---

## Footprints are GPU-only: host RAM is not modelled (open gap, not yet fixed)

**Nothing has changed in the tool.** This entry records a limitation that the
`[roles]` change above makes materially easier to hit, so an existing config can plan
around it until the tool covers it. Filed from a live box, 2026-08-29.

### The gap

`measure` records `d_vram`, `d_gtt` and `d_total`: memory the GPU driver reports.
It records nothing about **host RAM**, so `build` packs against the GPU budget alone.
A pack that fits the ceiling can still exhaust the box it runs on.

Recent llama.cpp (upstream PR 16391; observed on b10644) ships a host-RAM prompt
cache, `-cram` / `--cache-ram`, **on by default at 8192 MiB per llama-server
process**. It is anonymous, private-dirty memory: the kernel cannot reclaim it, and
llama.cpp evicts only against its own cap, never against host pressure. Nothing in a
llama-swap `cmd` has to mention the flag for the process to take the memory, which is
why it is easy to miss.

Measured on the box that surfaced this (unified memory, 96 GB carved out as VRAM,
leaving 31.7 GB of host RAM, no swap, no zram):

| | |
|---|---|
| host RAM total | 31.7 GB |
| two resident LLMs, anonymous RSS | 9.84 GB + 9.18 GB |
| available | 2.5 GB |

Both processes were sitting at the 8 GiB cap and thrashing it:

```
W srv alloc: - making room for prompt cache entry, removing oldest entry (size = 2183.906 MiB)
```

So the per-LLM host cost is roughly `8 GiB cache + 0.6-1.3 GB baseline`, and it is
invisible to the matrix. Note it is largely a flat per-process constant set by
`-cram`, not a function of model size, so the tool does not need a per-model
measurement to bound it usefully.

### Why the `[roles]` change sharpens it

Narrowing `aux` took that box from 90 packs / 3 LLMs to 205 packs / 5 LLMs. On the
GPU that is sound and was verified live at 98.05 GB against a 107.5 GB ceiling. But
the same widening multiplies the *host* cost by the number of co-resident LLMs, and
nothing bounds it:

| co-resident LLMs | host RAM needed | on a 31.7 GB box |
|---|---|---|
| 2 | ~19 GB | fits (this is what was verified) |
| 3 | ~28.5 GB, plus 2.2 GB for a TTS sidecar | no headroom (187 of the 205 packs) |
| 4 | ~38 GB | over budget (3 of the 205 packs) |

The entry above frames the risk of narrowing `aux` purely as a GPU commitment
planned against `d_total`. That framing is incomplete: it is also a host-RAM
commitment that no measurement covers.

With no cgroup cap on the container, the failure mode is the host OOM killer picking
the largest RSS, which is a llama-server. It presents as an unexplained upstream
death, not as a matrix error, so it will not be traced back here on its own.

### What an operator should do today

Bound it in the `cmd`, because the tool will not:

```
-cram 4096      # per LLM entry; 4 co-resident LLMs then need ~21 GB, not ~38 GB
```

Choose the value as `(host RAM - OS - non-LLM services) / max co-resident LLMs`, less
headroom. `-cram 0` disables the cache outright, which is usually the wrong trade:
the cache is what avoids reprocessing a long prompt when a third conversation switches
back in, and prompt reprocessing is exactly the cost the matrix exists to avoid.

Budget one re-measure per entry when you do. `-cram` is not on the `STRIP_WITH_VALUE`
allowlist in `param_hash.rs`, so it stays in the hash and adding it produces a new
key. The conservative default behaves correctly here (an unknown flag is assumed to
matter), but the resulting measurement is close to pure waste: it will record the
same GPU footprint as the old key, because the only thing `-cram` moved was host RAM,
which nothing samples. That is the gap stated from the other direction. The strip-list
doc comment reasons entirely in GPU terms ("never a wrong cache hit, which would
under-count the matrix and could OOM"), and for a host-RAM flag both branches of that
argument are silent about the memory that actually changed.

To confirm the attribution on your own box, set `-cram` on one entry, reload, and
compare `Anonymous` in `/proc/<pid>/smaps_rollup` before and after.

### If the tool were to cover it

Sketch, not a commitment. `measure` already owns the model load, so the sample point
exists: record the process anonymous RSS alongside the GPU delta. `llama-matrix.toml`
would need a `host_budget` (with the same explicit-scalar escape hatch `budget` has,
since the sensor question is easier here), and the knapsack would need a second
dimension. A cruder first cut that would still have caught this: a configurable flat
`host_gb_per_llm`, checked against `host_budget` when emitting each pack, warning
rather than excluding.
