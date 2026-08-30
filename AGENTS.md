# Agent guidelines

Instructions for any AI coding agent (Claude Code, opencode, Cursor, …) working
in this repository.

## Attribution - never attribute AI in the repo

- **Never** add AI/assistant attribution to commits or pull requests: no
  `Co-Authored-By: Claude` (or any other assistant) trailer, and no
  "🤖 Generated with …" line. Author every commit solely as the repository owner.
- AI assistance is disclosed **once**, in the README's "AI disclosure" section -
  that is the only place it belongs. Keep it out of the commit history entirely.
- If your tooling adds attribution by default, **turn it off at the source instead of
  fighting it per commit, and help the user do the same.** For Claude Code, set
  `includeCoAuthoredBy` to `false` in `~/.claude/settings.json` (it is on by default).
  A one-liner to hand the user (needs `jq`):

  ```sh
  f=~/.claude/settings.json; [ -f "$f" ] || printf '{}' > "$f"; \
    tmp=$(mktemp); jq '.includeCoAuthoredBy = false' "$f" > "$tmp" && mv "$tmp" "$f"
  ```

  Once it is off, no attribution is emitted at all and this rule holds effortlessly.

## No em-dashes

Never use an em-dash (`—`) in agent-authored text: commit messages, PR
descriptions, code comments, or prose in the docs. Use a plain hyphen, a comma, a
colon, parentheses, or just rewrite the sentence. A stray em-dash is one of the
strongest tells of machine-written text, and nothing in this repo should read as
AI-generated (the same motivation as the no-attribution rule above); agents reach
for em-dashes by default, so the rule has to be explicit.

The only em-dashes in this repository are the two that state this rule, here and in
CLAUDE.md, where the character has to appear to be named. A `grep -rn '—'` that
returns anything else is a defect.

## Releases & versioning - auto-incremented from commit type

CI cuts a release on every push to `main`, and the version is **derived
automatically from the commit history** (Conventional Commits) - nobody bumps a
version by hand, so a forgotten manual release still versions correctly. The
commit **subject prefix** decides the bump:

- `feat: …` - a new feature → **minor** bump (1.2.0 → 1.3.0)
- `fix: …` - a bug fix / hotfix → **patch** bump (1.2.3 → 1.2.4)
- `feat!: …` (or any `type!:`, e.g. `fix!:`) - a breaking change → **major** bump
  (1.4.2 → 2.0.0). Declare it with a `!` in the subject; a `BREAKING CHANGE`
  footer is **not** scanned (the version awk reads commit subjects only).
- anything else (`docs:`, `chore:`, `refactor:`, …) or an un-prefixed subject →
  **patch** bump

So **pick the right commit-subject prefix for the change** and the release version
follows automatically. Never hand-edit `version` in `Cargo.toml` to release - CI
computes and stamps it.

**One push per batch of work, not one push per commit.** Every push to `main` cuts a
release, so ten commits pushed individually burn ten version numbers and start ten
concurrent release runs. Commit as often as is useful; push when a piece of work is
finished. A public version line is not a scratchpad, and version numbers only go up:
churn cannot be undone, only avoided. (Releases are serialized by a `concurrency`
group in `release.yml`, which stops overlapping runs racing the Homebrew tap, but
serializing the runs does not un-burn the versions.)

**Green-gate before you push, or no release is cut.** The release job first runs
`cargo clippy --workspace --all-targets -- -D warnings` and the test suite; if
either fails, the push does **not** publish. `cargo build`/`cargo test` alone is
not enough - **clippy is the gate** (warnings are errors), so run
`cargo clippy --workspace --all-targets -- -D warnings` locally before every
push.

There is deliberately **no** `cargo fmt` gate. The reason, so it isn't
re-litigated: the tree is already rustfmt-shaped (the clippy gate keeps it tidy),
so a gate would catch drift that isn't happening - while forcing a `rust-toolchain.toml`
pin, since a fmt check against the floating `stable` channel breaks whenever
rustfmt's output shifts. Near-zero benefit for real ongoing cost. If you ever *do*
want it, add the toolchain pin in the same change and run it as a separate lint
workflow (never in `release.yml`, where a whitespace diff would block a release).

## Every confirmed bug gets a reproducing test

`crates/llama-matrix/tests/regressions.rs` holds one test per bug that was
*observable from the CLI*, each carrying the observation that produced it and the
commit that fixed it. It is separate from `cli.rs` deliberately: `cli.rs` says what the
CLI is supposed to do, and this says what it once did wrong, so a failure there means
a fixed bug came back rather than a feature regressing.

A bug only visible inside one function gets its reproducing test next to that
function instead. The rule is that the bug is reproduced *before* it is fixed, not
that every test lives in one file.

## The docs describe the tool as it is. They are not a log.

Every document in this repository answers one of two questions, and mixing them is
the failure this section exists to prevent.

**What the tool does** goes in the behaviour docs, in present tense, as though it had
always worked this way:

| file | answers |
|---|---|
| `README.md` | what this is and why it exists; the front door |
| `WORKFLOWS.md` | what you run and when; the operating loops |
| `SPEC.md` | schemas and contracts of record: config, store, DSL, param-hash |
| `ARCHITECTURE.md` | how it is built; the memory model and the module map |
| `PRINCIPLES.md` | the design rules, and why they are the rules |

**What the tool does not do yet** goes in `ROADMAP.md`, and nowhere else.

**What changed between two versions** goes in the commit message and the release
history. That is what they are for, they carry a date and a diff, and nothing else
has to be kept in step with them.

### ROADMAP.md

- It lists **only unbuilt work**. An item that ships is **deleted** from the file; its
  explanation moves into the behaviour docs above.
- **Never leave a "this has shipped" note behind.** A reader looking for how the tool
  behaves does not read the backlog, and would not trust it if they did. Two parallel
  descriptions also drift, and the one in the backlog is the one nobody updates.
- No "v1.0 scope" section, no summary of what already works. If a reader needs to know
  what the tool does, they are in the wrong file and the roadmap should not answer.
- An item **may** state the minimum baseline needed to define its gap ("images take
  the headroom the LLM knapsack left rather than competing for it"). It may **not**
  restate evidence, tables or measurements that live in the docs: cross-reference the
  section instead, so there is one copy and it is the maintained one.

### ADOPT.md, when it exists

`ADOPT.md` is a **temporary** file with a defined end. It exists only while there is
something an existing config must know about that is **not yet handled by the tool**:
an open gap, or a migration in flight. It is a holding area, not a document.

- The moment the implementation is complete, whatever in it is durable moves into the
  behaviour docs, and **`ADOPT.md` is deleted**. Not trimmed, not left with the
  entries marked done. Deleted.
- It must never accumulate entries for shipped behaviour. An entry that begins "this
  changed in 1.x" and then explains how the tool works is behaviour documentation
  filed under the wrong heading, where nobody looking for it will read it.
- If you find it present with every entry implemented, the correct action is to move
  what is durable and `git rm` the file. That has happened once already; the entries
  had grown into a second, parallel description of shipped behaviour.

## "Documenting the diff": the doc failure mode, named

Updating a doc or a comment is not the same as narrating the update. The reflex is
to *edit around* the stale sentence so it describes the transition: "this **is** now
checked", "it **no longer** needs root", "skipped **instead of** re-run". Every word
can be true and the doc still be wrong, because it describes your commit rather than
the software.

It costs twice. A reader cannot tell a live constraint from a dead one, so the old
state keeps steering them and they route around a problem that is already gone. It
also ages into trivia: one release on, "used to do X" is a fact about a version
nobody runs.

The tells are greppable, so grep **your own diff** before committing: `now`,
`now that`, `no longer`, `used to`, `actually`, `really`, `instead of <the old
behaviour>`, emphatic italics (`*is*` checked, arguing with a claim the reader never
saw), `(fixed in 1.2.3)`, `DONE`.

The fix, in order:

1. **Rewrite as if the new behaviour were the only one that ever existed:** present
   tense, no memory. Keep the *why* only where it is non-obvious and durable.
2. **Then ask whether the line still earns its place.** One that only made sense as
   a contrast with the old state should be *deleted*, not reworded.
3. **Put the before/after in the commit message,** the artefact built for it, kept by
   git with a date and a diff.
