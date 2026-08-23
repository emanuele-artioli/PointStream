# B′15 — Retire the pre-rewrite tree and its 433 tests

**Owns:** `src/encoder/**`, `src/decoder/**`, `src/shared/**`, and the 69
top-level `tests/test_*.py` files.

## The numbers

| Origin | Files | Tests |
|---|---|---|
| **pre-rewrite** (`tests/*.py`) | 69 | **433** |
| contracts | — | 195 |
| components | — | 295 |
| pipeline | — | 110 |
| runner | — | 11 |
| invariants | — | 12 |
| | | **~1056** |

`PLAN.md` §8 already anticipated this: *"The ~436 pre-rewrite tests are untouched
and test modules Phase B and C delete. They die with their modules; no separate
culling is needed."* 433 is that number. **The plan was right; the deletion just
has not happened**, because Phase C landed without removing what it replaced.

**623 rewrite tests across 16 component axes, the contracts, the pipeline and the
runner is not bloat** — that is roughly 40 per axis with misuse cases. The 433
are the removable half, and they are dead weight now: they slow every run, and
two of them are already `xfail` for pollution nobody will chase (`DEFERRED.md`
D6).

## The boundary is nearly clean

Only **three** modules in the pre-rewrite tree are still imported by new code:

| Module | Lines | Used by |
|---|---|---|
| `src/shared/torch_dtype.py` | 137 | `components/generation/controlnet.py` |
| `src/shared/spade4tennis_arch.py` | 138 | `components/generation/spade.py` |
| `src/decoder/animate_anyone_runtime.py` | 465 | `components/generation/animate_anyone.py` |

Everything else — 24 files and 6378 lines of `src/encoder`, most of
`src/decoder` (4175), most of `src/shared` (4782) — has no inbound edge from the
new tree.

## What to do

1. **Move the three modules** into the new tree, under `src/components/generation/`
   or a small `src/shared/` successor that respects the layer check. Keep their
   tests, ported.
2. **Confirm nothing else imports the old tree** — `src/main.py` does, so decide
   whether it is replaced by the runner CLI or retired with the rest.
3. **Delete the rest, with their tests**, in one commit per subtree so a mistake
   is easy to read and revert.
4. **Re-run the required-behaviour suite** and the layer check after each.

## Traps

**Read before deleting.** These modules are prior art we have already mined
twice — the two-naming-convention discovery came out of `tennis_dataset.py`, and
that file's *correct* positional pairing is the pattern the probe set now
follows. Grep for anything the new tree should inherit before removing a file,
and say what you took.

**`tennis_dataset.py` is still live for training** (`scripts/train_controlnet.py`
imports it) and must survive this cull, or move with the training code.

**Do not delete a test to make the suite green.** The two `xfail`ed AA tests
(D6) go when their module goes, not before, and `D5`'s architectural test is not
part of this cull at all.

**A smaller suite is not the goal; an honest one is.** If a pre-rewrite test
covers behaviour the new tree also has and does not test, port the test rather
than dropping it — and say which ones you ported.

## Done when

- The three shared modules live in the new tree with their tests.
- `src/encoder`, and the retired parts of `src/decoder` and `src/shared`, are
  gone with their tests.
- `python -m src.contracts.layers` is clean and the suite is green without the
  D6 `xfail`s.
- The report says what was ported rather than deleted, and why.
