# BP55 — timing contract retained by Codex

Design decision recorded during BP51/BP52 integration. Not a parallel dispatch
while BP53 changes background reconstruction; implement after reviewing that path
and before a final publication-ready comparison. Runtime is measured throughout
search, but unknown split timings must remain unknown rather than be inferred.

The current runner separately records backend preparation, stage wall times,
chunk finish/scoring and aggregate assembly/scoring. Both finish paths combine
reconstruction and metrics; summing them is not codec encode or decode time.
The low-rate wrapper then scores delivered frames again. This explains why
attempt wall, runner wall and codec-only times answer different questions.

Required instrument boundaries:

- encoder: from source input already decoded to complete transmitted payload,
  including scene analysis, canonical preparation, encoder-side synthesis and
  corrective coding. Distinguish cold initialization and steady-state separately.
- client: from transmitted payload/metadata to delivered full-resolution frames,
  including chain decoding, restoration/warp, foreground synthesis and correction.
  An independent client path must not receive encoder-side objects or source pixels.
- evaluation: metric calculation, null controls and result serialization, excluded
  from encoder/client clocks. Keep their cost in the total experiment wall ledger.
- recovery: time for every attempt, checkpoint I/O and known lost work retained;
  hard-kill gaps remain labelled lower bounds, never zero-cost retries.

GPU clocks require synchronization at measured boundaries. Overlapping work cannot
be naively added into an elapsed-time claim. Full end-to-end wall and component
durations should both be retained; no subtraction of an assumed scoring estimate.

Before implementing, Codex scopes tests using test-design: controlled clocks
prove each boundary, metrics excluded from codec clocks, generation-off path,
client reconstruction correctness, and cumulative attempt accounting. Reuse
real reconstruction paths and lightweight fake clocks, not mocked codec internals.
Then delegate a bounded implementation if appropriate. No speed ranking or
real-time claim until this boundary is driven and verified.
