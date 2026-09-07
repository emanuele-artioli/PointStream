# BP57 — approved two-source acquisition and shot pilot

Antigravity executes in a NEW worktree/branch from updated origin/main. Read
AGENTS.md, PLAN.md, done/BP54-source-shortlist.md and SESSION-REPORT.md.
User explicitly approved this bounded pilot on 4 September 2026.

## Authority and hard limits

Only these provisional sources:
- AO 2024 women's final, Sabalenka–Zheng: YouTube 8EShbWpBm_0.
- US Open 2023 women's final, Gauff–Sabalenka: YouTube PH6VpEfTMVQ.

At most 45 minutes PER SOURCE, 10 GB combined downloaded media including retries,
temporary download payload and audio if any. Prefer video-only. No substitutes,
full-match fallback, training, segmentation/annotation campaign, codec comparison
or confirmation encodes. No changes to active experiment manifests or paper.
These are candidates, not accepted confirmation matches.

## Execution

1. Verify current official uploader/title and available formats before download.
   Preserve metadata evidence, URL, date, tool path/version and format IDs. oEmbed
   verifies title/uploader, not exact media duration, frame rate or access rights.
2. Select one contiguous interval per source, based on match structure (prefer
   second-set coverage), NOT PointStream quality. Record exact start/end and
   selection reason before acquisition. Verify the downloader truly limits the
   fetched interval; if it must fetch the whole match, stop and ask. Bound bytes
   operationally, not just by a resolution guess. Prefer SDR; record native
   resolution/fps. Do not upscale a 1080p source and call it native 4K.
3. Use the configured external data root under assets/confirmation_raw/bp57/.
   Respect platform access controls; do not bypass restrictions or claim that
   research automatically grants a license. Record access/redistribution limits.
   No raw media goes into Git or the paper repository.
4. ffprobe downloaded media and record exact dimensions, duration, frame rate,
   pixel format, codec, color metadata, bytes and source-file SHA256. Separate
   observed values from estimates. Preserve raw source; no overwrite on retry.
5. Metadata/shot checks only: identify candidate uninterrupted main-court windows
   of 2/4/8/16 seconds; visually check two visible players and camera cuts. Use
   existing lightweight shot tools/contact sheets, not model training or dense
   annotation. Selection stays blind to PointStream rate–quality. Record failures
   and borderline shots rather than guaranteeing 16-second yield from rally lore.
6. Audit prior-use/compilation uncertainty before any future acceptance. The
   provisional non-Alcaraz classification is not proof based on decoded footage.
   No confirmation_videos entries or confirmation_eligible=true in this pilot.

Run detached if long; record progress and bounded/resumable downloads. Stop on
source/access mismatch, uncertain cap enforcement, corruption or exhausted cap.
Do not modify Cursor's worktree, outputs or encoder implementation.

## Return

Own plans/BP57-acquisition-report.md and manifests/bp57-acquisition-pilot.json.
Return one PR: commands, paths/hashes, exact fetched intervals and formats,
per-source/combined bytes (including failed attempts), time, shot candidates,
provenance uncertainty and restrictions, submitted/succeeded/failed counts.
Propose the next annotation/acquisition step and cost; do not execute it.
No requirement to fill a shortfall by exceeding two sources or 10 GB.
