# PRESLEY review-format alignment — 2026-11-01

The user explicitly chose to align PointStream's review manuscript with the actual PRESLEY review format. This records that user-directed design choice and a read-only comparison; it does **not** verify the live TOMM submission portal's current rules. The earlier PointStream `b910` ACM-format build was an intermediate corrected artifact, not scientifically invalid; an older `f68` style trial was superseded. The completed implementation and verified page counts are recorded below.

## Verified reference

Root's bounded read-only inspection of `/home/itec/emanuele/presley/68e8b6bb11d0dd9e62a67aef/main.tex` on gpu6 recorded PRESLEY Git HEAD `07b46952dc2783ef8857b277795fed58ff777a37`; `main.tex` SHA256 `ef306b80af22f198bb282d4112e3d632e7e06cf7fd20695b33691ee86f0387e7`. Complete small inspection receipts are in `/private/tmp/pointstream-presley-format-inspection-20261101/` and `/private/tmp/pointstream-presley-preamble-20261101/` (`stdout`, `execution.json`, `handle.json` and command/launcher files). Their captured preamble establishes:

- `\documentclass[manuscript, screen, review, anonymous, natbib=false]{acmart}`.
- `\textfloatsep`, `\floatsep`, `\intextsep`: `9pt plus 2pt minus 2pt`; `\abovecaptionskip` `4pt`; `\belowcaptionskip` `0pt`.
- `biblatex` with `datamodel=acmdatamodel`, `style=acmnumeric`, `sorting=none`; no explicit backend option in the captured preamble; `\addbibresource{references.bib}` and `\printbibliography`.

The reference also loads other packages and defines other layout details in the captured preamble. The items above are the agreed alignment profile; a matching class/profile alone does not establish TOMM portal compliance, nor does it certify scientific claims.

## PointStream implementation boundary

The user-selected profile is implemented and pushed to Overleaf as `eb7e3a3f67a3af7ce3e0da20387f6f138a66648f`. The master/named source retains all four confirmed authors, AAU and ATHENA. The anonymous review output hides the author block, funding and self-identifying URLs; research AI methods remain visible and names in cited works remain normal. This was tested against the targeted identifying strings, not every possible identity inference. The empirical checker (26 keys), five registry tests, both-variant visibility checks and fresh source builds with native Biber 2.22 passed. Each PDF has 22 body/reference plus 3 appendix pages.

The frozen package `/private/tmp/pointstream-submission-20261101-eb7e3a3/` contains `PointStream-anonymous-review.pdf` SHA256 `c40150cc7801873ca7462e9bd7ef966d263865559761051d79e8de625507d2e6`, `PointStream-named-author-copy.pdf` SHA256 `fe24d4958c5fd648d37eeb621edf7216fb2aa216a3e7b8d2434e911535a73e27` and author/editor-only `PointStream-source-AUTHOR-EDITOR-ONLY.zip` SHA256 `84fc7723b6deba5a335c601b04fe88d7a350e3176373fa1abfb827ca209c074b`; its master source is identifying. The manifest SHA256 is `7f1d8c01b332ef79591934cd62d5702dce436dd11a0111b5823676972efa5e70`. Source builds use declared external TeX dependencies and include no BBL/BST or vendored third-party styles. `b910` was a valid genuine-BibTeX intermediate, now format-superseded; the earlier `f68` style trial was unsuitable. No PRESLEY source was changed and no fresh-decoder success is claimed.

The user states the paper is under review nowhere, but coauthors have not reviewed this manuscript. Coauthor review/approval and explicit filing instruction remain; actual submission has not been authorized. The selected profile reflects the user's instruction, not independent verification of live TOMM portal rules.
