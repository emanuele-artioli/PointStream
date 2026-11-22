# Identity of the official TEST sample

The twelve fixed clips show twelve distinct singles matches and twenty distinct players. The pairings come from the original broadcast scoreboards, independently of the receiver, ball marks and quality comparisons. Inspect frame 12 from each SHA-identified original in `staged.json` using FFmpeg `select=eq(n\,12)`, and inspect the later source panel for match120, whose opening scoreboard is absent. The score and on-court branding distinguish the tournaments listed below. The ATP and broadcaster records corroborate the year and pairing. The sample supplies one shot per match.

| TEST ID | Players on the original source | Event | Official match record |
|---|---|---|---|
| 108 | Fritz, Draper | Munich 2024 | [ATP](https://www.atptour.com/en/news/fritz-draper-munich-2024-qf) |
| 102 | Sinner, Shelton | Shanghai 2023 | [ATP](https://www.atptour.com/en/news/sinner-shelton-shanghai-2023-tuesday) |
| 120 | Gaston, Thiem | Gstaad 2022 | [ATP match notes](https://www.atptour.com/-/media/09b23c846d834c58af56f6c93608c817.pdf) |
| 127 | Shapovalov, Safiullin | Sydney, ATP Cup 2022 | [ATP head-to-head record](https://www.atptour.com/-/media/2033aa1b663c4eefa58c71fa31cba055.pdf) |
| 105 | Sinner, Marozsan | Halle 2024 | [ATP](https://www.atptour.com/en/news/sinner-marozsan-halle-2024-thursday) |
| 116 | O'Connell, Dimitrov | Geneva 2023 | [ATP](https://www.atptour.com/en/news/fritz-ivashka-geneva-2023-thursday) |
| 107 | Marozsan, Rune | Miami 2024 | [ATP](https://www.atptour.com/en/news/zverev-marozsan-miami-2024-qf) |
| 131 | Ruud, Rune | Acapulco 2024 | [ATP](https://www.atptour.com/en/news/acapulco-2024-sf) |
| 119 | de Minaur, Wawrinka | Monte-Carlo 2024 | [ATP](https://www.atptour.com/en/news/de-minaur-wawrinka-monte-carlo-2024-tuesday) |
| 104 | Herbert, Auger-Aliassime | Marseille 2020 | [Tennis TV tournament replay](https://www.tennistv.com/tournaments/496_2020/marseille) |
| 130 | Draper, Paul | Queen's Club 2024 | [ATP](https://www.atptour.com/en/news/london-2024-qf-report-friday) |
| 126 | Evans, Rinderknech | Lyon 2024 | [ATP](https://www.atptour.com/en/news/lyon-2024-monday-report) |

The earlier Shanghai confirmation shows Arnaldi and Medvedev in 2024, a different match from Sinner–Shelton's 2023 deciding tiebreak in TEST 102. The two complete independent-coordinate foreground rallies also feature different pairings, Nadal–Kyrgios and Alcaraz–Hurkacz. The tasks have different input and annotation requirements. [Shanghai source provenance](../third-match-source-feasibility-20261122/README.md)

**Detector exposure.** All twelve matches postdate the 2019 description of the original ten-match TrackNet dataset. The published WASB tennis configuration trains on TrackNet games 1–7. Seven of these TEST events (108, 105, 107, 131, 119, 130, 126) additionally postdate the publicly released November 2023 WASB checkpoint. The exact checkpoint checksum and configuration are verified on each submitted judge run. The receiver and sender have separate exposure statements. [Original TrackNet study](https://arxiv.org/abs/1907.03698) · [WASB training audit](../k5b.md) · [Dated model zoo](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/MODEL_ZOO.md)
