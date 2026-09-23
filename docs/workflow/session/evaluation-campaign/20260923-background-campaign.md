# Background campaign — 23 September 2026

This is the background record for the later campaigns. Players are removed in
every row below, so none of these rows is a PointStream claim. A claim still
requires weighted PSNR at least as high as the anchor at no more bytes, as in
[the development campaign](20260923-development-campaign.md).

Encoder: `/opt/local/bin/ffmpeg` n7.1.1, VVC preset `faster`. Windows are 48
frames at 3840×2160. The mask is `masks_48.npz` on that window. The 11
September probe is not used; its masks were a different loader. Players are
painted out with a registered plate. The source anchor at a given QP is the
untouched frames encoded on this same path. On Federer scene 007 that source
at QP 46 is the campaign anchor: 112,295 B, foreground 21.69 dB, background
31.36 dB, weighted 24.59 dB.

Raw rows: `outputs/modular/background-arms/`.

## Rules

Weighted PSNR is `0.7 * foreground + 0.3 * background`.

- **Bytes left** are `source bytes − arm bytes`, side data included. Positive
  means room for the player.
- **Quality choice:** highest background PSNR among arms under the source.
- **Setup choice:** among arms with at least 8 kB left, the one with the lowest
  foreground PSNR needed to tie the source’s weighted score,
  `(weighted_source − 0.3 × bg_arm) / 0.7`.
- **Background-rate win:** background PSNR within 0.5 dB of the source and at
  least 8 kB under it. The player is still missing, so this is not a codec claim.
- **Latency** is recorded and does not decide. After a weighted tie or win at
  no more bytes, say whether the sender, the client, or both are faster than
  the baseline. Sender time includes the offline plate. Client time is decode
  plus that arm’s render. A faster decode with a worse picture is not a win.

## Representations

| Arm | What is coded | Side data |
|---|---|---|
| `still_frame0` | Cleaned frame 0, repeated, no warp | 10 B. Control |
| `best_frame` | Lowest background-MSE frame, players painted out, repeated, no warp | 12 B |
| `registered_panorama` | One registered plate, warped back per frame | 1,742 B at 48 frames (14 B header + 36 B per homography) |
| `cleaned_video` | The plate-inpainted frames as one video | 10 B |

Background PSNR order on a moving camera is video, then panorama, then best
frame, then frame 0. On a nearly still camera the stills sit close to the video.

## One clip in each player-size band

| Band | Clip | Mask fraction | Plate |
|---|---|---:|---|
| Small, under 0.5% | Federer scene 007 | 0.289% | 3926×2182. Best frame is index 13 |
| Medium, 0.5–2% | Alcaraz scene 000 | 0.510% | 3842×2162. Best frame is index 38 |
| Large, over 2% | Perricard scene 002 | 2.99% | 3870×2210. Best frame is index 20 |

Federer scene 001 (0.13%) repeats the small band and is not the medium clip.
Its headroom study area was a different mask.

## QP 46, against the source at the same QP

### Federer scene 007 — small

Source: 112,295 B, foreground 21.69 dB, background 31.36 dB, weighted 24.59 dB.

| Arm | Bytes | Background | Left | Required foreground above the anchor |
|---|---:|---:|---:|---:|
| Frame 0 | 35,886 | 19.20 | 76,409 | +5.2 dB |
| Best frame | 36,042 | 20.61 | 76,253 | +4.6 dB |
| Panorama | 35,763 | 23.15 | 76,532 | +3.5 dB |
| Inpainted video | 108,192 | 31.35 | 4,103 | matches the court, under 8 kB |

Setup choice: panorama, 76,532 B left, foreground must reach about 25.2 dB.
Quality choice: inpainted video. No background-rate win.

QP 50 and QP 54 keep the video within 1–2 kB of the source (2,196 B and
1,312 B left) with the court matched. The panorama’s required foreground gap
shrinks to +2.7 dB and +2.0 dB. QP 40 video is 200,711 B, over this source.
Against the separate AV1 QP 54 anchor (316,061 B, background 36.29 dB) that
QP 40 video leaves 115,350 B at background 34.26 dB.

### Alcaraz scene 000 — medium

Source: 65,149 B, foreground 21.73 dB, background 33.84 dB, weighted 25.37 dB.
The plate is two pixels larger than the frame, and the best-frame background
MSE is 18. The camera is nearly still, so a repeated frame already holds the court.

| Arm | Bytes | Background | Left | Required foreground above the anchor |
|---|---:|---:|---:|---:|
| Frame 0 | 22,965 | 31.88 | 42,184 | +0.8 dB |
| Best frame | 23,086 | 32.25 | 42,063 | +0.7 dB |
| Panorama | 24,648 | 32.33 | 40,501 | +0.65 dB |
| Inpainted video | 58,476 | 33.81 | 6,673 | matches the court, under 8 kB |

Setup choice: panorama, 40,501 B left, foreground must reach about 22.4 dB.
The court is 1.5 dB under the source. Quality choice: inpainted video, 6,673 B
left (10% under the source). No background-rate win, because 6,673 B is under
8 kB. At QP 50 the video leaves 3,521 B and the panorama’s gap is +0.47 dB.

### Perricard scene 002 — large

Source: 104,482 B, foreground 24.93 dB, background 32.45 dB.

| Arm | Bytes | Background | Left | Required foreground above the anchor |
|---|---:|---:|---:|---:|
| Frame 0 | 23,748 | 22.11 | 80,734 | |
| Best frame | 24,174 | 22.75 | 80,308 | |
| Panorama | 24,501 | 28.16 | 79,981 | about +1.8 dB |
| Inpainted video | 86,894 | 32.51 | 17,588 | −0.03 dB |

The inpainted video is the quality choice, the setup choice, and a
background-rate win: court matched, 17,588 B left (16.8%). Foreground only
has to match the anchor’s 24.9 dB. At QP 50 the same video leaves 10,387 B
with the court matched.

## What the foreground campaign inherits

Pick the band on purpose. A result on one clip does not transfer.

- **Large.** Composite on Perricard’s inpainted video at QP 46. Budget
  17,588 B. The court is already the source court, so the foreground has to
  match 24.9 dB, not beat it by several dB.
- **Medium.** Composite on Alcaraz scene 000’s panorama at QP 46. Budget
  40,501 B. Foreground has to reach about 22.4 dB, which is 0.65 dB over the
  anchor foreground. The inpainted video matches the court and leaves only
  6,673 B, under the 8 kB line.
- **Small.** Federer scene 007’s panorama at QP 46 leaves 76,532 B and needs
  foreground about 25.2 dB, 3.5 dB over the anchor. Its inpainted video matches
  the court and leaves 4,103 B. A 48-frame small player does not free 8 kB at
  a matched court.

Foreground PSNR on every background-only row is about 11–14 dB, because the
players are painted out. That number is not the foreground target.

## Latency

Ratio versus the source at the same QP. Above 1 is faster. Plate build is
104–122 s and is offline.

| Arm | Encode | Decode | Sender, plate included |
|---|---:|---:|---:|
| Still or best frame | 3.2–3.9× | 11–12× | 0.08× |
| Panorama | 3.0–3.6× | 8× | 0.08× |
| Inpainted video | 1.0× | 1.0× | 0.08× |

The client does not build the plate. A still or panorama client is faster
than VVC decode and, on the small clip, much worse on the court. The
inpainted video matches source decode time. On Perricard it is a rate win at
matched court and matched decode time, with a slower sender. Use the latency
sentence only after the weighted tie or win.

## 192 frames, projected, not measured

A real long-window encode is deferred until the 48-frame PointStream answers
are in. Fixed cost stays at the 48-frame measurement. Per-frame cost scales
by four. Background PSNR is not scaled. The one attempt to build a 192-frame
Perricard plate was still running after 45 minutes, so plate time is not
actually fixed; the sender line in this section is the assumption.

| Clip, inpainted video, QP 46 | 48-frame bytes left | 192-frame bytes left |
|---|---:|---:|
| Federer 007 | 4,103 (3.7%) | 16,442 |
| Alcaraz 000 | 6,673 (10%) | 26,722 |
| Perricard | 17,588 (16.8%) | 70,382 |

The percentage stays put. The 8 kB gate is absolute, so the two clips that
miss it at 48 frames would clear it at 192 frames under this rule. A panorama
client becomes relatively faster if the intra decode stays fixed and only the
warps scale: about 8× at 48 frames, about 18× at 192. The sender, with the
plate held at about 120 s, moves from about 0.08× the source encode to about
0.25× and is still slower.
