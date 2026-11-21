"""Collect paper panels for author review; private image inputs are optional."""

import argparse
from io import BytesIO
from pathlib import Path
from textwrap import wrap

from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).resolve().parent
SIZE = (792, 612)
PANELS = [
    (
        "system/architecture.pdf",
        "How the receiver reconstructs a broadcast",
        "At each eligible shot the object path supplies signals for an application and frames for playback. The conventional path handles other shots.",
    ),
    (
        "system/algorithms.pdf",
        "Implemented procedures",
        "The tennis sender tests court eligibility. The receiver renders players from paid appearance and keypoints and composites generated masks. It applies a paid ball update where a valid location exists.",
    ),
    (
        "players-starved/players-starved.pdf",
        "Player fidelity across three matches",
        "Eight paired historical windows use a same-match adaptation and a modeled two-hour asset lifetime. Each player-region comparison uses the actual competing DCVC-UF rates for that window.",
    ),
    (
        "starved-choice/players-rate-quality.pdf",
        "Player fidelity at the available bandwidth",
        "One rally with common scoring. Solid red charges the full startup. Open red prices already valid player references. All people have equal weight.",
    ),
    (
        "starved-choice/ball-rate-quality.pdf",
        "Ball fidelity at the available bandwidth",
        "Source-fitted trajectory, native video, tracked-pixel and joint modes share the same 12 source-reviewed instants. The lower-left star represents a complete 19.965 kbps stream.",
    ),
    (
        "selection/65kbps-regional-dists.pdf",
        "Where the joint reconstruction gains and loses quality",
        "Twelve shared instants, with a fixed 70% weight on both bodies and 30% on their complement. The joint stream charges 469.335 kbps cold and 65.473 kbps with ideal reference reuse.",
    ),
    (
        "ball/starved-display.pdf",
        "What a ball channel buys at 61 kbps",
        "One 240-tick rally. Twelve independently reviewed instants. The tile and plain-video pair differ by 28 bytes. The larger-resolution plain and ROI streams bracket that budget.",
    ),
    (
        "ball/ball-three-match-rates.pdf",
        "Ball fidelity by total delivered bitrate",
        "The earlier three-match study supports a common-rate average only across the overlap indicated in the plot. Each point includes the base stream and any object payload.",
    ),
    (
        "ball-starved/ball-starved.pdf",
        "Ball fidelity near 61 kbps in three broadcasts",
        "Nine, twelve and eleven independently labeled instants. Each match includes its best measured affordable plain or ROI AV1 control. The 60-fps match favors conventional AV1.",
    ),
    (
        "ball/matched-low-detection.pdf",
        "Can a machine locate the ball?",
        "On the 292-frame clip, 261 visible and 30 absent frames have definite labels. Direct paid refit locations score 240/261 and 2/30. The same protocol also measures a detector running on every decoded RGB arm.",
    ),
    (
        "system/starved-budget.pdf",
        "What the sender pays at startup and during play",
        "Actual native and joint packets, with image payloads itemized. The cache row assumes both references are already valid. The AVIF startup prices hold the other packet components fixed.",
    ),
    (
        "references/reuse-budget.pdf",
        "How much play pays for the first player references?",
        "The duration is active encoded play. At a 52.293 kbps ongoing price the lossless pair takes 26.7 seconds to average 125 kbps. Lossy costs are scenarios until their rendered output passes review.",
    ),
    (
        "system/crossshot-reference.pdf",
        "What the receiver actually paid when the players returned",
        "A 354-byte mapping reused both player images on the return 221.3 seconds later. The author judged reuse barely acceptable and preferred it to refresh for both tracks. This comparison prices appearance only.",
    ),
    (
        "references/rendered-reference-quality.pdf",
        "Which appearance references passed the motion review?",
        "One author viewed two 24-frame player tracks. Both shared lossy rungs passed for Player 2. Player 1 accepted the lossless control and judged Q78 almost acceptable. The plotted objective metric includes crop background.",
    ),
    (
        "player-tasks/decoded-masks.pdf",
        "Player segmentation on ordinary and reconstructed video",
        "Direct receiver masks and ordinary AV1 had similar mean IoU on 24 bodies. All arms localized both players. Six unprompted labels favored ordinary AV1 over the receiver masks.",
    ),
    (
        "ball/three-match-ball-bd-rate.pdf",
        "Supporting result · Equal-quality ball rate",
        "Per-match comparisons over common ball-crop DISTS support. The arithmetic mean of the three bitrate savings is 26.28%.",
    ),
    (
        "selection/65kbps-regional-lpips.pdf",
        "Appendix · Regional sensitivity to the perceptual metric",
        "The same fixed 70/30 weights evaluated with spatial LPIPS on the same twelve instants.",
    ),
    (
        "selection/rate-quality.pdf",
        "Appendix · Cold start over 4.8 seconds",
        "The native-scale comparison measures foreground fidelity on two players and the ball with full reference cost.",
    ),
    (
        "ball/matched-rate-ball-fidelity.pdf",
        "Appendix · Higher-rate ball fidelity",
        "At 303 kbps, 12 reviewed instants show a ball-crop improvement alongside a 0.104 dB whole-frame RGB PSNR cost.",
    ),
    (
        "selection/bandwidth-choice.pdf",
        "Appendix · Best stored stream under each cap",
        "This retrospective comparison selects by reconstructed quality and excludes switching overhead.",
    ),
    (
        "player-tasks/pose-coding.pdf",
        "Appendix · Distortion of transmitted joints",
        "Comparison with the original sender detections measures coding error. Human source joint labels are needed to measure pose accuracy.",
    ),
    (
        "background/court-reconstruction.pdf",
        "Appendix · How to restore court behind a player",
        "Artificial masks over known court pixels permit scoring aligned temporal donors and spatial inpainting. Inpainting supplies pixels where donors do not observe the court.",
    ),
    (
        "references/reference-ladder.pdf",
        "Appendix · Still-image reference ladder",
        "The image curves measure two appearance references. The highest-PSNR still image does not guarantee acceptable generated motion.",
    ),
]


def page(title, caption, index, raster=None):
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=SIZE)
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(32, 579, title)
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(760, 580, f"PointStream  |  Author review  |  {index}")
    if raster:
        labels = (
            "Original court",
            "Artificial occlusion",
            "Spatial inpainting",
            "Aligned donors with hole filling",
        )
        with Image.open(raster) as strip:
            if strip.size != (1920, 140):
                raise ValueError(
                    "Expected four 480 by 140 court crops from the registered comparison"
                )
            for i, label in enumerate(labels):
                x, y = 31 + 375 * (i % 2), 326 - 164 * (i // 2)
                crop = strip.crop((480 * i, 0, 480 * (i + 1), 140))
                canvas.drawImage(ImageReader(crop), x, y, 355, 355 * 140 / 480)
                canvas.setFont("Helvetica-Bold", 12)
                canvas.drawString(x, y + 114, label)
    canvas.setFont("Helvetica", 10)
    for i, line in enumerate(wrap(caption, width=145)):
        canvas.drawString(32, 45 - i * 13, line)
    canvas.save()
    return PdfReader(buffer).pages[0]


def build(output, timeline=None, court=None, rally=None, stroke=None, starved_ball=None, main_only=False):
    panels = [(HERE / name, title, caption) for name, title, caption in PANELS]

    def after(target, entry):
        i = next(i for i, (path, _, _) in enumerate(panels) if path.name == target)
        panels.insert(i + 1, entry)

    if timeline:
        panels.insert(
            0,
            (
                timeline,
                "Court, cutaway and return",
                "Private broadcast imagery. Cut times are measured. Coding paths are schematic. Publication rights need verification.",
            ),
        )
    if stroke:
        after(
            "players-starved.pdf",
            (
                stroke,
                "A serve at constrained bandwidth",
                "Three source-selected instants. Ordinary AV1 costs 66.105 kbps. The joint decode costs 469.335 kbps cold with a 65.473 kbps ideal running cost. Racket grip remains an open equipment question.",
            ),
        )
    if starved_ball:
        after(
            "starved-display.pdf",
            (
                starved_ball,
                "The ball at 61 kbps",
                "Source, baseline and tiled reconstructions at identical instants from the 60.623 and 60.670 kbps comparison. Private imagery.",
            ),
        )
    if rally:
        after(
            "matched-rate-ball-fidelity.pdf",
            (
                rally,
                "Appendix · Ball pixels at 303 kbps",
                "Three source-aligned instants from the higher-rate ablation. The tile covers the ball on two of the three instants. Private imagery.",
            ),
        )
    if court:
        after(
            "court-reconstruction.pdf",
            (
                court,
                "Appendix · Court reconstruction examples",
                "The same artificial test masks cover two marked court regions. The source supplies ground truth for both methods. Private imagery.",
            ),
        )
    if main_only:
        selected = {
            "architecture.pdf", "players-starved.pdf", "players-rate-quality.pdf",
            "ball-rate-quality.pdf", "ball-starved.pdf", "matched-low-detection.pdf",
            "starved-budget.pdf", "crossshot-reference.pdf",
        }
        selected.update(path.name for path in (timeline, stroke, starved_ball) if path)
        panels = [entry for entry in panels if entry[0].name in selected]
    writer = PdfWriter()
    for source, title, caption in panels:
        pages = PdfReader(source).pages if source.suffix == ".pdf" else [None]
        for j, panel in enumerate(pages):
            index = len(writer.pages) + 1
            current_title = (
                title
                if len(pages) == 1
                else title + (" · court and players" if j == 0 else " · tracked ball")
            )
            canvas = page(current_title, caption, index, source if panel is None else None)
            if panel is not None:
                width, height = float(panel.mediabox.width), float(panel.mediabox.height)
                scale = min(724 / width, 471 / height)
                translate = Transformation().translate(
                    -float(panel.mediabox.left), -float(panel.mediabox.bottom)
                )
                matrix = translate.scale(scale).translate(
                    (792 - width * scale) / 2, 71 + (471 - height * scale) / 2
                )
                canvas.merge_transformed_page(panel, matrix)
            writer.add_page(canvas)
            writer.add_outline_item(current_title, index - 1)
    writer.add_metadata(
        {"/Title": "PointStream figure review", "/Subject": "Development figures for author review"}
    )
    with output.open("xb") as stream:
        writer.write(stream)
    assert len(PdfReader(output).pages) == len(writer.pages)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--timeline", type=Path)
    cli.add_argument("--court", type=Path)
    cli.add_argument("--rally", type=Path)
    cli.add_argument("--stroke", type=Path)
    cli.add_argument("--starved-ball", type=Path)
    cli.add_argument("--main-only", action="store_true", help="Select the proposed manuscript figures")
    args = cli.parse_args()
    build(args.output, args.timeline, args.court, args.rally, args.stroke, args.starved_ball, args.main_only)
