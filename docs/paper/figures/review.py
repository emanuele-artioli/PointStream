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
    ("system/architecture.pdf", "Scene-selective coding", "The architecture shows both paths to playback and the state retained by the receiver. Its switching policy is schematic."),
    ("selection/rate-quality.pdf", "Rate and foreground fidelity", "One 4.8-second cold start. All 17 points use comparable object scoring. Completing the main figure requires the same foreground objective for DCVC-UF and PointStream at overlapping low rates."),
    ("selection/bandwidth-choice.pdf", "Which stream fits the bandwidth?", "Each strip picks the best measured stream under its bandwidth ceiling for one object. A deployed selector must choose before seeing the reconstructed score and must pay for switches."),
    ("selection/historical-low-rate.pdf", "Why revisit the low-rate player regime?", "The historical 25-fps Alcaraz window uses sampled frames and modeled two-hour reference and adapter reuse. Its points have their own comparison panel."),
    ("ball/matched-rate-ball-fidelity.pdf", "Ball fidelity at essentially equal bandwidth", "On 12 reviewed ticks, allocating bytes to source-pixel tiles improved ball-crop fidelity while whole-frame RGB PSNR fell by 0.104 dB."),
    ("ball/three-match-ball-bd-rate.pdf", "Tracked ball pixels across matches", "The 26.28% equal-match saving concerns ball-crop DISTS at labeled visible instants. The complete stream pays for both the native video and ball channel."),
    ("ball/ball-detection.pdf", "Can a machine locate the ball?", "The independent judge recovered 224 additional visible frames with a rendered streak. False detections increased on two absent frames. A matched-rate rerun is needed for this particular machine-task comparison."),
    ("player-tasks/pose-coding.pdf", "How accurately are the sender's joints conveyed?", "The reference consists of the sender's uncoded YOLO estimates. Human joint labels and an independent judge on decoded video would measure player pose accuracy."),
    ("player-tasks/receiver-masks.pdf", "How closely do receiver silhouettes follow the people?", "Mean frame IoU compares the receiver's own masks against 12 human-reviewed source frames. An RGB segmentation judge on the same matched-rate decodes is the next comparison."),
    ("background/court-reconstruction.pdf", "Reconstructing court behind an occluder", "Four frames have artificial masks over known court pixels. Temporal donors help where they observe the court. Naturally hidden pixels lack this same-frame ground truth."),
    ("references/reference-ladder.pdf", "Choose shared reference rungs by byte budget", "At all three two-image budgets AVIF led both development references on RGB PSNR. The rendering study checks whether that still-image advantage preserves generated player appearance."),
]


def page(title, caption, index, raster=None):
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=SIZE)
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(32, 579, title)
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(760, 580, f"PointStream  |  Author review  |  {index}")
    if raster:
        labels = ("Original court", "Artificial occlusion", "Spatial inpainting", "Aligned donors with hole filling")
        with Image.open(raster) as strip:
            if strip.size != (1920, 140):
                raise ValueError("Expected four 480 by 140 court crops from the registered comparison")
            for i, label in enumerate(labels):
                x, y = 31 + 375 * (i % 2), 326 - 164 * (i // 2)
                crop = strip.crop((480*i, 0, 480*(i+1), 140))
                canvas.drawImage(ImageReader(crop), x, y, 355, 355*140/480)
                canvas.setFont("Helvetica-Bold", 12)
                canvas.drawString(x, y + 114, label)
    canvas.setFont("Helvetica", 10)
    for i, line in enumerate(wrap(caption, width=145)):
        canvas.drawString(32, 45 - i*13, line)
    canvas.save()
    return PdfReader(buffer).pages[0]


def build(output, timeline, court, rally):
    panels = [(HERE / name, title, caption) for name, title, caption in PANELS]
    if rally:
        panels.insert(4, (rally, "Where the extra ball bits went", "Private review imagery. Three prespecified source-aligned instants show the exact 64-pixel ball crops. The tile covers the ball on two of the three instants."))
    if timeline:
        panels.insert(0, (timeline, "Court, cutaway and return", "Private broadcast imagery for author review. The two cut times are measured; coding paths are schematic. Publication rights need verification."))
    if court:
        panels.insert(-1, (court, "Court reconstruction examples", "Private review imagery. The original court pixels under these artificial masks provide a reference for spatial and temporal reconstruction."))
    writer = PdfWriter()
    for index, (source, title, caption) in enumerate(panels, start=1):
        canvas = page(title, caption, index, source if source.suffix == ".png" else None)
        if source.suffix == ".pdf":
            pdf = PdfReader(source)
            assert len(pdf.pages) == 1
            panel = pdf.pages[0]
            width, height = float(panel.mediabox.width), float(panel.mediabox.height)
            scale = min(724 / width, 471 / height)
            translate = Transformation().translate(-float(panel.mediabox.left), -float(panel.mediabox.bottom))
            matrix = translate.scale(scale).translate((792-width*scale)/2, 71+(471-height*scale)/2)
            canvas.merge_transformed_page(panel, matrix)
        writer.add_page(canvas)
        writer.add_outline_item(title, index - 1)
    writer.add_metadata({"/Title": "PointStream figure review", "/Subject": "Development figures for author review"})
    with output.open("xb") as stream:
        writer.write(stream)
    assert len(PdfReader(output).pages) == len(panels)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--timeline", type=Path)
    cli.add_argument("--court", type=Path)
    cli.add_argument("--rally", type=Path)
    args = cli.parse_args()
    build(args.output, args.timeline, args.court, args.rally)
