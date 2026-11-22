"""Typeset each procedure as a separate, tightly sized vector algorithm box."""

from pathlib import Path

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).resolve().parent


def main():
    text = (HERE / "algorithms.md").read_text()
    canvas = Canvas(str(HERE / "algorithms.pdf"), invariant=1)
    canvas.setTitle("PointStream tennis procedures")
    for section in text.split("\n## ")[1:]:
        title = section.splitlines()[0]
        code = section.split("```text\n")[1].split("\n```")[0].splitlines()
        width = max(stringWidth(line, "Courier", 11) for line in code) + 30
        height = 48 + 17 * len(code)
        canvas.setPageSize((width, height))
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(15, height - 19, title)
        canvas.setFont("Courier", 11)
        for i, line in enumerate(code):
            canvas.drawString(15, height - 46 - i * 17, line)
        canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    main()
