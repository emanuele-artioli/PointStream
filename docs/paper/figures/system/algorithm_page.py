"""Typeset the implemented algorithms from their shared, auditable paper text."""
from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).resolve().parent


def main():
    text = (HERE / "algorithms.md").read_text()
    canvas = Canvas(str(HERE / "algorithms.pdf"), pagesize=(650, 520), invariant=1)
    canvas.setTitle("PointStream tennis procedures")
    y = 492
    for index, section in enumerate(text.split("\n## ")[1:]):
        if index == 2:
            canvas.showPage()
            y = 492
        title = section.splitlines()[0]
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(21, y, title)
        y -= 25
        canvas.setFont("Courier", 10.6)
        for line in section.split("```text\n")[1].split("\n```")[0].splitlines():
            canvas.drawString(24, y, line)
            y -= 16.5
        y -= 20
    canvas.save()


if __name__ == "__main__":
    main()
