"""Outline synthetic test rectangles in a private, registered four-image strip."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def mark(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        if image.size != (1920, 140):
            raise ValueError('Expected four 480 by 140 crops from compare.py')
        image = image.convert('RGB')
        draw = ImageDraw.Draw(image)
        for panel in range(4):
            for x in (20, 395):
                draw.rectangle((panel * 480 + x, 20, panel * 480 + x + 63, 123),
                               outline=(255, 209, 0), width=2)
        image.save(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    arguments = parser.parse_args()
    mark(arguments.source, arguments.destination)
