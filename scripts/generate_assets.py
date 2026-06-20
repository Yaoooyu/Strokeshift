"""Generate privacy-safe placeholder figures from synthetic values only."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FONT = ImageFont.load_default()


def rounded_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fill: str) -> None:
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline="#17365d", width=3)
    draw.multiline_text(((xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2), text, fill="#17202a", font=FONT, anchor="mm", align="center")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((start, end), fill="#555555", width=4)
    x, y = end
    draw.polygon([(x, y), (x - 12, y - 7), (x - 12, y + 7)], fill="#555555")


def pipeline() -> None:
    image = Image.new("RGB", (1600, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((800, 35), "CGMS-Stacking pipeline (synthetic schematic placeholder)", fill="#17202a", font=FONT, anchor="mm")
    entries = [("CT probability", "#d9eaf7"), ("EHR probability", "#e2f0d9"), ("Report probability", "#fff2cc")]
    for index, (label, color) in enumerate(entries):
        y = 130 + index * 130
        rounded_box(draw, (60, y - 38, 320, y + 38), label, color)
        arrow(draw, (320, y), (480, 280))
    rounded_box(draw, (480, 195, 770, 365), "20-shot candidate\ncalibration bank", "#fce4d6")
    arrow(draw, (770, 280), (900, 280))
    rounded_box(draw, (900, 195, 1150, 365), "Center-wise\nselection", "#e4dfec")
    arrow(draw, (1150, 280), (1270, 280))
    rounded_box(draw, (1270, 195, 1510, 365), "Logistic stacking", "#d9ead3")
    draw.text((800, 515), "Query labels are used only for final evaluation", fill="#7f1d1d", font=FONT, anchor="mm")
    image.save(ASSETS / "fig_cgms_pipeline.png")


def domain_shift() -> None:
    rng = np.random.default_rng(42)
    image = Image.new("RGB", (900, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((450, 35), "Synthetic domain-shift illustration", fill="#17202a", font=FONT, anchor="mm")
    draw.rectangle((90, 80, 850, 620), outline="#555555", width=2)
    centers = [("A", (-0.9, 0.2), "#4472c4"), ("B", (1.0, 0.8), "#ed7d31"), ("C", (0.4, -1.0), "#70ad47")]
    for label, mean, color in centers:
        points = rng.normal(mean, (0.45, 0.4), size=(45, 2))
        for x_value, y_value in points:
            x = int(470 + x_value * 180)
            y = int(350 - y_value * 150)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.text((720, 105 + 30 * (ord(label) - ord("A"))), f"Center {label}", fill=color, font=FONT)
    draw.text((450, 680), "Placeholder only; not derived from patient embeddings", fill="#7f1d1d", font=FONT, anchor="mm")
    image.save(ASSETS / "fig_domain_shift_umap.png")


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    pipeline()
    domain_shift()
