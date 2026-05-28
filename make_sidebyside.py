"""
For each surface, stitch v0 and v1 PNG side by side.
Output: comparison_output/sidebyside/{label}_compare.png
"""

import glob
import os
from PIL import Image, ImageDraw, ImageFont

IN_DIR  = "comparator_output_v1"
OUT_DIR = "comparison_output/sidebyside"
os.makedirs(OUT_DIR, exist_ok=True)

v0_files = sorted(glob.glob(f"{IN_DIR}/Surface*_comparator.png"))

LABEL_H  = 40   # height of label banner above each panel
BG_COLOR = (240, 240, 240)
FG_COLOR = (30, 30, 30)

for v0_path in v0_files:
    label = os.path.basename(v0_path).replace("_comparator.png", "")
    v1_path = f"{IN_DIR}/{label}_comparator_v1.png"
    if not os.path.exists(v1_path):
        print(f"  skip {label} — v1 not found")
        continue

    img0 = Image.open(v0_path).convert("RGB")
    img1 = Image.open(v1_path).convert("RGB")

    # Keep only the left panel (map), discard the right histogram panel
    img0 = img0.crop((0, 0, img0.width // 2, img0.height))
    img1 = img1.crop((0, 0, img1.width // 2, img1.height))

    # Resize to same height if needed
    h = max(img0.height, img1.height)
    if img0.height != h:
        img0 = img0.resize((int(img0.width * h / img0.height), h), Image.LANCZOS)
    if img1.height != h:
        img1 = img1.resize((int(img1.width * h / img1.height), h), Image.LANCZOS)

    total_w = img0.width + img1.width
    total_h = h + LABEL_H

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    # Labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, img0.width, LABEL_H], fill=(200, 220, 240))
    draw.text((10, 8), f"v0  —  {label}", fill=FG_COLOR, font=font)

    draw.rectangle([img0.width, 0, total_w, LABEL_H], fill=(240, 210, 210))
    draw.text((img0.width + 10, 8), f"v1  —  {label}", fill=FG_COLOR, font=font)

    canvas.paste(img0, (0,         LABEL_H))
    canvas.paste(img1, (img0.width, LABEL_H))

    out_path = f"{OUT_DIR}/{label}_compare.png"
    canvas.save(out_path, optimize=True)
    print(f"  {label}")

print(f"\nDone → {OUT_DIR}/")
