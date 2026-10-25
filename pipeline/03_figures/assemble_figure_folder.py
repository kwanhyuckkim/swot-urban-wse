#!/usr/bin/env python3
"""최종 납품 figure/ 폴더 조립.

fig01-fig09, figS1-figS7을 정식 이름의 600 dpi PNG로 복사하고, 벡터가 있는
그림은 figure/pdf/에 같이 둔다. figS7만 원본 노트북 export(4200 px 폭,
6.5 in 배치 시 유효 646 dpi)를 그대로 쓴다.
"""
import shutil
from pathlib import Path
from PIL import Image

BASE = Path("/Users/targetkim/target/SWOT_urbanerror")
FP = BASE / "figures_print"
OUT = BASE / "figure"
PDF = OUT / "pdf"
OUT.mkdir(exist_ok=True)
PDF.mkdir(exist_ok=True)

MAP = {
    "fig01": FP / "fig01_orig_print.png",
    "fig02": FP / "fig02_study_map_v2.png",
    "fig03": FP / "fig03_map_schematic.png",
    "fig04": FP / "fig04_orig_print.png",
    "fig05": FP / "fig05_flags_screen.png",
    "fig06": FP / "fig06_orig_print.png",
    "fig07": FP / "fig07_orig_print.png",
    "fig08": FP / "fig08_orig_print.png",
    "fig09": FP / "fig09_orig_print.png",
    "figS1": FP / "figS1_thresholds.png",
    "figS2": FP / "figS3_orig_print.png",
    "figS3": FP / "figS4_region_decomposition.png",
    "figS4": FP / "figS5_structure_check.png",
    "figS5": FP / "figS5_displacement_print.png",
}

for name, src in MAP.items():
    dst = OUT / f"{name}.png"
    shutil.copy2(src, dst)
    im = Image.open(dst)
    if round(float(im.info.get("dpi", (600,))[0])) != 600:
        im.save(dst, dpi=(600, 600))
        im = Image.open(dst)
    print(f"{name}.png  {im.size[0]}x{im.size[1]}  dpi={round(float(im.info['dpi'][0]))}")
    vec = src.with_suffix(".pdf")
    if vec.exists():
        shutil.copy2(vec, PDF / f"{name}.pdf")

print("\npdf/:", sorted(p.name for p in PDF.glob("*.pdf")))
