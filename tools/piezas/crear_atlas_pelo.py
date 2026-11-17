#!/usr/bin/env python3
"""Convierte una lámina de 'hair cards' (mechones sobre fondo oscuro) al atlas del estudio.

Formato del estudio: 1024x1024, 4 columnas de 256 px con un mechón vertical cada una (raíz arriba):
  0 = tupido (interior), 1 = normal, 2 = con pelos sueltos (contorno), 3 = fino y ralo (bordes/nacimiento).
  RGB = luminancia de la fibra (el color lo pone el estudio), A = cobertura.

Uso: python3 tools/piezas/crear_atlas_pelo.py [lamina.jpg]
Salida: public/assets/aistudio/textures/hair_atlas_cards.png
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'referencias', 'atlas_pelo_tarjetas.jpg')
OUT = os.path.join(REPO, 'public', 'assets', 'aistudio', 'textures', 'hair_atlas_cards.png')
OUT_CLUMPS = os.path.join(REPO, 'public', 'assets', 'aistudio', 'textures', 'hair_atlas_clumps.png')

# Celdas elegidas de la lámina (x0, y0, x1, y1) medidas sobre la imagen de 2000 px de ancho
CELLS = [
    (25, 375, 230, 745),     # 0 tupido: mechón ancho y liso
    (220, 750, 330, 1100),   # 1 normal: mechón medio y liso
    (330, 750, 437, 1100),   # 2 suelto: mechón con pelos sueltos
    (1495, 20, 1577, 370),   # 3 fino: pocos pelos
]


# Mechones GRANDES (para las láminas anchas de 3-4 cm): el mechón ocupa todo el ancho de la lámina
CLUMP_CELLS = [
    (25, 375, 230, 745),     # 06: ancho, liso y tupido
    (25, 20, 230, 370),      # 01: ancho, liso y tupido
    (230, 375, 445, 745),    # 55: ancho, tupido y ondulado
    (230, 20, 445, 370),     # 02: ancho, tupido y ondulado
]


def cell_to_column(img: np.ndarray, box, bg: float, dense: bool = False) -> np.ndarray:
    x0, y0, x1, y1 = box
    m = 13   # fuera de las líneas de la rejilla y de los números de celda
    c = img[y0 + m:y1 - m, x0 + m:x1 - m]
    lum = c @ np.array([0.30, 0.59, 0.11])
    alpha = np.clip((lum - bg - (3.0 if dense else 6.0)) / (14.0 if dense else 28.0), 0, 1)   # tupido: mechón casi opaco
    ys, xs = np.nonzero(alpha > 0.15)
    c = c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    lum = lum[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # luminancia de la fibra normalizada (sin el fondo): 0.25..1
    hl = np.percentile(lum[alpha > 0.5], 99) if (alpha > 0.5).any() else lum.max()
    fiber = np.clip((lum - bg) / max(1.0, hl - bg), 0, 1) * 0.75 + 0.25
    rgba = np.dstack([fiber, fiber, fiber, alpha])
    im = Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA')
    # columna 256x1024 con un pequeño margen lateral (las tarjetas usan casi todo el ancho)
    col = Image.new('RGBA', (256, 1024), (0, 0, 0, 0))
    im = im.resize((240, 1012), Image.LANCZOS)
    col.paste(im, (8, 6))
    return np.asarray(col)


def main():
    src = Image.open(SRC).convert('RGB')
    if src.width != 2000:
        src = src.resize((2000, round(src.height * 2000 / src.width)), Image.LANCZOS)
    a = np.asarray(src).astype(np.float32)
    bg = float(np.median(a.reshape(-1, 3) @ np.array([0.30, 0.59, 0.11])))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for cells, path, dense in ((CELLS, OUT, False), (CLUMP_CELLS, OUT_CLUMPS, True)):
        atlas = np.concatenate([cell_to_column(a, b, bg, dense) for b in cells], 1)
        out = Image.fromarray(atlas, 'RGBA')
        # rellena el color bajo el alfa (evita halos oscuros al filtrar)
        rgb = out.convert('RGB').filter(ImageFilter.MaxFilter(3))
        out = Image.merge('RGBA', (*rgb.split(), out.split()[3]))
        out.save(path, optimize=True)
        print('Atlas', path, out.size, os.path.getsize(path), 'bytes')


if __name__ == '__main__':
    main()
