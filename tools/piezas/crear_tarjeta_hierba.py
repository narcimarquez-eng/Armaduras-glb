#!/usr/bin/env python3
"""Atlas de tarjetas de hierba (1024x512, 4 columnas, raíz abajo, fondo transparente).

0 = hierba densa corta, 1 = hierba alta, 2 = hierba seca/doblada, 3 = matas con flores.
Cada brizna es una curva cónica (ancha en la base, afilada en la punta) con degradado de color:
oscuro y sombreado en la base, más claro y amarillento hacia la punta, y un nervio central de luz.
Salida: public/assets/aistudio/textures/hierba_atlas.png
"""
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'public', 'assets', 'aistudio', 'textures', 'hierba_atlas.png')
W, H, COLS, SS = 1024, 512, 4, 2   # supermuestreo x2 para bordes suaves


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def blade(dr, x0, y0, height, width, bend, base, tip, rnd):
    """Brizna: polígono afilado que se curva con 'bend'; se pinta por tramos para el degradado."""
    N = 14
    pts_l, pts_r, mids = [], [], []
    for i in range(N + 1):
        t = i / N
        x = x0 + bend * height * t * t + rnd.uniform(-0.3, 0.3)
        y = y0 - height * t
        w = width * (1 - t) ** 0.9 * 0.5
        # dirección perpendicular aproximada
        dx = 2 * bend * height * t; nx, ny = 1.0, dx / max(1e-3, height)
        n = math.hypot(nx, ny); nx, ny = nx / n, ny / n
        pts_l.append((x - nx * w, y - ny * w)); pts_r.append((x + nx * w, y + ny * w)); mids.append((x, y))
    for i in range(N):
        t = (i + 0.5) / N
        col = lerp(base, tip, t ** 0.8) + (255,)
        dr.polygon([pts_l[i], pts_l[i + 1], pts_r[i + 1], pts_r[i]], fill=col)
    # nervio de luz
    for i in range(2, N - 1):
        t = i / N
        dr.line([mids[i], mids[i + 1]], fill=lerp(tip, (235, 245, 190), 0.35) + (int(70 * (1 - t)),), width=max(1, int(width * 0.12)))


def column(kind, seed):
    rnd = random.Random(seed)
    cw, ch = (W // COLS) * SS, H * SS
    img = Image.new('RGBA', (cw, ch), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    pal = {
        0: ((28, 52, 18), (118, 168, 58)),
        1: ((24, 48, 16), (140, 180, 70)),
        2: ((60, 58, 26), (196, 178, 104)),
        3: ((26, 50, 18), (126, 170, 62)),
    }[kind]
    n = {0: 70, 1: 45, 2: 40, 3: 55}[kind]
    hmax = {0: 0.55, 1: 0.95, 2: 0.80, 3: 0.75}[kind] * ch
    blades = []
    for _ in range(n):
        x0 = cw * 0.5 + rnd.gauss(0, cw * 0.18)
        x0 = min(cw * 0.95, max(cw * 0.05, x0))
        h = hmax * rnd.uniform(0.45, 1.0)
        bend = rnd.uniform(-0.35, 0.35) * (1.6 if kind == 2 else 1.0)
        w = cw * rnd.uniform(0.035, 0.06)
        shade = rnd.uniform(0.75, 1.1)
        base = tuple(min(255, int(c * shade)) for c in pal[0])
        tip = tuple(min(255, int(c * shade)) for c in pal[1])
        blades.append((h, x0, bend, w, base, tip))
    # las más altas detrás
    for h, x0, bend, w, base, tip in sorted(blades, key=lambda b: -b[0]):
        blade(dr, x0, ch - 2, h, w, bend, base, tip, rnd)
    if kind == 3:
        for _ in range(9):
            fx = cw * 0.5 + rnd.gauss(0, cw * 0.2); fy = ch - hmax * rnd.uniform(0.5, 0.95)
            col = rnd.choice([(240, 230, 120), (250, 250, 240), (200, 140, 220)])
            for k in range(5):
                a = k / 5 * 2 * math.pi
                dr.ellipse([fx + math.cos(a) * 10 - 9, fy + math.sin(a) * 10 - 9, fx + math.cos(a) * 10 + 9, fy + math.sin(a) * 10 + 9], fill=col + (255,))
            dr.ellipse([fx - 7, fy - 7, fx + 7, fy + 7], fill=(230, 160, 40, 255))
    # oclusión en la base: las raíces más oscuras
    a = np.asarray(img).astype(np.float32)
    yy = np.linspace(0, 1, ch)[:, None]
    ao = 0.55 + 0.45 * np.clip((1 - yy) / 0.35, 0, 1)
    a[..., :3] *= ao[..., None]
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8), 'RGBA')
    return img.resize((W // COLS, H), Image.LANCZOS)


def main():
    atlas = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    for k in range(COLS):
        atlas.paste(column(k, 100 + k), (k * W // COLS, 0))
    # color bajo el alfa (sin halos al filtrar)
    rgb = atlas.convert('RGB').filter(ImageFilter.MaxFilter(3))
    atlas = Image.merge('RGBA', (*rgb.split(), atlas.split()[3]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    atlas.save(OUT, optimize=True)
    print('atlas', os.path.abspath(OUT), os.path.getsize(OUT))


if __name__ == '__main__':
    main()
