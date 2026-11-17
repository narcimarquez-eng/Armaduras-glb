#!/usr/bin/env python3
"""Atlas de mechones TUPIDOS a partir de las láminas de color con fondo de cuadros (rubio, oscuro, castaño).

El fondo de cuadros está pintado (no hay transparencia real): se separa el pelo por saturación/luminancia
(los cuadros son grises claros sin color). Salida por lámina:
  public/assets/aistudio/textures/hair_atlas_clumps_<color>.png  (4 columnas, raíz arriba, gris + alfa)
  public/assets/aistudio/textures/hair_atlas_clumps_colores.json  (raíz, punta y brillo medidos de cada lámina)
"""
import json
import os

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
TEX = os.path.join(REPO, 'public', 'assets', 'aistudio', 'textures')
SHEETS = ['rubio', 'oscuro', 'castano']
# Mechones más poblados de la lámina (cajas sobre la imagen a 2000 px)
# Rejilla 4x2 de celdas de 256x512: fila 0 = mechones LARGOS, fila 1 = mechones CORTOS/medios;
# columnas de liso a muy curvado
CELLS = [
    (25, 375, 230, 745),    # largo liso tupido
    (230, 375, 445, 745),   # largo ondulado
    (445, 375, 665, 745),   # largo con ondas marcadas
    (665, 375, 833, 745),   # largo curvado en S
    (25, 750, 220, 1100),   # corto liso tupido
    (330, 750, 437, 1100),  # corto curvado
    (833, 20, 1000, 370),   # medio ondulado
    (1000, 20, 1167, 370),  # medio muy curvado
]
W = np.array([0.30, 0.59, 0.11])


def hex_(c):
    return '#' + ''.join('%02x' % int(np.clip(v, 0, 255)) for v in c)


def column(a, box):
    x0, y0, x1, y1 = box
    m = 14
    c = a[y0 + m:y1 - m, x0 + m:x1 - m]
    lum = c @ W
    sat = c.max(2) - c.min(2)
    # cuadros: grises claros (sat < ~10, lum > ~170); pelo: más oscuro o con color
    alpha = np.clip(np.maximum((178 - lum) / 30.0, (sat - 12) / 18.0), 0, 1)
    ys, xs = np.nonzero(alpha > 0.4)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    c, lum, alpha = c[sl], lum[sl], alpha[sl]
    hair = alpha > 0.6
    lo, hi = np.percentile(lum[hair], 2), np.percentile(lum[hair], 99)
    fiber = np.clip((lum - lo) / max(1.0, hi - lo), 0, 1) * 0.8 + 0.2
    im = Image.fromarray((np.dstack([fiber, fiber, fiber, alpha]) * 255).astype(np.uint8), 'RGBA')
    col = Image.new('RGBA', (256, 512), (0, 0, 0, 0))
    col.paste(im.resize((244, 504), Image.LANCZOS), (6, 4))
    return np.asarray(col), c[hair]


def main():
    colors = {}
    for name in SHEETS:
        src = Image.open(os.path.join(HERE, 'referencias', 'atlas_pelo_%s.jpg' % name)).convert('RGB')
        if src.width != 2000:
            src = src.resize((2000, round(src.height * 2000 / src.width)), Image.LANCZOS)
        a = np.asarray(src).astype(np.float32)
        cols, px = zip(*[column(a, b) for b in CELLS])
        out = Image.fromarray(np.concatenate([np.concatenate(cols[:4], 1), np.concatenate(cols[4:], 1)], 0), 'RGBA')
        rgb = out.convert('RGB').filter(ImageFilter.MaxFilter(3))
        out = Image.merge('RGBA', (*rgb.split(), out.split()[3]))
        path = os.path.join(TEX, 'hair_atlas_clumps_%s.png' % name)
        out.save(path, optimize=True)
        allpx = np.concatenate(px)
        order = np.argsort(allpx @ W)
        pick = lambda q: allpx[order[int(q * (len(order) - 1))]]
        colors[name] = {'root': hex_(pick(0.12)), 'tip': hex_(pick(0.55)), 'high': hex_(pick(0.95))}
        print(name, path, colors[name])
    with open(os.path.join(TEX, 'hair_atlas_clumps_colores.json'), 'w') as f:
        json.dump(colors, f, indent=1)


if __name__ == '__main__':
    main()
