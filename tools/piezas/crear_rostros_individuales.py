#!/usr/bin/env python3
"""Texturas de rostro a partir de retratos individuales (cara sola, de frente, fondo blanco o transparente).

Cada retrato se describe en referencias/rostros_individuales/rostros.json con sus anclajes en píxeles
(ojos, comisuras y barbilla). Se escala y coloca en el formato del estudio (410x543, ojos en y=219, barbilla
en y=464); el fondo blanco pasa a transparente (la frente difuminada a blanco se funde así con la piel 3D) y
se miden los colores. Salida: textures/rostro_ind_<id>.png + rostros_individuales.json.
"""
import json
import os

import numpy as np
from PIL import Image, ImageFilter

from crear_rostros_lamina import TEX, W, H, EY, CHIN, CX, LUM, hexc, mid_to_body, median_patch

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'referencias', 'rostros_individuales')


def main():
    cfg = json.load(open(os.path.join(REF, 'rostros.json')))
    defs = {}
    for fid, c in cfg.items():
        img = Image.open(os.path.join(REF, c['archivo']))
        has_alpha = img.mode in ('RGBA', 'LA')
        img = img.convert('RGBA')
        if c.get('crop'):
            # varias caras en una imagen: todo lo que queda fuera de la caja de esta cara pasa a fondo
            bg = Image.new('RGBA', img.size, (255, 255, 255, 0 if has_alpha else 255))
            bg.paste(img.crop(tuple(c['crop'])), tuple(c['crop'][:2]))
            img = bg
        ex = 0.5 * (c['eyeL'][0] + c['eyeR'][0]); ey = 0.5 * (c['eyeL'][1] + c['eyeR'][1])
        s = (CHIN - EY) / (c['chinY'] - ey)
        out = img.transform((W, H), Image.AFFINE, (1 / s, 0, ex - CX / s, 0, 1 / s, ey - EY / s),
                            resample=Image.BICUBIC, fillcolor=(255, 255, 255, 0))
        a = np.asarray(out).astype(np.float32)
        rgb, al = a[..., :3], a[..., 3] / 255.0
        if not has_alpha:
            # fondo blanco -> transparente (la piel muy clara conserva algo de opacidad)
            dw = 255.0 - rgb.min(2)
            al = al * np.clip((dw - 4.0) / 22.0, 0, 1)
            if not c.get('contour', True):
                # máscaras de borde neto: el fondo es SOLO el blanco conectado con el borde de la imagen
                # (los brillos casi blancos de la piel no se agujerean) y se queda la pieza más grande
                inside = al > 0.5
                bg = np.zeros_like(inside); bg[0, :] = ~inside[0, :]; bg[-1, :] = ~inside[-1, :]; bg[:, 0] = ~inside[:, 0]; bg[:, -1] = ~inside[:, -1]
                while True:
                    n = bg | (~inside & (np.roll(bg, 1, 0) | np.roll(bg, -1, 0) | np.roll(bg, 1, 1) | np.roll(bg, -1, 1)))
                    if (n == bg).all(): break
                    bg = n
                face = ~bg
                seed = np.zeros_like(face); seed[int(EY), int(CX)] = True
                while True:
                    n = seed | (face & (np.roll(seed, 1, 0) | np.roll(seed, -1, 0) | np.roll(seed, 1, 1) | np.roll(seed, -1, 1)))
                    if (n == seed).all(): break
                    seed = n
                al = np.where(seed, 1.0, 0.0)
            # Por debajo de los ojos la cara la cierra el trazo oscuro del contorno: todo lo que queda entre los
            # dos trazos es piel (aunque sea casi blanca) -> opaco, con un fundido suave hacia arriba
            dark = rgb.max(2) < (120 if c.get('contour', True) else -1)
            for y in range(int(EY + 20), H):
                xs = np.nonzero(dark[y])[0]
                if len(xs) < 2: continue
                l, r = xs.min(), xs.max()
                if r - l < 40: continue
                k = min(1.0, (y - (EY + 20)) / 60.0)
                al[y, l:r + 1] = np.maximum(al[y, l:r + 1], k)
                al[y, max(0, l - 2):l + 5] = 0; al[y, r - 4:r + 3] = 0     # sin el trazo del contorno
            for x in range(W):   # trazo de la barbilla (el más bajo de cada columna)
                ys = np.nonzero(dark[int(CHIN - 40):, x])[0]
                if len(ys): al[int(CHIN - 40) + ys.max() - 4:, x] = 0
        al8 = Image.fromarray((al * 255).astype(np.uint8))
        if c.get('erode'):
            # sin el sombreado oscuro del canto de la máscara
            al8 = al8.filter(ImageFilter.MinFilter(2 * int(c['erode']) + 1))
        al = np.asarray(al8.filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32) / 255.0
        # Borde de la cara fundido hacia el tono MEDIO de la mejilla (el que se calibra con el cuerpo): el canto
        # sombreado o difuminado del dibujo deja de verse como una línea entre la cara y la cabeza
        ex0 = CX; hw0 = 0.5 * (c['eyeR'][0] - c['eyeL'][0]) * s; Fd0 = CHIN - EY
        mid = median_patch(rgb, [(ex0 - hw0 * 0.95, EY + Fd0 * 0.30), (ex0 + hw0 * 0.95, EY + Fd0 * 0.30)], Fd0 * 0.05)
        m = Image.fromarray(((al > 0.5) * 255).astype(np.uint8))
        dist = np.zeros(al.shape, np.float32)
        for _ in range(int(c.get('edgeFade', 22))):
            dist += np.asarray(m, np.float32) / 255.0
            m = m.filter(ImageFilter.MinFilter(3))
        wgt = np.clip(dist / float(c.get('edgeFade', 22)), 0, 1) ** 0.8
        rgb = mid + (rgb - mid) * wgt[..., None]
        al = al * np.clip(dist / 6.0, 0, 1)
        name = f'rostro_ind_{fid}.png'
        Image.fromarray(np.dstack([rgb, al * 255]).clip(0, 255).astype(np.uint8), 'RGBA').save(os.path.join(TEX, name), optimize=True)
        eL = CX + (c['eyeL'][0] - ex) * s; eR = CX + (c['eyeR'][0] - ex) * s
        mY = EY + ((c['mouthL'][1] + c['mouthR'][1]) * 0.5 - ey) * s
        mL = CX + (c['mouthL'][0] - ex) * s; mR = CX + (c['mouthR'][0] - ex) * s
        Fd = CHIN - EY; hw = (eR - eL) / 2
        skin = median_patch(rgb, [(CX - hw * 0.95, EY + Fd * 0.30), (CX + hw * 0.95, EY + Fd * 0.30)], Fd * 0.05)
        lips = median_patch(rgb, [(CX, mY + 8)], 5)
        iris = np.concatenate([rgb[int(EY - 5):int(EY + 6), int(x - 5):int(x + 6)].reshape(-1, 3) for x in (eL, eR)])
        iris = iris[np.argsort(iris @ LUM)][len(iris) // 4: len(iris) * 3 // 4].mean(0)
        defs['ind_' + fid] = {
            'label': c['label'], 'gender': c['gender'], 'hairCut': c.get('hairCut', 'corto_clasico'),
            **({'beardExt': c['beardExt']} if c.get('beardExt') else {}),
            'url': 'textures/' + name, 'size': [W, H],
            'eyeL': [round(eL, 1), EY], 'eyeR': [round(eR, 1), EY], 'chinY': CHIN,
            'eyeWidth': round(c.get('eyeWidthK', 0.40) * (eR - eL), 1), 'lidOpen': c.get('lidOpen', 0.80),
            'browY': round(EY - Fd * 0.13, 1), 'noseY': round(EY + Fd * 0.45, 1),
            'mouthL': [round(mL, 1), round(mY, 1)], 'mouthR': [round(mR, 1), round(mY, 1)],
            'colors': {'eyes': hexc(iris), **c.get('colors', {}), 'skin': c.get('skinOverride') or mid_to_body(skin), 'lips': hexc(lips)},   # skinOverride: calibrado en render (mejilla = cuello)
        }
        print(fid, round(s, 3), defs['ind_' + fid]['colors'])
    json.dump(defs, open(os.path.join(TEX, 'rostros_individuales.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
