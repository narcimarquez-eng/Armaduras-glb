#!/usr/bin/env python3
"""Texturas de rostro a partir de las láminas de retratos (5 láminas x 6 rostros).

Para cada rostro: ojos y boca detectados (referencias/rostros_lamina/anclajes_detectados.json, coordenadas
dentro del recorte de 520x550 centrado en cada retrato), se escala y coloca en el formato de las texturas del
estudio (410x543, ojos en y=219, barbilla en y=464), se recorta con la máscara facial de las texturas
existentes y se miden los colores (piel -> color de cuerpo, labios, ojos y pelo).
Salida: public/assets/aistudio/textures/rostro_lam_<id>.png + rostros_lamina.json (definiciones para el estudio).
"""
import json
import os

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
REF = os.path.join(HERE, 'referencias', 'rostros_lamina')
TEX = os.path.join(REPO, 'public', 'assets', 'aistudio', 'textures')
W, H, EY, CHIN, CX = 410, 543, 219.0, 464.0, 205.0
CY_ROW, CX_COL = [290, 830], [370, 1010, 1660]

# id -> (etiqueta, género, corte de pelo, extensión de barba)
FACES = {
    '18_00': ('Nómada encapuchada', 'mujer', 'trenzas_laterales', 0),
    '18_01': ('Guerrero de la estepa', 'hombre', 'coleta_alta', 0.15),
    '18_02': ('Guerrero de trenzas', 'hombre', 'raya_medio_lisa', 0),
    '18_10': ('Druida pelirroja', 'mujer', 'trenzas_laterales', 0),
    '18_11': ('Dama de los palillos', 'mujer', 'coleta_alta', 0),
    '18_12': ('Guerrero tatuado', 'hombre', 'rizado_3d', 0),
    '19_00': ('Elfa oscura de la diadema', 'mujer', 'raya_medio_lisa', 0),
    '19_01': ('Orco veterano', 'hombre', 'raya_medio_lisa', 0.35),
    '19_02': ('Enana pelirroja', 'mujer', 'trenzas_laterales', 0),
    '19_10': ('Elfa plateada', 'mujer', 'raya_medio_lisa', 0),
    '19_11': ('Enano barbirrojo', 'hombre', 'raya_medio_lisa', 0.6),
    '19_12': ('Elfo oscuro marcado', 'hombre', 'raya_medio_lisa', 0),
    '20_00': ('Guerrera de trenzas doradas', 'mujer', 'trenzas_laterales', 0),
    '20_01': ('Cazadora de plumas', 'mujer', 'trenzas_laterales', 0),
    '20_02': ('Hechicera de plata', 'mujer', 'raya_medio_lisa', 0),
    '20_10': ('Pirata tuerta', 'mujer', 'ondulado_volumen', 0),
    '20_11': ('Asesina de pelo corto', 'mujer', 'bob_flequillo', 0),
    '20_12': ('Escudera rubia', 'mujer', 'trenzas_laterales', 0),
    '21_00': ('Mercenario de pelo largo', 'hombre', 'raya_medio_lisa', 0),
    '21_01': ('Viejo tuerto', 'hombre', 'peinado_atras', 0),
    '21_02': ('Bárbaro pintado', 'hombre', 'corto_guerrero', 0),
    '21_10': ('Explorador tuerto', 'hombre', 'raya_medio_lisa', 0),
    '21_11': ('Asesino rapado', 'hombre', 'militar', 0),
    '21_12': ('Luchador rizado', 'hombre', 'corto_guerrero', 0),
    '22_00': ('Mercenario barbado', 'hombre', 'raya_medio_lisa', 0.10),
    '22_01': ('Viejo tuerto barbado', 'hombre', 'peinado_atras', 0.45),
    '22_02': ('Bárbaro barbudo', 'hombre', 'corto_guerrero', 0.35),
    '22_10': ('Explorador tuerto barbado', 'hombre', 'raya_medio_lisa', 0.10),
    '22_11': ('Asesino de perilla', 'hombre', 'militar', 0.20),
    '22_12': ('Luchador barbudo', 'hombre', 'corto_guerrero', 0.30),
}
LUM = np.array([0.30, 0.59, 0.11])


def hexc(c):
    return '#' + ''.join('%02x' % int(np.clip(round(v), 0, 255)) for v in c)


def mid_to_body(mid, k=0.44, s=1.8):
    # misma fórmula que faceMidToBodySkin del estudio
    c = (np.asarray(mid) / 255.0) ** 2.2
    L = c @ LUM
    o = np.clip(k * (L + s * (c - L)), 0, 1) ** (1 / 2.2)
    return hexc(o * 255)


def median_patch(a, pts, r):
    px = []
    for x, y in pts:
        p = a[int(y - r):int(y + r), int(x - r):int(x + r)].reshape(-1, 3)
        px.append(p)
    p = np.concatenate(px)
    o = np.argsort(p @ LUM)
    p = p[o[int(len(o) * 0.3):int(len(o) * 0.7)]]
    return p.mean(0)


def main():
    lm = json.load(open(os.path.join(REF, 'anclajes_detectados.json')))
    masks = {k: np.asarray(Image.open(os.path.join(TEX, f'rostro_{k}.png')))[..., 3] for k in ('guerrero', 'elfa')}
    defs = {}
    for fid, (label, gender, cut, beard) in FACES.items():
        sheet, rc = fid.split('_')
        r, c = int(rc[0]), int(rc[1])
        img = Image.open(os.path.join(REF, f'lamina_{sheet}.jpg')).convert('RGB')
        L = lm[fid]
        ox, oy = CX_COL[c] - 260, CY_ROW[r] - 290
        exL, exR, ey = L['eyeL'][0] + ox, L['eyeR'][0] + ox, L['eyeL'][1] + oy
        ex = 0.5 * (exL + exR)
        F = (L['mouth'] - L['eyeL'][1]) * 1.52          # ojos -> barbilla (la boca está al 66%)
        s = (CHIN - EY) / F
        # transformación inversa: pixel destino -> lámina
        out = img.transform((W, H), Image.AFFINE, (1 / s, 0, ex - CX / s, 0, 1 / s, ey - EY / s), resample=Image.BICUBIC)
        a = np.asarray(out).astype(np.float32)
        mask = masks['guerrero' if beard > 0 else 'elfa'].astype(np.float32) / 255.0
        # el papel de la lámina (beige claro y poco saturado) no es rostro: transparente
        paper = np.median(np.asarray(img).reshape(-1, 3)[::97], 0)
        dpap = np.linalg.norm(a - paper, axis=2)
        mask = mask * np.clip((dpap - 18.0) / 30.0, 0, 1)
        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32) / 255.0
        rgba = np.dstack([a, mask * 255]).clip(0, 255).astype(np.uint8)
        name = f'rostro_lam_{fid}.png'
        Image.fromarray(rgba, 'RGBA').save(os.path.join(TEX, name), optimize=True)
        # anclajes en la textura
        eL, eR = CX - (ex - exL) * s, CX + (exR - ex) * s
        Fd = CHIN - EY
        mouthY = EY + (L['mouth'] - L['eyeL'][1]) * s
        mw = 0.41 * (eR - eL)
        # colores
        hw = (eR - eL) / 2
        skin = median_patch(a, [(CX - hw * 1.05, EY + Fd * 0.30), (CX + hw * 1.05, EY + Fd * 0.30), (CX, EY - Fd * 0.45)], Fd * 0.06)
        lips = median_patch(a, [(CX, mouthY + 6)], 5)
        iris = []
        for x in (eL, eR):
            p = a[int(EY - 5):int(EY + 6), int(x - 5):int(x + 6)].reshape(-1, 3); iris.append(p)
        iris = np.concatenate(iris); iris = iris[np.argsort(iris @ LUM)][len(iris) // 4: len(iris) * 3 // 4].mean(0)
        top = a[max(0, int(EY - Fd * 0.78)):int(EY - Fd * 0.58), int(CX - hw * 1.3):int(CX + hw * 1.3)].reshape(-1, 3)
        sat = top.max(1) - top.min(1)
        hp = top[(top @ LUM < 200) | (sat > 40)]
        if len(hp) < 50: hp = top
        o = np.argsort(hp @ LUM)
        q = lambda f: hp[o[int(f * (len(o) - 1))]]
        defs['lam_' + fid] = {
            'label': label, 'gender': gender, 'hairCut': cut,
            **({'beardExt': beard} if beard > 0 else {}),
            'url': 'textures/' + name, 'size': [W, H],
            'eyeL': [round(eL, 1), EY], 'eyeR': [round(eR, 1), EY], 'chinY': CHIN, 'eyeWidth': round(0.42 * (eR - eL), 1),
            'browY': round(EY - Fd * 0.13, 1), 'noseY': round(EY + Fd * 0.45, 1),
            'mouthL': [round(CX - mw, 1), round(mouthY, 1)], 'mouthR': [round(CX + mw, 1), round(mouthY, 1)],
            'colors': {'hairRoot': hexc(q(0.15)), 'hairTip': hexc(q(0.45)), 'hairHigh': hexc(q(0.85)),
                       'skin': mid_to_body(skin), 'lips': hexc(lips), 'eyes': hexc(iris)},
        }
        print(fid, label, round(s, 3))
    json.dump(defs, open(os.path.join(TEX, 'rostros_lamina.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
