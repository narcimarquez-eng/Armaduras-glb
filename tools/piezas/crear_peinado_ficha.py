#!/usr/bin/env python3
"""Peinado texturizado a partir de la ficha de personaje (vistas frontal, trasera y laterales).

1. Separa cabeza y pelo de cada vista de la referencia (papel claro, pelo castaño muy oscuro).
2. Modela el volumen del peinado como casco visual (intersección de las siluetas de las 4 vistas)
   en coordenadas esféricas alrededor del centro de la cabeza.
3. Proyecta la ilustración sobre ese casco (mezcla de vistas según la orientación de cada punto)
   y usa la máscara de pelo como canal alfa: el contorno del peinado (nacimiento, puntas) sale
   del dibujo, no de la geometría.
4. Escribe un GLB estático con malla + textura y, en extras, las referencias (orejas, línea de ojos)
   para que el estudio lo encaje en la cabeza de cualquier personaje.

Uso:  python3 tools/piezas/crear_peinado_ficha.py
Salida: public/assets/aistudio/peinados/despeinado_ficha.glb
"""
from __future__ import annotations

import io
import json
import os
import struct

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
REF = os.path.join(HERE, 'referencias', 'peinado_despeinado_ficha.jpg')
OUT_DIR = os.path.join(REPO, 'public', 'assets', 'aistudio', 'peinados')
OUT = os.path.join(OUT_DIR, 'despeinado_ficha.glb')

K = 1.376  # la ficha original mide 2752 px; las cajas están medidas sobre la vista a 2000 px
# Cajas de cada vista (x0, y0, x1, y1). Todas comparten la línea de ojos (y = 228 dentro del recorte)
BOXES = {
    'front': (220, 100, 480, 400),
    'back': (780, 100, 1020, 400),
    'left': (1190, 100, 1530, 400),   # perfil mirando a la derecha de la imagen
    'right': (1550, 100, 1900, 400),  # perfil mirando a la izquierda de la imagen
}
EYE_V = 228.0
# Centros de oreja medidos en cada recorte (px): x de la oreja en los perfiles
EAR_U = {'left': 200.0, 'right': 254.0}
EAR_V = 258.0
LIMIT_V = {'front': 300.0, 'back': 352.0, 'left': 305.0, 'right': 305.0}   # por debajo: barba/cuello, no pelo


def morph(m: np.ndarray, r: int, op: str) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            sh = np.roll(np.roll(m, dy, 0), dx, 1)
            out = (out & sh) if op == 'e' else (out | sh)
    return out


def fill_holes(m: np.ndarray) -> np.ndarray:
    bg = np.zeros_like(m)
    bg[0, :] = ~m[0, :]; bg[-1, :] = ~m[-1, :]; bg[:, 0] = ~m[:, 0]; bg[:, -1] = ~m[:, -1]
    free = ~m
    while True:
        n = bg | (free & (np.roll(bg, 1, 0) | np.roll(bg, -1, 0) | np.roll(bg, 1, 1) | np.roll(bg, -1, 1)))
        if (n == bg).all():
            break
        bg = n
    return ~bg


def remove_guides(a: np.ndarray) -> np.ndarray:
    # Las líneas discontinuas azules/rojas de la ficha (y la cuadrícula) no son pelo: se rellenan con las filas vecinas
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    guide = ((b - r > 4) & (b > 60)) | ((r - g > 70) & (r - b > 70) & (g < 120))
    guide = guide | np.roll(guide, 1, 0) | np.roll(guide, -1, 0)
    out = a.copy()
    for dy in (4, 6, 8):
        src = 0.5 * (np.roll(a, dy, 0) + np.roll(a, -dy, 0))
        srcg = np.roll(guide, dy, 0) | np.roll(guide, -dy, 0)
        fill = guide & ~srcg
        out[fill] = src[fill]
        guide = guide & ~fill
    return out


def load_views():
    ref = Image.open(REF).convert('RGB')
    views = {}
    for name, box in BOXES.items():
        crop = ref.crop(tuple(int(v * K) for v in box))
        a = remove_guides(np.asarray(crop).astype(np.float32))
        lum = a @ np.array([0.3, 0.59, 0.11], np.float32)
        paper = np.median(np.concatenate([a[:6].reshape(-1, 3), a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)]), 0)
        d = np.linalg.norm(a - paper, axis=2)
        head = (d > 38) | (lum < 150)
        head = morph(morph(head, 2, 'e'), 2, 'd')
        head = morph(fill_holes(morph(head, 3, 'd')), 3, 'e')
        hair = (lum < 60) & head
        hair = morph(morph(hair, 2, 'd'), 2, 'e')
        hair = morph(morph(hair, 2, 'e'), 2, 'd')
        hair = morph(morph(hair, 4, 'd'), 4, 'e')          # cierra los brillos dentro de la melena
        vv = np.arange(a.shape[0])[:, None]
        uu = np.arange(a.shape[1])[None, :]
        hair &= vv < LIMIT_V[name]
        # La cara no es pelo: cejas, ojos y sombras quedan fuera (debajo del nacimiento del pelo)
        if name == 'front':
            cu = 0.5 * (np.nonzero(head[int(EYE_V)])[0].min() + np.nonzero(head[int(EYE_V)])[0].max())
            hair &= ~((np.abs(uu - cu) < 112) & (vv > 168))
        elif name == 'left':
            hair &= ~(((uu - EAR_U['left']) > 60 + 0.9 * (vv - 172)) & (vv > 150))
        elif name == 'right':
            hair &= ~(((EAR_U['right'] - uu) > 60 + 0.9 * (vv - 172)) & (vv > 150))
        # alfa suave (borde de mechones)
        alpha = np.asarray(Image.fromarray((hair * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255.0
        views[name] = {'img': a / 255.0, 'head': head, 'alpha': alpha, 'w': a.shape[1], 'h': a.shape[0]}
    # Vista cenital (coronilla): solo la mancha de pelo central (sin hombreras)
    top = ref.crop(tuple(int(v * K) for v in (1450, 800, 1940, 1070)))
    a = np.asarray(top).astype(np.float32); lum = a @ np.array([0.3, 0.59, 0.11], np.float32)
    hh, ww = lum.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    hair = (lum < 70) & (xx > ww * 0.30) & (xx < ww * 0.70) & (yy < hh * 0.80)
    hair = morph(morph(hair, 3, 'd'), 3, 'e'); hair = morph(morph(hair, 3, 'e'), 3, 'd')
    hair = fill_holes(morph(hair, 5, 'd')); hair = morph(hair, 5, 'e')
    alpha = np.asarray(Image.fromarray((hair * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32) / 255.0
    ys, xs = np.nonzero(hair)
    views['top'] = {'img': a / 255.0, 'head': hair, 'alpha': alpha, 'w': ww, 'h': hh,
                    'bbox': (xs.min(), xs.max(), ys.min(), ys.max())}
    # Centros horizontales de las vistas frontal/trasera: mitad de la silueta a la altura de los ojos
    for name in ('front', 'back'):
        row = views[name]['head'][int(EYE_V)]
        xs = np.nonzero(row)[0]
        views[name]['cu'] = 0.5 * (xs.min() + xs.max())
    return views


def project(views, name, P):
    """Punto(s) 3D en px (X: izquierda del personaje = derecha de la imagen frontal, Y: arriba desde la línea
    de ojos, Z: hacia delante desde la oreja) -> coordenadas de imagen (u, v) de esa vista."""
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    v = EYE_V - Y
    if name == 'top':
        t = views['top']
        return t['cu'] + X * t['su'], t['cv'] + Z * t['sv']
    if name == 'front':
        u = views['front']['cu'] + X
    elif name == 'back':
        u = views['back']['cu'] - X
    elif name == 'left':
        u = EAR_U['left'] + Z
    else:
        u = EAR_U['right'] - Z
    return u, v


def sample_mask(m, u, v):
    h, w = m.shape
    ui = np.round(u).astype(int); vi = np.round(v).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.zeros(u.shape, bool)
    out[ok] = m[vi[ok], ui[ok]]
    return out


def sample_img(img, u, v):
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001); v = np.clip(v, 0, h - 1.001)
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    fu = (u - u0)[..., None]; fv = (v - v0)[..., None]
    if img.ndim == 2:
        img = img[..., None]
    c = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv) +
         img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)
    return c


VIEW_DIR = {'front': np.array([0, 0, 1.0]), 'back': np.array([0, 0, -1.0]),
            'left': np.array([-1.0, 0, 0]), 'right': np.array([1.0, 0, 0]), 'top': np.array([0, 1.0, 0])}
SIDE_VIEWS = ('front', 'back', 'left', 'right')
CENTER = np.array([0.0, 40.0, 18.0])
PHI_MIN, PHI_MAX = np.radians(-56.0), np.radians(89.0)


def hull_radius(views, NT, NP):
    th = np.linspace(-np.pi, np.pi, NT, endpoint=False)          # 0 = delante (+Z)
    ph = np.linspace(PHI_MIN, PHI_MAX, NP)
    TH, PH = np.meshgrid(th, ph)
    D = np.stack([np.cos(PH) * np.sin(TH), np.sin(PH), np.cos(PH) * np.cos(TH)], -1)
    R = np.zeros(TH.shape)
    alive = np.ones(TH.shape, bool)
    for r in np.arange(20.0, 280.0, 1.0):
        P = CENTER + D * r
        inside = np.ones(TH.shape, bool)
        for n in SIDE_VIEWS:
            u, v = project(views, n, P)
            inside &= sample_mask(views[n]['head'], u, v)
        alive &= inside
        R[alive] = r
    # Redondea la coronilla: la intersección de siluetas deja aristas de 'caja'; se limita con un elipsoide
    # (exponente 2.6, algo cuadrado como un peinado con volumen) ajustado a la extensión del casco
    P = CENTER + D * R[..., None]
    ext = np.maximum(np.abs(P - CENTER).reshape(-1, 3).max(0), 1.0)
    lo = (P - CENTER).reshape(-1, 3).min(0)
    ax = np.where(D > 0, ext, np.abs(lo))
    e = 2.0
    Rell = 1.0 / (np.abs(D / ax) ** e).sum(-1) ** (1.0 / e)
    R = np.minimum(R, Rell * 1.0)
    # suavizado ligero (quita escalones del recorte)
    for _ in range(3):
        R = 0.5 * R + 0.125 * (np.roll(R, 1, 1) + np.roll(R, -1, 1) + np.roll(R, 1, 0) + np.roll(R, -1, 0))
    return TH, PH, D, R


def normals_of(P):
    du = np.roll(P, -1, 1) - np.roll(P, 1, 1)
    dv = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    dv[0] = P[1] - P[0]; dv[-1] = P[-1] - P[-2]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    radial = P - CENTER
    flip = (n * radial).sum(-1) < 0
    n[flip] *= -1
    return n


def bake(views, P, N):
    col = np.zeros(P.shape[:-1] + (3,)); a = np.zeros(P.shape[:-1]); wsum = np.zeros(P.shape[:-1])
    for name, d in VIEW_DIR.items():
        w = np.clip((N * d).sum(-1), 0, 1) ** 4
        u, v = project(views, name, P)
        c = sample_img(views[name]['img'], u, v)
        al = sample_img(views[name]['alpha'], u, v)[..., 0]
        col += c * w[..., None]; a += al * w; wsum += w
    col /= wsum[..., None] + 1e-6; a /= wsum + 1e-6
    return col, a


def calibrate_top(views, P):
    # Encaja la vista cenital con la parte alta del casco (anchura X y fondo Z del pelo por encima de la frente)
    up = P[P[..., 1] > CENTER[1] + 35]
    x0, x1 = up[:, 0].min(), up[:, 0].max(); z0, z1 = up[:, 2].min(), up[:, 2].max()
    bx0, bx1, by0, by1 = views['top']['bbox']
    t = views['top']
    t['su'] = (bx1 - bx0) / (x1 - x0); t['cu'] = bx0 - x0 * t['su']
    t['sv'] = (by1 - by0) / (z1 - z0); t['cv'] = by0 - z0 * t['sv']


def write_glb(path, pos, nor, uv, idx, png_bytes, extras):
    def pad4(b, fill=b'\x00'):
        return b + fill * ((4 - len(b) % 4) % 4)
    bins = []
    views_meta = []
    off = 0
    for arr, target in ((pos.astype('<f4'), 34962), (nor.astype('<f4'), 34962), (uv.astype('<f4'), 34962),
                        (idx.astype('<u4'), 34963)):
        b = arr.tobytes(); views_meta.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(b), 'target': target})
        b = pad4(b); bins.append(b); off += len(b)
    img_off = off; bins.append(pad4(png_bytes)); off += len(pad4(png_bytes))
    views_meta.append({'buffer': 0, 'byteOffset': img_off, 'byteLength': len(png_bytes)})
    bin_blob = b''.join(bins)
    gltf = {
        'asset': {'version': '2.0', 'generator': 'crear_peinado_ficha.py'},
        'scene': 0, 'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0, 'name': 'Peinado_Despeinado_Ficha', 'extras': extras}],
        'meshes': [{'name': 'Peinado_Despeinado_Ficha', 'primitives': [{
            'attributes': {'POSITION': 0, 'NORMAL': 1, 'TEXCOORD_0': 2}, 'indices': 3, 'material': 0}]}],
        'materials': [{'name': 'Pelo_Ficha', 'doubleSided': True, 'alphaMode': 'MASK', 'alphaCutoff': 0.42,
                       'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0.0, 'roughnessFactor': 0.62}}],
        'textures': [{'source': 0, 'sampler': 0}],
        'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 33071}],
        'images': [{'bufferView': 4, 'mimeType': 'image/png'}],
        'accessors': [
            {'bufferView': 0, 'componentType': 5126, 'count': len(pos), 'type': 'VEC3',
             'min': pos.min(0).tolist(), 'max': pos.max(0).tolist()},
            {'bufferView': 1, 'componentType': 5126, 'count': len(nor), 'type': 'VEC3'},
            {'bufferView': 2, 'componentType': 5126, 'count': len(uv), 'type': 'VEC2'},
            {'bufferView': 3, 'componentType': 5125, 'count': len(idx), 'type': 'SCALAR'},
        ],
        'bufferViews': views_meta,
        'buffers': [{'byteLength': len(bin_blob)}],
    }
    js = pad4(json.dumps(gltf, separators=(',', ':')).encode(), b' ')
    total = 12 + 8 + len(js) + 8 + len(bin_blob)
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total))
        f.write(struct.pack('<I4s', len(js), b'JSON')); f.write(js)
        f.write(struct.pack('<I4s', len(bin_blob), b'BIN\x00')); f.write(bin_blob)


def main():
    views = load_views()
    # Textura: rejilla fina en (azimut, elevación); malla: rejilla más gruesa sobre el mismo radio
    TT, TP = 1024, 512
    TH, PH, D, R = hull_radius(views, TT, TP)
    P = CENTER + D * R[..., None] * 1.02          # 2% hacia fuera: el pelo queda sobre la piel
    N = normals_of(P)
    calibrate_top(views, P)
    col, alpha = bake(views, P, N)
    tex = np.concatenate([col, alpha[..., None]], -1)
    tex = np.clip(tex * 255, 0, 255).astype(np.uint8)          # fila k de la imagen = elevación k (glTF: v=0 arriba)
    tex_img = Image.fromarray(tex, 'RGBA')
    # rellena el color bajo el alfa (evita halos al filtrar)
    rgb = tex_img.convert('RGB').filter(ImageFilter.MedianFilter(3))
    tex_img = Image.merge('RGBA', (*rgb.split(), tex_img.split()[3]))
    bio = io.BytesIO(); tex_img.save(bio, 'PNG', optimize=True)

    # Malla: 1 de cada 8 columnas/filas, solo las bandas que tienen pelo
    st = 8
    Pm = P[::st, ::st]; Nm = N[::st, ::st]; Am = alpha[::st, ::st]
    nr, nc = Pm.shape[:2]
    uv = np.stack(np.meshgrid(np.arange(nc) * st / TT, np.arange(nr) * st / (TP - 1)), -1)
    # cerrar el azimut duplicando la primera columna
    Pm = np.concatenate([Pm, Pm[:, :1]], 1); Nm = np.concatenate([Nm, Nm[:, :1]], 1); Am = np.concatenate([Am, Am[:, :1]], 1)
    uv = np.concatenate([uv, np.stack([np.ones(nr), uv[:, 0, 1]], -1)[:, None]], 1)
    nc += 1
    # px -> metros: separación entre orejas de la ficha ~ 0.15 m (el estudio reescala con sus orejas)
    # media anchura del cráneo a la altura de las orejas (sin el pabellón de la oreja, ~24 px por lado)
    ear_x = 0.5 * float(np.ptp(views['front']['head'][int(EAR_V)].nonzero()[0])) - 24.0
    m_per_px = 0.075 / max(1.0, ear_x)
    # origen en el centro entre orejas (Y de la oreja = EYE_V - EAR_V px respecto a la línea de ojos; Z = 0)
    pos = (Pm - np.array([0.0, EYE_V - EAR_V, 0.0])) * m_per_px
    idx = []
    keep = Am > 0.02
    for i in range(nr - 1):
        for j in range(nc - 1):
            q = [i * nc + j, i * nc + j + 1, (i + 1) * nc + j + 1, (i + 1) * nc + j]
            if not (keep[i, j] or keep[i, j + 1] or keep[i + 1, j] or keep[i + 1, j + 1]):
                continue
            idx += [q[0], q[2], q[1], q[0], q[3], q[2]]
    idx = np.array(idx, np.uint32)
    used = np.unique(idx)
    remap = -np.ones(nr * nc, np.int64); remap[used] = np.arange(len(used))
    pos = pos.reshape(-1, 3)[used]; nor = Nm.reshape(-1, 3)[used]; uvf = uv.reshape(-1, 2)[used]
    idx = remap[idx].astype(np.uint32)
    extras = {
        'tipo': 'peinado_texturizado',
        'nombre': 'Despeinado (ficha)',
        'orejas_separacion_m': float(2 * ear_x * m_per_px),
        'linea_ojos_y_m': float((EAR_V - EYE_V) * m_per_px),
        'origen': 'centro entre orejas',
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    write_glb(OUT, pos, nor, uvf, idx, bio.getvalue(), extras)
    tex_img.save(os.path.join(OUT_DIR, 'despeinado_ficha_textura.png'))
    print('GLB', OUT, 'verts', len(pos), 'tris', len(idx) // 3, 'bytes', os.path.getsize(OUT))


if __name__ == '__main__':
    main()
