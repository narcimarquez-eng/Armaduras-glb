#!/usr/bin/env python3
"""Generador procedural en Python de la armadura asimétrica Segador de Cuervo & Hueso (Ravenbone Reaper).

Rompe el molde clásico de "placas simétricas con ribetes dorados":
  - Estética orgánica / asimétrica / cazador oscuro: paño carbón mate, lino carmesí,
    hueso tallado marfil, hierro forjado ennegrecido, cuero aceitado y ascuas vivas (CERO oro).
  - Genera .glb compatibles con el esqueleto `CreadorHumano_hombre` y Three.js / Blender
    para cada ranura física:
      helmet/10, shoulders/13, torso/21, gloves/7, belt/9,
      tassets/10, cuisses/9, knees/9, boots/14, weapon/82, shield/12.
  - Actualiza idempotente `piezas/catalogo-estudio.json` (incluyendo `underlayer` 17 y `shield` 12)
    y regenera `piezas/catalogo.json` mediante `tools/piezas/generar-catalogo.py`.

Uso:
    python3 tools/piezas/crear_ravenbone.py
"""
from __future__ import annotations

import io
import json
import math
import os
import struct
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIEZAS_DIR = os.path.join(ROOT_DIR, "piezas")
CATALOGO_ESTUDIO_PATH = os.path.join(PIEZAS_DIR, "catalogo-estudio.json")
GENERAR_CATALOGO_PATH = os.path.join(os.path.dirname(__file__), "generar-catalogo.py")

SKELETON_TEMPLATE = [
    {"name": "CreadorHumano_hombre", "translation": [0.0, 1.3138342750793461, 0.0], "children": [1]},
    {"name": "Pelvis", "children": [2, 13, 18]},
    {"name": "Torso", "children": [3, 5, 9]},
    {"name": "Neck", "translation": [0.0, 0.29652864, -0.005], "children": [4]},
    {"name": "Head", "translation": [0.0, 0.0792, 0.008], "scale": [0.9, 0.9, 0.9]},
    {"name": "Shoulder_L", "translation": [0.24168, 0.22712832, 0.0],
     "rotation": [0.0, 0.0, 0.31456656061611776, 0.9492354180824408], "children": [6]},
    {"name": "UpperArm_L", "children": [7]},
    {"name": "Elbow_L", "translation": [0.0, -0.297024, 0.0],
     "rotation": [-0.04996355159372734, -0.0012493490817475706, -0.024966155679015008, 0.9984381671939551], "children": [8]},
    {"name": "Forearm_L", "children": [23]},
    {"name": "Shoulder_R", "translation": [-0.24168, 0.22712832, 0.0],
     "rotation": [0.0, 0.0, -0.31456656061611776, 0.9492354180824408], "children": [10]},
    {"name": "UpperArm_R", "children": [11]},
    {"name": "Elbow_R", "translation": [0.0, -0.297024, 0.0],
     "rotation": [-0.04996355159372734, 0.0012493490817475706, 0.024966155679015008, 0.9984381671939551], "children": [12]},
    {"name": "Forearm_R", "children": [24]},
    {"name": "Hip_L", "translation": [0.0987, -0.29021952, 0.0],
     "rotation": [0.0, 0.0, 0.01999866669333308, 0.9998000066665778], "children": [14]},
    {"name": "Thigh_L", "children": [15]},
    {"name": "Knee_L", "translation": [0.0, -0.483, 0.0],
     "rotation": [0.01999866669333308, 0.0, 0.0, 0.9998000066665778], "children": [16]},
    {"name": "Shin_L", "children": [17]},
    {"name": "Ankle_L", "translation": [0.0, -0.46, 0.0],
     "rotation": [-0.01999866669333308, 0.0, 0.0, 0.9998000066665778]},
    {"name": "Hip_R", "translation": [-0.0987, -0.29021952, 0.0],
     "rotation": [0.0, 0.0, -0.01999866669333308, 0.9998000066665778], "children": [19]},
    {"name": "Thigh_R", "children": [20]},
    {"name": "Knee_R", "translation": [0.0, -0.483, 0.0],
     "rotation": [0.01999866669333308, 0.0, 0.0, 0.9998000066665778], "children": [21]},
    {"name": "Shin_R", "children": [22]},
    {"name": "Ankle_R", "translation": [0.0, -0.46, 0.0],
     "rotation": [-0.01999866669333308, 0.0, 0.0, 0.9998000066665778]},
    {"name": "Wrist_L", "translation": [0.0, -0.275808, 0.0]},
    {"name": "Wrist_R", "translation": [0.0, -0.275808, 0.0]},
]


# ==============================================================================
# 1. TEXTURAS PBR ORGÁNICAS RAVENBONE (PAÑO, HUESO, HIERRO, CARMESÍ, CUERO)
# ==============================================================================
def _img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def generate_ravenbone_textures() -> Dict[str, bytes]:
    rng = np.random.default_rng(77)

    # A) Paño carbón mate con trama diagonal de abrigo gastado
    s = 256
    cloth = Image.new("RGB", (s, s), (22, 23, 28))
    cd = ImageDraw.Draw(cloth)
    for y in range(-s, s * 2, 6):
        cd.line([(0, y), (s, y + s // 2)], fill=(30, 32, 39), width=1)
        cd.line([(0, y + 3), (s, y - s // 2)], fill=(16, 17, 21), width=1)
    arr = np.clip(np.array(cloth, dtype=np.int16) + rng.integers(-5, 6, (s, s, 1)), 0, 255).astype(np.uint8)
    cloth = Image.fromarray(arr, "RGB")

    # B) Hueso tallado envejecido con vetas y poros marfil/ocre oscuro
    bone = Image.new("RGB", (128, 128), (168, 152, 126))
    bd = ImageDraw.Draw(bone)
    for _ in range(42):
        y0 = int(rng.integers(0, 128))
        x0 = int(rng.integers(0, 128))
        bd.line([(x0, y0), (x0 + int(rng.integers(-14, 15)), y0 + int(rng.integers(12, 42)))],
                fill=(124, 108, 86), width=1)
    barr = np.clip(np.array(bone, dtype=np.int16) + rng.integers(-8, 9, (128, 128, 1)), 0, 255).astype(np.uint8)
    bone = Image.fromarray(barr, "RGB")

    # C) Hierro forjado ennegrecido con marcas de martillo
    iron = Image.new("RGB", (128, 128), (34, 38, 46))
    idraw = ImageDraw.Draw(iron)
    for _ in range(90):
        cx, cy = int(rng.integers(0, 128)), int(rng.integers(0, 128))
        r = int(rng.integers(3, 9))
        tone = int(rng.integers(22, 54))
        idraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(tone, tone + 2, tone + 8))

    # D) Lino sangre carmesí profundo (oxblood)
    crim = Image.new("RGB", (128, 128), (68, 12, 18))
    carr = np.clip(np.array(crim, dtype=np.int16) + rng.integers(-7, 8, (128, 128, 1)), 0, 255).astype(np.uint8)
    crim = Image.fromarray(carr, "RGB")

    return {
        "cloth_albedo": _img_to_png_bytes(cloth),
        "bone_albedo": _img_to_png_bytes(bone),
        "iron_albedo": _img_to_png_bytes(iron),
        "crimson_albedo": _img_to_png_bytes(crim),
    }


# ==============================================================================
# 2. UTILIDADES DE GEOMETRÍA 3D
# ==============================================================================
class MeshPart:
    def __init__(self) -> None:
        self.positions: List[np.ndarray] = []
        self.normals: List[np.ndarray] = []
        self.uvs: List[np.ndarray] = []
        self.indices: List[np.ndarray] = []
        self.v_offset = 0

    def add(self, pos: np.ndarray, idx: np.ndarray,
            uv: Optional[np.ndarray] = None,
            nrm: Optional[np.ndarray] = None) -> "MeshPart":
        pos = np.asarray(pos, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.uint32).reshape(-1)
        if len(pos) == 0 or len(idx) == 0:
            return self
        if uv is None:
            uv = np.stack([pos[:, 0] * 2.5 + 0.5, pos[:, 1] * 2.5 + 0.5], axis=1).astype(np.float32)
        else:
            uv = np.asarray(uv, dtype=np.float32)
        if nrm is None:
            nrm = compute_vertex_normals(pos, idx)
        else:
            nrm = np.asarray(nrm, dtype=np.float32)
        self.positions.append(pos)
        self.normals.append(nrm)
        self.uvs.append(uv)
        self.indices.append(idx + self.v_offset)
        self.v_offset += len(pos)
        return self

    def is_empty(self) -> bool:
        return self.v_offset == 0

    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.vstack(self.positions),
            np.vstack(self.normals),
            np.vstack(self.uvs),
            np.concatenate(self.indices),
        )


def compute_vertex_normals(pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    tris = idx.reshape(-1, 3)
    v0, v1, v2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    nrm = np.zeros_like(pos, dtype=np.float32)
    np.add.at(nrm, tris[:, 0], fn)
    np.add.at(nrm, tris[:, 1], fn)
    np.add.at(nrm, tris[:, 2], fn)
    lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-8, 1.0, lengths)
    return (nrm / lengths).astype(np.float32)


def transform_mesh(pos: np.ndarray, translate=(0, 0, 0),
                   rot_xyz=(0, 0, 0), scale=(1, 1, 1)) -> np.ndarray:
    p = pos.astype(np.float32).copy()
    p[:, 0] *= scale[0]
    p[:, 1] *= scale[1]
    p[:, 2] *= scale[2]
    rx, ry, rz = rot_xyz
    if abs(rx) > 1e-6:
        c, s = math.cos(rx), math.sin(rx)
        p[:, 1], p[:, 2] = p[:, 1] * c - p[:, 2] * s, p[:, 1] * s + p[:, 2] * c
    if abs(ry) > 1e-6:
        c, s = math.cos(ry), math.sin(ry)
        p[:, 0], p[:, 2] = p[:, 0] * c + p[:, 2] * s, -p[:, 0] * s + p[:, 2] * c
    if abs(rz) > 1e-6:
        c, s = math.cos(rz), math.sin(rz)
        p[:, 0], p[:, 1] = p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c
    p[:, 0] += translate[0]
    p[:, 1] += translate[1]
    p[:, 2] += translate[2]
    return p


def make_box(sx: float, sy: float, sz: float, center=(0, 0, 0),
             rot=(0, 0, 0), bevel: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    if bevel <= 0:
        verts = np.array([
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
            [hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz],
            [-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz],
            [hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz],
            [-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz],
        ], dtype=np.float32)
        idx = []
        for f in range(6):
            o = f * 4
            idx.extend([o, o + 1, o + 2, o, o + 2, o + 3])
        return transform_mesh(verts, center, rot), np.array(idx, dtype=np.uint32)
    b = min(bevel, hx * 0.45, hy * 0.45)
    pts = [
        (-hx + b, -hy), (hx - b, -hy), (hx, -hy + b), (hx, hy - b),
        (hx - b, hy), (-hx + b, hy), (-hx, hy - b), (-hx, -hy + b),
    ]
    return make_extruded_polygon(pts, sz, bevel=min(b, sz * 0.3), center=center, rot=rot)


def make_extruded_polygon(pts2d: List[Tuple[float, float]], depth: float,
                          bevel: float = 0.0, center=(0, 0, 0),
                          rot=(0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    n = len(pts2d)
    poly = np.array(pts2d, dtype=np.float32)
    cx, cy = poly.mean(axis=0)
    hz = depth * 0.5
    b = min(bevel, hz * 0.45)
    rings = []
    if b > 1e-5:
        inner = (poly - [cx, cy]) * 0.88 + [cx, cy]
        rings = [(inner, -hz - b * 0.6), (poly, -hz + b * 0.4), (poly, hz - b * 0.4), (inner, hz + b * 0.6)]
    else:
        rings = [(poly, -hz), (poly, hz)]
    verts, tris, ring_starts = [], [], []
    for r_pts, zval in rings:
        ring_starts.append(len(verts))
        for x, y in r_pts:
            verts.append([float(x), float(y), float(zval)])
    for r in range(len(rings) - 1):
        s0, s1 = ring_starts[r], ring_starts[r + 1]
        for i in range(n):
            ni = (i + 1) % n
            a, b0, c, d = s0 + i, s0 + ni, s1 + ni, s1 + i
            tris.extend([a, b0, c, a, c, d])
    back_c = len(verts); verts.append([float(cx), float(cy), float(rings[0][1])])
    bs = ring_starts[0]
    for i in range(n):
        tris.extend([back_c, bs + (i + 1) % n, bs + i])
    front_c = len(verts); verts.append([float(cx), float(cy), float(rings[-1][1])])
    fs = ring_starts[-1]
    for i in range(n):
        tris.extend([front_c, fs + i, fs + (i + 1) % n])
    return transform_mesh(np.array(verts, dtype=np.float32), center, rot), np.array(tris, dtype=np.uint32)


def make_cylinder(r_top: float, r_bot: float, height: float, segs: int = 12,
                  center=(0, 0, 0), rot=(0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    verts, tris = [], []
    hy = height * 0.5
    for i in range(segs):
        a = (i / segs) * math.tau
        verts.append([math.cos(a) * r_bot, -hy, math.sin(a) * r_bot])
    for i in range(segs):
        a = (i / segs) * math.tau
        verts.append([math.cos(a) * r_top, hy, math.sin(a) * r_top])
    bot_c = len(verts); verts.append([0.0, -hy, 0.0])
    top_c = len(verts); verts.append([0.0, hy, 0.0])
    for i in range(segs):
        ni = (i + 1) % segs
        tris.extend([i, segs + i, segs + ni, i, segs + ni, ni])
        tris.extend([bot_c, i, ni])
        tris.extend([top_c, segs + ni, segs + i])
    return transform_mesh(np.array(verts, dtype=np.float32), center, rot), np.array(tris, dtype=np.uint32)


def make_sphere(radius: float, segs: int = 14, rings: int = 10,
                center=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)) -> Tuple[np.ndarray, np.ndarray]:
    verts, tris = [], []
    for ir in range(rings + 1):
        phi = (ir / rings) * math.pi
        y = math.cos(phi) * radius
        rr = math.sin(phi) * radius
        for iseg in range(segs):
            th = (iseg / segs) * math.tau
            verts.append([math.cos(th) * rr, y, math.sin(th) * rr])
    for ir in range(rings):
        for iseg in range(segs):
            ni = (iseg + 1) % segs
            a = ir * segs + iseg
            b = ir * segs + ni
            c = (ir + 1) * segs + ni
            d = (ir + 1) * segs + iseg
            tris.extend([a, b, c, a, c, d])
    return transform_mesh(np.array(verts, dtype=np.float32), center, rot, scale), np.array(tris, dtype=np.uint32)


def make_tube_curve(points: List[Tuple[float, float, float]], radii: List[float],
                    segs: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Tubo orgánico con radio variable a lo largo de una polilínea 3D (cuernos, costillas, garras, astas)."""
    pts = [np.array(p, dtype=np.float32) for p in points]
    n = len(pts)
    verts, tris = [], []
    prev_n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for k in range(n):
        if k == 0:
            tan = pts[1] - pts[0]
        elif k == n - 1:
            tan = pts[-1] - pts[-2]
        else:
            tan = pts[k + 1] - pts[k - 1]
        t_len = float(np.linalg.norm(tan))
        tan = tan / max(t_len, 1e-6)
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(tan[1]) > 0.88 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        norm = np.cross(ref, tan)
        if np.linalg.norm(norm) < 1e-5:
            norm = prev_n
        else:
            norm = norm / np.linalg.norm(norm)
        prev_n = norm
        binorm = np.cross(tan, norm)
        rad = max(0.0008, float(radii[k]))
        for s in range(segs):
            ang = (s / segs) * math.tau
            p = pts[k] + (math.cos(ang) * norm + math.sin(ang) * binorm) * rad
            verts.append([float(p[0]), float(p[1]), float(p[2])])
    for k in range(n - 1):
        for s in range(segs):
            ns = (s + 1) % segs
            a, b = k * segs + s, k * segs + ns
            c, d = (k + 1) * segs + ns, (k + 1) * segs + s
            tris.extend([a, b, c, a, c, d])
    # Tapas
    c0 = len(verts); verts.append([float(pts[0][0]), float(pts[0][1]), float(pts[0][2])])
    c1 = len(verts); verts.append([float(pts[-1][0]), float(pts[-1][1]), float(pts[-1][2])])
    last = (n - 1) * segs
    for s in range(segs):
        ns = (s + 1) % segs
        tris.extend([c0, ns, s])
        tris.extend([c1, last + s, last + ns])
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.uint32)


def make_curved_shell(
    y0: float, y1: float,
    rx0: float, rz0: float,
    rx1: float, rz1: float,
    ang_min: float, ang_max: float,
    thickness: float = 0.008,
    ny: int = 10, na: int = 18,
    tatter_waves: int = 0,
    tatter_amp: float = 0.0,
    flare_top: float = 0.0,
    flare_bot: float = 0.0,
    peak_front_y: float = 0.0,
    center=(0, 0, 0),
    rot=(0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(0.0, 1.0, ny)
    angs = np.linspace(ang_min, ang_max, na)

    def eval_pt(vn: float, th: float, off: float) -> Tuple[float, float, float]:
        ym = y0 + (y1 - y0) * vn
        if tatter_waves > 0 and tatter_amp != 0.0:
            ym -= tatter_amp * (0.5 + 0.5 * math.sin(th * tatter_waves)) * (1.0 - vn)
        if peak_front_y != 0.0:
            ym += peak_front_y * (max(0.0, math.cos(th)) ** 2.2) * vn
        rx = rx0 + (rx1 - rx0) * vn + flare_top * (vn ** 2) + flare_bot * ((1.0 - vn) ** 2)
        rz = rz0 + (rz1 - rz0) * vn + flare_top * (vn ** 2) + flare_bot * ((1.0 - vn) ** 2)
        return math.sin(th) * (rx + off), ym, math.cos(th) * (rz + off)

    verts = []
    for off in (thickness * 0.5, -thickness * 0.5):
        for v in ys:
            for th in angs:
                verts.append(eval_pt(float(v), float(th), off))
    tris, stride, ib = [], na, ny * na
    for j in range(ny - 1):
        for i in range(na - 1):
            a = j * stride + i
            b, c, d = a + 1, (j + 1) * stride + i + 1, (j + 1) * stride + i
            tris.extend([a, b, c, a, c, d])
            ia, ib0, ic, id0 = ib + a, ib + b, ib + c, ib + d
            tris.extend([ia, ic, ib0, ia, id0, ic])
    for i in range(na - 1):
        tris.extend([i, ib + i, ib + i + 1, i, ib + i + 1, i + 1])
        row = (ny - 1) * stride
        tris.extend([row + i, row + i + 1, ib + row + i + 1, row + i, ib + row + i + 1, ib + row + i])
    for j in range(ny - 1):
        o0, o1 = j * stride, (j + 1) * stride
        tris.extend([o0, o1, ib + o1, o0, ib + o1, ib + o0])
        r0, r1 = j * stride + na - 1, (j + 1) * stride + na - 1
        tris.extend([r0, ib + r0, ib + r1, r0, ib + r1, r1])
    return transform_mesh(np.array(verts, dtype=np.float32), center, rot), np.array(tris, dtype=np.uint32)


def add_sculpted_skull(part_dict: Dict[str, MeshPart], scale: float = 1.0,
                       center=(0, 0, 0), rot=(0, 0, 0), with_jaw: bool = True) -> None:
    """Cráneo de cuervo/bestia alargado con cuencas de ascua y correas de hierro."""
    def put(pos_arr, idx_arr, mkey: str):
        part_dict[mkey].add(transform_mesh(pos_arr, translate=center, rot_xyz=rot), idx_arr)

    put(*make_sphere(0.054 * scale, segs=14, rings=10, scale=(0.92, 1.04, 1.28)), "bone")
    put(*make_cylinder(0.004 * scale, 0.036 * scale, 0.088 * scale, segs=6,
                       center=(0, -0.012 * scale, 0.068 * scale), rot=(math.pi * 0.56, 0, 0)), "bone")
    for sgn in (-1, 1):
        put(*make_box(0.028 * scale, 0.012 * scale, 0.044 * scale,
                      center=(sgn * 0.028 * scale, 0.016 * scale, 0.042 * scale),
                      rot=(0.18, sgn * 0.25, sgn * -0.22), bevel=0.002), "bone")
        put(*make_sphere(0.0085 * scale, segs=8, rings=6,
                         center=(sgn * 0.024 * scale, 0.004 * scale, 0.056 * scale),
                         scale=(1.1, 0.7, 0.9)), "ember")
    if with_jaw:
        put(*make_box(0.054 * scale, 0.024 * scale, 0.066 * scale,
                      center=(0, -0.038 * scale, 0.034 * scale), bevel=0.003), "bone")
        put(*make_box(0.086 * scale, 0.014 * scale, 0.018 * scale,
                      center=(0, -0.010 * scale, 0.012 * scale), bevel=0.002), "iron")


# ==============================================================================
# 3. CONSTRUCTORES DE LAS 11 PIEZAS ASIMÉTRICAS SEGADOR DE CUERVO & HUESO
# ==============================================================================
def _parts_dict() -> Dict[str, MeshPart]:
    return {
        "cloth": MeshPart(),
        "bone": MeshPart(),
        "iron": MeshPart(),
        "crimson": MeshPart(),
        "leather": MeshPart(),
        "ember": MeshPart(),
    }


def build_rb_helmet() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Head": _parts_dict()}
    h = out["Head"]
    # Capucha profunda con pico frontal proyectado sobre la frente (Head local)
    h["cloth"].add(*make_curved_shell(
        0.020, 0.245, 0.132, 0.145, 0.118, 0.165,
        -math.pi * 0.94, math.pi * 0.94, thickness=0.011,
        flare_top=0.018, peak_front_y=0.042
    ))
    h["crimson"].add(*make_curved_shell(
        0.026, 0.236, 0.125, 0.138, 0.110, 0.156,
        -math.pi * 0.75, math.pi * 0.75, thickness=0.005,
        flare_top=0.012, peak_front_y=0.032
    ))
    # Media máscara ósea inferior en mandíbula/mejillas + dientes de hierro
    h["bone"].add(*make_curved_shell(
        -0.010, 0.082, 0.106, 0.116, 0.112, 0.122,
        -0.98, 0.98, thickness=0.010, peak_front_y=0.018
    ))
    for k in range(5):
        a = -0.52 + k * 0.26
        h["iron"].add(*make_box(0.007, 0.022, 0.008,
                                center=(math.sin(a) * 0.120, 0.036, math.cos(a) * 0.126),
                                rot=(0, a, 0)))
    # Ojos ascua brillantes bajo la sombra de la capucha
    for sgn in (-1, 1):
        h["ember"].add(*make_sphere(0.010, segs=8, rings=6,
                                    center=(sgn * 0.036, 0.112, 0.118), scale=(1.2, 0.65, 0.7)))
        # Cuernos curvos barridos hacia atrás
        pts = [
            (sgn * 0.085, 0.175, 0.030),
            (sgn * 0.135, 0.235, -0.025),
            (sgn * 0.165, 0.285, -0.105),
            (sgn * 0.140, 0.345, -0.185),
        ]
        h["bone"].add(*make_tube_curve(pts, [0.020, 0.014, 0.009, 0.002], segs=9))
    return out


def build_rb_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    out = {"UpperArm_L": _parts_dict(), "Torso": _parts_dict(), "UpperArm_R": _parts_dict()}
    # IZQUIERDA: Hombrera protagonista de Cráneo Cornudo + Manto carmesí unilateral
    ul = out["UpperArm_L"]
    add_sculpted_skull(ul, scale=1.42, center=(0.068, 0.025, 0.010), rot=(-0.15, math.pi * 0.42, -0.25))
    for k in (-1, 1):
        pts = [
            (0.075, 0.055, k * 0.032),
            (0.145, 0.115, k * 0.052),
            (0.205, 0.165, k * 0.015 - 0.04),
            (0.225, 0.235, -0.085),
        ]
        ul["bone"].add(*make_tube_curve(pts, [0.019, 0.013, 0.007, 0.002], segs=9))
    # Correas de hierro sujetando el cráneo
    ul["iron"].add(*make_curved_shell(-0.045, 0.035, 0.088, 0.085, 0.095, 0.092, 0.4, 2.7, thickness=0.010))

    # Manto asimétrico unilateral en Torso que cae desde el hombro izquierdo hacia la espalda
    tr = out["Torso"]
    tr["cloth"].add(*make_curved_shell(
        -0.28, 0.28, 0.185, 0.148, 0.225, 0.165,
        math.pi * 0.58, math.pi * 1.40, thickness=0.009,
        ny=12, na=16, tatter_waves=5, tatter_amp=0.065, flare_bot=0.065
    ))
    tr["crimson"].add(*make_curved_shell(
        -0.29, -0.22, 0.246, 0.208, 0.250, 0.212,
        math.pi * 0.60, math.pi * 1.38, thickness=0.010,
        tatter_waves=5, tatter_amp=0.060
    ))

    # DERECHA: SIN hombrera pesada — arnés liviano de cuero + 2 vainas de cuchillos
    ur = out["UpperArm_R"]
    ur["leather"].add(*make_curved_shell(-0.055, 0.010, 0.082, 0.080, 0.085, 0.082, -2.8, -0.35, thickness=0.009))
    for i in range(2):
        oz = -0.025 + i * 0.045
        ur["leather"].add(*make_box(0.018, 0.115, 0.026, center=(-0.086, -0.055, oz), rot=(0.15, 0, -0.22), bevel=0.003))
        ur["bone"].add(*make_cylinder(0.006, 0.008, 0.048, segs=6, center=(-0.096, 0.018, oz), rot=(0.15, 0, -0.22)))
    return out


def build_rb_torso() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _parts_dict()}
    t = out["Torso"]
    # Abrigo carbón abierto al frente (sin peto cerrado en V frontal) + solapas carmesí
    t["cloth"].add(*make_curved_shell(
        -0.17, 0.29, 0.168, 0.130, 0.218, 0.156,
        0.34, math.tau - 0.34, thickness=0.014, ny=12, na=22, flare_top=0.010
    ))
    for sgn in (-1, 1):
        lapel = [(0.0, -0.15), (sgn * 0.058, 0.06), (sgn * 0.092, 0.26), (sgn * 0.025, 0.28)]
        t["crimson"].add(*make_extruded_polygon(lapel, 0.012, bevel=0.003,
                                               center=(sgn * 0.062, 0.04, 0.155),
                                               rot=(0.08, sgn * -0.32, 0)))
    # Cuello alto acampanado y rasgado alrededor de nuca y sienes
    t["cloth"].add(*make_curved_shell(
        0.24, 0.385, 0.105, 0.102, 0.145, 0.135,
        0.72, math.tau - 0.72, thickness=0.011, tatter_waves=4, tatter_amp=0.022, flare_top=0.025
    ))
    # Bandolera diagonal de cuero con 3 cuchillos arrojadizos
    strap_pts = [(-0.155, 0.235, 0.148), (-0.040, 0.095, 0.172), (0.060, -0.025, 0.164), (0.145, -0.125, 0.138)]
    t["leather"].add(*make_tube_curve(strap_pts, [0.022, 0.022, 0.022, 0.022], segs=8))
    for idx_k in range(3):
        frac = 0.22 + idx_k * 0.26
        px = -0.155 + (0.145 + 0.155) * frac
        py = 0.235 + (-0.125 - 0.235) * frac
        pz = 0.168
        t["iron"].add(*make_box(0.014, 0.075, 0.008, center=(px, py - 0.018, pz + 0.012), rot=(0.15, 0, -0.58)))
        t["bone"].add(*make_cylinder(0.007, 0.006, 0.048, segs=6, center=(px - 0.022, py + 0.025, pz + 0.018), rot=(0.15, 0, -0.58)))
    # 4 costillas óseas asimétricas SOLO en el costado izquierdo
    for r in range(4):
        ry = 0.175 - r * 0.052
        rib_pts = [
            (0.048, ry - 0.014, 0.168),
            (0.135, ry + 0.005, 0.145),
            (0.195, ry + 0.018, 0.045),
            (0.168, ry + 0.022, -0.095),
        ]
        t["bone"].add(*make_tube_curve(rib_pts, [0.009, 0.011, 0.010, 0.006], segs=7))
    return out


def build_rb_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {
        "Forearm_R": _parts_dict(), "Wrist_R": _parts_dict(),
        "Forearm_L": _parts_dict(), "Wrist_L": _parts_dict(),
    }
    # DERECHA: Vendas en espiral + garfio en codo + 3 garras segadoras sobre el dorso de la mano
    fr, wr = out["Forearm_R"], out["Wrist_R"]
    for b in range(6):
        y0 = -0.255 + b * 0.038
        rad_b = 0.048 + b * 0.0028
        fr["cloth"].add(*make_curved_shell(y0, y0 + 0.032, rad_b, rad_b, rad_b + 0.002, rad_b + 0.002,
                                           -math.pi, math.pi, thickness=0.006))
    fr["iron"].add(*make_tube_curve([(0.0, -0.02, -0.058), (-0.01, 0.045, -0.092), (0.0, 0.095, -0.072)],
                                    [0.012, 0.009, 0.002], segs=7))
    wr["iron"].add(*make_box(0.074, 0.068, 0.014, center=(-0.024, -0.032, 0.0), bevel=0.003))
    for c in (-0.024, 0.0, 0.024):
        pts = [(-0.030, -0.045, c), (-0.038, -0.105, c * 1.15), (-0.028, -0.185, c * 1.10)]
        wr["iron"].add(*make_tube_curve(pts, [0.006, 0.0045, 0.0012], segs=6))

    # IZQUIERDA: Brazal de tablillas óseas atado con tiras de cuero + cinta carmesí
    fl, wl = out["Forearm_L"], out["Wrist_L"]
    for sp in (-1, 0, 1):
        fl["bone"].add(*make_box(0.020, 0.195, 0.012, center=(0.056, -0.145, sp * 0.022),
                                 rot=(0, sp * 0.35, 0), bevel=0.003))
    for st in (-0.085, -0.195):
        fl["leather"].add(*make_curved_shell(st, st + 0.020, 0.058, 0.058, 0.058, 0.058, -math.pi, math.pi, thickness=0.007))
    wl["crimson"].add(*make_box(0.018, 0.165, 0.005, center=(0.045, -0.11, -0.025), rot=(0.15, 0, 0.18)))
    return out


def build_rb_belt() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Pelvis": _parts_dict()}
    p = out["Pelvis"]
    # Triple fajín carmesí + cincha cuero (Pelvis local y ~ -0.06..0.06)
    p["crimson"].add(*make_curved_shell(-0.07, 0.05, 0.172, 0.136, 0.176, 0.140, -math.pi, math.pi, thickness=0.012))
    p["leather"].add(*make_curved_shell(-0.03, 0.02, 0.182, 0.144, 0.182, 0.144, -math.pi, math.pi, thickness=0.010))
    # Hebilla-cráneo asimétrica en cadera izquierda
    add_sculpted_skull(p, scale=0.68, center=(0.135, 0.00, 0.135), rot=(0, 0.72, 0), with_jaw=False)
    # Faldones largos partidos de gabardina (duster coat tails) con forro carmesí
    for a0, a1 in ((0.42, 1.85), (math.pi * 0.62, math.pi * 1.38), (math.tau - 1.85, math.tau - 0.42)):
        p["cloth"].add(*make_curved_shell(-0.62, -0.04, 0.245, 0.215, 0.180, 0.142,
                                          a0, a1, thickness=0.010, ny=12, na=14,
                                          tatter_waves=4, tatter_amp=0.065, flare_bot=0.055))
        p["crimson"].add(*make_curved_shell(-0.60, -0.06, 0.236, 0.206, 0.173, 0.135,
                                            a0 + 0.05, a1 - 0.05, thickness=0.006,
                                            tatter_waves=4, tatter_amp=0.055, flare_bot=0.050))
    return out


def build_rb_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _parts_dict()
        # Placas óseas colgando de cadenas en cada cadera
        for row in range(2):
            yc = -0.065 - row * 0.092
            p["bone"].add(*make_box(0.024, 0.076, 0.088, center=(sgn * 0.105, yc, 0.010),
                                    rot=(0, 0, sgn * -0.08), bevel=0.004))
            p["iron"].add(*make_cylinder(0.004, 0.004, 0.055, segs=6, center=(sgn * 0.106, yc + 0.052, 0.032)))
            p["iron"].add(*make_cylinder(0.004, 0.004, 0.055, segs=6, center=(sgn * 0.106, yc + 0.052, -0.012)))
        # Plumas metálicas de cuervo colgando
        for f in range(3):
            p["iron"].add(*make_box(0.006, 0.085, 0.018, center=(sgn * 0.114, -0.235, -0.028 + f * 0.028),
                                    rot=(0.1 * (f - 1), 0, sgn * -0.14)))
        out[side] = p
    return out


def build_rb_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _parts_dict()
        # Vendas carbón mate y correas alrededor del muslo
        p["cloth"].add(*make_curved_shell(-0.38, -0.08, 0.088, 0.088, 0.105, 0.105, -math.pi, math.pi, thickness=0.008))
        for yband in (-0.14, -0.28):
            p["leather"].add(*make_curved_shell(yband, yband + 0.024, 0.104, 0.104, 0.104, 0.104, -math.pi, math.pi, thickness=0.009))
        if sgn < 0:
            # Muslo derecho: bolsa de caza + viales de ascua
            p["leather"].add(*make_box(0.044, 0.095, 0.078, center=(-0.112, -0.21, 0.01), bevel=0.006))
            for k in (-1, 1):
                p["ember"].add(*make_cylinder(0.009, 0.009, 0.048, segs=8, center=(-0.128, -0.20, k * 0.020)))
        else:
            # Muslo izquierdo: 2 tablillas de hueso atadas
            for oz in (-0.028, 0.028):
                p["bone"].add(*make_box(0.018, 0.195, 0.032, center=(0.108, -0.22, oz), bevel=0.004))
        out[side] = p
    return out


def build_rb_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _parts_dict(), "Knee_R": _parts_dict()}
    # Rodilla izquierda: cráneo cornudo con ojos ascua
    kl = out["Knee_L"]
    kl["leather"].add(*make_curved_shell(-0.045, 0.045, 0.082, 0.082, 0.082, 0.082, -math.pi, math.pi, thickness=0.008))
    add_sculpted_skull(kl, scale=0.88, center=(0.0, 0.005, 0.078), rot=(0.08, 0.12, 0), with_jaw=False)
    for sg in (-1, 1):
        kl["bone"].add(*make_tube_curve([(sg * 0.042, 0.025, 0.082), (sg * 0.075, 0.068, 0.062), (sg * 0.085, 0.105, 0.020)],
                                        [0.011, 0.007, 0.002], segs=7))
    # Rodilla derecha: garfio forjado curvo de segador
    kr = out["Knee_R"]
    kr["leather"].add(*make_curved_shell(-0.045, 0.045, 0.082, 0.082, 0.082, 0.082, -math.pi, math.pi, thickness=0.008))
    kr["iron"].add(*make_curved_shell(-0.055, 0.055, 0.084, 0.088, 0.084, 0.088, -1.25, 1.25, thickness=0.010))
    kr["iron"].add(*make_tube_curve([(0, -0.01, 0.090), (0, 0.025, 0.135), (-0.015, 0.085, 0.115)],
                                    [0.014, 0.009, 0.002], segs=7))
    return out


def build_rb_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side in ("L", "R"):
        sh, ak = _parts_dict(), _parts_dict()
        # Caña alta de cuero con vuelta superior doblada y ribete carmesí
        sh["leather"].add(*make_curved_shell(-0.44, -0.05, 0.068, 0.072, 0.088, 0.092, -math.pi, math.pi, thickness=0.010))
        sh["crimson"].add(*make_curved_shell(-0.14, -0.04, 0.094, 0.098, 0.102, 0.106, -math.pi, math.pi, thickness=0.009, flare_top=0.008))
        for ystr in (-0.19, -0.29, -0.39):
            sh["iron"].add(*make_curved_shell(ystr, ystr + 0.022, 0.084, 0.088, 0.084, 0.088, -math.pi, math.pi, thickness=0.009))
        # Pie con suela dentada, puntera de hierro y espolón óseo trasero
        ak["leather"].add(*make_box(0.096, 0.068, 0.225, center=(0, -0.022, 0.052), bevel=0.012))
        ak["iron"].add(*make_box(0.104, 0.022, 0.238, center=(0, -0.056, 0.052), bevel=0.004))
        ak["iron"].add(*make_sphere(0.050, segs=10, rings=8, center=(0, -0.024, 0.148), scale=(0.95, 0.65, 1.05)))
        ak["bone"].add(*make_cylinder(0.002, 0.014, 0.065, segs=6, center=(0, -0.032, -0.082), rot=(-1.35, 0, 0)))
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_rb_weapon() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_R": _parts_dict()}
    w = out["Wrist_R"]
    gy = -0.035
    # Asta larga torcida de hierro (orientada en +Z en Wrist_R local como Dark Crystal)
    shaft_pts = [(0.0, gy, -0.44), (-0.010, gy + 0.008, 0.0), (0.014, gy - 0.006, 0.56), (-0.008, gy + 0.008, 1.08), (0.008, gy, 1.32)]
    w["iron"].add(*make_tube_curve(shaft_pts, [0.014, 0.015, 0.015, 0.015, 0.015], segs=10))
    w["leather"].add(*make_cylinder(0.018, 0.018, 0.24, segs=12, center=(0, gy, 0.01), rot=(math.pi * 0.5, 0, 0)))
    w["bone"].add(*make_cylinder(0.002, 0.022, 0.11, segs=6, center=(0, gy, -0.48), rot=(-math.pi * 0.5, 0, 0)))
    # Cráneo de cuervo en el nacimiento de la hoja
    add_sculpted_skull(w, scale=0.72, center=(0.024, gy, 1.28), rot=(0, math.pi * 0.45, 0), with_jaw=False)
    # Hoja creciente monumental de guadaña + filo ascua
    blade_pts = [(0.01, 1.24), (0.24, 1.44), (0.56, 1.38), (0.62, 0.90), (0.50, 1.16), (0.30, 1.26), (0.02, 1.18)]
    w["iron"].add(*make_extruded_polygon(blade_pts, 0.012, bevel=0.004, center=(0.0, gy, 0.0), rot=(-math.pi * 0.5, 0, 0)))
    rune_pts = [(0.06, gy, 1.21), (0.26, gy, 1.27), (0.45, gy, 1.18), (0.57, gy, 0.96)]
    w["ember"].add(*make_tube_curve(rune_pts, [0.0045, 0.0045, 0.004, 0.002], segs=6))
    # Cintas carmesí colgantes
    w["crimson"].add(*make_box(0.022, 0.005, 0.28, center=(0.045, gy - 0.015, 1.02), rot=(0.15, 0, 0.08)))
    return out


def build_rb_shield() -> Dict[str, Dict[str, MeshPart]]:
    """Farol Abisal & Garra Ósea montado en Wrist_L."""
    out = {"Wrist_L": _parts_dict()}
    s = out["Wrist_L"]
    gy = -0.035
    # Empuñadura + garra ósea sobre los nudillos (+X)
    s["leather"].add(*make_cylinder(0.016, 0.016, 0.24, segs=12, center=(0, gy, 0.02), rot=(math.pi * 0.5, 0, 0)))
    s["bone"].add(*make_cylinder(0.002, 0.020, 0.085, segs=6, center=(0, gy, -0.14), rot=(-math.pi * 0.5, 0, 0)))
    s["bone"].add(*make_tube_curve([(0.016, gy - 0.008, -0.075), (0.044, gy - 0.018, 0.015), (0.038, gy - 0.005, 0.115), (0.012, gy + 0.010, 0.175)],
                                   [0.010, 0.012, 0.010, 0.005], segs=7))
    # Brazo curvo de hierro + farol colgante hacia -Y (hacia el suelo)
    s["iron"].add(*make_tube_curve([(0, gy, 0.12), (0, gy - 0.01, 0.24), (0, gy + 0.028, 0.32), (0, gy + 0.072, 0.30)],
                                   [0.010, 0.010, 0.010, 0.008], segs=8))
    lx, ly, lz = 0.0, gy - 0.10, 0.30
    s["iron"].add(*make_cylinder(0.006, 0.058, 0.055, segs=6, center=(lx, ly + 0.072, lz)))
    s["iron"].add(*make_cylinder(0.048, 0.004, 0.085, segs=6, center=(lx, ly - 0.088, lz)))
    for b in range(6):
        ang = (b / 6) * math.tau
        s["iron"].add(*make_cylinder(0.004, 0.004, 0.125, segs=6,
                                     center=(lx + math.cos(ang) * 0.046, ly, lz + math.sin(ang) * 0.046)))
    s["ember"].add(*make_sphere(0.028, segs=8, rings=6, center=(lx, ly, lz), scale=(0.85, 1.45, 0.85)))
    return out


# ==============================================================================
# 4. EMPAQUETADOR GLB BINARIO AUTOCONTENIDO
# ==============================================================================
def write_glb(out_path: str, bone_meshes: Dict[str, Dict[str, MeshPart]],
              textures: Dict[str, bytes]) -> dict:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nodes = [dict(n) for n in SKELETON_TEMPLATE]
    name_to_node = {n["name"]: i for i, n in enumerate(nodes)}

    bin_buf = bytearray()
    buffer_views, accessors, meshes = [], [], []

    def append_bytes(raw: bytes, target: Optional[int] = None) -> int:
        while len(bin_buf) % 4 != 0:
            bin_buf.append(0)
        offset = len(bin_buf)
        bin_buf.extend(raw)
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            bv["target"] = target
        buffer_views.append(bv)
        return len(buffer_views) - 1

    tex_keys = ("cloth_albedo", "bone_albedo", "iron_albedo", "crimson_albedo")
    for k in tex_keys:
        append_bytes(textures[k])
    images = [{"bufferView": i, "mimeType": "image/png", "name": k} for i, k in enumerate(tex_keys)]
    samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    gltf_textures = [{"sampler": 0, "source": i} for i in range(4)]

    materials = [
        {"name": "Ravenbone_Cloth", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.04, "roughnessFactor": 0.90}, "doubleSided": True},
        {"name": "Ravenbone_Bone", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.05, "roughnessFactor": 0.56}, "doubleSided": True},
        {"name": "Ravenbone_Iron", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}, "metallicFactor": 0.84, "roughnessFactor": 0.42}, "doubleSided": True},
        {"name": "Ravenbone_Crimson", "pbrMetallicRoughness": {"baseColorTexture": {"index": 3}, "metallicFactor": 0.06, "roughnessFactor": 0.84}, "doubleSided": True},
        {"name": "Ravenbone_Leather", "pbrMetallicRoughness": {"baseColorFactor": [0.10, 0.07, 0.055, 1.0], "metallicFactor": 0.12, "roughnessFactor": 0.68}, "doubleSided": True},
        {"name": "Ravenbone_Ember", "pbrMetallicRoughness": {"baseColorFactor": [0.98, 0.57, 0.24, 1.0], "metallicFactor": 0.10, "roughnessFactor": 0.14}, "emissiveFactor": [0.95, 0.40, 0.08], "doubleSided": True},
    ]
    mat_idx_map = {"cloth": 0, "bone": 1, "iron": 2, "crimson": 3, "leather": 4, "ember": 5}

    total_verts, total_tris = 0, 0
    for bone_name, parts in bone_meshes.items():
        parent_idx = name_to_node[bone_name]
        for mat_key in ("cloth", "bone", "iron", "crimson", "leather", "ember"):
            part = parts.get(mat_key)
            if part is None or part.is_empty():
                continue
            pos, nrm, uv, idx = part.build()
            total_verts += len(pos)
            total_tris += len(idx) // 3

            pos_bv = append_bytes(pos.astype("<f4").tobytes(), target=34962)
            nrm_bv = append_bytes(nrm.astype("<f4").tobytes(), target=34962)
            uv_bv = append_bytes(uv.astype("<f4").tobytes(), target=34962)
            idx_u32 = idx.astype("<u4")
            idx_bv = append_bytes(idx_u32.tobytes(), target=34963)

            pos_acc = len(accessors)
            accessors.append({
                "bufferView": pos_bv, "componentType": 5126, "count": len(pos), "type": "VEC3",
                "min": [float(x) for x in pos.min(axis=0)],
                "max": [float(x) for x in pos.max(axis=0)],
            })
            nrm_acc = len(accessors)
            accessors.append({"bufferView": nrm_bv, "componentType": 5126, "count": len(nrm), "type": "VEC3"})
            uv_acc = len(accessors)
            accessors.append({"bufferView": uv_bv, "componentType": 5126, "count": len(uv), "type": "VEC2"})
            idx_acc = len(accessors)
            accessors.append({"bufferView": idx_bv, "componentType": 5125, "count": len(idx_u32), "type": "SCALAR"})

            mesh_idx = len(meshes)
            meshes.append({
                "name": f"{bone_name}_{mat_key}",
                "primitives": [{
                    "attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc, "TEXCOORD_0": uv_acc},
                    "indices": idx_acc, "material": mat_idx_map[mat_key],
                }],
            })
            node_idx = len(nodes)
            nodes.append({"name": f"{bone_name}_Ravenbone_{mat_key}", "mesh": mesh_idx})
            nodes[parent_idx].setdefault("children", []).append(node_idx)

    while len(bin_buf) % 4 != 0:
        bin_buf.append(0)

    gltf = {
        "asset": {"version": "2.0", "generator": "Armaduras-glb/crear_ravenbone.py"},
        "scene": 0, "scenes": [{"name": "RavenboneReaper", "nodes": [0]}],
        "nodes": nodes, "meshes": meshes, "materials": materials,
        "textures": gltf_textures, "images": images, "samplers": samplers,
        "accessors": accessors, "bufferViews": buffer_views, "buffers": [{"byteLength": len(bin_buf)}],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_len))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes)
        f.write(struct.pack("<II", len(bin_buf), 0x004E4942) + bytes(bin_buf))
    return {"path": out_path, "bytes": total_len, "verts": total_verts, "tris": total_tris}


# ==============================================================================
# 5. ACTUALIZACIÓN DE CATÁLOGOS
# ==============================================================================
RAVENBONE_CATALOG_ENTRIES = {
    "underlayer": {"idx": 17, "name": "Lino Oscuro & Vendas de Cuervo (Asimétrico)"},
    "helmet":     {"idx": 10, "name": "Capucha Profunda & Máscara Ósea Cornuda (Segador)"},
    "shoulders":  {"idx": 13, "name": "Cráneo Cornudo Izquierdo & Manto de Cuervo (Asimétrico)"},
    "torso":      {"idx": 21, "name": "Abrigo Abierto Carmesí, Bandolera y Costillas Óseas (Segador)"},
    "gloves":     {"idx": 7,  "name": "Garras Forjadas Derechas & Vendas/Hueso Izq. (Asimétrico)"},
    "belt":       {"idx": 9,  "name": "Fajín Triple Carmesí & Gabardina Partida (Segador)"},
    "tassets":    {"idx": 10, "name": "Colgantes Óseos con Cadenas y Plumas (Segador)"},
    "cuisses":    {"idx": 9,  "name": "Vendas de Muslo, Alquimia Der. y Tablillas Izq. (Asimétrico)"},
    "knees":      {"idx": 9,  "name": "Cráneo Cornudo Izquierdo & Garfio de Hierro Der. (Asimétrico)"},
    "boots":      {"idx": 14, "name": "Botas Altas de Cazador con Vuelta Carmesí y Espolón Óseo"},
    "weapon":     {"idx": 82, "name": "Guadaña Segadora de Cuervo y Hueso", "familia": "hacha", "nivel": 40},
    "shield":     {"idx": 12, "name": "Farol Abisal & Garra Ósea (Mano Izquierda)", "nivel": 40},
}


def update_catalogo_estudio() -> None:
    with open(CATALOGO_ESTUDIO_PATH, encoding="utf-8") as f:
        estudio = json.load(f)
    for ranura, entry in RAVENBONE_CATALOG_ENTRIES.items():
        lst = estudio.setdefault(ranura, [])
        existing = next((it for it in lst if it["idx"] == entry["idx"]), None)
        if existing is not None:
            existing.update(entry)
        else:
            lst.append(dict(entry))
        lst.sort(key=lambda x: x["idx"])
    with open(CATALOGO_ESTUDIO_PATH, "w", encoding="utf-8") as f:
        json.dump(estudio, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    textures = generate_ravenbone_textures()
    builders = {
        ("helmet", 10): build_rb_helmet,
        ("shoulders", 13): build_rb_shoulders,
        ("torso", 21): build_rb_torso,
        ("gloves", 7): build_rb_gloves,
        ("belt", 9): build_rb_belt,
        ("tassets", 10): build_rb_tassets,
        ("cuisses", 9): build_rb_cuisses,
        ("knees", 9): build_rb_knees,
        ("boots", 14): build_rb_boots,
        ("weapon", 82): build_rb_weapon,
        ("shield", 12): build_rb_shield,
    }
    print("Generando piezas Segador de Cuervo & Hueso (.glb)...")
    for (slot, idx), fn in builders.items():
        out_path = os.path.join(PIEZAS_DIR, slot, f"{idx}.glb")
        stats = write_glb(out_path, fn(), textures)
        print(f"  [OK] {slot}/{idx}.glb -> {stats['verts']} verts, {stats['tris']} tris, {stats['bytes']} bytes")

    update_catalogo_estudio()
    print(f"Actualizado {os.path.relpath(CATALOGO_ESTUDIO_PATH, ROOT_DIR)}")
    subprocess.run([sys.executable, GENERAR_CATALOGO_PATH], check=True)


if __name__ == "__main__":
    main()
