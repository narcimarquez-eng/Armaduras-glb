#!/usr/bin/env python3
"""Generador procedural en Python de la armadura Dark Crystal (Lineage II - Grado A).

Genera:
  - Modelos .glb compatibles con el esqueleto `CreadorHumano_hombre` y Three.js / Blender
    para cada ranura física: helmet/9, shoulders/12, torso/20, gloves/6, belt/8,
    tassets/9, cuisses/8, knees/8, boots/13, weapon/81, shield/11.
  - Texturas PBR embebidas (albedo Dark Crystal índigo/oro, roughness/metallic, emisivo cian).
  - Actualización idempotente de `piezas/catalogo-estudio.json` en todas las ranuras
    (incluyendo `underlayer` 16 y la nueva ranura `shield` 11).
  - Regeneración de `piezas/catalogo.json` mediante `tools/piezas/generar-catalogo.py`.

Uso:
    python3 tools/piezas/crear_dark_crystal.py
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

# Esqueleto base exacto observado en piezas/*.glb (CreadorHumano_hombre)
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
# 1. TEXTURAS PBR PROCEDURALES DARK CRYSTAL (PIL -> PNG EN MEMORIA)
# ==============================================================================
def _img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def generate_dark_crystal_textures() -> Dict[str, bytes]:
    """Genera las texturas PBR características de la armadura Dark Crystal (L2)."""
    size = 256

    # A) Placa principal: Índigo oscuro facetado con filigrana dorada y vetas cian
    albedo = Image.new("RGB", (size, size), (28, 36, 62))
    draw = ImageDraw.Draw(albedo)
    mr = Image.new("RGB", (size, size), (0, 62, 212))  # G=roughness(~0.24), B=metalness(~0.83)
    mr_draw = ImageDraw.Draw(mr)
    emissive = Image.new("RGB", (size, size), (0, 0, 0))
    em_draw = ImageDraw.Draw(emissive)

    # Trama metálica cepillada sutil
    rng = np.random.default_rng(42)
    arr = np.array(albedo, dtype=np.int16)
    noise = rng.integers(-6, 7, size=(size, size, 1))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    albedo = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(albedo)

    step = 64
    for gy in range(0, size + step, step):
        for gx in range(0, size + step, step):
            # Rombo facetado oscuro superior/inferior para dar efecto cristalino
            poly_top = [(gx, gy - 26), (gx + 26, gy), (gx, gy), (gx - 26, gy)]
            poly_bot = [(gx, gy + 26), (gx + 26, gy), (gx, gy), (gx - 26, gy)]
            draw.polygon(poly_top, fill=(44, 58, 98))
            draw.polygon(poly_bot, fill=(22, 29, 50))
            draw.line([(gx, gy - 26), (gx + 26, gy), (gx, gy + 26), (gx - 26, gy), (gx, gy - 26)],
                      fill=(212, 169, 78), width=2)
            # Costura luminosa cian interior
            draw.line([(gx, gy - 17), (gx + 17, gy), (gx, gy + 17), (gx - 17, gy), (gx, gy - 17)],
                      fill=(56, 189, 248), width=1)
            em_draw.line([(gx, gy - 17), (gx + 17, gy), (gx, gy + 17), (gx - 17, gy), (gx, gy - 17)],
                         fill=(14, 116, 178), width=1)
            mr_draw.polygon([(gx, gy - 26), (gx + 26, gy), (gx, gy + 26), (gx - 26, gy)],
                            outline=(0, 48, 235))

    # B) Oro antiguo Dark Crystal (para ribetes, garras, coronas y marcos)
    gold = Image.new("RGB", (128, 128), (218, 175, 84))
    g_arr = np.array(gold, dtype=np.int16)
    streak = rng.integers(-8, 9, size=(128, 1, 1))
    g_arr = np.clip(g_arr + streak, 0, 255).astype(np.uint8)
    gold = Image.fromarray(g_arr, "RGB")
    g_draw = ImageDraw.Draw(gold)
    for y in range(0, 128, 16):
        g_draw.line([(0, y), (128, y)], fill=(180, 136, 52), width=1)

    return {
        "plate_albedo": _img_to_png_bytes(albedo),
        "plate_mr": _img_to_png_bytes(mr),
        "plate_emissive": _img_to_png_bytes(emissive),
        "gold_albedo": _img_to_png_bytes(gold),
    }


# ==============================================================================
# 2. UTILIDADES DE GEOMETRÍA 3D (NUMPY -> TRIÁNGULOS CON NORMALES Y UVs)
# ==============================================================================
class MeshPart:
    """Acumulador de geometría indexada con posición, normal y UV."""

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
    v0 = pos[tris[:, 0]]
    v1 = pos[tris[:, 1]]
    v2 = pos[tris[:, 2]]
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
        y, z = p[:, 1] * c - p[:, 2] * s, p[:, 1] * s + p[:, 2] * c
        p[:, 1], p[:, 2] = y, z
    if abs(ry) > 1e-6:
        c, s = math.cos(ry), math.sin(ry)
        x, z = p[:, 0] * c + p[:, 2] * s, -p[:, 0] * s + p[:, 2] * c
        p[:, 0], p[:, 2] = x, z
    if abs(rz) > 1e-6:
        c, s = math.cos(rz), math.sin(rz)
        x, y = p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c
        p[:, 0], p[:, 1] = x, y
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
    # Prisma biselado octogonal en X/Y con profundidad Z
    b = min(bevel, hx * 0.45, hy * 0.45)
    pts = [
        (-hx + b, -hy), (hx - b, -hy), (hx, -hy + b), (hx, hy - b),
        (hx - b, hy), (-hx + b, hy), (-hx, hy - b), (-hx, -hy + b),
    ]
    return make_extruded_polygon(pts, sz, bevel=min(b, sz * 0.3), center=center, rot=rot)


def make_extruded_polygon(pts2d: List[Tuple[float, float]], depth: float,
                          bevel: float = 0.0, center=(0, 0, 0),
                          rot=(0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """Extruye un polígono 2D (asumido CCW y convexo/estrella respecto a su centroide) con bisel opcional."""
    n = len(pts2d)
    poly = np.array(pts2d, dtype=np.float32)
    cx, cy = poly.mean(axis=0)
    hz = depth * 0.5
    b = min(bevel, hz * 0.45)

    rings: List[Tuple[np.ndarray, float]] = []
    if b > 1e-5:
        shrink = 0.88
        inner = (poly - [cx, cy]) * shrink + [cx, cy]
        rings.append((inner, -hz - b * 0.6))
        rings.append((poly, -hz + b * 0.4))
        rings.append((poly, hz - b * 0.4))
        rings.append((inner, hz + b * 0.6))
    else:
        rings.append((poly, -hz))
        rings.append((poly, hz))

    verts = []
    tris = []
    # Anillos laterales
    ring_starts = []
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

    # Tapa trasera (z-)
    back_c = len(verts)
    verts.append([float(cx), float(cy), float(rings[0][1])])
    bs = ring_starts[0]
    for i in range(n):
        ni = (i + 1) % n
        tris.extend([back_c, bs + ni, bs + i])

    # Tapa frontal (z+)
    front_c = len(verts)
    verts.append([float(cx), float(cy), float(rings[-1][1])])
    fs = ring_starts[-1]
    for i in range(n):
        ni = (i + 1) % n
        tris.extend([front_c, fs + i, fs + ni])

    pos = transform_mesh(np.array(verts, dtype=np.float32), center, rot)
    return pos, np.array(tris, dtype=np.uint32)


def make_crystal_gem(radius: float, height: float, sides: int = 6,
                     center=(0, 0, 0), rot=(0, 0, 0),
                     double_pointed: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Genera un cristal Dark Crystal facetado tipo obelisco/octaedro."""
    verts = []
    tris = []
    half_h = height * 0.5
    crown_h = half_h * 0.42
    for i in range(sides):
        ang = (i / sides) * math.tau
        verts.append([math.cos(ang) * radius, crown_h, math.sin(ang) * radius])
    for i in range(sides):
        ang = (i / sides) * math.tau
        verts.append([math.cos(ang) * radius * 0.92, -crown_h if double_pointed else -half_h, math.sin(ang) * radius * 0.92])
    top_idx = len(verts)
    verts.append([0.0, half_h, 0.0])
    bot_idx = len(verts)
    verts.append([0.0, -half_h if double_pointed else -half_h, 0.0])

    for i in range(sides):
        ni = (i + 1) % sides
        tris.extend([top_idx, ni, i])
        tris.extend([i, ni, sides + ni, i, sides + ni, sides + i])
        tris.extend([bot_idx, sides + i, sides + ni])

    pos = transform_mesh(np.array(verts, dtype=np.float32), center, rot)
    return pos, np.array(tris, dtype=np.uint32)


def make_curved_armor_shell(
    y0: float, y1: float,
    rx0: float, rz0: float,
    rx1: float, rz1: float,
    ang_min: float, ang_max: float,
    thickness: float = 0.014,
    ny: int = 10, na: int = 18,
    swallow_tail: float = 0.0,
    front_ridge: float = 0.0,
    flutes: int = 0,
    flare_top: float = 0.0,
    flare_bot: float = 0.0,
    center=(0, 0, 0),
    rot=(0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera una coraza o greba anatómica sólida con grosor real, cola de golondrina, cresta y acanaladuras."""
    ys = np.linspace(0.0, 1.0, ny)
    angs = np.linspace(ang_min, ang_max, na)

    def eval_ring(v_norm: float, theta: float, out_offset: float) -> Tuple[float, float, float]:
        # Cola de golondrina en el borde inferior (pico central hacia abajo)
        front_bias = math.cos(theta)
        y_mod = y0 + (y1 - y0) * v_norm
        if swallow_tail != 0.0:
            edge_drop = swallow_tail * max(0.0, front_bias) ** 1.6 * (1.0 - v_norm)
            y_mod -= edge_drop

        rx = rx0 + (rx1 - rx0) * v_norm
        rz = rz0 + (rz1 - rz0) * v_norm
        flare = flare_top * (v_norm ** 2) + flare_bot * ((1.0 - v_norm) ** 2)
        rx += flare
        rz += flare

        ridge = front_ridge * max(0.0, front_bias) ** 3.0 * math.sin(v_norm * math.pi * 0.9 + 0.15)
        flute_w = 0.0
        if flutes > 0 and abs(front_bias) > 0.25:
            flute_w = 0.0032 * math.cos(theta * flutes * 2.0) * math.sin(v_norm * math.pi)

        r_x = rx + out_offset + flute_w
        r_z = rz + out_offset + ridge + flute_w
        x = math.sin(theta) * r_x
        z = math.cos(theta) * r_z
        return x, y_mod, z

    verts = []
    # 0..ny*na-1: capa exterior; ny*na..2*ny*na-1: capa interior
    for layer_off in (thickness * 0.5, -thickness * 0.5):
        for v in ys:
            for th in angs:
                verts.append(eval_ring(float(v), float(th), layer_off))

    tris = []
    stride = na
    inner_base = ny * na

    # Superficie exterior & interior
    for j in range(ny - 1):
        for i in range(na - 1):
            a = j * stride + i
            b = a + 1
            c = (j + 1) * stride + i + 1
            d = (j + 1) * stride + i
            tris.extend([a, b, c, a, c, d])

            ia, ib, ic, id_ = inner_base + a, inner_base + b, inner_base + c, inner_base + d
            tris.extend([ia, ic, ib, ia, id_, ic])

    # Tapas superior e inferior
    for i in range(na - 1):
        # Inferior (j=0)
        o0, o1 = i, i + 1
        i0, i1 = inner_base + i, inner_base + i + 1
        tris.extend([o0, i0, i1, o0, i1, o1])
        # Superior (j=ny-1)
        row = (ny - 1) * stride
        to0, to1 = row + i, row + i + 1
        ti0, ti1 = inner_base + to0, inner_base + to1
        tris.extend([to0, to1, ti1, to0, ti1, ti0])

    # Tapas laterales (i=0 y i=na-1)
    for j in range(ny - 1):
        o0 = j * stride
        o1 = (j + 1) * stride
        i0 = inner_base + o0
        i1 = inner_base + o1
        tris.extend([o0, o1, i1, o0, i1, i0])

        ro0 = j * stride + (na - 1)
        ro1 = (j + 1) * stride + (na - 1)
        ri0 = inner_base + ro0
        ri1 = inner_base + ro1
        tris.extend([ro0, ri0, ri1, ro0, ri1, ro1])

    pos = transform_mesh(np.array(verts, dtype=np.float32), center, rot)
    return pos, np.array(tris, dtype=np.uint32)


def make_cylinder(r_top: float, r_bot: float, height: float, segs: int = 14,
                  center=(0, 0, 0), rot=(0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    tris = []
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
    pos = transform_mesh(np.array(verts, dtype=np.float32), center, rot)
    return pos, np.array(tris, dtype=np.uint32)


# ==============================================================================
# 3. CONSTRUCTORES POR RANURA DARK CRYSTAL (LINEAGE II)
# ==============================================================================
# Cada constructor devuelve dict: bone_name -> dict(material_key -> MeshPart)
# material_key in ('plate', 'trim', 'crystal', 'glow')

def _parts_dict() -> Dict[str, MeshPart]:
    return {"plate": MeshPart(), "trim": MeshPart(), "crystal": MeshPart(), "glow": MeshPart()}


def build_dc_helmet() -> Dict[str, Dict[str, MeshPart]]:
    b = {"Head": _parts_dict()}
    h = b["Head"]
    # Banda diadema trasera + sienes (Head scale=0.9 en esqueleto, coords locales)
    h["plate"].add(*make_curved_armor_shell(
        0.135, 0.215, 0.118, 0.128, 0.112, 0.122,
        -math.pi * 0.85, math.pi * 0.85, thickness=0.012, swallow_tail=0.025, front_ridge=0.015
    ))
    # Ribete dorado inferior de la diadema
    h["trim"].add(*make_curved_armor_shell(
        0.128, 0.148, 0.121, 0.132, 0.121, 0.132,
        -math.pi * 0.82, math.pi * 0.82, thickness=0.014, swallow_tail=0.028
    ))
    # Aletas cristalinas laterales tipo Dark Crystal
    wing_pts = [(0.0, 0.0), (0.045, 0.025), (0.085, 0.105), (0.032, 0.078), (0.0, 0.038)]
    for sgn in (-1, 1):
        h["plate"].add(*make_extruded_polygon(
            wing_pts, 0.010, bevel=0.003,
            center=(sgn * 0.108, 0.185, 0.015),
            rot=(0.15, sgn * -0.45, sgn * -0.22)
        ))
        h["trim"].add(*make_box(0.012, 0.085, 0.014, center=(sgn * 0.116, 0.205, 0.012),
                               rot=(0.15, 0.0, sgn * -0.35), bevel=0.002))
    # Crestón central y cristal cian frontal
    crest_pts = [(-0.012, 0.0), (0.012, 0.0), (0.018, 0.065), (0.0, 0.095), (-0.018, 0.065)]
    h["trim"].add(*make_extruded_polygon(crest_pts, 0.014, bevel=0.004, center=(0.0, 0.175, 0.132)))
    h["crystal"].add(*make_crystal_gem(0.017, 0.056, 6, center=(0.0, 0.195, 0.146), rot=(0.18, 0, 0)))
    h["glow"].add(*make_box(0.006, 0.042, 0.006, center=(0.0, 0.235, 0.125), rot=(-0.25, 0, 0)))
    return b


def build_dc_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("UpperArm_L", 1), ("UpperArm_R", -1)):
        p = _parts_dict()
        # 3 láminas escalonadas sobre el deltoides
        for k in range(3):
            y0 = -0.075 + k * 0.042
            y1 = 0.045 + k * 0.032
            r = 0.105 + k * 0.012
            ang_c = sgn * math.pi * 0.5
            p["plate"].add(*make_curved_armor_shell(
                y0, y1, r, r * 0.94, r * 0.96, r * 0.90,
                ang_c - 1.15, ang_c + 1.15,
                thickness=0.014, swallow_tail=0.022, flutes=3, flare_top=0.010
            ))
            p["trim"].add(*make_curved_armor_shell(
                y0 - 0.006, y0 + 0.016, r + 0.003, r * 0.94 + 0.003, r + 0.002, r * 0.94 + 0.002,
                ang_c - 1.18, ang_c + 1.18,
                thickness=0.015, swallow_tail=0.024
            ))
        # Aletas verticales Dark Crystal hacia arriba/afuera
        blade_pts = [(0.0, 0.0), (0.055, 0.03), (0.095, 0.145), (0.028, 0.105), (-0.012, 0.03)]
        p["plate"].add(*make_extruded_polygon(
            blade_pts, 0.014, bevel=0.004,
            center=(sgn * 0.095, 0.068, 0.0),
            rot=(0.0, 0.0 if sgn > 0 else math.pi, sgn * -0.35)
        ))
        # Clúster de 3 cristales cian en la hombrera
        for c_off, h_c, tilt in ((-0.028, 0.055, -0.35), (0.0, 0.078, -0.15), (0.028, 0.052, 0.12)):
            p["crystal"].add(*make_crystal_gem(
                0.014, h_c, 6,
                center=(sgn * 0.122, 0.048, c_off),
                rot=(tilt, 0.0, sgn * -0.58)
            ))
        p["glow"].add(*make_box(0.010, 0.060, 0.010, center=(sgn * 0.118, 0.005, 0.0), rot=(0, 0, sgn * -0.4)))
        out[side] = p
    return out


def build_dc_torso() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _parts_dict(), "UpperArm_L": _parts_dict(), "UpperArm_R": _parts_dict()}
    t = out["Torso"]
    # Peto frontal anatómico Dark Crystal (y en Torso: -0.18..0.31)
    t["plate"].add(*make_curved_armor_shell(
        -0.16, 0.28, 0.165, 0.128, 0.215, 0.155,
        -math.pi * 0.62, math.pi * 0.62,
        thickness=0.018, ny=12, na=22, swallow_tail=0.055, front_ridge=0.028, flutes=2, flare_top=0.014
    ))
    # Espaldar con escudos dorsales
    t["plate"].add(*make_curved_armor_shell(
        -0.14, 0.29, 0.162, 0.125, 0.212, 0.148,
        math.pi * 0.44, math.pi * 1.56,
        thickness=0.017, ny=10, na=20, swallow_tail=0.030, front_ridge=0.012
    ))
    # Cuello choker alto Dark Crystal con solapas doradas
    t["plate"].add(*make_curved_armor_shell(
        0.265, 0.345, 0.092, 0.095, 0.104, 0.105,
        -math.pi * 0.72, math.pi * 0.72,
        thickness=0.013, flare_top=0.012
    ))
    t["trim"].add(*make_curved_armor_shell(
        0.325, 0.350, 0.102, 0.104, 0.108, 0.110,
        -math.pi * 0.74, math.pi * 0.74,
        thickness=0.014
    ))
    # Cenefa inferior del peto en V cola de golondrina
    t["trim"].add(*make_curved_armor_shell(
        -0.17, -0.115, 0.169, 0.133, 0.172, 0.136,
        -math.pi * 0.58, math.pi * 0.58,
        thickness=0.019, swallow_tail=0.058
    ))
    # Yugo clavicular y costillas diagonales pectorales
    for sgn in (-1, 1):
        t["trim"].add(*make_box(0.115, 0.018, 0.020, center=(sgn * 0.075, 0.185, 0.165),
                               rot=(0.15, sgn * -0.28, sgn * 0.42), bevel=0.004))
        t["trim"].add(*make_box(0.095, 0.016, 0.020, center=(sgn * 0.082, 0.085, 0.162),
                               rot=(0.05, sgn * -0.25, sgn * -0.36), bevel=0.004))
        # Cristales laterales pectorales
        t["crystal"].add(*make_crystal_gem(
            0.013, 0.044, 6, center=(sgn * 0.085, 0.155, 0.178),
            rot=(0.25, 0.0, sgn * -0.55)
        ))
        # Escudos dorsales alados 3D
        d_pts = [(0.0, -0.06), (0.095, 0.01), (0.125, 0.14), (0.045, 0.12), (-0.01, 0.02)]
        t["plate"].add(*make_extruded_polygon(
            d_pts, 0.016, bevel=0.005,
            center=(sgn * 0.095, 0.14, -0.165),
            rot=(-0.18, sgn * 0.30, sgn * -0.12)
        ))

    # Engaste romboidal central + Gran Cristal Cian en el esternón
    dia_pts = [(0.0, -0.065), (0.045, 0.0), (0.0, 0.065), (-0.045, 0.0)]
    t["trim"].add(*make_extruded_polygon(dia_pts, 0.018, bevel=0.005, center=(0.0, 0.145, 0.188)))
    t["crystal"].add(*make_crystal_gem(0.026, 0.082, 6, center=(0.0, 0.145, 0.204), rot=(0.10, 0, 0)))
    t["glow"].add(*make_box(0.010, 0.14, 0.010, center=(0.0, 0.02, 0.180)))

    # Espaldarones integrados del peto sobre los brazos
    for side, sgn in (("UpperArm_L", 1), ("UpperArm_R", -1)):
        ap = out[side]
        ang_c = sgn * math.pi * 0.5
        ap["plate"].add(*make_curved_armor_shell(
            -0.06, 0.05, 0.095, 0.092, 0.092, 0.090,
            ang_c - 0.95, ang_c + 0.95, thickness=0.013, swallow_tail=0.018
        ))
    return out


def build_dc_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side in ("L", "R"):
        sgn = 1 if side == "L" else -1
        fa = _parts_dict()
        wr = _parts_dict()
        # Brazal del antebrazo (Forearm: y de -0.26 a 0.0)
        fa["plate"].add(*make_curved_armor_shell(
            -0.255, -0.035, 0.048, 0.048, 0.064, 0.064,
            -math.pi * 0.88, math.pi * 0.88,
            thickness=0.011, swallow_tail=0.018, front_ridge=0.009, flutes=3, flare_top=0.008
        ))
        fa["trim"].add(*make_curved_armor_shell(
            -0.075, -0.032, 0.063, 0.063, 0.068, 0.068,
            -math.pi * 0.90, math.pi * 0.90, thickness=0.012, swallow_tail=0.016
        ))
        fa["trim"].add(*make_curved_armor_shell(
            -0.258, -0.228, 0.049, 0.049, 0.052, 0.052,
            -math.pi * 0.90, math.pi * 0.90, thickness=0.012
        ))
        fa["glow"].add(*make_box(0.012, 0.085, 0.008, center=(sgn * 0.012, -0.145, 0.063)))
        fa["crystal"].add(*make_crystal_gem(0.011, 0.032, 6, center=(sgn * 0.052, -0.075, 0.020), rot=(0, 0, sgn * -0.4)))

        # Dorso de mano / guantelete y nudillera
        wr["plate"].add(*make_box(0.082, 0.11, 0.048, center=(sgn * 0.008, -0.068, 0.0), bevel=0.008))
        wr["trim"].add(*make_box(0.088, 0.022, 0.056, center=(sgn * 0.008, -0.045, 0.004), bevel=0.004))
        wr["crystal"].add(*make_crystal_gem(0.010, 0.028, 6, center=(sgn * 0.036, -0.065, 0.0), rot=(0, 0, sgn * -1.2)))
        out[f"Forearm_{side}"] = fa
        out[f"Wrist_{side}"] = wr
    return out


def build_dc_belt() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _parts_dict()}
    t = out["Torso"]
    # Cinturón articulado swallow-tail en Torso y=[-0.26, -0.13]
    t["plate"].add(*make_curved_armor_shell(
        -0.235, -0.145, 0.168, 0.142, 0.166, 0.138,
        -math.pi * 0.95, math.pi * 0.95,
        thickness=0.016, swallow_tail=0.036, front_ridge=0.016
    ))
    t["trim"].add(*make_curved_armor_shell(
        -0.160, -0.135, 0.171, 0.145, 0.171, 0.145,
        -math.pi * 0.95, math.pi * 0.95, thickness=0.017
    ))
    # Hebilla romboidal alada con cristal cian
    buckle_pts = [(0.0, -0.048), (0.052, 0.0), (0.0, 0.048), (-0.052, 0.0)]
    t["trim"].add(*make_extruded_polygon(buckle_pts, 0.018, bevel=0.005, center=(0.0, -0.185, 0.165)))
    t["crystal"].add(*make_crystal_gem(0.018, 0.052, 6, center=(0.0, -0.185, 0.178)))
    # Tabardo frontal en punta chevrón con gota cristalina
    tab_pts = [(-0.045, 0.0), (0.045, 0.0), (0.038, -0.18), (0.0, -0.255), (-0.038, -0.18)]
    t["plate"].add(*make_extruded_polygon(tab_pts, 0.012, bevel=0.004, center=(0.0, -0.220, 0.162), rot=(0.06, 0, 0)))
    t["glow"].add(*make_box(0.008, 0.13, 0.008, center=(0.0, -0.320, 0.172), rot=(0.06, 0, 0)))
    t["crystal"].add(*make_crystal_gem(0.012, 0.036, 6, center=(0.0, -0.430, 0.174)))
    return out


def build_dc_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _parts_dict(), "Thigh_L": _parts_dict(), "Thigh_R": _parts_dict()}
    # Culera trasera en Torso
    out["Torso"]["plate"].add(*make_curved_armor_shell(
        -0.29, -0.18, 0.178, 0.145, 0.172, 0.140,
        math.pi * 0.55, math.pi * 1.45, thickness=0.014, swallow_tail=0.025
    ))
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        th = out[side]
        # 3 láminas swallow-tail sobre el muslo superior
        for k in range(3):
            y0 = -0.12 - k * 0.075
            y1 = -0.01 - k * 0.070
            r = 0.112 - k * 0.005
            th["plate"].add(*make_curved_armor_shell(
                y0, y1, r, r * 0.96, r * 1.04, r,
                -0.85 + sgn * 0.35, 0.85 + sgn * 0.35,
                thickness=0.013, swallow_tail=0.024, flutes=3
            ))
            th["trim"].add(*make_curved_armor_shell(
                y0 - 0.005, y0 + 0.012, r + 0.003, r * 0.96 + 0.003, r + 0.003, r * 0.96 + 0.003,
                -0.88 + sgn * 0.35, 0.88 + sgn * 0.35, thickness=0.014, swallow_tail=0.026
            ))
        # Rondel hexagonal de cadera con cristal cian
        rondel = [(math.cos(i * math.tau / 6) * 0.042, math.sin(i * math.tau / 6) * 0.042) for i in range(6)]
        th["trim"].add(*make_extruded_polygon(rondel, 0.012, bevel=0.003,
                                              center=(sgn * 0.118, -0.02, 0.02),
                                              rot=(0, sgn * math.pi * 0.45, 0)))
        th["crystal"].add(*make_crystal_gem(0.013, 0.032, 6,
                                            center=(sgn * 0.128, -0.02, 0.02),
                                            rot=(0, 0, sgn * -math.pi * 0.5)))
    return out


def build_dc_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        th = _parts_dict()
        th["plate"].add(*make_curved_armor_shell(
            -0.42, -0.08, 0.088, 0.088, 0.106, 0.104,
            -math.pi * 0.78, math.pi * 0.78,
            thickness=0.013, swallow_tail=0.028, front_ridge=0.012, flutes=4, flare_top=0.010
        ))
        th["trim"].add(*make_curved_armor_shell(
            -0.12, -0.075, 0.105, 0.103, 0.110, 0.108,
            -math.pi * 0.80, math.pi * 0.80, thickness=0.014, swallow_tail=0.020
        ))
        # Aleta lateral defensiva
        fin_pts = [(0.0, 0.0), (0.045, 0.04), (0.025, 0.19), (-0.01, 0.14)]
        th["plate"].add(*make_extruded_polygon(
            fin_pts, 0.010, bevel=0.003,
            center=(sgn * 0.105, -0.26, 0.0),
            rot=(0, 0 if sgn > 0 else math.pi, sgn * -0.10)
        ))
        th["crystal"].add(*make_crystal_gem(0.012, 0.036, 6, center=(0.0, -0.24, 0.118)))
        out[side] = th
    return out


def build_dc_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Shin_L", 1), ("Shin_R", -1)):
        sh = _parts_dict()
        # Copa rotuliana facetada en Shin y=[-0.08, 0.09]
        sh["plate"].add(*make_curved_armor_shell(
            -0.07, 0.085, 0.080, 0.088, 0.086, 0.095,
            -math.pi * 0.60, math.pi * 0.60,
            thickness=0.015, swallow_tail=0.022, front_ridge=0.026
        ))
        sh["trim"].add(*make_curved_armor_shell(
            0.055, 0.092, 0.085, 0.094, 0.088, 0.098,
            -math.pi * 0.62, math.pi * 0.62, thickness=0.016
        ))
        # Ala lateral de rodillera en cuchilla
        wing_pts = [(0.0, -0.05), (0.055, -0.02), (0.068, 0.08), (0.015, 0.06)]
        sh["plate"].add(*make_extruded_polygon(
            wing_pts, 0.010, bevel=0.003,
            center=(sgn * 0.088, 0.01, 0.01),
            rot=(0, sgn * -0.35 + (0 if sgn > 0 else math.pi), 0)
        ))
        # Espina/púa central + cristal cian
        sh["trim"].add(*make_box(0.016, 0.13, 0.018, center=(0.0, 0.015, 0.112), bevel=0.004))
        sh["crystal"].add(*make_crystal_gem(0.016, 0.048, 6, center=(0.0, 0.012, 0.125), rot=(0.12, 0, 0)))
        out[side] = sh
    return out


def build_dc_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("L", 1), ("R", -1)):
        sh = _parts_dict()
        ak = _parts_dict()
        # Greba tibial con caña wing-sweep (Shin y=[-0.43, -0.04])
        sh["plate"].add(*make_curved_armor_shell(
            -0.42, -0.04, 0.062, 0.066, 0.078, 0.082,
            -math.pi * 0.88, math.pi * 0.88,
            thickness=0.012, swallow_tail=0.020, front_ridge=0.016, flutes=4, flare_top=0.008
        ))
        sh["trim"].add(*make_curved_armor_shell(
            -0.085, -0.035, 0.080, 0.084, 0.086, 0.090,
            -math.pi * 0.90, math.pi * 0.90, thickness=0.014, swallow_tail=0.022
        ))
        sh["glow"].add(*make_box(0.010, 0.10, 0.008, center=(0.0, -0.24, 0.088)))

        # Escarpe articulado sobre el pie (Ankle: y de -0.07 a 0.12, z de -0.10 a 0.21)
        ak["plate"].add(*make_box(0.112, 0.065, 0.24, center=(0.0, -0.02, 0.055), bevel=0.012))
        # Puntera gótica dragon-wedge
        toe_pts = [(-0.048, 0.0), (0.048, 0.0), (0.032, 0.075), (0.0, 0.105), (-0.032, 0.075)]
        ak["trim"].add(*make_extruded_polygon(toe_pts, 0.045, bevel=0.008,
                                              center=(0.0, -0.028, 0.155), rot=(math.pi * 0.5, 0, 0)))
        # Cuello del tobillo con aletas laterales y espolón trasero
        ak["plate"].add(*make_curved_armor_shell(
            0.02, 0.13, 0.068, 0.072, 0.075, 0.078,
            -math.pi * 0.85, math.pi * 0.85, thickness=0.011, flare_top=0.010
        ))
        ak["trim"].add(*make_box(0.014, 0.020, 0.055, center=(0.0, -0.01, -0.095), bevel=0.003))
        ak["crystal"].add(*make_crystal_gem(0.012, 0.034, 6, center=(0.0, 0.045, 0.095), rot=(0.45, 0, 0)))

        out[f"Shin_{side}"] = sh
        out[f"Ankle_{side}"] = ak
    return out


def build_dc_weapon() -> Dict[str, Dict[str, MeshPart]]:
    """Espada Dark Crystal montada en Wrist_R en orientación de agarre natural (+Z hacia adelante)."""
    out = {"Wrist_R": _parts_dict()}
    w = out["Wrist_R"]
    # En Wrist_R local (como en weapon/1.glb): puño en y~-0.07, hoja apuntando hacia +Z
    gy = -0.070
    # Empuñadura (eje Z)
    w["plate"].add(*make_cylinder(0.015, 0.016, 0.19, 14, center=(0.0, gy, 0.0), rot=(math.pi * 0.5, 0, 0)))
    for rz in (-0.05, 0.0, 0.05):
        w["trim"].add(*make_cylinder(0.019, 0.019, 0.012, 14, center=(0.0, gy, rz), rot=(math.pi * 0.5, 0, 0)))
    # Pomo + cristal cian
    w["trim"].add(*make_crystal_gem(0.026, 0.052, 6, center=(0.0, gy, -0.115), rot=(math.pi * 0.5, 0, 0)))
    w["crystal"].add(*make_crystal_gem(0.018, 0.040, 6, center=(0.0, gy, -0.145), rot=(math.pi * 0.5, 0, 0)))
    # Guarda alada swept Dark Crystal
    guard_pts = [
        (-0.038, -0.016), (0.038, -0.016), (0.130, 0.018), (0.160, 0.065),
        (0.092, 0.026), (0.0, 0.030), (-0.092, 0.026), (-0.160, 0.065), (-0.130, 0.018)
    ]
    w["trim"].add(*make_extruded_polygon(guard_pts, 0.026, bevel=0.006,
                                         center=(0.0, gy, 0.096), rot=(-math.pi * 0.5, 0, 0)))
    w["crystal"].add(*make_crystal_gem(0.024, 0.044, 6, center=(0.0, gy + 0.022, 0.106), rot=(0, 0, 0)))
    # Hoja ancha Dark Crystal con escalón en ricasso
    blade_pts = [
        (-0.030, 0.0), (-0.044, 0.14), (-0.034, 0.20), (-0.038, 0.65),
        (0.0, 0.84), (0.038, 0.65), (0.034, 0.20), (0.044, 0.14), (0.030, 0.0)
    ]
    w["plate"].add(*make_extruded_polygon(blade_pts, 0.016, bevel=0.006,
                                          center=(0.0, gy, 0.110), rot=(-math.pi * 0.5, 0, 0)))
    # Nervio dorado central y línea de energía cian
    w["trim"].add(*make_box(0.014, 0.022, 0.54, center=(0.0, gy, 0.410), bevel=0.002))
    w["glow"].add(*make_box(0.008, 0.026, 0.48, center=(0.0, gy, 0.430)))
    return out


def build_dc_shield() -> Dict[str, Dict[str, MeshPart]]:
    """Escudo Dark Crystal (Lineage II) montado en Wrist_L."""
    out = {"Wrist_L": _parts_dict()}
    s = out["Wrist_L"]
    # Posición natural respecto a Wrist_L (antebrazo izquierdo / puño izquierdo)
    cx, cy, cz = -0.055, -0.140, 0.0
    rot_sh = (0.0, -math.pi * 0.5, 0.0)

    hex_pts = [
        (0.0, 0.36), (0.21, 0.25), (0.24, 0.02), (0.15, -0.26),
        (0.0, -0.44), (-0.15, -0.26), (-0.24, 0.02), (-0.21, 0.25)
    ]
    body_p, body_i = make_extruded_polygon(hex_pts, 0.038, bevel=0.012, center=(0, 0, 0))
    body_p = transform_mesh(body_p, translate=(cx, cy, cz), rot_xyz=rot_sh)
    s["plate"].add(body_p, body_i)

    # Marco perimetral dorado (barras siguiendo los 8 segmentos del escudo)
    for k in range(len(hex_pts)):
        x0, y0 = hex_pts[k]
        x1, y1 = hex_pts[(k + 1) % len(hex_pts)]
        mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        seg_len = math.hypot(x1 - x0, y1 - y0) + 0.018
        ang = math.atan2(y1 - y0, x1 - x0)
        bp, bi = make_box(seg_len, 0.026, 0.048, center=(mx * 1.03, my * 1.03, 0.004), rot=(0, 0, ang), bevel=0.004)
        s["trim"].add(transform_mesh(bp, translate=(cx, cy, cz), rot_xyz=rot_sh), bi)

    # Cresta vertical + chevrones cian
    sp, si = make_box(0.026, 0.72, 0.022, center=(0, -0.03, 0.028), bevel=0.004)
    s["trim"].add(transform_mesh(sp, translate=(cx, cy, cz), rot_xyz=rot_sh), si)
    for sg in (-1, 1):
        gp, gi = make_box(0.12, 0.012, 0.014, center=(sg * 0.075, 0.14, 0.034), rot=(0, 0, sg * -0.45))
        s["glow"].add(transform_mesh(gp, translate=(cx, cy, cz), rot_xyz=rot_sh), gi)
        gp2, gi2 = make_box(0.10, 0.012, 0.014, center=(sg * 0.062, -0.17, 0.034), rot=(0, 0, sg * 0.52))
        s["glow"].add(transform_mesh(gp2, translate=(cx, cy, cz), rot_xyz=rot_sh), gi2)

    # 4 garras doradas y gran cristal central cian
    for c in range(4):
        ang = c * math.pi * 0.5 + math.pi * 0.25
        cp, ci = make_box(0.022, 0.090, 0.022,
                          center=(math.cos(ang) * 0.078, 0.02 + math.sin(ang) * 0.088, 0.038),
                          rot=(0, 0, ang + math.pi * 0.5), bevel=0.004)
        s["trim"].add(transform_mesh(cp, translate=(cx, cy, cz), rot_xyz=rot_sh), ci)

    gem_p, gem_i = make_crystal_gem(0.072, 0.16, 6, center=(0, 0.02, 0.048), rot=(math.pi * 0.5, 0, 0))
    s["crystal"].add(transform_mesh(gem_p, translate=(cx, cy, cz), rot_xyz=rot_sh), gem_i)

    # Enarmes (barra de agarre trasera)
    bar_p, bar_i = make_cylinder(0.015, 0.015, 0.16, 12, center=(0, 0, -0.042))
    s["trim"].add(transform_mesh(bar_p, translate=(cx, cy, cz), rot_xyz=rot_sh), bar_i)
    return out


# ==============================================================================
# 4. EMPAQUETADOR GLTF 2.0 BINARIO (.GLB) AUTOCONTENIDO
# ==============================================================================
def write_glb(out_path: str, bone_meshes: Dict[str, Dict[str, MeshPart]],
              textures: Dict[str, bytes]) -> dict:
    """Empaqueta las mallas por hueso + texturas PBR embebidas en un archivo .glb válido."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nodes = [dict(n) for n in SKELETON_TEMPLATE]
    name_to_node = {n["name"]: i for i, n in enumerate(nodes)}

    bin_buf = bytearray()
    buffer_views = []
    accessors = []
    meshes = []

    def append_bytes(raw: bytes, target: Optional[int] = None) -> int:
        # Alinear inicio a 4 bytes
        while len(bin_buf) % 4 != 0:
            bin_buf.append(0)
        offset = len(bin_buf)
        bin_buf.extend(raw)
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            bv["target"] = target
        bv_idx = len(buffer_views)
        buffer_views.append(bv)
        return bv_idx

    # Embebido de imágenes PNG
    img_idx_map = {}
    for key in ("plate_albedo", "plate_mr", "plate_emissive", "gold_albedo"):
        bv = append_bytes(textures[key])
        img_idx_map[key] = len(img_idx_map)
        # Se agregan en la lista `images` abajo en el mismo orden
    images = [
        {"bufferView": i, "mimeType": "image/png", "name": k}
        for i, k in enumerate(("plate_albedo", "plate_mr", "plate_emissive", "gold_albedo"))
    ]
    samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    gltf_textures = [{"sampler": 0, "source": i} for i in range(4)]

    materials = [
        {
            "name": "DarkCrystal_Plate",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicRoughnessTexture": {"index": 1},
                "metallicFactor": 0.85,
                "roughnessFactor": 0.28,
            },
            "emissiveTexture": {"index": 2},
            "emissiveFactor": [0.45, 0.85, 1.0],
            "doubleSided": True,
        },
        {
            "name": "DarkCrystal_GoldTrim",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 3},
                "metallicFactor": 0.92,
                "roughnessFactor": 0.22,
            },
            "doubleSided": True,
        },
        {
            "name": "DarkCrystal_CyanGem",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.22, 0.78, 0.98, 1.0],
                "metallicFactor": 0.18,
                "roughnessFactor": 0.08,
            },
            "emissiveFactor": [0.06, 0.42, 0.68],
            "doubleSided": True,
        },
        {
            "name": "DarkCrystal_Glow",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.48, 0.88, 1.0, 1.0],
                "metallicFactor": 0.10,
                "roughnessFactor": 0.15,
            },
            "emissiveFactor": [0.35, 0.82, 1.0],
            "doubleSided": True,
        },
    ]
    mat_idx_map = {"plate": 0, "trim": 1, "crystal": 2, "glow": 3}

    total_verts = 0
    total_tris = 0

    for bone_name, parts in bone_meshes.items():
        parent_idx = name_to_node[bone_name]
        for mat_key in ("plate", "trim", "crystal", "glow"):
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
                    "indices": idx_acc,
                    "material": mat_idx_map[mat_key],
                }],
            })
            node_idx = len(nodes)
            nodes.append({"name": f"{bone_name}_DarkCrystal_{mat_key}", "mesh": mesh_idx})
            nodes[parent_idx].setdefault("children", []).append(node_idx)

    while len(bin_buf) % 4 != 0:
        bin_buf.append(0)

    gltf = {
        "asset": {"version": "2.0", "generator": "Armaduras-glb/crear_dark_crystal.py"},
        "scene": 0,
        "scenes": [{"name": "DarkCrystal", "nodes": [0]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": gltf_textures,
        "images": images,
        "samplers": samplers,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_buf)}],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack("<II", len(bin_buf), 0x004E4942) + bytes(bin_buf)

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(json_chunk)
        f.write(bin_chunk)

    return {"path": out_path, "bytes": total_len, "verts": total_verts, "tris": total_tris}


# ==============================================================================
# 5. ACTUALIZACIÓN DE CATÁLOGOS
# ==============================================================================
DARK_CRYSTAL_CATALOG_ENTRIES = {
    "underlayer": {"idx": 16, "name": "Cota Lamellar Dark Crystal (L2)"},
    "helmet":     {"idx": 9,  "name": "Diadema Dark Crystal con Crestón Cian (L2)"},
    "shoulders":  {"idx": 12, "name": "Hombreras Dark Crystal con Aletas Cian (L2)"},
    "torso":      {"idx": 20, "name": "Coraza Pesada Dark Crystal (L2)"},
    "gloves":     {"idx": 6,  "name": "Guanteletes Dark Crystal (L2)"},
    "belt":       {"idx": 8,  "name": "Cinturón & Tabardo Dark Crystal (L2)"},
    "tassets":    {"idx": 9,  "name": "Escarcelas Dark Crystal (L2)"},
    "cuisses":    {"idx": 8,  "name": "Musleras Dark Crystal (L2)"},
    "knees":      {"idx": 8,  "name": "Rodilleras Dark Crystal con Aletas de Cristal (L2)"},
    "boots":      {"idx": 13, "name": "Botas & Escarpes Dark Crystal (L2)"},
    "weapon":     {"idx": 81, "name": "Espada Dark Crystal (L2)", "familia": "espada", "nivel": 40},
    "shield":     [
        {"idx": 0,  "name": "Sin Escudo"},
        {"idx": 11, "name": "Escudo Dark Crystal (L2)", "nivel": 40},
    ],
}


def update_catalogo_estudio() -> None:
    with open(CATALOGO_ESTUDIO_PATH, encoding="utf-8") as f:
        estudio = json.load(f)

    for ranura, entry in DARK_CRYSTAL_CATALOG_ENTRIES.items():
        if isinstance(entry, list):
            lst = estudio.setdefault(ranura, [])
            by_idx = {it["idx"]: it for it in lst}
            for item in entry:
                if item["idx"] in by_idx:
                    by_idx[item["idx"]].update(item)
                else:
                    lst.append(dict(item))
            lst.sort(key=lambda x: x["idx"])
        else:
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
    textures = generate_dark_crystal_textures()
    builders = {
        ("helmet", 9): build_dc_helmet,
        ("shoulders", 12): build_dc_shoulders,
        ("torso", 20): build_dc_torso,
        ("gloves", 6): build_dc_gloves,
        ("belt", 8): build_dc_belt,
        ("tassets", 9): build_dc_tassets,
        ("cuisses", 8): build_dc_cuisses,
        ("knees", 8): build_dc_knees,
        ("boots", 13): build_dc_boots,
        ("weapon", 81): build_dc_weapon,
        ("shield", 11): build_dc_shield,
    }

    print("Generando piezas Dark Crystal (.glb)...")
    for (slot, idx), fn in builders.items():
        out_path = os.path.join(PIEZAS_DIR, slot, f"{idx}.glb")
        stats = write_glb(out_path, fn(), textures)
        print(f"  [OK] {slot}/{idx}.glb -> {stats['verts']} verts, {stats['tris']} tris, {stats['bytes']} bytes")

    update_catalogo_estudio()
    print(f"Actualizado {os.path.relpath(CATALOGO_ESTUDIO_PATH, ROOT_DIR)}")

    subprocess.run([sys.executable, GENERAR_CATALOGO_PATH], check=True)


if __name__ == "__main__":
    main()
