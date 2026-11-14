#!/usr/bin/env python3
"""Generador procedural en Python de los 3 nuevos arquetipos asimétricos (uno por clase):
  1) PESADA — Coloso de Basalto & Fragua (helmet/11, shoulders/14, torso/22, gloves/8, belt/10, tassets/11, cuisses/10, knees/10, boots/15, weapon/83, shield/13)
  2) LIGERA — Cazador de Dunas & Tormenta (helmet/12, shoulders/15, torso/23, gloves/9, belt/11, tassets/12, cuisses/11, knees/11, boots/16, weapon/84, shield/14)
  3) MAGO   — Tejedor del Astrolabio & Eclipse (helmet/13, shoulders/16, torso/24, gloves/10, belt/12, tassets/13, cuisses/12, knees/12, boots/17, weapon/85, shield/15)

Genera los 33 archivos .glb compatibles con `CreadorHumano_hombre`,
actualiza `piezas/catalogo-estudio.json` (incluyendo underlayer 18..20 y shield 13..15)
y regenera `piezas/catalogo.json`.

Uso:
    python3 tools/piezas/crear_arquetipos.py
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from crear_ravenbone import (  # noqa: E402
    CATALOGO_ESTUDIO_PATH,
    GENERAR_CATALOGO_PATH,
    PIEZAS_DIR,
    ROOT_DIR,
    SKELETON_TEMPLATE,
    MeshPart,
    _img_to_png_bytes,
    make_box,
    make_curved_shell,
    make_cylinder as _base_make_cylinder,
    make_extruded_polygon,
    make_sphere,
    make_tube_curve,
    transform_mesh,
)


def make_cylinder(
    r_top: float, r_bot: float, height: float, segs: int = 12,
    center=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1),
) -> Tuple[np.ndarray, np.ndarray]:
    pos, idx = _base_make_cylinder(r_top, r_bot, height, segs=segs, center=(0, 0, 0), rot=(0, 0, 0))
    return transform_mesh(pos, translate=center, rot_xyz=rot, scale=scale), idx


# ==============================================================================
# UTILIDAD EXTRA: TOROIDE / ANILLO PARAMÉTRICO
# ==============================================================================
def make_torus(
    major_r: float,
    minor_r: float,
    seg_u: int = 24,
    seg_v: int = 8,
    arc: float = math.tau,
    center=(0, 0, 0),
    rot=(0, 0, 0),
    scale=(1, 1, 1),
) -> Tuple[np.ndarray, np.ndarray]:
    verts, tris = [], []
    closed_u = abs(arc - math.tau) < 1e-4
    nu = seg_u if closed_u else seg_u + 1
    for i in range(nu):
        u = (i / seg_u) * arc
        cu, su = math.cos(u), math.sin(u)
        for j in range(seg_v):
            v = (j / seg_v) * math.tau
            cv, sv = math.cos(v), math.sin(v)
            r = major_r + minor_r * cv
            verts.append([r * cu, minor_r * sv, r * su])
    limit_u = seg_u if closed_u else seg_u
    for i in range(limit_u):
        ni = (i + 1) % seg_u if closed_u else (i + 1)
        for j in range(seg_v):
            nj = (j + 1) % seg_v
            a = i * seg_v + j
            b = ni * seg_v + j
            c = ni * seg_v + nj
            d = i * seg_v + nj
            tris.extend([a, b, c, a, c, d])
    return transform_mesh(np.array(verts, dtype=np.float32), center, rot, scale), np.array(tris, dtype=np.uint32)


def _parts(*keys: str) -> Dict[str, MeshPart]:
    return {k: MeshPart() for k in keys}


def write_glb_custom(
    out_path: str,
    bone_meshes: Dict[str, Dict[str, MeshPart]],
    textures: Dict[str, bytes],
    tex_keys: Tuple[str, ...],
    material_specs: List[Tuple[str, dict]],
    scene_name: str,
    prefix: str,
) -> dict:
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

    for k in tex_keys:
        append_bytes(textures[k])
    images = [{"bufferView": i, "mimeType": "image/png", "name": k} for i, k in enumerate(tex_keys)]
    samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    gltf_textures = [{"sampler": 0, "source": i} for i in range(len(tex_keys))]

    materials = [spec for _, spec in material_specs]
    mat_idx_map = {key: i for i, (key, _) in enumerate(material_specs)}

    total_verts, total_tris = 0, 0
    for bone_name, parts in bone_meshes.items():
        parent_idx = name_to_node[bone_name]
        for mat_key in mat_idx_map:
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
            nodes.append({"name": f"{bone_name}_{prefix}_{mat_key}", "mesh": mesh_idx})
            nodes[parent_idx].setdefault("children", []).append(node_idx)

    while len(bin_buf) % 4 != 0:
        bin_buf.append(0)

    gltf = {
        "asset": {"version": "2.0", "generator": "Armaduras-glb/crear_arquetipos.py"},
        "scene": 0, "scenes": [{"name": scene_name, "nodes": [0]}],
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
# ARQUETIPO 1 (PESADA): COLOSO DE BASALTO & FRAGUA
# ==============================================================================
def gen_basalt_textures() -> Dict[str, bytes]:
    rng = np.random.default_rng(101)
    s = 128
    # Basalto volcánico gris pizarra con grietas de lava sutiles
    basalt = Image.new("RGB", (s, s), (42, 44, 52))
    bd = ImageDraw.Draw(basalt)
    for _ in range(36):
        x0, y0 = int(rng.integers(0, s)), int(rng.integers(0, s))
        x1, y1 = x0 + int(rng.integers(-28, 29)), y0 + int(rng.integers(-28, 29))
        bd.line([(x0, y0), (x1, y1)], fill=(26, 27, 33), width=2)
    arr = np.clip(np.array(basalt, dtype=np.int16) + rng.integers(-6, 7, (s, s, 1)), 0, 255).astype(np.uint8)
    basalt = Image.fromarray(arr, "RGB")

    # Hierro forjado colado oscuro
    iron = Image.new("RGB", (s, s), (28, 32, 38))
    idraw = ImageDraw.Draw(iron)
    for _ in range(70):
        cx, cy = int(rng.integers(0, s)), int(rng.integers(0, s))
        r = int(rng.integers(3, 8))
        v = int(rng.integers(20, 48))
        idraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(v, v + 3, v + 8))

    # Bronce forjado envejecido (acento coherente con obsidiana y magma)
    copper = Image.new("RGB", (s, s), (92, 54, 34))
    cdraw = ImageDraw.Draw(copper)
    for _ in range(45):
        cx, cy = int(rng.integers(0, s)), int(rng.integers(0, s))
        r = int(rng.integers(4, 11))
        cdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(118, 72, 46))

    return {
        "basalt_albedo": _img_to_png_bytes(basalt),
        "iron_albedo": _img_to_png_bytes(iron),
        "copper_albedo": _img_to_png_bytes(copper),
    }


BASALT_MAT_SPECS = [
    ("basalt", {"name": "Basalt_Stone", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.10, "roughnessFactor": 0.86}, "doubleSided": True}),
    ("iron", {"name": "Basalt_Iron", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.88, "roughnessFactor": 0.40}, "doubleSided": True}),
    ("copper", {"name": "Basalt_Bronze", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}, "metallicFactor": 0.78, "roughnessFactor": 0.42}, "doubleSided": True}),
    ("brass", {"name": "Basalt_Gunmetal", "pbrMetallicRoughness": {"baseColorFactor": [0.23, 0.26, 0.30, 1.0], "metallicFactor": 0.84, "roughnessFactor": 0.34}, "doubleSided": True}),
    ("leather", {"name": "Basalt_CharredLeather", "pbrMetallicRoughness": {"baseColorFactor": [0.05, 0.04, 0.04, 1.0], "metallicFactor": 0.06, "roughnessFactor": 0.88}, "doubleSided": True}),
    ("magma", {"name": "Basalt_Magma", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.34, 0.05, 1.0], "metallicFactor": 0.05, "roughnessFactor": 0.14}, "emissiveFactor": [1.0, 0.32, 0.04], "doubleSided": True}),
]
BASALT_KEYS = ("basalt", "iron", "copper", "brass", "leather", "magma")


def _bp() -> Dict[str, MeshPart]:
    return _parts(*BASALT_KEYS)


def _horn(part: MeshPart, pts: List[Tuple[float, float, float]], r0: float, segs: int = 8) -> None:
    """Cuerno / espina orgánica cónica suave mediante make_tube_curve."""
    n = len(pts)
    radii = [max(0.0018, r0 * ((1.0 - i / max(1, n - 1)) ** 0.78)) for i in range(n)]
    part.add(*make_tube_curve(pts, radii, segs=segs))


def _boot_foot(
    parts: Dict[str, MeshPart],
    sole_key: str,
    vamp_key: str,
    toe_key: str,
    spur_key: str,
    foot_w: float = 0.106,
    toe_len: float = 0.175,
    heel_len: float = 0.068,
    ground_y: float = -0.052,
    pointed: float = 0.25,
) -> None:
    """Pie anatómico esculpido sin bloques rectangulares."""
    sole_pts = [
        (-foot_w * 0.45, -heel_len),
        (foot_w * 0.45, -heel_len),
        (foot_w * 0.50, toe_len * 0.45),
        (foot_w * 0.28, toe_len * 0.88),
        (0.0, toe_len * (1.02 + pointed)),
        (-foot_w * 0.28, toe_len * 0.88),
        (-foot_w * 0.50, toe_len * 0.45),
    ]
    parts[sole_key].add(*make_extruded_polygon(
        sole_pts, 0.020, bevel=0.004,
        center=(0, ground_y + 0.012, 0), rot=(math.pi / 2, 0, 0)
    ))
    parts[vamp_key].add(*make_sphere(
        foot_w * 0.50, segs=14, rings=10,
        center=(0, ground_y + 0.024, toe_len * 0.34),
        scale=(0.96, 0.62, (toe_len * 0.86) / (foot_w * 0.50))
    ))
    # Puntera afilada curva hacia arriba
    _horn(parts[toe_key], [
        (0, ground_y + 0.026, toe_len * 0.72),
        (0, ground_y + 0.032, toe_len * (0.98 + pointed * 0.55)),
        (0, ground_y + 0.068 + pointed * 0.04, toe_len * (1.08 + pointed * 0.90)),
    ], foot_w * 0.28, segs=8)
    # Espolón trasero de talón
    _horn(parts[spur_key], [
        (0, ground_y + 0.034, -heel_len * 0.85),
        (0, ground_y + 0.038, -heel_len - 0.035),
        (0, ground_y + 0.056, -heel_len - 0.058),
    ], 0.013, segs=6)


def build_basalt_helmet() -> Dict[str, Dict[str, MeshPart]]:
    """Yelmo sallet volcánico esculpido con visera biselada, rendija de magma y cuernos forjados."""
    out = {"Head": _bp()}
    h = out["Head"]
    # Calota craneal anatómica esculpida + cresta dorsal
    h["basalt"].add(*make_curved_shell(
        0.04, 0.24, 0.118, 0.128, 0.102, 0.112,
        -math.pi * 0.95, math.pi * 0.95, thickness=0.013, ny=12, na=24, flare_top=0.012, peak_front_y=0.022
    ))
    # Visera gótica en pico frontal biselada y nuca en cola de langosta
    h["iron"].add(*make_curved_shell(
        0.095, 0.185, 0.126, 0.136, 0.118, 0.126,
        -1.05, 1.05, thickness=0.012, ny=8, na=16, peak_front_y=-0.028
    ))
    h["basalt"].add(*make_curved_shell(
        -0.06, 0.09, 0.128, 0.138, 0.116, 0.126,
        math.pi * 0.58, math.pi * 1.42, thickness=0.012, ny=8, na=14, flare_bot=0.034, tatter_waves=3, tatter_amp=0.018
    ))
    # Rendija ocular incandescente + barba de hierro forjado
    h["magma"].add(*make_curved_shell(
        0.104, 0.120, 0.128, 0.140, 0.128, 0.140,
        -0.68, 0.68, thickness=0.008, ny=4, na=12
    ))
    h["iron"].add(*make_curved_shell(
        0.015, 0.102, 0.124, 0.134, 0.126, 0.136,
        -0.78, 0.78, thickness=0.011, ny=6, na=14, peak_front_y=-0.025
    ))
    # Cuernos asimétricos de bronce y obsidiana curvados hacia atrás
    _horn(h["brass"], [(0.088, 0.185, 0.02), (0.148, 0.248, -0.02), (0.135, 0.335, -0.09), (0.092, 0.385, -0.14)], 0.022)
    _horn(h["copper"], [(-0.085, 0.185, 0.02), (-0.138, 0.238, -0.01), (-0.128, 0.305, -0.07)], 0.019)
    # Cresta dorsal espinosa
    for k in range(3):
        _horn(h["iron"], [(0.0, 0.235 - k * 0.025, 0.04 - k * 0.065), (0.0, 0.285 - k * 0.02, 0.01 - k * 0.075), (0.0, 0.305 - k * 0.02, -0.03 - k * 0.08)], 0.012)
    return out


def build_basalt_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    out = {"UpperArm_L": _bp(), "Torso": _bp(), "UpperArm_R": _bp()}
    # IZQUIERDO: Gran hombrera esculpida en capas + 2 cuernos Juggernaut + manto de cuero rasgado
    ul = out["UpperArm_L"]
    ul["basalt"].add(*make_curved_shell(
        -0.03, 0.095, 0.102, 0.104, 0.084, 0.086,
        -0.5, math.pi * 1.15, thickness=0.014, ny=10, na=16, flare_bot=0.028
    ))
    ul["iron"].add(*make_curved_shell(
        0.02, 0.118, 0.114, 0.114, 0.092, 0.092,
        -0.35, math.pi * 1.05, thickness=0.011, ny=8, na=14, peak_front_y=0.022
    ))
    ul["leather"].add(*make_curved_shell(
        -0.24, 0.04, 0.112, 0.110, 0.092, 0.092,
        0.1, math.pi * 1.15, thickness=0.008, ny=10, na=16, flare_bot=0.038, tatter_waves=4, tatter_amp=0.042
    ))
    _horn(ul["brass"], [(0.055, 0.085, 0.01), (0.125, 0.165, -0.01), (0.145, 0.265, -0.05)], 0.024)
    _horn(ul["copper"], [(0.085, 0.055, -0.025), (0.155, 0.115, -0.045), (0.195, 0.185, -0.08)], 0.018)

    # DERECHO: Hombrera escalonada vanguardia + costilla de bronce
    ur = out["UpperArm_R"]
    ur["iron"].add(*make_curved_shell(
        -0.02, 0.082, 0.098, 0.098, 0.082, 0.082,
        -math.pi * 1.05, 0.4, thickness=0.012, ny=8, na=14, flare_bot=0.020
    ))
    ur["brass"].add(*make_tube_curve([(-0.02, 0.075, 0.06), (-0.095, 0.052, 0.0), (-0.03, 0.068, -0.06)], [0.012, 0.014, 0.011], segs=8))

    # DORSAL EN TORSO: Aletas de obsidiana y chimeneas curvas con magma
    tr = out["Torso"]
    _horn(tr["copper"], [(0.045, 0.12, -0.14), (0.095, 0.24, -0.18), (0.125, 0.34, -0.14)], 0.020)
    _horn(tr["iron"], [(-0.045, 0.10, -0.14), (-0.088, 0.21, -0.17), (-0.112, 0.30, -0.13)], 0.017)
    tr["magma"].add(*make_sphere(0.034, segs=10, rings=8, center=(0.0, 0.14, -0.175), scale=(0.8, 1.8, 0.6)))
    return out


def build_basalt_torso() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _bp()}
    t = out["Torso"]
    # Coraza V-esculpida de basalto con faldón inferior y clavículas marcadas
    t["basalt"].add(*make_curved_shell(
        -0.16, 0.28, 0.176, 0.138, 0.224, 0.162,
        -math.pi, math.pi, thickness=0.016, ny=12, na=28, peak_front_y=0.028
    ))
    # Peto frontal biselado en escudo gótico + núcleo de magma vivo
    chest_pts = [(0.0, -0.04), (0.095, 0.06), (0.115, 0.22), (0.0, 0.25), (-0.115, 0.22), (-0.095, 0.06)]
    t["iron"].add(*make_extruded_polygon(chest_pts, 0.026, bevel=0.008, center=(0, 0.04, 0.158), rot=(-0.10, 0, 0)))
    t["magma"].add(*make_sphere(0.036, segs=10, rings=8, center=(0, 0.145, 0.175), scale=(0.85, 1.35, 0.55)))
    # Costillas forjadas curvas en el torso + collar dentado alto
    for side in (-1, 1):
        for ry in (0.06, 0.13, 0.20):
            t["brass"].add(*make_tube_curve([
                (side * 0.032, ry, 0.172),
                (side * 0.145, ry + 0.015, 0.115),
                (side * 0.185, ry + 0.025, -0.02),
            ], [0.009, 0.011, 0.007], segs=7))
    t["iron"].add(*make_curved_shell(
        0.24, 0.35, 0.118, 0.114, 0.146, 0.134,
        0.48, math.tau - 0.48, thickness=0.012, ny=6, na=20, flare_top=0.024, tatter_waves=5, tatter_amp=0.016
    ))
    # Cadena diagonal de fragua cruzando el pecho
    t["brass"].add(*make_tube_curve([(-0.15, 0.25, 0.14), (-0.02, 0.09, 0.185), (0.14, -0.10, 0.13)], [0.011, 0.012, 0.010], segs=8))
    return out


def build_basalt_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Forearm_R": _bp(), "Wrist_R": _bp(), "Forearm_L": _bp(), "Wrist_L": _bp()}
    for sd, sgn in (("R", -1), ("L", 1)):
        fa, wr = out["Forearm_" + sd], out["Wrist_" + sd]
        fa["basalt"].add(*make_curved_shell(-0.25, -0.02, 0.056, 0.056, 0.072, 0.072, -math.pi, math.pi, thickness=0.011, flare_top=0.012))
        # Aleta codera gótica biselada
        fin_pts = [(0.0, -0.04), (sgn * 0.065, 0.01), (sgn * 0.045, 0.085), (0.0, 0.04)]
        fa["iron"].add(*make_extruded_polygon(fin_pts, 0.018, bevel=0.004, center=(sgn * 0.045, -0.04, -0.02), rot=(0, sgn * 0.35, 0)))
        fa["brass"].add(*make_tube_curve([(sgn * 0.055, -0.06, 0.02), (sgn * 0.068, -0.15, 0.0), (sgn * 0.052, -0.23, 0.02)], [0.009, 0.011, 0.008], segs=7))
        # Guantelete con garras forjadas en los nudillos
        wr["iron"].add(*make_curved_shell(-0.075, 0.005, 0.048, 0.042, 0.052, 0.046, -math.pi, math.pi, thickness=0.008))
        for kz in (-0.024, 0.0, 0.024):
            _horn(wr["brass"], [(sgn * 0.040, -0.035, kz), (sgn * 0.065, -0.052, kz), (sgn * 0.078, -0.082, kz)], 0.008, segs=6)
    return out


def build_basalt_belt() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Pelvis": _bp()}
    p = out["Pelvis"]
    p["iron"].add(*make_curved_shell(-0.07, 0.05, 0.182, 0.146, 0.188, 0.152, -math.pi, math.pi, thickness=0.015, ny=6, na=24))
    # Hebilla gótica de bronce con ascua de magma
    b_pts = [(0.0, -0.065), (0.055, -0.01), (0.042, 0.055), (0.0, 0.072), (-0.042, 0.055), (-0.055, -0.01)]
    p["brass"].add(*make_extruded_polygon(b_pts, 0.022, bevel=0.006, center=(0, -0.01, 0.158)))
    p["magma"].add(*make_sphere(0.020, segs=8, rings=6, center=(0, -0.005, 0.172), scale=(0.85, 1.35, 0.6)))
    # Faldón de guerra rasgado en girones asimétricos + delantal central biselado
    p["leather"].add(*make_curved_shell(
        -0.52, -0.03, 0.228, 0.195, 0.184, 0.148,
        0.38, math.tau - 0.38, thickness=0.009, ny=12, na=28, flare_bot=0.045, tatter_waves=5, tatter_amp=0.055
    ))
    apron_pts = [(0.0, -0.34), (0.072, -0.22), (0.084, 0.0), (-0.084, 0.0), (-0.072, -0.22)]
    p["basalt"].add(*make_extruded_polygon(apron_pts, 0.016, bevel=0.005, center=(0, -0.06, 0.152), rot=(-0.08, 0, 0)))
    # Cadenas pesadas arqueadas laterales
    for sgn in (-1, 1):
        p["brass"].add(*make_tube_curve([
            (sgn * 0.12, 0.01, 0.13), (sgn * 0.195, -0.09, 0.02), (sgn * 0.12, -0.01, -0.13)
        ], [0.008, 0.009, 0.008], segs=7))
    return out


def build_basalt_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _bp()
        for row in range(3):
            yc = -0.045 - row * 0.076
            w = 0.056 - row * 0.005
            plate_pts = [(0.0, -0.052), (w, 0.0), (w * 0.85, 0.042), (-w * 0.85, 0.042), (-w, 0.0)]
            p["basalt" if row != 1 else "iron"].add(*make_extruded_polygon(
                plate_pts, 0.016, bevel=0.005,
                center=(sgn * (0.118 + row * 0.004), yc, 0.015),
                rot=(0.0, sgn * math.pi * 0.42, sgn * -0.14)
            ))
            p["brass"].add(*make_tube_curve([(sgn * 0.110, yc + 0.032, -0.02), (sgn * 0.126, yc + 0.028, 0.05)], [0.005, 0.005], segs=6))
        out[side] = p
    return out


def build_basalt_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _bp()
        p["iron"].add(*make_curved_shell(-0.37, -0.07, 0.094, 0.094, 0.115, 0.115, -math.pi, math.pi, thickness=0.012, peak_front_y=-0.020))
        ridge_pts = [(0.0, -0.13), (0.042, 0.0), (0.0, 0.13), (-0.042, 0.0)]
        p["basalt"].add(*make_extruded_polygon(ridge_pts, 0.018, bevel=0.006, center=(sgn * 0.012, -0.22, 0.105)))
        p["brass"].add(*make_tube_curve([(sgn * 0.08, -0.12, 0.06), (sgn * 0.11, -0.22, 0.0), (sgn * 0.07, -0.32, -0.05)], [0.008, 0.010, 0.008], segs=7))
        out[side] = p
    return out


def build_basalt_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _bp(), "Knee_R": _bp()}
    for side, sgn in (("Knee_L", 1), ("Knee_R", -1)):
        k = out[side]
        k["iron"].add(*make_curved_shell(-0.048, 0.048, 0.086, 0.086, 0.086, 0.086, -1.6, 1.6, thickness=0.011))
        wing_pts = [(0.0, -0.068), (0.055, -0.015), (0.068, 0.035), (0.0, 0.072), (-0.048, 0.025), (-0.045, -0.025)]
        k["basalt"].add(*make_extruded_polygon(wing_pts, 0.022, bevel=0.007, center=(sgn * 0.012, 0.0, 0.088), rot=(0, sgn * 0.18, 0)))
        _horn(k["brass"], [(sgn * 0.055, 0.02, 0.065), (sgn * 0.095, 0.055, 0.04), (sgn * 0.115, 0.105, 0.005)], 0.012, segs=6)
        if sgn > 0:
            k["magma"].add(*make_sphere(0.020, segs=8, rings=6, center=(0, 0.0, 0.104), scale=(0.8, 1.3, 0.6)))
    return out


def build_basalt_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("L", 1), ("R", -1)):
        sh, ak = _bp(), _bp()
        sh["iron"].add(*make_curved_shell(-0.44, -0.05, 0.068, 0.072, 0.094, 0.098, -math.pi, math.pi, thickness=0.013, peak_front_y=0.024))
        shin_pts = [(0.0, -0.17), (0.046, 0.04), (0.0, 0.18), (-0.046, 0.04)]
        sh["basalt"].add(*make_extruded_polygon(shin_pts, 0.022, bevel=0.007, center=(0, -0.23, 0.088)))
        sh["copper"].add(*make_curved_shell(-0.14, -0.04, 0.096, 0.102, 0.108, 0.112, -math.pi, math.pi, thickness=0.010, flare_top=0.012))
        _horn(sh["brass"], [(sgn * 0.075, -0.10, 0.02), (sgn * 0.105, -0.05, -0.01), (sgn * 0.118, 0.01, -0.03)], 0.011, segs=6)
        _boot_foot({"basalt": ak["basalt"], "iron": ak["iron"], "brass": ak["brass"], "copper": ak["copper"]},
                   "iron", "basalt", "brass", "copper", foot_w=0.114, toe_len=0.182, pointed=0.22)
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_basalt_weapon() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_R": _bp()}
    w = out["Wrist_R"]
    gy = -0.035
    # Asta de hierro forjado orgánicamente curvada
    w["iron"].add(*make_tube_curve([
        (0, gy, -0.44), (0.010, gy - 0.008, -0.02), (-0.012, gy + 0.006, 0.42), (0.006, gy - 0.006, 0.80), (0, gy, 1.02)
    ], [0.016, 0.016, 0.016, 0.016, 0.016], segs=10))
    w["leather"].add(*make_cylinder(0.021, 0.021, 0.28, segs=12, center=(0, gy, 0.01), rot=(math.pi / 2, 0, 0)))
    _horn(w["basalt"], [(0, gy, -0.44), (0.008, gy, -0.51), (0, gy, -0.58)], 0.022)
    # Cabeza hacha-martillo volcánica esculpida biselada + cuernos y núcleo magma
    head_pts = [
        (-0.08, 0.88), (-0.18, 0.78), (-0.23, 0.94), (-0.20, 1.07),
        (-0.06, 1.00), (0.06, 1.00), (0.21, 1.08), (0.26, 0.94), (0.16, 0.75), (0.04, 0.84)
    ]
    w["basalt"].add(*make_extruded_polygon(head_pts, 0.056, bevel=0.014, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    frame_pts = [(-0.04, 0.84), (-0.12, 0.82), (-0.14, 1.01), (0.11, 1.02), (0.12, 0.83)]
    w["iron"].add(*make_extruded_polygon(frame_pts, 0.072, bevel=0.008, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    w["magma"].add(*make_sphere(0.042, segs=10, rings=8, center=(0, gy, 0.925), scale=(0.85, 1.15, 1.35)))
    w["magma"].add(*make_tube_curve([(-0.08, gy, 0.92), (-0.16, gy, 0.96), (-0.21, gy, 1.03)], [0.006, 0.006, 0.004], segs=6))
    w["magma"].add(*make_tube_curve([(0.08, gy, 0.91), (0.17, gy, 0.93), (0.23, gy, 0.88)], [0.006, 0.006, 0.004], segs=6))
    _horn(w["brass"], [(0, gy, 1.00), (-0.04, gy + 0.01, 1.11), (-0.09, gy - 0.01, 1.20)], 0.018)
    _horn(w["copper"], [(0.05, gy, 0.99), (0.11, gy - 0.01, 1.09), (0.16, gy + 0.01, 1.17)], 0.014)
    return out


def build_basalt_shield() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_L": _bp()}
    s = out["Wrist_L"]
    gy = -0.035
    s["leather"].add(*make_cylinder(0.015, 0.015, 0.18, segs=10, center=(0, gy, 0.0), rot=(math.pi / 2, 0, 0)))
    sh_pts = [(0.0, 0.34), (0.18, 0.25), (0.16, 0.02), (0.11, -0.22), (0.0, -0.34), (-0.11, -0.22), (-0.16, 0.02), (-0.18, 0.25)]
    s["basalt"].add(*make_extruded_polygon(sh_pts, 0.030, bevel=0.010, center=(0.052, gy + 0.02, 0.0), rot=(0, math.pi / 2, 0)))
    s["magma"].add(*make_sphere(0.050, segs=10, rings=8, center=(0.072, gy + 0.05, 0.0), scale=(0.55, 1.25, 0.90)))
    s["iron"].add(*make_torus(0.084, 0.012, 18, 6, center=(0.068, gy + 0.05, 0.0), rot=(0, 0, math.pi / 2)))
    for dz in (-0.035, 0.0, 0.035):
        s["iron"].add(*make_tube_curve([(0.075, gy - 0.05, dz), (0.092, gy + 0.05, dz * 0.8), (0.075, gy + 0.15, dz)], [0.0065, 0.007, 0.0065], segs=6))
    for sd in (-1, 1):
        _horn(s["brass"], [(0.052, gy + 0.24, sd * 0.15), (0.066, gy + 0.34, sd * 0.22), (0.042, gy + 0.44, sd * 0.25)], 0.017)
    _horn(s["copper"], [(0.050, gy - 0.31, 0.0), (0.058, gy - 0.40, 0.0), (0.038, gy - 0.46, 0.0)], 0.015)
    return out


# ==============================================================================
# ARQUETIPO 2 (LIGERA): CAZADOR DE DUNAS & TORMENTA (ROSTRO ABIERTO S-RANK)
# ==============================================================================
def gen_dune_textures() -> Dict[str, bytes]:
    rng = np.random.default_rng(202)
    s = 128
    sand = Image.new("RGB", (s, s), (156, 128, 86))
    sd = ImageDraw.Draw(sand)
    for k in range(0, s, 4):
        sd.line([(0, k), (s, k)], fill=(138, 112, 74), width=1)
        sd.line([(k, 0), (k, s)], fill=(166, 138, 94), width=1)
    arr = np.clip(np.array(sand, dtype=np.int16) + rng.integers(-6, 7, (s, s, 1)), 0, 255).astype(np.uint8)
    sand = Image.fromarray(arr, "RGB")

    indigo = Image.new("RGB", (s, s), (28, 48, 78))
    iarr = np.clip(np.array(indigo, dtype=np.int16) + rng.integers(-5, 6, (s, s, 1)), 0, 255).astype(np.uint8)
    indigo = Image.fromarray(iarr, "RGB")

    leather = Image.new("RGB", (s, s), (68, 40, 24))
    larr = np.clip(np.array(leather, dtype=np.int16) + rng.integers(-7, 8, (s, s, 1)), 0, 255).astype(np.uint8)
    leather = Image.fromarray(larr, "RGB")

    return {
        "sand_albedo": _img_to_png_bytes(sand),
        "indigo_albedo": _img_to_png_bytes(indigo),
        "leather_albedo": _img_to_png_bytes(leather),
    }


DUNE_MAT_SPECS = [
    ("sand", {"name": "Dune_SandLinen", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.02, "roughnessFactor": 0.90}, "doubleSided": True}),
    ("indigo", {"name": "Dune_Indigo", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.04, "roughnessFactor": 0.86}, "doubleSided": True}),
    ("vermilion", {"name": "Dune_Vermilion", "pbrMetallicRoughness": {"baseColorFactor": [0.42, 0.10, 0.10, 1.0], "metallicFactor": 0.04, "roughnessFactor": 0.84}, "doubleSided": True}),
    ("leather", {"name": "Dune_Leather", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}, "metallicFactor": 0.12, "roughnessFactor": 0.68}, "doubleSided": True}),
    ("brass", {"name": "Dune_Brass", "pbrMetallicRoughness": {"baseColorFactor": [0.56, 0.44, 0.22, 1.0], "metallicFactor": 0.82, "roughnessFactor": 0.34}, "doubleSided": True}),
    ("steel", {"name": "Dune_Steel", "pbrMetallicRoughness": {"baseColorFactor": [0.64, 0.68, 0.74, 1.0], "metallicFactor": 0.90, "roughnessFactor": 0.24}, "doubleSided": True}),
    ("turq", {"name": "Dune_Turquoise", "pbrMetallicRoughness": {"baseColorFactor": [0.18, 0.83, 0.75, 1.0], "metallicFactor": 0.12, "roughnessFactor": 0.16}, "emissiveFactor": [0.12, 0.72, 0.64], "doubleSided": True}),
]
DUNE_KEYS = ("sand", "indigo", "vermilion", "leather", "brass", "steel", "turq")


def _dp() -> Dict[str, MeshPart]:
    return _parts(*DUNE_KEYS)


def build_dune_helmet() -> Dict[str, Dict[str, MeshPart]]:
    """Capucha picuda de asesino nómada + cuernos curvos de latón + semivelo bajo ojos (rostro superior 100% libre)."""
    out = {"Head": _dp()}
    h = out["Head"]
    # Capucha en arco picudo que enmarca la cara por encima de las cejas
    h["indigo"].add(*make_curved_shell(
        0.155, 0.275, 0.128, 0.138, 0.072, 0.110,
        -math.pi * 0.96, math.pi * 0.96, thickness=0.010, ny=12, na=24, peak_front_y=0.030
    ))
    h["sand"].add(*make_curved_shell(
        -0.06, 0.215, 0.132, 0.142, 0.126, 0.136,
        math.pi * 0.45, math.pi * 1.55, thickness=0.009, ny=12, na=18, flare_bot=0.042, tatter_waves=4, tatter_amp=0.032
    ))
    # Cola picuda trasera de la capucha
    _horn(h["indigo"], [(0, 0.245, -0.05), (0, 0.265, -0.145), (0, 0.235, -0.225)], 0.048)
    # Semivelo inferior (por debajo de nariz/ojos, y = -0.02..0.068) dejando ojos y cejas 100% visibles
    h["vermilion"].add(*make_curved_shell(
        -0.02, 0.068, 0.118, 0.126, 0.120, 0.128,
        -1.35, 1.35, thickness=0.008, ny=8, na=18, peak_front_y=0.016
    ))
    # Diadema con joya turquesa sobre el entrecejo y dos cuernos laterales de latón
    h["brass"].add(*make_curved_shell(0.168, 0.196, 0.128, 0.136, 0.128, 0.136, -1.15, 1.15, thickness=0.007, peak_front_y=-0.018))
    h["turq"].add(*make_sphere(0.015, segs=8, rings=6, center=(0, 0.180, 0.142), scale=(0.8, 1.3, 0.6)))
    for sgn in (-1, 1):
        _horn(h["brass"], [(sgn * 0.095, 0.195, 0.04), (sgn * 0.155, 0.255, -0.01), (sgn * 0.150, 0.335, -0.08)], 0.016)
    return out


def build_dune_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    out = {"UpperArm_L": _dp(), "Torso": _dp(), "UpperArm_R": _dp()}
    tr = out["Torso"]
    # Manto asimétrico del desierto cayendo por hombro izquierdo y espalda en picos rasgados
    tr["sand"].add(*make_curved_shell(
        -0.18, 0.28, 0.225, 0.185, 0.155, 0.140,
        -0.05, math.pi * 1.28, thickness=0.009, ny=12, na=22, flare_bot=0.055, tatter_waves=5, tatter_amp=0.048
    ))
    tr["vermilion"].add(*make_curved_shell(
        0.21, 0.30, 0.134, 0.132, 0.148, 0.142,
        -math.pi, math.pi, thickness=0.011, ny=6, na=22, flare_top=0.016
    ))
    # Izquierdo: hombrera alada de halcón en latón con 2 cuernos curvos
    ul = out["UpperArm_L"]
    ul["leather"].add(*make_curved_shell(-0.04, 0.06, 0.092, 0.092, 0.082, 0.082, -0.35, 2.75, thickness=0.010, flare_bot=0.018))
    ul["brass"].add(*make_curved_shell(0.01, 0.08, 0.100, 0.100, 0.086, 0.086, -0.2, 2.4, thickness=0.009, peak_front_y=0.022))
    _horn(ul["brass"], [(0.055, 0.065, 0.01), (0.125, 0.135, -0.01), (0.165, 0.215, -0.04)], 0.017)
    _horn(ul["brass"], [(0.080, 0.035, -0.02), (0.145, 0.085, -0.04), (0.185, 0.135, -0.07)], 0.012)
    # Derecho: arnés ligero y pañuelo
    ur = out["UpperArm_R"]
    ur["indigo"].add(*make_curved_shell(-0.045, 0.015, 0.086, 0.086, 0.082, 0.082, -2.6, -0.3, thickness=0.008))
    return out


def build_dune_torso() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _dp()}
    t = out["Torso"]
    # Camisa plisada arena + chaleco solapa V abierto índigo + cuello viento bermellón dentado
    t["sand"].add(*make_curved_shell(-0.16, 0.27, 0.162, 0.124, 0.208, 0.148, -math.pi, math.pi, thickness=0.009))
    t["indigo"].add(*make_curved_shell(-0.17, 0.28, 0.172, 0.132, 0.220, 0.156, 0.34, math.tau - 0.34, thickness=0.010, flare_bot=0.016))
    t["vermilion"].add(*make_curved_shell(
        0.24, 0.36, 0.118, 0.116, 0.150, 0.138,
        0.52, math.tau - 0.52, thickness=0.010, flare_top=0.028, tatter_waves=5, tatter_amp=0.018
    ))
    # Bandolera diagonal con 3 cuchillos Jambiya biselados
    t["leather"].add(*make_tube_curve([(-0.155, 0.24, 0.145), (-0.03, 0.08, 0.170), (0.15, -0.14, 0.132)], [0.020, 0.020, 0.018], segs=8))
    for i, (dx, dy) in enumerate(((-0.08, 0.14), (-0.01, 0.06), (0.06, -0.02))):
        dag_pts = [(0.0, -0.045), (0.012, 0.01), (0.0, 0.065), (-0.012, 0.01)]
        t["steel"].add(*make_extruded_polygon(dag_pts, 0.008, bevel=0.002, center=(dx, dy, 0.176), rot=(0, 0, -0.65)))
        t["brass"].add(*make_box(0.024, 0.014, 0.016, center=(dx - 0.02, dy + 0.02, 0.178), rot=(0, 0, -0.65)))
    for sgn in (-1, 1):
        t["brass"].add(*make_tube_curve([(sgn * 0.05, 0.05, 0.165), (sgn * 0.155, 0.07, 0.105), (sgn * 0.175, 0.08, -0.02)], [0.008, 0.009, 0.007], segs=6))
    return out


def build_dune_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Forearm_R": _bp() if False else _dp(), "Wrist_R": _dp(), "Forearm_L": _dp(), "Wrist_L": _dp()}
    for sd, sgn in (("R", -1), ("L", 1)):
        fa, wr = out["Forearm_" + sd], out["Wrist_" + sd]
        fa["leather"].add(*make_curved_shell(-0.245, -0.03, 0.052, 0.052, 0.066, 0.066, -math.pi, math.pi, thickness=0.008, flare_top=0.010))
        fa["indigo"].add(*make_curved_shell(-0.250, -0.185, 0.058, 0.058, 0.060, 0.060, -math.pi, math.pi, thickness=0.007))
        if sgn > 0:
            fin_pts = [(0.0, -0.04), (0.058, 0.01), (0.038, 0.082), (0.0, 0.04)]
            fa["steel"].add(*make_extruded_polygon(fin_pts, 0.010, bevel=0.003, center=(0.042, -0.04, -0.015)))
            fa["brass"].add(*make_cylinder(0.025, 0.025, 0.014, segs=12, center=(0.054, -0.20, 0.0), rot=(0, 0, math.pi / 2)))
            fa["turq"].add(*make_sphere(0.012, segs=8, rings=6, center=(0.062, -0.20, 0.0)))
        else:
            wr["brass"].add(*make_curved_shell(-0.072, 0.005, 0.046, 0.040, 0.048, 0.042, -math.pi, math.pi, thickness=0.006))
            for kz in (-0.018, 0.018):
                _horn(wr["steel"], [(-0.036, -0.035, kz), (-0.062, -0.055, kz), (-0.082, -0.095, kz)], 0.007, segs=5)
    return out


def build_dune_belt() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Pelvis": _dp()}
    p = out["Pelvis"]
    p["vermilion"].add(*make_curved_shell(-0.08, 0.05, 0.174, 0.138, 0.178, 0.142, -math.pi, math.pi, thickness=0.012))
    p["indigo"].add(*make_curved_shell(-0.04, 0.02, 0.180, 0.144, 0.182, 0.146, -math.pi, math.pi, thickness=0.010))
    clasp_pts = [(0.0, -0.055), (0.062, 0.0), (0.045, 0.045), (0.0, 0.025), (-0.045, 0.045), (-0.062, 0.0)]
    p["brass"].add(*make_extruded_polygon(clasp_pts, 0.016, bevel=0.004, center=(0, -0.008, 0.152)))
    p["turq"].add(*make_sphere(0.016, segs=8, rings=6, center=(0, -0.008, 0.162)))
    # Faldón partido nómada abierto al frente en girones afilados
    p["indigo"].add(*make_curved_shell(
        -0.64, -0.04, 0.252, 0.215, 0.178, 0.142,
        0.44, math.tau - 0.44, thickness=0.008, ny=12, na=28, flare_bot=0.055, tatter_waves=5, tatter_amp=0.062
    ))
    p["vermilion"].add(*make_curved_shell(
        -0.56, -0.04, 0.232, 0.198, 0.174, 0.138,
        0.48, 1.05, thickness=0.006, ny=10, na=10, tatter_waves=3, tatter_amp=0.04
    ))
    return out


def build_dune_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _dp()
        for row in range(2):
            yc = -0.055 - row * 0.088
            scale_pts = [(0.0, -0.062), (0.054, 0.0), (0.045, 0.042), (-0.045, 0.042), (-0.054, 0.0)]
            p["indigo" if row == 0 else "leather"].add(*make_extruded_polygon(
                scale_pts, 0.012, bevel=0.004,
                center=(sgn * 0.114, yc, 0.015),
                rot=(0.0, sgn * math.pi * 0.42, sgn * -0.14)
            ))
        if sgn > 0:
            # Cuchillos arrojadizos curvos Jambiya
            for k in (-1, 1):
                dag_pts = [(0.0, -0.06), (0.014, 0.01), (0.022, 0.07), (0.0, 0.02)]
                p["steel"].add(*make_extruded_polygon(dag_pts, 0.008, bevel=0.002, center=(0.122, -0.14, k * 0.032), rot=(0.18 * k, 0, -0.1)))
        else:
            p["brass"].add(*make_cylinder(0.044, 0.044, 0.032, segs=14, center=(-0.126, -0.08, -0.02), rot=(0, 0, math.pi / 2)))
        out[side] = p
    return out


def build_dune_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _dp()
        p["sand"].add(*make_curved_shell(-0.38, -0.06, 0.094, 0.094, 0.116, 0.116, -math.pi, math.pi, thickness=0.009, peak_front_y=-0.015))
        p["brass"].add(*make_tube_curve([(sgn * 0.06, -0.12, 0.08), (sgn * 0.11, -0.22, 0.01), (sgn * 0.07, -0.32, -0.05)], [0.007, 0.009, 0.007], segs=6))
        out[side] = p
    return out


def build_dune_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _dp(), "Knee_R": _dp()}
    for side, sgn in (("Knee_L", 1), ("Knee_R", -1)):
        k = out[side]
        k["leather"].add(*make_curved_shell(-0.044, 0.044, 0.084, 0.084, 0.084, 0.084, -1.6, 1.6, thickness=0.008))
        wing_pts = [(0.0, -0.058), (0.052, -0.01), (0.064, 0.035), (0.0, 0.062), (-0.048, 0.025), (-0.042, -0.02)]
        k["brass"].add(*make_extruded_polygon(wing_pts, 0.014, bevel=0.004, center=(sgn * 0.01, 0, 0.084), rot=(0, sgn * 0.18, 0)))
        k["turq"].add(*make_sphere(0.016, segs=8, rings=6, center=(0, 0, 0.094)))
    return out


def build_dune_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side in ("L", "R"):
        sh, ak = _dp(), _dp()
        sh["leather"].add(*make_curved_shell(-0.44, -0.06, 0.068, 0.070, 0.090, 0.094, -math.pi, math.pi, thickness=0.009, peak_front_y=0.022))
        sh["vermilion"].add(*make_curved_shell(-0.15, -0.05, 0.090, 0.094, 0.102, 0.106, -math.pi, math.pi, thickness=0.008, flare_top=0.016, tatter_waves=4, tatter_amp=0.014))
        _boot_foot({"leather": ak["leather"], "brass": ak["brass"], "indigo": ak["indigo"]},
                   "indigo", "leather", "brass", "brass", foot_w=0.100, toe_len=0.172, pointed=0.35)
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_dune_weapon() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_R": _dp()}
    w = out["Wrist_R"]
    gy = -0.035
    w["leather"].add(*make_cylinder(0.015, 0.016, 0.23, segs=12, center=(0, gy, 0.0), rot=(math.pi / 2, 0, 0)))
    _horn(w["brass"], [(0, gy, -0.115), (0.026, gy, -0.155), (0.052, gy, -0.140)], 0.016)
    w["vermilion"].add(*make_curved_shell(-0.36, -0.15, 0.022, 0.022, 0.012, 0.012, -0.4, 0.4, thickness=0.004, center=(0.03, gy, 0.0), rot=(math.pi / 2, 0, 0)))
    guard_pts = [(-0.105, 0.16), (0.0, 0.105), (0.125, 0.18), (0.0, 0.155)]
    w["brass"].add(*make_extruded_polygon(guard_pts, 0.018, bevel=0.005, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    w["turq"].add(*make_sphere(0.018, segs=8, rings=6, center=(0, gy + 0.015, 0.135)))
    scim_pts = [(-0.016, 0.13), (0.042, 0.50), (0.140, 0.92), (0.062, 1.12), (0.072, 0.85), (0.025, 0.52), (-0.036, 0.13)]
    w["steel"].add(*make_extruded_polygon(scim_pts, 0.010, bevel=0.004, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    w["turq"].add(*make_tube_curve([(-0.020, gy, 0.18), (0.018, gy, 0.48), (0.072, gy, 0.76), (0.095, gy, 0.96)], [0.0045, 0.0045, 0.004, 0.003], segs=6))
    return out


def build_dune_shield() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_L": _dp()}
    s = out["Wrist_L"]
    gy = -0.035
    s["leather"].add(*make_cylinder(0.015, 0.015, 0.18, segs=10, center=(0, gy, 0), rot=(math.pi / 2, 0, 0)))
    _horn(s["brass"], [(0, gy, -0.09), (0.018, gy, -0.13), (0.038, gy, -0.115)], 0.014)
    g_pts = [(-0.07, 0.11), (0.0, 0.075), (0.07, 0.11), (0.0, 0.095)]
    s["brass"].add(*make_extruded_polygon(g_pts, 0.014, bevel=0.003, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    jam_pts = [(-0.016, 0.10), (-0.032, 0.25), (0.008, 0.37), (0.055, 0.48), (0.015, 0.34), (0.018, 0.10)]
    s["steel"].add(*make_extruded_polygon(jam_pts, 0.009, bevel=0.003, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    # Broquel estrella de 8 puntas en dorso de mano
    star_pts = []
    for i in range(8):
        ang = (i / 8) * math.tau
        r = 0.145 if i % 2 == 0 else 0.102
        star_pts.append((math.cos(ang) * r, math.sin(ang) * r * 1.12))
    s["leather"].add(*make_extruded_polygon(star_pts, 0.016, bevel=0.004, center=(0.044, gy + 0.02, 0.02), rot=(0, math.pi / 2, 0)))
    s["brass"].add(*make_torus(0.118, 0.009, 20, 6, center=(0.054, gy + 0.02, 0.02), rot=(0, 0, math.pi / 2)))
    s["turq"].add(*make_sphere(0.032, segs=8, rings=6, center=(0.062, gy + 0.02, 0.02), scale=(0.55, 1.2, 1.0)))
    return out


# ==============================================================================
# ARQUETIPO 3 (MAGO): TEJEDOR DEL ASTROLABIO & ECLIPSE (ROSTRO ABIERTO S-RANK)
# ==============================================================================
def gen_astrolabe_textures() -> Dict[str, bytes]:
    rng = np.random.default_rng(303)
    s = 128
    velvet = Image.new("RGB", (s, s), (16, 22, 46))
    vd = ImageDraw.Draw(velvet)
    for _ in range(48):
        sx, sy = int(rng.integers(0, s)), int(rng.integers(0, s))
        vd.point((sx, sy), fill=(195, 175, 245))
    arr = np.clip(np.array(velvet, dtype=np.int16) + rng.integers(-4, 5, (s, s, 1)), 0, 255).astype(np.uint8)
    velvet = Image.fromarray(arr, "RGB")

    silk = Image.new("RGB", (s, s), (98, 108, 134))
    sarr = np.clip(np.array(silk, dtype=np.int16) + rng.integers(-5, 6, (s, s, 1)), 0, 255).astype(np.uint8)
    silk = Image.fromarray(sarr, "RGB")

    bronze = Image.new("RGB", (s, s), (124, 98, 58))
    bdraw = ImageDraw.Draw(bronze)
    for k in range(16, s, 32):
        bdraw.ellipse([k - 10, 20, k + 10, 40], outline=(170, 140, 88), width=1)

    return {
        "velvet_albedo": _img_to_png_bytes(velvet),
        "silk_albedo": _img_to_png_bytes(silk),
        "bronze_albedo": _img_to_png_bytes(bronze),
    }


ASTRO_MAT_SPECS = [
    ("velvet", {"name": "Astro_Velvet", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.08, "roughnessFactor": 0.84}, "doubleSided": True}),
    ("silk", {"name": "Astro_Silk", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.22, "roughnessFactor": 0.56}, "doubleSided": True}),
    ("bronze", {"name": "Astro_Bronze", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}, "metallicFactor": 0.85, "roughnessFactor": 0.30}, "doubleSided": True}),
    ("silver", {"name": "Astro_Silver", "pbrMetallicRoughness": {"baseColorFactor": [0.58, 0.62, 0.72, 1.0], "metallicFactor": 0.90, "roughnessFactor": 0.22}, "doubleSided": True}),
    ("parchment", {"name": "Astro_Parchment", "pbrMetallicRoughness": {"baseColorFactor": [0.56, 0.50, 0.40, 1.0], "metallicFactor": 0.02, "roughnessFactor": 0.78}, "doubleSided": True}),
    ("astral", {"name": "Astro_VioletGlow", "pbrMetallicRoughness": {"baseColorFactor": [0.65, 0.54, 0.98, 1.0], "metallicFactor": 0.12, "roughnessFactor": 0.12}, "emissiveFactor": [0.58, 0.42, 0.98], "doubleSided": True}),
    ("amber", {"name": "Astro_StarAmber", "pbrMetallicRoughness": {"baseColorFactor": [0.96, 0.62, 0.08, 1.0], "metallicFactor": 0.15, "roughnessFactor": 0.16}, "emissiveFactor": [0.85, 0.50, 0.06], "doubleSided": True}),
]
ASTRO_KEYS = ("velvet", "silk", "bronze", "silver", "parchment", "astral", "amber")


def _ap() -> Dict[str, MeshPart]:
    return _parts(*ASTRO_KEYS)


def build_astro_helmet() -> Dict[str, Dict[str, MeshPart]]:
    """Corona catedralicia abierta con grandes cuernos crecientes asimétricos y halo roto (100% rostro abierto)."""
    out = {"Head": _ap()}
    h = out["Head"]
    h["bronze"].add(*make_curved_shell(
        0.168, 0.215, 0.124, 0.132, 0.128, 0.136,
        -math.pi * 0.92, math.pi * 0.92, thickness=0.008, ny=6, na=22, peak_front_y=-0.025
    ))
    h["astral"].add(*make_sphere(0.017, segs=8, rings=6, center=(0, 0.195, 0.140), scale=(0.8, 1.35, 0.6)))
    # Dos grandes cuernos crecientes asimétricos de bronce y plata en los temporales
    _horn(h["bronze"], [(0.095, 0.195, 0.02), (0.165, 0.285, -0.02), (0.155, 0.395, -0.09)], 0.020)
    _horn(h["silver"], [(-0.092, 0.195, 0.02), (-0.155, 0.265, -0.01), (-0.142, 0.355, -0.07)], 0.016)
    # Halo celestial roto sobre la coronilla + lunas en las orejas
    h["silver"].add(*make_torus(0.175, 0.007, 24, 6, arc=math.tau * 0.72, center=(0, 0.275, -0.03), rot=(0.45, 0.0, 0.35)))
    for sgn in (-1, 1):
        _horn(h["bronze"], [(sgn * 0.105, 0.09, 0.0), (sgn * 0.128, 0.135, -0.02), (sgn * 0.105, 0.18, 0.01)], 0.008, segs=6)
    return out


def build_astro_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    """Gran Halo Astrolabio dorsal con 4 agujas aladas + hombrera de media luna izquierda + chal derecho de seda."""
    out = {"Torso": _ap(), "UpperArm_L": _ap(), "UpperArm_R": _ap()}
    tr = out["Torso"]
    cx, cy, cz = 0.0, 0.24, -0.24
    tr["bronze"].add(*make_torus(0.32, 0.012, 32, 8, center=(cx, cy, cz), rot=(math.pi / 2 + 0.12, 0.0, 0.08)))
    tr["silver"].add(*make_torus(0.24, 0.009, 28, 6, center=(cx, cy, cz), rot=(math.pi / 2 + 0.35, 0.42, -0.25)))
    tr["amber"].add(*make_torus(0.16, 0.007, 24, 6, center=(cx, cy, cz), rot=(math.pi / 2 - 0.35, -0.5, 0.32)))
    # 4 agujas celestiales emergiendo del halo dorsal
    for sx, sy0, sx1, sy1 in ((0.18, 0.36, 0.38, 0.62), (-0.18, 0.36, -0.38, 0.62), (0.24, 0.14, 0.44, 0.05), (-0.24, 0.14, -0.44, 0.05)):
        _horn(tr["bronze"], [(sx, sy0, cz), ((sx + sx1) * 0.5, (sy0 + sy1) * 0.5, cz - 0.03), (sx1, sy1, cz - 0.01)], 0.016, segs=6)
    for s in range(4):
        ang = (s / 4) * math.tau + 0.35
        tr["astral"].add(*make_sphere(0.022, segs=8, rings=6, center=(math.cos(ang) * 0.32, cy + math.sin(ang) * 0.32, cz)))
    # Manto asimétrico izquierdo de terciopelo y chal derecho de seda
    tr["velvet"].add(*make_curved_shell(
        -0.24, 0.28, 0.218, 0.175, 0.155, 0.140,
        0.1, math.pi * 1.25, thickness=0.009, ny=12, na=20, flare_bot=0.055, tatter_waves=4, tatter_amp=0.045
    ))
    tr["silk"].add(*make_curved_shell(
        -0.16, 0.28, 0.212, 0.172, 0.152, 0.138,
        math.pi * 0.85, math.tau - 0.15, thickness=0.007, ny=10, na=16, flare_bot=0.042, tatter_waves=3, tatter_amp=0.035
    ))
    ul = out["UpperArm_L"]
    ul["bronze"].add(*make_curved_shell(-0.02, 0.08, 0.098, 0.098, 0.084, 0.084, -0.3, 2.5, thickness=0.009, peak_front_y=0.022))
    _horn(ul["bronze"], [(0.055, 0.075, 0.0), (0.125, 0.165, -0.02), (0.145, 0.275, -0.05)], 0.018)
    ur = out["UpperArm_R"]
    ur["silk"].add(*make_curved_shell(-0.04, 0.04, 0.090, 0.090, 0.084, 0.084, -2.5, -0.2, thickness=0.008))
    return out


def build_astro_torso() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _ap()}
    t = out["Torso"]
    t["silk"].add(*make_curved_shell(-0.16, 0.27, 0.164, 0.128, 0.210, 0.150, -math.pi, math.pi, thickness=0.009))
    # Abrigo V abierto de terciopelo con cuello catedralicio dentado
    t["velvet"].add(*make_curved_shell(-0.17, 0.28, 0.172, 0.134, 0.220, 0.158, 0.36, math.tau - 0.36, thickness=0.011, flare_bot=0.018))
    t["velvet"].add(*make_curved_shell(
        0.24, 0.40, 0.116, 0.114, 0.162, 0.146,
        0.55, math.tau - 0.55, thickness=0.010, flare_top=0.034, tatter_waves=5, tatter_amp=0.020
    ))
    for sgn in (-1, 1):
        t["bronze"].add(*make_tube_curve([(sgn * 0.045, 0.08, 0.168), (sgn * 0.155, 0.12, 0.105), (sgn * 0.178, 0.14, -0.02)], [0.008, 0.009, 0.007], segs=6))
    t["bronze"].add(*make_torus(0.052, 0.009, 18, 6, center=(0.0, 0.145, 0.172), rot=(math.pi / 2, 0, 0)))
    t["astral"].add(*make_sphere(0.022, segs=8, rings=6, center=(0.0, 0.145, 0.175)))
    return out


def build_astro_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Forearm_R": _ap(), "Wrist_R": _ap(), "Forearm_L": _ap(), "Wrist_L": _ap()}
    for sd, sgn in (("R", -1), ("L", 1)):
        fa, wr = out["Forearm_" + sd], out["Wrist_" + sd]
        fa["velvet"].add(*make_curved_shell(
            -0.25, -0.02, 0.116, 0.116, 0.058, 0.058,
            -math.pi, math.pi, thickness=0.008, flare_bot=0.034, tatter_waves=4, tatter_amp=0.022
        ))
        fa["silk"].add(*make_curved_shell(-0.245, -0.19, 0.112, 0.112, 0.098, 0.098, -math.pi, math.pi, thickness=0.006))
        fa["bronze"].add(*make_torus(0.118, 0.006, 18, 6, center=(0, -0.22, 0)))
        wr["astral"].add(*make_sphere(0.012, segs=6, rings=5, center=(sgn * 0.026, -0.025, 0)))
    return out


def build_astro_belt() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Pelvis": _ap()}
    p = out["Pelvis"]
    p["silk"].add(*make_curved_shell(-0.07, 0.05, 0.175, 0.138, 0.178, 0.142, -math.pi, math.pi, thickness=0.012, peak_front_y=-0.022))
    clasp_pts = [(0.0, -0.052), (0.052, 0.0), (0.035, 0.045), (0.0, 0.025), (-0.035, 0.045), (-0.052, 0.0)]
    p["bronze"].add(*make_extruded_polygon(clasp_pts, 0.014, bevel=0.004, center=(0, -0.01, 0.152)))
    p["astral"].add(*make_sphere(0.016, segs=8, rings=6, center=(0, -0.01, 0.162)))
    # Faldón ceremonial de 4 colas de golondrina catedralicias
    for a0, a1 in ((-0.42, 0.42), (0.62, 1.98), (math.pi * 0.66, math.pi * 1.34), (math.tau - 1.98, math.tau - 0.62)):
        p["velvet"].add(*make_curved_shell(
            -0.70, -0.04, 0.268, 0.232, 0.178, 0.142,
            a0, a1, thickness=0.009, ny=12, na=12, flare_bot=0.068, tatter_waves=3, tatter_amp=0.048
        ))
        p["silk"].add(*make_curved_shell(-0.70, -0.63, 0.270, 0.234, 0.262, 0.228, a0 + 0.05, a1 - 0.05, thickness=0.006))
    return out


def build_astro_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _ap()
        scale_pts = [(0.0, -0.065), (0.050, 0.0), (0.042, 0.045), (-0.042, 0.045), (-0.050, 0.0)]
        p["silk"].add(*make_extruded_polygon(scale_pts, 0.010, bevel=0.003, center=(sgn * 0.114, -0.06, 0.012), rot=(0, sgn * math.pi * 0.42, sgn * -0.14)))
        for offz in (-0.032, 0.032):
            p["bronze"].add(*make_cylinder(0.003, 0.003, 0.20, segs=6, center=(sgn * 0.118, -0.15, offz)))
            p["amber"].add(*make_sphere(0.016, segs=8, rings=6, center=(sgn * 0.118, -0.255, offz), scale=(0.8, 1.4, 0.8)))
        p["bronze"].add(*make_torus(0.044, 0.006, 16, 6, center=(sgn * 0.120, -0.15, 0.0), rot=(0, 0, math.pi / 2)))
        p["astral"].add(*make_sphere(0.016, segs=8, rings=6, center=(sgn * 0.122, -0.15, 0.0)))
        out[side] = p
    return out


def build_astro_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _ap()
        p["velvet"].add(*make_curved_shell(-0.38, -0.06, 0.092, 0.092, 0.112, 0.112, -math.pi, math.pi, thickness=0.009, peak_front_y=-0.015))
        p["silk"].add(*make_tube_curve([(sgn * 0.05, -0.10, 0.08), (sgn * 0.106, -0.22, 0.0), (sgn * 0.06, -0.34, -0.05)], [0.007, 0.008, 0.007], segs=6))
        out[side] = p
    return out


def build_astro_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _ap(), "Knee_R": _ap()}
    for side, sgn in (("Knee_L", 1), ("Knee_R", -1)):
        k = out[side]
        cres_pts = [(0.0, -0.058), (0.046, -0.01), (0.055, 0.035), (0.0, 0.062), (-0.044, 0.025)]
        k["bronze"].add(*make_extruded_polygon(cres_pts, 0.012, bevel=0.004, center=(sgn * 0.008, 0, 0.086), rot=(0, sgn * 0.18, 0)))
        _horn(k["silver"], [(sgn * 0.042, 0.015, 0.072), (sgn * 0.078, 0.048, 0.048), (sgn * 0.095, 0.092, 0.015)], 0.009, segs=6)
        k["astral"].add(*make_sphere(0.014, segs=8, rings=6, center=(0, 0, 0.096)))
    return out


def build_astro_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side in ("L", "R"):
        sh, ak = _ap(), _ap()
        sh["velvet"].add(*make_curved_shell(-0.44, -0.12, 0.068, 0.070, 0.088, 0.092, -math.pi, math.pi, thickness=0.008, peak_front_y=0.022))
        sh["silk"].add(*make_curved_shell(-0.18, -0.10, 0.086, 0.090, 0.098, 0.102, -math.pi, math.pi, thickness=0.008, flare_top=0.015, tatter_waves=4, tatter_amp=0.012))
        _boot_foot({"velvet": ak["velvet"], "bronze": ak["bronze"], "silver": ak["silver"]},
                   "silver", "velvet", "bronze", "silver", foot_w=0.096, toe_len=0.170, pointed=0.40)
        ak["amber"].add(*make_sphere(0.012, segs=6, rings=5, center=(0, 0.012, 0.092)))
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_astro_weapon() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_R": _ap()}
    w = out["Wrist_R"]
    gy = -0.035
    w["bronze"].add(*make_tube_curve([
        (0, gy, -0.46), (0.012, gy + 0.008, -0.05), (-0.014, gy - 0.008, 0.42), (0.010, gy + 0.006, 0.85), (0, gy, 1.12)
    ], [0.014, 0.014, 0.014, 0.014, 0.014], segs=10))
    w["velvet"].add(*make_cylinder(0.019, 0.019, 0.24, segs=12, center=(0, gy, 0.01), rot=(math.pi / 2, 0, 0)))
    _horn(w["silver"], [(0, gy, -0.46), (0, gy + 0.012, -0.54), (0, gy, -0.60)], 0.018)
    # Hoja creciente lunar enmarcando el planetario (y -> Z = 1.24)
    cres_pts = [(-0.01, 1.06), (0.22, 1.24), (0.44, 1.18), (0.48, 0.82), (0.34, 1.02), (0.18, 1.08), (0.01, 0.99)]
    w["silver"].add(*make_extruded_polygon(cres_pts, 0.010, bevel=0.004, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    _horn(w["bronze"], [(-0.01, gy, 1.08), (-0.14, gy + 0.02, 1.22), (-0.18, gy - 0.01, 1.38)], 0.015)
    cz = 1.24
    w["bronze"].add(*make_torus(0.118, 0.008, 24, 6, center=(0.04, gy, cz), rot=(math.pi / 2, 0.35, 0)))
    w["silver"].add(*make_torus(0.088, 0.006, 22, 6, center=(0.04, gy, cz), rot=(0.65, -0.5, 0.35)))
    w["amber"].add(*make_torus(0.056, 0.005, 18, 6, center=(0.04, gy, cz), rot=(-0.7, 0.6, 0.8)))
    w["astral"].add(*make_sphere(0.036, segs=10, rings=8, center=(0.04, gy, cz)))
    w["silk"].add(*make_curved_shell(0.62, 1.02, 0.025, 0.025, 0.012, 0.012, -0.4, 0.4, thickness=0.004, center=(0.04, gy, 0), rot=(-math.pi / 2, 0, 0)))
    return out


def build_astro_shield() -> Dict[str, Dict[str, MeshPart]]:
    """Grimorio Catedralicio Abierto & Planetario Flotante en mano izquierda."""
    out = {"Wrist_L": _ap()}
    s = out["Wrist_L"]
    gy = -0.035
    s["bronze"].add(*make_cylinder(0.012, 0.012, 0.14, segs=8, center=(0, gy, 0), rot=(math.pi / 2, 0, 0)))
    s["bronze"].add(*make_cylinder(0.016, 0.016, 0.24, segs=8, center=(0.032, gy + 0.04, 0.0), rot=(math.pi / 2, 0, 0)))
    for sd in (-1, 1):
        s["velvet"].add(*make_box(0.012, 0.125, 0.23, center=(0.048, gy + 0.04, sd * 0.062), rot=(0, -sd * 0.42, 0), bevel=0.004))
        s["parchment"].add(*make_box(0.014, 0.112, 0.20, center=(0.056, gy + 0.04, sd * 0.058), rot=(0, -sd * 0.42, 0)))
        _horn(s["bronze"], [(0.068, gy + 0.15, sd * 0.11), (0.078, gy + 0.21, sd * 0.14), (0.068, gy + 0.25, sd * 0.10)], 0.008)
    ox, oy, oz = 0.148, gy + 0.06, 0.0
    s["astral"].add(*make_sphere(0.032, segs=10, rings=8, center=(ox, oy, oz)))
    s["bronze"].add(*make_torus(0.068, 0.0055, 22, 6, center=(ox, oy, oz), rot=(0.55, 0.4, 0.2)))
    s["amber"].add(*make_torus(0.050, 0.0045, 18, 6, center=(ox, oy, oz), rot=(-0.6, 0.8, 0.5)))
    _horn(s["silver"], [(ox, oy + 0.12, -0.055), (ox + 0.008, oy + 0.16, 0.0), (ox, oy + 0.12, 0.055)], 0.009)
    return out


# ==============================================================================
# CATÁLOGOS
# ==============================================================================
NEW_CATALOG_ENTRIES = {
    "underlayer": [
        {"idx": 18, "name": "Hulla & Remaches de Fragua (Basalto)"},
        {"idx": 19, "name": "Lino Arena & Faja Índigo de Nómada (Dunas)"},
        {"idx": 20, "name": "Túnica Terciopelo Astral & Hilo Plateado (Astrolabio)"},
    ],
    "helmet": [
        {"idx": 11, "name": "Yelmo Rejilla de Caldera & Cresta Yunque (Basalto)"},
        {"idx": 12, "name": "Turbante del Desierto & Gafas de Aviador de Latón (Dunas)"},
        {"idx": 13, "name": "Diadema Orbital de Eclipse & Sextante Astral (Astrolabio)"},
    ],
    "shoulders": [
        {"idx": 14, "name": "Hombrera Chimenea Fumante & Yunque Asimétrico (Basalto)"},
        {"idx": 15, "name": "Poncho Asimétrico de Arena & Bufanda Bermellón (Dunas)"},
        {"idx": 16, "name": "Astrolabio Cósmico Giratorio & Estola Dorsal (Astrolabio)"},
    ],
    "torso": [
        {"idx": 22, "name": "Coraza Horno de Basalto con Rejilla de Magma (Pesada)"},
        {"idx": 23, "name": "Túnica Cruzada de Nómada, Bandolera & Poncho (Ligera)"},
        {"idx": 24, "name": "Túnica Ceremonial de Seda, Cuello Plisado & Estola (Mago)"},
    ],
    "gloves": [
        {"idx": 8,  "name": "Guantelete Pistón Hidráulico & Cadena (Basalto)"},
        {"idx": 9,  "name": "Vendas de Cuero, Brújula & Guardamano (Dunas)"},
        {"idx": 10, "name": "Mangas Acampanadas Ceremoniales & Mitones Rúnicos (Astrolabio)"},
    ],
    "belt": [
        {"idx": 10, "name": "Cinturón de Cadenas de Forja & Delantal de Basalto (Basalto)"},
        {"idx": 11, "name": "Faja Multicapa de Tela del Desierto & Cantimplora (Dunas)"},
        {"idx": 12, "name": "Cíngulo Astral & Faldón Ceremonial de 4 Paneles (Astrolabio)"},
    ],
    "tassets": [
        {"idx": 11, "name": "Lozas Hexagonales de Basalto & Escoria Ígnea (Basalto)"},
        {"idx": 12, "name": "Faldones Abiertos de Lino & Vainas de Caza (Dunas)"},
        {"idx": 13, "name": "Láminas Colgantes de Péndulos & Relojes Estelares (Astrolabio)"},
    ],
    "cuisses": [
        {"idx": 10, "name": "Musleras Pistón de Hierro Colado & Basalto (Basalto)"},
        {"idx": 11, "name": "Bombachos de Lino Sujetos con Correas & Mapa (Dunas)"},
        {"idx": 12, "name": "Calzas de Terciopelo Bordado con Borde Plateado (Astrolabio)"},
    ],
    "knees": [
        {"idx": 10, "name": "Rodilleras Yunque de Hierro & Escudo Caldera (Basalto)"},
        {"idx": 11, "name": "Rodilleras Ligeras de Cuero con Hebilla de Latón (Dunas)"},
        {"idx": 12, "name": "Rodilleras Filigrana Astral de Lunas Crecientes (Astrolabio)"},
    ],
    "boots": [
        {"idx": 15, "name": "Botas Pilar de Hierro Fundido & Escoria (Basalto)"},
        {"idx": 16, "name": "Botas Bajas Curvas de Nómada con Polainas (Dunas)"},
        {"idx": 17, "name": "Chapines Ceremoniales Puntiagudos de Mago (Astrolabio)"},
    ],
    "weapon": [
        {"idx": 83, "name": "Martillo Yunque de Fragua & Magma (Basalto)", "familia": "maza", "nivel": 50},
        {"idx": 84, "name": "Cimitarra Creciente de Dunas (Dunas)", "familia": "espada", "nivel": 50},
        {"idx": 85, "name": "Báculo Orbital de Astrolabio Cósmico (Astrolabio)", "familia": "cetro", "nivel": 50},
    ],
    "shield": [
        {"idx": 13, "name": "Escudo Puerta de Horno con Mirilla de Magma (Basalto)", "nivel": 50},
        {"idx": 14, "name": "Daga Parada & Rodela Estrella de Latón (Dunas)", "nivel": 50},
        {"idx": 15, "name": "Grimorio Arcano Abierto & Orbe Flotante (Astrolabio)", "nivel": 50},
    ],
}


def update_catalogo_estudio() -> None:
    with open(CATALOGO_ESTUDIO_PATH, encoding="utf-8") as f:
        estudio = json.load(f)
    for ranura, entries in NEW_CATALOG_ENTRIES.items():
        lst = estudio.setdefault(ranura, [])
        for entry in entries:
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
    packs = [
        (
            "Coloso de Basalto & Fragua (Pesada)",
            gen_basalt_textures(),
            ("basalt_albedo", "iron_albedo", "copper_albedo"),
            BASALT_MAT_SPECS,
            "BasaltColossus",
            "Basalt",
            {
                ("helmet", 11): build_basalt_helmet,
                ("shoulders", 14): build_basalt_shoulders,
                ("torso", 22): build_basalt_torso,
                ("gloves", 8): build_basalt_gloves,
                ("belt", 10): build_basalt_belt,
                ("tassets", 11): build_basalt_tassets,
                ("cuisses", 10): build_basalt_cuisses,
                ("knees", 10): build_basalt_knees,
                ("boots", 15): build_basalt_boots,
                ("weapon", 83): build_basalt_weapon,
                ("shield", 13): build_basalt_shield,
            },
        ),
        (
            "Cazador de Dunas & Tormenta (Ligera)",
            gen_dune_textures(),
            ("sand_albedo", "indigo_albedo", "leather_albedo"),
            DUNE_MAT_SPECS,
            "DuneNomad",
            "Dune",
            {
                ("helmet", 12): build_dune_helmet,
                ("shoulders", 15): build_dune_shoulders,
                ("torso", 23): build_dune_torso,
                ("gloves", 9): build_dune_gloves,
                ("belt", 11): build_dune_belt,
                ("tassets", 12): build_dune_tassets,
                ("cuisses", 11): build_dune_cuisses,
                ("knees", 11): build_dune_knees,
                ("boots", 16): build_dune_boots,
                ("weapon", 84): build_dune_weapon,
                ("shield", 14): build_dune_shield,
            },
        ),
        (
            "Tejedor del Astrolabio & Eclipse (Mago)",
            gen_astrolabe_textures(),
            ("velvet_albedo", "silk_albedo", "bronze_albedo"),
            ASTRO_MAT_SPECS,
            "AstrolabeMage",
            "Astrolabe",
            {
                ("helmet", 13): build_astro_helmet,
                ("shoulders", 16): build_astro_shoulders,
                ("torso", 24): build_astro_torso,
                ("gloves", 10): build_astro_gloves,
                ("belt", 12): build_astro_belt,
                ("tassets", 13): build_astro_tassets,
                ("cuisses", 12): build_astro_cuisses,
                ("knees", 12): build_astro_knees,
                ("boots", 17): build_astro_boots,
                ("weapon", 85): build_astro_weapon,
                ("shield", 15): build_astro_shield,
            },
        ),
    ]

    for title, tex, tex_keys, mat_specs, scene_name, prefix, builders in packs:
        print(f"Generando {title}...")
        for (slot, idx), fn in builders.items():
            out_path = os.path.join(PIEZAS_DIR, slot, f"{idx}.glb")
            stats = write_glb_custom(out_path, fn(), tex, tex_keys, mat_specs, scene_name, prefix)
            print(f"  [OK] {slot}/{idx}.glb -> {stats['verts']} verts, {stats['tris']} tris, {stats['bytes']} bytes")

    update_catalogo_estudio()
    print(f"Actualizado {os.path.relpath(CATALOGO_ESTUDIO_PATH, ROOT_DIR)}")
    subprocess.run([sys.executable, GENERAR_CATALOGO_PATH], check=True)


if __name__ == "__main__":
    main()
