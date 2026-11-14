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

    # Cobre oxidado (patina verdosa)
    copper = Image.new("RGB", (s, s), (54, 114, 102))
    cdraw = ImageDraw.Draw(copper)
    for _ in range(45):
        cx, cy = int(rng.integers(0, s)), int(rng.integers(0, s))
        r = int(rng.integers(4, 11))
        cdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(72, 138, 122))

    return {
        "basalt_albedo": _img_to_png_bytes(basalt),
        "iron_albedo": _img_to_png_bytes(iron),
        "copper_albedo": _img_to_png_bytes(copper),
    }


BASALT_MAT_SPECS = [
    ("basalt", {"name": "Basalt_Stone", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.08, "roughnessFactor": 0.86}, "doubleSided": True}),
    ("iron", {"name": "Basalt_Iron", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.86, "roughnessFactor": 0.44}, "doubleSided": True}),
    ("copper", {"name": "Basalt_Patina", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}, "metallicFactor": 0.58, "roughnessFactor": 0.56}, "doubleSided": True}),
    ("brass", {"name": "Basalt_Brass", "pbrMetallicRoughness": {"baseColorFactor": [0.42, 0.27, 0.16, 1.0], "metallicFactor": 0.78, "roughnessFactor": 0.38}, "doubleSided": True}),
    ("leather", {"name": "Basalt_Leather", "pbrMetallicRoughness": {"baseColorFactor": [0.11, 0.08, 0.07, 1.0], "metallicFactor": 0.06, "roughnessFactor": 0.82}, "doubleSided": True}),
    ("magma", {"name": "Basalt_Magma", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.34, 0.05, 1.0], "metallicFactor": 0.05, "roughnessFactor": 0.14}, "emissiveFactor": [1.0, 0.32, 0.04], "doubleSided": True}),
]
BASALT_KEYS = ("basalt", "iron", "copper", "brass", "leather", "magma")


def _bp() -> Dict[str, MeshPart]:
    return _parts(*BASALT_KEYS)


def build_basalt_helmet() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Head": _bp()}
    h = out["Head"]
    # Caldera-yelmo prismática pesada cubriendo el cráneo + rejilla frontal de horno
    h["basalt"].add(*make_curved_shell(
        0.02, 0.25, 0.126, 0.138, 0.118, 0.145,
        -math.pi * 0.96, math.pi * 0.96, thickness=0.015, flare_top=0.010
    ))
    # Tapa superior hexagonal estilo yunque
    h["iron"].add(*make_cylinder(0.110, 0.128, 0.055, segs=6, center=(0, 0.255, 0.0)))
    # Cresta metálica desplazada asimétricamente a la derecha (-X) + respiradero lateral izquierdo (+X)
    h["brass"].add(*make_box(0.028, 0.095, 0.185, center=(-0.032, 0.278, 0.0), rot=(0.08, 0, -0.14), bevel=0.004))
    h["copper"].add(*make_cylinder(0.018, 0.022, 0.095, segs=6, center=(0.125, 0.165, -0.025), rot=(0, 0, -0.45)))
    # Rejilla frontal de hierro con hendidura de magma incandescente
    h["iron"].add(*make_curved_shell(-0.015, 0.125, 0.132, 0.146, 0.132, 0.146, -0.82, 0.82, thickness=0.012))
    h["magma"].add(*make_box(0.125, 0.016, 0.020, center=(0, 0.115, 0.138), bevel=0.002))
    for bx in (-0.048, -0.016, 0.016, 0.048):
        h["iron"].add(*make_box(0.010, 0.110, 0.014, center=(bx, 0.055, 0.146), bevel=0.001))
    return out


def build_basalt_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    out = {"UpperArm_L": _bp(), "Torso": _bp(), "UpperArm_R": _bp()}
    # IZQUIERDO: Gran caldera + doble chimenea fumante inclinada
    ul = out["UpperArm_L"]
    ul["basalt"].add(*make_cylinder(0.105, 0.122, 0.12, segs=6, center=(0.045, 0.035, 0.0), rot=(0, 0, -0.25)))
    ul["iron"].add(*make_curved_shell(-0.055, 0.035, 0.095, 0.092, 0.112, 0.108, -0.4, 2.8, thickness=0.014))
    for idx_st, (dx, dy, dz, ht) in enumerate(((0.035, 0.145, -0.028, 0.155), (0.075, 0.125, 0.024, 0.120))):
        ul["iron"].add(*make_cylinder(0.024, 0.030, ht, segs=6, center=(dx, dy, dz), rot=(0.12 * idx_st, 0, -0.30)))
        ul["brass"].add(*make_cylinder(0.032, 0.026, 0.024, segs=6, center=(dx + 0.022, dy + ht * 0.45, dz), rot=(0.12 * idx_st, 0, -0.30)))
        ul["magma"].add(*make_cylinder(0.018, 0.018, 0.012, segs=6, center=(dx + 0.024, dy + ht * 0.49, dz), rot=(0.12 * idx_st, 0, -0.30)))

    # DERECHO: Escudo yunque prismático bajo y anguloso con remaches de cobre
    ur = out["UpperArm_R"]
    ur["iron"].add(*make_box(0.145, 0.095, 0.155, center=(-0.048, 0.022, 0.0), rot=(0, 0, 0.35), bevel=0.008))
    ur["copper"].add(*make_box(0.115, 0.022, 0.125, center=(-0.056, 0.068, 0.0), rot=(0, 0, 0.35), bevel=0.003))
    ur["brass"].add(*make_cylinder(0.012, 0.012, 0.022, segs=6, center=(-0.098, 0.042, 0.045), rot=(0, 0, math.pi / 2)))
    ur["brass"].add(*make_cylinder(0.012, 0.012, 0.022, segs=6, center=(-0.098, 0.042, -0.045), rot=(0, 0, math.pi / 2)))

    # DORSAL EN TORSO: Caldera dorsal asimétrica desplazada a izquierda con tubo de escape y brillo de magma
    tr = out["Torso"]
    tr["iron"].add(*make_cylinder(0.088, 0.098, 0.24, segs=6, center=(0.055, 0.095, -0.175), rot=(0.08, 0, -0.10)))
    tr["basalt"].add(*make_cylinder(0.072, 0.078, 0.18, segs=6, center=(-0.052, 0.065, -0.165), rot=(0.08, 0, 0.10)))
    tr["magma"].add(*make_box(0.082, 0.145, 0.024, center=(0.0, 0.095, -0.228), bevel=0.004))
    tr["copper"].add(*make_tube_curve([(0.065, 0.21, -0.175), (0.12, 0.28, -0.135), (0.16, 0.32, -0.06)],
                                      [0.018, 0.016, 0.015], segs=6))
    return out


def build_basalt_torso() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _bp()}
    t = out["Torso"]
    # Peto hexagonal acorazado + jaula frontal de rejilla de hierro con núcleo magma vivo
    t["basalt"].add(*make_curved_shell(-0.16, 0.29, 0.178, 0.144, 0.228, 0.168, -math.pi, math.pi, thickness=0.018))
    t["iron"].add(*make_box(0.185, 0.235, 0.055, center=(0, 0.085, 0.165), bevel=0.010))
    t["magma"].add(*make_cylinder(0.054, 0.054, 0.028, segs=6, center=(0, 0.105, 0.184), rot=(math.pi / 2, 0, 0)))
    for bx in (-0.042, 0.0, 0.042):
        t["iron"].add(*make_box(0.012, 0.145, 0.022, center=(bx, 0.105, 0.198), bevel=0.002))
    # Gorjal de fragua alto asimétrico (protege mandíbula izquierda)
    t["copper"].add(*make_curved_shell(0.25, 0.35, 0.125, 0.122, 0.142, 0.135, -0.2, 2.2, thickness=0.014))
    t["brass"].add(*make_box(0.22, 0.026, 0.20, center=(0, 0.255, 0.02), bevel=0.004))
    return out


def build_basalt_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Forearm_R": _bp(), "Wrist_R": _bp(), "Forearm_L": _bp(), "Wrist_L": _bp()}
    for sd, sgn in (("R", -1), ("L", 1)):
        fa, wr = out["Forearm_" + sd], out["Wrist_" + sd]
        fa["basalt"].add(*make_cylinder(0.068, 0.054, 0.23, segs=6, center=(0, -0.145, 0)))
        # Pistón hidráulico en el lado exterior del antebrazo
        fa["brass"].add(*make_cylinder(0.013, 0.015, 0.21, segs=8, center=(sgn * 0.065, -0.135, -0.01)))
        fa["copper"].add(*make_cylinder(0.020, 0.020, 0.085, segs=6, center=(sgn * 0.065, -0.085, -0.01)))
        # Nudilleras ariete hexagonales
        wr["iron"].add(*make_box(0.042, 0.085, 0.092, center=(sgn * 0.026, -0.045, 0.0), bevel=0.006))
        if sgn > 0:
            # Cadena pesada envuelta en muñeca izquierda
            wr["brass"].add(*make_torus(0.058, 0.009, 14, 6, center=(0, -0.005, 0)))
    return out


def build_basalt_belt() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Pelvis": _bp()}
    p = out["Pelvis"]
    p["iron"].add(*make_curved_shell(-0.07, 0.05, 0.184, 0.148, 0.188, 0.152, -math.pi, math.pi, thickness=0.016))
    p["brass"].add(*make_cylinder(0.048, 0.048, 0.028, segs=6, center=(0, -0.01, 0.162), rot=(math.pi / 2, 0, 0)))
    # Delantal central de herrero + cadenas colgantes laterales
    p["leather"].add(*make_box(0.165, 0.42, 0.022, center=(0, -0.25, 0.142), rot=(-0.08, 0, 0), bevel=0.004))
    p["basalt"].add(*make_box(0.115, 0.145, 0.026, center=(0, -0.22, 0.155), rot=(-0.08, 0, 0), bevel=0.005))
    for sgn in (-1, 1):
        for r in range(3):
            p["brass"].add(*make_torus(0.020, 0.006, 10, 6, center=(sgn * 0.175, -0.095 - r * 0.036, 0.02),
                                       rot=(0, math.pi / 2 if r % 2 == 0 else 0, 0)))
    return out


def build_basalt_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _bp()
        # Losas hexagonales pesadas de basalto superpuestas con bordes de cobre
        for row in range(2):
            yc = -0.065 - row * 0.105
            p["basalt"].add(*make_cylinder(0.068, 0.075, 0.028, segs=6,
                                           center=(sgn * 0.118, yc, 0.012), rot=(0, 0, math.pi * 0.5 - sgn * 0.12)))
            p["copper"].add(*make_box(0.018, 0.082, 0.095, center=(sgn * 0.106, yc, 0.012), bevel=0.003))
        out[side] = p
    return out


def build_basalt_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _bp()
        p["iron"].add(*make_curved_shell(-0.37, -0.07, 0.096, 0.096, 0.115, 0.115, -math.pi, math.pi, thickness=0.014))
        p["basalt"].add(*make_cylinder(0.075, 0.088, 0.22, segs=6, center=(sgn * 0.025, -0.22, 0.045)))
        if sgn < 0:
            p["brass"].add(*make_cylinder(0.014, 0.014, 0.24, segs=8, center=(-0.108, -0.22, 0.0)))
        out[side] = p
    return out


def build_basalt_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _bp(), "Knee_R": _bp()}
    for side, sgn in (("Knee_L", 1), ("Knee_R", -1)):
        k = out[side]
        k["basalt"].add(*make_cylinder(0.065, 0.068, 0.042, segs=6, center=(0, 0.0, 0.082), rot=(math.pi / 2, 0, 0)))
        k["iron"].add(*make_box(0.110, 0.095, 0.036, center=(sgn * 0.018, 0.01, 0.086), rot=(0, sgn * 0.2, 0), bevel=0.006))
        if sgn > 0:
            k["magma"].add(*make_cylinder(0.026, 0.026, 0.022, segs=6, center=(0, 0, 0.104), rot=(math.pi / 2, 0, 0)))
    return out


def build_basalt_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side in ("L", "R"):
        sh, ak = _bp(), _bp()
        sh["iron"].add(*make_cylinder(0.095, 0.082, 0.38, segs=6, center=(0, -0.24, 0.008)))
        sh["basalt"].add(*make_box(0.085, 0.30, 0.042, center=(0, -0.24, 0.075), bevel=0.006))
        sh["copper"].add(*make_curved_shell(-0.12, -0.05, 0.100, 0.104, 0.108, 0.112, -math.pi, math.pi, thickness=0.012))
        ak["basalt"].add(*make_box(0.118, 0.078, 0.245, center=(0, -0.022, 0.055), bevel=0.010))
        ak["iron"].add(*make_box(0.128, 0.028, 0.258, center=(0, -0.056, 0.055), bevel=0.005))
        ak["brass"].add(*make_box(0.122, 0.056, 0.065, center=(0, -0.022, 0.145), bevel=0.008))
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_basalt_weapon() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_R": _bp()}
    w = out["Wrist_R"]
    gy = -0.035
    w["iron"].add(*make_cylinder(0.018, 0.019, 1.38, segs=10, center=(0, gy, 0.28), rot=(math.pi / 2, 0, 0)))
    w["leather"].add(*make_cylinder(0.023, 0.023, 0.26, segs=10, center=(0, gy, 0.0), rot=(math.pi / 2, 0, 0)))
    w["brass"].add(*make_cylinder(0.026, 0.014, 0.09, segs=6, center=(0, gy, -0.44), rot=(math.pi / 2, 0, 0)))
    # Gran cabeza hexagonal de yunque transversal en eje X/Y + anillo magma
    w["basalt"].add(*make_cylinder(0.092, 0.082, 0.34, segs=6, center=(0, gy, 0.90), rot=(0, 0, math.pi / 2)))
    w["magma"].add(*make_cylinder(0.095, 0.095, 0.048, segs=6, center=(0, gy, 0.90), rot=(0, 0, math.pi / 2)))
    w["iron"].add(*make_cylinder(0.104, 0.098, 0.036, segs=6, center=(-0.14, gy, 0.90), rot=(0, 0, math.pi / 2)))
    w["iron"].add(*make_cylinder(0.104, 0.098, 0.036, segs=6, center=(0.14, gy, 0.90), rot=(0, 0, math.pi / 2)))
    w["iron"].add(*make_cylinder(0.002, 0.036, 0.16, segs=6, center=(0, gy, 1.04), rot=(math.pi / 2, 0, 0)))
    return out


def build_basalt_shield() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_L": _bp()}
    s = out["Wrist_L"]
    gy = -0.035
    s["leather"].add(*make_cylinder(0.015, 0.015, 0.18, segs=10, center=(0, gy, 0.0), rot=(math.pi / 2, 0, 0)))
    # Compuerta de horno hexagonal en dorso de mano izquierda (+X)
    s["basalt"].add(*make_cylinder(0.24, 0.21, 0.038, segs=6, center=(0.055, gy + 0.04, 0.05),
                                   rot=(0, 0, math.pi / 2), scale=(1.45, 1.0, 0.88)))
    s["iron"].add(*make_cylinder(0.252, 0.222, 0.024, segs=6, center=(0.044, gy + 0.04, 0.05),
                                 rot=(0, 0, math.pi / 2), scale=(1.46, 1.0, 0.89)))
    s["magma"].add(*make_cylinder(0.072, 0.072, 0.026, segs=6, center=(0.072, gy + 0.05, 0.05), rot=(0, 0, math.pi / 2)))
    for dz in (-0.042, 0.0, 0.042):
        s["iron"].add(*make_box(0.018, 0.175, 0.014, center=(0.086, gy + 0.05, 0.05 + dz)))
    return out


# ==============================================================================
# ARQUETIPO 2 (LIGERA): CAZADOR DE DUNAS & TORMENTA (ROSTRO ABIERTO)
# ==============================================================================
def gen_dune_textures() -> Dict[str, bytes]:
    rng = np.random.default_rng(202)
    s = 128
    # Lino ocre arena con tejido cruzado
    sand = Image.new("RGB", (s, s), (156, 128, 86))
    sd = ImageDraw.Draw(sand)
    for k in range(0, s, 4):
        sd.line([(0, k), (s, k)], fill=(138, 112, 74), width=1)
        sd.line([(k, 0), (k, s)], fill=(166, 138, 94), width=1)
    arr = np.clip(np.array(sand, dtype=np.int16) + rng.integers(-6, 7, (s, s, 1)), 0, 255).astype(np.uint8)
    sand = Image.fromarray(arr, "RGB")

    # Índigo profundo del desierto
    indigo = Image.new("RGB", (s, s), (28, 48, 78))
    iarr = np.clip(np.array(indigo, dtype=np.int16) + rng.integers(-5, 6, (s, s, 1)), 0, 255).astype(np.uint8)
    indigo = Image.fromarray(iarr, "RGB")

    # Cuero sillero tostado
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
    """Turbante voluminoso + gafas de aviador subidas a la frente (deja ojos y rostro 100% visibles)."""
    out = {"Head": _dp()}
    h = out["Head"]
    # Turbante alto enrollado sobre la coronilla (por encima de y = 0.175 para despejar cejas y ojos)
    for k in range(3):
        ry = 0.195 + k * 0.032
        rad = 0.126 - k * 0.010
        h["sand" if k % 2 == 0 else "indigo"].add(*make_torus(rad, 0.028, 22, 8, center=(0, ry, -0.008),
                                                             rot=(0.14 - k * 0.04, 0, 0.06 if k == 1 else -0.05)))
    # Cúpula superior del turbante + joya turquesa frontal
    h["sand"].add(*make_sphere(0.112, segs=16, rings=10, center=(0, 0.235, -0.01), scale=(1.02, 0.72, 1.05)))
    h["brass"].add(*make_cylinder(0.020, 0.020, 0.014, segs=8, center=(0, 0.228, 0.122), rot=(math.pi / 2, 0, 0)))
    h["turq"].add(*make_sphere(0.014, segs=8, rings=6, center=(0, 0.228, 0.128), scale=(0.8, 1.2, 0.6)))
    # Cola del turbante cayendo por detrás de la nuca
    h["indigo"].add(*make_curved_shell(-0.08, 0.16, 0.118, 0.128, 0.122, 0.132,
                                       math.pi * 0.72, math.pi * 1.28, thickness=0.008, tatter_waves=3, tatter_amp=0.025))
    # Gafas de aviador de latón sobre la frente (y = 0.188, fuera de los ojos)
    h["leather"].add(*make_curved_shell(0.172, 0.202, 0.124, 0.132, 0.124, 0.132, -math.pi, math.pi, thickness=0.006))
    for sgn in (-1, 1):
        h["brass"].add(*make_torus(0.024, 0.005, 14, 6, center=(sgn * 0.036, 0.188, 0.126), rot=(math.pi / 2 - 0.18, 0, 0)))
        h["turq"].add(*make_cylinder(0.019, 0.019, 0.006, segs=12, center=(sgn * 0.036, 0.188, 0.126), rot=(math.pi / 2 - 0.18, 0, 0)))
    return out


def build_dune_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    out = {"UpperArm_L": _bp(), "Torso": _dp(), "UpperArm_R": _dp()}
    out["UpperArm_L"] = _dp()
    # Poncho asimétrico de arena cayendo sobre hombro IZQUIERDO + bufanda bermellón al cuello
    tr = out["Torso"]
    tr["sand"].add(*make_curved_shell(0.02, 0.28, 0.182, 0.145, 0.238, 0.172,
                                      -0.15, math.pi * 1.28, thickness=0.009, tatter_waves=4, tatter_amp=0.045, flare_bot=0.045))
    tr["vermilion"].add(*make_curved_shell(0.22, 0.30, 0.132, 0.130, 0.148, 0.140,
                                           -math.pi, math.pi, thickness=0.012))
    # Hombro izquierdo: hombrera ligera de cuero con catalejo de latón sujeto
    ul = out["UpperArm_L"]
    ul["leather"].add(*make_curved_shell(-0.06, 0.04, 0.088, 0.085, 0.095, 0.092, -0.2, 2.6, thickness=0.010))
    ul["brass"].add(*make_cylinder(0.018, 0.014, 0.135, segs=10, center=(0.086, -0.02, 0.045), rot=(0.45, 0, -0.22)))
    # Hombro derecho despejado: tirante fino + pañuelo
    ur = out["UpperArm_R"]
    ur["indigo"].add(*make_curved_shell(-0.04, 0.01, 0.082, 0.080, 0.084, 0.082, -2.5, -0.4, thickness=0.007))
    return out


def build_dune_torso() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _dp()}
    t = out["Torso"]
    # Túnica cruzada índigo/arena ceñida sin placas pesadas + bandolera diagonal con bolsas
    t["indigo"].add(*make_curved_shell(-0.17, 0.28, 0.166, 0.128, 0.214, 0.152, -math.pi, math.pi, thickness=0.010))
    t["sand"].add(*make_extruded_polygon([(0.0, -0.15), (0.085, 0.05), (0.115, 0.26), (0.015, 0.27)], 0.010,
                                         center=(0.045, 0.04, 0.152), rot=(0.06, -0.25, 0)))
    strap = [(-0.155, 0.24, 0.145), (-0.03, 0.08, 0.168), (0.07, -0.04, 0.160), (0.15, -0.14, 0.132)]
    t["leather"].add(*make_tube_curve(strap, [0.020, 0.020, 0.020, 0.020], segs=8))
    t["leather"].add(*make_box(0.046, 0.064, 0.032, center=(0.06, 0.00, 0.172), rot=(0, 0, -0.5), bevel=0.005))
    t["brass"].add(*make_box(0.018, 0.022, 0.014, center=(-0.03, 0.08, 0.176)))
    return out


def build_dune_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Forearm_R": _dp(), "Wrist_R": _dp(), "Forearm_L": _dp(), "Wrist_L": _dp()}
    for sd, sgn in (("R", -1), ("L", 1)):
        fa, wr = out["Forearm_" + sd], out["Wrist_" + sd]
        for b in range(5):
            yb = -0.245 + b * 0.040
            rb = 0.046 + b * 0.0025
            fa["sand" if b % 2 == 0 else "leather"].add(*make_curved_shell(yb, yb + 0.034, rb, rb, rb + 0.002, rb + 0.002,
                                                                           -math.pi, math.pi, thickness=0.006))
        if sgn > 0:
            # Brújula de latón con aguja turquesa en muñeca izquierda
            fa["brass"].add(*make_cylinder(0.026, 0.026, 0.014, segs=12, center=(0.056, -0.20, 0.0), rot=(0, 0, math.pi / 2)))
            fa["turq"].add(*make_sphere(0.012, segs=8, rings=6, center=(0.064, -0.20, 0.0)))
        else:
            # Guardamano ligero de cuero y acero en mano derecha
            wr["steel"].add(*make_box(0.036, 0.072, 0.084, center=(-0.022, -0.032, 0.0), bevel=0.004))
    return out


def build_dune_belt() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Pelvis": _dp()}
    p = out["Pelvis"]
    p["vermilion"].add(*make_curved_shell(-0.08, 0.05, 0.174, 0.138, 0.178, 0.142, -math.pi, math.pi, thickness=0.012))
    p["indigo"].add(*make_curved_shell(-0.04, 0.02, 0.180, 0.144, 0.182, 0.146, -math.pi, math.pi, thickness=0.010))
    # Cantimplora redonda en cadera izquierda + manta enrollada trasera
    p["brass"].add(*make_cylinder(0.048, 0.048, 0.036, segs=14, center=(0.185, -0.055, -0.025), rot=(0, 0, math.pi / 2)))
    p["leather"].add(*make_cylinder(0.046, 0.046, 0.24, segs=12, center=(0.0, 0.025, -0.168), rot=(0, 0, math.pi / 2)))
    # Faja colgante frontal abierta
    p["sand"].add(*make_curved_shell(-0.48, -0.05, 0.215, 0.182, 0.174, 0.138, -0.45, 0.45, thickness=0.009, tatter_waves=3, tatter_amp=0.045))
    return out


def build_dune_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _dp()
        p["indigo"].add(*make_curved_shell(-0.34, -0.02, 0.118, 0.105, 0.104, 0.092,
                                           0.4 if sgn > 0 else -2.6, 2.6 if sgn > 0 else -0.4,
                                           thickness=0.008, tatter_waves=3, tatter_amp=0.045))
        # Vainas de cuchillos arrojadizos
        for k in (-1, 1):
            p["leather"].add(*make_box(0.016, 0.105, 0.024, center=(sgn * 0.112, -0.14, k * 0.028), rot=(0.15 * k, 0, sgn * -0.1)))
            p["brass"].add(*make_cylinder(0.006, 0.006, 0.040, segs=6, center=(sgn * 0.116, -0.075, k * 0.028)))
        out[side] = p
    return out


def build_dune_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _dp()
        p["sand"].add(*make_curved_shell(-0.39, -0.06, 0.095, 0.095, 0.116, 0.116, -math.pi, math.pi, thickness=0.009))
        for y in (-0.16, -0.30):
            p["leather"].add(*make_curved_shell(y, y + 0.022, 0.116, 0.116, 0.116, 0.116, -math.pi, math.pi, thickness=0.008))
        if sgn < 0:
            # Rollo de mapa estelar en muslo derecho
            p["sand"].add(*make_cylinder(0.022, 0.022, 0.17, segs=10, center=(-0.118, -0.22, 0.02), rot=(0.25, 0, 0)))
        out[side] = p
    return out


def build_dune_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _dp(), "Knee_R": _dp()}
    for side in ("Knee_L", "Knee_R"):
        k = out[side]
        k["leather"].add(*make_curved_shell(-0.042, 0.042, 0.082, 0.082, 0.082, 0.082, -math.pi, math.pi, thickness=0.008))
        k["brass"].add(*make_cylinder(0.036, 0.036, 0.018, segs=8, center=(0, 0, 0.084), rot=(math.pi / 2, 0, 0)))
    return out


def build_dune_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side in ("L", "R"):
        sh, ak = _dp(), _dp()
        sh["leather"].add(*make_curved_shell(-0.44, -0.10, 0.070, 0.072, 0.086, 0.088, -math.pi, math.pi, thickness=0.009))
        sh["vermilion"].add(*make_curved_shell(-0.16, -0.09, 0.088, 0.090, 0.094, 0.096, -math.pi, math.pi, thickness=0.008))
        ak["leather"].add(*make_box(0.096, 0.064, 0.220, center=(0, -0.022, 0.052), bevel=0.012))
        # Puntera curva levantada estilo nómada
        ak["brass"].add(*make_tube_curve([(0, -0.036, 0.14), (0, -0.022, 0.19), (0, 0.024, 0.215)],
                                         [0.026, 0.016, 0.005], segs=8))
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_dune_weapon() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_R": _dp()}
    w = out["Wrist_R"]
    gy = -0.035
    w["leather"].add(*make_cylinder(0.015, 0.016, 0.22, segs=10, center=(0, gy, 0.0), rot=(math.pi / 2, 0, 0)))
    w["brass"].add(*make_sphere(0.024, segs=8, rings=6, center=(0, gy, -0.125)))
    w["brass"].add(*make_torus(0.075, 0.011, 16, 8, arc=math.pi * 1.05, center=(0, gy, 0.125), rot=(0, -math.pi * 0.52, 0)))
    w["turq"].add(*make_sphere(0.018, segs=8, rings=6, center=(0, gy + 0.016, 0.125)))
    # Hoja alfanje creciente en el plano XZ orientada en +Z
    scim_pts = [(-0.016, 0.12), (0.040, 0.48), (0.115, 0.80), (0.055, 1.06), (0.055, 0.78), (0.005, 0.45), (-0.032, 0.12)]
    w["steel"].add(*make_extruded_polygon(scim_pts, 0.009, bevel=0.003, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    w["vermilion"].add(*make_cylinder(0.004, 0.022, 0.14, segs=8, center=(-0.02, gy, -0.20), rot=(math.pi / 2, 0, 0)))
    return out


def build_dune_shield() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_L": _dp()}
    s = out["Wrist_L"]
    gy = -0.035
    s["leather"].add(*make_cylinder(0.015, 0.015, 0.18, segs=10, center=(0, gy, 0), rot=(math.pi / 2, 0, 0)))
    s["brass"].add(*make_box(0.022, 0.016, 0.11, center=(0, gy, 0.09), rot=(0, math.pi / 2, 0)))
    # Hoja de daga de parada hacia +Z
    s["steel"].add(*make_cylinder(0.002, 0.020, 0.34, segs=4, center=(0, gy, 0.27), rot=(math.pi / 2, 0, 0), scale=(0.4, 1.0, 1.0)))
    # Broquel compacto solar lateral (+X)
    s["leather"].add(*make_cylinder(0.135, 0.145, 0.018, segs=16, center=(0.046, gy + 0.02, 0.02), rot=(0, 0, math.pi / 2)))
    s["brass"].add(*make_torus(0.138, 0.011, 18, 6, center=(0.052, gy + 0.02, 0.02), rot=(0, 0, math.pi / 2)))
    s["turq"].add(*make_sphere(0.030, segs=8, rings=6, center=(0.060, gy + 0.02, 0.02), scale=(0.55, 1.0, 1.0)))
    return out


# ==============================================================================
# ARQUETIPO 3 (MAGO): TEJEDOR DEL ASTROLABIO & ECLIPSE (ROSTRO ABIERTO)
# ==============================================================================
def gen_astrolabe_textures() -> Dict[str, bytes]:
    rng = np.random.default_rng(303)
    s = 128
    # Terciopelo azul medianoche estrellado
    velvet = Image.new("RGB", (s, s), (16, 22, 46))
    vd = ImageDraw.Draw(velvet)
    for _ in range(48):
        sx, sy = int(rng.integers(0, s)), int(rng.integers(0, s))
        vd.point((sx, sy), fill=(195, 175, 245))
    arr = np.clip(np.array(velvet, dtype=np.int16) + rng.integers(-4, 5, (s, s, 1)), 0, 255).astype(np.uint8)
    velvet = Image.fromarray(arr, "RGB")

    # Seda astral gris plateada
    silk = Image.new("RGB", (s, s), (98, 108, 134))
    sarr = np.clip(np.array(silk, dtype=np.int16) + rng.integers(-5, 6, (s, s, 1)), 0, 255).astype(np.uint8)
    silk = Image.fromarray(sarr, "RGB")

    # Bronce planetario grabado
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
    """Diadema abierta orbital + sextante astral sobre la coronilla (rostro 100% despejado)."""
    out = {"Head": _ap()}
    h = out["Head"]
    # Banda diadema de bronce sobre la frente alta
    h["bronze"].add(*make_curved_shell(0.185, 0.215, 0.120, 0.128, 0.120, 0.128, -math.pi, math.pi, thickness=0.008))
    # Corona orbital inclinada flotando sobre la cabeza (halo de eclipse)
    h["bronze"].add(*make_torus(0.175, 0.008, 28, 8, center=(0, 0.275, -0.025), rot=(0.45, 0.0, 0.18)))
    h["silver"].add(*make_torus(0.145, 0.006, 24, 6, center=(0, 0.268, -0.020), rot=(-0.35, 0.2, -0.22)))
    # Sextante & gema astral sobre el entrecejo superior (y = 0.22)
    h["bronze"].add(*make_torus(0.038, 0.006, 16, 6, arc=math.pi, center=(0, 0.22, 0.130), rot=(math.pi / 2, 0, math.pi * 0.5)))
    h["astral"].add(*make_sphere(0.016, segs=8, rings=6, center=(0, 0.228, 0.136)))
    # Satélites solares laterales
    for ang in (0.6, math.pi - 0.6, math.pi + 0.6):
        h["amber"].add(*make_sphere(0.013, segs=8, rings=6,
                                    center=(math.cos(ang) * 0.175, 0.275 + math.sin(ang) * 0.06, -0.025 + math.sin(ang) * 0.15)))
    return out


def build_astro_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    """Astrolabio Cósmico Gigante de 3 anillos flotando tras los hombros + estola dorsal ceremonial."""
    out = {"Torso": _ap(), "UpperArm_L": _ap(), "UpperArm_R": _ap()}
    tr = out["Torso"]
    cx, cy, cz = 0.0, 0.24, -0.24
    # 3 anillos concéntricos inclinados del gran astrolabio dorsal
    tr["bronze"].add(*make_torus(0.32, 0.012, 32, 8, center=(cx, cy, cz), rot=(math.pi / 2 + 0.12, 0.0, 0.08)))
    tr["silver"].add(*make_torus(0.24, 0.009, 28, 6, center=(cx, cy, cz), rot=(math.pi / 2 + 0.35, 0.42, -0.25)))
    tr["amber"].add(*make_torus(0.16, 0.007, 24, 6, center=(cx, cy, cz), rot=(math.pi / 2 - 0.35, -0.5, 0.32)))
    # Orbes estelares en los anillos
    for s in range(4):
        ang = (s / 4) * math.tau + 0.35
        tr["astral"].add(*make_sphere(0.022, segs=8, rings=6, center=(math.cos(ang) * 0.32, cy + math.sin(ang) * 0.32, cz)))
    # Estola sacerdotal colgante dorsal y por hombros
    tr["silk"].add(*make_curved_shell(-0.26, 0.28, 0.180, 0.145, 0.220, 0.165,
                                      math.pi * 0.65, math.pi * 1.35, thickness=0.008, tatter_waves=4, tatter_amp=0.04))
    for k, sd in (("UpperArm_L", 1), ("UpperArm_R", -1)):
        ua = out[k]
        ua["velvet"].add(*make_curved_shell(-0.04, 0.05, 0.088, 0.086, 0.105, 0.102,
                                            0.0 if sd > 0 else -math.pi, math.pi if sd > 0 else 0.0, thickness=0.009))
        ua["bronze"].add(*make_torus(0.082, 0.007, 16, 6, center=(sd * 0.04, 0.045, 0), rot=(0, 0, sd * 0.3)))
    return out


def build_astro_torso() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Torso": _ap()}
    t = out["Torso"]
    # Túnica ceremonial de terciopelo + cuello plisado en abanico + medallón reloj solar en el pecho
    t["velvet"].add(*make_curved_shell(-0.17, 0.28, 0.168, 0.132, 0.216, 0.155, -math.pi, math.pi, thickness=0.011))
    # Cuello plisado alto abierto delante
    t["silk"].add(*make_curved_shell(0.24, 0.39, 0.108, 0.106, 0.152, 0.140, 0.58, math.tau - 0.58,
                                     thickness=0.009, flare_top=0.022))
    # Estola frontal asimétrica + reloj solar astral pectoral
    t["silk"].add(*make_box(0.095, 0.42, 0.012, center=(-0.035, 0.04, 0.162), rot=(0.05, 0, 0.12), bevel=0.003))
    t["bronze"].add(*make_torus(0.055, 0.009, 18, 6, center=(0.0, 0.135, 0.172), rot=(math.pi / 2, 0, 0)))
    t["astral"].add(*make_sphere(0.022, segs=8, rings=6, center=(0.0, 0.135, 0.174)))
    return out


def build_astro_gloves() -> Dict[str, Dict[str, MeshPart]]:
    """Mangas acampanadas ceremoniales muy anchas en puño + mitones rúnicos."""
    out = {"Forearm_R": _ap(), "Wrist_R": _ap(), "Forearm_L": _ap(), "Wrist_L": _ap()}
    for sd, sgn in (("R", -1), ("L", 1)):
        fa, wr = out["Forearm_" + sd], out["Wrist_" + sd]
        fa["velvet"].add(*make_curved_shell(-0.25, -0.02, 0.108, 0.108, 0.058, 0.058, -math.pi, math.pi,
                                            thickness=0.008, flare_bot=0.028))
        fa["silk"].add(*make_curved_shell(-0.255, -0.215, 0.115, 0.115, 0.105, 0.105, -math.pi, math.pi, thickness=0.007))
        fa["bronze"].add(*make_torus(0.112, 0.006, 18, 6, center=(0, -0.23, 0)))
        wr["astral"].add(*make_sphere(0.012, segs=6, rings=5, center=(sgn * 0.026, -0.025, 0)))
    return out


def build_astro_belt() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Pelvis": _ap()}
    p = out["Pelvis"]
    p["silk"].add(*make_curved_shell(-0.07, 0.05, 0.175, 0.138, 0.178, 0.142, -math.pi, math.pi, thickness=0.012))
    p["bronze"].add(*make_torus(0.042, 0.008, 16, 6, center=(0, -0.01, 0.152), rot=(math.pi / 2, 0, 0)))
    # Faldón ceremonial largo de 4 paneles partidos
    for a0, a1 in ((-0.45, 0.45), (0.65, 1.95), (math.pi * 0.65, math.pi * 1.35), (math.tau - 1.95, math.tau - 0.65)):
        p["velvet"].add(*make_curved_shell(-0.68, -0.05, 0.258, 0.228, 0.178, 0.142, a0, a1,
                                           thickness=0.009, flare_bot=0.060))
        p["silk"].add(*make_curved_shell(-0.685, -0.62, 0.262, 0.232, 0.256, 0.226, a0 + 0.04, a1 - 0.04, thickness=0.007))
    return out


def build_astro_tassets() -> Dict[str, Dict[str, MeshPart]]:
    """Péndulos colgantes y pequeños relojes estelares laterales en caderas."""
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _ap()
        for offz in (-0.032, 0.032):
            p["bronze"].add(*make_cylinder(0.003, 0.003, 0.20, segs=6, center=(sgn * 0.115, -0.13, offz)))
            p["amber"].add(*make_sphere(0.016, segs=8, rings=6, center=(sgn * 0.115, -0.235, offz), scale=(0.8, 1.4, 0.8)))
        p["bronze"].add(*make_torus(0.044, 0.006, 16, 6, center=(sgn * 0.118, -0.14, 0.0), rot=(0, 0, math.pi / 2)))
        p["astral"].add(*make_sphere(0.016, segs=8, rings=6, center=(sgn * 0.120, -0.14, 0.0)))
        out[side] = p
    return out


def build_astro_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _ap()
        p["velvet"].add(*make_curved_shell(-0.38, -0.06, 0.092, 0.092, 0.112, 0.112, -math.pi, math.pi, thickness=0.009))
        p["silk"].add(*make_box(0.012, 0.28, 0.022, center=(sgn * 0.106, -0.22, 0.0)))
        out[side] = p
    return out


def build_astro_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _ap(), "Knee_R": _ap()}
    for side in ("Knee_L", "Knee_R"):
        k = out[side]
        k["bronze"].add(*make_torus(0.042, 0.006, 16, 6, center=(0, 0, 0.088), rot=(math.pi / 2, 0, 0)))
        k["astral"].add(*make_sphere(0.014, segs=8, rings=6, center=(0, 0, 0.092)))
    return out


def build_astro_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side in ("L", "R"):
        sh, ak = _ap(), _ap()
        sh["velvet"].add(*make_curved_shell(-0.44, -0.18, 0.068, 0.070, 0.084, 0.086, -math.pi, math.pi, thickness=0.008))
        sh["silk"].add(*make_torus(0.086, 0.010, 16, 6, center=(0, -0.18, 0)))
        ak["velvet"].add(*make_box(0.092, 0.058, 0.215, center=(0, -0.024, 0.050), bevel=0.010))
        ak["bronze"].add(*make_tube_curve([(0, -0.036, 0.14), (0, -0.020, 0.195), (0, 0.035, 0.225)],
                                         [0.022, 0.014, 0.004], segs=8))
        ak["amber"].add(*make_sphere(0.011, segs=6, rings=5, center=(0, 0.012, 0.09)))
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_astro_weapon() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Wrist_R": _ap()}
    w = out["Wrist_R"]
    gy = -0.035
    w["bronze"].add(*make_cylinder(0.014, 0.016, 1.54, segs=10, center=(0, gy, 0.34), rot=(math.pi / 2, 0, 0)))
    w["velvet"].add(*make_cylinder(0.018, 0.018, 0.22, segs=10, center=(0, gy, 0.02), rot=(math.pi / 2, 0, 0)))
    w["silver"].add(*make_cylinder(0.002, 0.018, 0.12, segs=8, center=(0, gy, -0.49), rot=(math.pi / 2, 0, 0)))
    # Corona cardán de 3 anillos girados + sol astral flotante en punta (+Z = 1.24)
    cz = 1.24
    w["bronze"].add(*make_torus(0.095, 0.010, 18, 6, arc=math.pi * 1.15, center=(0, gy, 1.21), rot=(math.pi / 2, -0.5, 0)))
    w["bronze"].add(*make_torus(0.118, 0.008, 24, 6, center=(0, gy, cz), rot=(math.pi / 2, 0.35, 0)))
    w["silver"].add(*make_torus(0.088, 0.006, 22, 6, center=(0, gy, cz), rot=(0.65, -0.5, 0.35)))
    w["amber"].add(*make_torus(0.058, 0.005, 18, 6, center=(0, gy, cz), rot=(-0.7, 0.6, 0.8)))
    w["astral"].add(*make_sphere(0.036, segs=10, rings=8, center=(0, gy, cz)))
    return out


def build_astro_shield() -> Dict[str, Dict[str, MeshPart]]:
    """Grimorio Abierto & Planetario Flotante en mano izquierda."""
    out = {"Wrist_L": _ap()}
    s = out["Wrist_L"]
    gy = -0.035
    s["bronze"].add(*make_cylinder(0.012, 0.012, 0.14, segs=8, center=(0, gy, 0), rot=(math.pi / 2, 0, 0)))
    # Lomo y dos tapas abiertas en V
    s["velvet"].add(*make_box(0.034, 0.026, 0.23, center=(0.032, gy + 0.04, 0.0)))
    for sd in (-1, 1):
        s["velvet"].add(*make_box(0.012, 0.125, 0.22, center=(0.048, gy + 0.04, sd * 0.062), rot=(0, -sd * 0.42, 0)))
        s["parchment"].add(*make_box(0.014, 0.112, 0.20, center=(0.056, gy + 0.04, sd * 0.058), rot=(0, -sd * 0.42, 0)))
    # Planetario flotante delante del libro abierto
    ox, oy, oz = 0.145, gy + 0.06, 0.0
    s["astral"].add(*make_sphere(0.032, segs=10, rings=8, center=(ox, oy, oz)))
    s["bronze"].add(*make_torus(0.064, 0.005, 20, 6, center=(ox, oy, oz), rot=(0.55, 0.4, 0.2)))
    s["amber"].add(*make_torus(0.050, 0.004, 18, 6, center=(ox, oy, oz), rot=(-0.6, 0.8, 0.5)))
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
