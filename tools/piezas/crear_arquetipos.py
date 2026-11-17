#!/usr/bin/env python3
"""Generador procedural en Python de las 3 nuevas armaduras S-Rank de alta estética (una por clase):
  1) PESADA — Soberano Dracónico Carmesí & Plata Negra (helmet/11, shoulders/14, torso/22, gloves/8, belt/10, tassets/11, cuisses/10, knees/10, boots/15, weapon/83, shield/13)
  2) LIGERA — Cazador Élfico de Esmeralda & Luz Lunar (helmet/12, shoulders/15, torso/23, gloves/9, belt/11, tassets/12, cuisses/11, knees/11, boots/16, weapon/84, shield/14)
  3) MAGO   — Archimago Serafín del Eclipse & Terciopelo Astral (helmet/13, shoulders/16, torso/24, gloves/10, belt/12, tassets/13, cuisses/12, knees/12, boots/17, weapon/85, shield/15)

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
    nodes = json.loads(json.dumps(SKELETON_TEMPLATE))  # copia profunda: las listas children no se comparten entre GLB
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
# ARQUETIPO 1 (PESADA): SOBERANO DRACÓNICO CARMESÍ & PLATA NEGRA
# ==============================================================================
def gen_basalt_textures() -> Dict[str, bytes]:
    rng = np.random.default_rng(101)
    s = 256
    # Acero obsidiana damasquinado con escamas de dragón y filigrana sutil
    basalt = Image.new("RGB", (s, s), (22, 20, 28))
    bd = ImageDraw.Draw(basalt)
    step = 32
    for row in range(0, s + step, step // 2):
        offset = (step // 2) if (row // (step // 2)) % 2 else 0
        for col in range(-step, s + step, step):
            cx = col + offset
            bd.arc([cx - 16, row - 16, cx + 16, row + 16], 15, 165, fill=(68, 56, 72), width=2)
            bd.arc([cx - 14, row - 14, cx + 14, row + 14], 25, 155, fill=(138, 108, 56), width=1)
    arr = np.clip(np.array(basalt, dtype=np.int16) + rng.integers(-4, 5, (s, s, 1)), 0, 255).astype(np.uint8)
    basalt = Image.fromarray(arr, "RGB")

    # Plata negra espejo imperial con ondas de damasco
    iron = Image.new("RGB", (s, s), (44, 42, 52))
    idraw = ImageDraw.Draw(iron)
    for y in range(0, s, 8):
        pts = [(x, int(y + math.sin(x * 0.08 + y * 0.05) * 4)) for x in range(0, s, 8)]
        idraw.line(pts, fill=(76, 74, 88), width=1)

    # Terciopelo y escama carmesí imperial con bordado de oro
    copper = Image.new("RGB", (s, s), (124, 16, 30))
    cdraw = ImageDraw.Draw(copper)
    for y in range(0, s, 32):
        for x in range(0, s, 32):
            cdraw.polygon([(x + 16, y + 2), (x + 30, y + 16), (x + 16, y + 30), (x + 2, y + 16)], outline=(186, 138, 58))

    return {
        "basalt_albedo": _img_to_png_bytes(basalt),
        "iron_albedo": _img_to_png_bytes(iron),
        "copper_albedo": _img_to_png_bytes(copper),
    }


BASALT_MAT_SPECS = [
    ("basalt", {"name": "Sovereign_ObsidianPlate", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.90, "roughnessFactor": 0.24}, "doubleSided": True}),
    ("iron", {"name": "Sovereign_DarkSilver", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.94, "roughnessFactor": 0.20}, "doubleSided": True}),
    ("copper", {"name": "Sovereign_CrimsonVelvet", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}, "metallicFactor": 0.18, "roughnessFactor": 0.56}, "doubleSided": True}),
    ("brass", {"name": "Sovereign_RoyalGold", "pbrMetallicRoughness": {"baseColorFactor": [0.90, 0.72, 0.32, 1.0], "metallicFactor": 0.95, "roughnessFactor": 0.18}, "doubleSided": True}),
    ("leather", {"name": "Sovereign_CrimsonScale", "pbrMetallicRoughness": {"baseColorFactor": [0.28, 0.05, 0.08, 1.0], "metallicFactor": 0.25, "roughnessFactor": 0.52}, "doubleSided": True}),
    ("magma", {"name": "Sovereign_BloodRuby", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.10, 0.22, 1.0], "metallicFactor": 0.15, "roughnessFactor": 0.08}, "emissiveFactor": [0.95, 0.06, 0.18], "doubleSided": True}),
]
BASALT_KEYS = ("basalt", "iron", "copper", "brass", "leather", "magma")


def _bp() -> Dict[str, MeshPart]:
    return _parts(*BASALT_KEYS)


def _horn(part: MeshPart, pts: List[Tuple[float, float, float]], r0: float, segs: int = 8) -> None:
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
    _horn(parts[toe_key], [
        (0, ground_y + 0.026, toe_len * 0.72),
        (0, ground_y + 0.032, toe_len * (0.98 + pointed * 0.55)),
        (0, ground_y + 0.068 + pointed * 0.04, toe_len * (1.08 + pointed * 0.90)),
    ], foot_w * 0.28, segs=8)
    _horn(parts[spur_key], [
        (0, ground_y + 0.034, -heel_len * 0.85),
        (0, ground_y + 0.038, -heel_len - 0.035),
        (0, ground_y + 0.056, -heel_len - 0.058),
    ], 0.013, segs=6)


def build_basalt_helmet() -> Dict[str, Dict[str, MeshPart]]:
    """Corona-Visor Imperial de Dragón Carmesí & Plata Negra (rostro abierto con corona real, guardas de pómulo y cuernos dracónicos)."""
    out = {"Head": _bp()}
    h = out["Head"]
    # Diadema-corona imperial en plata negra y oro real dejando el rostro descubierto
    h["basalt"].add(*make_curved_shell(
        0.155, 0.225, 0.122, 0.132, 0.118, 0.128,
        -math.pi * 0.92, math.pi * 0.92, thickness=0.010, ny=8, na=24, flare_top=0.016, peak_front_y=-0.022
    ))
    h["brass"].add(*make_curved_shell(
        0.162, 0.205, 0.126, 0.136, 0.124, 0.134,
        -1.25, 1.25, thickness=0.008, ny=6, na=18, peak_front_y=-0.018
    ))
    # Guardas laterales de mejilla en plata negra y oro que enmarcan la mandíbula sin cubrir ojos/boca
    for sgn in (-1, 1):
        a0 = 0.72 if sgn > 0 else -1.65
        a1 = 1.65 if sgn > 0 else -0.72
        h["iron"].add(*make_curved_shell(
            0.01, 0.155, 0.118, 0.128, 0.122, 0.132,
            a0, a1, thickness=0.009, ny=8, na=10
        ))
        # Cuernos dracónicos reales estriados en oro y plata negra
        _horn(h["brass"], [(sgn * 0.092, 0.195, 0.02), (sgn * 0.155, 0.265, -0.02), (sgn * 0.145, 0.365, -0.09), (sgn * 0.105, 0.425, -0.14)], 0.020)
        _horn(h["iron"], [(sgn * 0.075, 0.205, 0.04), (sgn * 0.115, 0.285, 0.00), (sgn * 0.095, 0.355, -0.05)], 0.013)
    # Cresta central de lanza imperial y rubí tallado en el entrecejo
    _horn(h["brass"], [(0.0, 0.185, 0.135), (0.0, 0.265, 0.145), (0.0, 0.335, 0.115)], 0.014)
    h["magma"].add(*make_sphere(0.018, segs=8, rings=6, center=(0, 0.185, 0.142), scale=(0.8, 1.35, 0.6)))
    return out


def build_basalt_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    """Hombreras Aladas del Soberano Dracónico + 6 Alas Dorsales + Media Capa Imperial Carmesí."""
    out = {"UpperArm_L": _bp(), "Torso": _bp(), "UpperArm_R": _bp()}
    for side, sgn in (("UpperArm_L", 1), ("UpperArm_R", -1)):
        u = out[side]
        u["basalt"].add(*make_curved_shell(
            -0.03, 0.095, 0.102, 0.104, 0.084, 0.086,
            -0.5 if sgn > 0 else -math.pi * 1.15, math.pi * 1.15 if sgn > 0 else 0.5,
            thickness=0.013, ny=10, na=16, flare_bot=0.028
        ))
        u["brass"].add(*make_curved_shell(
            0.02, 0.115, 0.112, 0.112, 0.090, 0.090,
            -0.35 if sgn > 0 else -math.pi * 1.05, math.pi * 1.05 if sgn > 0 else 0.35,
            thickness=0.010, ny=8, na=14, peak_front_y=0.022
        ))
        _horn(u["brass"], [(sgn * 0.055, 0.085, 0.01), (sgn * 0.135, 0.175, -0.02), (sgn * 0.165, 0.275, -0.06)], 0.021)
        _horn(u["iron"], [(sgn * 0.082, 0.052, -0.02), (sgn * 0.155, 0.118, -0.05), (sgn * 0.195, 0.195, -0.09)], 0.015)
        u["magma"].add(*make_sphere(0.016, segs=8, rings=6, center=(sgn * 0.098, 0.055, 0.045)))

    # Torso: Gran capa larga fluida estilo cuervo en carmesí con cenefa bordada en oro (sin alas rígidas ni adorno dorsal)
    tr = out["Torso"]
    tr["copper"].add(*make_curved_shell(
        -0.68, 0.22, 0.240, 0.195, 0.162, 0.145,
        math.pi * 0.45, math.pi * 1.55, thickness=0.007, ny=16, na=24, flare_bot=0.095, tatter_waves=5, tatter_amp=0.048
    ))
    tr["brass"].add(*make_curved_shell(
        -0.68, -0.56, 0.245, 0.200, 0.232, 0.190,
        math.pi * 0.47, math.pi * 1.53, thickness=0.006, ny=4, na=22, tatter_waves=5, tatter_amp=0.042
    ))
    return out


def build_basalt_torso() -> Dict[str, Dict[str, MeshPart]]:
    """Coraza Real de Escamas de Dragón & Filigrana de Oro."""
    out = {"Torso": _bp()}
    t = out["Torso"]
    t["basalt"].add(*make_curved_shell(
        -0.16, 0.28, 0.174, 0.136, 0.222, 0.160,
        -math.pi, math.pi, thickness=0.015, ny=12, na=28, peak_front_y=0.028
    ))
    chest_pts = [(0.0, -0.05), (0.092, 0.05), (0.118, 0.22), (0.0, 0.26), (-0.118, 0.22), (-0.092, 0.05)]
    t["iron"].add(*make_extruded_polygon(chest_pts, 0.022, bevel=0.007, center=(0, 0.04, 0.156), rot=(-0.08, 0, 0)))
    t["magma"].add(*make_sphere(0.032, segs=10, rings=8, center=(0, 0.15, 0.174), scale=(0.82, 1.38, 0.55)))
    for side in (-1, 1):
        for ry in (0.05, 0.12, 0.19):
            t["brass"].add(*make_tube_curve([
                (side * 0.028, ry, 0.172),
                (side * 0.145, ry + 0.018, 0.115),
                (side * 0.185, ry + 0.030, -0.02),
            ], [0.008, 0.010, 0.006], segs=7))
    t["brass"].add(*make_curved_shell(
        0.24, 0.34, 0.116, 0.112, 0.144, 0.132,
        0.48, math.tau - 0.48, thickness=0.010, ny=6, na=20, flare_top=0.022, tatter_waves=5, tatter_amp=0.014
    ))
    return out


def build_basalt_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Forearm_R": _bp(), "Wrist_R": _bp(), "Forearm_L": _bp(), "Wrist_L": _bp()}
    for sd, sgn in (("R", -1), ("L", 1)):
        fa, wr = out["Forearm_" + sd], out["Wrist_" + sd]
        fa["basalt"].add(*make_curved_shell(-0.25, -0.02, 0.055, 0.055, 0.071, 0.071, -math.pi, math.pi, thickness=0.010, flare_top=0.014))
        fin_pts = [(0.0, -0.04), (sgn * 0.068, 0.01), (sgn * 0.048, 0.095), (0.0, 0.04)]
        fa["brass"].add(*make_extruded_polygon(fin_pts, 0.014, bevel=0.004, center=(sgn * 0.045, -0.04, -0.02), rot=(0, sgn * 0.35, 0)))
        fa["magma"].add(*make_sphere(0.014, segs=8, rings=6, center=(sgn * 0.056, -0.14, 0.0)))
        wr["iron"].add(*make_curved_shell(-0.075, 0.005, 0.047, 0.041, 0.051, 0.045, -math.pi, math.pi, thickness=0.008))
        for kz in (-0.024, 0.0, 0.024):
            _horn(wr["brass"], [(sgn * 0.040, -0.035, kz), (sgn * 0.066, -0.055, kz), (sgn * 0.082, -0.088, kz)], 0.0075, segs=6)
    return out


def build_basalt_belt() -> Dict[str, Dict[str, MeshPart]]:
    """Fajín Carmesí de Escamas de Dragón & Llamas (hebilla flamígera en V grande, escamas superpuestas, cadenas doradas con borlas y estandarte ampliado >=40%)."""
    out = {"Pelvis": _bp()}
    p = out["Pelvis"]
    # Cinturón base ceñido en V en obsidiana/plata negra y filete dorado (sin abultar en cintura)
    p["basalt"].add(*make_curved_shell(-0.062, 0.052, 0.172, 0.136, 0.178, 0.142, -math.pi, math.pi, thickness=0.011, ny=6, na=24, peak_front_y=-0.032))
    p["brass"].add(*make_curved_shell(0.038, 0.058, 0.176, 0.140, 0.180, 0.144, -math.pi, math.pi, thickness=0.009, ny=3, na=22))
    # Gran hebilla flamígera en V esculpida en oro real con rubí central (+45%)
    b_pts = [(0.0, -0.125), (0.068, -0.035), (0.115, 0.048), (0.064, 0.092), (0.0, 0.058), (-0.064, 0.092), (-0.115, 0.048), (-0.068, -0.035)]
    p["brass"].add(*make_extruded_polygon(b_pts, 0.022, bevel=0.007, center=(0, -0.008, 0.150)))
    p["magma"].add(*make_sphere(0.030, segs=10, rings=8, center=(0, -0.004, 0.170), scale=(0.84, 1.40, 0.58)))
    # 3 filas de escamas de dragón ampliadas sin sobresalir del muslo
    scale_pts = [(0.0, -0.058), (0.044, -0.016), (0.038, 0.032), (-0.038, 0.032), (-0.044, -0.016)]
    for row in range(3):
        yc = -0.075 - row * 0.074
        n_cols = 4 - row
        for c in range(n_cols):
            for sgn in (-1, 1):
                ang = sgn * (0.48 + c * 0.28 + row * 0.08)
                rx = 0.186 + row * 0.012
                rz = 0.150 + row * 0.012
                px = math.sin(ang) * rx
                pz = math.cos(ang) * rz
                mat_k = "leather" if (row + c) % 2 == 0 else "copper"
                p[mat_k].add(*make_extruded_polygon(
                    scale_pts, 0.011, bevel=0.003,
                    center=(px, yc, pz),
                    rot=(0.14, ang, 0.0)
                ))
                p["brass"].add(*make_sphere(0.008, segs=6, rings=5, center=(px * 1.03, yc + 0.014, pz * 1.03)))
    # Gran estandarte central puntiagudo con llamas doradas (+45% longitud y anchura)
    apron_pts = [(0.0, -0.56), (0.104, -0.33), (0.122, -0.04), (-0.122, -0.04), (-0.104, -0.33)]
    p["copper"].add(*make_extruded_polygon(apron_pts, 0.014, bevel=0.004, center=(0, -0.06, 0.156), rot=(-0.08, 0, 0)))
    flame_trim_pts = [(0.0, -0.50), (0.066, -0.31), (0.026, -0.19), (0.078, -0.08), (0.0, -0.12), (-0.078, -0.08), (-0.026, -0.19), (-0.066, -0.31)]
    p["brass"].add(*make_extruded_polygon(flame_trim_pts, 0.009, bevel=0.002, center=(0, -0.06, 0.166), rot=(-0.08, 0, 0)))
    # Faldón trasero ampliado con holgura sobre muslos
    p["copper"].add(*make_curved_shell(
        -0.62, -0.04, 0.240, 0.205, 0.178, 0.142,
        0.46, math.tau - 0.46, thickness=0.009, ny=14, na=26, flare_bot=0.068, tatter_waves=5, tatter_amp=0.062
    ))
    # Cadenas doradas arqueadas y borlas de seda ampliadas ~45%
    for sgn in (-1, 1):
        p["brass"].add(*make_tube_curve([
            (sgn * 0.065, -0.032, 0.154), (sgn * 0.155, -0.165, 0.132), (sgn * 0.208, -0.075, 0.025)
        ], [0.008, 0.009, 0.008], segs=7))
        p["brass"].add(*make_tube_curve([
            (sgn * 0.075, -0.060, 0.150), (sgn * 0.172, -0.235, 0.112), (sgn * 0.215, -0.115, 0.005)
        ], [0.007, 0.008, 0.007], segs=7))
        for tx, ty, tz in ((sgn * 0.145, -0.235, 0.135), (sgn * 0.188, -0.305, 0.092), (0.0, -0.595, 0.190)):
            if sgn < 0 and tx == 0.0:
                continue
            p["brass"].add(*make_sphere(0.016, segs=7, rings=5, center=(tx, ty + 0.045, tz)))
            p["basalt"].add(*make_cylinder(0.007, 0.021, 0.082, segs=8, center=(tx, ty, tz)))
    return out


def build_basalt_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _bp()
        for row in range(3):
            yc = -0.045 - row * 0.076
            w = 0.056 - row * 0.005
            plate_pts = [(0.0, -0.055), (w, 0.0), (w * 0.85, 0.042), (-w * 0.85, 0.042), (-w, 0.0)]
            p["basalt" if row != 1 else "brass"].add(*make_extruded_polygon(
                plate_pts, 0.015, bevel=0.005,
                center=(sgn * (0.118 + row * 0.004), yc, 0.015),
                rot=(0.0, sgn * math.pi * 0.42, sgn * -0.14)
            ))
        out[side] = p
    return out


def build_basalt_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _bp()
        p["basalt"].add(*make_curved_shell(-0.37, -0.07, 0.094, 0.094, 0.115, 0.115, -math.pi, math.pi, thickness=0.012, peak_front_y=-0.020))
        ridge_pts = [(0.0, -0.14), (0.042, 0.0), (0.0, 0.14), (-0.042, 0.0)]
        p["brass"].add(*make_extruded_polygon(ridge_pts, 0.016, bevel=0.005, center=(sgn * 0.012, -0.22, 0.105)))
        p["magma"].add(*make_sphere(0.016, segs=8, rings=6, center=(sgn * 0.012, -0.22, 0.116)))
        out[side] = p
    return out


def build_basalt_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _bp(), "Knee_R": _bp()}
    for side, sgn in (("Knee_L", 1), ("Knee_R", -1)):
        k = out[side]
        k["iron"].add(*make_curved_shell(-0.048, 0.048, 0.086, 0.086, 0.086, 0.086, -1.6, 1.6, thickness=0.011))
        wing_pts = [(0.0, -0.072), (0.056, -0.015), (0.068, 0.038), (0.0, 0.076), (-0.048, 0.025), (-0.045, -0.025)]
        k["basalt"].add(*make_extruded_polygon(wing_pts, 0.020, bevel=0.006, center=(sgn * 0.012, 0.0, 0.088), rot=(0, sgn * 0.18, 0)))
        _horn(k["brass"], [(sgn * 0.055, 0.02, 0.065), (sgn * 0.098, 0.060, 0.04), (sgn * 0.120, 0.115, 0.005)], 0.012, segs=6)
        k["magma"].add(*make_sphere(0.018, segs=8, rings=6, center=(0, 0.0, 0.104), scale=(0.8, 1.3, 0.6)))
    return out


def build_basalt_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("L", 1), ("R", -1)):
        sh, ak = _bp(), _bp()
        sh["basalt"].add(*make_curved_shell(-0.44, -0.05, 0.068, 0.072, 0.094, 0.098, -math.pi, math.pi, thickness=0.012, peak_front_y=0.024))
        shin_pts = [(0.0, -0.17), (0.044, 0.04), (0.0, 0.18), (-0.044, 0.04)]
        sh["brass"].add(*make_extruded_polygon(shin_pts, 0.020, bevel=0.006, center=(0, -0.23, 0.088)))
        _horn(sh["brass"], [(sgn * 0.075, -0.10, 0.02), (sgn * 0.108, -0.04, -0.01), (sgn * 0.122, 0.02, -0.03)], 0.011, segs=6)
        _boot_foot({"basalt": ak["basalt"], "iron": ak["iron"], "brass": ak["brass"], "copper": ak["copper"]},
                   "iron", "basalt", "brass", "brass", foot_w=0.110, toe_len=0.184, pointed=0.28)
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_basalt_weapon() -> Dict[str, Dict[str, MeshPart]]:
    """Mandoble Imperial del Soberano Dracónico."""
    out = {"Wrist_R": _bp()}
    w = out["Wrist_R"]
    gy = -0.035
    w["leather"].add(*make_cylinder(0.016, 0.017, 0.28, segs=14, center=(0, gy, -0.02), rot=(math.pi / 2, 0, 0)))
    w["brass"].add(*make_sphere(0.026, segs=10, rings=8, center=(0, gy, -0.18), scale=(1.0, 0.85, 1.35)))
    w["magma"].add(*make_sphere(0.018, segs=8, rings=6, center=(0, gy, -0.20)))
    guard_pts = [
        (0.0, 0.10), (0.10, 0.12), (0.22, 0.20), (0.24, 0.12),
        (0.05, 0.09), (0.0, 0.07), (-0.05, 0.09), (-0.24, 0.12), (-0.22, 0.20), (-0.10, 0.12)
    ]
    w["brass"].add(*make_extruded_polygon(guard_pts, 0.024, bevel=0.006, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    blade_pts = [
        (-0.038, 0.11), (-0.056, 0.24), (-0.036, 0.30), (-0.048, 0.44),
        (-0.034, 0.88), (0.0, 1.08), (0.034, 0.88), (0.048, 0.44), (0.036, 0.30), (0.056, 0.24), (0.038, 0.11)
    ]
    w["iron"].add(*make_extruded_polygon(blade_pts, 0.014, bevel=0.005, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    w["brass"].add(*make_box(0.016, 0.022, 0.72, center=(0, gy, 0.48)))
    w["magma"].add(*make_box(0.007, 0.026, 0.64, center=(0, gy, 0.50)))
    w["magma"].add(*make_sphere(0.025, segs=8, rings=6, center=(0, gy, 0.13), scale=(0.85, 0.75, 1.35)))
    return out


def build_basalt_shield() -> Dict[str, Dict[str, MeshPart]]:
    """Égida Alada del Dragón Carmesí."""
    out = {"Wrist_L": _bp()}
    s = out["Wrist_L"]
    gy = -0.035
    s["leather"].add(*make_cylinder(0.015, 0.015, 0.18, segs=10, center=(0, gy, 0.0), rot=(math.pi / 2, 0, 0)))
    sh_pts = [(0.0, 0.36), (0.22, 0.27), (0.18, 0.04), (0.13, -0.22), (0.0, -0.36), (-0.13, -0.22), (-0.18, 0.04), (-0.22, 0.27)]
    s["basalt"].add(*make_extruded_polygon(sh_pts, 0.026, bevel=0.009, center=(0.052, gy + 0.02, 0.0), rot=(0, math.pi / 2, 0)))
    for sd in (-1, 1):
        wing_pts = [(sd * 0.03, -0.18), (sd * 0.24, 0.30), (sd * 0.16, 0.32), (sd * 0.02, -0.08)]
        s["brass"].add(*make_extruded_polygon(wing_pts, 0.014, bevel=0.004, center=(0.068, gy + 0.02, 0.0), rot=(0, math.pi / 2, 0)))
        _horn(s["brass"], [(0.052, gy + 0.26, sd * 0.17), (0.066, gy + 0.38, sd * 0.24), (0.042, gy + 0.48, sd * 0.20)], 0.015)
    s["magma"].add(*make_sphere(0.050, segs=10, rings=8, center=(0.074, gy + 0.05, 0.0), scale=(0.55, 1.35, 0.85)))
    s["brass"].add(*make_torus(0.068, 0.011, 20, 8, center=(0.068, gy + 0.05, 0.0), rot=(0, 0, math.pi / 2)))
    return out


# ==============================================================================
# ARQUETIPO 2 (LIGERA): CAZADOR ÉLFICO DE ESMERALDA & LUZ LUNAR
# ==============================================================================
def gen_dune_textures() -> Dict[str, bytes]:
    rng = np.random.default_rng(202)
    s = 256
    # Mithril estelar plateado-azulado con arabescos élficos
    sand = Image.new("RGB", (s, s), (206, 222, 232))
    sd = ImageDraw.Draw(sand)
    for y in range(0, s, 32):
        for x in range(0, s, 32):
            sd.arc([x, y, x + 32, y + 32], 0, 180, fill=(238, 246, 252), width=2)
            sd.arc([x + 4, y + 4, x + 28, y + 28], 180, 360, fill=(196, 168, 92), width=1)
    arr = np.clip(np.array(sand, dtype=np.int16) + rng.integers(-4, 5, (s, s, 1)), 0, 255).astype(np.uint8)
    sand = Image.fromarray(arr, "RGB")

    # Damasco esmeralda silvano profundo con nervaduras foliares doradas
    indigo = Image.new("RGB", (s, s), (16, 68, 50))
    idraw = ImageDraw.Draw(indigo)
    for y in range(0, s, 32):
        for x in range(0, s, 32):
            idraw.polygon([(x + 16, y + 2), (x + 28, y + 16), (x + 16, y + 30), (x + 4, y + 16)], outline=(42, 118, 86))
    iarr = np.clip(np.array(indigo, dtype=np.int16) + rng.integers(-4, 5, (s, s, 1)), 0, 255).astype(np.uint8)
    indigo = Image.fromarray(iarr, "RGB")

    # Cuero élfico verde bosque oscuro
    leather = Image.new("RGB", (s, s), (24, 52, 40))
    larr = np.clip(np.array(leather, dtype=np.int16) + rng.integers(-5, 6, (s, s, 1)), 0, 255).astype(np.uint8)
    leather = Image.fromarray(larr, "RGB")

    return {
        "sand_albedo": _img_to_png_bytes(sand),
        "indigo_albedo": _img_to_png_bytes(indigo),
        "leather_albedo": _img_to_png_bytes(leather),
    }


DUNE_MAT_SPECS = [
    ("sand", {"name": "Elven_MoonSilk", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.35, "roughnessFactor": 0.38}, "doubleSided": True}),
    ("indigo", {"name": "Elven_EmeraldDamask", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.14, "roughnessFactor": 0.56}, "doubleSided": True}),
    ("vermilion", {"name": "Elven_SylvanCloak", "pbrMetallicRoughness": {"baseColorFactor": [0.06, 0.28, 0.20, 1.0], "metallicFactor": 0.12, "roughnessFactor": 0.58}, "doubleSided": True}),
    ("leather", {"name": "Elven_ForestLeather", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}, "metallicFactor": 0.22, "roughnessFactor": 0.48}, "doubleSided": True}),
    ("brass", {"name": "Elven_LeafGold", "pbrMetallicRoughness": {"baseColorFactor": [0.92, 0.78, 0.42, 1.0], "metallicFactor": 0.94, "roughnessFactor": 0.18}, "doubleSided": True}),
    ("steel", {"name": "Elven_StarlightMithril", "pbrMetallicRoughness": {"baseColorFactor": [0.84, 0.92, 0.97, 1.0], "metallicFactor": 0.95, "roughnessFactor": 0.14}, "doubleSided": True}),
    ("turq", {"name": "Elven_EmeraldStar", "pbrMetallicRoughness": {"baseColorFactor": [0.18, 0.98, 0.66, 1.0], "metallicFactor": 0.15, "roughnessFactor": 0.08}, "emissiveFactor": [0.12, 0.90, 0.56], "doubleSided": True}),
]
DUNE_KEYS = ("sand", "indigo", "vermilion", "leather", "brass", "steel", "turq")


def _dp() -> Dict[str, MeshPart]:
    return _parts(*DUNE_KEYS)


def build_dune_helmet() -> Dict[str, Dict[str, MeshPart]]:
    """Tiara Alada de Mithril Lunar & Esmeralda (100% rostro abierto con alas de pluma élfica en las sienes)."""
    out = {"Head": _dp()}
    h = out["Head"]
    # Diadema élfica en V sobre la frente con gema esmeralda estelar
    h["steel"].add(*make_curved_shell(
        0.165, 0.202, 0.124, 0.134, 0.124, 0.134,
        -math.pi * 0.88, math.pi * 0.88, thickness=0.007, ny=6, na=22, peak_front_y=-0.022
    ))
    h["brass"].add(*make_curved_shell(
        0.172, 0.194, 0.126, 0.136, 0.126, 0.136,
        -1.15, 1.15, thickness=0.006, ny=4, na=18, peak_front_y=-0.018
    ))
    h["turq"].add(*make_sphere(0.016, segs=8, rings=6, center=(0, 0.182, 0.142), scale=(0.8, 1.35, 0.6)))
    # Alas élficas de mithril y oro desplegadas hacia arriba y atrás desde las sienes
    for sgn in (-1, 1):
        wing_pts = [(0.0, 0.0), (sgn * 0.04, 0.08), (sgn * 0.09, 0.16), (sgn * 0.05, 0.02)]
        h["steel"].add(*make_extruded_polygon(
            wing_pts, 0.008, bevel=0.003,
            center=(sgn * 0.105, 0.185, 0.02),
            rot=(-0.25, sgn * 0.45, 0.0)
        ))
        _horn(h["brass"], [(sgn * 0.098, 0.192, 0.03), (sgn * 0.145, 0.265, -0.02), (sgn * 0.135, 0.345, -0.08)], 0.014)
    return out


def build_dune_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    """Guardahombros Foliares de Mithril + Capa Larga Silvana Bordada Estilo Cuervo."""
    out = {"UpperArm_L": _dp(), "Torso": _dp(), "UpperArm_R": _dp()}
    tr = out["Torso"]
    # Gran capa larga fluida estilo cuervo en damasco esmeralda con cenefa bordada dorada
    tr["vermilion"].add(*make_curved_shell(
        -0.68, 0.24, 0.238, 0.195, 0.160, 0.145,
        math.pi * 0.44, math.pi * 1.56, thickness=0.007, ny=16, na=24, flare_bot=0.092, tatter_waves=5, tatter_amp=0.046
    ))
    tr["brass"].add(*make_curved_shell(
        -0.68, -0.56, 0.242, 0.200, 0.230, 0.190,
        math.pi * 0.46, math.pi * 1.54, thickness=0.006, ny=4, na=22, tatter_waves=5, tatter_amp=0.040
    ))
    for sgn in (-1, 1):
        u = out["UpperArm_L" if sgn > 0 else "UpperArm_R"]
        u["steel"].add(*make_curved_shell(
            -0.02, 0.085, 0.096, 0.096, 0.082, 0.082,
            -0.35 if sgn > 0 else -2.75, 2.75 if sgn > 0 else 0.35,
            thickness=0.010, flare_bot=0.020, peak_front_y=0.020
        ))
        _horn(u["brass"], [(sgn * 0.052, 0.068, 0.01), (sgn * 0.125, 0.145, -0.01), (sgn * 0.160, 0.225, -0.05)], 0.015)
        u["turq"].add(*make_sphere(0.014, segs=8, rings=6, center=(sgn * 0.088, 0.050, 0.042)))
    return out


def build_dune_torso() -> Dict[str, Dict[str, MeshPart]]:
    """Corselete-Peto Élfico de Mithril & Damasco Esmeralda con Tirantes de Cuero y Hebillas."""
    out = {"Torso": _dp()}
    t = out["Torso"]
    # Peto ligero más corto arriba (deja clavículas y pecho alto despejados para los tirantes)
    t["indigo"].add(*make_curved_shell(-0.16, 0.16, 0.164, 0.126, 0.205, 0.146, -math.pi, math.pi, thickness=0.010))
    t["steel"].add(*make_curved_shell(-0.15, 0.17, 0.172, 0.134, 0.212, 0.152, -1.60, 1.60, thickness=0.011, peak_front_y=0.018))
    t["turq"].add(*make_sphere(0.022, segs=8, rings=6, center=(0, 0.08, 0.166), scale=(0.8, 1.35, 0.55)))
    # Tirantes de cuero élfico con hebillas de oro foliar cruzando desde el peto sobre las clavículas y hombros hasta la espalda
    for sgn in (-1, 1):
        strap_pts = [
            (sgn * 0.082, 0.14, 0.152),
            (sgn * 0.105, 0.24, 0.118),
            (sgn * 0.118, 0.31, 0.010),
            (sgn * 0.102, 0.23, -0.115),
            (sgn * 0.075, 0.12, -0.142),
        ]
        t["leather"].add(*make_tube_curve(strap_pts, [0.013, 0.014, 0.014, 0.014, 0.013], segs=8))
        # Hebilla de oro sobre cada clavícula
        t["brass"].add(*make_box(0.032, 0.026, 0.014, center=(sgn * 0.095, 0.20, 0.136), rot=(-0.35, 0.0, 0.0)))
    # Correa pectoral horizontal entre ambos tirantes
    t["leather"].add(*make_tube_curve([(-0.095, 0.20, 0.136), (0.0, 0.195, 0.148), (0.095, 0.20, 0.136)], [0.009, 0.010, 0.009], segs=6))
    t["brass"].add(*make_box(0.028, 0.024, 0.014, center=(0.0, 0.195, 0.152)))
    return out


def build_dune_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Forearm_R": _dp(), "Wrist_R": _dp(), "Forearm_L": _dp(), "Wrist_L": _dp()}
    for sd, sgn in (("R", -1), ("L", 1)):
        fa, wr = out["Forearm_" + sd], out["Wrist_" + sd]
        fa["steel"].add(*make_curved_shell(-0.245, -0.03, 0.052, 0.052, 0.066, 0.066, -math.pi, math.pi, thickness=0.009, flare_top=0.012))
        fin_pts = [(0.0, -0.04), (sgn * 0.058, 0.01), (sgn * 0.042, 0.090), (0.0, 0.04)]
        fa["brass"].add(*make_extruded_polygon(fin_pts, 0.010, bevel=0.003, center=(sgn * 0.042, -0.04, -0.015)))
        fa["turq"].add(*make_sphere(0.012, segs=8, rings=6, center=(sgn * 0.054, -0.15, 0.0)))
        wr["brass"].add(*make_curved_shell(-0.072, 0.005, 0.046, 0.040, 0.048, 0.042, -math.pi, math.pi, thickness=0.006))
    return out


def build_dune_belt() -> Dict[str, Dict[str, MeshPart]]:
    """Ceñidor Silvano Foliar (corsé de hojas esculpidas, 3 gemas en frente y caderas, fajín ceñido, doble estola frontal amplia y faldones >=40% más grandes)."""
    out = {"Pelvis": _dp()}
    p = out["Pelvis"]
    # Corsé esculpido ceñido en cintura (sin abultar) en oro foliar y mithril
    p["brass"].add(*make_curved_shell(-0.072, 0.056, 0.168, 0.132, 0.172, 0.136, -math.pi, math.pi, thickness=0.010, ny=6, na=24, peak_front_y=-0.028))
    p["steel"].add(*make_curved_shell(-0.052, 0.036, 0.172, 0.136, 0.176, 0.140, -1.65, 1.65, thickness=0.008, ny=4, na=18))
    # Hojas esculpidas más amplias radiando desde el broche central y las caderas (+45%)
    leaf_pts = [(0.0, -0.068), (0.031, -0.014), (0.020, 0.058), (0.0, 0.078), (-0.020, 0.058), (-0.031, -0.014)]
    for ang, sc in ((0.0, 1.12), (-0.42, 0.92), (0.42, 0.92), (-1.15, 0.88), (1.15, 0.88)):
        px = math.sin(ang) * 0.178
        pz = math.cos(ang) * 0.144
        p["brass"].add(*make_extruded_polygon(
            [(x * sc, y * sc) for x, y in leaf_pts], 0.010, bevel=0.003,
            center=(px, -0.006, pz), rot=(0.0, ang, -ang * 0.25)
        ))
    # 3 gemas ovaladas ampliadas (centro + ambas caderas)
    p["turq"].add(*make_sphere(0.025, segs=10, rings=8, center=(0, -0.006, 0.156), scale=(0.84, 1.36, 0.58)))
    for sgn in (-1, 1):
        p["turq"].add(*make_sphere(0.021, segs=8, rings=6, center=(sgn * 0.170, -0.012, 0.065), scale=(0.65, 1.28, 0.85)))
    # Fajín de tela drapeada ajustado bajo el corsé
    p["vermilion"].add(*make_curved_shell(-0.125, -0.058, 0.184, 0.148, 0.178, 0.142, -math.pi, math.pi, thickness=0.008, ny=5, na=24))
    # Doble estola/cinta frontal amplia (+45% longitud y anchura) con bordado foliar
    ribbon_pts = [(0.0, -0.62), (0.036, -0.31), (0.038, -0.04), (-0.038, -0.04), (-0.036, -0.31)]
    for sgn in (-1, 1):
        p["sand"].add(*make_extruded_polygon(
            ribbon_pts, 0.007, bevel=0.002,
            center=(sgn * 0.052, -0.06, 0.158), rot=(-0.06, sgn * 0.08, sgn * -0.05)
        ))
        p["brass"].add(*make_tube_curve([
            (sgn * 0.052, -0.10, 0.165), (sgn * 0.062, -0.34, 0.176), (sgn * 0.052, -0.59, 0.184)
        ], [0.006, 0.0065, 0.0055], segs=6))
    # Faldones esmeralda asimétricos ampliados con holgura sobre muslos y cenefa dorada
    p["indigo"].add(*make_curved_shell(
        -0.70, -0.06, 0.262, 0.222, 0.184, 0.146,
        0.38, math.tau - 0.38, thickness=0.008, ny=14, na=28, flare_bot=0.075, tatter_waves=4, tatter_amp=0.062
    ))
    p["brass"].add(*make_curved_shell(
        -0.70, -0.63, 0.266, 0.226, 0.258, 0.220,
        0.42, math.tau - 0.42, thickness=0.007, ny=3, na=24
    ))
    return out


def build_dune_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _dp()
        for row in range(2):
            yc = -0.055 - row * 0.088
            scale_pts = [(0.0, -0.065), (0.054, 0.0), (0.045, 0.044), (-0.045, 0.044), (-0.054, 0.0)]
            p["steel" if row == 0 else "indigo"].add(*make_extruded_polygon(
                scale_pts, 0.012, bevel=0.004,
                center=(sgn * 0.114, yc, 0.015),
                rot=(0.0, sgn * math.pi * 0.42, sgn * -0.14)
            ))
        out[side] = p
    return out


def build_dune_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _dp()
        p["indigo"].add(*make_curved_shell(-0.38, -0.06, 0.094, 0.094, 0.116, 0.116, -math.pi, math.pi, thickness=0.009, peak_front_y=-0.015))
        p["steel"].add(*make_curved_shell(-0.35, -0.08, 0.098, 0.098, 0.118, 0.118, -1.2, 1.2, thickness=0.008))
        p["brass"].add(*make_tube_curve([(sgn * 0.06, -0.12, 0.08), (sgn * 0.11, -0.22, 0.01), (sgn * 0.07, -0.32, -0.05)], [0.007, 0.008, 0.007], segs=6))
        out[side] = p
    return out


def build_dune_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _dp(), "Knee_R": _dp()}
    for side, sgn in (("Knee_L", 1), ("Knee_R", -1)):
        k = out[side]
        k["steel"].add(*make_curved_shell(-0.044, 0.044, 0.084, 0.084, 0.084, 0.084, -1.6, 1.6, thickness=0.008))
        wing_pts = [(0.0, -0.062), (0.052, -0.01), (0.064, 0.042), (0.0, 0.068), (-0.048, 0.025), (-0.042, -0.02)]
        k["brass"].add(*make_extruded_polygon(wing_pts, 0.014, bevel=0.004, center=(sgn * 0.01, 0, 0.084), rot=(0, sgn * 0.18, 0)))
        k["turq"].add(*make_sphere(0.015, segs=8, rings=6, center=(0, 0, 0.094)))
    return out


def build_dune_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side in ("L", "R"):
        sh, ak = _dp(), _dp()
        sh["steel"].add(*make_curved_shell(-0.44, -0.06, 0.068, 0.070, 0.090, 0.094, -math.pi, math.pi, thickness=0.009, peak_front_y=0.022))
        sh["brass"].add(*make_curved_shell(-0.14, -0.05, 0.090, 0.094, 0.102, 0.106, -math.pi, math.pi, thickness=0.008, flare_top=0.016))
        _boot_foot({"leather": ak["leather"], "brass": ak["brass"], "indigo": ak["steel"]},
                   "leather", "indigo", "brass", "brass", foot_w=0.100, toe_len=0.176, pointed=0.32)
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_dune_weapon() -> Dict[str, Dict[str, MeshPart]]:
    """Hoja Lunar Élfica de Esmeralda."""
    out = {"Wrist_R": _dp()}
    w = out["Wrist_R"]
    gy = -0.035
    w["leather"].add(*make_cylinder(0.014, 0.015, 0.24, segs=12, center=(0, gy, 0.0), rot=(math.pi / 2, 0, 0)))
    _horn(w["brass"], [(0, gy, -0.12), (0.020, gy, -0.165), (0.042, gy, -0.195)], 0.015)
    guard_pts = [(-0.09, 0.17), (0.0, 0.105), (0.15, 0.22), (0.0, 0.15)]
    w["brass"].add(*make_extruded_polygon(guard_pts, 0.016, bevel=0.005, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    w["turq"].add(*make_sphere(0.019, segs=8, rings=6, center=(0, gy, 0.135)))
    scim_pts = [(-0.012, 0.13), (0.032, 0.50), (0.115, 1.00), (0.042, 1.16), (0.058, 0.92), (0.016, 0.56), (-0.034, 0.13)]
    w["steel"].add(*make_extruded_polygon(scim_pts, 0.009, bevel=0.004, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    w["turq"].add(*make_tube_curve([(-0.018, gy, 0.18), (0.012, gy, 0.50), (0.052, gy, 0.80), (0.068, gy, 1.04)], [0.0042, 0.0042, 0.0038, 0.003], segs=6))
    return out


def build_dune_shield() -> Dict[str, Dict[str, MeshPart]]:
    """Broquel Estelar de Mithril & Daga Élfica."""
    out = {"Wrist_L": _dp()}
    s = out["Wrist_L"]
    gy = -0.035
    s["leather"].add(*make_cylinder(0.014, 0.015, 0.18, segs=10, center=(0, gy, 0), rot=(math.pi / 2, 0, 0)))
    for d in (-1, 1):
        jam_pts = [(-0.018, d * 0.08), (-0.036, d * 0.24), (0.018, d * 0.42), (0.022, d * 0.08)]
        s["steel"].add(*make_extruded_polygon(jam_pts, 0.008, bevel=0.003, center=(0.02, gy, 0), rot=(-math.pi / 2, 0, 0)))
    star_pts = []
    for i in range(8):
        ang = (i / 8) * math.tau
        r = 0.152 if i % 2 == 0 else 0.102
        star_pts.append((math.cos(ang) * r, math.sin(ang) * r * 1.16))
    s["steel"].add(*make_extruded_polygon(star_pts, 0.014, bevel=0.004, center=(0.044, gy + 0.02, 0.02), rot=(0, math.pi / 2, 0)))
    s["brass"].add(*make_torus(0.120, 0.009, 24, 8, center=(0.054, gy + 0.02, 0.02), rot=(0, 0, math.pi / 2)))
    s["turq"].add(*make_sphere(0.034, segs=8, rings=6, center=(0.062, gy + 0.02, 0.02), scale=(0.55, 1.25, 0.95)))
    return out


# ==============================================================================
# ARQUETIPO 3 (MAGO): ARCHIMAGO SERAFÍN DEL ECLIPSE & TERCIOPELO ASTRAL
# ==============================================================================
def gen_astrolabe_textures() -> Dict[str, bytes]:
    rng = np.random.default_rng(303)
    s = 256
    # Terciopelo amatista imperial con constelaciones en oro rosa y estrellas
    velvet = Image.new("RGB", (s, s), (40, 16, 68))
    vd = ImageDraw.Draw(velvet)
    for k in range(32, s, 64):
        vd.ellipse([k - 24, k - 24, k + 24, k + 24], outline=(206, 152, 118), width=1)
    for _ in range(80):
        sx, sy = int(rng.integers(0, s)), int(rng.integers(0, s))
        vd.point((sx, sy), fill=(245, 232, 255))
    arr = np.clip(np.array(velvet, dtype=np.int16) + rng.integers(-4, 5, (s, s, 1)), 0, 255).astype(np.uint8)
    velvet = Image.fromarray(arr, "RGB")

    # Seda perlada astral blanca-lavanda
    silk = Image.new("RGB", (s, s), (232, 228, 244))
    sarr = np.clip(np.array(silk, dtype=np.int16) + rng.integers(-4, 5, (s, s, 1)), 0, 255).astype(np.uint8)
    silk = Image.fromarray(sarr, "RGB")

    # Oro rosa celestial espejo con grabados solares
    bronze = Image.new("RGB", (s, s), (222, 172, 134))
    bdraw = ImageDraw.Draw(bronze)
    for k in range(24, s, 48):
        bdraw.ellipse([k - 16, 24, k + 16, 56], outline=(250, 214, 178), width=2)

    return {
        "velvet_albedo": _img_to_png_bytes(velvet),
        "silk_albedo": _img_to_png_bytes(silk),
        "bronze_albedo": _img_to_png_bytes(bronze),
    }


ASTRO_MAT_SPECS = [
    ("velvet", {"name": "Seraph_AmethystVelvet", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.16, "roughnessFactor": 0.56}, "doubleSided": True}),
    ("silk", {"name": "Seraph_StarlightSilk", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "metallicFactor": 0.28, "roughnessFactor": 0.32}, "doubleSided": True}),
    ("bronze", {"name": "Seraph_RoseGold", "pbrMetallicRoughness": {"baseColorTexture": {"index": 2}, "metallicFactor": 0.95, "roughnessFactor": 0.16}, "doubleSided": True}),
    ("silver", {"name": "Seraph_PlatinumSilver", "pbrMetallicRoughness": {"baseColorFactor": [0.90, 0.92, 0.98, 1.0], "metallicFactor": 0.96, "roughnessFactor": 0.14}, "doubleSided": True}),
    ("parchment", {"name": "Seraph_IlluminatedParchment", "pbrMetallicRoughness": {"baseColorFactor": [0.95, 0.90, 0.78, 1.0], "metallicFactor": 0.05, "roughnessFactor": 0.58}, "doubleSided": True}),
    ("astral", {"name": "Seraph_AmethystStar", "pbrMetallicRoughness": {"baseColorFactor": [0.76, 0.56, 1.0, 1.0], "metallicFactor": 0.15, "roughnessFactor": 0.08}, "emissiveFactor": [0.70, 0.44, 1.0], "doubleSided": True}),
    ("amber", {"name": "Seraph_SolarHalo", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.84, 0.42, 1.0], "metallicFactor": 0.25, "roughnessFactor": 0.12}, "emissiveFactor": [0.95, 0.72, 0.25], "doubleSided": True}),
]
ASTRO_KEYS = ("velvet", "silk", "bronze", "silver", "parchment", "astral", "amber")


def _ap() -> Dict[str, MeshPart]:
    return _parts(*ASTRO_KEYS)


def build_astro_helmet() -> Dict[str, Dict[str, MeshPart]]:
    """Corona Serafín del Eclipse (100% rostro abierto, sin aros)."""
    out = {"Head": _ap()}
    h = out["Head"]
    h["bronze"].add(*make_curved_shell(
        0.166, 0.212, 0.124, 0.132, 0.128, 0.136,
        -math.pi * 0.92, math.pi * 0.92, thickness=0.008, ny=6, na=22, peak_front_y=-0.024
    ))
    h["astral"].add(*make_sphere(0.017, segs=8, rings=6, center=(0, 0.192, 0.142), scale=(0.8, 1.35, 0.6)))
    # Alas de serafín en las sienes (sin halos tras la cabeza)
    for sgn in (-1, 1):
        _horn(h["bronze"], [(sgn * 0.095, 0.195, 0.02), (sgn * 0.160, 0.285, -0.02), (sgn * 0.148, 0.395, -0.08)], 0.018)
        _horn(h["silver"], [(sgn * 0.072, 0.205, 0.04), (sgn * 0.118, 0.295, 0.00), (sgn * 0.105, 0.375, -0.05)], 0.012)
    return out


def build_astro_shoulders() -> Dict[str, Dict[str, MeshPart]]:
    """Guardahombros Serafín + Gran Capa Larga de Terciopelo Amatista Bordada en Oro Rosa (sin aros ni alas pequeñas)."""
    out = {"Torso": _ap(), "UpperArm_L": _ap(), "UpperArm_R": _ap()}
    tr = out["Torso"]
    # Gran capa larga fluida estilo cuervo en terciopelo amatista con cenefa bordada en oro rosa y plata
    tr["velvet"].add(*make_curved_shell(
        -0.70, 0.23, 0.242, 0.198, 0.162, 0.146,
        math.pi * 0.44, math.pi * 1.56, thickness=0.007, ny=16, na=24, flare_bot=0.098, tatter_waves=5, tatter_amp=0.046
    ))
    tr["bronze"].add(*make_curved_shell(
        -0.70, -0.57, 0.246, 0.202, 0.234, 0.192,
        math.pi * 0.46, math.pi * 1.54, thickness=0.006, ny=4, na=22, tatter_waves=5, tatter_amp=0.040
    ))
    for sgn in (-1, 1):
        u = out["UpperArm_L" if sgn > 0 else "UpperArm_R"]
        u["bronze"].add(*make_curved_shell(-0.02, 0.08, 0.096, 0.096, 0.084, 0.084, -0.3 if sgn > 0 else -2.5, 2.5 if sgn > 0 else 0.3, thickness=0.009, peak_front_y=0.022))
        _horn(u["silver"], [(sgn * 0.055, 0.075, 0.0), (sgn * 0.125, 0.165, -0.02), (sgn * 0.145, 0.265, -0.05)], 0.016)
    return out


def build_astro_torso() -> Dict[str, Dict[str, MeshPart]]:
    """Túnica Real Cruzada de Terciopelo Amatista, Seda Perlada & Bandoleras Cruzadas Estrechas."""
    out = {"Torso": _ap()}
    t = out["Torso"]
    t["silk"].add(*make_curved_shell(-0.16, 0.27, 0.164, 0.128, 0.210, 0.150, -math.pi, math.pi, thickness=0.008))
    t["velvet"].add(*make_curved_shell(-0.17, 0.28, 0.172, 0.134, 0.220, 0.158, 0.36, math.tau - 0.36, thickness=0.010, flare_bot=0.018))
    # Solapas cruzadas en V de la túnica de mago
    for sgn in (-1, 1):
        t["bronze"].add(*make_tube_curve([(sgn * 0.095, 0.26, 0.135), (sgn * 0.032, 0.10, 0.168), (-sgn * 0.045, -0.10, 0.152)], [0.008, 0.009, 0.007], segs=8))
    # Bandolera estrecha cruzada sobre el pecho con hebilla y viales arcanos
    t["bronze"].add(*make_tube_curve([(-0.125, 0.24, 0.125), (0.0, 0.08, 0.172), (0.145, -0.09, 0.115), (0.0, 0.06, -0.162), (-0.125, 0.24, -0.115)], [0.007, 0.007, 0.007, 0.007, 0.007], segs=8))
    t["silver"].add(*make_box(0.024, 0.022, 0.012, center=(0.0, 0.08, 0.176), rot=(0.0, 0.0, -0.58)))
    for v_idx in range(3):
        vx = -0.06 + v_idx * 0.035
        vy = 0.14 - v_idx * 0.035
        t["astral"].add(*make_cylinder(0.006, 0.006, 0.026, segs=8, center=(vx, vy, 0.172)))
    return out


def build_astro_gloves() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Forearm_R": _ap(), "Wrist_R": _ap(), "Forearm_L": _ap(), "Wrist_L": _ap()}
    for sd, sgn in (("R", -1), ("L", 1)):
        fa, wr = out["Forearm_" + sd], out["Wrist_" + sd]
        fa["velvet"].add(*make_curved_shell(
            -0.25, -0.02, 0.092, 0.092, 0.058, 0.058,
            -math.pi, math.pi, thickness=0.008, flare_bot=0.028
        ))
        fa["silk"].add(*make_curved_shell(-0.245, -0.18, 0.096, 0.096, 0.088, 0.088, -math.pi, math.pi, thickness=0.006))
        fa["bronze"].add(*make_torus(0.098, 0.005, 20, 6, center=(0, -0.21, 0)))
        wr["astral"].add(*make_sphere(0.012, segs=6, rings=5, center=(sgn * 0.026, -0.025, 0)))
    return out


def build_astro_belt() -> Dict[str, Dict[str, MeshPart]]:
    """Cíngulo Rúnico Astral & Faldón Catedralicio (cintura ceñida, chevrones, amatistas, viales, estandarte y faldón ampliados >=40%)."""
    out = {"Pelvis": _ap()}
    p = out["Pelvis"]
    # Cinturón base ceñido en cintura (sin abultar)
    p["silver"].add(*make_curved_shell(-0.052, 0.054, 0.174, 0.138, 0.176, 0.140, -math.pi, math.pi, thickness=0.009, ny=6, na=24, peak_front_y=-0.026))
    # 3 placas frontales escalonadas en V de plata astral ampliadas (+45%)
    for tier, dy, sc in ((0, 0.028, 1.38), (1, -0.028, 1.24), (2, -0.080, 1.08)):
        chev_pts = [(0.0, -0.044 * sc), (0.094 * sc, 0.012 * sc), (0.080 * sc, 0.036 * sc), (0.0, -0.008 * sc), (-0.080 * sc, 0.036 * sc), (-0.094 * sc, 0.012 * sc)]
        p["silver" if tier != 1 else "bronze"].add(*make_extruded_polygon(
            chev_pts, 0.013, bevel=0.003, center=(0, dy, 0.148 + tier * 0.005)
        ))
        p["astral"].add(*make_sphere(0.014 * sc, segs=8, rings=6, center=(0, dy - 0.012 * sc, 0.162 + tier * 0.005)))
    # Broche superior ampliado con cabujón ovalado de amatista y marco esculpido
    p["bronze"].add(*make_sphere(0.032, segs=10, rings=8, center=(0, 0.030, 0.156), scale=(0.95, 1.40, 0.45)))
    p["astral"].add(*make_sphere(0.026, segs=10, rings=8, center=(0, 0.030, 0.162), scale=(0.82, 1.32, 0.58)))
    # Cadenas de plata arqueadas + medallones sólidos de astrolabio + 3 viales ampliados (+45%)
    for sgn in (-1, 1):
        p["silver"].add(*make_tube_curve([
            (sgn * 0.072, -0.015, 0.152), (sgn * 0.155, -0.135, 0.132), (sgn * 0.205, -0.048, 0.025)
        ], [0.007, 0.0075, 0.007], segs=7))
        mx, my, mz = sgn * 0.178, -0.118, 0.092
        p["silver"].add(*make_cylinder(0.032, 0.032, 0.009, segs=12, center=(mx, my, mz), rot=(0, 0, sgn * 0.95)))
        p["astral"].add(*make_sphere(0.014, segs=8, rings=6, center=(mx, my, mz + 0.004)))
    for vx, vy, vz in ((-0.068, -0.155, 0.168), (0.0, -0.180, 0.174), (0.068, -0.155, 0.168)):
        p["silver"].add(*make_cylinder(0.013, 0.013, 0.014, segs=8, center=(vx, vy + 0.026, vz)))
        p["astral"].add(*make_cylinder(0.010, 0.007, 0.048, segs=8, center=(vx, vy, vz)))
    # Gran estandarte frontal puntiagudo de terciopelo (+45%) y borla lavanda
    banner_pts = [(0.0, -0.64), (0.092, -0.39), (0.104, -0.09), (-0.104, -0.09), (-0.092, -0.39)]
    p["velvet"].add(*make_extruded_polygon(banner_pts, 0.011, bevel=0.003, center=(0, -0.06, 0.162), rot=(-0.07, 0, 0)))
    p["silver"].add(*make_sphere(0.016, segs=8, rings=6, center=(0, -0.71, 0.204)))
    p["silk"].add(*make_cylinder(0.008, 0.023, 0.088, segs=8, center=(0, -0.758, 0.208)))
    # Faldón catedralicio plisado ampliado con holgura sobre muslos
    p["silk"].add(*make_curved_shell(-0.74, -0.05, 0.258, 0.220, 0.174, 0.140, -math.pi, math.pi, thickness=0.006, ny=12, na=24, flare_bot=0.065))
    for a0, a1 in ((0.38, math.pi - 0.12), (math.pi + 0.12, math.tau - 0.38)):
        p["velvet"].add(*make_curved_shell(
            -0.78, -0.04, 0.278, 0.238, 0.180, 0.144,
            a0, a1, thickness=0.009, ny=14, na=16, flare_bot=0.082, tatter_waves=4, tatter_amp=0.042
        ))
        p["bronze"].add(*make_curved_shell(-0.78, -0.70, 0.282, 0.242, 0.272, 0.234, a0 + 0.04, a1 - 0.04, thickness=0.007))
    return out


def build_astro_tassets() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _ap()
        scale_pts = [(0.0, -0.068), (0.052, 0.0), (0.044, 0.046), (-0.044, 0.046), (-0.052, 0.0)]
        p["silk"].add(*make_extruded_polygon(scale_pts, 0.010, bevel=0.003, center=(sgn * 0.114, -0.06, 0.012), rot=(0, sgn * math.pi * 0.42, sgn * -0.14)))
        for offz in (-0.032, 0.032):
            p["bronze"].add(*make_cylinder(0.003, 0.003, 0.20, segs=6, center=(sgn * 0.118, -0.15, offz)))
            p["amber"].add(*make_sphere(0.016, segs=8, rings=6, center=(sgn * 0.118, -0.255, offz), scale=(0.8, 1.4, 0.8)))
        p["bronze"].add(*make_torus(0.044, 0.006, 18, 6, center=(sgn * 0.120, -0.15, 0.0), rot=(0, 0, math.pi / 2)))
        p["astral"].add(*make_sphere(0.016, segs=8, rings=6, center=(sgn * 0.122, -0.15, 0.0)))
        out[side] = p
    return out


def build_astro_cuisses() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side, sgn in (("Thigh_L", 1), ("Thigh_R", -1)):
        p = _ap()
        p["velvet"].add(*make_curved_shell(-0.38, -0.06, 0.092, 0.092, 0.112, 0.112, -math.pi, math.pi, thickness=0.009, peak_front_y=-0.015))
        p["bronze"].add(*make_tube_curve([(sgn * 0.05, -0.10, 0.08), (sgn * 0.106, -0.22, 0.0), (sgn * 0.06, -0.34, -0.05)], [0.007, 0.008, 0.007], segs=6))
        out[side] = p
    return out


def build_astro_knees() -> Dict[str, Dict[str, MeshPart]]:
    out = {"Knee_L": _ap(), "Knee_R": _ap()}
    for side, sgn in (("Knee_L", 1), ("Knee_R", -1)):
        k = out[side]
        cres_pts = [(0.0, -0.060), (0.048, -0.01), (0.058, 0.038), (0.0, 0.065), (-0.044, 0.025)]
        k["bronze"].add(*make_extruded_polygon(cres_pts, 0.012, bevel=0.004, center=(sgn * 0.008, 0, 0.086), rot=(0, sgn * 0.18, 0)))
        _horn(k["silver"], [(sgn * 0.042, 0.015, 0.072), (sgn * 0.078, 0.048, 0.048), (sgn * 0.095, 0.092, 0.015)], 0.009, segs=6)
        k["astral"].add(*make_sphere(0.015, segs=8, rings=6, center=(0, 0, 0.096)))
    return out


def build_astro_boots() -> Dict[str, Dict[str, MeshPart]]:
    out = {}
    for side in ("L", "R"):
        sh, ak = _ap(), _ap()
        sh["velvet"].add(*make_curved_shell(-0.44, -0.10, 0.068, 0.070, 0.088, 0.092, -math.pi, math.pi, thickness=0.009, peak_front_y=0.022))
        sh["bronze"].add(*make_curved_shell(-0.18, -0.08, 0.086, 0.090, 0.098, 0.102, -math.pi, math.pi, thickness=0.008, flare_top=0.015))
        _boot_foot({"velvet": ak["velvet"], "bronze": ak["bronze"], "silver": ak["silver"]},
                   "silver", "velvet", "bronze", "silver", foot_w=0.096, toe_len=0.172, pointed=0.36)
        ak["astral"].add(*make_sphere(0.013, segs=6, rings=5, center=(0, 0.012, 0.092)))
        out["Shin_" + side] = sh
        out["Ankle_" + side] = ak
    return out


def build_astro_weapon() -> Dict[str, Dict[str, MeshPart]]:
    """Gran Cetro Planetario del Serafín Astral."""
    out = {"Wrist_R": _ap()}
    w = out["Wrist_R"]
    gy = -0.035
    w["bronze"].add(*make_tube_curve([
        (0, gy, -0.48), (0.008, gy + 0.006, -0.05), (-0.008, gy - 0.006, 0.42), (0.006, gy + 0.004, 0.86), (0, gy, 1.10)
    ], [0.014, 0.014, 0.014, 0.014, 0.014], segs=12))
    w["velvet"].add(*make_cylinder(0.018, 0.018, 0.26, segs=12, center=(0, gy, 0.01), rot=(math.pi / 2, 0, 0)))
    # Alas de serafín a ambos lados del planetario en el cabezal
    for sd in (-1, 1):
        wing_pts = [(sd * 0.02, 1.02), (sd * 0.20, 1.12), (sd * 0.26, 1.32), (sd * 0.14, 1.24), (sd * 0.02, 1.14)]
        w["bronze"].add(*make_extruded_polygon(wing_pts, 0.008, bevel=0.003, center=(0, gy, 0), rot=(-math.pi / 2, 0, 0)))
    cz = 1.24
    w["bronze"].add(*make_torus(0.125, 0.008, 28, 8, center=(0.0, gy, cz), rot=(math.pi / 2, 0.0, 0)))
    w["silver"].add(*make_torus(0.092, 0.006, 24, 6, center=(0.0, gy, cz), rot=(0.65, -0.5, 0.35)))
    w["amber"].add(*make_torus(0.060, 0.005, 20, 6, center=(0.0, gy, cz), rot=(-0.7, 0.6, 0.8)))
    w["astral"].add(*make_sphere(0.038, segs=10, rings=8, center=(0.0, gy, cz)))
    return out


def build_astro_shield() -> Dict[str, Dict[str, MeshPart]]:
    """Grimorio Iluminado del Eclipse & Orbe Serafín."""
    out = {"Wrist_L": _ap()}
    s = out["Wrist_L"]
    gy = -0.035
    s["bronze"].add(*make_cylinder(0.012, 0.012, 0.14, segs=8, center=(0, gy, 0), rot=(math.pi / 2, 0, 0)))
    s["bronze"].add(*make_cylinder(0.016, 0.016, 0.25, segs=8, center=(0.032, gy + 0.04, 0.0), rot=(math.pi / 2, 0, 0)))
    for sd in (-1, 1):
        s["velvet"].add(*make_box(0.012, 0.130, 0.24, center=(0.048, gy + 0.04, sd * 0.064), rot=(0, -sd * 0.42, 0), bevel=0.004))
        s["parchment"].add(*make_box(0.014, 0.118, 0.21, center=(0.056, gy + 0.04, sd * 0.060), rot=(0, -sd * 0.42, 0)))
        _horn(s["bronze"], [(0.068, gy + 0.15, sd * 0.11), (0.078, gy + 0.22, sd * 0.15), (0.068, gy + 0.27, sd * 0.11)], 0.0085)
    ox, oy, oz = 0.148, gy + 0.06, 0.0
    s["astral"].add(*make_sphere(0.034, segs=10, rings=8, center=(ox, oy, oz)))
    s["bronze"].add(*make_torus(0.070, 0.0055, 24, 6, center=(ox, oy, oz), rot=(0.55, 0.4, 0.2)))
    s["silver"].add(*make_torus(0.052, 0.0045, 20, 6, center=(ox, oy, oz), rot=(-0.6, 0.8, 0.5)))
    _horn(s["silver"], [(ox, oy + 0.12, -0.060), (ox + 0.008, oy + 0.17, 0.0), (ox, oy + 0.12, 0.060)], 0.009)
    return out


# ==============================================================================
# CATÁLOGOS
# ==============================================================================
NEW_CATALOG_ENTRIES = {
    "underlayer": [
        {"idx": 18, "name": "Escamas Carmesí & Brocado de Oro Imperial (Soberano Dracónico)"},
        {"idx": 19, "name": "Seda Esmeralda Silvana & Filigrana Lunar (Cazador Élfico)"},
        {"idx": 20, "name": "Terciopelo Amatista Real & Seda Perlada (Archimago Serafín)"},
    ],
    "helmet": [
        {"idx": 11, "name": "Corona-Visor Imperial de Dragón Carmesí & Plata Negra (Soberano)"},
        {"idx": 12, "name": "Tiara Alada de Mithril Lunar & Esmeralda (Cazador Élfico)"},
        {"idx": 13, "name": "Corona Serafín del Eclipse & Halo Radiante (Archimago)"},
    ],
    "shoulders": [
        {"idx": 14, "name": "Hombreras Aladas de Dragón & Media Capa Imperial (Soberano)"},
        {"idx": 15, "name": "Guardahombros Foliares de Mithril & Manto Élfico (Cazador)"},
        {"idx": 16, "name": "6 Alas de Serafín en Oro Rosa & Planetario Dorsal (Archimago)"},
    ],
    "torso": [
        {"idx": 22, "name": "Coraza Real de Escamas de Dragón & Filigrana de Oro (Pesada)"},
        {"idx": 23, "name": "Corselete-Peto Élfico de Mithril & Damasco Esmeralda (Ligera)"},
        {"idx": 24, "name": "Vestidura Real de Terciopelo Amatista & Cuello Serafín (Mago)"},
    ],
    "gloves": [
        {"idx": 8,  "name": "Guanteletes de Garras Dracónicas en Plata Negra & Rubí (Soberano)"},
        {"idx": 9,  "name": "Brazales Foliares de Mithril & Filigrana de Oro (Cazador Élfico)"},
        {"idx": 10, "name": "Brazales de Oro Rosa & Mangas Catedralicias de Seda (Archimago)"},
    ],
    "belt": [
        {"idx": 10, "name": "Fajín Real Dracónico & Faldón Heráldico Carmesí (Soberano)"},
        {"idx": 11, "name": "Ceñidor Élfico de Mithril & Faldones de Guardabosques (Cazador)"},
        {"idx": 12, "name": "Cíngulo Serafín & Faldón Catedralicio de Terciopelo (Archimago)"},
    ],
    "tassets": [
        {"idx": 11, "name": "Escarcelas Aladas de Dragón en Plata Negra & Oro (Soberano)"},
        {"idx": 12, "name": "Escarcelas Foliares de Mithril & Esmeraldas (Cazador Élfico)"},
        {"idx": 13, "name": "Escarcelas Astrales con Péndulos de Incensario (Archimago)"},
    ],
    "cuisses": [
        {"idx": 10, "name": "Musleras Esculpidas de Plata Negra & Nervios de Oro (Soberano)"},
        {"idx": 11, "name": "Musleras Élficas de Mithril & Damasco Esmeralda (Cazador)"},
        {"idx": 12, "name": "Musleras de Seda Perlada & Filigrana de Oro Rosa (Archimago)"},
    ],
    "knees": [
        {"idx": 10, "name": "Rodilleras de Garra Dracónica & Rubí de Sangre (Soberano)"},
        {"idx": 11, "name": "Rodilleras Aladas de Mithril Lunar & Esmeralda (Cazador)"},
        {"idx": 12, "name": "Rodilleras de Media Luna Serafín en Oro Rosa (Archimago)"},
    ],
    "boots": [
        {"idx": 15, "name": "Escarpes Imperiales de Plata Negra & Espolón de Dragón (Soberano)"},
        {"idx": 16, "name": "Botas Altas Élficas de Mithril & Filigrana de Hoja (Cazador)"},
        {"idx": 17, "name": "Escarpes Reales de Terciopelo Amatista & Oro Rosa (Archimago)"},
    ],
    "weapon": [
        {"idx": 83, "name": "Mandoble Imperial del Soberano Dracónico (Soberano)", "familia": "mandoble", "nivel": 50},
        {"idx": 84, "name": "Hoja Lunar Élfica de Esmeralda (Cazador Élfico)", "familia": "espada", "nivel": 50},
        {"idx": 85, "name": "Gran Cetro Planetario del Serafín Astral (Archimago)", "familia": "cetro", "nivel": 50},
    ],
    "shield": [
        {"idx": 13, "name": "Égida Alada del Dragón Carmesí (Soberano)", "nivel": 50},
        {"idx": 14, "name": "Broquel Estelar de Mithril & Daga Élfica (Cazador)", "nivel": 50},
        {"idx": 15, "name": "Grimorio Iluminado del Eclipse & Orbe Serafín (Archimago)", "nivel": 50},
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
            "Soberano Dracónico Carmesí & Plata Negra (Pesada)",
            gen_basalt_textures(),
            ("basalt_albedo", "iron_albedo", "copper_albedo"),
            BASALT_MAT_SPECS,
            "DragonSovereign",
            "Sovereign",
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
            "Cazador Élfico de Esmeralda & Luz Lunar (Ligera)",
            gen_dune_textures(),
            ("sand_albedo", "indigo_albedo", "leather_albedo"),
            DUNE_MAT_SPECS,
            "ElvenLunar",
            "Elven",
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
            "Archimago Serafín del Eclipse & Terciopelo Astral (Mago)",
            gen_astrolabe_textures(),
            ("velvet_albedo", "silk_albedo", "bronze_albedo"),
            ASTRO_MAT_SPECS,
            "SeraphimEclipse",
            "Seraph",
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
