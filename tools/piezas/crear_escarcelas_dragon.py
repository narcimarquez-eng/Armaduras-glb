#!/usr/bin/env python3
"""Generador en Python de las Escarcelas y el Cinturón del Dragón Rúnico.

Construye la pieza de la lámina de referencia `tools/piezas/referencias/escarcelas_dragon_runico.jpg`
usando SUS PROPIAS TEXTURAS, recortadas de la lámina (con el fondo de papel eliminado por transparencia):

  - piezas/tassets/15.glb  (huesos Thigh_L / Thigh_R):
      placa del dragón rúnico (vista lateral de la lámina) curvada sobre la cara externa del muslo,
      con su cota de malla y flecos de cuero; faldón trasero de cuero oscuro con correas cruzadas
      (vista trasera de la lámina); cuernos 3D curvos y correas de suspensión hacia el cinturón.
  - piezas/belt/13.glb     (hueso Torso):
      cinturón de cuero tachonado (textura del cinturón de la lámina) con la placa-hebilla frontal
      de la gema cian y la hebilla trasera.

Geometría en el espacio de los huesos del esqueleto `CreadorHumano_hombre` (mismo que el resto de piezas).
Actualiza `piezas/catalogo-estudio.json` y regenera `piezas/catalogo.json`.

Uso:
    python3 tools/piezas/crear_escarcelas_dragon.py
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
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
from crear_dark_crystal import (  # noqa: E402  (reutiliza esqueleto y utilidades de malla)
    SKELETON_TEMPLATE, MeshPart, compute_vertex_normals, make_crystal_gem, transform_mesh,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIEZAS_DIR = os.path.join(ROOT_DIR, "piezas")
REF_PATH = os.path.join(os.path.dirname(__file__), "referencias", "escarcelas_dragon_runico.jpg")
CATALOGO_ESTUDIO_PATH = os.path.join(PIEZAS_DIR, "catalogo-estudio.json")
GENERAR_CATALOGO_PATH = os.path.join(os.path.dirname(__file__), "generar-catalogo.py")

TASSET_IDX = 15
BELT_IDX = 13
BG = np.array([244, 236, 225], dtype=np.float32)   # color del papel de la lámina


# ==============================================================================
# 1. TEXTURAS RECORTADAS DE LA LÁMINA (fondo de papel -> transparente)
# ==============================================================================
def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def cut_out(ref: Image.Image, box: Tuple[int, int, int, int], size: Tuple[int, int],
            tol: float = 34.0, open_px: int = 3) -> Image.Image:
    """Recorta `box` de la lámina y vuelve transparente el papel conectado con el borde del recorte.
    Una apertura morfológica elimina las líneas guía finas que cruzan el fondo."""
    crop = ref.crop(box).convert("RGB")
    a = np.asarray(crop).astype(np.float32)
    h, w, _ = a.shape
    near_bg = np.linalg.norm(a - BG, axis=2) < tol
    # Relleno desde los bordes sobre píxeles de papel (fondo exterior, no huecos interiores)
    bg = np.zeros((h, w), bool)
    stack = [(y, x) for x in range(w) for y in (0, h - 1)] + [(y, x) for y in range(h) for x in (0, w - 1)]
    while stack:
        y, x = stack.pop()
        if y < 0 or x < 0 or y >= h or x >= w or bg[y, x] or not near_bg[y, x]:
            continue
        bg[y, x] = True
        stack.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
    fg = Image.fromarray(((~bg) * 255).astype(np.uint8))
    # Apertura: borra líneas de 1-3 px que no forman parte de la pieza
    fg = fg.filter(ImageFilter.MinFilter(open_px * 2 + 1)).filter(ImageFilter.MaxFilter(open_px * 2 + 1))
    fg = fg.filter(ImageFilter.GaussianBlur(0.8))
    rgba = crop.copy()
    rgba.putalpha(fg)
    return rgba.resize(size, Image.LANCZOS)


def build_textures() -> Dict[str, bytes]:
    ref = Image.open(REF_PATH).convert("RGB")
    tex = {}
    # Placa del dragón rúnico con cota de malla y flecos (vista lateral)
    tex["dragon"] = _png(cut_out(ref, (1340, 178, 1562, 618), (256, 512)))
    # Faldón trasero de cuero con correas cruzadas, malla y flecos (pierna izquierda, vista trasera)
    tex["back"] = _png(cut_out(ref, (842, 215, 1000, 785), (176, 512)))
    # Cinto de cuero tachonado (tramo liso del cinturón), se repite alrededor de la cintura
    band = ref.crop((222, 128, 302, 170)).convert("RGB").resize((128, 64), Image.LANCZOS)
    tex["belt"] = _png(band)
    # (cuerno: tramo central del cuerno de la placa en la vista lateral, sin fondo)
    # Placa-hebilla frontal con la gema cian
    buckle = cut_out(ref, (300, 108, 488, 238), (256, 176))
    # Silueta de la placa-hebilla (hexágono alargado): fuera de ella, el cinto de detrás queda transparente
    shape = Image.new("L", buckle.size, 0)
    ImageDraw.Draw(shape).polygon([(18, 88), (70, 40), (128, 4), (186, 40), (238, 88), (186, 136), (128, 172), (70, 136)], fill=255)
    shape = shape.filter(ImageFilter.GaussianBlur(1.0))
    al = np.minimum(np.asarray(buckle.getchannel("A")), np.asarray(shape))
    buckle.putalpha(Image.fromarray(al.astype(np.uint8)))
    tex["buckle"] = _png(buckle)
    # Hebilla trasera
    tex["buckle_back"] = _png(cut_out(ref, (955, 100, 1045, 160), (128, 96), tol=30))
    # Cuerno: tira vertical del cuerno de la placa (vista trasera, cuerno superior)
    horn = ref.crop((1438, 222, 1452, 262)).convert("RGB").resize((32, 128), Image.LANCZOS)
    tex["horn"] = _png(horn)
    return tex


# ==============================================================================
# 2. GEOMETRÍA CON UV (lámina curvada, cuernos, tiras)
# ==============================================================================
def curved_sheet(y_top: float, y_bot: float, r_top: Tuple[float, float], r_bot: Tuple[float, float],
                 ang_min: float, ang_max: float, thickness: float = 0.006,
                 nu: int = 24, nv: int = 32, flare: float = 0.0, belly: float = 0.0,
                 u_flip: bool = False):
    """Lámina con grosor sobre un cilindro elíptico (x = sin(a)·rx, z = cos(a)·rz).
    UV: u a lo ancho (ángulo), v de arriba (0) a abajo (1) — la imagen se proyecta entera."""
    verts, uvs = [], []
    for layer in (0.5, -0.5):
        for j in range(nv + 1):
            v = j / nv
            y = y_top + (y_bot - y_top) * v
            rx = r_top[0] + (r_bot[0] - r_top[0]) * v + flare * v * v
            rz = r_top[1] + (r_bot[1] - r_top[1]) * v + flare * v * v
            for i in range(nu + 1):
                u = i / nu
                a = ang_min + (ang_max - ang_min) * u
                bump = belly * math.sin(u * math.pi) * math.sin(v * math.pi)
                off = layer * thickness + bump
                verts.append((math.sin(a) * (rx + off), y, math.cos(a) * (rz + off)))
                uvs.append(((1.0 - u) if u_flip else u, v))
    tris = []
    row = nu + 1
    inner = (nv + 1) * row
    for j in range(nv):
        for i in range(nu):
            a = j * row + i
            b, c, d = a + 1, a + row + 1, a + row
            tris += [a, d, b, b, d, c]                       # exterior
            tris += [inner + a, inner + b, inner + d, inner + b, inner + c, inner + d]
    return np.array(verts, np.float32), np.array(tris, np.uint32), np.array(uvs, np.float32)


def curved_horn(base, direction, length: float, radius: float, bend, segs: int = 10, sides: int = 10):
    """Cuerno cónico curvado: eje cuadrático desde `base` en `direction`, curvándose hacia `bend`."""
    base = np.array(base, np.float32)
    d = np.array(direction, np.float32); d /= np.linalg.norm(d)
    bnd = np.array(bend, np.float32)
    pts = [base + d * length * t + bnd * length * t * t for t in np.linspace(0, 1, segs + 1)]
    verts, uvs, tris = [], [], []
    for k, p in enumerate(pts):
        t = k / segs
        tang = (pts[min(k + 1, segs)] - pts[max(k - 1, 0)]); tang /= np.linalg.norm(tang)
        ref = np.array([0, 1, 0], np.float32) if abs(tang[1]) < 0.9 else np.array([1, 0, 0], np.float32)
        n1 = np.cross(tang, ref); n1 /= np.linalg.norm(n1)
        n2 = np.cross(tang, n1)
        r = radius * (1.0 - t) ** 0.9 + 0.0006
        for s in range(sides + 1):
            ang = s / sides * math.tau
            verts.append(p + (n1 * math.cos(ang) + n2 * math.sin(ang)) * r)
            uvs.append((s / sides, 1.0 - t))
    row = sides + 1
    for k in range(segs):
        for s in range(sides):
            a = k * row + s
            tris += [a, a + row, a + 1, a + 1, a + row, a + row + 1]
    return np.array(verts, np.float32), np.array(tris, np.uint32), np.array(uvs, np.float32)


def strap(p0, p1, width: float, thick: float = 0.004, normal=(0, 0, 1)):
    """Correa recta (caja fina) entre p0 y p1, con la cara ancha hacia `normal`."""
    p0, p1 = np.array(p0, np.float32), np.array(p1, np.float32)
    ax = p1 - p0; L = float(np.linalg.norm(ax)); ax /= L
    nrm = np.array(normal, np.float32); nrm -= ax * float(nrm @ ax); nrm /= np.linalg.norm(nrm)
    side = np.cross(ax, nrm)
    verts, uvs = [], []
    for e in (0, 1):
        for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(p0 + ax * L * e + side * sx * width * 0.5 + nrm * sz * thick * 0.5)
            uvs.append(((sx + 1) * 0.5, e * L * 6.0))
    q = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = []
    for a, b, c, d in q:
        tris += [a, b, c, a, c, d]
    tris += [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7]
    return np.array(verts, np.float32), np.array(tris, np.uint32), np.array(uvs, np.float32)


def flat_plate(width: float, height: float, center, thickness: float = 0.008, curve_r: float = 0.0):
    """Placa frontal (hebilla) con la imagen proyectada; opcionalmente curvada alrededor del eje Y."""
    nu, nv = 16, 8
    cx, cy, cz = center
    verts, uvs = [], []
    for layer in (0.5, -0.5):
        for j in range(nv + 1):
            v = j / nv
            for i in range(nu + 1):
                u = i / nu
                x = (u - 0.5) * width
                z = cz + layer * thickness - (x * x / (2 * curve_r) if curve_r else 0.0)
                verts.append((cx + x, cy + (0.5 - v) * height, z))
                uvs.append((u, v))
    tris = []
    row = nu + 1; inner = (nv + 1) * row
    for j in range(nv):
        for i in range(nu):
            a = j * row + i; b, c, d = a + 1, a + row + 1, a + row
            tris += [a, d, b, b, d, c, inner + a, inner + b, inner + d, inner + b, inner + c, inner + d]
    return np.array(verts, np.float32), np.array(tris, np.uint32), np.array(uvs, np.float32)


def mirror_x(pos: np.ndarray, idx: np.ndarray):
    p = pos.copy(); p[:, 0] *= -1
    i = idx.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
    return p, i


# ==============================================================================
# 3. CONSTRUCTORES
# ==============================================================================
MAT_KEYS = ("dragon", "back", "belt", "buckle", "buckle_back", "horn", "gem")


def _parts() -> Dict[str, MeshPart]:
    return {k: MeshPart() for k in MAT_KEYS}


def build_tassets() -> Dict[str, Dict[str, MeshPart]]:
    left = _parts()
    # Placa del dragón: cara externa-frontal del muslo izquierdo (+X), de la cadera hasta encima de la rodilla
    a_c, a_h = math.pi * 0.32, 0.88   # cara externa-frontal del muslo, como en la vista frontal de la lámina
    pos, idx, uv = curved_sheet(0.035, -0.400, (0.132, 0.124), (0.120, 0.112), a_c - a_h, a_c + a_h,
                                thickness=0.007, flare=0.020, belly=0.010)
    left["dragon"].add(pos, idx, uv)
    # Faldón trasero de cuero con correas cruzadas: de la cara externa-trasera a la interna-trasera
    pos, idx, uv = curved_sheet(0.000, -0.430, (0.124, 0.118), (0.112, 0.106), math.pi * 0.62, math.pi * 1.30,
                                thickness=0.005, flare=0.016)
    left["back"].add(pos, idx, uv)
    # Cuernos curvos de la placa (como en la vista frontal/cenital): dos arriba, dos en el lateral
    for ang, y, up, ln, rad in ((a_c + 0.62, 0.010, 0.75, 0.085, 0.0105),
                                (a_c - 0.55, 0.000, 0.85, 0.070, 0.0090),
                                (a_c + 0.80, -0.120, 0.20, 0.075, 0.0095),
                                (a_c + 0.78, -0.215, -0.05, 0.060, 0.0085)):
        base = (math.sin(ang) * 0.136, y, math.cos(ang) * 0.128)
        outward = (math.sin(ang), up, math.cos(ang))
        bend = (0.0, 0.55, -0.25)
        left["horn"].add(*curved_horn(base, outward, ln, rad, bend))
    # Correas de suspensión desde el cinturón (en el espacio del muslo: el cinto queda ~0.09 m más arriba)
    for ang in (a_c - 0.55, a_c + 0.40):
        p0 = (math.sin(ang) * 0.128, 0.120, math.cos(ang) * 0.120)
        p1 = (math.sin(ang) * 0.134, 0.020, math.cos(ang) * 0.127)
        left["belt"].add(*strap(p0, p1, 0.020, 0.004, normal=(math.sin(ang), 0, math.cos(ang))))
    # Gema cian de la placa, en relieve sobre la que ya está pintada
    gx, gy = math.sin(a_c) * 0.152, -0.255
    left["gem"].add(*make_crystal_gem(0.011, 0.030, 6, center=(gx, gy, math.cos(a_c) * 0.142),
                                      rot=(math.pi * 0.5, 0, -a_c)))

    right = _parts()
    for k, part in left.items():
        if part.is_empty():
            continue
        p, n, u, i = part.build()
        mp, mi = mirror_x(p, i)
        right[k].add(mp, mi, u)
    return {"Thigh_L": left, "Thigh_R": right}


def build_belt() -> Dict[str, Dict[str, MeshPart]]:
    t = _parts()
    # Cinto: anillo completo en la cadera (espacio Torso: y≈-0.19), textura repetida 9 veces alrededor
    pos, idx, uv = curved_sheet(-0.162, -0.218, (0.176, 0.150), (0.178, 0.152), -math.pi, math.pi,
                                thickness=0.010, nu=72, nv=4)
    uv = uv.copy(); uv[:, 0] *= 9.0
    t["belt"].add(pos, idx, uv)
    # Placa-hebilla frontal con gema y la hebilla trasera
    t["buckle"].add(*flat_plate(0.150, 0.104, (0.0, -0.188, 0.172), thickness=0.010, curve_r=0.30))
    t["gem"].add(*make_crystal_gem(0.013, 0.036, 6, center=(0.0, -0.182, 0.176), rot=(math.pi * 0.5, 0, 0)))
    pb, ib, ub = flat_plate(0.070, 0.048, (0.0, -0.186, 0.160), thickness=0.008, curve_r=0.30)
    pb[:, 2] *= -1; ib = ib.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1); ub = ub.copy(); ub[:, 0] = 1 - ub[:, 0]
    t["buckle_back"].add(pb, ib, ub)
    # Gemas laterales de los enganches de las escarcelas
    for sgn in (1, -1):
        t["gem"].add(*make_crystal_gem(0.010, 0.026, 6, center=(sgn * 0.186, -0.190, 0.018),
                                       rot=(0, 0, sgn * -math.pi * 0.5)))
    return {"Torso": t}


# ==============================================================================
# 4. ESCRITURA GLB (materiales con las texturas de la lámina)
# ==============================================================================
MATERIALS = {
    "dragon":      dict(tex="dragon", metal=0.35, rough=0.48, alpha=True, emissive=[0.55, 0.25, 0.08]),
    "back":        dict(tex="back", metal=0.10, rough=0.62, alpha=True),
    "belt":        dict(tex="belt", metal=0.08, rough=0.60),
    "buckle":      dict(tex="buckle", metal=0.45, rough=0.40, alpha=True),
    "buckle_back": dict(tex="buckle_back", metal=0.45, rough=0.40, alpha=True),
    "horn":        dict(tex="horn", metal=0.20, rough=0.45),
    "gem":         dict(color=[0.18, 0.82, 0.78, 1.0], metal=0.10, rough=0.06, emissive=[0.04, 0.45, 0.42]),
}


def write_glb(out_path: str, bone_meshes: Dict[str, Dict[str, MeshPart]], textures: Dict[str, bytes],
              scene_name: str) -> dict:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nodes = json.loads(json.dumps(SKELETON_TEMPLATE))   # copia profunda: las listas children no se comparten
    name_to_node = {n["name"]: i for i, n in enumerate(nodes)}
    bin_buf = bytearray(); buffer_views, accessors, meshes = [], [], []

    def append(raw: bytes, target: Optional[int] = None) -> int:
        while len(bin_buf) % 4:
            bin_buf.append(0)
        bv = {"buffer": 0, "byteOffset": len(bin_buf), "byteLength": len(raw)}
        if target is not None:
            bv["target"] = target
        bin_buf.extend(raw); buffer_views.append(bv)
        return len(buffer_views) - 1

    used = [k for k in MAT_KEYS if any(not parts[k].is_empty() for parts in bone_meshes.values())]
    images, gtex, materials, mat_index = [], [], [], {}
    for key in used:
        spec = MATERIALS[key]
        pbr = {"metallicFactor": spec["metal"], "roughnessFactor": spec["rough"]}
        if "tex" in spec:
            images.append({"bufferView": append(textures[spec["tex"]]), "mimeType": "image/png", "name": spec["tex"]})
            gtex.append({"sampler": 0, "source": len(images) - 1})
            pbr["baseColorTexture"] = {"index": len(gtex) - 1}
        else:
            pbr["baseColorFactor"] = spec["color"]
        mat = {"name": f"DragonRunico_{key}", "pbrMetallicRoughness": pbr, "doubleSided": True}
        if spec.get("alpha"):
            mat["alphaMode"] = "MASK"; mat["alphaCutoff"] = 0.5
        if spec.get("emissive"):
            mat["emissiveFactor"] = spec["emissive"]
            if "tex" in spec and key == "dragon":
                mat["emissiveTexture"] = {"index": len(gtex) - 1}   # runas y ojos incandescentes
        mat_index[key] = len(materials); materials.append(mat)

    total_v = total_t = 0
    for bone, parts in bone_meshes.items():
        for key in used:
            part = parts[key]
            if part.is_empty():
                continue
            pos, nrm, uv, idx = part.build()
            total_v += len(pos); total_t += len(idx) // 3
            acc0 = len(accessors)
            accessors += [
                {"bufferView": append(pos.astype("<f4").tobytes(), 34962), "componentType": 5126,
                 "count": len(pos), "type": "VEC3", "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
                {"bufferView": append(nrm.astype("<f4").tobytes(), 34962), "componentType": 5126, "count": len(nrm), "type": "VEC3"},
                {"bufferView": append(uv.astype("<f4").tobytes(), 34962), "componentType": 5126, "count": len(uv), "type": "VEC2"},
                {"bufferView": append(idx.astype("<u4").tobytes(), 34963), "componentType": 5125, "count": len(idx), "type": "SCALAR"},
            ]
            meshes.append({"name": f"{bone}_{key}", "primitives": [{
                "attributes": {"POSITION": acc0, "NORMAL": acc0 + 1, "TEXCOORD_0": acc0 + 2},
                "indices": acc0 + 3, "material": mat_index[key]}]})
            nodes.append({"name": f"{bone}_DragonRunico_{key}", "mesh": len(meshes) - 1})
            nodes[name_to_node[bone]].setdefault("children", []).append(len(nodes) - 1)

    while len(bin_buf) % 4:
        bin_buf.append(0)
    gltf = {
        "asset": {"version": "2.0", "generator": "Armaduras-glb/crear_escarcelas_dragon.py"},
        "scene": 0, "scenes": [{"name": scene_name, "nodes": [0]}],
        "nodes": nodes, "meshes": meshes, "materials": materials, "textures": gtex, "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "accessors": accessors, "bufferViews": buffer_views, "buffers": [{"byteLength": len(bin_buf)}],
    }
    js = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_buf)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(bin_buf), 0x004E4942)); f.write(bytes(bin_buf))
    return {"bytes": total, "verts": total_v, "tris": total_t}


# ==============================================================================
# 5. CATÁLOGO
# ==============================================================================
CATALOG_ENTRIES = {
    "tassets": {"idx": TASSET_IDX, "name": "Escarcelas del Dragón Rúnico (Placa Lateral + Cuero)"},
    "belt":    {"idx": BELT_IDX, "name": "Cinturón del Dragón Rúnico con Gema Cian"},
}


def update_catalogo_estudio() -> None:
    with open(CATALOGO_ESTUDIO_PATH, encoding="utf-8") as f:
        estudio = json.load(f)
    for ranura, entry in CATALOG_ENTRIES.items():
        lst = estudio.setdefault(ranura, [])
        cur = next((it for it in lst if it["idx"] == entry["idx"]), None)
        if cur is not None:
            cur.update(entry)
        else:
            lst.append(dict(entry))
        lst.sort(key=lambda x: x["idx"])
    with open(CATALOGO_ESTUDIO_PATH, "w", encoding="utf-8") as f:
        json.dump(estudio, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    textures = build_textures()
    for slot, idx, fn in (("tassets", TASSET_IDX, build_tassets), ("belt", BELT_IDX, build_belt)):
        out = os.path.join(PIEZAS_DIR, slot, f"{idx}.glb")
        st = write_glb(out, fn(), textures, "DragonRunico")
        print(f"  [OK] {slot}/{idx}.glb -> {st['verts']} verts, {st['tris']} tris, {st['bytes']} bytes")
    if "--sin-catalogo" not in sys.argv:
        update_catalogo_estudio()
        subprocess.run([sys.executable, GENERAR_CATALOGO_PATH], check=True)
    if "--texturas" in sys.argv:
        dest = os.path.join(ROOT_DIR, "tools", "piezas", "referencias", "dragon_texturas")
        os.makedirs(dest, exist_ok=True)
        for k, b in textures.items():
            with open(os.path.join(dest, f"{k}.png"), "wb") as f:
                f.write(b)


if __name__ == "__main__":
    main()
