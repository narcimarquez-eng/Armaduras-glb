"""Genera piezas/catalogo.json a partir de piezas/catalogo-estudio.json (exportado del estudio).

Cada pieza real (índice > 0; el 0 de cada ranura es «Sin …») recibe un nivel mínimo para equiparla.
Las piezas de una ranura se reparten de forma uniforme entre el nivel 1 y el nivel MAX_NIVEL, en el
orden del catálogo del estudio. La cota de malla europea (underlayer 2) es la armadura inicial: nivel 1.
Para cambiar niveles, edita catalogo.json o ajusta MAX_NIVEL y vuelve a ejecutar:
    python3 tools/piezas/generar-catalogo.py
"""
import json
import os

MAX_NIVEL = 50
INICIAL = {('underlayer', 2)}  # Cota de Malla Europea 4-en-1: la que sale equipada

RAIZ = os.path.join(os.path.dirname(__file__), '..', '..', 'piezas')
with open(os.path.join(RAIZ, 'catalogo-estudio.json'), encoding='utf-8') as f:
    estudio = json.load(f)

catalogo = {}
for ranura, items in estudio.items():
    reales = [it for it in items if it['idx'] > 0]
    n = len(reales)
    piezas = []
    for k, it in enumerate(reales):
        nivel = 1 if (ranura, it['idx']) in INICIAL else round(1 + k * (MAX_NIVEL - 1) / max(1, n - 1))
        piezas.append({
            'idx': it['idx'],
            'nombre': it['name'],
            'nivel': nivel,
            # La capa interior no se exporta como malla: sin modelo hasta que el estudio la tenga
            'archivo': None if ranura == 'underlayer' else f"piezas/{ranura}/{it['idx']}.glb",
        })
    catalogo[ranura] = piezas

# Equipo inicial: la cota (capa interior) y, en cada ranura, la pieza de nivel 1 que tenga modelo
inicial = {}
for r, ps in catalogo.items():
    if r == 'underlayer':
        inicial[r] = 2
        continue
    primera = next((p for p in ps if p['nivel'] == 1 and p['archivo']), None)
    if primera:
        inicial[r] = primera['idx']

salida = {
    'version': 1,
    'maxNivel': MAX_NIVEL,
    'inicial': inicial,
    'ranuras': catalogo,
    'clases': {'guerrero': sorted(catalogo.keys()), 'mago': []},  # mago: túnicas pendientes de crear en el estudio
}
with open(os.path.join(RAIZ, 'catalogo.json'), 'w', encoding='utf-8') as f:
    json.dump(salida, f, ensure_ascii=False, indent=2)
print('piezas:', sum(len(v) for v in catalogo.values()), '→ piezas/catalogo.json')
