"""Genera piezas/catalogo.json a partir de piezas/catalogo-estudio.json (exportado del estudio).

Cada pieza real (índice > 0; el 0 de cada ranura es «Sin …») recibe un nivel mínimo para equiparla.
Las piezas de una ranura se reparten de forma uniforme entre el nivel 1 y el nivel MAX_NIVEL, en el
orden del catálogo del estudio. Las armas se reparten así dentro de su familia (legado, espada, hacha,
maza, daga, lanza, báculo, arco): cada tipo de arma tiene piezas desde el nivel 1 hasta el MAX_NIVEL. La cota de malla europea (underlayer 2) es la armadura inicial: nivel 1.
Para cambiar niveles, edita catalogo.json o ajusta MAX_NIVEL y vuelve a ejecutar:
    python3 tools/piezas/generar-catalogo.py
"""
import json
import os

MAX_NIVEL = 40
INICIAL = {('underlayer', 2)}  # Cota de Malla Europea 4-en-1: la que sale equipada

RAIZ = os.path.join(os.path.dirname(__file__), '..', '..', 'piezas')
with open(os.path.join(RAIZ, 'catalogo-estudio.json'), encoding='utf-8') as f:
    estudio = json.load(f)

def repartir(items):
    """Niveles de 1 a MAX_NIVEL repartidos de forma uniforme, en orden."""
    n = len(items)
    return {it['idx']: round(1 + k * (MAX_NIVEL - 1) / max(1, n - 1)) for k, it in enumerate(items)}


catalogo = {}
for ranura, items in estudio.items():
    reales = [it for it in items if it['idx'] > 0]
    if ranura == 'weapon':
        # Por familia (la del estudio, en catalogo-estudio.json): todas empiezan en el nivel 1
        niveles = {}
        for fam in dict.fromkeys(it.get('familia', 'arma') for it in reales):
            niveles.update(repartir([it for it in reales if it.get('familia', 'arma') == fam]))
    else:
        niveles = repartir(reales)
    piezas = []
    for it in reales:
        pieza = {
            'idx': it['idx'],
            'nombre': it['name'],
            'nivel': 1 if (ranura, it['idx']) in INICIAL else niveles[it['idx']],
            # La capa interior no se exporta como malla: sin modelo hasta que el estudio la tenga
            'archivo': None if ranura == 'underlayer' else f"piezas/{ranura}/{it['idx']}.glb",
        }
        if 'familia' in it:
            pieza['familia'] = it['familia']
        piezas.append(pieza)
    catalogo[ranura] = piezas

# Equipo inicial: solo la cota (capa interior). El arma básica la elige js/equipo.js.
inicial = {'underlayer': 2}

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
