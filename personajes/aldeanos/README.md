# Aldeanos con ropa actual (.glb)

Los 12 looks de «Ropa Actual» de la pestaña **Aldea** del Wardrobe Studio
(`public/assets/aistudio/Parametric_Character_Outfitter…html`), exportados con el botón **GLB ligero**
(texturas de 512 px, malla simplificada al 50 %, cuantizada).

Se regeneran con:

```bash
npm i -D playwright        # o usar uno ya instalado
node tools/aldeanos/exportar-aldeanos.cjs                 # todos, en personajes/aldeanos/
node tools/aldeanos/exportar-aldeanos.cjs salida mod_suit # solo uno
```

El script sirve el estudio desde el propio proceso, abre Chromium sin ventana, elige cada look (complexión
«natural»; el género está fijado en el script) y guarda la descarga. `exportados.json` lista lo exportado.

| Archivo | Look | Género |
|---|---|---|
| streetwear.glb | Urbano / Streetwear | hombre |
| casual_denim.glb | Casual Vaquero (Denim) | hombre |
| suit.glb | Ejecutivo / Traje Formal | hombre |
| biker.glb | Motero / Rockero Urbano | hombre |
| sport.glb | Deportivo / Fitness | mujer |
| winter.glb | Invierno / Plumífero | hombre |
| summer.glb | Verano / Playero Casual | mujer |
| college.glb | Universitario / Varsity | mujer |
| chic_dress.glb | Moda Urbana / Falda Midi | mujer |
| trench.glb | Gabardina / Trench Elegante | mujer |
| worker.glb | Obrero / Técnico Actual | hombre |
| doctor.glb | Médico / Científico | mujer |

## Qué traen
- **Sin skinning**: piezas rígidas colgadas de una jerarquía de nodos con nombres fijos, la misma en todos:
  `CreadorHumano_keep › Pelvis › Torso › Neck › Head`, `Shoulder_L/R › UpperArm › Elbow › Forearm › Wrist`,
  `Hip_L/R › Thigh › Knee › Shin › Ankle`. Por eso una animación vale para cualquier aldeano.
- **48 animaciones horneadas** sobre esos nodos (rotación de cada articulación y traslación de `Pelvis`):
  `idle`, `walk`, `run`, `jump` (agacharse + aire + aterrizar en un clip), `crouch`, `swim`, `victory`,
  `backflip` y las de combate.
- **Unidades**: metros; mira hacia **+Z**; la raíz está a ~1,45 m (pelvis) y los pies en y ≈ 0.
- **Peso**: 4,6–8 MB y ~110–140 k triángulos cada uno, 50–60 mallas. Para un juego en móvil conviene prepararlos
  (quitar animaciones que no se usan, unir mallas, reducir texturas): el juego Plataformas-Reales- lo hace con su
  propio script.
