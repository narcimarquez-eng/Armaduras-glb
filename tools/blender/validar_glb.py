"""Valida un .glb exportado por el estudio importándolo en Blender (sin interfaz) y renderiza una vista previa.

Uso (Blender instalado o `pip install bpy`):
    blender -b -P tools/blender/validar_glb.py -- Personaje_Armadura_Ligero.glb preview.png
    python tools/blender/validar_glb.py -- Personaje_Armadura_Ligero.glb preview.png   # con el módulo bpy
"""
import sys
import bpy
import mathutils

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[-2:]
glb_path, out_png = args[0], (args[1] if len(args) > 1 else "preview.png")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
print(f"Mallas: {len(meshes)} | Triángulos: {tris} | Animaciones: {len(bpy.data.actions)} | "
      f"Materiales: {len(bpy.data.materials)} | Imágenes: {len(bpy.data.images)}")
print("Animaciones:", ", ".join(a.name for a in bpy.data.actions))

sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 24
sc.render.resolution_x, sc.render.resolution_y = 640, 800
sc.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("Fondo")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.06, 0.09, 1)

for loc, energy in [((2.5, -3.0, 3.5), 900), ((-3, -2, 2.5), 450), ((0, 3, 3), 600)]:
    light = bpy.data.objects.new("Luz", bpy.data.lights.new("Luz", "AREA"))
    light.data.energy, light.data.size = energy, 2.0
    light.location = loc
    light.rotation_euler = (mathutils.Vector((0, 0, 1.2)) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(light)

cam = bpy.data.objects.new("Camara", bpy.data.cameras.new("Camara"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.location = (0.0, -3.2, 1.25)
cam.rotation_euler = (mathutils.Vector((0, 0, 1.1)) - cam.location).to_track_quat("-Z", "Y").to_euler()

sc.render.filepath = out_png
bpy.ops.render.render(write_still=True)
print("Render guardado en", out_png)
