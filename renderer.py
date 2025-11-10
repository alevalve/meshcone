#!/usr/bin/env python3
"""
Simplified ShapeNet Blender Renderer (robust, orthographic, always-in-frame)
Run with: blender --background --python shapenet_blender_render.py -- /input/dir /output/dir
"""

import bpy
import sys
import math
import os
from mathutils import Vector, Matrix
from pathlib import Path

# --------------------------
# Parse command line args
# --------------------------
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
    if len(argv) < 2:
        print("Usage: blender --background --python shapenet_blender_render.py -- <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = Path(argv[0])
    output_dir = Path(argv[1])
else:
    input_dir = Path("/data1/alex/convmesh/meshes/original_meshes")
    output_dir = Path("/data1/alex/convmesh/meshes/renders/original")

output_dir.mkdir(parents=True, exist_ok=True)

obj_files = list(input_dir.rglob("*.obj"))


def clean_scene():
    # Select everything and delete
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Purge orphans
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def create_camera_and_light():
    # Camera
    bpy.ops.object.camera_add(location=(0, -3, 0))
    cam = bpy.context.object
    scene = bpy.context.scene
    scene.camera = cam

    # Make orthographic to avoid "out of frame" surprises
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 2.2
    cam.data.clip_start = 0.001
    cam.data.clip_end = 1000.0

    # Sun
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(45))

    return cam, sun


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = 'PNG'

    # Make background opaque (no alpha)
    scene.render.film_transparent = False

    # Color management: avoid Filmic gray cast
    scene.display_settings.display_device = 'sRGB'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # White background (world)
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in nt.nodes:
        if n.type not in {'OUTPUT_WORLD', 'BACKGROUND'}:
            nt.nodes.remove(n)
    bg = nt.nodes.get('Background') or nt.nodes.new('ShaderNodeBackground')
    out = nt.nodes.get('World Output') or nt.nodes.new('ShaderNodeOutputWorld')
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])


def import_obj(path: Path):
    # Only remove mesh objects, keep camera and lights
    for obj in [o for o in bpy.data.objects if o.type == 'MESH']:
        bpy.data.objects.remove(obj, do_unlink=True)

    try:
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis='Y', up_axis='Z')
    except Exception as e:
        print(f"  Error importing {path.name}: {e}")
        return None

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            return obj
    return None


def object_space_bounds(obj):
    pts = [Vector(corner) for corner in obj.bound_box]
    min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    size = max((max_v - min_v).x, (max_v - min_v).y, (max_v - min_v).z)
    center = (min_v + max_v) * 0.5
    return min_v, max_v, size, center


def normalize_to_unit(obj, target_extent=2.0):
    if bpy.context.object is not None and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    _, _, size, center = object_space_bounds(obj)
    if size == 0:
        return False

    T = Matrix.Translation(-center)
    obj.data.transform(T)

    _, _, size, _ = object_space_bounds(obj)

    scale = target_extent / size
    S = Matrix.Scale(scale, 4)
    obj.data.transform(S)

    bpy.context.view_layer.update()

    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
    except:
        pass
    
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = math.radians(60)

    return True


def ensure_simple_material(obj):
    mat = bpy.data.materials.new(name="ClayMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.75, 0.73, 0.72, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Specular"].default_value = 0.2
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def add_tracker_at_origin():
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    return bpy.context.object


def make_camera_track(cam, target_obj):
    c = cam.constraints.new(type='TRACK_TO')
    c.target = target_obj
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'


def set_camera_view(cam, name):
    locs = {
        'front':  (0, -3, 0),
        'back':   (0,  3, 0),
        'left':   (-3, 0, 0),
        'right':  (3,  0, 0),
        'top':    (0,  0, 3),
        'bottom': (0,  0, -3),
    }
    cam.location = Vector(locs[name])
    bpy.context.view_layer.update()


def auto_set_ortho_scale(cam, margin=1.10):
    if cam.data.type != 'ORTHO':
        return
    cam.data.ortho_scale = 2.0 * margin


def render_views(output_base: Path, cam):
    views = ['front', 'back', 'left', 'right', 'top', 'bottom']
    for v in views:
        set_camera_view(cam, v)
        bpy.context.scene.render.filepath = str(output_dir / f"{output_base.stem}_{v}.png")
        bpy.ops.render.render(write_still=True)
    print(f"  ✓ Rendered {output_base.stem}")


def clean_mesh_only():
    """Safely remove only mesh objects between renders"""
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    for obj in mesh_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Clean up orphaned mesh data
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    
    # Clean up orphaned materials
    for mat in bpy.data.materials:
        if mat.users == 0:
            bpy.data.materials.remove(mat)


print("Setting up scene…")
clean_scene()
setup_render()
cam, _sun = create_camera_and_light()
tracker = add_tracker_at_origin()
make_camera_track(cam, tracker)

for i, obj_path in enumerate(obj_files, 1):
    print(f"[{i}/{len(obj_files)}] {obj_path.name}")

    mesh = import_obj(obj_path)
    if mesh is None:
        print("  ✗ Failed to import")
        continue

    if not normalize_to_unit(mesh, target_extent=2.0):
        print("  ✗ Empty or degenerate geometry")
        clean_mesh_only()
        continue

    ensure_simple_material(mesh)
    auto_set_ortho_scale(cam, margin=1.12)

    try:
        render_views(obj_path, cam)
    except Exception as e:
        print(f"  ✗ Render failed: {e}")
    
    # Clean up mesh after rendering
    clean_mesh_only()

print("\nRendering complete!")