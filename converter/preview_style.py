"""Soft library-preview lighting, independent of pose, crop and camera fitting.

Blender imports stay inside functions. Workbench uses the original review studio;
EEVEE/Cycles use a camera-space emission shader so headless rendering does not
introduce specular highlights, occlusion or size-dependent area-light shadows.
The browser implements the same two diffuse lobes in previewStyle.ts.
"""
from pathlib import Path

PREVIEW_STYLE_VERSION = "library-soft-v1"
STUDIO_FILE = Path(__file__).with_name("preview_studio.sl")
BACKGROUND_LINEAR = (0.31855, 0.31855, 0.31855)  # sRGB #999999
SURFACE_GAIN = 0.72  # Match Workbench's diffuse response without washed-out whites.


def studio_parameters():
    values = dict(line.split(maxsplit=1) for line in STUDIO_FILE.read_text().splitlines())
    def vector(prefix):
        return tuple(float(values[f"{prefix}.{axis}"]) for axis in "xyz")
    lights = []
    for index in range(4):
        prefix = f"light[{index}]"
        if values[f"{prefix}.flag"] == "1":
            lights.append({"color": vector(f"{prefix}.col"),
                           "direction": vector(f"{prefix}.vec"),
                           "wrap": float(values[f"{prefix}.smooth"])})
    return {"ambient": vector("light_ambient"), "lights": lights}


def configure_color(scene):
    import bpy
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Library preview background")
    scene.world.color = BACKGROUND_LINEAR


def configure_workbench(scene):
    import bpy
    configure_color(scene)
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    studio = next((light for light in bpy.context.preferences.studio_lights
                   if light.name == STUDIO_FILE.name), None)
    if studio is None:
        studio = bpy.context.preferences.studio_lights.load(str(STUDIO_FILE), "STUDIO")
    shading.studio_light = studio.name
    shading.studiolight_rotate_z = 0
    shading.color_type = "SINGLE"
    shading.single_color = (1.0, 1.0, 1.0)
    shading.show_shadows = True
    shading.show_cavity = False
    shading.show_specular_highlight = False
    shading.show_object_outline = False
    shading.background_type = "WORLD"


def configure_surface(scene, meshes):
    """Camera-relative matte shader for EEVEE/Cycles and the browser fallback."""
    import bpy
    from mathutils import Vector
    configure_color(scene)
    world = scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*BACKGROUND_LINEAR, 1)
    background.inputs["Strength"].default_value = 1
    material = bpy.data.materials.new("Library soft preview")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    geometry = nodes.new("ShaderNodeNewGeometry")
    # Use the actual camera basis: shader CAMERA coordinates differ between
    # engines, while geometry.Normal is consistently a world-space unit normal.
    camera_basis = scene.camera.matrix_world.to_3x3()
    parameters = studio_parameters()
    ambient = nodes.new("ShaderNodeRGB")
    ambient.outputs[0].default_value = (*parameters["ambient"], 1)
    color = ambient.outputs[0]
    for light in parameters["lights"]:
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        links.new(geometry.outputs["Normal"], dot.inputs[0])
        dot.inputs[1].default_value = camera_basis @ Vector(light["direction"])
        add = nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        links.new(dot.outputs["Value"], add.inputs[0])
        add.inputs[1].default_value = light["wrap"]
        divide = nodes.new("ShaderNodeMath")
        divide.operation, divide.use_clamp = "DIVIDE", True
        links.new(add.outputs[0], divide.inputs[0])
        divide.inputs[1].default_value = 1 + light["wrap"]
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.inputs[0].default_value = light["color"]
        links.new(divide.outputs[0], scale.inputs["Scale"])
        total = nodes.new("ShaderNodeVectorMath")
        total.operation = "ADD"
        links.new(color, total.inputs[0])
        links.new(scale.outputs[0], total.inputs[1])
        color = total.outputs[0]
    gain = nodes.new("ShaderNodeVectorMath")
    gain.operation = "SCALE"
    links.new(color, gain.inputs[0])
    gain.inputs["Scale"].default_value = SURFACE_GAIN
    emission = nodes.new("ShaderNodeEmission")
    links.new(gain.outputs[0], emission.inputs["Color"])
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs[0], output.inputs["Surface"])
    for mesh in meshes:
        mesh.data.materials.clear()
        mesh.data.materials.append(material)
