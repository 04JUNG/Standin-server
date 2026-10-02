"""Optional primitive prop guides for authored scenes, separate from the BVH.

Guides explain grip and scale. They are not production assets or collision
evidence. Body diagnostics run before these objects are added.
"""

import math
import bpy
from mathutils import Vector


def _sphere(name, center, radius, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=radius, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    return obj


def _box(name, center, size, direction=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    if direction is not None:
        obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    return obj


def _rod(name, start, end, radius):
    delta = end - start
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=delta.length, location=(start + end) / 2)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = delta.to_track_quat('Z', 'Y').to_euler()
    return obj


def add_guides(arm, specification, support=''):
    if not specification and not support:
        return {'count': 0, 'kind': 'none'}
    def point(name):
        return arm.matrix_world @ arm.pose.bones['mixamorig:' + name].head
    height = (point('Head') - point('Hips')).length / .65
    up = Vector((0, 0, 1))
    forward = Vector((0, -1, 0))
    centers = {}
    for side in ('L', 'R'):
        name = 'Left' if side == 'L' else 'Right'
        centers[side] = (point(name + 'HandMiddle1') + point(name + 'HandMiddle2')) / 2
    count = 0
    for item in specification.split(';') if specification else []:
        kind, grip, *orientation = item.split(':')
        center = (centers['L'] + centers['R']) / 2 if grip == 'B' else centers[grip]
        axis = (centers['L'] - centers['R']).normalized() if grip == 'B' else up
        if orientation:
            axis = {'up':up, 'down':-up, 'forward':forward}[orientation[0]]
        if not orientation and grip == 'B' and axis.z < 0:
            axis = -axis
        scale = height
        if kind in {'sword', 'greatsword', 'dagger', 'spear', 'staff', 'broom', 'mop', 'axe', 'hammer', 'pickaxe', 'wand', 'torch', 'umbrella'}:
            lengths = {'dagger': .3, 'wand': .3, 'sword': .8, 'greatsword': 1.2, 'spear': 1.7,
                       'staff': 1.6, 'broom': 1.25, 'mop': 1.25, 'axe': .75, 'hammer': .8,
                       'pickaxe': .85, 'torch': .5, 'umbrella': .85}
            length = lengths[kind] * scale
            lower = .72 if kind in {'broom','mop'} else .18 if grip != 'B' else .45
            start, end = center - axis * length * lower, center + axis * length * (1-lower)
            _rod('Guide ' + kind, start, end, .018 * scale)
            if kind in {'sword', 'greatsword', 'dagger'}:
                _box('Blade', center + axis * length * .37, (.055*scale, .012*scale, length*.65), axis)
                _box('Guard', center + axis * length * .03, (.16*scale, .03*scale, .025*scale))
            elif kind in {'axe', 'hammer', 'pickaxe', 'broom', 'mop'}:
                _box('Tool head', start if kind in {'broom','mop'} else end, (.26*scale, .09*scale, .15*scale))
            elif kind in {'spear', 'torch'}:
                _sphere('Tip', end, .035*scale, (1, 1, 2.3))
            elif kind == 'umbrella':
                _sphere('Canopy guide', end, .40*scale, (1, 1, .18))
        elif kind in {'cup', 'goblet', 'potion', 'jar', 'watering_can'}:
            radius = (.038 if kind == 'potion' else .054) * scale
            # Both-hand holds locate the vessel between palms; single-hand
            # grips place its side beside the fingers, leaving room for a handle.
            c = center + (forward*.045*scale if grip != 'B' else Vector((0,0,0)))
            _rod('Vessel guide', c - up*.045*scale, c + up*.07*scale, radius)
            if kind == 'goblet':
                _rod('Stem', c-up*.10*scale, c-up*.045*scale, .012*scale)
        elif kind in {'phone','book','letter','photo','scroll','shield','tray','plate','lunchbox','cake','gift','ringbox','crate','chest','basket','bag','pouch','pillow','laundry','teddy'}:
            sizes = {'phone':(.07,.018,.14),'book':(.26,.06,.30),'letter':(.21,.008,.15),'photo':(.16,.005,.11),
                     'scroll':(.35,.018,.23),'shield':(.42,.05,.54),'tray':(.52,.30,.025),'plate':(.22,.22,.015),
                     'lunchbox':(.32,.20,.08),'cake':(.28,.24,.10),'gift':(.26,.20,.20),'ringbox':(.06,.05,.05),
                     'crate':(.48,.32,.30),'chest':(.52,.34,.30),'basket':(.50,.32,.26),'bag':(.28,.18,.30),
                     'pouch':(.10,.08,.13),'pillow':(.42,.19,.28),'laundry':(.40,.22,.24),'teddy':(.38,.22,.42)}
            size = sizes[kind]
            c = center.copy()
            if kind in {'tray','plate','cake','lunchbox','crate','chest','basket'}:
                c.z += size[2]*scale*.42
            _box('Guide '+kind,c,tuple(v*scale for v in size))
        elif kind in {'orb','gem','boulder','crown'}:
            radius = {'orb':.14,'gem':.035,'boulder':.24,'crown':.13}[kind]*scale
            _sphere('Guide '+kind, center + up*radius*.5, radius)
        elif kind in {'pen','spoon','brush','comb','knife','key','flower','bouquet'}:
            length = (.38 if kind in {'flower','bouquet'} else .18)*scale
            _rod('Guide '+kind, center-up*length*.2, center+up*length*.8, .008*scale)
            if kind in {'flower','bouquet'}:
                _sphere('Flowers',center+up*length*.8,(.065 if kind=='flower' else .15)*scale)
        elif kind in {'towel','scarf','handkerchief','rope'}:
            if grip == 'B':
                _rod('Cloth/rope guide', centers['L'], centers['R'], (.008 if kind=='rope' else .025)*scale)
            else:
                _box('Cloth guide',center-up*.055*scale,(.12*scale,.01*scale,.12*scale))
        elif kind in {'pistol','crossbow'}:
            _box('Grip',center,(.03*scale,.06*scale,.12*scale))
            _box('Aim guide',center+forward*.12*scale+up*.055*scale,(.06*scale,.27*scale,.04*scale))
        elif kind == 'bow':
            aim = (center-point('Head')).normalized()
            arc = [center+up*(t*.46*scale)-aim*(t*t*.13*scale) for t in [i/6 for i in range(-6,7)]]
            for a,b in zip(arc,arc[1:]):
                _rod('Bow limb',a,b,.012*scale)
            draw = centers['R'] if grip=='L' else centers['L']
            _rod('Bowstring',arc[0],draw,.002*scale)
            _rod('Bowstring',draw,arc[-1],.002*scale)
        elif kind in {'barrier'}:
            continue
        elif kind == 'dryer':
            _rod('Dryer handle',center-up*.07*scale,center+up*.04*scale,.025*scale)
            inward = (point('Head')-center).normalized()
            _rod('Dryer nozzle',center+up*.08*scale,center+up*.08*scale+inward*.15*scale,.04*scale)
        elif kind == 'lantern':
            _box('Lantern',center-up*.12*scale,(.13*scale,.13*scale,.18*scale))
            _rod('Lantern handle',center-up*.03*scale,center+up*.04*scale,.009*scale)
        else:
            raise ValueError('unsupported prop guide: '+kind)
        count += 1
    if support == 'seat':
        pelvis = point('Hips')
        _box('Seat guide',pelvis-up*.105*height,(.53*height,.43*height,.045*height))
    return {'count':count,'specification':specification,'support':support,
            'kind':'primitive placement guides, separate from BVH; not contact/collision evidence'}
