"""Appended to a pinned authoring controller for the approved anatomy revision.

The inherited neutral body controls are retained. Pose deformation is provided
by the approved 54-bone rig and PSDs, rather than stacking two shoulder solvers.
"""
import base64, zlib

ANATOMY_CACHE = {}


def _anatomy_source():
    m, a = objects()
    if m.as_pointer() not in ANATOMY_CACHE:
        base = arr(m.data.shape_keys.key_blocks['Basis'].data)
        ANATOMY_CACHE[m.as_pointer()] = {
            'base': base,
            'psd': {k.name: arr(k.data) - base for k in m.data.shape_keys.key_blocks if k.name.startswith('PSD_')},
        }
    return ANATOMY_CACHE[m.as_pointer()]


def correct_pose(scene=None, check_intersections=False, contacts=False):
    m, a = objects(scene)
    # Drivers were restored from the approved manifests at authoring time.
    a.update_tag()
    m.data.update()
    bpy.context.view_layer.update()
    if not np.isfinite(evaluated(m)).all():
        raise ValueError('non-finite corrected anatomy')
    m['pose_qa'] = canonical({'rules': 'approved-anatomy-psd-v1', 'finite': True,
        'active': {k.name: float(k.value) for k in m.data.shape_keys.key_blocks
                   if k.name.startswith('PSD_') and k.value > 1e-5}})
    m['contact_qa'] = canonical({'checked': False, 'reason': 'approved anatomical PSDs; no collision solver'})


def synchronize(scene=None, correct=True):
    global BUSY
    if BUSY:
        return
    scene = scene or bpy.context.scene
    m, a = objects(scene)
    s = source(scene)
    c = normalize(controls(scene))
    result, _, reference = compute(c, scene)
    validate_geometry(c, scene)
    saved = _anatomy_source()
    BUSY = True
    try:
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = a
        bpy.ops.object.select_all(action='DESELECT')
        a.hide_set(False)
        a.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in a.data.edit_bones:
            row = s['rest'][bone.name]
            h, t = warp([row['head'], row['tail']], c, m)
            bone.use_connect = False
            bone.head, bone.tail = h, t
            axis = warp([np.array(row['head']) + .001*np.array(row['roll_axis'])], c, m)[0] - h
            bone.align_roll(Vector(axis))
        bpy.ops.object.mode_set(mode='OBJECT')
        keys = m.data.shape_keys.key_blocks
        for key in keys:
            if key.name.startswith(('BODY_', 'AUTO_')):
                key.value = 0
        keys['SYSTEM_Proportions'].data.foreach_set('co', (s['co'] + result - saved['base']).ravel())
        keys['SYSTEM_Proportions'].value = 1
        # Transport PSD displacement through the same stature/proportion warp.
        for name, delta in saved['psd'].items():
            transported = warp(saved['base'] + delta, c, m) - reference
            keys[name].data.foreach_set('co', (s['co'] + transported).ravel())
        m['current_controls'] = canonical(c)
        m.data.update()
        a.update_tag()
        bpy.context.view_layer.update()
        if correct:
            correct_pose(scene)
    finally:
        if a.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        BUSY = False


def hl_prepare_export_copy(source, target):
    # All approved bone channels, including the two scapular helpers, are baked.
    for pb in source.pose.bones:
        target.pose.bones[pb.name].matrix_basis = pb.matrix_basis.copy()


def sl_export_copy(source, target):
    hl_prepare_export_copy(source, target)


def register():
    global BUSY
    # Headless pinned controller: no persistent editor handlers or old rig rules.
    old = getattr(bpy.types, 'STANDIN_PG_studio', None)
    if hasattr(bpy.types.Scene, 'standin_studio'):
        del bpy.types.Scene.standin_studio
    if old:
        bpy.utils.unregister_class(old)
    bpy.utils.register_class(STANDIN_PG_studio)
    bpy.types.Scene.standin_studio = PointerProperty(type=STANDIN_PG_studio)
    BUSY = True
    try:
        for k, v in defaults(objects()[0]).items():
            setattr(bpy.context.scene.standin_studio, k, v)
    finally:
        BUSY = False
    _anatomy_source()


def preview_mesh():
    return next(o for o in bpy.context.scene.objects if o.get('standin_preview_lod'))


def preview_compute(c):
    """Barycentric projection of the full export mesh for the existing SSR2 UI."""
    lod = preview_mesh()
    if 'lod' not in ANATOMY_CACHE:
        raw = zlib.decompress(base64.b64decode(lod['projection']))
        n = len(lod.data.vertices)
        ids = np.frombuffer(raw[:n*12], dtype='<i4').reshape(n, 3)
        weights = np.frombuffer(raw[n*12:], dtype='<f4').reshape(n, 3)
        ANATOMY_CACHE['lod'] = (ids, weights)
    ids, weights = ANATOMY_CACHE['lod']
    result = compute(c)[0]
    return np.einsum('ni,nij->nj', weights, result[ids])
