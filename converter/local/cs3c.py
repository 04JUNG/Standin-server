"""Bounded CS3C v2 decoder and original skinned-character adapter.

Decoder parameters are a separately packaged, hash-pinned build resource. Neither
Modeler nor user-supplied code is executed. No proprietary binary/parameter bytes
or third-party character files are stored in this source tree.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import zlib

from converter.local.contract import LIMITS, LocalError

MODELER_SHA256 = "2ebe2d90f8609496cb2e81a7c9defae4e851479b8e5db76eb9dd8ec05d943150"
PARAMETER_SHA256 = "235841bf2ff7904bd86553dad6298b08eb03b707920f93b186deb713a89e4b41"
RESOURCE = Path(__file__).with_name("cs3c_decoder_data.json")
DECODER_VERSION = "cs3c-catalog2/1"
SCHEMAS = {
    "name": "636fb79c94e67ad63b539afee4638beed01714a0795c1c8ca6510972bdc49896",
    "location": "d1797e0c9528097c4125ffeb369f1c71c43c968b57637e9668bd55cb5301a1b2",
    "node": "2389d66f450495690d8aee487803ab42c52a664b8120dc12d1048e0ab8a3ccda",
    "flyweight_data_vector3_array": "c1120fda52229684110dffa27d9e16f02478bb9e018f3c0c886ba131e2b6482f",
    "flyweight_data_vector2_array": "5b412964e13dc8b73bd33c4b1316902a8c7076fafe42f27bf94c5ae391204c19",
    "flyweight_data_uint32_array": "8871c0394b64daf6fbcc559fc32961166e85e20cb294f1db929c96d4a994f7f3",
    "vertex_array": "e6978b4c964a7a86a9b1be0070ae6cffd7e42ef76e95f9e0cff46f1a871c99d1",
    "face_data": "2ba3f6a3414ae7b8e1ab571701ba368d3537597aba6543e9ce9df8ff1bbed2f6",
    "flyweight_data_face_array": "7620abea8a88d4de894d8e1c45f50fa2c34e23b65488794a1b454db260042c4e",
    "face_array": "e99bf89b599edfa7f5d4d091082564175d685662be59067b09395436022272a3",
    "texture": "099af093eea3eb945b43748b8e88f07a51fefb614340b409b177e833fe032084",
    "material": "a5e94f53206951a0976e688d0fb57a69e4b3e2b4ddf1e200ee4e06c5030485b8",
    "mesh": "6e6266f4c6da4bf3f570e5d0962aacff1f85ca2c928e4e2cb77df790380d412a",
    "flyweight_data_weightinfo_map": "37aa5feb35cae1f56ca7e416e9e39305e17d3856315f90931c1cc2731193de18",
    "skin": "942692594788cc2800c5eb8c0bcdf4d0a204b071d996dd142e8b313128dbe719",
}


def require(ok, code="UNSUPPORTED_CS3C_VERSION", message="Unsupported or malformed CS3C structure"):
    if not ok:
        raise LocalError(code, message)


def build_decoder_resource(exe, destination):
    """Build-only extraction; never copy/execute the Modeler executable."""
    exe = Path(exe)
    require(exe.is_file() and exe.stat().st_size <= LIMITS["sourceBytes"],
            "CS3C_DECODER_UNAVAILABLE", "Invalid decoder build input")
    data = exe.read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELER_SHA256,
            "CS3C_DECODER_UNAVAILABLE", "Decoder build input hash is not the reviewed Modeler version")
    payload = data[0x17ea970:0x17ea970 + 48] + data[0x17ea9a0:0x17ea9a0 + 1042 * 4]
    require(hashlib.sha256(payload).hexdigest() == PARAMETER_SHA256,
            "CS3C_DECODER_UNAVAILABLE", "Decoder parameter identity mismatch")
    Path(destination).write_text(json.dumps({"schemaVersion": 1, "sourceExeSha256": MODELER_SHA256,
                                           "parameterPayloadHex": payload.hex()}))
    return {"decoderVersion": DECODER_VERSION, "parameterSha256": PARAMETER_SHA256,
            "sourceExeSha256": MODELER_SHA256, "modelerExecuted": False, "modelerBundled": False}


def read_parameters(path=RESOURCE):
    try:
        path = Path(path)
        require(path.is_file() and path.stat().st_size <= 16384, "CS3C_DECODER_UNAVAILABLE", "Missing local CS3C decoder data")
        data = json.loads(path.read_text())
        payload = bytes.fromhex(data["parameterPayloadHex"])
        require(data["schemaVersion"] == 1 and data["sourceExeSha256"] == MODELER_SHA256
                and len(payload) == 4216 and hashlib.sha256(payload).hexdigest() == PARAMETER_SHA256,
                "CS3C_DECODER_UNAVAILABLE", "Local CS3C decoder data failed its identity check")
        return struct.unpack("<1042I", payload[48:]), payload[:48]
    except (OSError, ValueError, KeyError, TypeError):
        raise LocalError("CS3C_DECODER_UNAVAILABLE", "Local CS3C decoder data is unavailable or invalid") from None


def decoder_available():
    try:
        read_parameters()
        return True
    except LocalError:
        return False


class Cipher:
    """Observed fixed-parameter Feistel/feedback codec, no native EXE dependency."""
    MASK = 0xffffffff

    def __init__(self, parameters, key):
        require(len(parameters) == 1042 and bool(key))
        self.p, self.s = list(parameters[:18]), list(parameters[18:])
        for i in range(18):
            self.p[i] ^= int.from_bytes(bytes(key[(i * 4 + j) % len(key)] for j in range(4)), "big")
        left = right = 0
        for array in (self.p, self.s):
            for i in range(0, len(array), 2):
                left, right = self.block(left, right)
                array[i], array[i + 1] = left, right

    def f(self, value):
        s = self.s
        return (((s[value >> 24] + s[256 + ((value >> 16) & 255)]) & self.MASK)
                ^ s[512 + ((value >> 8) & 255)]) + s[768 + (value & 255)] & self.MASK

    def block(self, left, right, decrypt=False):
        p = self.p[::-1] if decrypt else self.p
        for i in range(16):
            left ^= p[i]
            right ^= self.f(left)
            left, right = right, left
        left, right = right, left
        return left ^ p[17], right ^ p[16]

    def decode(self, data):
        require(len(data) >= 16 and len(data) % 8 == 0)
        words = [self.block(a, b, True) for a, b in struct.iter_unpack(">II", data)]
        left = (words[-1][0] - sum(a for a, _ in words[:-1])) & self.MASK
        right = (words[-1][1] - sum(b for _, b in words[:-1])) & self.MASK
        out = bytearray()
        for a, b in words[:-1]:
            left ^= a
            right ^= b
            out.extend(struct.pack(">II", left, right))
        return bytes(out)


def _catalog(path):
    try:
        with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, LIMITS["expandedBytes"])
            db.set_progress_handler(lambda: 1, 100_000)
            db.execute("PRAGMA trusted_schema=OFF")
            db.execute("PRAGMA query_only=ON")
            row = db.execute("SELECT catalog_data FROM character LIMIT 1").fetchone()
        blob = row[0] if row else b""
        require(isinstance(blob, bytes) and len(blob) >= 89)
        require(blob[:24] == struct.pack("<I", 20) + b"CLIP_STUDIO_3D_DATA2"
                and blob[24:45] == struct.pack("<I", 17) + b"catalog_character")
        version, plain, encrypted = struct.unpack_from("<III", blob, 61)
        require(version == 2 and encrypted % 8 == 0 and len(blob) == 73 + encrypted and plain <= encrypted - 8)
        require(plain <= LIMITS["expandedBytes"], "LIMIT_EXCEEDED", "CS3C catalog exceeds the expanded-data limit")
        return blob, {"container": "SQLite3", "catalogVersion": version, "declaredPlainBytes": plain,
                      "decoderAvailable": decoder_available(), "decoderVersion": DECODER_VERSION}
    except (sqlite3.Error, struct.error, TypeError, IndexError):
        raise LocalError("UNSUPPORTED_CS3C_VERSION", "Unrecognized CS3C container or catalog version") from None


def inspect_container(path):
    return _catalog(path)[1]


class Reader:
    def __init__(self, data):
        self.data, self.pos = memoryview(data), 0

    def take(self, size):
        require(0 <= size <= len(self.data) - self.pos)
        out = self.data[self.pos:self.pos + size]
        self.pos += size
        return out

    def row(self, fmt):
        try:
            return struct.unpack(fmt, self.take(struct.calcsize(fmt)))
        except struct.error:
            raise LocalError("UNSUPPORTED_CS3C_VERSION", "Truncated CS3C record") from None

    def count(self, maximum=1_000_000):
        n = self.row("<I")[0]
        require(n <= maximum, "LIMIT_EXCEEDED", "CS3C row/array count exceeds the limit")
        return n

    def string(self):
        try:
            return bytes(self.take(self.count(65536))).decode("utf-8")
        except UnicodeError:
            raise LocalError("UNSUPPORTED_CS3C_VERSION", "Invalid CS3C string") from None

    def done(self):
        require(self.pos == len(self.data))


class Tables:
    def __init__(self, data):
        require(len(data) <= LIMITS["expandedBytes"], "LIMIT_EXCEEDED", "CS3C table stream exceeds the limit")
        self.tables = {}
        offset = 0
        # Observed table framing. Required table headers AND exact row consumption
        # are checked, so marker collisions fail closed rather than guessing a schema.
        while offset < len(data):
            require(len(self.tables) < 128)
            reader = Reader(data[offset:])
            name = reader.string()
            require(name not in self.tables and name.isascii() and len(name) < 128)
            begin = data.find(b"\x0b\0\0\0table_begin", offset + reader.pos)
            end = data.find(b"\x09\0\0\0table_end", begin + 19)
            require(begin >= offset + reader.pos and end >= begin + 19 and begin - offset < 16384)
            header = data[offset:begin]
            if name in SCHEMAS:
                require(hashlib.sha256(header).hexdigest() == SCHEMAS[name])
            count = struct.unpack_from("<I", data, begin + 15)[0]
            require(count <= 1_000_000, "LIMIT_EXCEEDED", "CS3C table row count exceeds the limit")
            self.tables[name] = (count, data[begin + 19:end])
            offset = end + 13
        require(offset == len(data))

    def records(self, name, fmt):
        count, data = self.tables[name]
        require(len(data) == count * struct.calcsize(fmt))
        return list(struct.iter_unpack(fmt, data))

    def arrays(self, name, fmt):
        count, data = self.tables[name]
        reader = Reader(data)
        result = []
        for _ in range(count):
            size = reader.count()
            result.append(list(struct.iter_unpack(fmt, reader.take(size * struct.calcsize(fmt)))))
        reader.done()
        return result


def decode_tables(path, resource=RESOURCE):
    blob, info = _catalog(path)
    plain = Cipher(*read_parameters(resource)).decode(blob[73:])[:info["declaredPlainBytes"]]
    reader = Reader(plain)
    expanded, field_size = reader.row("<II")
    compressed = reader.row(">I")[0]
    require(expanded <= LIMITS["expandedBytes"], "LIMIT_EXCEEDED", "CS3C expanded tables exceed the limit")
    require(field_size == compressed + 4)
    stream = bytes(reader.take(compressed))
    inflater = zlib.decompressobj()
    try:
        data = inflater.decompress(stream, expanded + 1)
    except zlib.error:
        raise LocalError("UNSUPPORTED_CS3C_VERSION", "CS3C compressed catalog is invalid") from None
    require(len(data) == expanded and inflater.eof and not inflater.unused_data,
            "LIMIT_EXCEEDED", "CS3C actual expansion does not match its declaration")
    tables = Tables(data)
    Tables(bytes(reader.take(len(reader.data) - reader.pos)))  # Validate offscreen table framing too.
    require(SCHEMAS.keys() <= tables.tables.keys())
    info.update({"decoderAvailable": True, "expandedBytes": len(data), "tableStreamSha256": hashlib.sha256(data).hexdigest()})
    return tables, info


def character_data(path, resource=RESOURCE):
    """Recover source geometry, source materials, hierarchy and sparse weights."""
    import numpy as np
    tables, info = decode_tables(path, resource)
    require(tables.tables["node"][0] <= LIMITS["bones"] and tables.tables["mesh"][0] <= LIMITS["meshes"]
            and tables.tables["material"][0] <= 2048, "LIMIT_EXCEEDED", "CS3C character table limit exceeded")
    count, data = tables.tables["name"]
    reader = Reader(data)
    names = []
    for _ in range(count):
        labels = {}
        for _ in range(reader.count(64)):
            language = reader.row("<I")[0]
            require(language not in labels)
            labels[language] = reader.string()
        names.append({"label": labels.get(1, labels.get(0, "")), "uuid": bytes(reader.take(16)).hex()})
    reader.done()
    def at(values, index):
        require(0 <= index < len(values))
        return values[index]
    locations = tables.records("location", "<7d")
    require(all(math.isfinite(x) for row in locations for x in row))
    nodes = []
    for name, parent, location, *scale in tables.records("node", "<III3d"):
        label = at(names, name)
        require(label["label"] and len(label["label"].encode()) <= 63)
        require(all(abs(x-1) < 1e-8 for x in scale), "UNSUPPORTED_TRANSFORM", "Scaled CS3C joints are not certified")
        pos_rot = at(locations, location)
        require(abs(sum(x*x for x in pos_rot[3:]) - 1) < 1e-5, "UNSUPPORTED_TRANSFORM", "Invalid CS3C joint quaternion")
        nodes.append({**label, "parentUuid": at(names, parent)["uuid"] if at(names, parent)["label"] else None,
                      "position": pos_rot[:3], "rotation": pos_rot[3:]})
    require(0 < len(nodes) <= LIMITS["bones"], "LIMIT_EXCEEDED", "CS3C bone limit exceeded")
    by_uuid = {n["uuid"]: i for i, n in enumerate(nodes)}
    require(len(by_uuid) == len(nodes) and len({n["label"] for n in nodes}) == len(nodes),
            "INVALID_HIERARCHY", "Duplicate CS3C bone identities")
    for n in nodes:
        require(n["parentUuid"] is None or n["parentUuid"] in by_uuid, "INVALID_HIERARCHY", "CS3C parent is missing")
        n["parent"] = by_uuid[n["parentUuid"]] if n["parentUuid"] else None
    world = [None] * len(nodes)
    visiting = set()
    def visit(index):
        if world[index] is not None:
            return world[index]
        require(index not in visiting, "INVALID_HIERARCHY", "CS3C hierarchy is cyclic")
        visiting.add(index)
        n = nodes[index]
        w, x, y, z = n["rotation"]
        m = np.eye(4)
        m[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        m[:3, 3] = n["position"]
        if n["parent"] is not None:
            m = visit(n["parent"]) @ m
        world[index] = m
        visiting.remove(index)
        return m
    for i in range(len(nodes)):
        visit(i)
    require(sum(n["parent"] is None for n in nodes) == 1, "INVALID_HIERARCHY", "CS3C must contain one character hierarchy")
    require(tables.tables["texture"] == (0, b""), "UNSUPPORTED_MATERIAL", "CS3C textures are not yet certified")
    count, data = tables.tables["material"]
    reader = Reader(data)
    materials = []
    for _ in range(count):
        row = reader.row("<I14f")
        textures = [reader.row("<I")[0] for _ in range(reader.count(64))]
        require(all(t == 0xffffffff for t in textures), "UNSUPPORTED_MATERIAL", "CS3C texture references are not certified")
        require(all(math.isfinite(x) for x in row[1:]) and all(0 <= c <= 1 for c in row[4:7])
                and abs(row[13]-1) < 1e-6 and 0 <= row[14] <= 100
                and all(abs(x) < 1e-8 for x in row[7:13]),
                "UNSUPPORTED_MATERIAL", "Only opaque nonemissive matte CS3C colors are certified")
        materials.append({"name": at(names, row[0])["label"], "ambient": row[1:4], "diffuse": row[4:7],
                          "specular": row[7:10], "emission": row[10:13], "opacity": row[13], "shininess": row[14]})
    reader.done()
    vectors = tables.arrays("flyweight_data_vector3_array", "<3f")
    uv = tables.arrays("flyweight_data_vector2_array", "<2f")
    integers = tables.arrays("flyweight_data_uint32_array", "<I")
    face_records = tables.arrays("face_data", "<I")
    face_groups = tables.arrays("flyweight_data_face_array", "<I")
    vertex_refs = tables.records("vertex_array", "<4I")
    face_refs = tables.records("face_array", "<2I")
    count, data = tables.tables["mesh"]
    require(0 < count <= LIMITS["meshes"], "LIMIT_EXCEEDED", "CS3C mesh limit exceeded")
    reader = Reader(data)
    meshes = []
    vertex_count = 0
    for _ in range(count):
        name, node, location, vr, fr = reader.row("<5I")
        slots = [reader.row("<I")[0] for _ in range(reader.count(1024))]
        require(slots and all(i < len(materials) for i in slots), "UNSUPPORTED_MATERIAL", "Invalid CS3C material slots")
        require(at(locations, location) == (0,0,0,1,0,0,0), "UNSUPPORTED_TRANSFORM", "Nonidentity CS3C mesh transforms are not certified")
        coord, normal, uvref, link = at(vertex_refs, vr)
        positions = np.asarray(at(vectors, coord), dtype=float)
        normals = np.asarray(at(vectors, normal), dtype=float)
        vertex_count += len(positions)
        require(vertex_count <= LIMITS["vertices"], "LIMIT_EXCEEDED", "CS3C vertex limit exceeded")
        require(positions.shape == normals.shape and positions.ndim == 2 and positions.shape[1] == 3
                and len(positions) > 0 and np.isfinite(positions).all() and np.isfinite(normals).all()
                and (np.linalg.norm(normals, axis=1) > 1e-8).all(), "INVALID_SKIN", "Invalid CS3C vertices or normals")
        coords_uv = np.asarray(at(uv, uvref), dtype=float) if uvref != 0xffffffff else None
        require(coords_uv is None or (coords_uv.shape == (len(positions), 2) and np.isfinite(coords_uv).all()))
        faces, material_ref = at(face_refs, fr)
        triangles = []
        for (fi,) in at(face_groups, faces):
            indices = [i for (i,) in at(face_records, fi) if i != 0xffffffff]
            require(len(indices) == 3 and len(set(indices)) == 3 and all(i < len(positions) for i in indices),
                    "UNSUPPORTED_FEATURE", "Only valid triangular CS3C faces are certified")
            triangles.append(indices)
        assignment = [i for (i,) in at(integers, material_ref)]
        require(len(assignment) == len(triangles) and all(i < len(slots) for i in assignment),
                "UNSUPPORTED_MATERIAL", "Invalid CS3C per-face materials")
        meshes.append({**at(names, name), "positions": positions, "normals": normals, "uv": coords_uv,
                       "triangles": triangles, "slots": slots, "assignment": assignment,
                       "weights": [[] for _ in positions], "sourceNode": at(names, node)["uuid"],
                       "nextLinkReference": link})
    reader.done()
    mesh_by_uuid = {m["uuid"]: i for i, m in enumerate(meshes)}
    require(len(mesh_by_uuid) == len(meshes))
    weight_maps = tables.arrays("flyweight_data_weightinfo_map", "<Id")
    binds = {}
    max_bind_error = 0.
    for bone, mesh, wi, *raw in tables.records("skin", "<III16d"):
        require(at(names, bone)["uuid"] in by_uuid and at(names, mesh)["uuid"] in mesh_by_uuid, "INVALID_SKIN", "Invalid CS3C skin binding")
        joint = by_uuid[at(names, bone)["uuid"]]
        target = meshes[mesh_by_uuid[at(names, mesh)["uuid"]]]
        inverse = np.asarray(raw, dtype=float).reshape(4,4).T
        error = float(np.max(np.abs(world[joint] @ inverse - np.eye(4))))
        require(np.isfinite(inverse).all() and error < 1e-4, "INVALID_SKIN", "Source CS3C inverse bind does not match its hierarchy")
        max_bind_error = max(max_bind_error, error)
        require(joint not in binds or np.max(np.abs(binds[joint]-inverse)) < 1e-8, "INVALID_SKIN", "Inconsistent CS3C inverse binds")
        binds[joint] = inverse
        for vertex, weight in at(weight_maps, wi):
            require(vertex < len(target["positions"]) and math.isfinite(weight) and 0 <= weight <= 1,
                    "INVALID_SKIN", "Invalid CS3C source weight")
            require(all(j != joint for j, _ in target["weights"][vertex]), "INVALID_SKIN", "Duplicate CS3C vertex/joint weight")
            if weight > 0:
                target["weights"][vertex].append((joint, weight))
    rest_error = 0.
    for mesh in meshes:
        require(all(abs(sum(w for _, w in row)-1) < 1e-6 for row in mesh["weights"]),
                "INVALID_SKIN", "Incomplete or unnormalized original CS3C weights")
        for point, row in zip(mesh["positions"], mesh["weights"]):
            skinned = sum(w * (world[j] @ binds[j] @ np.r_[point, 1])[:3] for j,w in row)
            rest_error = max(rest_error, float(np.linalg.norm(skinned-point)))
    require(rest_error < 1e-4, "INVALID_SKIN", "CS3C original rest skin does not reproduce its geometry")
    info.update({"hierarchyNodes": len(nodes), "meshes": len(meshes), "vertices": vertex_count,
                 "materials": materials, "materialSupport": "source_diffuse_colors_and_face_assignment",
                 "sourceShadingMatchToCSP": "unverified", "maxInverseBindError": max_bind_error,
                 "maxSourceRestSkinError": rest_error, "sourceAxisUp": "Y", "sourceUnits": "unconfirmed",
                 "internalDisplayScale": .01, "originalWeightsRenormalized": False})
    return {"nodes": nodes, "world": world, "binds": binds, "meshes": meshes, "materials": materials, "info": info}


def verify_original_skin(data, arm, meshes, conversion):
    """Compare Blender skinning to source FK + source inverse binds/weights.

    This is independent of FBX roundtrip self-consistency: rebuilding a visually
    identical rest mesh with the wrong bind frames/weights must not pass import.
    """
    import bpy
    import numpy as np
    from mathutils import Matrix
    from converter.local.inspection import surface
    by_name = {n["label"]: i for i, n in enumerate(data["nodes"])}
    c = np.asarray(conversion)
    maximum = 0.
    for name in ("LeftLowerArm", "RightLowerArm", "LeftLowerLeg", "RightLowerLeg", "Spine"):
        require(name in by_name, "UNSUPPORTED_RIG", "Unsupported CS3C humanoid joint mapping")
        selected = by_name[name]
        rotation = np.asarray(Matrix.Rotation(.25, 4, "X"))
        worlds = {}
        def visit(i):
            if i not in worlds:
                parent = data["nodes"][i]["parent"]
                local = (np.linalg.inv(data["world"][parent]) @ data["world"][i]
                         if parent is not None else data["world"][i])
                if i == selected:
                    local = local @ rotation
                worlds[i] = visit(parent) @ local if parent is not None else local
            return worlds[i]
        joints = {j: visit(j) @ inverse for j, inverse in data["binds"].items()}
        bone = arm.pose.bones[name]
        saved = bone.matrix_basis.copy()
        bone.matrix_basis = Matrix(rotation.tolist())
        bpy.context.view_layer.update()
        observed = surface(meshes)
        for mesh, source in zip(meshes, data["meshes"]):
            expected = []
            for point, row in zip(source["positions"], source["weights"]):
                skinned = sum(w * (joints[j] @ np.r_[point, 1]) for j,w in row)
                expected.append((c @ skinned)[:3])
            error = float(np.linalg.norm(np.asarray(expected)-observed[mesh.name], axis=1).max())
            require(error < 2e-5, "INVALID_SKIN", "Blender deformation differs from the original CS3C skin")
            maximum = max(maximum, error)
        bone.matrix_basis = saved
        bpy.context.view_layer.update()
    return maximum


def import_character(path):
    """Create the same Blender character representation used by the FBX adapter."""
    import bpy
    from mathutils import Matrix
    data = character_data(path)
    conversion = Matrix.Rotation(math.pi/2, 4, "X") @ Matrix.Scale(.01, 4)
    skeleton = bpy.data.armatures.new("CS3COriginalSkeleton")
    arm = bpy.data.objects.new("CS3CCharacter", skeleton)
    bpy.context.collection.objects.link(arm)
    arm.matrix_world = conversion
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for i, n in enumerate(data["nodes"]):
        bone = skeleton.edit_bones.new(n["label"])
        # A new EditBone has zero length. Assigning its matrix at that point
        # loses the requested Y direction/roll; initialize length first.
        bone.head = (0, 0, 0)
        bone.tail = (0, 1, 0)
        children = [j for j, child in enumerate(data["nodes"]) if child["parent"] == i]
        bone.length = min((math.dist(data["world"][i][:3,3], data["world"][j][:3,3]) for j in children), default=5.)
        require(bone.length > 1e-6, "UNSUPPORTED_RIG", "Degenerate CS3C joint")
        bone.matrix = Matrix(data["world"][i].tolist())
    for n in data["nodes"]:
        if n["parent"] is not None:
            skeleton.edit_bones[n["label"]].parent = skeleton.edit_bones[data["nodes"][n["parent"]]["label"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    rest_matrix_error = max(max(abs(skeleton.bones[n["label"]].matrix_local[r][c] - data["world"][i][r,c])
                                for r in range(4) for c in range(4)) for i,n in enumerate(data["nodes"]))
    require(rest_matrix_error < 1e-4, "INVALID_SKIN", "Blender rest matrices differ from the original CS3C hierarchy")
    data["info"]["maxBlenderRestMatrixErrorSourceUnits"] = rest_matrix_error
    materials = []
    for i, source in enumerate(data["materials"]):
        mat = bpy.data.materials.new(f"CS3CMaterial_{i}")
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*source["diffuse"], source["opacity"])
        shader.inputs["Roughness"].default_value = 1 - math.sqrt(source["shininess"])/10
        shader.inputs["Specular IOR Level"].default_value = 0.
        mat.diffuse_color = (*source["diffuse"], source["opacity"])
        mat["cs3c_source_material"] = json.dumps(source)
        materials.append(mat)
    meshes = []
    for index, source in enumerate(data["meshes"]):
        geometry = bpy.data.meshes.new(f"CS3CGeometry_{index}")
        geometry.from_pydata(source["positions"].tolist(), [], source["triangles"])
        geometry.update()
        geometry.normals_split_custom_set_from_vertices(source["normals"].tolist())
        mesh = bpy.data.objects.new(f"CS3CMesh_{index}", geometry)
        bpy.context.collection.objects.link(mesh)
        mesh.matrix_world = conversion
        mesh["cs3c_source_name"] = source["label"]
        for slot in source["slots"]:
            geometry.materials.append(materials[slot])
        for polygon, slot in zip(geometry.polygons, source["assignment"]):
            polygon.material_index = slot
            polygon.use_smooth = True
        if source["uv"] is not None:
            layer = geometry.uv_layers.new(name="SourceUV")
            for loop in geometry.loops:
                layer.data[loop.index].uv = source["uv"][loop.vertex_index]
        groups = {}
        for vertex, row in enumerate(source["weights"]):
            for joint, weight in row:
                groups.setdefault((joint, weight), []).append(vertex)
        vertex_groups = {j: mesh.vertex_groups.new(name=data["nodes"][j]["label"]) for j in {j for j, _ in groups}}
        for (joint, weight), indices in groups.items():
            vertex_groups[joint].add(indices, weight, "REPLACE")
        mesh.modifiers.new("CS3COriginalSkin", "ARMATURE").object = arm
        meshes.append(mesh)
    bpy.context.view_layer.update()
    data["info"]["maxOriginalFKProbeError"] = verify_original_skin(data, arm, meshes, conversion)
    data["info"]["originalFKProbeCount"] = 5
    arm["standin_cs3c_provenance"] = json.dumps(data["info"])
    return arm, meshes
