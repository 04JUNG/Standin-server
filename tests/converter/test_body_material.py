import array
from types import SimpleNamespace as Node

import pytest

from converter.body_material import COLOR, PROPERTIES, restore_tree
from converter.outline_fbx import fingerprint


def node(kind, props=(), types=b"", children=()):
    return Node(id=kind, props=list(props), props_type=bytearray(types), elems=list(children))


def fixture(name=b"Standin anatomical clay", textured=False):
    props = [node(b"P", [key, b"Color", b"", b"A", *([.8] * len(value))],
                  b"SSSS" + b"D" * len(value)) for key, value in PROPERTIES.items()]
    material = node(b"Material", [1, name + b"\x00\x01Material", b""], b"LSS",
                    [node(b"Properties70", children=props)])
    geometry = node(b"Geometry", [2, b"body", b"Mesh"], b"LSS", [
        node(b"Vertices", [array.array("d", [0., 1., 2.])], b"d"),
        node(b"Normals", [array.array("d", [0., 0., 1.])], b"d")])
    objects = [material, geometry, node(b"Deformer", [3, b"skin", b"Cluster"], b"LSS")]
    links = []
    if textured:
        objects.append(node(b"Texture", [4, b"skin texture", b""], b"LSS"))
        links.append(node(b"C", [b"OP", 4, 1, b"DiffuseColor"], b"SLLS"))
    return node(b"", children=[node(b"Objects", children=objects), node(b"Connections", children=links)])


def test_gray_restores_only_known_properties_and_is_idempotent():
    tree = fixture()
    objects = tree.elems[0].elems
    protected = [fingerprint(obj) for obj in objects[1:]]
    result = restore_tree(tree)
    assert result["changed_materials"] == ["Standin anatomical clay"]
    fields = {p.props[0]: p.props[4:] for p in objects[0].elems[0].elems}
    assert fields[b"DiffuseColor"] == list(COLOR)
    assert [fingerprint(obj) for obj in objects[1:]] == protected
    corrected = fingerprint(tree)
    assert restore_tree(tree)["changed_materials"] == []
    assert fingerprint(tree) == corrected


@pytest.mark.parametrize("name,textured", [(b"Pilot neutral clay", False),
    (b"Standin anatomical clay custom", False), (b"Standin anatomical clay", True)])
def test_other_and_textured_materials_are_unchanged(name, textured):
    tree = fixture(name, textured)
    before = fingerprint(tree)
    assert restore_tree(tree)["changed_materials"] == []
    assert fingerprint(tree) == before


def test_unknown_material_schema_fails_before_file_replacement():
    tree = fixture()
    tree.elems[0].elems[0].elems[0].elems.pop()
    with pytest.raises(ValueError, match="missing expected"):
        restore_tree(tree)
