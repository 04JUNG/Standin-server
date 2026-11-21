"""Model-selection dimensions must not remove actual geometry controls."""
import copy

import numpy as np
import pytest

from scripts.character_upgrade.check_sampled_runtime import make_evaluator


def runtime(mode):
    header = {
        "vertexCount": 1, "basisCount": 1, "coefCount": 4,
        "controls": {k: {"min": 0, "max": 1, "default": 0} for k in ("shape", "detail")},
        "presets": {"A": {"shape": 0, "detail": 0}, "B": {"shape": 1, "detail": 0}},
        "models": {
            name: {"base": index, "axes": [{"key": "detail", "knots": [0, 1], "offset": 2}], "pairs": []}
            for index, name in enumerate(("A", "B"))
        },
        "selection": {"mode": mode, "cutRatio": 2},
    }
    arrays = {"basis": np.array([1, 0, 0]), "basisScale": np.array([1]),
              "coef": np.array([0, 10, 0, 1]), "mean": np.zeros(3)}
    return header, arrays


def test_varying_mode_keeps_common_detail_in_geometry_and_catalog_domain():
    header, arrays = runtime("idw2-varying")
    original = copy.deepcopy(header)
    evaluate = make_evaluator(header, arrays, 1)
    np.testing.assert_array_equal(evaluate("A", {"shape": .3, "detail": 0}), [[0, 0, 0]])
    np.testing.assert_array_equal(evaluate("A", {"shape": .3, "detail": 1}), [[1, 0, 0]])
    assert header == original


def test_legacy_mode_retains_all_dimension_inverse_distance_weights():
    header, arrays = runtime("idw2")
    result = make_evaluator(header, arrays, 1)("A", {"shape": .3, "detail": 1})
    da, db = .3**2 + 1, .7**2 + 1
    cutoff = 1 / (4 * da**2)
    wa, wb = 1 / da**2 - cutoff, 1 / db**2 - cutoff
    assert result[0, 0] == pytest.approx(1 + 10 * wb / (wa + wb))


def test_unknown_selection_mode_is_rejected():
    header, arrays = runtime("unknown")
    with pytest.raises(ValueError, match="unsupported"):
        make_evaluator(header, arrays, 1)


@pytest.mark.parametrize('values',[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,1),(.2,.4,.7),(-1,.4,2)])
def test_three_control_interpolation_matches_independent_polynomial(values):
    polynomial=lambda x,y,z:1+2*x+3*y+4*z+5*x*y*z
    rows=[0]+[polynomial(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]
    model=dict(base=0,axes=[],pairs=[],triples=[dict(a='x',b='y',c='z',ga=[0,1],gb=[0,1],gc=[0,1],offset=1)])
    header=dict(vertexCount=1,basisCount=1,coefCount=9,controls={k:dict(min=0,max=1,default=0) for k in ('x','y','z')},
        presets={'A':dict(x=0,y=0,z=0),'B':dict(x=1,y=0,z=0)},models={'A':model,'B':model},selection=dict(mode='idw2-varying-trilinear',cutRatio=2))
    arrays=dict(basis=np.array([1,0,0]),basisScale=np.ones(1),coef=np.array(rows),mean=np.zeros(3))
    actual=make_evaluator(header,arrays,1)('A',dict(zip(('x','y','z'),values)))
    assert actual[0,0]==pytest.approx(polynomial(*(min(1,max(0,v)) for v in values)))
    header['selection']['mode']='idw2'
    with pytest.raises(ValueError,match='explicit trilinear'):
        make_evaluator(header,arrays,1)
