import numpy as np
from converter.hand_contact import triangle_crossing


def test_crossing_and_separated_triangles():
    a=np.array([[0,0,0],[2,0,0],[0,2,0]])
    b=np.array([[.4,.4,-1],[.4,.4,1],[1,.4,1]])
    assert triangle_crossing(a,b)
    assert not triangle_crossing(a,b+[5,0,0])
    assert not triangle_crossing(a,a+[0,0,1])


def test_scale_and_translation_preserve_crossing():
    a=np.array([[0,0,0],[2,0,0],[0,2,0]])
    b=np.array([[.4,.4,-1],[.4,.4,1],[1,.4,1]])
    for scale in (.01,1,100):assert triangle_crossing(a*scale+3,b*scale+3)
