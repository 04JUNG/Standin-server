"""Check approximation on dense independent positions, not just retained knots."""
import numpy as np
import pytest

from scripts.character_upgrade.sampled_table_simplification import simplify_axis


def test_linear_table_keeps_endpoints_and_declared_preset_knot():
    grid=np.linspace(-1,1,65)
    table=np.stack([2*grid+1,-3*grid,grid/7],axis=1)
    keep=simplify_axis(table,0,grid,{0.0},1e-12)
    np.testing.assert_array_equal(grid[keep],[-1,0,1])
    assert np.array_equal(table[keep[1]],table[32])


def test_nonlinear_curve_error_stays_bounded_between_training_knots():
    grid=np.linspace(-2,2,129)
    table=np.stack([np.sin(3*grid),.3*np.cos(5*grid)],axis=1)
    tolerance=.015
    keep=simplify_axis(table,0,grid,{0.0},tolerance)
    assert 3<len(keep)<len(grid)
    positions=np.random.default_rng(817273).uniform(-2,2,12037)
    before=np.stack([np.interp(positions,grid,table[:,i]) for i in range(2)],axis=1)
    after=np.stack([np.interp(positions,grid[keep],table[keep,i]) for i in range(2)],axis=1)
    assert np.linalg.norm(after-before,axis=1).max()<=tolerance+1e-12


def test_shared_dimension_is_checked_against_every_other_grid_slice():
    grid=np.linspace(0,1,65)
    table=np.zeros((3,len(grid),2))
    table[0,:,0]=grid
    table[1,:,0]=grid**3
    table[2,:,1]=np.sin(12*grid)
    tolerance=.008
    keep=simplify_axis(table,1,grid,{.5},tolerance)
    positions=np.linspace(0,1,7001)
    for line in table:
        error=np.stack([np.interp(positions,grid[keep],line[keep,i])-np.interp(positions,grid,line[:,i]) for i in range(2)],axis=1)
        assert np.linalg.norm(error,axis=1).max()<=tolerance+1e-12


@pytest.mark.parametrize('grid,tolerance', [([0,0,1],.01),([0,float('nan'),1],.01),([0,.5,1],-1)])
def test_invalid_domains_cannot_create_a_simplification_proof(grid,tolerance):
    with pytest.raises(ValueError):
        simplify_axis(np.zeros((3,2)),0,grid,set(),tolerance)


def test_tensor_reduction_certificate_bounds_independent_geometry_positions():
    x=np.linspace(-1,1,65);y=np.linspace(-2,2,49)
    xx,yy=np.meshgrid(x,y,indexing='ij')
    table=np.stack([np.sin(2*xx)*np.cos(yy),xx*yy,.2*np.sin(3*yy)+xx**2],axis=-1)
    tolerance=.006
    ix=simplify_axis(table,0,x,{0.0},tolerance)
    iy=simplify_axis(table,1,y,{0.0},tolerance)
    trimmed=table[np.ix_(ix,iy)]
    positions=np.random.default_rng(11791).uniform([-1,-2],[1,2],size=(1501,2))
    # Independent sequential 1-D interpolation at arbitrary 2-D positions.
    def interpolate(values,gx,gy):
        rows=[]
        for px,py in positions:
            along_y=np.array([[np.interp(py,gy,line[:,k]) for k in range(3)] for line in values])
            rows.append([np.interp(px,gx,along_y[:,k]) for k in range(3)])
        return np.asarray(rows)
    delta=interpolate(trimmed,x[ix],y[iy])-interpolate(table,x,y)
    assert np.linalg.norm(delta,axis=1).max()<=2*tolerance+1e-12
    basis=np.random.default_rng(703).normal(size=(3,17,3))
    vertex_bound=np.sqrt(np.sum(basis**2,axis=(0,2))).max()
    geometry_delta=np.einsum('pk,kvc->pvc',delta,basis)
    assert np.linalg.norm(geometry_delta,axis=2).max()<=2*tolerance*vertex_bound+1e-12
