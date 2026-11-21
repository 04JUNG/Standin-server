"""Conservative original-knot simplification for multilinear coefficient tables."""
import numpy as np

def simplify_axis(table,axis,grid,preserve,tolerance):
 table=np.asarray(table,dtype=np.float64);grid=np.asarray(grid,dtype=np.float64)
 if (table.ndim<2 or not 0<=axis<table.ndim-1 or table.shape[axis]!=len(grid)
     or len(grid)<2 or not np.isfinite(table).all() or not np.isfinite(grid).all()
     or not np.all(np.diff(grid)>0) or not np.isfinite(tolerance) or tolerance<0):
  raise ValueError('Invalid interpolation table or tolerance')
 K=table.shape[-1]
 line=np.moveaxis(table,axis,0).reshape(len(grid),-1,K);keep=list(range(len(grid)));i=1
 while i<len(keep)-1:
  left,mid,right=keep[i-1:i+2]
  if grid[mid] in preserve:i+=1;continue
  positions=np.arange(left+1,right);t=((np.asarray(grid)[positions]-grid[left])/(grid[right]-grid[left]))[:,None,None]
  delta=line[positions]-((1-t)*line[left]+t*line[right]);error=float(np.linalg.norm(delta,axis=2).max())
  if error<=tolerance:keep.pop(i);i=max(1,i-1)
  else:i+=1
 return keep
