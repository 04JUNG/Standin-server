"""Linearize hinge and existing-contact limits on the active patch and its halo."""
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import csr_matrix, vstack

from scripts.local_converter.experimental.contact import baseline_contact_pins
from scripts.local_converter.experimental.contact_features import measures
from scripts.local_converter.experimental.deformation_guard import DeformationGuard


class PoseLimits:
    def __init__(self, guard, free, contacts, selected_pairs=None):
        features=[]
        if contacts:
            fixed,scan=baseline_contact_pins(guard.baseline,guard.faces,free,sliding_features=features)
            if not scan["complete"] or (fixed & free).any():
                raise ValueError("Unclassified baseline contacts on free vertices")
        if selected_pairs is not None:
            features = [feature for feature in features if tuple(sorted(feature[:2])) in selected_pairs]
            if len(features)!=len(selected_pairs):
                raise ValueError("Missing or unsupported contact witness")
        pairs=np.asarray([(i,j) for i,j,_ in features],dtype=int).reshape(-1,2)
        affected=free[guard.faces].any(1)
        left,right=guard.graph.face_pairs.T
        halo=affected.copy()
        halo[left[affected[right]]]=True
        halo[right[affected[left]]]=True
        halo[pairs.ravel()]=True
        ids=np.unique(guard.faces[halo])
        faces=np.searchsorted(ids,guard.faces[halo])
        self.guard=DeformationGuard(guard.rest[ids],guard.baseline[ids],faces,~free[ids])
        self.movable=np.flatnonzero(free[ids])
        self.pairs=np.searchsorted(ids,guard.faces[pairs])
        self.spec=np.asarray([s for _,_,s in features],dtype=int).reshape(-1,2,2)
        self.contact_reference=measures(self.guard.baseline,self.pairs,self.spec)
        self.unit=guard.scale*.01
        self.crease_limit=np.maximum(1e-7,abs(self.guard.angle_difference(self.guard.reference["bend"],self.guard.rest_bend)))
        left,right=self.guard.graph.face_pairs.T
        hinge=np.column_stack((faces[left],faces[right]))
        supports=(hinge,hinge,hinge[self.guard.crease],hinge[self.guard.crease],
                  np.repeat(self.pairs.reshape(-1,6),7,axis=0))
        mapping=np.full(len(ids),-1)
        mapping[self.movable]=np.arange(len(self.movable))
        rows,cols=[],[]
        offset=0
        for support in supports:
            for i,vertices in enumerate(support):
                variables=mapping[np.unique(vertices)]
                variables=variables[variables>=0]
                rows.extend([offset+i]*3*len(variables))
                cols.extend((3*variables[:,None]+np.arange(3)).ravel())
            offset+=len(support)
        self.structure=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(offset,3*len(self.movable)))

        self.contact_structure=self.structure[-len(self.contact_reference):] if len(self.contact_reference) else None
        self.initial_rate=None
        self.swept_pairs = contacts and selected_pairs is not None and bool(features)

    def contact_residual(self,x):
        points=self.guard.baseline.copy()
        points[self.movable]+=x.reshape(-1,3)*self.unit
        return (measures(points,self.pairs,self.spec)-self.contact_reference)/self.unit

    def residual(self,x):
        g=self.guard
        q=g.baseline.copy()
        q[self.movable]+=x.reshape(-1,3)*self.unit
        bend=g.measure(q)["bend"]
        crease=g.angle_difference(bend,g.rest_bend)[g.crease]
        return np.r_[bend-g.bend_limit,-bend-g.bend_limit,
                     crease-self.crease_limit[g.crease],-crease-self.crease_limit[g.crease],
                     (measures(q,self.pairs,self.spec)-self.contact_reference)/self.unit]

    def approximation(self,x):
        value=self.residual(x)
        jac=approx_derivative(self.residual,x,method="3-point",sparsity=self.structure,abs_step=1e-6)
        scale=np.asarray(abs(jac).max(axis=1).toarray()).ravel()
        keep=scale>1e-10
        mat=jac[keep].multiply(1/scale[keep,None]).tocsr()
        rhs=mat@x-value[keep]/scale[keep]
        if not self.swept_pairs:
            return mat,rhs
        if self.initial_rate is None:
            self.initial_rate=approx_derivative(self.contact_residual,np.zeros_like(x),method="3-point",
                                                sparsity=self.contact_structure,abs_step=1e-6)
        pieces=[mat,self.initial_rate]
        limits=[rhs,np.full(len(self.contact_reference),-1e-7)]
        for fraction in (.25,.5,.75):
            point=x*fraction
            derivative=approx_derivative(self.contact_residual,point,method="3-point",
                                         sparsity=self.contact_structure,abs_step=1e-6)*fraction
            pieces.append(derivative)
            limits.append(derivative@x-self.contact_residual(point)-1e-8)
        return vstack(pieces,format="csr"),np.concatenate(limits)


def hinge_jacobian(guard, variable, unit):
    left,right=guard.graph.face_pairs.T
    lf,rf=guard.faces[left],guard.faces[right]
    active=((variable[lf]>=0)|(variable[rf]>=0)).any(1)
    lf,rf=lf[active],rf[active]
    corners=np.column_stack((lf,rf))
    shared=(lf[:,:,None]==rf[:,None,:]).any(2)
    endpoints=np.nonzero(shared)[1].reshape(-1,2)
    row=np.arange(len(corners))
    points=guard.baseline[corners].copy()
    def angle():
        first=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
        second=np.cross(points[:,4]-points[:,3],points[:,5]-points[:,3])
        axis=points[row,endpoints[:,1]]-points[row,endpoints[:,0]]
        axis/=np.linalg.norm(axis,axis=1)[:,None]
        return np.arctan2((np.cross(first,second)*axis).sum(1),(first*second).sum(1))
    rows,cols,values=[],[],[]
    eps=guard.scale*2e-8
    for corner in range(6):
        ids=variable[corners[:,corner]]
        include=ids>=0
        for j in range(3):
            original=points[:,corner,j].copy()
            points[:,corner,j]=original+eps
            plus=angle()
            points[:,corner,j]=original-eps
            minus=angle()
            points[:,corner,j]=original
            derivative=unit*guard.angle_difference(plus,minus)/(2*eps)
            rows.extend(row[include])
            cols.extend(3*ids[include]+j)
            values.extend(derivative[include])
    return csr_matrix((values,(rows,cols)),shape=(len(corners),3*(variable.max()+1)))
