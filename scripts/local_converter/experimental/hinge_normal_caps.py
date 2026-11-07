"""Sufficient fixed-axis normal bounds for an originally oriented hinge."""
import numpy as np


def caps(guard, indices):
    left,right=guard.graph.face_pairs.T
    points=guard.baseline[guard.faces]
    normals=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    result=[]
    for index in indices:
        i,j=left[index],right[index]
        first,second=normals[i],normals[j]
        angle=np.arccos(np.clip(first@second,-1.,1.))
        bisector=first+second
        if np.linalg.norm(bisector)<1e-9:
            raise ValueError("Ambiguous normal bisector")
        low,high=0.,guard.bend_limit[index]
        if guard.crease[index]:
            signed=guard.reference["bend"][index]
            error=guard.angle_difference(signed,guard.rest_bend[index])
            center=signed-error
            width=max(abs(error),1e-7)
            sign=1 if signed>=0 else -1
            low=max(0.,sign*center-width)
            high=min(high,sign*center+width)
            if angle>1e-6 and low==0:
                low=angle*.01
        high=min(np.pi,max(angle,high))
        axis=bisector/np.linalg.norm(bisector)
        result.extend([(int(i),axis,high/2),(int(j),axis,high/2)])
        if low>1e-8:
            tangent=second-first*(first@second)
            magnitude=np.linalg.norm(tangent)
            if magnitude<1e-9:
                raise ValueError("Ambiguous lower hinge bound")
            tangent/=magnitude
            separation=max(angle,low+min(np.pi/6,(np.pi-low)/2))
            radius=(separation-low)/2
            for face,t in ((i,(angle-separation)/2),(j,(angle+separation)/2)):
                result.append((int(face),np.cos(t)*first+np.sin(t)*tangent,radius))
    return result


def cap_frame(triangle,axis):
    sides=(triangle[1:]-triangle[0]).T
    first=sides[:,0]/np.linalg.norm(sides[:,0])
    second=sides[:,1]-first*(first@sides[:,1])
    second/=np.linalg.norm(second)
    tangent=np.column_stack((first,second))
    inverse=np.linalg.inv(tangent.T@sides)
    weights=np.vstack((-inverse.sum(0),inverse))
    projected=tangent-np.outer(axis,axis@tangent)
    left,_,right=np.linalg.svd(projected,full_matrices=False)
    return weights,left@right
