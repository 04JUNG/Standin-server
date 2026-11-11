"""Offline, measured-volume blocking. Safety is a constraint, never a score.

This is an authoring proxy, not a substitute for deformed native-mesh checks.
Coordinates are subject-left / up / forward in metres.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def unit(v):
    v=np.asarray(v,float)
    return v/max(np.linalg.norm(v),1e-12)


def hand_frame(direction,twist):
    u=unit(direction)
    n=unit(np.array([0.,0.,-1.])-u*np.dot(u,[0,0,-1]))
    n=Rotation.from_rotvec(u*twist).apply(n)
    return np.column_stack((u,n,np.cross(u,n)))

def pair_frame(a,b):
    a=unit(a);b=unit(b-a*(a@b))
    return np.column_stack((a,b,np.cross(a,b)))

def align(a,b):
    a=unit(a);b=unit(b);v=np.cross(a,b);c=np.clip(a@b,-1,1)
    if np.linalg.norm(v)<1e-9:return np.eye(3)
    return Rotation.from_rotvec(unit(v)*np.arccos(c)).as_matrix()


def line_clearance(points,a,b,r0,r1):
    d=b-a
    t=np.clip((points-a)@d/max(d@d,1e-12),0,1)
    return np.linalg.norm(points-a-t[:,None]*d,axis=1)-(r0+(r1-r0)*t)


class FoldBlockout:
    """Full paired gesture with measured arm volumes and articulated hand proxies."""
    def __init__(self,data):
        self.data=data
        self.rest={n:np.array(v) for n,v in data['rest'].items()}
        self.seed={n:np.array(v) for n,v in data['seed'].items()}
        self.tail={n:np.array(v) for n,v in data['seed_tail'].items()}
        hull=ConvexHull(np.array(data['torso_vertices']))
        self.planes=hull.equations
        self.profiles={};self.hands={};self.hand_frames={}
        self.rest_hand_frames={}
        for side,sign in [('Left',1),('Right',-1)]:
            for part in ['Arm','ForeArm']:
                self.profiles[side+part]=np.array(data['radial_95_m'][side+part])
            wrist=self.seed[side+'Hand']
            u=unit(self.seed[side+'HandMiddle1']-wrist)
            n=unit(np.cross(self.seed[side+'HandIndex1']-wrist,self.seed[side+'HandPinky1']-wrist)*(-1 if sign==1 else 1))
            n=unit(n-u*(n@u));frame=np.column_stack((u,n,np.cross(u,n)))
            self.hand_frames[side]=frame
            rw=self.rest[side+'Hand'];ru=unit(self.rest[side+'HandMiddle1']-rw)
            rn=unit(np.cross(self.rest[side+'HandIndex1']-rw,self.rest[side+'HandPinky1']-rw)*(-1 if sign==1 else 1))
            self.rest_hand_frames[side]=pair_frame(ru,rn)
            points=[];radii=[];tags=[]
            # Four metacarpal rails retain real palm width and thickness.
            for finger in ['Index','Middle','Ring','Pinky']:
                base=self.seed[side+'Hand'+finger+'1']
                for t in [.2,.45,.7,1.]:
                    points.append(frame.T@((1-t)*wrist+t*base-wrist));radii.append(.014);tags.append('palm')
            for finger in ['Thumb','Index','Middle','Ring','Pinky']:
                for segment in [1,2,3]:
                    name=side+'Hand'+finger+str(segment)
                    for t in [0.,.5,1.]:
                        p=(1-t)*self.seed[name]+t*self.tail[name]
                        points.append(frame.T@(p-wrist));radii.append(.009 if finger!='Thumb' else .011);tags.append(finger)
            self.hands[side]=(np.array(points),np.array(radii),tags)

    def seed_parameters(self,spec):
        values=[]
        for side in ['Left','Right']:
            c=spec[side.lower()];f=self.hand_frames[side]
            u=f[:,0];yaw=np.arctan2(u[2],u[0]);elev=np.arcsin(u[1])
            zero=hand_frame(u,0)[:,1]
            twist=np.arctan2(u@np.cross(zero,f[:,1]),zero@f[:,1])
            values.extend([*self.seed[side+'ForeArm'],*self.seed[side+'Hand'],yaw,elev,twist,
                           np.radians(c['protraction_deg']),np.radians(c.get('depression_deg',0))])
        return np.array(values)

    def decode(self,x):
        result={}
        for k,(side,sign) in enumerate([('Left',1),('Right',-1)]):
            v=x[k*11:k*11+11];e,w=v[:3],v[3:6]
            # Blender torso coordinates form a right-handed X/Y/Z frame here.
            pro=Rotation.from_rotvec(np.array([0.,-sign*v[9],0.]))
            dep=Rotation.from_rotvec(np.array([0.,0.,-sign*v[10]]))
            sh=self.rest[side+'Shoulder'];s=sh+dep.apply(pro.apply(self.rest[side+'Arm']-sh))
            u=np.array([np.cos(v[7])*np.cos(v[6]),np.sin(v[7]),np.cos(v[7])*np.sin(v[6])])
            h=hand_frame(u,v[8]);local,radii,tags=self.hands[side]
            hp=local@h.T+w
            upper=unit(e-s);lower=unit(w-e)
            ru=unit(self.rest[side+'ForeArm']-self.rest[side+'Arm']);rl=unit(self.rest[side+'Hand']-self.rest[side+'ForeArm'])
            du=pair_frame(upper,lower)@pair_frame(ru,np.array([0.,0.,1.])).T
            neutral=align(du@rl,lower)@du
            rolls=np.radians(np.arange(-100,101,2));fore=Rotation.from_rotvec(rolls[:,None]*lower).as_matrix()@neutral
            dh=h@self.rest_hand_frames[side].T
            q=Rotation.from_matrix(fore.transpose(0,2,1)@dh).as_quat()
            mag=np.clip(np.sqrt(q[:,3]**2+(q[:,:3]@rl)**2),1e-12,1)
            swing=np.degrees(2*np.arccos(mag));twist=np.degrees(2*np.arccos(np.clip(abs(q[:,3])/mag,0,1)))
            rank=np.lexsort((twist+abs(np.degrees(rolls))*.01,np.maximum(0,swing-65)+np.maximum(0,twist-35)))
            best=rank[0]
            result[side]={'shoulder':s,'elbow':e,'wrist':w,'hand_frame':h,'hand_points':hp,
                          'hand_radii':radii,'hand_tags':tags,'Arm_delta':du,'ForeArm_delta':fore[best],
                          'wrist_swing':float(swing[best]),'wrist_twist':float(twist[best])}
        return result

    def bone_equalities(self,x):
        pose=self.decode(x);rows=[]
        for s,p in pose.items():
            a=np.linalg.norm(self.rest[s+'ForeArm']-self.rest[s+'Arm'])
            b=np.linalg.norm(self.rest[s+'Hand']-self.rest[s+'ForeArm'])
            rows.extend([np.linalg.norm(p['elbow']-p['shoulder'])-a,np.linalg.norm(p['wrist']-p['elbow'])-b])
        return np.array(rows)

    def volumes(self,pose):
        regions={}
        for s,p in pose.items():
            for part,a,b in [('Arm',p['shoulder'],p['elbow']),('ForeArm',p['elbow'],p['wrist'])]:
                t=np.linspace(0,1,13)
                # Piecewise profile from actual 95th-percentile radial skin data.
                rs=np.interp(t,[.125,.375,.625,.875],self.profiles[s+part])
                if 'ellipse_profiles' in self.data:
                    ell=self.data['ellipse_profiles'][s+part];r=np.array(ell['radii'])
                    major=np.interp(t,[.125,.375,.625,.875],r[:,0]);minor=np.interp(t,[.125,.375,.625,.875],r[:,1])
                    axis=p[part+'_delta']@np.array(ell['major_axis_rest'])
                    center=a+(b-a)*t[:,None]
                    # Three measured minor-radius rails form an oval section.
                    ps=np.concatenate([center+sign*(major-minor)[:,None]*axis for sign in [-1,0,1]])
                    regions[s+part]=(ps,np.tile(minor+.001,3))
                else:regions[s+part]=(a+(b-a)*t[:,None],rs)
            regions[s+'Hand']=(p['hand_points'],p['hand_radii'])
        return regions

    def joint_limits(self,x):
        rows=[]
        for p in self.decode(x).values():
            upper=unit(p['elbow']-p['shoulder']);lower=unit(p['wrist']-p['elbow'])
            rows.extend([upper@lower-np.cos(np.radians(150)),(65-p['wrist_swing'])/65,(35-p['wrist_twist'])/35])
        return np.array(rows)

    def safety(self,x,with_names=False):
        pose=self.decode(x);regions=self.volumes(pose);rows=[];names=[]
        for side in ['Left','Right']:
            p=pose[side]
            for part in ['Arm','ForeArm','Hand']:
                points,radii=regions[side+part]
                # Match the native local shoulder seam; no whole-region exemption.
                length=np.linalg.norm(self.rest[side+'ForeArm']-self.rest[side+'Arm'])
                use=np.tile(np.linspace(0,length,13)>=.155,len(points)//13) if part=='Arm' else np.ones(len(points),dtype=bool)
                distances=np.max(points@self.planes[:,:3].T+self.planes[:,3],axis=1)-radii-.003
                rows.extend(distances[use]);names.extend([side+part+' / torso']*sum(use))
        # No cross-arm seam exemption, including the hands and all fingers.
        for a in ['LeftArm','LeftForeArm','LeftHand']:
            for b in ['RightArm','RightForeArm','RightHand']:
                pa,ra=regions[a];pb,rb=regions[b]
                d=np.linalg.norm(pa[:,None,:]-pb[None,:,:],axis=2)-ra[:,None]-rb[None,:]-.003
                # Sphere samples are overlapping along each rail. This is a
                # measured blockout only; exact native skin is the final screen.
                rows.extend(d.ravel());names.extend([a+' / '+b]*d.size)
        result=np.array(rows)
        return (result,names) if with_names else result

    def roles(self,x):
        p=self.decode(x);l,r=p['Left'],p['Right'];rows=[]
        # The lower forearm crosses UNDER the front forearm. Its wrist has
        # already emerged ABOVE that crossing in the supplied photograph.
        def at_x(a,b,x):
            t=np.clip((x-a[0])/(b[0]-a[0]+1e-9),0,1)
            return a+t*(b-a)
        front=at_x(l['elbow'],l['wrist'],r['wrist'][0])
        def cross(a,b):return a[0]*b[1]-a[1]*b[0]
        a=l['elbow'];d=l['wrist']-a;b=r['elbow'];e=r['wrist']-b
        denom=cross(d[:2],e[:2]);denom=denom if abs(denom)>1e-8 else np.copysign(1e-8,denom or 1.)
        t=cross((b-a)[:2],e[:2])/denom;u=cross((b-a)[:2],d[:2])/denom
        lc=a+t*d;rc=b+u*e
        rows.extend([t*.24,(1-t)*.24,u*.24,(1-u)*.24,lc[2]-rc[2]-.03,r['wrist'][1]-front[1]-.005])
        mid=[j for j,t in enumerate(r['hand_tags']) if t=='Middle'][-1]
        tip=r['hand_points'][mid]
        rows.extend([tip[1]-front[1]-.04,tip[0]+.01])
        a,b=l['shoulder'],l['elbow'];axis=b-a
        t=np.clip((tip-a)@axis/(axis@axis),.2,.9);target=a+t*axis
        arm_radius=np.interp(t,[.125,.375,.625,.875],self.profiles['LeftArm'])
        rows.extend([target[0]-tip[0],arm_radius+.03-np.linalg.norm(tip-target)])
        # The hidden fingertips wrap behind the bicep. The whole palm need not
        # sit behind its axis: that would force an unnecessary torso collision.
        ids=[[j for j,t in enumerate(l['hand_tags']) if t==f][-1] for f in ['Index','Middle','Ring']]
        tucked=l['hand_points'][ids].mean(axis=0)
        a,b=r['shoulder'],r['elbow'];t=np.clip((tucked[1]-a[1])/(b[1]-a[1]+1e-9),.15,.95);target=a+t*(b-a)
        rad=np.interp(t,[.125,.375,.625,.875],self.profiles['RightArm'])
        rows.extend([rad-.008-abs(tucked[0]-target[0]),target[2]-tucked[2]-.025,
                     rad+.035-np.linalg.norm(tucked-target)])
        return np.array(rows)

    def appearance(self,x):
        p=self.decode(x)
        # A single front-view block from the supplied photo, not a pose grid.
        targets={'Left':([.18,-.265],[-.075,-.205]),'Right':([-.17,-.285],[.065,-.24])}
        cost=0.
        for s,(e,w) in targets.items():
            cost+=np.sum((p[s]['elbow'][:2]-e)**2)*2+np.sum((p[s]['wrist'][:2]-w)**2)*3
            mid=[j for j,t in enumerate(p[s]['hand_tags']) if t=='Middle'][-1]
            target=[-.18,-.18] if s=='Left' else [.125,-.125]
            cost+=np.sum((p[s]['hand_points'][mid,:2]-target)**2)*4
        cost+=.0002*np.sum((x[[9,10,20,21]]-np.radians([40,5,40,5]))**2)
        return cost

    def gesture_axes(self,x):
        p=self.decode(x)
        return np.array([-p['Left']['hand_frame'][0,0]-.5,p['Right']['hand_frame'][1,0]-.65,
                         p['Left']['wrist'][1]-p['Left']['elbow'][1]-.025,
                         p['Right']['wrist'][1]-p['Right']['elbow'][1]-.03])

    def to_spec(self,x,template):
        import copy
        spec=copy.deepcopy(template);p=self.decode(x)
        for k,s in enumerate(['Left','Right']):
            q=p[s];c=spec[s.lower()]
            c.update(self.data.get('hand_forms',{}).get(s.lower(),{}))
            c.update(wrist=q['wrist'].tolist(),pole=q['elbow'].tolist(),hand_direction=q['hand_frame'][:,0].tolist(),
                     palm_normal=q['hand_frame'][:,1].tolist(),protraction_deg=float(np.degrees(x[k*11+9])),depression_deg=float(np.degrees(x[k*11+10])))
        spec.update(id='contact_blockout',optimize=False,between_arm_and_torso=True)
        for name in ['raised_wrist_y_range','raised_hand_side_range','rising_wrist_depth_range']:
            spec.pop(name,None)
        return spec
