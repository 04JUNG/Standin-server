"""Image-only, closed-set pose retrieval pilot; never reads GT in inference.

Render -> freeze answers -> infer PNGs -> retrieve -> score exact source IDs.
Both arms share RTMW3D image XY, masks, gallery and view aggregation.
This is a Z ablation, not a comparison with the deployed RTMPose model.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_z_axis_search import load_library
from standin_eval.z_axis_search import (
    build_search_index, distances, predicted_xyz_features, rotate_y,
)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def render_pose(joints, yaw, size=512, style="solid", seed=0, *, pitch=0., distance=None):
    """Orthographic 3D ellipsoid mannequin with z-buffer and surface lighting.

    Screen X right, Y down; world Y up; camera looks along +Z (larger Z
    means farther). Unlike 2D sticks, overlapping limbs have depth occlusion.
    Faces/feet are approximated from body axes, not a rigged character mesh.
    """
    if style not in ("solid", "rough"):
        raise ValueError("unknown rendering style")
    points = rotate_y(joints, yaw).astype(float)
    hip = (points[11] + points[12]) / 2
    shoulder = (points[5] + points[6]) / 2
    torso = float(np.linalg.norm(shoulder - hip))
    if torso < 1e-6:
        raise ValueError("degenerate torso")
    points = (points - hip) / torso
    if pitch:
        angle = np.deg2rad(pitch)
        rotation = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],
                             [0,np.sin(angle),np.cos(angle)]])
        points = points @ rotation.T
    hip = (points[11] + points[12]) / 2
    shoulder = (points[5] + points[6]) / 2
    up = unit(shoulder - hip)
    lateral = unit(points[5] - points[6])
    # COCO left means the subject's anatomical left. In a frontal image it
    # points screen-right, so front points toward -Z for this camera.
    forward = unit(np.cross(up, lateral))
    lateral = unit(np.cross(forward, up))
    frame = np.column_stack((lateral, up, forward))
    objects = []
    skin = np.array([218, 178, 132.])
    shirt = np.array([95, 133, 170.])
    pants = np.array([70, 79, 99.])

    def ellipsoid(center, radii, basis, color):
        objects.append((np.asarray(center), np.asarray(radii), basis, color))

    def limb(a, b, radius, color):
        axis = unit(b - a)
        helper = np.array([1., 0, 0]) if abs(axis[0]) < .9 else np.array([0., 1, 0])
        cross = unit(np.cross(axis, helper))
        basis = np.column_stack((cross, axis, np.cross(cross, axis)))
        ellipsoid((a+b)/2, (radius, np.linalg.norm(b-a)/2+radius*.45, radius), basis, color)

    width = np.linalg.norm(points[5]-points[6]) / 2
    ellipsoid(hip+up*.56, (max(.25, width), .57, .22), frame, shirt)
    ellipsoid(hip+up*.02, (.30, .23, .21), frame, pants)
    for a, b in ((5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)):
        leg = a >= 11
        radius = (.13 if a in (11,12) else .10) if leg else .085
        color = pants if leg else skin
        limb(points[a], points[b], radius, color)
        ellipsoid(points[b], (radius, radius, radius), np.eye(3), color)
    for wrist in (9,10):
        ellipsoid(points[wrist], (.105,.13,.075), frame, skin)
    for ankle in (15,16):
        ellipsoid(points[ankle]+forward*.10, (.105,.09,.22), frame, pants)
    head = points[0]
    limb(shoulder, head, .11, skin)
    ellipsoid(head, (.22,.29,.21), frame, skin)
    # Small facial geometry provides an orientation cue without identity text.
    ellipsoid(head+forward*.215, (.045,.065,.07), frame, skin)
    for side in (-1,1):
        ellipsoid(head+forward*.19+up*.07+lateral*side*.083,
                  (.028,.018,.022), frame, np.array([48.,43,39]))

    # Conservative bounds include the actual rendered volume, not only joints.
    extent = [(c-np.abs(b)@r, c+np.abs(b)@r) for c,r,b,_ in objects]
    lo = np.min([v[0] for v in extent], axis=0)
    hi = np.max([v[1] for v in extent], axis=0)
    if distance is not None:
        # Project conservative 3D bounding-box corners, not an existing image.
        if distance <= 0 or lo[2]+distance <= .1:
            raise ValueError("camera intersects rendered volume")
        corners = np.array([[x,y,z] for x in (lo[0],hi[0])
                            for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        bounds = corners[:,:2]/(corners[:,2,None]+distance)
        lo[:2],hi[:2] = bounds.min(axis=0),bounds.max(axis=0)
    scale = (size-80)/max(hi[0]-lo[0], hi[1]-lo[1])
    cx, cy = (lo[:2]+hi[:2])/2
    yy, xx = np.mgrid[:size, :size]
    wx = (xx-size/2)/scale+cx
    wy = -(yy-size/2)/scale+cy
    depth = np.full((size,size), np.inf)
    surface_ids = np.full((size,size), -1, dtype=np.int16)
    canvas = np.full((size,size,3), 246., dtype=float)
    light = unit(np.array([-.5,.7,-1.]))
    for object_id, (center, radii, basis, color) in enumerate(objects):
        extent = np.abs(basis)@radii
        if distance is None:
            x0 = max(0, int((center[0]-extent[0]-cx)*scale+size/2)-2)
            x1 = min(size, int((center[0]+extent[0]-cx)*scale+size/2)+3)
            y0 = max(0, int(-(center[1]+extent[1]-cy)*scale+size/2)-2)
            y1 = min(size, int(-(center[1]-extent[1]-cy)*scale+size/2)+3)
        else:
            corners = np.array([[x,y,z] for x in (center[0]-extent[0],center[0]+extent[0])
                                for y in (center[1]-extent[1],center[1]+extent[1])
                                for z in (center[2]-extent[2],center[2]+extent[2])])
            bounds = corners[:,:2]/(corners[:,2,None]+distance)
            left,bottom = bounds.min(axis=0); right,top = bounds.max(axis=0)
            x0=max(0,int((left-cx)*scale+size/2)-2)
            x1=min(size,int((right-cx)*scale+size/2)+3)
            y0=max(0,int(-(top-cy)*scale+size/2)-2)
            y1=min(size,int(-(bottom-cy)*scale+size/2)+3)
        if x0>=x1 or y0>=y1:
            continue
        q = basis@np.diag(1/radii**2)@basis.T
        if distance is None:
            dx = wx[y0:y1,x0:x1]-center[0]
            dy = wy[y0:y1,x0:x1]-center[1]
            linear = q[2,0]*dx+q[2,1]*dy
            constant = q[0,0]*dx**2+2*q[0,1]*dx*dy+q[1,1]*dy**2-1
            discriminant = linear**2-q[2,2]*constant
            dz = (-linear-np.sqrt(np.maximum(discriminant,0)))/q[2,2]
            z = dz+center[2]
            visible = (discriminant>=0)&(z<depth[y0:y1,x0:x1])
            normals = np.stack((dx,dy,dz),axis=-1)@q.T
        else:
            rays = np.stack((wx[y0:y1,x0:x1],wy[y0:y1,x0:x1],
                             np.ones((y1-y0,x1-x0))),axis=-1)
            origin = np.array([0.,0.,-distance])-center
            quadratic = np.einsum('...i,ij,...j->...',rays,q,rays)
            linear = rays @ q @ origin
            discriminant = linear**2-quadratic*(origin @ q @ origin-1)
            ray_depth = (-linear-np.sqrt(np.maximum(discriminant,0)))/quadratic
            z=ray_depth-distance
            visible=(discriminant>=0)&(ray_depth>.1)&(z<depth[y0:y1,x0:x1])
            normals=(origin+rays*ray_depth[...,None])@q.T
        normals /= np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-8)
        diffuse = np.maximum(normals@light,0)
        rgb = np.clip(color*(.48+.52*diffuse[...,None]),0,255)
        canvas[y0:y1,x0:x1][visible] = rgb[visible]
        depth[y0:y1,x0:x1][visible] = z[visible]
        surface_ids[y0:y1,x0:x1][visible] = object_id
    plane=points[:,:2] if distance is None else points[:,:2]/(points[:,2,None]+distance)
    projected = np.column_stack(((plane[:,0]-cx)*scale+size/2,
                                  -(plane[:,1]-cy)*scale+size/2))
    if style == "rough":
        # Generate vector pen strokes directly from BVH-derived surface geometry.
        # No existing image is filtered, traced by a model, or used as input.
        # Depth-buffer boundaries retain occlusion; only the stroke varies.
        import cv2
        drawing = Image.new("RGB", (size*2,size*2), "white")
        pen = ImageDraw.Draw(drawing)
        rng = np.random.default_rng(seed)
        for object_id in range(len(objects)):
            contours, _ = cv2.findContours(
                (surface_ids==object_id).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour)<3:
                    continue
                path = cv2.approxPolyDP(contour,.65,True).reshape(-1,2).astype(float)
                path = np.concatenate((path,path[:1]))
                pen.line([tuple(v*2) for v in path],fill=(80,80,80),width=2)
                # Subpixel duplicate strokes imitate the inspected roughs;
                # the underlying projected joint coordinates never change.
                if len(path)>8:
                    jitter = rng.normal(0,.65,path.shape)
                    pen.line([tuple(v*2) for v in path+jitter],fill=(175,175,175),width=1)
        return drawing.resize((size,size),Image.Resampling.LANCZOS), projected
    return Image.fromarray(canvas.astype(np.uint8)), projected


def retrieve(index, xy, z, mask, weight):
    """No answer ID or source coordinates are accepted by retrieval."""
    values = distances(index, xy, z, valid_mask=mask, z_weight=weight)
    seen, candidates = set(), []
    for row in np.argsort(values, kind="stable"):
        pose_id = index.pose_ids[row]
        if pose_id in seen:
            continue
        seen.add(pose_id)
        candidates.append({"pose_id": pose_id, "distance": float(values[row]),
                           "view_degrees": float(index.view_degrees[row])})
    return candidates


def score_records(records):
    """Failed extractions stay in the denominator and count as misses."""
    n = len(records)
    return {arm: {f"top{k}": sum(r.get(arm+"_rank") is not None and
                                 r[arm+"_rank"]<=k for r in records)/n
                  for k in (1,5)} for arm in ("xy","xyz")}


def infer_images(paths, model):
    """Inference only sees anonymous PNG pixels. Save raw output for auditing."""
    import cv2
    output = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        try:
            k3, scores, _, k2 = model(image)
            output.append({"image":path.name,"keypoints2d":k2[0,:17].tolist(),
                           "keypoints3d":k3[0,:17].tolist(),"scores":scores[0,:17].tolist()})
        except Exception as error:
            output.append({"image":path.name,"error":type(error).__name__+": "+str(error)})
        print(f"[infer] {path.name}",flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root",required=True,type=Path)
    parser.add_argument("--model",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--samples",type=int,default=20)
    parser.add_argument("--seed",type=int,default=20261012)
    parser.add_argument("--z-weight",type=float,default=.25)
    parser.add_argument("--render-style",choices=("solid","rough"),default="solid")
    parser.add_argument("--exclude-answers",type=Path,help="Exclude development query families from query sampling, not gallery")
    args = parser.parse_args()
    if args.samples<=0 or args.z_weight<0:
        raise ValueError("invalid sample count or weight")
    args.out.mkdir(parents=True,exist_ok=False)
    images = args.out/"images"
    images.mkdir()
    poses = load_library(args.library_root.resolve())
    index = build_search_index(poses)
    # One representative per family in query sampling; exact pose ID scoring.
    unique = {}
    for pose in poses:
        unique.setdefault(pose.family_id,pose)
    pool = list(unique.values())
    excluded = set()
    if args.exclude_answers:
        excluded = {r['family_id'] for r in json.loads(args.exclude_answers.read_text())}
        pool = [pose for pose in pool if pose.family_id not in excluded]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(pool),args.samples,replace=False)
    answers, paths = [], []
    for i, selected in enumerate(chosen):
        pose = pool[selected]
        yaw = (0.,45.,90.,180.)[i%4]
        rendered, gt_xy = render_pose(pose.joints,yaw,style=args.render_style,seed=args.seed+i)
        path = images/f"query_{i+1:03}.png"
        rendered.save(path)
        paths.append(path)
        answers.append({"image":path.name,"pose_id":pose.pose_id,
                        "family_id":pose.family_id,"yaw":yaw,"sha256":sha256(path),
                        "gt_projected_xy":gt_xy.tolist()})
    # This protocol is written BEFORE any image inference or metric inspection.
    protocol = {"samples":args.samples,"seed":args.seed,"gallery_poses":len(poses),
                "excluded_query_families":sorted(excluded),
                "gallery_views":[0,45,90,180],"query_views":[0,45,90,180],
                "target":"exact pose_id; views collapsed, mirrors NOT accepted as equivalent",
                "renderer":"ellipsoid-depth-buffer-v2; anatomical front=up cross left; camera looks +Z; Y up",
                "render_style":args.render_style,
                "rough_source":"BVH vector contours with subpixel duplicate strokes; NOT human or AI-drawn" if args.render_style=="rough" else None,
                "input":"PNG only; full image single person crop; no detector or VLM",
                "arms":"same RTMW3D XY and mask; weights 0 versus fixed .25 by default",
                "z_weight":args.z_weight,"torso_meters_prior":.5,"kpt_threshold":.3,
                "model":str(args.model.resolve()),"model_sha256":sha256(args.model),
                "library_db_sha256":sha256(args.library_root/"poses.db"),
                "script_sha256":sha256(Path(__file__)),
                "failure_rule":"<6 body joints or any missing shoulder/hip anchor: both miss",
                "limitations":["synthetic mannequin, not real rough art","20-case exploratory pilot",
                               "same-camera closed-set retrieval; no held-out-pose generalization",
                               "fixed metric torso prior and model Z calibration remain experimental",
                               "XY arm is RTMW3D XY, not deployed RTMPose XY"]}
    save_json(args.out/"protocol.json",protocol)
    save_json(args.out/"answers.json",answers)
    from rtmlib import RTMPose3d
    model = RTMPose3d(str(args.model.resolve()),model_input_size=(288,384),
                     backend="onnxruntime",device="cpu")
    predictions = infer_images(paths,model)
    save_json(args.out/"predictions.json",predictions)
    records = []
    for prediction, answer in zip(predictions,answers):
        record = {"image":prediction["image"],"target":answer["pose_id"],"yaw":answer["yaw"],
                  "xy_rank":None,"xyz_rank":None}
        if "error" in prediction:
            record["error"] = prediction["error"]
        else:
            k3=np.asarray(prediction["keypoints3d"])
            k2=np.asarray(prediction["keypoints2d"])
            scores=np.asarray(prediction["scores"])
            if not all(np.isfinite(v).all() for v in (k3,k2,scores)):
                record["error"]="nonfinite_model_output"
            else:
                xy,z,mask=predicted_xyz_features(k3,k2,scores)
                record["valid_body_joints"]=int(mask[5:].sum())
                if mask[5:].sum()<6 or not mask[[5,6,11,12]].all():
                    record["error"]="insufficient_body_or_anchors"
                else:
                    for arm,weight in (("xy",0.),("xyz",args.z_weight)):
                        candidates=retrieve(index,xy,z,mask,weight)
                        record[arm+"_rank"]=next(j+1 for j,c in enumerate(candidates)
                                                if c["pose_id"]==record["target"])
                        record[arm+"_top5"]=candidates[:5]
                # Diagnostic only, after predictions; never affects retrieval.
                gt=np.asarray(answer["gt_projected_xy"])
                torso_px=np.linalg.norm((gt[5]+gt[6]-gt[11]-gt[12])/2)
                record["xy_error_torso_ratio"]=float(np.linalg.norm(k2[5:]-gt[5:],axis=1).mean()/torso_px)
        records.append(record)
    metrics=score_records(records)
    failed=sum("error" in r for r in records)
    wins=sum(r["xyz_rank"] is not None and r["xyz_rank"]<=5 and
             (r["xy_rank"] is None or r["xy_rank"]>5) for r in records)
    losses=sum(r["xy_rank"] is not None and r["xy_rank"]<=5 and
               (r["xyz_rank"] is None or r["xyz_rank"]>5) for r in records)
    result={"protocol":protocol,"metrics":metrics,"failures":failed,
            "top5_gains":wins,"top5_losses":losses,"records":records}
    save_json(args.out/"results.json",result)
    render_label = "합성 러프 선화" if args.render_style=="rough" else "입체 마네킹"
    report=["# 포즈 이미지 → 원본 포즈 검색: Z 추가 실험","",
            "동일한 RTMW3D 추정 XY에 Z를 추가하는 통제 실험. 기존 배포 모델과의 비교는 아님.","",
            f"라이브러리 {len(poses):,}개에서 고정 시드로 고른 {args.samples}개 포즈를 {render_label} PNG로 렌더링.",
            "검색에는 PNG에서 추정한 좌표만 전달하고, 원본 ID는 채점에만 사용. 원본 XYZ는 쿼리에 넣지 않음.","",
            "| 지표 | XY만 | XY+Z | 변화 |","|---|---:|---:|---:|"]
    for k in (5,1):
        a,b=metrics["xy"][f"top{k}"],metrics["xyz"][f"top{k}"]
        report.append(f"| 정답 Top-{k} 적중률 | {a:.0%} ({round(a*args.samples)}/{args.samples}) | "
                      f"{b:.0%} ({round(b*args.samples)}/{args.samples}) | {(b-a)*100:+.0f}%p |")
    report += ["",f"추출 실패 {failed}건도 분모에 포함. Top-5 성공 전환 {wins}건, 실패 전환 {losses}건.","",
               "## 해석 범위","",
               "- 정답은 원본 pose_id 일치. 같은 포즈의 여러 카메라 뷰만 합치며, 미러/유사 포즈는 정답 처리하지 않음.",
               "- 검색 라이브러리에 정답이 포함된 closed-set 시험. 카메라는 색인과 같은 0/45/90/180도로 고정.",
               "- 원본을 이미 알고 만든 이미지이므로 실제 러프 성능이나 작가의 사용 가능성을 의미하지 않음.",
               "- XY와 Z 모두 같은 RTMW3D 출력. 모델 교체 효과를 제외한 Z 기여도만 비교.",
               "- Z 가중치 0.25, 몸통 길이 가정 0.5m, 점수 기준 0.3은 결과를 보기 전에 고정. 테스트 결과로 튜닝하지 않음.",
               "- 몸통 길이 가정·Z 추정·단순 마네킹의 표현 한계가 결과에 영향을 줌. 20개는 탐색용이며 도입 결론을 내리지 않음.",
               "- VLM·검출기·후속 보정은 실행하지 않은 단일 인물 검색 실험.","",
               "- rough 모드는 BVH 표면 경계를 선으로 그린 합성 러프. 사람이 그린 러프나 생성 AI의 그림이 아님. 원본 관절·카메라·가림 관계를 유지한다.","",
               "## 확인 파일","","- review.html: 입력/추정 관절과 각 사례의 정답 순위",
               "- protocol.json / answers.json: 고정 조건과 정답표",
               "- predictions.json: 이미지에서 추정한 원시 XY/Z/점수",
               "- results.json: 후보 ID와 정확한 채점 결과","",
               "모델 구현 참고: https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/pose_estimation/rtmpose3d.py",""]
    (args.out/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    cards=[]
    sheet=Image.new("RGB",(5*230,((len(records)+4)//5)*290),"white")
    for i,(r,p) in enumerate(zip(records,predictions)):
        im=Image.open(images/r["image"]).convert("RGB")
        overlay=im.copy()
        draw=ImageDraw.Draw(overlay)
        if "keypoints2d" in p:
            for j,(x,y) in enumerate(p["keypoints2d"]):
                if j<5: continue
                color="green" if p["scores"][j]>=.3 else "red"
                draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
        overlay.save(images/(Path(r["image"]).stem+"_overlay.png"))
        sheet.paste(im.resize((220,220)),((i%5)*230,(i//5)*290))
        draw=ImageDraw.Draw(sheet)
        draw.text(((i%5)*230+5,(i//5)*290+224),
                  f'{i+1:02} yaw={r["yaw"]:g}\nXY #{r["xy_rank"]} / XYZ #{r["xyz_rank"]}',fill="black")
        candidates="".join(f'<p>{arm.upper()}: '+html.escape(
            " → ".join(c["pose_id"] for c in r.get(arm+"_top5",[])))+"</p>" for arm in ("xy","xyz"))
        cards.append(f'<article><h2>{i+1:02}. {html.escape(r["target"])}</h2>'
                     f'<img src="images/{r["image"]}"><img src="images/{Path(r["image"]).stem}_overlay.png">'
                     f'<p>XY 정답 순위: {r["xy_rank"]} / XY+Z: {r["xyz_rank"]} · '
                     f'{html.escape(r.get("error","추출 성공"))}</p>{candidates}</article>')
    sheet.save(args.out/"contact_sheet.png")
    (args.out/"review.html").write_text('<!doctype html><meta charset="utf-8"><title>Z 검색 실험</title>'
        '<style>body{font:15px system-ui;background:#f5f5f4;margin:30px}article{background:white;padding:20px;margin:20px 0;overflow-wrap:anywhere}img{width:280px}h2{font-size:16px}</style>'
        '<h1>원본 포즈 검색 · XY vs XY+Z</h1><p>왼쪽: 입력 PNG / 오른쪽: 이미지에서 추정한 관절. 초록: 점수 ≥ 0.3, 빨강: 미달. 숫자는 원본 포즈의 검색 순위.</p>'
        +''.join(cards),encoding="utf-8")
    print(json.dumps({"metrics":metrics,"failures":failed,"top5_gains":wins,"top5_losses":losses},indent=2),flush=True)
    print(f"[saved] {args.out.resolve()}")


if __name__=="__main__":
    main()
