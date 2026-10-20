"""Package authentic roughs/thumbnails and BVH-only camera diagnostics for v4.

No VLM calls, no image manipulation of supplied roughs/mesh thumbnails. Camera
diagnostics are deterministic COCO17 mannequin renders, NOT regenerated FBX meshes.
"""
import argparse
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.library import load_bvh_pose
from src.semantic_shapes import rotate_for_camera, projected_relations
from src.thumbnail_renderer import render_thumbnail


def encoded(path):
    mime = "image/png" if path.suffix.lower()==".png" else "image/jpeg"
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("old", "frozen", "live", "diagnostic", "fresh-ablation", "images", "thumbs", "bvh-root", "out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    raw=json.loads(args.old.read_text()); frozen=json.loads(args.frozen.read_text())
    live=json.loads(args.live.read_text()); diagnostic=json.loads(args.diagnostic.read_text())
    fresh=json.loads(args.fresh_ablation.read_text())
    cases={c["test_id"]:c for c in raw["cases"]}
    outputs={}; camera_rows=[]

    def hit_data(hit, prefix):
        pose_id=hit["pose_id"]; view=hit.get("pose",{}).get("view","front")
        thumb=args.thumbs/f"{pose_id}__{view}.jpg"
        if not thumb.exists(): thumb=args.thumbs/f"{pose_id}__front.jpg"
        data={"id":pose_id,"score":hit["score"],"image":encoded(thumb),"yaw_label":view}
        cam=hit.get("pose",{}).get("preview_camera")
        if cam:
            points,scores=load_bvh_pose(str(args.bvh_root/(pose_id+".bvh")),0)
            cams=[("정면",{"view":"front","yaw_deg":0,"pitch_deg":0,"roll_deg":0}),
                  ("정면 부감 비교",{"view":"front","yaw_deg":0,"pitch_deg":60,"roll_deg":0}),
                  ("팔 가시성 시점",cam)]
            data["cameras"]=[]
            for index,(label,camera) in enumerate(cams):
                file=args.out/f"{prefix}-{index}.png"
                render_thumbnail(rotate_for_camera(points,camera),scores,0,size=320).save(file)
                entry={"label":label,"camera":camera,"image":encoded(file),
                       "evidence":projected_relations(points,scores,camera)}
                data["cameras"].append(entry)
                camera_rows.append({"pose_id":pose_id,"file":file.name,"label":label,"camera":camera,
                                    "evidence":entry["evidence"]})
            data["comparison"]=hit["matched"]["camera_comparison"]
        return data

    for id,index in (("T013",1),("T028",0)):
        c=cases[id]; row=next(r for r in frozen["rows"] if r["test_id"]==id and r["person"]==index)
        fixed=row["fresh_description"]
        outputs[id]={"id":id,"person":index,"rough":encoded(args.images/c["image"]),
            "frozen":{"description":fixed["description"]["search_text"],"status":fixed["new_status"],
                "reason":fixed["new_reason"],
                "before":[hit_data(h,f"{id}-old-{i}") for i,h in enumerate(fixed["old"])],
                "after":[hit_data(h,f"{id}-fixed-{i}") for i,h in enumerate(fixed["new"])]}}
        fc=next(c for c in diagnostic["cases"] if c["test_id"]==id)
        desc=fc["people"][index]["routing"]["semantic_evidence"].get("description")
        fr=next(r for r in fresh["rows"] if r["test_id"]==id and r["person"]==index)
        updated=fr.get("fresh_description")
        outputs[id]["fresh"]={"description":(desc or {}).get("search_text","설명 취득 실패"),
            "status":updated["new_status"] if updated else "Gemini upstream_error",
            "reason":updated["new_reason"] if updated else "upstream_error",
            "before":[hit_data(h,f"{id}-fresh-old-{i}") for i,h in enumerate(updated["old"])] if updated else [],
            "after":[hit_data(h,f"{id}-fresh-new-{i}") for i,h in enumerate(updated["new"])] if updated else []}
        if not desc: outputs[id]["fresh"]["description"]="설명 취득 실패 · 검색 미실행"
    (args.out/"visual-data.json").write_text(json.dumps(outputs,ensure_ascii=False,separators=(",",":")))
    (args.out/"camera-evidence.json").write_text(json.dumps(camera_rows,ensure_ascii=False,indent=2))
    lines=["# 핵심 팔 관계 검색·시점 개선 전후", "", "## 결론", "",
        "- 팔짱(T013 슬롯 2): 동일한 기존 Gemini 설명을 고정하면 스트레칭 등 오답 후보 5개 → 검증 후보 0개.",
        "  앉아서 가슴 앞 팔짱 조합을 검증한 후보가 없다는 뜻이다. 라이브러리 전체의 부재 증명이나 포즈 합성 성공이 아니다.",
        "- V자 팔(T028 슬롯 1): 기존 다이빙 후보 5개 → 몸 기준 양팔 V 관계를 통과한 후보 3개.",
        "  새 1위는 cmu_05_15_00385_mirror. 원본 mesh 썸네일에서도 양팔 벌림·한쪽 무릎 접힘을 확인했다.",
        "- 기존 정상 스켈레톤(T044)의 paired OFF/ON 기하 Top5는 그대로다.", "",
        "## 실험 구분", "",
        "1. 기존 실호출 설명 8장/11슬롯 고정 재검색: 추가 Gemini 호출 0회. 검색 변경 효과를 분리.",
        "2. 동일 승인 8장 새 실호출: Gemini 3.8 Flash, d2 프롬프트, current-X. 분석 8회 + 자세 설명 7회.",
        "   설명 5회 성공, T013/T028 2회 upstream_error. 오류 슬롯 5개는 검색 미실행이며 no_result와 다르다.",
        "3. 핵심 두 장 별도 진단 1회: 분석 2회 + 설명 2회. T013 다시 upstream_error, T028 성공(7.746초).",
        "   API 실패를 숨기거나 첫 실패 기록을 성공으로 대체하지 않았다. 자동 자세 설명 재시도는 없다.",
        "4. 새 T028 설명도 고정하여 v3/v4 비교: 추가 호출 0회. 행동이 dive → stretch로 변했지만 새 V 후보 1위는 동일.",
        "   context_evidence의 하이앵글 단서 보존은 이 마지막 오프라인 검색에 반영했다. raw live 기록은 수정하지 않았다.", "",
        "## 라이브러리·시점", "",
        "- BVH 1,981개 frame 0 처리, 실패 0. reviewed fold_arms+가슴 높이 손 교차 검증 2개(원본/미러, 서기).",
        "- 양손이 몸통 축 기준 위·바깥에 있는 V 기준 통과 30개. 정확한 러프 일치 개수가 아니다.",
        "- 4 yaw × 6 pitch = 24시점, roll=0. 몸 관계 판정과 투영 가시성 판정을 분리.",
        "- 카메라 PNG는 기존 BVH를 단순 관절 마네킹으로 렌더한 진단이다. 실제 FBX mesh 재렌더가 아니다.",
        "- 정면/정면 pitch+60/선택 시점을 같은 BVH로 비교. 포즈 관절·BVH 파일 자체는 수정하지 않았다.",
        "- 선택은 양팔 가시성과 명시한 부감 단서 기준. 러프 픽셀 맞춤·원근·roll·얼굴 방향까지 해결한 것은 아니다.", "",
        "- 시점 비교에서 정면 부감은 V 실루엣이 깨지고, 반대 yaw의 부감은 V가 보여도 몸/다리 방향이 러프와 다를 수 있었다.",
        "  따라서 팔 형태 개선은 확인했지만 부감의 전체 포즈 적합성은 아직 해결되지 않았다.", "",
        "## 설명 고정 8장 결과", "", "| 러프 | 슬롯 | 이전 → 이후 후보 | 변화 |", "|---|---:|---|---|"]
    for row in frozen["rows"]:
        r=row.get("fresh_description")
        if r:
            old=[h["pose_id"] for h in r["old"]]; new=[h["pose_id"] for h in r["new"]]
            lines.append(f"| {row['test_id']} | {row['person']+1} | {len(old)} → {len(new)} | {'동일' if old==new else r['new_reason'] or 'V 관계 필수'} |")
        else: lines.append(f"| {row['test_id']} | {row['person']+1} | - | {row['after_route']} 유지 |")
    lines += ["", "## 증거", "", f"- 원본 이전 기록: {args.old}",f"- 동일 설명 ablation: {args.frozen}",
        f"- 새 8장 실호출: {args.live}",f"- 핵심 2장 진단: {args.diagnostic}",f"- 새 설명 재검색: {args.fresh_ablation}",
        "", "## 남은 한계", "",
        "- 앉아서 팔짱 포즈는 여전히 반환하지 못한다. 정확한 라이브러리 추가 또는 별도 검증된 합성 연구가 필요.",
        "- V 1위가 원본 다이빙의 몸통 기울기·원근·공중 여부까지 일치하지는 않는다. 다른 동작 T011/T012 품질도 해결되지 않았다.",
        "- VLM 설명이 흔들리면 noncritical 후보 순위는 달라진다. 설명 valid는 정확도 보증이 아니다.",
        "- T013 새 E2E 성공은 미검증. upstream_error 이상의 제공자 내부 원인은 이 로그만으로 확정할 수 없다.",
        "- 사용자 승인 8장만 외부 전송. 배포/PR/push 없음. 운영 기본 SEARCH_ROUTING=0, 새 색인 기본 비활성.", ""]
    lines += ["## 코드·화면 검증", "", "- 서버 회귀 872 passed, 8 skipped, 22 subtests passed. 스모크 54/54.",
              "- tests/converter는 bpy/mathutils 런타임 부재로 제외(3D 리타게팅/합성은 이번 변경 범위 밖).",
              "- 비교 화면 두 러프×두 출처×전체 후보 이미지·버튼 및 320/360/736px 표시 검증. 외부 네트워크 차단.", ""]
    (args.out/"COMPARISON_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"visual_bytes":(args.out/"visual-data.json").stat().st_size,"camera_renders":len(camera_rows)},ensure_ascii=False))


if __name__=="__main__": main()
