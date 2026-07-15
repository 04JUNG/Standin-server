"""
VLM 태그 추출·측정(5순위): 러프 컷 → VLM → shot/action/view/relationship/count/대략박스.

목적: "VLM이 우리 판단 태스크를 실제로 잘 하는가"를 측정.
  - shot 3갈래(full_half/bust/face) 정확도
  - action/view/relationship 태그가 검색 필터로 쓸 만한지
  - 사람 수(검출 보정 신호)

실행:
    # provider 선택: mock(오프라인 기본) | gemini | openai
    VLM_PROVIDER=gemini GEMINI_API_KEY=... py -3.12 scripts/vlm_tag.py <이미지 또는 폴더>
    py -3.12 scripts/vlm_tag.py cut.png                 # mock(플러밍 확인)
옵션:
    --provider gemini    (env 대신 인자로)
    --csv out.csv        결과를 CSV로 저장(정답과 비교·정확도 산출용)
"""
import sys, os, glob, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path):
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return img, img.width, img.height
    except Exception:
        class H(str):
            @property
            def hint(self): return str(self)
        return H(path), 512, 768   # mock 폴백(파일명 힌트)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="이미지 파일 또는 폴더")
    ap.add_argument("--provider")
    ap.add_argument("--csv")
    a = ap.parse_args()
    if a.provider:
        os.environ["VLM_PROVIDER"] = a.provider

    from src.vlm.client import build_vlm_client
    client = build_vlm_client()
    from src.config import CFG
    print(f"[vlm] provider={CFG.vlm_provider} model={CFG.gemini_model}\n")

    if os.path.isdir(a.path):
        files = sorted(glob.glob(os.path.join(a.path, "*.png")) +
                       glob.glob(os.path.join(a.path, "*.jpg")))
    else:
        files = [a.path]

    rows = []
    hdr = f"{'file':28s} {'#':>2s} {'shot':10s} {'action':9s} {'view':13s} {'relation':13s}"
    print(hdr); print("-"*len(hdr))
    for f in files:
        img, w, h = _load(f)
        try:
            r = client.analyze(img, w, h)
            name = os.path.basename(f)[:28]
            print(f"{name:28s} {r.num_people:>2d} {r.shot.value:10s} {r.action.value:9s} "
                  f"{r.view.value:13s} {r.relationship.value:13s}"
                  + (f"  dlg={r.dialogue}" if r.dialogue else ""))
            rows.append({"file": os.path.basename(f), "num_people": r.num_people,
                         "shot": r.shot.value, "action": r.action.value,
                         "view": r.view.value, "relationship": r.relationship.value})
        except Exception as e:
            print(f"{os.path.basename(f)[:28]:28s} ERR {e}")

    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as fp:
            w_ = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
        print(f"\nsaved {a.csv} ({len(rows)} rows) — 정답 컬럼 추가해 정확도 계산")


if __name__ == "__main__":
    main()
