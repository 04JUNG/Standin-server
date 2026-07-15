"""
VLM 실연결 테스트(5순위): 러프 컷 → Gemini/OpenAI로 개수·shot·action·view·relationship 태그.
좌표는 안 만든다(설계 불변식 #1). approx_boxes는 대략 박스만.

실행:
    $env:VLM_PROVIDER="gemini"        # .env보다 우선(load_dotenv override=False)
    py -3.12 scripts/vlm_test.py <이미지 또는 폴더>

출력: 컷별 [num/shot/action/view/rel] 한 줄 + 원본 JSON. provider가 mock이면 그대로 표시(폴백 감지용).
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.config import CFG
from src.vlm.client import build_vlm_client


def iter_images(path):
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "*.png")) +
                        glob.glob(os.path.join(path, "*.jpg"))):
            yield f
    else:
        yield path


def main():
    if len(sys.argv) < 2:
        print("usage: py -3.12 scripts/vlm_test.py <img|dir>"); return
    target = sys.argv[1]

    client = build_vlm_client()
    print(f"provider={CFG.vlm_provider}  model={CFG.gemini_model}  client={type(client).__name__}")
    if type(client).__name__ == "MockVLMClient" and CFG.vlm_provider.lower() != "mock":
        print("  ⚠ mock으로 폴백됨(키/패키지 확인). 아래 결과는 규칙기반 스텁임.")
    print("-" * 88)

    for f in iter_images(target):
        name = os.path.basename(f)
        try:
            img = Image.open(f)                    # PIL은 한글 경로 OK
            w, h = img.size
            a = client.analyze(img, w, h)
            print(f"[{name[:40]:40s}] n={a.num_people} shot={a.shot.value:9s} "
                  f"action={a.action.value:9s} view={a.view.value:13s} rel={a.relationship.value}")
            if a.dialogue:
                print(f"    dialogue: {a.dialogue!r}")
        except Exception as e:
            print(f"[{name[:40]:40s}] ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
