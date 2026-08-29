"""Gemini 상류 상태 프로브 — 키 교체·모델 변경이 실제로 먹혔는지 실측한다.

우리 재시도 래퍼(GeminiVLMClient)를 **거치지 않고** SDK를 직접 때린다.
재시도가 섞이면 "상류가 실제로 몇 %나 503을 내는가"를 못 본다.

실행:
    pip install google-genai pillow
    $env:GEMINI_API_KEY="..."                       # 또는 .env
    py -3.12 scripts/vlm_probe.py                   # 순차 10회
    py -3.12 scripts/vlm_probe.py --burst 15        # 동시 15회(무료 티어 RPM 초과 → 429 유발)
    py -3.12 scripts/vlm_probe.py --model gemini-flash-latest,gemini-2.5-flash

읽는 법:
    429 가 나오면      → 여전히 무료 티어 쿼터(빌링 미연결 또는 옛 키가 배포에 남음)
    429 0건 / burst 15 → 유료 티어로 붙었다(무료 RPM 상한을 넘겼는데 안 막힘)
    503 은 유료여도 난다 → 구글 쪽 모델 과부하. 비율을 보고 폴백 모델을 결정한다.
"""
from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG                                    # noqa: E402  (.env 로드 포함)
from src.vlm import prompts                                   # noqa: E402
from src.vlm.client import _http_status, _is_timeout          # noqa: E402


def _load_image_bytes(path: str | None) -> bytes:
    """실 러프 컷이 있으면 그걸 쓰고, 없으면 최소 스틱피겨를 합성한다.

    503/429는 내용과 무관하므로 합성으로도 가용성 측정은 성립한다.
    토큰 수만 실제 컷보다 조금 작다.
    """
    from PIL import Image, ImageDraw
    if path:
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (768, 1024), "white")
        d = ImageDraw.Draw(img)
        d.ellipse((360, 200, 420, 260), outline="black", width=4)   # 머리
        d.line((390, 260, 390, 560), fill="black", width=4)          # 몸통
        d.line((390, 320, 300, 430), fill="black", width=4)          # 팔
        d.line((390, 320, 480, 430), fill="black", width=4)
        d.line((390, 560, 330, 760), fill="black", width=4)          # 다리
        d.line((390, 560, 450, 760), fill="black", width=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _one_call(client, model: str, png: bytes, timeout_ms: int) -> dict:
    from google.genai import types
    started = time.monotonic()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[prompts.USER_TEMPLATE,
                      types.Part.from_bytes(data=png, mime_type="image/png")],
            config=types.GenerateContentConfig(
                system_instruction=prompts.SYSTEM,
                response_mime_type="application/json",
                temperature=0,
                http_options=types.HttpOptions(timeout=timeout_ms),
            ),
        )
        usage = getattr(resp, "usage_metadata", None)
        return {
            "outcome": "ok",
            "elapsed": time.monotonic() - started,
            "thoughtTokens": getattr(usage, "thoughts_token_count", None),
            "totalTokens": getattr(usage, "total_token_count", None),
        }
    except Exception as error:                      # noqa: BLE001 — 분류가 목적이다
        status = _http_status(error)
        outcome = (
            "timeout" if _is_timeout(error)
            else str(status) if status is not None
            else type(error).__name__
        )
        return {"outcome": outcome, "elapsed": time.monotonic() - started,
                "detail": str(error)[:160]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="순차 호출 수")
    ap.add_argument("--burst", type=int, default=0,
                    help="동시 호출 수(무료 티어 RPM 초과 판별용). 주면 --n 대신 이걸 쓴다")
    ap.add_argument("--model", default=None, help="쉼표로 여러 개 주면 모델별 비교")
    ap.add_argument("--image", default=None, help="실 러프 컷 경로(없으면 합성)")
    ap.add_argument("--timeout-ms", type=int, default=CFG.gemini_request_timeout_ms)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("GEMINI_API_KEY가 없다. .env 또는 환경변수로 넣고 다시 실행.")
        return 2

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key,
                          http_options=types.HttpOptions(timeout=args.timeout_ms))
    png = _load_image_bytes(args.image)
    models = [m.strip() for m in (args.model or CFG.gemini_model).split(",") if m.strip()]

    # 키 자체는 절대 찍지 않는다. 배포에 들어간 키와 같은 키인지 대조할 지문만 남긴다.
    print(f"key=...{key[-4:]} (len={len(key)})  timeout={args.timeout_ms}ms  "
          f"image={'합성' if not args.image else args.image}  bytes={len(png)}")

    exit_code = 0
    for model in models:
        mode = f"burst x{args.burst}" if args.burst else f"순차 x{args.n}"
        print(f"\n=== {model}  ({mode}) ===")
        if args.burst:
            with ThreadPoolExecutor(max_workers=args.burst) as pool:
                results = list(pool.map(
                    lambda _: _one_call(client, model, png, args.timeout_ms),
                    range(args.burst)))
        else:
            results = [_one_call(client, model, png, args.timeout_ms)
                       for _ in range(args.n)]

        counts: dict[str, int] = {}
        for r in results:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
            mark = "ok " if r["outcome"] == "ok" else "FAIL"
            extra = ""
            if r["outcome"] == "ok" and r.get("thoughtTokens") is not None:
                extra = f"  thought={r['thoughtTokens']} total={r['totalTokens']}"
            elif r.get("detail"):
                extra = f"  {r['detail']}"
            print(f"  {mark} {r['outcome']:<12s} {r['elapsed']:6.1f}s{extra}")

        elapsed = sorted(r["elapsed"] for r in results)
        ok = counts.get("ok", 0)
        print(f"  --- 성공 {ok}/{len(results)}  "
              f"p50={statistics.median(elapsed):.1f}s max={elapsed[-1]:.1f}s")
        print(f"  --- 분포 {counts}")
        if counts.get("429"):
            print("  ⚠ 429 발생 → 이 키는 아직 무료 티어 쿼터에 걸린다(빌링 연결 확인).")
            exit_code = 1
        if counts.get("503"):
            print("  ⚠ 503 발생 → 구글 쪽 모델 과부하. 유료여도 난다. 폴백 모델이 답이다.")
        if ok == 0:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
