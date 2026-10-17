"""search-taxonomy-v2.json → src/semantic_vocab.py 스냅샷 재생성.

    python scripts/sync_semantic_vocab.py ../../../pose-library-viewer/dist/data/search-taxonomy-v2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ("posture", "action", "domain", "activity", "object", "intent", "emotion")
TARGET = Path(__file__).resolve().parents[1] / "src" / "semantic_vocab.py"


def render(taxonomy: dict) -> str:
    vocab = taxonomy["vocabulary"]
    lines = [
        '"""search-taxonomy-v2 어휘 스냅샷(pose-library-viewer/dist/data/search-taxonomy-v2.json, '
        f'version {taxonomy["version"]}).',
        "",
        "러프 설명(VLM)과 라이브러리 태깅이 같은 어휘를 써야 시맨틱 검색이 맞는다",
        "(tagging/ROUGH_TO_LANGUAGE.md). 원본이 바뀌면 이 파일을 다시 생성한다:",
        "    python scripts/sync_semantic_vocab.py <search-taxonomy-v2.json>",
        '"""',
        "",
        f'TAXONOMY_VERSION = "{taxonomy["version"]}"',
        "",
        "VOCABULARY = {",
    ]
    for group in ORDER:
        lines.append(f'    "{group}": {{')
        for key, label in vocab[group].items():
            lines.append(f"        {json.dumps(key)}: {json.dumps(label, ensure_ascii=False)},")
        lines.append("    },")
    lines += ["}", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    taxonomy = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    TARGET.write_text(render(taxonomy), encoding="utf-8")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
