// 손 프리셋 칩용 마네킹 손 썸네일을 만든다(04J-290). 개발자 도구 — 라이브러리를 바꾼 뒤 한 번 돌린다.
//
//   python scripts/hand_library_preview.py out/hand-library.html
//   node scripts/render_hand_thumbnails.mjs out/hand-library.html converter/hand_thumbs
//
// 미리보기 페이지와 같은 렌더러(관통 검사와 같은 두께)로 프리셋마다 왼손·오른손 PNG(투명 배경)를
// 그리고, 어떤 라이브러리에서 만든 그림인지 manifest.json에 남긴다. 서버는 manifest의
// library_sha256이 현재 라이브러리와 다르면 썸네일을 내주지 않는다(옛 그림 방지).
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";

const [, , pageArg = "out/hand-library.html", outArg = "converter/hand_thumbs", sizeArg = "160"] = process.argv;
const size = Number(sizeArg);
let chromium;
try { ({ chromium } = await import("playwright")); }
catch { ({ chromium } = (await import("/opt/node22/lib/node_modules/playwright/index.js")).default); }

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 2, colorScheme: "light" });
await page.goto(pathToFileURL(resolve(pageArg)).href);
await page.waitForFunction(() => typeof window.__standinRenderHand === "function");
const meta = await page.evaluate(() => {
  const data = JSON.parse(document.getElementById("hand-data").textContent);
  return { version: data.library_version, sha: data.library_sha256, ids: Object.keys(data.presets) };
});
const out = resolve(outArg);
mkdirSync(out, { recursive: true });
const files = {};
for (const id of meta.ids) {
  for (const side of ["left", "right"]) {
    const url = await page.evaluate(([i, s, n]) => window.__standinRenderHand(i, s, n), [id, side, size]);
    const bytes = Buffer.from(url.split(",")[1], "base64");
    const name = `${id}__${side}.png`;
    writeFileSync(join(out, name), bytes);
    files[name] = createHash("sha256").update(bytes).digest("hex");
  }
}
writeFileSync(join(out, "manifest.json"), JSON.stringify({
  library_version: meta.version, library_sha256: meta.sha, size_css_px: size, scale: 2,
  view: "palm-oblique(yaw 35, pitch 25)", files,
}, null, 1) + "\n");
console.log(`${Object.keys(files).length} thumbnails -> ${out}`);
await browser.close();
