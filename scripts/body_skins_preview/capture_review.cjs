const { chromium } = require("playwright");
const fs = require("fs"),
  path = require("path");
const output = process.argv[3];
if (!output) throw Error("Usage: node capture_review.cjs URL OUTPUT_DIRECTORY");
(async () => {
  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1060 },
    deviceScaleFactor: 1,
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error" && !m.text().includes("404"))
      errors.push(m.text());
  });
  await page.goto(process.argv[2]);
  await page.waitForFunction(() => window.skinPreview?.ready, {
    timeout: 60000,
  });
  const out = path.join(output, "renders");
  fs.mkdirSync(out, { recursive: true });
  await page.screenshot({ path: path.join(output, "preview-screen.png") });
  await page.addStyleTag({
    content:
      "body.capture header,body.capture aside,body.capture .stage-top,body.capture .pose-bar,body.capture .view-tools,body.capture .canvas-foot,body.capture #compare-labels{display:none!important}body.capture main{display:block;max-width:none}body.capture .studio{padding:0}body.capture .viewport{width:100vw;height:100vh;min-height:0;border:0;border-radius:0}",
  });
  await page.evaluate(() => document.body.classList.add("capture"));
  async function save(name) {
    await page.waitForTimeout(80);
    const data = await page.evaluate(() => skinPreview.capture());
    fs.writeFileSync(
      path.join(out, name + ".png"),
      Buffer.from(data.split(",")[1], "base64"),
    );
  }
  for (const model of ["male", "female"]) {
    await page.evaluate((m) => skinPreview.setModel(m), model);
    await page.evaluate(() => skinPreview.setAngle("front"));
    await page.setViewportSize({ width: 2400, height: 1200 });
    await page.evaluate(() => skinPreview.setCompare(true));
    await save(`${model}-comparison`);
    await page.evaluate(() => skinPreview.setCompare(false));
    await page.setViewportSize({ width: 900, height: 1100 });
    for (const skin of [
      "default",
      "gray-guide",
      "flesh",
      "flesh-guide",
      "muscle",
    ]) {
      await page.evaluate((s) => skinPreview.setSkin(s), skin);
      for (const angle of ["front", "quarter", "side", "back"]) {
        await page.evaluate((v) => skinPreview.setAngle(v), angle);
        await save(`${model}-${skin}-${angle}`);
      }
    }
    for (const pose of ["bent", "seated", "broader"]) {
      await page.evaluate((p) => skinPreview.setPose(p), pose);
      await page.evaluate(() => skinPreview.setAngle("quarter"));
      for (const skin of ["gray-guide", "muscle"]) {
        await page.evaluate((s) => skinPreview.setSkin(s), skin);
        await save(`${model}-${pose}-${skin}`);
      }
    }
    await page.evaluate(() => skinPreview.setPose("standing"));
    await page.evaluate(() => skinPreview.setAngle("quarter"));
    await page.evaluate(() => skinPreview.setDetail(true));
    for (const skin of ["gray-guide", "muscle"]) {
      await page.evaluate((s) => skinPreview.setSkin(s), skin);
      await save(`${model}-${skin}-detail`);
    }
    await page.evaluate(() => skinPreview.setDetail(false));
  }
  const comparisonLayouts = [];
  await page.evaluate(() => skinPreview.setCompare(true));
  for (const angle of ["front", "quarter", "side", "back"]) {
    await page.evaluate((v) => skinPreview.setAngle(v), angle);
    const centers = await page.evaluate(() => skinPreview.comparisonCenters());
    if (centers.some((v, i) => i && v[0] - centers[i - 1][0] < 0.1))
      throw Error("Comparison models overlap at " + angle);
    comparisonLayouts.push({ angle, projectedCenters: centers });
  }
  await page.evaluate(() => skinPreview.setCompare(false));
  // Mouse orbit/zoom, skin swap and material controls must retain the inspection camera.
  await page.evaluate(() => skinPreview.setModel("male"));
  await page.evaluate(() => skinPreview.setSkin("gray-guide"));
  await page.mouse.move(420, 450);
  await page.mouse.down();
  await page.mouse.move(560, 510, { steps: 8 });
  await page.mouse.up();
  await page.mouse.wheel(0, -130);
  await page.waitForTimeout(150);
  const before = await page.evaluate(() => skinPreview.cameraState());
  await page.evaluate(() => skinPreview.setSkin("flesh-guide"));
  const after = await page.evaluate(() => skinPreview.cameraState());
  if (JSON.stringify(before) !== JSON.stringify(after))
    throw Error("Skin swap moved the inspection camera");
  await page.evaluate(() => document.body.classList.remove("capture"));
  await page.setViewportSize({ width: 1600, height: 1060 });
  await page.evaluate(() => skinPreview.setAngle("front"));
  await page.screenshot({ path: path.join(output, "final-screen.png") });
  fs.writeFileSync(
    path.join(output, "browser-qa.json"),
    JSON.stringify(
      {
        errors,
        comparisonLayouts,
        loads: await page.evaluate(() => skinPreview.diagnostics.loads),
        cameraPreservedOnSkinSwap:
          JSON.stringify(before) === JSON.stringify(after),
        cameraBefore: before,
        cameraAfter: after,
        renderedImages: fs.readdirSync(out).length,
      },
      null,
      2,
    ),
  );
  await browser.close();
  if (errors.length) throw Error(errors.join("\n"));
  console.log("Browser QA passed; all render images saved");
})();
