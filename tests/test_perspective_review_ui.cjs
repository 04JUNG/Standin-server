// Headless software QA uses its own isolated browser context and no human ratings.
const assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.argv[3]||'playwright');
let activeBrowser;
(async()=>{
  const browser=await chromium.launch({headless:true,...(process.argv[4]?{executablePath:process.argv[4]}:{})});
  activeBrowser=browser;
  const context=await browser.newContext({viewport:{width:1200,height:1000}});
  const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(process.argv[2]).href);
  const frame=await page.locator('iframe').count()?page.frameLocator('iframe'):page;
  const root=frame.locator('#rough-perspective-review');
  await root.locator('[data-query] option').last().waitFor({state:'attached'});
  assert.equal(await root.locator('[data-query] option').count(),10);
  assert.equal(await root.locator('[data-query]').inputValue(),'rough_298');
  await root.locator('[data-input]').evaluate(im=>im.decode());
  for(const arm of ['before','after']){
    assert.equal(await root.locator(`[data-ranks="${arm}"] button`).count(),5);
    await root.locator(`[data-image="${arm}"]`).evaluate(im=>im.decode());
  }
  const first=await root.locator('[data-label="after"]').textContent();
  await root.locator('[data-ranks="after"] button').nth(1).click();
  assert.notEqual(await root.locator('[data-label="after"]').textContent(),first);
  const matched=await root.locator('[data-image="after"]').getAttribute('src');
  await root.locator('[data-view]').selectOption('canonical');
  assert.notEqual(await root.locator('[data-image="after"]').getAttribute('src'),matched);
  const raw=await root.locator('[data-input]').getAttribute('src');
  await root.locator('[data-overlay]').check();
  assert.notEqual(await root.locator('[data-input]').getAttribute('src'),raw);
  await root.locator('[data-useful="before"]').selectOption('no');
  await root.locator('[data-useful="after"]').selectOption('yes');
  await root.locator('[data-preference]').selectOption('after');
  await root.locator('[data-note]').fill('TEST ONLY · isolated context');
  assert.match(await root.locator('[data-progress]').textContent(),/1장 검수 완료/);
  await root.locator('[data-query]').selectOption('rough_178');
  assert.match(await root.locator('[data-status]').textContent(),/검색 실패/);
  assert.equal(await root.locator('[data-image="before"]').isVisible(),false);
  assert.equal(await root.locator('[data-rating-controls]').isVisible(),false);
  await root.locator('[data-query]').selectOption('rough_298');
  assert.equal(await root.locator('[data-useful="after"]').inputValue(),'yes');
  assert.equal(await root.locator('[data-note]').inputValue(),'TEST ONLY · isolated context');
  // Reload verifies state storage used by the exported review.
  await page.reload(); await root.locator('[data-query]').waitFor();
  assert.equal(await root.locator('[data-useful="after"]').inputValue(),'yes');
  assert.equal(errors.length,0,errors.join('\n'));
  await page.screenshot({path:'/private/tmp/standin-perspective-review-desktop.png',fullPage:true});
  await page.setViewportSize({width:360,height:1000});
  await root.locator('[data-query]').selectOption('rough_315');
  const width=await root.evaluate(el=>({scroll:el.scrollWidth,width:el.clientWidth}));
  assert(width.scroll<=width.width+1,JSON.stringify(width));
  await page.screenshot({path:'/private/tmp/standin-perspective-review-mobile.png',fullPage:true});
  console.log('PASS 10 cases, image loading, rank/view/overlay, ratings persistence, failure display, mobile width');
  await context.close();await browser.close();
})().catch(async e=>{console.error(e);if(activeBrowser)await activeBrowser.close();process.exit(1)});
