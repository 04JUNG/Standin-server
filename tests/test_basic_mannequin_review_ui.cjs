// Separate headless context: no interaction with the user's saved review.
const assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.argv[3]);
let browser;
(async()=>{
  browser=await chromium.launch({headless:true,executablePath:process.argv[4]});
  const context=await browser.newContext({viewport:{width:1200,height:1000}});
  const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(process.argv[2]).href);
  const frame=await page.locator('iframe').count()?page.frameLocator('iframe'):page;
  const root=frame.locator('#basic-mannequin-review');
  await root.locator('[data-query] option').last().waitFor({state:'attached'});
  const expected=process.argv[5]==='inline'?4:10;
  assert.equal(await root.locator('[data-query] option').count(),expected);
  assert.equal(await root.locator('[data-query]').inputValue(),'rough_288');
  assert.equal(await root.locator('[data-view]').inputValue(),'front');
  const ids=await root.locator('[data-query] option').evaluateAll(items=>items.map(x=>x.value));
  let checked=0;
  for(const id of ids){
    await root.locator('[data-query]').selectOption(id);
    await root.locator('[data-input]').evaluate(im=>im.decode());
    for(const view of ['front','three_quarter','side','back']){
      await root.locator('[data-view]').selectOption(view);
      assert.match(await root.locator('[data-view-label]').textContent(),/검색 맞춤 시점 아님/);
      for(const arm of ['before','after']){
        const n=await root.locator(`[data-ranks="${arm}"] button`).count();
        for(let i=0;i<n;i++){
          await root.locator(`[data-ranks="${arm}"] button`).nth(i).click();
          const im=root.locator(`[data-image="${arm}"]`);
          assert.match(await im.getAttribute('src'),/^data:image\/jpeg;base64,/);
          await im.evaluate(x=>x.decode());
          assert.match(await root.locator(`[data-label="${arm}"]`).textContent(),/기본 인형/);checked++;
        }
      }
    }
  }
  await root.locator('[data-query]').selectOption('rough_288');
  await root.locator('[data-view]').selectOption('front');
  if(expected===10){
    const old=await root.locator('[data-input]').getAttribute('src');
    await root.locator('[data-overlay]').check();
    assert.notEqual(await root.locator('[data-input]').getAttribute('src'),old);
    await root.locator('[data-overlay]').uncheck();
  }else assert.equal(await root.locator('[data-overlay]').isVisible(),false);
  await root.locator('[data-useful="before"]').selectOption('no');
  await root.locator('[data-useful="after"]').selectOption('no');
  await root.locator('[data-preference]').selectOption('neither');
  await root.locator('[data-note]').fill('TEST ONLY · isolated browser');
  await page.reload();
  assert.equal(await root.locator('[data-note]').inputValue(),'TEST ONLY · isolated browser');
  assert.equal(await root.locator('[data-preference]').inputValue(),'neither');
  await root.locator('[data-note]').fill('');
  for(const id of ['rough_288','rough_315','rough_301','rough_291']){
    await root.locator('[data-query]').selectOption(id);
    for(const arm of ['before','after'])await root.locator(`[data-ranks="${arm}"] button`).first().click();
    await root.locator('[data-view]').selectOption('front');
    await page.screenshot({path:`/private/tmp/basic-mannequin-${expected}-${id}.png`,fullPage:true});
  }
  await page.setViewportSize({width:360,height:1000});
  const dimensions=await root.evaluate(el=>({width:el.clientWidth,scroll:el.scrollWidth}));
  assert(dimensions.scroll<=dimensions.width+1,JSON.stringify(dimensions));
  assert.equal(errors.length,0,errors.join('\n'));
  console.log('PASS',expected,'queries;',checked,'candidate/view images; fixed-view labels, ratings persistence, mobile width');
  await context.close();await browser.close();
})().catch(async e=>{console.error(e);if(browser)await browser.close();process.exit(1)});
