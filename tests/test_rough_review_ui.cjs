// In-memory DOM contract test. This is NOT a browser/layout/screenshot test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=JSON.parse(html.match(/<script id="study-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const code=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(code);
class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.handlers={};this.dataset={};this.value='';this.textContent='';}
  append(...els){this.children.push(...els);}
  replaceChildren(...els){this.children=[...els];}
  addEventListener(name,fn){this.handlers[name]=fn;}
  setAttribute(k,v){this[k]=v;}
  focus(){this.focused=true;}
  click(){this.handlers.click?.();}
  showModal(){this.open=true;}
  close(){this.open=false;}
}
const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
elements['study-data'].textContent=JSON.stringify(data);
const memory=new Map();let exported=null;
const document={getElementById:id=>{assert(elements[id],'missing element '+id);return elements[id];},createElement:tag=>new Element(tag)};
vm.runInNewContext(code,{document,localStorage:{getItem:k=>memory.get(k)||null,setItem:(k,v)=>memory.set(k,v)},
  Blob:class{constructor(parts){exported=JSON.parse(parts.join(''));}},
  URL:{createObjectURL:()=> 'memory:test-only',revokeObjectURL:()=>{}},setTimeout:fn=>fn(),Date,JSON,Set,Object,Number,Math});
const all=el=>[el,...el.children.flatMap(all)];
const selects=()=>all(elements.methods).filter(e=>e.dataset.candidate);
assert.equal(selects().length,15);assert(elements.progress.textContent.includes('0 / 120'));
const select=selects()[0];select.value='yes';select.handlers.change();
assert(elements.progress.textContent.includes('1 / 120'));
select.value='';select.handlers.change();assert(elements.progress.textContent.includes('0 / 120'));
elements.export.click();assert.equal(exported,null);assert(elements.rater.focused);
elements.rater.value='unit-test-memory-only';elements.rater.handlers.input();elements.export.click();
assert.equal(exported.complete,false);assert.equal(Object.keys(exported.ratings).length,0);
elements.query.value=data.queries.findIndex(q=>q.failed);elements.query.handlers.change();
assert.equal(selects().length,0);assert(elements.methods.children[0].textContent.includes('세 방법 모두'));
elements.query.value=data.queries.findIndex(q=>!q.failed);elements.query.handlers.change();
const zoom=all(elements.methods).find(e=>e.className==='enlarge');zoom.click();assert(elements.zoom.open);
elements['close-zoom'].click();assert(!elements.zoom.open);
assert(!html.includes('anchor_4'));assert(!html.includes('pooled_117'));assert(!html.includes('pose_id'));
console.log('PASS JS syntax, empty defaults, 15 controls/query, local state, partial export, failure page, modal handlers, blinded metadata');
console.log('LIMIT: in-memory DOM only; browser rendering and actual download behavior not verified');
