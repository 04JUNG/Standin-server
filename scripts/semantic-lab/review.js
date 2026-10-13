const $ = id => document.getElementById(id);
const create = (tag, className, value) => {const n=document.createElement(tag);if(className)n.className=className;if(value!==undefined)n.textContent=value;return n;};
const labels={match:'맞음',partial:'부분 일치',wrong:'다름',deferred:'판단 보류'};
const views={front:'정면',side:'측면',three_quarter:'¾ 시점',back:'후면'};
let data=null, activeId=null, selected=0, comparison=false, timer=null, dirty=false, queue=Promise.resolve(), pending=0;
let draftKey='', drafts={};
function row(){return data?.rows.find(q=>q.id===activeId);}
function candidate(){return row()?.current[selected]||null;}
function isDone(r){return r && r.verdict!=='deferred';}
function currentVerdict(){const r=data.reviews[activeId];return r?.selected_pose_id===(candidate()?.pose_id||null)?r.verdict:'deferred';}
function visibleRows(){if(!data)return [];return data.rows.filter(q=>{
  const status=data.reviews[q.id], f=$('filter').value;
  return ($('category').value==='all'||q.category===$('category').value) &&
    (f==='all'||(f==='pending'&&!isDone(status))||(f==='done'&&isDone(status))||(f==='weak'&&q.ai_verdict!=='match'));
});}
async function api(url,options){const r=await fetch(url,options);const body=await r.json();if(!r.ok)throw Error(typeof body.detail==='string'?body.detail:'저장에 실패했습니다. 다시 시도해주세요.');return body;}
function remember(){try{localStorage.setItem(draftKey,JSON.stringify(drafts));}catch{}}
function progress(){const all=Object.values(data.reviews), done=all.filter(isDone).length, deferred=all.filter(r=>r.verdict==='deferred').length;
  $('progress').max=data.total;$('progress').value=done;$('progressText').textContent=`검수 ${done} / ${data.total}개 완료`;$('deferredCount').textContent=deferred?`보류·메모 ${deferred}개`:'';
}
function persist(snapshot){pending++;paintButtons();
  queue=queue.catch(()=>{}).then(async()=>{
    const out=await api('/review-feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(snapshot)});
    data.reviews[snapshot.query_id]=out.review;
    if(drafts[snapshot.query_id]===snapshot.note){delete drafts[snapshot.query_id];remember();}
    progress();
    if(activeId===snapshot.query_id){$('saveState').textContent=`로컬 저장 완료 · ${labels[snapshot.verdict]}`;$('message').textContent='';paintButtons();}
    return out;
  }).catch(error=>{if(activeId===snapshot.query_id){$('saveState').textContent='저장 실패 · 메모는 이 브라우저에 임시 보관 중';$('message').textContent=error.message;}throw error;}).finally(()=>{pending--;paintButtons();});
  return queue;
}
function snapshot(verdict){return {review_id:data.review_id,query_id:activeId,verdict,selected_pose_id:candidate()?.pose_id||null,note:$('note').value};}
async function flushDraft(){clearTimeout(timer);if(dirty){dirty=false;await persist(snapshot(currentVerdict()));}else if(pending){await queue;}}
function paintButtons(){if(!data)return;const saved=data.reviews[activeId];for(const b of document.querySelectorAll('[data-verdict]')){
  b.disabled=pending>0||(!candidate()&&['match','partial'].includes(b.dataset.verdict));
  b.classList.toggle('chosen',saved?.verdict===b.dataset.verdict&&saved.selected_pose_id===(candidate()?.pose_id||null));
}}
function zoom(c,view,url){$('zoomImage').src=url;$('zoomLabel').textContent=`${views[view]} · ${c.pose_id}`;$('zoom').showModal();}
function fitImages(){
  const bar=document.querySelector('.verdicts');
  if(comparison||getComputedStyle(bar).position!=='fixed')return;
  const height=document.documentElement.clientHeight,barHeight=bar.getBoundingClientRect().height;
  for(const img of $('stage').querySelectorAll('.pose-frame img')){
    const room=height-barHeight-img.getBoundingClientRect().top-12;
    img.style.height=Math.max(170,Math.min(380,room))+'px';
  }
}
function panel(c,title,view){if(!c){const blank=create('div','no-result');blank.append(create('strong','',title+' · 후보 없음'),create('p','','이 문장에 맞는 후보가 반환되지 않았습니다. 의견을 적고 다름 또는 보류로 기록할 수 있어요.'));return blank;}
  const p=create('article','pose-panel'), cap=create('div','panel-caption');cap.append(create('span','',title),create('span','','이미지를 눌러 확대'));p.append(cap);
  const chosen=view==='pair'?['front','side']:[view], images=create('div','images'+(chosen.length===1?' single':''));
  for(const v of chosen){const figure=create('figure','pose-frame'),url=c.thumbnails[v];if(url){const img=create('img');img.src=url;img.alt=`${title} ${views[v]} ${c.pose_id}`;img.tabIndex=0;img.setAttribute('role','button');img.onclick=()=>zoom(c,v,url);img.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();zoom(c,v,url);}};figure.append(img);}else figure.append(create('p','','이 시점의 이미지가 없습니다.'));figure.append(create('figcaption','',views[v]));images.append(figure);}p.append(images);
  const meta=create('div','pose-meta');meta.append(create('p','',c.description),create('p','pose-id',c.pose_id));const tags=create('div','tags');for(const t of Object.values(c.tags).flat().slice(0,12))tags.append(create('span','',t.label));meta.append(tags);p.append(meta);return p;
}
function renderPose(){const q=row(), c=candidate(), view=$('view').value;$('stage').replaceChildren();
  if(comparison){const both=create('div','comparison');both.append(panel(q.previous[0],'개선 전 · 1위',view),panel(c,`개선 후 · ${selected+1}위`,view));$('stage').append(both);}else $('stage').append(panel(c,`개선 후 · ${selected+1}위`,view));
  $('candidates').replaceChildren();q.current.forEach((c,n)=>{const b=create('button','candidate'+(selected===n?' active':''));b.setAttribute('aria-label',`${n+1}번 후보 보기`);b.setAttribute('aria-pressed',selected===n);const img=create('img');img.src=c.thumbnails.front||Object.values(c.thumbnails)[0]||'';img.alt='';const name=create('div','',`${n+1}번 후보`);if(selected===n)name.append(create('div','selected-text','지금 보는 포즈'));b.append(img,name);const queryId=activeId;b.onclick=async()=>{try{await flushDraft();if(activeId!==queryId)return;selected=n;window.scrollTo({top:0,behavior:'instant'});renderPose();paintButtons();}catch{}};$('candidates').append(b);});
  $('selectedLabel').textContent=c?`개선 후 ${selected+1}번 후보를 평가합니다. 다른 후보가 더 맞으면 먼저 골라주세요.`:'검색 결과가 없습니다. 찾지 못한 이유를 메모해주세요.';paintButtons();fitImages();
}
function render(){progress();const rows=visibleRows();$('jump').replaceChildren();for(const q of rows){const option=create('option','',`${isDone(data.reviews[q.id])?'✓ ':''}${q.id} · ${q.text}`);option.value=q.id;$('jump').append(option);}
  if(!rows.length){$('workspace').hidden=true;$('empty').hidden=false;$('question').textContent='';$('queryId').textContent='';$('queryCategory').textContent='';$('queryPosition').textContent='';$('prev').disabled=true;$('next').disabled=true;return;}
  if(!rows.some(q=>q.id===activeId))activeId=rows[0].id;
  const q=row(), saved=data.reviews[activeId];selected=Math.max(0,q.current.findIndex(c=>c.pose_id===saved?.selected_pose_id));
  $('workspace').hidden=false;$('empty').hidden=true;$('queryId').textContent=q.id;$('queryCategory').textContent=q.category;$('queryPosition').textContent=`${data.rows.indexOf(q)+1} / ${data.total}`;$('question').textContent='“'+q.text+'”';$('jump').value=activeId;
  const i=rows.findIndex(q=>q.id===activeId);$('prev').disabled=i<=0;$('next').disabled=i>=rows.length-1;
  $('note').value=drafts[activeId]??saved?.note??'';dirty=Object.hasOwn(drafts,activeId);$('noteBlock').open=!!$('note').value;
  $('saveState').textContent=dirty?'이 브라우저에 임시 메모가 있어요':saved?`저장된 평가 · ${labels[saved.verdict]}`:'아직 평가하지 않았어요';
  $('aiJudgment').textContent='AI의 1위 판정: '+(labels[q.ai_verdict]||'결과 없음');$('aiNote').textContent=q.ai_note;
  location.replace('#'+activeId);renderPose();
}
async function navigate(id){try{await flushDraft();activeId=id;$('message').textContent='';window.scrollTo({top:0,behavior:'instant'});render();}catch{}}
async function move(delta){const rows=visibleRows(),i=rows.findIndex(q=>q.id===activeId);if(rows[i+delta])await navigate(rows[i+delta].id);}
async function judge(verdict){if(pending||!row()||$('workspace').hidden)return;clearTimeout(timer);dirty=false;const id=activeId;try{await persist(snapshot(verdict));
    if($('autoNext').checked){const allIndex=data.rows.findIndex(q=>q.id===id), next=visibleRows().find(q=>data.rows.indexOf(q)>allIndex);if(next)await navigate(next.id);else{$('message').textContent='이 보기의 마지막 문장입니다. 다른 필터에서 이어서 검수할 수 있어요.';render();}}
    else paintButtons();
  }catch{dirty=true;}
}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('jump').onchange=()=>navigate($('jump').value);
for(const id of ['filter','category'])$(id).onchange=async()=>{try{await flushDraft();render();}catch{}};
$('nextPending').onclick=async()=>{const from=data.rows.findIndex(q=>q.id===activeId),ordered=[...data.rows.slice(from+1),...data.rows.slice(0,from+1)];const next=ordered.find(q=>!isDone(data.reviews[q.id]));if(next){$('filter').value='pending';$('category').value='all';await navigate(next.id);}else $('message').textContent='100개 모두 검수했습니다.';};
$('resetFilters').onclick=()=>{$('filter').value='all';$('category').value='all';render();};
$('view').onchange=renderPose;
function setMode(on){comparison=on;$('singleMode').classList.toggle('active',!on);$('compareMode').classList.toggle('active',on);$('singleMode').setAttribute('aria-pressed',!on);$('compareMode').setAttribute('aria-pressed',on);renderPose();}
$('singleMode').onclick=()=>setMode(false);$('compareMode').onclick=()=>setMode(true);
$('note').oninput=()=>{dirty=true;drafts[activeId]=$('note').value;remember();$('saveState').textContent='메모 저장 중…';clearTimeout(timer);timer=setTimeout(()=>{flushDraft().catch(()=>{});},700);};
for(const b of document.querySelectorAll('[data-verdict]'))b.onclick=()=>judge(b.dataset.verdict);
document.addEventListener('keydown',e=>{if(!data||$('workspace').hidden||$('zoom').open||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||e.metaKey||e.ctrlKey||e.altKey)return;
  if(e.key==='ArrowLeft'){e.preventDefault();move(-1);}if(e.key==='ArrowRight'){e.preventDefault();move(1);}const v={'1':'match','2':'partial','3':'wrong'}[e.key];if(v){e.preventDefault();judge(v);}
});
$('closeZoom').onclick=()=>$('zoom').close();$('zoom').onclick=e=>{if(e.target===$('zoom'))$('zoom').close();};
$('zoom').addEventListener('close',()=>{window.scrollTo({top:0,behavior:'instant'});fitImages();});
$('editNote').onclick=()=>{$('noteBlock').open=true;$('noteBlock').scrollIntoView({block:'center',behavior:'smooth'});$('note').focus({preventScroll:true});};
window.addEventListener('resize',()=>{if(data&&!$('workspace').hidden)renderPose();});
$('export').onclick=async()=>{try{await flushDraft();const out={review_id:data.review_id,total:data.total,completed:Object.values(data.reviews).filter(isDone).length,rows:data.rows.map(q=>({query_id:q.id,query:q.text,review:data.reviews[q.id]||null}))};const url=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));const a=create('a');a.href=url;a.download='standin-human-review-100.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch{}};
api('/review-data').then(out=>{data=out;draftKey='standin-review-drafts-'+out.review_id;try{drafts=JSON.parse(localStorage.getItem(draftKey)||'{}');}catch{};
  for(const category of new Set(data.rows.map(q=>q.category))){const option=create('option','',category);option.value=category;$('category').append(option);}
  const requested=location.hash.slice(1);activeId=data.rows.some(q=>q.id===requested)?requested:(data.rows.find(q=>!isDone(data.reviews[q.id]))||data.rows[0]).id;
  $('export').disabled=false;window.scrollTo({top:0,behavior:'instant'});render();
}).catch(error=>{$('message').textContent=error.message;$('question').textContent='검수 데이터를 불러오지 못했습니다';});
