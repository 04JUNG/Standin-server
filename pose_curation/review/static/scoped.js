import {request} from './api.js?v=qa-20261002';
const $ = id => document.getElementById(id);
let items = [], query, result, sequence = 0, controller;
const limbs = [[5,6],[5,7],[7,9],[6,8],[8,10]];
function el(tag, text, cls) { const e=document.createElement(tag); e.textContent=text; if(cls)e.className=cls; return e; }
function overlay(candidate) {
  const svg=$('query-overlay'); svg.replaceChildren();
  if (!query) return;
  svg.setAttribute('viewBox', `0 0 ${query.size[0]} ${query.size[1]}`);
  function draw(points, color, valid) {
    const width=Math.max(...query.size)/250;
    for(const [a,b] of limbs) {
      if(!valid(a)||!valid(b))continue;
      const line=document.createElementNS('http://www.w3.org/2000/svg','line');
      for(const [key,val] of Object.entries({x1:points[a][0],y1:points[a][1],x2:points[b][0],y2:points[b][1],stroke:color,'stroke-width':width}))line.setAttribute(key,String(val));
      svg.append(line);
    }
    for(let j=5;j<=10;j++)if(valid(j)) {
      const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');
      for(const [key,val] of Object.entries({cx:points[j][0],cy:points[j][1],r:width*1.3,fill:color}))circle.setAttribute(key,String(val));
      svg.append(circle);
    }
  }
  draw(query.keypoints,'#24845f',j=>query.observed_joints.includes(j));
  if(candidate){const points=Array.from({length:17},()=>[0,0]); candidate.projected.forEach((p,i)=>points[i+5]=p);draw(points,'#e88738',j=>result.observed_joints.includes(j));}
}
function showCandidate(candidate, index) {
  overlay(candidate);
  [...$('match-results').children].forEach((card,i)=>card.classList.toggle('selected',i===index));
}
function showResults(value) {
  result=value;
  $('match-results').replaceChildren(...value.candidates.map((candidate,index)=>{
    const card=el('article','','match-card');
    const angles=candidate.orientation;
    card.append(el('h3',`${index+1}. ${candidate.pose_id}`),el('p',`${candidate.group==='new'?'신규 · 현재 검수 통과':'기존 라이브러리'} · 투영 오차 ${candidate.error.toFixed(3)}`),el('p',`좌우 ${angles.yaw}° / 높낮이 ${angles.pitch}° / 기울기 ${angles.roll}°`));
    const actions=el('div','','match-actions'), button=el('button','러프와 관절 비교','quiet-button');
    button.addEventListener('click',()=>showCandidate(candidate,index));
    const params=new URLSearchParams({pose:candidate.key,scope:'half',yaw:angles.yaw,pitch:angles.pitch,roll:angles.roll,source_hash:candidate.content_hash});
    const link=el('a','이 각도로 미리보기·내보내기 ↗','quiet-button');link.href=`/?${params}`;link.target='_blank';link.rel='noopener';
    actions.append(button,link);card.append(actions);return card;
  }));
  showCandidate(value.candidates[0],0);
}
async function selectQuery() {
  const seq=++sequence; controller?.abort(); controller=new AbortController();
  $('fit-rough').disabled=true; $('match-results').replaceChildren();query=null;overlay();
  $('query-image').removeAttribute('src');
  const key=$('rough-select').value;
  if(!key){$('match-status').textContent='이 범위에는 팔 관측 조건을 충족하는 러프가 없습니다. 전체 러프의 상체 비교를 선택하거나 포즈 상세에서 수동 각도를 사용하세요.';return;}
  try{
    const value=await request(`/api/scoped/queries/${encodeURIComponent(key)}`,{signal:controller.signal});
    if(seq!==sequence)return;query=value;
    $('query-image').src=`/api/scoped/queries/${encodeURIComponent(key)}/image`;overlay();
    $('fit-rough').disabled=false;$('match-status').textContent=`${value.image_id} · 인물 ${value.person+1}의 초록색 관절을 확인한 뒤 검색하세요.`;
  }catch(error){if(seq===sequence)$('match-status').textContent=error.message;}
}
function filterQueries(){
  const mode=$('rough-filter').value;
  const filtered=items.filter(q=>mode==='all'||(mode==='upper'?q.upper_only:q.origin==='제공 러프'));
  $('rough-select').replaceChildren(...filtered.map(q=>new Option(`${q.origin} ${q.image_id} · 인물 ${q.person+1}`,q.key)));
  selectQuery();
}
$('rough-filter').addEventListener('change',filterQueries);$('rough-select').addEventListener('change',selectQuery);
$('fit-rough').addEventListener('click',async()=>{
  const seq=sequence;$('fit-rough').disabled=true;$('match-results').replaceChildren();overlay();
  $('match-status').textContent='상체 포즈와 시점을 비교 중입니다…';
  try{
    const value=await request('/api/scoped/match',{method:'POST',headers:{'Content-Type':'application/json','X-Pose-Review':'1'},body:JSON.stringify({query:query.key,scope:'half'}),signal:controller.signal});
    if(seq!==sequence)return;showResults(value);
    $('match-status').textContent=`${value.searched}개 포즈 비교 완료${value.fallback_to_all?' · 대표 포즈 오차가 커서 전체 포즈로 확장했습니다.':''} · 낮은 오차가 시각적 정답을 보장하지는 않습니다.`;
  }catch(error){if(seq===sequence)$('match-status').textContent=error.message;}
  finally{if(seq===sequence)$('fit-rough').disabled=false;}
});
try{
  const [library,queries]=await Promise.all([request('/api/scoped'),request('/api/scoped/queries')]);
  $('scope-summary').textContent=`검색 가능 ${library.source_count}개 → 반신 대표 ${library.scopes.half.count}개 · 흉상 대표 ${library.scopes.bust.count}개`;
  items=queries.items;
  if(!items.some(q=>q.upper_only))$('rough-filter').value='all';
  filterQueries();
}catch(error){$('match-status').textContent=error.message;$('scope-summary').textContent='대표 라이브러리를 확인할 수 없습니다.';}
