/* Browser/Node shared conservative arm-part text search. No model calls. */
(function(root){
const norm=s=>(s||'').normalize('NFKC').toLowerCase().replace(/[^0-9a-z가-힣ㄱ-ㆎ\s]/g,' ').replace(/\s+/g,' ').trim();
function parse(text){
 const normalized=norm(text),compact=normalized.replace(/ /g,'');
 let fold=/팔짱|folded?arms|armscrossed/.test(compact);
 const exclude_fold=fold&&/팔짱.{0,6}(?:말고|제외|아닌|안낀|않|풀고|풀기|안끼|끼지않)/.test(compact);if(exclude_fold)fold=false;
 let selection=/왼팔|왼손|leftarm|lefthand/.test(compact)?'left':(/오른팔|오른손|rightarm|righthand/.test(compact)?'right':null);
 if(fold||/양팔|양손|두손|두팔|both|팔포개|아래팔교차|엑스방어|x자막기/.test(compact))selection='both';
 let actions=[];
 if(/뒷짐|등뒤/.test(compact))actions=['hands_behind'];
 else if(/뒤통수|뒷목|뒷머리|머리뒤/.test(compact))actions=['head_back'];
 else if(/허리|골반/.test(compact)&&/손|짚/.test(compact))actions=/양손|두손/.test(compact)?['hands_on_hips']:/다른손|설명|내밀/.test(compact)?['hip_offer']:['hands_on_hips','hip_offer'];
 else if(/턱|입근처/.test(compact))actions=/(?:양손|두손)(?:으로)?턱/.test(compact)?['chin_support_pair']:['chin_support'];
 else if(/폰|휴대폰|물건들/.test(compact))actions=['phone_hold','phone_pair'];
 else if(/비대칭|앞뒤|한팔앞/.test(compact))actions=['asymmetric_reach'];
 else if(/위로|높이|높은곳/.test(compact)&&/뻗|들/.test(compact))actions=['reach_up','raise_hand'];
 else if(/앞으로|전방/.test(compact)&&/뻗|내밀/.test(compact))actions=['reach_forward','offer_hand'];
 return {text,normalized,compact,selection,fold,exclude_fold,actions,extended:/끝까지|팔꿈치펴|곧게|길게/.test(compact),tucked:/겨드랑|손.{0,8}(?:안쪽|숨|안보이)|tuck/.test(compact),loose:/느슨|포개|loose/.test(compact),guard:/방어|막기|가드|guard/.test(compact)};
}
function search(parts,text,{includeHeld=false,topK=50}={}){
 const query=parse(text),hits=[],held=[];
 for(const part of parts){
  const tags=part.semantics;if(!tags)continue;
  if(query.selection&&part.selection!==query.selection)continue;
  if(tags.relation_fragment&&(query.fold||query.guard))continue;
  const action=tags.action;
  if(query.exclude_fold&&['fold_arms','forearm_stack','forearm_cross'].includes(action))continue;
  if(query.fold&&action!=='fold_arms'&&!(query.loose&&action==='forearm_stack'))continue;
  if(query.guard&&action!=='cross_guard')continue;
  if(query.actions.length&&!(query.fold||query.guard)&&!query.actions.includes(action))continue;
  if(query.extended){const sides=part.selection==='both'?tags.active_sides:[part.selection];if(!sides.length||sides.some(s=>part.arms[s].elbow_flexion_deg>30))continue;}
  if(query.tucked&&!tags.tucked_hands_certified)continue;
  const aliases=tags.aliases.map(a=>norm(a).replace(/ /g,''));
  const exact=aliases.some(a=>query.compact.includes(a)||a.includes(query.compact));
  const chars=new Set(query.compact),doc=new Set(aliases.join(''));let count=0;for(const c of chars)if(doc.has(c))count++;
  const lexical=count/Math.max(chars.size,1);
  if(!(query.fold||query.guard||exact||lexical>=.6))continue;
  if(['left','right'].includes(part.selection)&&!tags.active_sides.includes(part.selection))continue;
  const quality=tags.pair_quality,isHeld=query.fold&&(!quality||!quality.eligible_strict_fold);
  const hit={part_id:part.id,label:part.label,selection:part.selection,action,score:Number(((exact?2:0)+lexical).toFixed(4)),quality_status:quality?quality.status:'requires_visual_review',reasons:isHeld&&quality?quality.reasons:[],requires_review:true};
  (isHeld?held:hits).push(hit);
 }
 const byScore=(a,b)=>b.score-a.score||(a.part_id<b.part_id?-1:a.part_id>b.part_id?1:0);hits.sort(byScore);held.sort(byScore);
 return {query,hits:(includeHeld?hits.concat(held):hits).slice(0,topK),held:held.slice(0,topK),status:hits.length?'candidates':held.length?'held_for_pair_quality':'no_verified_match',reason:query.tucked&&!hits.length?'손을 안쪽으로 숨기는 접촉이 검증된 파츠가 없습니다.':query.fold&&!hits.length?'팔짱은 반대팔 접촉과 실제 피부 검사까지 통과한 양팔 후보가 필요합니다.':null};
}
const api={parse,search};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ArmTextSearch=api;
})(typeof globalThis!=='undefined'?globalThis:this);
