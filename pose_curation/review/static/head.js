import {request} from './api.js?v=qa-20261002';
import {OrientationPreview} from './orientation.js?v=head-2';
import {FaceCandidates} from './head-candidates.js?v=1';
import {HeadAngleReviews} from './head-reviews.js?v=2';
import {BustControls} from './head-bust.js?v=1';

const $ = id => document.getElementById(id);
const preview = new OrientationPreview({endpointBase: '/api/head', extraSpec: () => bust.extraSpec()});
let config, query, points = Array(6).fill(null), active = 0, fitResult, revision = 0, fitController, recordUrl;
let inputMode = 'landmarks', corners = [], region = null, shoulders = [];
const svgNS = 'http://www.w3.org/2000/svg';
const angleReviews = new HeadAngleReviews({
  context: () => ({query, reference:config?.reference, scope:$('head-scope').value, angles:preview.angles(), get bodyAngles() { return bust.angles(); }, get shoulderPoints() { return bust.enabled() ? bust.points : null; }}),
  restore: row => {
    clearFit();
    points = row.provenance.points || Array(6).fill(null);
    region = row.provenance.region || null; corners = []; shoulders = []; inputMode = 'landmarks';
    $('head-scope').value = row.scope;
    preview.setContext(config.reference, row.scope);
    preview.applySuggestion(row.angles);
    bust.restore(row.body_angles, row.shoulder_points);
    shoulders = row.shoulder_points || [];
    angleReviews.setTarget(row.provenance.candidate ? {candidate:row.provenance.candidate} : region ? {region} : {points:row.provenance.points});
    draw();
    $('head-status').textContent = '저장한 대상 얼굴·각도를 불러왔습니다. 기본 모델 미리보기를 다시 생성해 주세요.';
    $('angle-status').textContent = '저장한 각도입니다. 미리보기·파일을 다시 생성해 확인해 주세요.';
  }
});
const faces = new FaceCandidates({
  onStart: () => {clearFit(); bust.restore(null); angleReviews.setTarget(null); points = Array(6).fill(null); region = null; corners = []; shoulders = []; inputMode = 'landmarks'; draw();},
  onSelected: result => acceptFit(result, '자동 검출 기준점')
});
const bust = new BustControls({
  faceAngles: () => preview.angles(), query: () => query,
  changed: () => {preview.invalidate(); angleReviews.changed();},
  pickShoulders: () => {
    if (!query || query.excluded) return;
    inputMode = 'shoulders'; shoulders = []; draw();
    $('head-status').textContent = '그림에서 같은 인물의 화면 왼쪽 어깨, 오른쪽 어깨 순서로 눌러 주세요.';
    $('head-image').scrollIntoView({block:'center',behavior:'smooth'});
  }
});

function clearFit({invalidatePreview = true} = {}) {
  ++revision;
  bust.cancel();
  fitController?.abort();
  faces.cancelApply();
  fitResult = null;
  if (recordUrl) URL.revokeObjectURL(recordUrl);
  recordUrl = null;
  $('head-fit-download').hidden = true;
  $('head-fit-download').removeAttribute('href');
  if (invalidatePreview) preview.invalidate();
}

function draw() {
  $('head-overlay').replaceChildren();
  for (const button of $('head-anchors').children) {
    button.classList.toggle('active', Number(button.dataset.index) === active);
    button.classList.toggle('placed', Boolean(points[button.dataset.index]));
  }
  if (!query) return;
  const radius = fitResult?.source === 'detected_face_user_selected'
    ? Math.max(fitResult.bbox[2]-fitResult.bbox[0], fitResult.bbox[3]-fitResult.bbox[1]) * .018
    : Math.max(...query.size) * .006;
  function marker(point, color, label) {
    const circle = document.createElementNS(svgNS, 'circle');
    for (const [k,v] of Object.entries({cx: point[0], cy: point[1], r: radius, fill: color, stroke: 'white', 'stroke-width': radius * .2})) circle.setAttribute(k,v);
    $('head-overlay').append(circle);
    if (label) {
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', point[0] + radius * 1.4); text.setAttribute('y', point[1]);
      text.setAttribute('font-size', radius * 3); text.setAttribute('fill', color);
      text.setAttribute('stroke', 'white'); text.setAttribute('stroke-width', radius * .08);
      text.textContent = label; $('head-overlay').append(text);
    }
  }
  points.forEach((p,i) => p && marker(p, '#287959', String(i+1)));
  if (fitResult?.source === 'detected_face_user_selected') fitResult.points.forEach(p => marker(p, '#287959'));
  fitResult?.projected.forEach(p => marker(p, '#ce7429'));
  corners.forEach(p => marker(p, '#7352a2'));
  shoulders.forEach((p,i) => marker(p, '#245baf', `어깨 ${i+1}`));
  if (region) {
    const box = document.createElementNS(svgNS, 'rect');
    for (const [k,v] of Object.entries({x:region[0],y:region[1],width:region[2]-region[0],height:region[3]-region[1],fill:'none',stroke:'#7352a2','stroke-width':radius*.6})) box.setAttribute(k,v);
    $('head-overlay').append(box);
  }
  $('head-fit').disabled = query.excluded || !points.every(Boolean) || !$('head-image').complete;
  $('head-overlay-crop').disabled = !region && !points.every(Boolean) && !fitResult?.points;
}

function acceptFit(result, origin) {
  angleReviews.setTarget(result.source === 'detected_face_user_selected' ? {candidate:result.id} : {points:result.points});
  fitResult = result; preview.applySuggestion(result.orientation); draw();
  $('head-status').textContent = `${origin} 차이 ${(100*result.error).toFixed(1)}% · 참고 각도입니다. 주황색 투영과 초록색 점을 비교하고 실제 모델을 생성해 확인하세요.`;
  recordUrl = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)], {type:'application/json'}));
  $('head-fit-download').href = recordUrl; $('head-fit-download').download = 'head-landmark-suggestion.json'; $('head-fit-download').hidden = false;
}

function selectQuery() {
  clearFit();
  $('head-region').disabled = true;
  $('body-shoulders').disabled = true;
  bust.cancel(); bust.restore(null); region = null; corners = []; shoulders = []; inputMode = 'landmarks';
  query = config.items.find(row => row.key === $('head-query').value);
  angleReviews.setTarget(null);
  angleReviews.load(query);
  $('head-review-note').value = '';
  $('head-review-decision').value = 'hold';
  points = Array(6).fill(null); active = 0;
  $('angle-rough').hidden = true;
  $('angle-rough').removeAttribute('src');
  $('head-exclude').hidden = Boolean(query?.excluded);
  $('head-exclude').disabled = !query;
  $('head-restore').hidden = !query?.excluded;
  if (!query) {
    faces.load(null, $('head-image'));
    $('head-image').removeAttribute('src'); $('head-overlay').replaceChildren();
    $('head-fit').disabled = true; $('head-overlay-crop').disabled = true;
    $('head-status').textContent = '이 목록에 이미지가 없습니다. 오른쪽에서 직접 각도를 조절할 수 있습니다.'; return;
  }
  $('head-overlay').setAttribute('viewBox', `0 0 ${query.size.join(' ')}`);
  $('head-image').onload = () => {
    $('head-region').disabled = query.excluded;
    $('body-shoulders').disabled = query.excluded;
    draw(); faces.render();
    $('head-status').textContent = query.excluded ? `제외한 이미지 · ${query.exclusion_reason}` : '얼굴 후보를 선택하거나 기준점·머리 영역을 직접 지정해 주세요.';
  };
  $('head-image').onerror = () => { $('head-status').textContent = '러프를 읽지 못했습니다. 다시 선택해 주세요.'; $('head-fit').disabled = true; };
  $('head-image').src = `/api/head/queries/${encodeURIComponent(query.key)}/image`;
  faces.load(query, $('head-image'));
  draw();
}

function populateQueries(desired) {
  const excluded = $('head-list-mode').value === 'excluded';
  const detected = $('head-list-mode').value === 'detected';
  $('head-query').replaceChildren();
  for (const row of config.items.filter(row => Boolean(row.excluded) === excluded && (!detected || row.face_candidates > 0))) {
    const option = document.createElement('option'); option.value = row.key;
    option.textContent = `${row.origin} · ${row.key}${row.face_candidates ? ` · 후보 ${row.face_candidates}` : ''}`; $('head-query').append(option);
  }
  if ([...$('head-query').options].some(row => row.value === desired)) $('head-query').value = desired;
  const count = config.items.filter(row => row.excluded).length;
  const ready = config.items.filter(row => !row.excluded && row.face_candidates > 0).length;
  $('head-list-status').textContent = `검수 대상 ${config.items.length-count}장 · 자동 후보 ${ready}장 · 제외 ${count}장 · 원본 파일 보관`;
  selectQuery();
}

async function reviewQuery(excluded) {
  if (!query) return;
  const selected = query;
  clearFit();
  for (const id of ['head-exclude','head-restore','head-query','head-list-mode']) $(id).disabled = true;
  try {
    await request(`/api/head/queries/${encodeURIComponent(selected.key)}/review`, {method:'POST', headers:{'Content-Type':'application/json','X-Pose-Review':'1'}, body:JSON.stringify({content_hash:selected.content_hash, excluded, reason:excluded ? '두상·흉상 대상 없음' : '사용자 복구'})});
    config = await request('/api/head');
    populateQueries();
  } catch(error) { $('head-status').textContent = error.message; }
  finally {
    for (const id of ['head-restore','head-query','head-list-mode']) $(id).disabled = false;
    $('head-exclude').disabled = !query;
  }
}
$('head-exclude').addEventListener('click', () => reviewQuery(true));
$('head-restore').addEventListener('click', () => reviewQuery(false));
$('head-list-mode').addEventListener('change', () => populateQueries());

$('head-overlay').addEventListener('pointerdown', event => {
  if (!query || query.excluded || !$('head-image').complete || !$('head-image').naturalWidth) return;
  const svg = $('head-overlay'), matrix = svg.getScreenCTM();
  if (!matrix) return;
  const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
  if (p.x < 0 || p.y < 0 || p.x >= query.size[0] || p.y >= query.size[1]) return;
  if (inputMode === 'shoulders') {
    shoulders.push([p.x,p.y]);
    if (shoulders.length === 2) { inputMode = 'landmarks'; bust.fitShoulders(shoulders); }
    $('head-status').textContent = shoulders.length === 1 ? '화면 오른쪽 어깨를 눌러 주세요.' : '어깨 기울기를 계산합니다. 오른쪽 몸통 설정에서 확인하세요.';
    draw(); return;
  }
  if (inputMode === 'region') {
    clearFit(); corners.push([p.x,p.y]);
    if (corners.length === 2) {
      const [a,b] = corners;
      const box = [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];
      corners = [];
      if (box[2]-box[0] < 16 || box[3]-box[1] < 16) {
        $('head-status').textContent = '얼굴 영역이 너무 작습니다. 16px 이상으로 다시 지정해 주세요.';
      } else {
        region = box; inputMode = 'landmarks'; angleReviews.setTarget({region});
        $('head-status').textContent = '얼굴 영역을 지정했습니다. 측면·후면 버튼과 각도로 맞춘 뒤 미리보기를 생성하고 저장하세요.';
      }
    } else $('head-status').textContent = '얼굴 영역의 반대쪽 모서리를 눌러 주세요.';
    draw(); return;
  }
  region = null; corners = [];
  bust.restore(null); shoulders = [];
  clearFit(); points[active] = [p.x, p.y];
  angleReviews.setTarget(null);
  const next = points.findIndex(p => p === null);
  if (next >= 0) active = next;
  $('head-status').textContent = next >= 0 ? `${active+1}번 ${config.labels[active]}을 지정해 주세요.` : '6개 점을 확인한 뒤 각도 추천을 눌러 주세요.';
  draw();
});
$('head-query').addEventListener('change', selectQuery);
function manualAngleChange() {
  const hadResult = Boolean(fitResult);
  clearFit({invalidatePreview: false}); draw();
  angleReviews.changed();
  if (hadResult) $('head-status').textContent = '각도를 직접 변경했습니다. 이전 추천 점 겹침을 지웠습니다.';
}
for (const name of ['yaw','pitch','roll']) {
  $(name+'-angle').addEventListener('input', manualAngleChange);
  $(name+'-number').addEventListener('input', manualAngleChange);
}
$('angle-presets').addEventListener('click', manualAngleChange);
$('head-scope').addEventListener('change', () => {
  if (!config) return;
  const angles = preview.angles();
  preview.setContext(config.reference, $('head-scope').value);
  preview.applySuggestion(angles);
  bust.update();
  angleReviews.changed();
  $('angle-status').textContent = '출력 범위가 바뀌었습니다. 미리보기·파일을 다시 생성해 주세요.';
});
$('head-reset').addEventListener('click', () => {clearFit(); bust.cancel(); angleReviews.setTarget(null); points = Array(6).fill(null); region = null; corners = []; shoulders = []; inputMode = 'landmarks'; active = 0; draw(); $('head-status').textContent = '기준점을 초기화했습니다.';});
$('head-region').addEventListener('click', () => {
  if (!query || query.excluded) return;
  clearFit(); bust.restore(null); angleReviews.setTarget(null); points = Array(6).fill(null); region = null; corners = []; shoulders = []; inputMode = 'region'; draw();
  $('head-status').textContent = '대상 머리 전체를 감싸는 영역의 두 모서리를 눌러 주세요. 가려진 눈·코를 지정할 필요는 없습니다.';
  $('head-image').scrollIntoView({block:'center',behavior:'smooth'});
});
$('head-fit').addEventListener('click', async () => {
  if (!query || query.excluded || !points.every(Boolean)) return;
  clearFit(); const token = revision;
  fitController = new AbortController(); $('head-fit').disabled = true;
  $('head-status').textContent = '지정한 기준점으로 참고 각도를 계산하고 있습니다…';
  try {
    const result = await request('/api/head/fit', {method:'POST', headers:{'Content-Type':'application/json','X-Pose-Review':'1'}, body:JSON.stringify({query:query.key, content_hash:query.content_hash, points}), signal:fitController.signal});
    if (token !== revision) return;
    acceptFit(result, '직접 지정한 기준점');
  } catch(error) { if (token === revision && error.name !== 'AbortError') $('head-status').textContent = error.message; }
  finally { if (token === revision) draw(); }
});
$('head-overlay-crop').addEventListener('click', () => {
  const selectedPoints = region ? [region.slice(0,2),region.slice(2)] : points.every(Boolean) ? points : fitResult?.points;
  if (!selectedPoints) return;
  const xs = selectedPoints.map(p=>p[0]), ys = selectedPoints.map(p=>p[1]);
  const cx = (Math.min(...xs)+Math.max(...xs))/2, cy = (Math.min(...ys)+Math.max(...ys))/2;
  const side = Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys))*1.35;
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 512;
  canvas.getContext('2d').drawImage($('head-image'), cx-side/2,cy-side/2,side,side,0,0,512,512);
  $('angle-rough').src = canvas.toDataURL('image/png'); $('angle-rough').hidden = false;
});
$('head-overlay-clear').addEventListener('click', () => { $('angle-rough').hidden = true; });
for (const name of ['scale','x','y']) $('head-overlay-'+name).addEventListener('input', () => {
  $('angle-rough').style.transform = `translate(${$('head-overlay-x').value}%,${$('head-overlay-y').value}%) scale(${$('head-overlay-scale').value/100})`;
});

try {
  config = await request('/api/head');
  config.labels.forEach((label,index) => { const button = document.createElement('button'); button.dataset.index = index; button.textContent = `${index+1}. ${label}`; button.addEventListener('click', () => { inputMode = 'landmarks'; active = index; draw(); }); $('head-anchors').append(button); });
  preview.setContext(config.reference, $('head-scope').value);
  $('head-query').disabled = false;
  if (!config.automatic_enabled) $('head-list-mode').value = 'active';
  populateQueries(new URLSearchParams(location.search).get('query'));
} catch(error) { $('head-status').textContent = error.message; $('angle-generate').disabled = true; }
