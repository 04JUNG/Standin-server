import {request} from './api.js?v=qa-20261002';
import {OrientationPreview} from './orientation.js?v=head-1';
import {FaceCandidates} from './head-candidates.js?v=1';

const $ = id => document.getElementById(id);
const preview = new OrientationPreview({endpointBase: '/api/head'});
let config, query, points = Array(6).fill(null), active = 0, fitResult, revision = 0, fitController, recordUrl;
const svgNS = 'http://www.w3.org/2000/svg';
const faces = new FaceCandidates({
  onStart: () => {clearFit(); points = Array(6).fill(null); draw();},
  onSelected: result => acceptFit(result, '자동 검출 기준점')
});

function clearFit({invalidatePreview = true} = {}) {
  ++revision;
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
  $('head-fit').disabled = query.excluded || !points.every(Boolean) || !$('head-image').complete;
  $('head-overlay-crop').disabled = !points.every(Boolean) && !fitResult?.points;
}

function acceptFit(result, origin) {
  fitResult = result; preview.applySuggestion(result.orientation); draw();
  $('head-status').textContent = `${origin} 차이 ${(100*result.error).toFixed(1)}% · 참고 각도입니다. 주황색 투영과 초록색 점을 비교하고 실제 모델을 생성해 확인하세요.`;
  recordUrl = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)], {type:'application/json'}));
  $('head-fit-download').href = recordUrl; $('head-fit-download').download = 'head-landmark-suggestion.json'; $('head-fit-download').hidden = false;
}

function selectQuery() {
  clearFit();
  query = config.items.find(row => row.key === $('head-query').value);
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
  $('head-image').onload = () => { draw(); faces.render(); $('head-status').textContent = query.excluded ? `제외한 이미지 · ${query.exclusion_reason}` : '얼굴 후보를 선택하거나 기준점을 직접 지정해 주세요.'; };
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
  clearFit(); points[active] = [p.x, p.y];
  const next = points.findIndex(p => p === null);
  if (next >= 0) active = next;
  $('head-status').textContent = next >= 0 ? `${active+1}번 ${config.labels[active]}을 지정해 주세요.` : '6개 점을 확인한 뒤 각도 추천을 눌러 주세요.';
  draw();
});
$('head-query').addEventListener('change', selectQuery);
function manualAngleChange() {
  const hadResult = Boolean(fitResult);
  clearFit({invalidatePreview: false}); draw();
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
  $('angle-status').textContent = '출력 범위가 바뀌었습니다. 미리보기·파일을 다시 생성해 주세요.';
});
$('head-reset').addEventListener('click', () => {clearFit(); points = Array(6).fill(null); active = 0; draw(); $('head-status').textContent = '기준점을 초기화했습니다.';});
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
  const selectedPoints = points.every(Boolean) ? points : fitResult?.points;
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
  config.labels.forEach((label,index) => { const button = document.createElement('button'); button.dataset.index = index; button.textContent = `${index+1}. ${label}`; button.addEventListener('click', () => { active = index; draw(); }); $('head-anchors').append(button); });
  preview.setContext(config.reference, $('head-scope').value);
  $('head-query').disabled = false;
  if (!config.automatic_enabled) $('head-list-mode').value = 'active';
  populateQueries(new URLSearchParams(location.search).get('query'));
} catch(error) { $('head-status').textContent = error.message; $('angle-generate').disabled = true; }
