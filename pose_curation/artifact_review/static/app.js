import {PoseViewer, Thumbnails} from './viewer.js';

const $ = selector => document.querySelector(selector);
const names = {pending: '미검수', accepted: '채택', hold: '보류', rejected: '제외'};
const scopeNames = {full: '전신', half: '반신', bust: '흉상', shoulder_head: '두상'};
const filterIds = ['query', 'group', 'category', 'cohort', 'status', 'auto', 'code'];
const autoNames = {pass: '자동 통과', hold: '자동 보류 권장', reject: '자동 제외 권장', unsupported: '자동 검사 불가', error: '자동 검사 오류'};
const limit = 12;
let viewer, thumbs, items = [], selected = null, total = 0, offset = 0;
let requestId = 0, selectionId = 0, dirty = false, saving = false;
let loadingPage = false, modelReady = false, activeFilters;

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.detail || '요청을 처리하지 못했습니다.');
  return data;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function summary() {
  const data = await api('/api/summary');
  $('#summary').textContent = `총 ${data.total.toLocaleString()}개 · 내 검수 ${data.total - (data.statuses.pending || 0)}개 완료 · 채택 ${data.statuses.accepted || 0} · 보류 ${data.statuses.hold || 0} · 제외 ${data.statuses.rejected || 0}`
    + (data.auto_version ? ` · 자동 검사: 제외 권장 ${data.auto.reject || 0} · 보류 권장 ${data.auto.hold || 0} · 통과 ${data.auto.pass || 0} (수정안으로 해결 ${data.auto_fixable})` : ' · 자동 검사 결과 없음');
  if ($('#code').options.length === 1) {
    for (const [code, label] of Object.entries(data.auto_codes || {})) $('#code').append(new Option(label, code));
  }
  if ($('#category').options.length === 1) {
    for (const category of data.categories) $('#category').append(new Option(category, category));
  }
}

function canLeave() {
  if (saving || loadingPage) return false;
  return !dirty || confirm('저장하지 않은 검수 내용이 있습니다. 저장하지 않고 이동할까요?');
}

function controls() {
  const index = items.findIndex(row => row.key === selected?.key);
  const busy = saving || loadingPage;
  $('#prev-pose').disabled = !selected || busy || offset + index <= 0;
  $('#next-pose').disabled = !selected || busy || offset + index >= total - 1;
  $('#prev-page').disabled = busy || offset === 0;
  $('#next-page').disabled = busy || offset + limit >= total;
  $('#scope').disabled = !selected || busy;
  for (const id of filterIds) $('#' + id).disabled = busy;
  for (const input of $('#review-form').elements) input.disabled = !selected || !modelReady || busy;
  $('#selected-position').textContent = selected ? `${offset + index + 1} / ${total.toLocaleString()}` : '';
}

function clearSelection(message = '포즈를 선택해 주세요') {
  selectionId++;
  selected = null;
  dirty = false;
  modelReady = false;
  viewer.clear();
  $('#pose-title').textContent = message;
  for (const id of ['pose-id', 'provenance', 'save-status', 'description']) $('#' + id).textContent = '';
  $('#auto-review').hidden = true;
  $('#note').value = '';
  $('#downloads').replaceChildren();
  $('#tags').replaceChildren();
  $('#model-status').hidden = false;
  $('#model-status').textContent = message;
  controls();
}

function cardFor(row) {
  const card = element('button', null, 'card');
  card.type = 'button';
  card.dataset.key = row.key;
  card.setAttribute('aria-pressed', 'false');
  card.setAttribute('aria-label', row.pose_id);
  card.append(element('div', '3D 미리보기 생성 중', 'placeholder'));
  const info = element('div', null, 'info');
  info.append(element('strong', row.title), element('small', row.pose_id),
    element('span', row.group === 'new' ? '추가' : '기존', 'badge'),
    element('span', names[row.review.status], 'badge decision ' + row.review.status));
  if (row.scene_reviewed) info.append(element('span', '손·소품 복구', 'badge'));
  if (row.auto) info.append(element('span', autoNames[row.auto.verdict] || row.auto.verdict, 'badge auto-' + row.auto.verdict));
  card.append(info);
  card.onclick = () => { if (canLeave()) select(row); };
  return card;
}

async function loadPage(pickLast = false, preferredKey = null) {
  const version = ++requestId;
  loadingPage = true;
  clearSelection('최신 포즈를 불러오는 중…');
  const params = new URLSearchParams({offset, limit, q: $('#query').value});
  for (const id of filterIds.slice(1)) params.set(id, $('#' + id).value);
  $('#list-message').textContent = '최신 포즈를 불러오는 중…';
  try {
    const data = await api('/api/poses?' + params);
    if (version !== requestId) return;
    items = data.items;
    total = data.total;
    $('#cards').replaceChildren(...items.map(cardFor));
    $('#page-info').textContent = `${total ? offset + 1 : 0}–${Math.min(offset + limit, total)} / ${total.toLocaleString()}`;
    $('#list-message').textContent = total ? '각 썸네일도 최신 3D 모델에서 생성합니다.' : '검색 결과가 없습니다.';
    thumbs.fill(items, (key, png, error) => {
      if (version !== requestId) return;
      const card = [...$('#cards').children].find(node => node.dataset.key === key);
      if (!card) return;
      if (png) {
        const img = element('img');
        img.src = png;
        img.alt = items.find(row => row.key === key)?.title || '포즈';
        card.firstChild.replaceWith(img);
      } else card.firstChild.textContent = '미리보기 오류: ' + error;
    });
    loadingPage = false;
    if (items.length) await select(items.find(row => row.key === preferredKey) || items[pickLast ? items.length - 1 : 0]);
    else clearSelection('검색 조건에 맞는 포즈가 없습니다.');
  } catch (error) {
    if (version === requestId) $('#list-message').textContent = error.message;
  } finally {
    if (version === requestId) { loadingPage = false; controls(); }
  }
}

async function loadModel() {
  if (!selected) return;
  const version = ++selectionId;
  modelReady = false;
  controls();
  $('#model-status').hidden = false;
  $('#model-status').textContent = '최신 3D 미리보기 불러오는 중…';
  try {
    const loaded = await viewer.load(`/api/assets/${selected.key}/${$('#scope').value}/glb`);
    if (version === selectionId && loaded) {
      modelReady = true;
      $('#model-status').hidden = true;
    }
  } catch (error) {
    if (version === selectionId) $('#model-status').textContent = '모델을 열지 못했습니다: ' + error.message;
  } finally {
    if (version === selectionId) controls();
  }
}

async function select(row) {
  selected = row;
  dirty = false;
  $('#pose-title').textContent = row.title;
  $('#pose-id').textContent = row.pose_id;
  $('#provenance').textContent = `${row.group === 'new' ? '추가한 포즈' : '기존 라이브러리'} · ${row.category}${row.scene_reviewed ? ' · 손·소품 검수 통과' : ''}`;
  $('#scope').replaceChildren(...row.scopes.map(scope => new Option(scopeNames[scope], scope)));
  $('#note').value = row.review.note;
  $(`input[name=decision][value=${row.review.status}]`).checked = true;
  $('#save-status').textContent = row.review.updated_at ? '저장된 검수 기록' : '아직 직접 검수하지 않았습니다.';
  $('#description').textContent = row.description || '등록된 설명이 없습니다.';
  $('#tags').replaceChildren(...row.tags.map(tag => element('span', tag, 'badge')));
  for (const card of $('#cards').children) card.setAttribute('aria-pressed', String(card.dataset.key === row.key));
  downloads();
  autoReview(row);
  history.replaceState(null, '', '?pose=' + row.key);
  await loadModel();
}

function downloads() {
  if (!selected) return;
  $('#downloads').replaceChildren();
  const scope = $('#scope').value;
  for (const kind of ['fbx', 'glb']) {
    const link = element('a', scopeNames[scope] + ' ' + kind.toUpperCase() + ' 다운로드');
    link.href = `/api/assets/${selected.key}/${scope}/${kind}`;
    link.download = '';
    $('#downloads').append(link);
  }
  if (selected.has_bvh) {
    const link = element('a', '원본 BVH');
    link.href = `/api/assets/${selected.key}/source/bvh`;
    link.download = '';
    $('#downloads').append(link);
  }
}

function autoReview(row) {
  const auto = row.auto;
  $('#auto-review').hidden = !auto;
  if (!auto) return;
  $('#auto-verdict').textContent = autoNames[auto.verdict] || auto.verdict;
  $('#auto-verdict').className = 'badge auto-' + auto.verdict;
  $('#auto-fill').hidden = !['hold', 'reject'].includes(auto.verdict);
  $('#auto-findings').replaceChildren(...auto.findings.map(f => {
    const limit = Array.isArray(f.limit) ? f.limit.join(' ~ ') : f.limit;
    const item = element('li', `${f.severity === 'reject' ? '제외' : '보류'} · ${f.label} ${f.value} (한계 ${limit})`, f.severity);
    item.title = '근거: ' + f.source;
    return item;
  }));
  const fix = auto.correction;
  $('#auto-fix').textContent = !fix ? '' : fix.usable
    ? `자동 수정안: ${autoNames[fix.verdict] || fix.verdict} · 손·발·머리 최대 이동 ${fix.max_move_torso} (몸통 길이 단위)${fix.max_move_torso < 0.01 ? ' — 관절 위치는 그대로, 비틀림만 교정' : ''}`
    : '자동 수정안이 결과를 개선하지 못해 제공하지 않습니다.';
  const ankle = fix?.ankle;
  if (ankle && ankle.status !== 'nothing') {
    const ankleText = {
      corrected: `바닥을 딛은 발목은 접지를 유지한 채 ${Math.round(ankle.alpha * 100)}%까지 보정했습니다.`,
      gross: '발이 크게 망가져 다시 세웠습니다. 반드시 눈으로 확인해 주세요.',
      replanted: '다른 관절을 고치며 움직인 발을 원래 접지 위치로 되돌렸습니다.',
      restored: '접지를 지키며 고칠 수 없어 바닥을 딛은 다리는 원본 그대로 두었습니다.',
    }[ankle.status];
    if (ankleText) $('#auto-fix').textContent += ' ' + ankleText;
  }
  $('#auto-sheet-box').hidden = !auto.findings.length || !row.has_bvh;
  $('#auto-sheet-box').open = false;
  $('#auto-sheet').removeAttribute('src');
  $('#auto-sheet-box').ontoggle = () => {
    if ($('#auto-sheet-box').open && !$('#auto-sheet').getAttribute('src')) $('#auto-sheet').src = `/api/poses/${row.key}/plausibility.png`;
  };
  if (fix?.usable) {
    const link = element('a', '자동 수정 BVH');
    link.href = `/api/assets/${row.key}/corrected/bvh`;
    link.download = '';
    $('#downloads').append(link);
  }
}

function fillFromAuto() {
  const auto = selected?.auto;
  if (!auto || !['hold', 'reject'].includes(auto.verdict)) return;
  $(`input[name=decision][value=${auto.verdict === 'reject' ? 'rejected' : 'hold'}]`).checked = true;
  const lines = auto.findings.slice(0, 4).map(f => `${f.label} ${f.value}`).join(', ');
  $('#note').value = ($('#note').value ? $('#note').value + '\n' : '') + '[자동검수] ' + lines;
  dirty = true;
  $('#save-status').textContent = '저장하지 않은 변경사항';
}

async function bulkApply() {
  const verdicts = ['reject', 'hold'];
  const post = body => api('/api/auto-review/apply', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  try {
    const plan = await post({verdicts, dry_run: true});
    if (!plan.count) { alert('적용할 미검수 포즈가 없습니다.'); return; }
    if (!confirm(`아직 검수하지 않은(메모도 없는) 포즈 ${plan.count.toLocaleString()}개에 자동 판정을 적용합니다.\n제외 ${plan.statuses.rejected || 0}개 · 보류 ${plan.statuses.hold || 0}개\n이미 검수한 포즈는 바꾸지 않으며, 메모 앞에 [자동검수]가 붙습니다. 진행할까요?`)) return;
    const done = await post({verdicts, dry_run: false});
    alert(`${done.count.toLocaleString()}개에 자동 판정을 적용했습니다.`);
    await summary();
    await loadPage(false, selected?.key);
  } catch (error) { alert(error.message); }
}

async function move(direction) {
  if (!canLeave() || !selected) return;
  const index = items.findIndex(row => row.key === selected.key) + direction;
  if (index >= 0 && index < items.length) return select(items[index]);
  offset += direction * limit;
  await loadPage(direction < 0);
}

async function save(next) {
  if (!selected || saving || !modelReady || loadingPage) return;
  const row = selected;
  saving = true;
  controls();
  $('#save-status').textContent = '저장 중…';
  let saved = false;
  try {
    row.review = await api(`/api/poses/${row.key}/review`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({content_hash: row.content_hash, status: $('input[name=decision]:checked').value, note: $('#note').value}),
    });
    dirty = false;
    saved = true;
    $('#save-status').textContent = '저장했습니다.';
    const card = [...$('#cards').children].find(node => node.dataset.key === row.key);
    const badge = card?.querySelector('.decision');
    if (badge) { badge.className = 'badge decision ' + row.review.status; badge.textContent = names[row.review.status]; }
    await summary();
  } catch (error) { $('#save-status').textContent = error.message; }
  finally { saving = false; controls(); }
  if (saved && next && offset + items.findIndex(item => item.key === row.key) < total - 1) await move(1);
}

function readFilters() { return Object.fromEntries(filterIds.map(id => [id, $('#' + id).value])); }
function filterChanged() {
  if (!canLeave()) {
    for (const id of filterIds) $('#' + id).value = activeFilters[id];
    return;
  }
  activeFilters = readFilters();
  offset = 0;
  loadPage();
}

// A shared pose URL reopens that exact entry, without retaining hidden filter state.
const initialKey = new URLSearchParams(location.search).get('pose');
try {
  viewer = new PoseViewer($('#viewer'));
  thumbs = new Thumbnails();
  controls();
  await summary();
  if (initialKey) {
    try { $('#query').value = (await api('/api/poses/' + encodeURIComponent(initialKey))).pose_id; }
    catch { $('#query').value = ''; }
  }
  activeFilters = readFilters();
  await loadPage(false, initialKey);
} catch (error) { $('#model-status').textContent = '검수 화면을 열지 못했습니다: ' + error.message; }

let timer;
$('#query').oninput = () => { clearTimeout(timer); timer = setTimeout(filterChanged, 350); };
for (const id of filterIds.slice(1)) $('#' + id).onchange = filterChanged;
$('#prev-page').onclick = () => { if (canLeave()) { offset = Math.max(0, offset - limit); loadPage(); } };
$('#next-page').onclick = () => { if (canLeave()) { offset += limit; loadPage(); } };
$('#prev-pose').onclick = () => move(-1);
$('#next-pose').onclick = () => move(1);
$('#scope').onchange = () => { downloads(); loadModel(); };
for (const button of document.querySelectorAll('[data-angle]')) button.onclick = () => viewer.fit(Number(button.dataset.angle));
$('#reset').onclick = () => viewer.fit();
$('#review-form').oninput = () => { dirty = true; $('#save-status').textContent = '저장하지 않은 변경사항'; };
$('#review-form').onsubmit = event => { event.preventDefault(); save(false); };
$('#save-next').onclick = () => save(true);
$('#auto-fill').onclick = fillFromAuto;
$('#auto-apply').onclick = () => { if (canLeave()) bulkApply(); };
window.addEventListener('beforeunload', event => { if (dirty || saving) { event.preventDefault(); event.returnValue = ''; } });
