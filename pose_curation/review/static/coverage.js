const $ = id => document.getElementById(id);
const labels = {library: '포즈 부족', extraction: '관절 인식 문제', camera: '원근·시점 문제', partial: '부분 신체·겹침'};
const edges = [[5,6], [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]];
const imageUrl = id => `/api/coverage/image/${encodeURIComponent(id)}`;
const thumbUrl = hit => `/api/poses/${hit.key}/thumbnail?view=${encodeURIComponent(hit.view)}`;
let report;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function figure(title, source, description) {
  const frame = element('figure');
  const image = element('img');
  image.src = source;
  image.alt = title;
  image.loading = 'lazy';
  const caption = element('figcaption');
  caption.append(element('strong', title), element('span', description));
  frame.append(image, caption);
  return frame;
}

function overlay(item) {
  const frame = element('figure');
  const canvas = element('canvas');
  canvas.width = canvas.height = 450;
  const caption = element('figcaption');
  caption.append(element('strong', '실제로 인식한 관절'), element('span', item.query
    ? `${item.query.body_visible}/12개 신체 관절 · 빨강 선` : '인물 검출 실패'));
  frame.append(canvas, caption);
  const image = new Image();
  image.onload = () => drawOverlay(canvas, image, item.query);
  image.src = imageUrl(item.image_id);
  return frame;
}

function drawOverlay(canvas, image, query) {
  const ctx = canvas.getContext('2d');
  const size = canvas.width;
  const scale = Math.min(size / image.width, size / image.height);
  const ox = (size - image.width * scale) / 2;
  const oy = (size - image.height * scale) / 2;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(image, ox, oy, image.width * scale, image.height * scale);
  if (!query) return;
  ctx.strokeStyle = '#d32642';
  ctx.lineWidth = 3;
  for (const [a, b] of edges) {
    if (Math.min(query.scores[a], query.scores[b]) < .3) continue;
    const points = query.keypoints;
    ctx.beginPath();
    ctx.moveTo(ox + points[a][0] * scale, oy + points[a][1] * scale);
    ctx.lineTo(ox + points[b][0] * scale, oy + points[b][1] * scale);
    ctx.stroke();
  }
}

function candidate(title, hit) {
  if (hit) return figure(title, thumbUrl(hit), `${hit.pose_id} · 거리 ${hit.distance.toFixed(3)}`);
  const frame = element('figure');
  frame.append(element('div', '비교 가능한 신체 관절이 부족합니다.', 'placeholder'), element('figcaption', title));
  return frame;
}

function caseCard(item) {
  const card = element('article');
  const title = element('div', null, 'case-title');
  title.append(element('h2', item.title), element('span', labels[item.category], 'badge'));
  card.append(title, element('div', item.image_id, 'ids'));
  if (item.related_ids.length) {
    const related = element('details', null, 'related');
    related.append(element('summary', `같은 유형의 러프 ${item.related_ids.length}장`));
    const links = element('div', null, 'ids');
    for (const id of item.related_ids) {
      const link = element('a', id);
      link.href = imageUrl(id);
      link.target = '_blank';
      link.rel = 'noopener';
      links.append(link, document.createTextNode(' · '));
    }
    related.append(links);
    card.append(related);
  }
  card.append(element('p', item.problem, 'description'));
  const grid = element('div', null, 'comparison');
  grid.append(figure('원본 러프', imageUrl(item.image_id), item.origin === 'user_upload'
    ? '실제 업로드 · 익명 처리' : '제공 ZIP'), overlay(item),
    candidate('확장 전 자동 검색', item.before), candidate('확장 후 자동 검색', item.after));
  card.append(grid, element('p', item.outcome, 'result'));
  const recommendations = element('div', null, 'recommendations');
  for (const pose of item.recommendations) {
    const link = element('a');
    link.href = `/?pose=${pose.key}`;
    link.append(figure('보강한 기준 자세', thumbUrl(pose), `${pose.label} · 4방향 검수 열기 ↗`));
    recommendations.append(link);
  }
  if (recommendations.childElementCount) card.append(recommendations);
  return card;
}

function render() {
  const query = $('search').value.trim().toLowerCase();
  const filter = $('filter').value;
  const rows = report.cases.filter(item =>
    (filter === 'all' || item.category === filter) &&
    (!query || [item.title, item.image_id, ...item.related_ids].join(' ').toLowerCase().includes(query)));
  $('count').textContent = `부족 사례 ${rows.length}종 · 거리 감소는 최종 사용 가능 판정과 다릅니다.`;
  $('cases').replaceChildren(...rows.map(caseCard));
}

function renderInventory() {
  if (!$('inventory-panel').open || $('inventory').childElementCount) return;
  const fragment = document.createDocumentFragment();
  for (const item of report.inventory) {
    const people = item.people.length ? item.people : [null];
    for (const person of people) {
      const row = element('tr');
      const cell = element('td');
      const link = element('a', item.id);
      link.href = imageUrl(item.id);
      link.target = '_blank';
      link.rel = 'noopener';
      cell.append(link);
      row.append(cell, element('td', person ? person.person : '없음'),
        element('td', person ? `${person.body_visible}/12` : '—'),
        element('td', person?.before ? person.before.distance.toFixed(3) : '—'),
        element('td', person?.after ? person.after.distance.toFixed(3) : '—'));
      fragment.append(row);
    }
  }
  $('inventory').append(fragment);
}

async function main() {
  const response = await fetch('/api/coverage', {cache: 'no-store'});
  if (!response.ok) throw new Error('비교 보고서를 불러오지 못했습니다.');
  report = await response.json();
  const stats = [
    ['제공 러프', report.summary.supplied_images],
    ['실제 업로드 / 중복 제거', `${report.summary.user_uploads} / ${report.summary.unique_user_images}`],
    ['보강 후 검색 포즈', report.summary.after_poses],
    ['기준 대비 검색 포즈 증감', report.summary.added_poses],
  ];
  for (const [label, value] of stats) {
    const stat = element('div', null, 'stat');
    stat.append(element('strong', value.toLocaleString()), element('span', label));
    $('summary').append(stat);
  }
  $('method').textContent = report.method;
  for (const line of report.limitations) $('limitations').append(element('p', line));
  $('filter').addEventListener('change', render);
  $('search').addEventListener('input', render);
  $('inventory-panel').addEventListener('toggle', renderInventory);
  render();
}

main().catch(error => { $('count').textContent = error.message; });
