import { request, thumbnailUrl, saveReview } from "./api.js?v=qa-20261002";
import { SkeletonViewer } from "./viewer.js";
import { FramedPreview } from "./framing.js?v=20261002";
import { OrientationPreview } from "./orientation.js?v=scoped-20261002";

const $ = (id) => document.getElementById(id);
const statusLabels = { pending: "검수 대기", accepted: "채택", hold: "보류", rejected: "제외" };
const viewLabels = { front: "정면", three_quarter: "45°", side: "측면", back: "후면" };
const movementLabels = { ID: "정지 자세", FW: "걷기", FR: "달리기", combat: "전투", sports: "스포츠", daily: "일상" };
const sourceLabel = meta => ({"100style": "100STYLE", quaternius: "Quaternius", accad: "ACCAD 모션캡처", cmu: "CMU 모션캡처", authored_combat: "직접 제작", authored_scenario: "상황별 제작"}[meta.source] || meta.source || "신규");
const initialParams = new URLSearchParams(location.search);
const initialCategory = initialParams.get('category') || '';
const initialStatus = initialParams.get('status') || 'all';
const state = { group: "all", q: initialParams.get('q') || "", status: Object.hasOwn(statusLabels, initialStatus) ? initialStatus : "all", batch: initialParams.get('batch') || "", category: initialCategory, offset: 0, limit: 48, view: "front", items: [], total: 0 };
state.library_scope = ['half','bust'].includes(initialParams.get('library_scope')) ? initialParams.get('library_scope') : 'all';
const drafts = new Map();
let activePose = null, decision = "pending", loadSequence = 0, detailSequence = 0, toastTimer;
let renderInProgress = false;
const viewer = new SkeletonViewer($("skeleton-canvas"));
let detailView = state.view;
const orientationPreview = new OrientationPreview();
const framedPreview = new FramedPreview(() => {
  setPreview(detailView);
  if (activePose) orientationPreview.setContext(activePose, framedPreview.scope);
});
framedPreview.scope = ['half','bust','head'].includes(initialParams.get('scope')) ? initialParams.get('scope')
  : state.library_scope !== 'all' ? state.library_scope : 'full';

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text != null) element.textContent = text;
  return element;
}

function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4000);
}

async function refreshSummary() {
  const data = await request("/api/summary");
  $("count-all").textContent = data.total.toLocaleString();
  $("total-heading").textContent = data.total.toLocaleString();
  $("count-existing").textContent = (data.groups.existing || 0).toLocaleString();
  $("count-new").textContent = (data.groups.new || 0).toLocaleString();
  $("count-excluded").textContent = (data.excluded || 0).toLocaleString();
  const reviewed = data.total - (data.reviews.pending || 0);
  $("review-count").textContent = `${reviewed.toLocaleString()}개`;
  $("review-progress").max = data.total || 1;
  $("review-progress").value = reviewed;
  const batchSelect = $("batch-filter");
  batchSelect.replaceChildren(new Option("모든 배치", ""), new Option("기존 S3 라이브러리", "s3-v1"));
  for (const batch of data.batches) batchSelect.add(new Option(`${batch.id} · ${batch.poses}`, batch.id));
  batchSelect.value = state.batch;
  const notices = [...data.errors];
  renderInProgress = false;
  for (const batch of data.batches) {
    if (batch.status === "building") notices.push(`${batch.id}: 후보 생성 중 · ${batch.poses}개 준비됨`);
    if (batch.failures.length) notices.push(`${batch.id}: ${batch.failures.length}개 클립 처리 실패. 완료된 후보는 검수할 수 있습니다.`);
    const render = batch.character_render;
    if (render?.status === "rendering") {
      renderInProgress = true;
      notices.push(`캐릭터 미리보기 생성 중 · ${render.completed}/${render.requested}개 완료`);
    } else if (render?.status === "partial") {
      notices.push(`캐릭터 미리보기 ${render.completed}/${render.requested}개 완료 · 실패한 항목은 다시 렌더링해 주세요.`);
    }
  }
  $("notice").textContent = notices.join(" / ");
  $("notice").hidden = !notices.length;
  return data;
}

function poseCard(pose) {
  const card = node("button", "pose-card");
  card.setAttribute("aria-label", `${pose.group === "new" ? "신규" : "기존"} 포즈 ${pose.pose_id} 열기`);
  const imageBox = node("div", "card-image");
  const image = node("img");
  if (pose.preview_kind === "character") image.src = thumbnailUrl(pose, state.view);
  else image.hidden = true;
  image.alt = pose.pose_id;
  image.loading = "lazy";
  imageBox.append(image, node("span", `badge ${pose.group}`, pose.group === "new" ? `NEW · ${sourceLabel(pose.metadata)}` : "기존"));
  if (pose.preview_kind !== "character") imageBox.append(node("span", "preview-pending", "캐릭터 미리보기 준비 중"));
  if (pose.review.status !== "pending") imageBox.append(node("span", `card-status ${pose.review.status}`, statusLabels[pose.review.status]));
  const body = node("div", "card-body");
  const title = pose.group === "new" ? `${pose.metadata.style} · ${movementLabels[pose.metadata.movement] || pose.metadata.movement}` : pose.pose_id;
  const heading = node("div", "card-title", title);
  heading.title = pose.pose_id;
  const subtitle = node("div", "card-subtitle");
  subtitle.append(node("span", "", pose.metadata.category_label || (pose.group === "new" ? `프레임 ${pose.metadata.source_frame_0based}` : "S3 라이브러리")));
  subtitle.append(node("span", "", pose.preview_kind === "character" ? "캐릭터 · 4 views" : "미리보기 준비 중"));
  body.append(heading, subtitle);
  card.append(imageBox, body);
  card.addEventListener("click", () => openDetail(pose.key).catch((error) => toast(error.message)));
  return card;
}

async function loadPage() {
  const sequence = ++loadSequence;
  const query = new URLSearchParams({ group: state.group, q: state.q, status: state.status, batch: state.batch, category: state.category, library_scope: state.library_scope, offset: state.offset, limit: state.limit });
  let data;
  try { data = await request(`/api/poses?${query}`); }
  catch (error) {
    if (sequence === loadSequence) {
      state.items = []; $('grid').replaceChildren();
      $('result-count').textContent = error.message;
      $('previous').disabled = true; $('next').disabled = true;
    }
    throw error;
  }
  if (sequence !== loadSequence) return;
  if (!data.items.length && data.total && state.offset >= data.total) {
    state.offset = Math.floor((data.total - 1) / state.limit) * state.limit;
    return loadPage();
  }
  state.items = data.items;
  state.total = data.total;
  $("grid").replaceChildren(...data.items.map(poseCard));
  $("empty").hidden = data.total > 0;
  $("result-count").textContent = `${data.total.toLocaleString()}개 포즈 · ${state.group === "excluded" || state.status === "rejected" ? "제외 목록 · 대기 또는 채택으로 저장하면 복구" : state.group === "new" ? "신규 후보" : state.group === "existing" ? "기존 라이브러리" : "전체 라이브러리"}`;
  if (state.library_scope !== 'all') $("result-count").textContent += ` · ${state.library_scope === 'half' ? '반신' : '흉상'} 대표 · 카드는 원본 전신, 상세에서 실제 부분 출력`;
  $("page-label").textContent = `${Math.floor(state.offset / state.limit) + 1} / ${Math.max(1, Math.ceil(data.total / state.limit))}`;
  $("previous").disabled = state.offset === 0;
  $("next").disabled = state.offset + state.limit >= data.total;
}

function rememberDraft() {
  if (activePose) drafts.set(activePose.key, { status: decision, note: $("review-note").value, visual_checks: checkedVisualItems() });
}

function checkedVisualItems() {
  return [...document.querySelectorAll('#visual-checklist input:checked')].map(input => input.value);
}

function selectDecision(status) {
  decision = status;
  for (const button of $("decision-buttons").children) {
    button.classList.toggle("selected", button.dataset.status === status);
    button.setAttribute("aria-pressed", String(button.dataset.status === status));
  }
}

function metadataRow(label, value, href) {
  const row = node("div", "meta-row");
  const text = node("strong");
  if (href) {
    const link = node("a", "", value);
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    text.append(link);
  } else text.textContent = value;
  row.append(node("span", "", label), text);
  return row;
}

function setPreview(view) {
  if (!activePose) return;
  detailView = view;
  const url = framedPreview.url(view);
  const ready = Boolean(url);
  $("detail-image").hidden = !ready;
  $("detail-pending").hidden = ready;
  if (ready) $("detail-image").src = url;
  else $("detail-image").removeAttribute("src");
  $("detail-pending").textContent = framedPreview.result?.status === "failed" ? "미리보기를 생성하지 못했습니다. 다시 생성해 주세요." : "캐릭터 미리보기 준비 중…";
  $("detail-image").alt = `${activePose.pose_id} ${viewLabels[view]}`;
  for (const button of $("detail-views").children) {
    button.classList.toggle("active", button.dataset.view === view);
    const image = button.querySelector("img");
    const source = framedPreview.url(button.dataset.view);
    image.hidden = !source;
    if (source) image.src = source;
    else image.removeAttribute("src");
  }
}

function setMode(mode) {
  for (const button of document.querySelectorAll(".visual-tabs button")) button.classList.toggle("active", button.dataset.mode === mode);
  $("preview-panel").hidden = mode !== "preview";
  $("skeleton-panel").hidden = mode !== "skeleton";
  if (mode === "skeleton") viewer.draw();
}

async function openDetail(key) {
  rememberDraft();
  framedPreview.stop();
  orientationPreview.invalidate();
  const sequence = ++detailSequence;
  const [pose, qa] = await Promise.all([request(`/api/poses/${key}`), request(`/api/poses/${key}/qa`)]);
  if (sequence !== detailSequence) return;
  activePose = pose;
  const meta = pose.metadata;
  $("hand-focus").hidden = !meta.hand_augmentation && meta.source_hand_joints !== 30;
  for (const button of $("hand-focus").children) {
    button.classList.toggle("active", button.dataset.hand === "");
    button.disabled = true;
  }
  $("detail-title").textContent = pose.pose_id;
  $("detail-badge").className = `badge ${pose.group}`;
  $("detail-badge").textContent = pose.group === "new" ? `신규 후보 · ${sourceLabel(meta)}` : "기존 라이브러리 · S3";
  $("download-bvh").href = `/api/poses/${key}/bvh`;
  $("preview-caption").textContent = pose.group === "new"
    ? (pose.preview_kind === "character"
      ? "기존 라이브러리와 같은 Standin Master V2 캐릭터로 렌더링했습니다. 3D 골격 탭에서 BVH 관절과 손 모양을 확인할 수 있습니다."
      : "캐릭터 미리보기를 준비하고 있습니다. 3D 골격 탭에서 원본 자세를 확인할 수 있습니다.")
    : "서비스에서 사용하던 캐릭터 썸네일입니다. 3D 골격 탭에서 원본 BVH도 확인할 수 있습니다.";
  const rows = [metadataRow("배치", pose.batch)];
  const qaLabels = { approved: "현재 검수 유효", excluded: "제외 유지", blocked: "보정·검사 필요", visual_review: "시각 검수 필요", existing_snapshot: "기존 스냅샷 · 신규 기준 자동 검사 대상 외" };
  rows.push(metadataRow("통합 검수", qaLabels[qa.status] || qa.status));
  if (qa.findings.length) rows.push(metadataRow("검수 항목", qa.findings.map(item => item.message).join(" / ")));
  if (pose.group === "new") {
    rows.push(metadataRow("스타일", `${meta.style} / ${movementLabels[meta.movement] || meta.movement}`));
    if (Number.isFinite(meta.source_time_seconds)) rows.push(metadataRow("원본 프레임", `${meta.source_frame_0based} · ${meta.source_time_seconds.toFixed(2)}초`));
    if (meta.selection) rows.push(metadataRow("선택 이유", { held_pose: "자세가 유지되는 구간", expressive_extreme: "자세 변화의 정점", limb_extreme: "팔다리 움직임의 정점", phase_coverage: "동작 단계의 대표 자세" }[meta.selection.reason] || meta.selection.reason));
    if (meta.source === "authored_combat" || meta.source === "authored_scenario") rows.push(metadataRow("제작 방식", "3D 자세 설계 · 모션캡처 및 러프 관절 추론 데이터가 아닙니다."));
    if (meta.category_label) rows.push(metadataRow("상황 카테고리", meta.category_label));
    if (meta.nearest_scenario) rows.push(metadataRow("가장 비슷한 상황 포즈", `${meta.nearest_scenario.pose_id} · 골격 거리 ${meta.nearest_scenario.distance.toFixed(3)}`));
    if (meta.prop_guides) rows.push(metadataRow("소품 미리보기", "파지·배치 확인용 단순 도형입니다. BVH에는 인물의 관절만 포함됩니다."));
    if (meta.category === 'romance') rows.push(metadataRow("인물 배치", "인물 1명의 자세입니다. 상대와의 거리는 장면에서 맞춰 주세요."));
    if (meta.composition_variant) {
      rows.push(metadataRow("구도 변형", "팔다리·손가락을 유지하고 전체 방향만 회전했습니다. 지면에 선 자세와 별도인 장면 배치용 변형입니다."));
      rows.push(metadataRow("기준 자세", meta.composition_variant.base_pose_id));
    }
    rows.push(metadataRow("출처", `${sourceLabel(meta)} · ${meta.author || ""}`, meta.catalog_url || meta.source_url));
    rows.push(metadataRow("라이선스", meta.license, meta.license_url));
    rows.push(metadataRow("자동 검사", "1프레임 · 관절 매핑 · 저장 전후 좌표 통과"));
    if (meta.quality_review) rows.push(metadataRow("자세 재검수", meta.quality_review.summary));
    if (meta.torso_correction) rows.push(metadataRow("허리 보정", "척추 회전 재정렬·분산 · 골반과 가슴 사이의 급격한 꺾임 보정" + (meta.torso_correction.post_torso_arm_clearance ? " · 허리 보정 후 팔 위치도 조정" : "")));
    if (meta.anatomy_check?.torso_segment_rotation_degrees) rows.push(metadataRow("척추 검사", `분절별 회전 ${Object.values(meta.anatomy_check.torso_segment_rotation_degrees).map(x => x.toFixed(1) + '°').join(' / ')} · 실제 캐릭터 검사`));
    if (meta.anatomy_correction) rows.push(metadataRow(meta.torso_correction?.post_torso_arm_clearance ? "이전 체형 보정" : "체형 보정", `${meta.torso_correction?.post_torso_arm_clearance ? "허리 수정 전 단계에서 " : ""}팔을 몸 바깥으로 ${meta.anatomy_correction.total_degrees}° 조정 · 손가락·팔꿈치의 관절 각도 보존.`));
    if (meta.anatomy_check) {
      const flags = meta.anatomy_check.flags || [];
      rows.push(metadataRow("팔·몸통 검사", flags.length ? "추가 검수 필요 · " + flags.join(" / ") : "현재 캐릭터의 팔꿈치 굽힘 평면·팔/몸통 메시 교차 검사에서 경고 없음. 4방향 시각 검수 별도."));
    }
    if (meta.hand_augmentation) {
      const hands = meta.hand_augmentation;
      const labels = { relaxed: "힘을 뺀 손", open: "편 손", fist: "주먹", grip: "손잡이 파지", cup: "컵 파지", support: "받치는 손", pinch: "집는 손" };
      rows.push(metadataRow("손가락", meta.scenario ? "양손 30개 관절 · 상황별 손 모양 제작" : "양손 30개 관절 추가 · 기본 모양으로 보완"));
      rows.push(metadataRow("손 모양", `왼손: ${labels[hands.left]} / 오른손: ${labels[hands.right]}`));
      rows.push(metadataRow("손 데이터 출처", meta.scenario ? "원본 손가락 리그의 회전을 보정해 제작 · 실제 소품 접촉을 캡처한 데이터가 아닙니다." : "프리셋으로 생성 · 원본 모션에 손가락 정보 없음"));
    }
    if (meta.source_hand_joints === 30) rows.push(metadataRow("손가락", "원본 애니메이션의 양손 30개 관절·회전 보존"));
    if (meta.preview) rows.push(metadataRow("미리보기 모델", "Standin Master V2 · Blender 5.2"));
  } else {
    rows.push(metadataRow("출처 기록", `${meta.source} / ${meta.license}`));
    rows.push(metadataRow("참고", "기존 DB의 출처·라이선스 표기는 보완이 필요합니다."));
  }
  $("metadata").replaceChildren(...rows);
  $("detail-views").replaceChildren(...Object.entries(viewLabels).map(([view, label]) => {
    const button = node("button");
    button.dataset.view = view;
    const image = node("img");
    if (pose.preview_kind === "character") image.src = thumbnailUrl(pose, view);
    else image.hidden = true;
    image.alt = ""; // The adjacent label names this view button.
    button.append(image, node("span", "", label));
    button.addEventListener("click", () => setPreview(view));
    return button;
  }));
  detailView = state.view;
  framedPreview.setPose(pose);
  setMode("preview");
  const draft = drafts.get(key) || pose.review;
  const checklist = $("visual-checklist");
  checklist.hidden = pose.group !== "new";
  checklist.replaceChildren(node("legend", "", "채택 전 시각 검수"));
  const checked = draft.visual_checks || pose.review.evidence?.visual_checks || [];
  for (const [id, label] of Object.entries(qa.visual_checks)) {
    const item = node("label");
    const input = node("input");
    input.type = "checkbox"; input.value = id; input.checked = checked.includes(id);
    item.append(input, document.createTextNode(label)); checklist.append(item);
  }
  selectDecision(draft.status);
  $("review-note").value = draft.note;
  $("save-state").textContent = pose.review.updated_at ? "저장된 검수 기록" : "검수 대기";
  $("nearest-section").hidden = !meta.nearest_existing_key;
  if (meta.nearest_existing_key) {
    const nearest = { key: meta.nearest_existing_key, content_hash: "baseline" };
    $("nearest-image").src = thumbnailUrl(nearest);
    $("nearest-name").textContent = meta.nearest_existing.pose_id;
    $("nearest-distance").textContent = `골격 거리 ${meta.nearest_existing.distance.toFixed(3)}${meta.nearest_existing.distance < 0.15 ? " · 유사 후보" : ""}`;
    $("open-nearest").onclick = () => openDetail(meta.nearest_existing_key).catch((error) => toast(error.message));
  }
  if (!$("detail-dialog").open) $("detail-dialog").showModal();
  $("detail-prev").disabled = !state.items.some((item) => item.key === key) || state.items[0]?.key === key;
  $("detail-next").disabled = !state.items.some((item) => item.key === key) || state.items.at(-1)?.key === key;
  viewer.setData({ joints: [] });
  const geometry = await request(`/api/poses/${key}/skeleton`);
  if (sequence === detailSequence) {
    viewer.setData(geometry);
    for (const button of $("hand-focus").children) button.disabled = false;
  }
}

function moveDetail(delta) {
  const index = state.items.findIndex((pose) => pose.key === activePose?.key);
  const target = state.items[index + delta];
  if (index >= 0 && target) openDetail(target.key).catch((error) => toast(error.message));
}

async function changeFilter() {
  state.offset = 0;
  await loadPage();
}

$("groups").addEventListener("click", (event) => {
  const button = event.target.closest("[data-group]");
  if (!button) return;
  state.group = button.dataset.group;
  if (state.group === 'excluded') { state.library_scope = 'all'; $('library-scope').value = 'all'; }
  state.batch = "";
  state.category = "";
  $("category-filter").value = "";
  state.status = "all";
  $("status-filter").value = "all";
  $("batch-filter").value = "";
  for (const item of $("groups").children) item.classList.toggle("active", item === button);
  changeFilter().catch((error) => toast(error.message));
});
let searchTimer;
$('library-scope').value = state.library_scope;
$('library-scope').addEventListener('change', () => {
  state.library_scope = $('library-scope').value;
  framedPreview.scope = state.library_scope === 'all' ? 'full' : state.library_scope;
  changeFilter().catch(error => toast(error.message));
});
$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.q = $("search").value; changeFilter().catch((error) => toast(error.message)); }, 180);
});
for (const [id, property] of [["status-filter", "status"], ["batch-filter", "batch"], ["category-filter", "category"]]) {
  $(id).addEventListener("change", () => { state[property] = $(id).value; changeFilter().catch((error) => toast(error.message)); });
}
$("view-switch").addEventListener("click", (event) => {
  const button = event.target.closest("[data-view]");
  if (!button) return;
  state.view = button.dataset.view;
  for (const item of $("view-switch").children) item.classList.toggle("active", item === button);
  $("grid").replaceChildren(...state.items.map(poseCard));
});
for (const [id, delta] of [["previous", -1], ["next", 1]]) {
  $(id).addEventListener("click", () => { state.offset += delta * state.limit; loadPage().catch((error) => toast(error.message)); window.scrollTo({ top: 0, behavior: "smooth" }); });
}
$("refresh").addEventListener("click", () => Promise.all([refreshSummary(), loadPage()]).catch((error) => toast(error.message)));
$("close-detail").addEventListener("click", () => $("detail-dialog").close());
$("detail-dialog").addEventListener("close", rememberDraft);
$("detail-dialog").addEventListener("close", () => { framedPreview.stop(); orientationPreview.invalidate(); ++detailSequence; });
$("detail-prev").addEventListener("click", () => moveDetail(-1));
$("detail-next").addEventListener("click", () => moveDetail(1));
$("reset-camera").addEventListener("click", () => viewer.reset());
$("hand-focus").addEventListener("click", (event) => {
  const button = event.target.closest("[data-hand]");
  if (!button) return;
  viewer.focusHand(button.dataset.hand || null);
  for (const item of $("hand-focus").children) item.classList.toggle("active", item === button);
});
for (const button of document.querySelectorAll(".visual-tabs button")) button.addEventListener("click", () => setMode(button.dataset.mode));
$("decision-buttons").addEventListener("click", (event) => {
  const button = event.target.closest("[data-status]");
  if (button) { selectDecision(button.dataset.status); $("save-state").textContent = "저장 전"; }
});
$("review-note").addEventListener("input", () => { $("save-state").textContent = "저장 전"; });
$("review-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  const pose = activePose;
  try {
    const review = await saveReview(pose, decision, $("review-note").value, checkedVisualItems());
    drafts.delete(pose.key);
    pose.review = review;
    $("save-state").textContent = "저장되었습니다";
    toast(review.status === "rejected" ? "라이브러리에서 제외했습니다. 제외 목록에서 복구할 수 있습니다." : "검수 결과를 저장했습니다.");
    await Promise.all([refreshSummary(), loadPage()]);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
});

$("category-filter").value = initialCategory;
$("search").value = state.q;
$("status-filter").value = state.status;
Promise.all([refreshSummary(), loadPage()]).then(async () => {
  const key = new URLSearchParams(location.search).get("pose");
  if (key) {
    await openDetail(key);
    const angles = Object.fromEntries(['yaw','pitch','roll'].map(name => [name, Number(initialParams.get(name))]));
    if (['yaw','pitch','roll'].every(name => initialParams.has(name) && Number.isFinite(angles[name]))
        && Math.abs(angles.yaw) <= 180 && Math.abs(angles.pitch) <= 90 && Math.abs(angles.roll) <= 180) {
      if (initialParams.get('source_hash') !== activePose.content_hash || activePose.excluded) {
        toast('검색 후 포즈가 변경되었습니다. 러프 검색을 다시 실행해 주세요.');
      } else {
        orientationPreview.applySuggestion(angles);
      }
    }
  }
}).catch((error) => toast(error.message));
setInterval(async () => {
  if (!renderInProgress || document.hidden || $("detail-dialog").open) return;
  try {
    await refreshSummary();
    await loadPage();
  } catch (error) { toast(error.message); }
}, 10000);
