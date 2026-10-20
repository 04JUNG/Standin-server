import {request} from './api.js?v=qa-20261002';

const $ = id => document.getElementById(id);
const labels = {accepted:'채택', hold:'보류', rejected:'제외'};

// Persistence UI has no fitting logic; the API validates target and preview revisions.
export class HeadAngleReviews {
  constructor({context, restore}) {
    this.context = context; this.restore = restore; this.sequence = 0;
    this.items = []; this.target = null; this.edits = 0;
    $('head-review-save').addEventListener('click', () => this.save());
    $('head-review-reload').addEventListener('click', () => this.load(this.context().query));
    $('head-review-restore').addEventListener('click', () => {
      const row = this.items.find(item => String(item.revision) === $('head-review-records').value);
      if (!row?.current) return;
      this.restore(row);
      $('head-review-decision').value = row.status;
      $('head-review-note').value = row.note;
      $('head-review-status').textContent = '저장한 각도를 불러왔습니다. 미리보기를 다시 생성해 확인하세요.';
    });
    $('head-review-records').addEventListener('change', () => this.updateRestore());
    for (const id of ['head-review-note','head-review-decision']) $(id).addEventListener('input', () => {++this.edits;});
  }

  setTarget(value) {
    if (JSON.stringify(value) !== JSON.stringify(this.target)) {
      $('head-review-note').value = '';
      $('head-review-decision').value = 'hold';
    }
    this.target = value;
    $('head-review-save').disabled = !value || this.context().query?.excluded;
    this.changed();
  }

  changed() {
    ++this.edits;
    $('head-review-confirm').checked = false;
    if (this.target) $('head-review-status').textContent = '현재 얼굴의 각도와 검수 결과를 저장할 수 있습니다.';
  }

  updateRestore() {
    const row = this.items.find(item => String(item.revision) === $('head-review-records').value);
    $('head-review-restore').disabled = !row?.current;
  }

  async load(query) {
    const sequence = ++this.sequence;
    this.controller?.abort(); this.controller = new AbortController();
    this.items = []; $('head-review-records').replaceChildren();
    $('head-review-restore').disabled = true;
    $('head-review-status').textContent = '저장한 검수 기록을 불러옵니다…';
    if (!query) { $('head-review-status').textContent = '러프를 선택해 주세요.'; return; }
    try {
      const result = await request(`/api/head/queries/${encodeURIComponent(query.key)}/angle-reviews`,{signal:this.controller.signal});
      if (sequence !== this.sequence) return;
      this.items = result.items;
      for (const row of result.items) {
        const option = document.createElement('option'); option.value = row.revision;
        const a = row.angles;
        const targetLabel = row.provenance.candidate ? '후보 '+(row.provenance.face_number || '') : row.provenance.region ? '영역 지정' : '기준점 지정';
        option.textContent = `${labels[row.status]} · ${row.scope === 'head' ? '두상' : '흉상'} · ${a.yaw}/${a.pitch}/${a.roll}° · ${targetLabel}${row.current ? '' : ' · 재검수 필요'}`;
        option.title = row.stale_reason || row.note;
        $('head-review-records').append(option);
      }
      this.updateRestore();
      $('head-review-status').textContent = result.items.length ? `얼굴·범위별 최근 기록 ${result.items.length}개. 불러오기를 누르면 저장한 각도로 돌아갑니다.` : '저장한 기록이 없습니다. 얼굴 후보·기준점 또는 얼굴 영역을 지정해 주세요.';
    } catch(error) {
      if (sequence === this.sequence && error.name !== 'AbortError') $('head-review-status').textContent = error.message;
    }
  }

  async save() {
    let context;
    try { context = this.context(); }
    catch(error) { $('head-review-status').textContent = error.message; return; }
    if (!context.query || !this.target || context.query.excluded) return;
    if (['yaw','pitch','roll'].some(name => !$(name+'-number').validity.valid || $(name+'-number').value === '')) {
      $('head-review-status').textContent = '각도 범위 안의 숫자를 입력해 주세요.'; return;
    }
    const same = row => row.scope === context.scope && (this.target.candidate
      ? row.provenance.candidate === this.target.candidate
      : this.target.region
        ? JSON.stringify(row.provenance.region) === JSON.stringify(this.target.region)
        : JSON.stringify(row.provenance.points) === JSON.stringify(this.target.points));
    const previous = this.items.find(same);
    const href = $('angle-fbx').getAttribute('href');
    const previewVisible = !$('angle-image').hidden && $('angle-image').complete && $('angle-image').naturalWidth && !$('angle-downloads').hidden;
    const version = previewVisible && href ? new URL(href,location.origin).searchParams.get('v') : null;
    const status = $('head-review-decision').value;
    const confirmed = $('head-review-confirm').checked;
    if (status === 'accepted' && (!version || !confirmed)) {
      $('head-review-status').textContent = '현재 각도의 미리보기를 생성하고 확인 체크를 해 주세요.'; return;
    }
    const sequence = this.sequence;
    const edits = this.edits;
    $('head-review-save').disabled = true;
    try {
      await request(`/api/head/queries/${encodeURIComponent(context.query.key)}/angle-reviews`, {
        method:'POST',headers:{'Content-Type':'application/json','X-Pose-Review':'1'},
        body:JSON.stringify({content_hash:context.query.content_hash,reference_hash:context.reference.content_hash,
          scope:context.scope,...this.target,angles:context.angles,body_angles:context.bodyAngles || null,shoulder_points:context.shoulderPoints || null,status,note:$('head-review-note').value,
          expected_revision:previous?.revision || 0,preview_version:version,visual_confirmed:confirmed})
      });
      if (sequence !== this.sequence) return;
      await this.load(context.query);
      if (this.context().query?.key === context.query.key) $('head-review-status').textContent = edits === this.edits
        ? `${labels[status]} 저장 완료 · 각도·대상 얼굴·검수 이력을 로컬에 보관했습니다.`
        : '이전 각도의 기록을 저장했습니다. 현재 수정한 각도는 아직 저장하지 않았습니다.';
    } catch(error) {
      if (sequence === this.sequence) $('head-review-status').textContent = error.message;
    } finally { $('head-review-save').disabled = !this.target || this.context().query?.excluded; }
  }
}
