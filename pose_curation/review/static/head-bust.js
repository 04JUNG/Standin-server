import {request} from './api.js?v=qa-20261002';
const $ = id => document.getElementById(id);
const names = ['yaw','pitch','roll'];

// Torso direction is explicit. Two observed shoulders constrain roll only.
export class BustControls {
  constructor({faceAngles, query, changed, pickShoulders}) {
    this.faceAngles = faceAngles; this.query = query; this.changed = changed;
    this.sequence = 0;
    this.points = null;
    $('bust-separate').addEventListener('change', () => {
      if ($('bust-separate').checked) this.setAngles(faceAngles());
      this.points = null;
      this.update(); changed();
    });
    for (const n of names) $('body-'+n).addEventListener('input', () => {this.cancel(); this.points = null; changed();});
    $('body-reset').addEventListener('click', () => {this.cancel(); this.points = null; this.setAngles(faceAngles()); changed();});
    $('body-shoulders').addEventListener('click', () => {this.cancel(); pickShoulders();});
  }
  cancel() { ++this.sequence; this.controller?.abort(); }
  enabled() { return $('head-scope').value === 'bust' && $('bust-separate').checked; }
  update() {
    $('bust-controls').hidden = $('head-scope').value !== 'bust';
    $('body-fields').hidden = !this.enabled();
    this.cancel();
  }
  setAngles(angles) { for (const n of names) $('body-'+n).value = angles[n]; }
  angles() {
    if (!this.enabled()) return null;
    if (names.some(n => !$('body-'+n).validity.valid || $('body-'+n).value === '')) throw new Error('몸통 각도 범위 안의 숫자를 입력해 주세요.');
    return Object.fromEntries(names.map(n => [n,Number($('body-'+n).value)]));
  }
  extraSpec() {
    const a = this.angles();
    return a ? Object.fromEntries(names.map(n => ['body_'+n,a[n]])) : {};
  }
  restore(angles, points = null) {
    this.points = points;
    $('bust-separate').checked = Boolean(angles);
    this.setAngles(angles || this.faceAngles()); this.update();
  }
  async fitShoulders(points) {
    const query = this.query();
    if (!query || !this.enabled()) return;
    this.cancel(); const sequence = this.sequence;
    this.controller = new AbortController();
    try {
      const result = await request('/api/head/shoulder-roll', {
        method:'POST', headers:{'Content-Type':'application/json','X-Pose-Review':'1'},
        body:JSON.stringify({query:query.key,content_hash:query.content_hash,points,body_angles:this.angles()}), signal:this.controller.signal
      });
      if (sequence !== this.sequence) return;
      this.setAngles(result.body_angles); this.points = result.points; this.changed();
      $('body-status').textContent = '어깨선의 화면 기울기를 적용했습니다. 몸통 좌우·높낮이는 직접 맞추고 목 모양을 확인해 주세요.';
    } catch(error) { if (sequence === this.sequence && error.name !== 'AbortError') $('body-status').textContent = error.message; }
  }
}
