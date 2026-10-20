import { request, thumbnailUrl } from './api.js?v=qa-20261002';

const labels = { full: '전신', half: '반신', bust: '흉상', head: '두상' };

// Owns only preview scope/loading state; decisions and skeleton remain separate.
export class FramedPreview {
  constructor(onChange) {
    this.onChange = onChange;
    this.scope = 'full';
    this.sequence = 0;
    document.getElementById('scope-switch').addEventListener('click', event => {
      const button = event.target.closest('[data-scope]');
      if (button) this.select(button.dataset.scope);
    });
    document.getElementById('scope-retry').addEventListener('click', () => this.select(this.scope));
  }

  setPose(pose) {
    this.pose = pose;
    this.select(this.scope);
  }

  stop() {
    ++this.sequence;
    clearTimeout(this.timer);
  }

  url(view) {
    if (this.scope === 'full') return this.pose.preview_kind === 'character' ? thumbnailUrl(this.pose, view) : null;
    return this.result?.status === 'ready'
      ? `/api/poses/${this.pose.key}/framing/${this.scope}/${view}?v=${this.result.version}` : null;
  }

  update() {
    for (const button of document.querySelectorAll('#scope-switch button')) {
      const selected = button.dataset.scope === this.scope;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', String(selected));
    }
    const failed = this.result?.status === 'failed';
    document.getElementById('scope-retry').hidden = !failed;
    document.getElementById('scope-status').textContent = this.scope === 'full'
      ? '전신 미리보기 · BVH와 3D 골격은 항상 전신입니다.'
      : failed ? this.result.error
      : this.result?.status === 'ready'
        ? `${labels[this.scope]} · 실제 메시 절단 결과입니다. 소품은 포함하지 않습니다. BVH와 3D 골격은 전신을 유지합니다.`
        : `${labels[this.scope]} 4방향 미리보기 생성 중… 처음 생성할 때 잠시 걸립니다.`;
    this.onChange();
  }

  async select(scope) {
    this.stop();
    this.scope = scope;
    this.result = null;
    this.update();
    if (scope === 'full' || !this.pose) return;
    const sequence = this.sequence;
    const url = `/api/poses/${this.pose.key}/framing?scope=${scope}`;
    const poll = async (start = false) => {
      try {
        const result = await request(url, start ? { method: 'POST', headers: { 'X-Pose-Review': '1' } } : {});
        if (sequence !== this.sequence) return;
        this.result = result;
        this.update();
        if (result.status === 'rendering') this.timer = setTimeout(poll, 1500);
      } catch (error) {
        if (sequence !== this.sequence) return;
        this.result = { status: 'failed', error: error.message };
        this.update();
      }
    };
    await poll(true);
  }
}
