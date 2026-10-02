import { request } from './api.js?v=qa-20261002';

const $ = id => document.getElementById(id);
const names = ['yaw', 'pitch', 'roll'];

export class OrientationPreview {
  constructor({endpointBase = '/api/poses'} = {}) {
    this.endpointBase = endpointBase;
    this.sequence = 0;
    this.mode = 'views';
    this.saved = new Map();
    $('preview-style').addEventListener('click', event => {
      const button = event.target.closest('[data-style]');
      if (!button) return;
      this.mode = button.dataset.style;
      for (const item of $('preview-style').children) {
        const selected = item === button;
        item.classList.toggle('active', selected);
        item.setAttribute('aria-pressed', String(selected));
      }
      $('standard-preview').hidden = this.mode !== 'views';
      $('orientation-panel').hidden = this.mode !== 'oriented';
    });
    for (const name of names) {
      $(name + '-angle').addEventListener('input', () => {
        $(name + '-value').textContent = $(name + '-angle').value + '°';
        $(name + '-number').value = $(name + '-angle').value;
        this.saved.set(this.contextKey, this.angles());
        this.invalidate();
      });
      $(name + '-number').addEventListener('input', () => {
        const input = $(name + '-number');
        this.invalidate();
        if (!input.validity.valid || input.value === '') {
          $('angle-generate').disabled = true;
          $('angle-status').textContent = '각도 범위 안의 숫자를 입력해 주세요.';
          return;
        }
        $(name + '-angle').value = input.value;
        $(name + '-value').textContent = input.value + '°';
        this.saved.set(this.contextKey, this.angles());
        $('angle-generate').disabled = names.some(key => !$(key + '-number').validity.valid || $(key + '-number').value === '');
      });
    }
    $('angle-presets').addEventListener('click', event => {
      const button = event.target.closest('[data-yaw]');
      if (!button) return;
      this.setAngles({yaw: Number(button.dataset.yaw), pitch: 0, roll: 0});
      this.saved.set(this.contextKey, this.angles());
      this.invalidate();
    });
    $('angle-generate').addEventListener('click', () => this.generate());
    $('rough-file').addEventListener('change', () => {
      if (this.roughUrl) URL.revokeObjectURL(this.roughUrl);
      const file = $('rough-file').files[0];
      this.roughUrl = file ? URL.createObjectURL(file) : null;
      $('angle-rough').hidden = !file;
      if (file) $('angle-rough').src = this.roughUrl;
      else $('angle-rough').removeAttribute('src');
    });
    $('rough-opacity').addEventListener('input', () => {
      $('angle-rough').style.opacity = $('rough-opacity').value / 100;
    });
  }

  setContext(pose, scope) {
    const key = `${pose.key}:${pose.content_hash}:${scope}`;
    if (this.contextKey === key) return;
    this.contextKey = key;
    this.pose = pose;
    this.scope = scope;
    this.setAngles(this.saved.get(key) || {yaw: 0, pitch: 0, roll: 0});
    this.invalidate();
  }

  setAngles(value) {
    for (const name of names) {
      $(name + '-angle').value = value[name];
      $(name + '-number').value = value[name];
      $(name + '-value').textContent = value[name] + '°';
    }
  }

  applySuggestion(value) {
    this.setAngles(value);
    this.saved.set(this.contextKey, this.angles());
    this.invalidate();
    $('preview-style').querySelector('[data-style="oriented"]').click();
    $('angle-status').textContent = '러프에서 추정한 참고 각도입니다. 미리보기·파일 생성 후 형태를 확인해 주세요.';
  }

  angles() {
    return Object.fromEntries(names.map(name => [name, Number($(name + '-angle').value)]));
  }

  stop() {
    ++this.sequence;
    clearTimeout(this.timer);
    this.controller?.abort();
  }

  invalidate() {
    this.stop();
    $('angle-image').hidden = true;
    $('angle-image').removeAttribute('src');
    $('angle-downloads').hidden = true;
    for (const link of $('angle-downloads').querySelectorAll('a')) link.removeAttribute('href');
    $('angle-generate').disabled = false;
    $('angle-status').textContent = '각도를 조절한 뒤 미리보기·파일 생성을 눌러 주세요.';
  }

  async generate() {
    if (!this.pose) return;
    if (names.some(name => !$(name + '-number').validity.valid || $(name + '-number').value === '')) return;
    this.invalidate();
    const sequence = this.sequence;
    const spec = {scope: this.scope, ...this.angles(), content_hash: this.pose.content_hash};
    const endpoint = `${this.endpointBase}/${this.pose.key}/oriented`;
    const params = new URLSearchParams(spec);
    this.controller = new AbortController();
    const signal = this.controller.signal;
    $('angle-generate').disabled = true;
    $('angle-status').textContent = '이 각도의 모델과 미리보기를 생성하고 있습니다…';
    const fail = message => {
      if (sequence !== this.sequence) return;
      $('angle-status').textContent = message;
      $('angle-generate').disabled = false;
    };
    const poll = async (start = false) => {
      try {
        const result = await request(start ? endpoint : `${endpoint}?${params}`, start
          ? {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Pose-Review': '1'}, body: JSON.stringify(spec), signal}
          : {signal});
        if (sequence !== this.sequence) return;
        if (result.status === 'ready') {
          params.set('v', result.version);
          const img = $('angle-image');
          img.onload = () => {
            if (sequence !== this.sequence) return;
            img.hidden = false;
            for (const kind of ['fbx', 'settings']) $('angle-' + kind).href = `${endpoint}/${kind}?${params}`;
            $('angle-downloads').hidden = false;
            $('angle-generate').disabled = false;
            $('angle-status').textContent = `출력 준비 완료 · 좌우 ${spec.yaw}° / 높낮이 ${spec.pitch}° / 기울기 ${spec.roll}°`;
          };
          img.onerror = () => fail('미리보기를 불러오지 못했습니다. 다시 생성해 주세요.');
          img.src = `${endpoint}/preview?${params}`;
        } else if (result.status === 'failed') fail(result.error);
        else if (result.status === 'missing') fail('서버가 재시작되었거나 생성 작업이 없습니다. 다시 생성해 주세요.');
        else this.timer = setTimeout(() => poll(), 1500);
      } catch (error) {
        if (error.name !== 'AbortError') fail(error.message);
      }
    };
    await poll(true);
  }
}
