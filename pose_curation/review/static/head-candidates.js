import {request} from './api.js?v=qa-20261002';

// Inference runs offline. This component only selects revision-bound local results.
export class FaceCandidates {
  constructor({onStart, onSelected}) {
    this.onStart = onStart;
    this.onSelected = onSelected;
    this.listSequence = 0;
    this.applySequence = 0;
    this.container = document.getElementById('face-candidates');
    this.status = document.getElementById('face-candidates-status');
  }

  cancelApply() {
    ++this.applySequence;
    this.applyController?.abort();
  }

  async load(query, image) {
    const token = ++this.listSequence;
    this.listController?.abort(); this.cancelApply();
    this.container.replaceChildren();
    this.query = query; this.image = image; this.items = [];
    if (!query || query.excluded) {
      this.status.textContent = query?.excluded ? '제외한 이미지는 자동 후보를 사용하지 않습니다.' : '';
      return;
    }
    this.status.textContent = '확대 검사한 얼굴 후보를 불러옵니다…';
    this.listController = new AbortController();
    try {
      const result = await request(`/api/head/queries/${encodeURIComponent(query.key)}/candidates`, {signal:this.listController.signal});
      if (token !== this.listSequence) return;
      this.items = result.items;
      this.status.textContent = result.items.length ? '참고 사진과 다른 인물이 섞여 있을 수 있습니다. 원하는 그림 얼굴의 버튼을 눌러 각도를 적용하세요.' : '안정적으로 검출한 얼굴이 없습니다. 아래에서 기준점을 지정하거나 직접 각도를 맞추세요.';
      this.render();
    } catch(error) {
      if (token === this.listSequence && error.name !== 'AbortError') this.status.textContent = '자동 후보를 사용할 수 없습니다. 기준점 또는 수동 각도로 진행할 수 있습니다.';
    }
  }

  render() {
    this.container.replaceChildren();
    if (!this.image?.complete || !this.image.naturalWidth) return;
    this.items.forEach((item,index) => {
      const card = document.createElement('article'); card.className = 'face-candidate';
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 180;
      canvas.setAttribute('role','img'); canvas.setAttribute('aria-label',`얼굴 후보 ${index+1}`);
      const [x0,y0,x1,y1] = item.bbox, side = Math.max(x1-x0,y1-y0)*1.4;
      const ctx = canvas.getContext('2d'); ctx.fillStyle = '#eef0eb'; ctx.fillRect(0,0,180,180);
      ctx.drawImage(this.image,(x0+x1-side)/2,(y0+y1-side)/2,side,side,0,0,180,180);
      const label = document.createElement('p'); label.textContent = `후보 ${index+1} · ${item.detector}`;
      const button = document.createElement('button'); button.className = 'quiet-button';
      button.textContent = `후보 ${index+1} 얼굴 각도 사용`;
      button.disabled = item.status !== 'suggested';
      button.addEventListener('click', () => this.apply(item,button));
      card.append(canvas,label,button);
      if (item.reasons.length) { const reason = document.createElement('p'); reason.className = 'preview-caption'; reason.textContent = item.reasons.join(' '); card.append(reason); }
      this.container.append(card);
    });
  }

  async apply(item, button) {
    this.onStart();
    const token = ++this.applySequence;
    this.applyController = new AbortController();
    button.disabled = true;
    this.status.textContent = '선택한 얼굴의 관측·검수 버전을 확인하고 있습니다…';
    try {
      const result = await request('/api/head/suggest',{method:'POST',headers:{'Content-Type':'application/json','X-Pose-Review':'1'},
        body:JSON.stringify({query:this.query.key,content_hash:this.query.content_hash,candidate:item.id,target_confirmed:true}),signal:this.applyController.signal});
      if (token !== this.applySequence) return;
      this.status.textContent = '선택한 얼굴의 참고 각도를 적용했습니다. 실제 모델을 생성해서 형태를 확인하세요.';
      this.onSelected(result);
    } catch(error) {
      if (token === this.applySequence && error.name !== 'AbortError') this.status.textContent = error.message;
    } finally { button.disabled = false; }
  }
}
