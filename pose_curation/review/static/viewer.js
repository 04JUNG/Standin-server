/** A dependency-free, orthographic skeleton inspector. No mesh/retarget claims. */
export class SkeletonViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.joints = [];
    this.reset();
    canvas.addEventListener("pointerdown", (event) => {
      this.drag = [event.clientX, event.clientY];
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!this.drag) return;
      this.yaw += (event.clientX - this.drag[0]) * 0.009;
      this.pitch = Math.max(-1.3, Math.min(1.3, this.pitch + (event.clientY - this.drag[1]) * 0.006));
      this.drag = [event.clientX, event.clientY];
      this.draw();
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
      canvas.addEventListener(name, () => { this.drag = null; });
    }
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.zoom = Math.max(0.45, Math.min(2.8, this.zoom * Math.exp(-event.deltaY * 0.001)));
      this.draw();
    }, { passive: false });
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
  }

  reset() {
    this.yaw = 0.35;
    this.pitch = -0.06;
    this.zoom = 1;
    this.draw();
  }

  setData(data) {
    this.rawData = data;
    this.focusSide = null;
    this.fitData();
  }

  focusHand(side = null) {
    this.focusSide = side;
    this.fitData();
  }

  fitData() {
    const data = this.rawData || { joints: [] };
    if (!data.joints.length) {
      this.joints = [];
      this.reset();
      return;
    }
    const joints = this.handCoordinates(data.joints);
    const focused = this.focusSide
      ? joints.filter((joint) => joint.name === `${this.focusSide}Wrist` || joint.name.startsWith(`${this.focusSide}Hand`))
      : joints;
    if (!focused.length) return;
    const positions = focused.map((joint) => joint.position);
    const min = [0, 1, 2].map((axis) => Math.min(...positions.map((point) => point[axis])));
    const max = [0, 1, 2].map((axis) => Math.max(...positions.map((point) => point[axis])));
    const center = min.map((value, axis) => (value + max[axis]) / 2);
    const scale = Math.max(...max.map((value, axis) => value - min[axis]), 1e-6);
    this.joints = joints.map((joint) => ({
      ...joint,
      position: joint.position.map((value, axis) => (value - center[axis]) / scale),
    }));
    this.floor = (min[1] - center[1]) / scale;
    this.reset();
  }

  handCoordinates(joints) {
    if (!this.focusSide) return joints;
    const position = (suffix) => joints.find((joint) => joint.name === this.focusSide + suffix)?.position;
    const wrist = position("Wrist"), middle = position("HandMiddle1");
    const index = position("HandIndex1"), pinky = position("HandPinky1");
    if (!wrist || !middle || !index || !pinky) return joints;
    const subtract = (a, b) => a.map((value, i) => value - b[i]);
    const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
    const normalize = (v) => v.map((value) => value / Math.max(Math.hypot(...v), 1e-9));
    const right = normalize(subtract(index, pinky));
    const forward = subtract(middle, wrist);
    const up = normalize(forward.map((value, i) => value - dot(forward, right) * right[i]));
    const depth = [right[1] * up[2] - right[2] * up[1], right[2] * up[0] - right[0] * up[2], right[0] * up[1] - right[1] * up[0]];
    return joints.map((joint) => {
      const relative = subtract(joint.position, wrist);
      return { ...joint, position: [dot(relative, right), dot(relative, up), dot(relative, depth)] };
    });
  }

  draw() {
    const { canvas, context: ctx } = this;
    if (!ctx || !canvas.clientWidth || !canvas.clientHeight) return;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    ctx.scale(pixelRatio, pixelRatio);
    const background = ctx.createLinearGradient(0, 0, 0, height);
    background.addColorStop(0, "#eff2eb");
    background.addColorStop(1, "#dfe7dc");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);
    if (!this.joints.length) return;
    const scale = Math.min(width, height) * 0.76 * this.zoom;
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const project = ([x, y, z]) => {
      const horizontal = cy * x + sy * z;
      const depth = -sy * x + cy * z;
      return [width / 2 + horizontal * scale, height / 2 - (cp * y - sp * depth) * scale, sp * y + cp * depth];
    };
    const line = (a, b, color, thickness) => {
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = thickness;
      ctx.lineCap = "round";
      ctx.stroke();
    };
    for (let step = -5; !this.focusSide && step <= 5; step++) {
      const offset = step / 10;
      line(project([offset, this.floor, -0.5]), project([offset, this.floor, 0.5]), "#cbd7c7", 0.7);
      line(project([-0.5, this.floor, offset]), project([0.5, this.floor, offset]), "#cbd7c7", 0.7);
    }
    const projected = this.joints.map((joint) => project(joint.position));
    const edges = this.joints.map((joint, index) => ({ joint, index }))
      .filter(({ joint }) => joint.parent >= 0 && (!this.focusSide || joint.name.startsWith(`${this.focusSide}Hand`)))
      .sort((a, b) => (projected[a.index][2] + projected[a.joint.parent][2]) - (projected[b.index][2] + projected[b.joint.parent][2]));
    for (const { joint, index } of edges) {
      const a = projected[joint.parent], b = projected[index];
      const core = /Spine|Chest|Neck/i.test(joint.name);
      const finger = /Hand(Thumb|Index|Middle|Ring|Pinky)/i.test(joint.name);
      const thickness = finger ? Math.max(.7, scale * (this.focusSide ? .009 : .0035)) : Math.max(2, scale * (joint.end ? 0.008 : core ? 0.027 : 0.018));
      const left = /Left|\.L/.test(joint.name);
      line(a, b, left ? "#456a59" : "#877250", thickness + (finger ? .7 : 2));
      line(a, b, left ? "#8db6a0" : "#d0b68c", thickness);
      if (!joint.end) {
        ctx.beginPath();
        ctx.arc(b[0], b[1], thickness * 0.53, 0, Math.PI * 2);
        ctx.fillStyle = "#e2dcc8";
        ctx.fill();
      }
      if (joint.name.split(":").pop() === "Head") {
        ctx.beginPath();
        ctx.ellipse(b[0], b[1] - scale * 0.013, scale * 0.032, scale * 0.045, 0, 0, Math.PI * 2);
        ctx.fillStyle = "#d5c5a4";
        ctx.fill();
        ctx.strokeStyle = "#9c947e";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
    ctx.fillStyle = "#8e9f8c";
    ctx.font = "10px Segoe UI, sans-serif";
    ctx.fillText(this.focusSide ? `${this.focusSide.toUpperCase()} HAND · BVH JOINTS` : "BVH SKELETON · Y UP", 15, height - 17);
  }
}
