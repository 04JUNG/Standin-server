/** Binary solved pose state reader (standin-binary-f64le-v2), shared with BFF and Client.
 * Byte-for-byte contract: converter/solved_state_v2.py. Same validation as the
 * typed-JSON v1 reader, without per-float JSON objects: numeric sections are
 * little-endian Float64/Uint32/Uint16 arrays that are returned as typed views.
 * The input is copied once; returned typed arrays must be treated as read-only.
 */
export const ENCODING = 'standin-binary-f64le-v2';
export const MEDIA_TYPE = 'application/vnd.standin.solved-pose-state-v2';
const MAX_BYTES = 64 * 1024 * 1024;
const MAX_HEADER = 4 * 1024 * 1024;
const STATE_MAGIC = 'STNDSPS2';
const MODEL_MAGIC = 'STNDMDL2';
const MAX_VERTICES = 500000, MAX_NORMALS = 3000000, MAX_TRIANGLES = 1000000, MAX_INFLUENCES = 32;
const ascii = new TextDecoder('ascii', { fatal: true });

if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw Error('little-endian platform required');

async function sha(view) {
  const digest = await crypto.subtle.digest('SHA-256', view);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}
const align = n => (n + 7) & ~7;
function untyped(value) {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const keys = Object.keys(value);
    if (keys.length === 1 && keys[0] === '$f64') {
      if (typeof value.$f64 !== 'string' || !/^[0-9a-f]{16}$/.test(value.$f64)) throw Error('invalid float');
      const bytes = Uint8Array.from(value.$f64.match(/../g), n => parseInt(n, 16));
      const number = new DataView(bytes.buffer).getFloat64(0, false);
      if (!Number.isFinite(number)) throw Error('nonfinite float');
      return number;
    }
    const out = {};
    for (const k of keys) {
      if (k === '__proto__' || k === 'constructor' || k === 'prototype') throw Error('reserved key');
      out[k] = untyped(value[k]);
    }
    return out;
  }
  if (Array.isArray(value)) return value.map(untyped);
  if (typeof value === 'number' && !Number.isSafeInteger(value)) throw Error('plain numbers must be safe integers');
  return value;
}
// Python json.dumps(sort_keys=True, ensure_ascii=True, separators=(',', ':')).
function canonical(value) {
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(k =>
    canonical(k) + ':' + canonical(value[k])).join(',') + '}';
  return JSON.stringify(value).replace(/[\u0080-￿]/g, c => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0'));
}
function zero(bytes, start, end) {
  if (end > bytes.length) throw Error('binary state truncated');
  for (let i = start; i < end; i++) if (bytes[i] !== 0) throw Error('binary state padding is not canonical');
}
function header(bytes, offset, magic) {
  if (offset + 12 > bytes.length || ascii.decode(bytes.subarray(offset, offset + 8)) !== magic) throw Error('binary state magic mismatch');
  const size = new DataView(bytes.buffer, bytes.byteOffset).getUint32(offset + 8, true);
  if (!(size > 0 && size <= MAX_HEADER) || offset + 12 + size > bytes.length) throw Error('invalid binary header size');
  const text = ascii.decode(bytes.subarray(offset + 12, offset + 12 + size));
  const typed = JSON.parse(text);
  if (canonical(typed) !== text) throw Error('binary header is not canonical');
  const end = offset + 12 + size;
  zero(bytes, end, align(end));
  return [untyped(typed), align(end)];
}
const int = (v, lo, hi, what) => { if (!Number.isSafeInteger(v) || v < lo || v > hi) throw Error('invalid ' + what); return v; };
function sameKeys(value, keys) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).length === keys.length && keys.every(k => Object.hasOwn(value, k));
}
function checkMeta(meta) {
  if (!sameKeys(meta, ['schema', 'coordinates', 'capabilities', 'bones', 'meshes']) || meta.schema !== 'interactive-pose-model-v1')
    throw Error('invalid binary model header');
  const c = meta.coordinates;
  if (!sameKeys(c, ['space', 'originBlender', 'matrixLayout', 'baked']) || c.space !== 'right-handed-y-up-metres-hip-origin'
      || c.matrixLayout !== 'row-major' || c.baked !== 'rigged_rest' || !Array.isArray(c.originBlender)
      || c.originBlender.length !== 3 || !c.originBlender.every(Number.isFinite)) throw Error('invalid model coordinates');
  if (!sameKeys(meta.capabilities, ['localHands']) || meta.capabilities.localHands !== false)
    throw Error('capability requires a separately validated variant table');
  if (!Array.isArray(meta.bones) || meta.bones.length < 1 || meta.bones.length > 256) throw Error('invalid bone count');
  const names = new Set();
  meta.bones.forEach((b, i) => {
    if (!sameKeys(b, ['name', 'parent']) || typeof b.name !== 'string' || !b.name || names.has(b.name)) throw Error('invalid bone record');
    names.add(b.name); int(b.parent, -1, i - 1, 'bone parent');
  });
  if (!Array.isArray(meta.meshes) || meta.meshes.length < 1 || meta.meshes.length > 32) throw Error('invalid mesh count');
  const meshNames = new Set(); let total = 0;
  for (const m of meta.meshes) {
    if (!sameKeys(m, ['name', 'vertices', 'normals', 'triangles', 'influences']) || typeof m.name !== 'string'
        || !m.name || meshNames.has(m.name)) throw Error('invalid mesh fields');
    meshNames.add(m.name);
    total += int(m.vertices, 1, MAX_VERTICES, 'vertex count');
    int(m.normals, 1, MAX_NORMALS, 'normal count'); int(m.triangles, 1, MAX_TRIANGLES, 'triangle count');
    int(m.influences, m.vertices, m.vertices * MAX_INFLUENCES, 'influence count');
    if (total > MAX_VERTICES) throw Error('invalid mesh bounds');
  }
}
function finite(array, what) { for (let i = 0; i < array.length; i++) if (!Number.isFinite(array[i])) throw Error('nonfinite ' + what); }
function below(array, limit, what) { for (let i = 0; i < array.length; i++) if (array[i] >= limit) throw Error('invalid ' + what); }

function readModel(bytes, offset) {
  const [meta, start] = header(bytes, offset, MODEL_MAGIC);
  checkMeta(meta);
  let at = start;
  const take = (Type, count) => {
    const end = at + Type.BYTES_PER_ELEMENT * count;
    if (end > bytes.length) throw Error('binary state truncated');
    const view = new Type(bytes.buffer, bytes.byteOffset + at, count);
    zero(bytes, end, align(end)); at = align(end);
    return view;
  };
  const B = meta.bones.length;
  const matrices = take(Float64Array, B * 48); finite(matrices, 'bone matrix');
  const bones = meta.bones.map((b, i) => Object.freeze({ name: b.name, parent: b.parent,
    restLocal: matrices.subarray(i * 48, i * 48 + 16), poseLocal: matrices.subarray(i * 48 + 16, i * 48 + 32),
    inverseBind: matrices.subarray(i * 48 + 32, i * 48 + 48) }));
  const meshes = meta.meshes.map(m => {
    const V = m.vertices, N = m.normals, T = m.triangles, W = m.influences;
    const positions = take(Float64Array, V * 3), normals = take(Float64Array, N * 3);
    const triangles = take(Uint32Array, T * 3), normalTriangles = take(Uint32Array, T * 3);
    const weightOffsets = take(Uint32Array, V + 1), weightBones = take(Uint16Array, W), weightValues = take(Float64Array, W);
    finite(positions, 'surface'); finite(normals, 'surface');
    below(triangles, V, 'surface triangle'); below(normalTriangles, N, 'normal triangle');
    if (weightOffsets[0] !== 0 || weightOffsets[V] !== W) throw Error('invalid skin weight offsets');
    for (let v = 0; v < V; v++) {
      const s = weightOffsets[v], e = weightOffsets[v + 1];
      if (!(e > s && e - s <= MAX_INFLUENCES)) throw Error('unbound or excessive vertex influences');
      let sum = 0;
      for (let j = s; j < e; j++) {
        const w = weightValues[j];
        if (weightBones[j] >= B || (j > s && weightBones[j] <= weightBones[j - 1])) throw Error('invalid skin weight');
        if (!(Number.isFinite(w) && w > 0 && w <= 1)) throw Error('invalid skin weight');
        sum += w;
      }
      if (Math.abs(sum - 1) > 1e-4) throw Error('skin weights are not normalized');
    }
    return Object.freeze({ name: m.name, vertexCount: V, normalCount: N, triangleCount: T, positions, normals,
      triangles, normalTriangles, weightOffsets, weightBones, weightValues });
  });
  if (at !== bytes.length) throw Error('binary model has trailing bytes');
  return Object.freeze({ schema: meta.schema, coordinates: Object.freeze({ ...meta.coordinates,
    originBlender: Object.freeze([...meta.coordinates.originBlender]) }),
    capabilities: Object.freeze({ ...meta.capabilities }), bones: Object.freeze(bones), meshes: Object.freeze(meshes) });
}

function deepFreeze(value) {
  if (value && typeof value === 'object' && !ArrayBuffer.isView(value)) {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}

/** Verify SHA256(bytes) === expectedSha256, then decode and validate. */
export async function readSolvedStateV2(input, expectedSha256) {
  if (!(input instanceof Uint8Array) || !input.byteLength || input.byteLength > MAX_BYTES) throw Error('invalid byte budget');
  // Private copy at byteOffset 0 (8-byte aligned). Not input.slice(): on a Node Buffer
  // that returns a view into a possibly misaligned shared pool instead of a copy.
  const bytes = new Uint8Array(input.byteLength);
  bytes.set(input);
  if (await sha(bytes) !== expectedSha256) throw Error('state hash mismatch');
  const [head, modelOffset] = header(bytes, 0, STATE_MAGIC);
  if (!sameKeys(head, ['encoding', 'schema', 'identity', 'safety', 'dependencies']) || head.encoding !== ENCODING
      || head.schema !== 'solved-pose-state-v1' || head.safety?.finalFileVerified !== false
      || head.safety?.bodyAccepted !== true || typeof head.safety?.handChecksPassed !== 'boolean'
      || head.safety?.surface !== 'existing-adapter-checks-only' || Object.keys(head.safety).length !== 4
      || head.dependencies?.initialFbxRequired !== true
      || head.dependencies?.boundary !== 'post-frozen-body-pre-final-export-v1' || Object.keys(head.dependencies).length !== 2)
    throw Error('unsupported solved state');
  const model = readModel(bytes, modelOffset);
  const modelSha256 = await sha(bytes.subarray(modelOffset));
  const { encoding, ...rest } = head;
  return Object.freeze({ ...deepFreeze(rest), model, stateHash: expectedSha256, modelSha256, byteLength: bytes.length });
}

/** JSON.stringify of the [[a,b,c],...] triangle list, for existing skin pins. */
export function triangleIndexJson(triangles) {
  const parts = new Array(triangles.length / 3);
  for (let t = 0, i = 0; i < triangles.length; t++, i += 3) parts[t] = '[' + triangles[i] + ',' + triangles[i + 1] + ',' + triangles[i + 2] + ']';
  return '[' + parts.join(',') + ']';
}

/** The equivalent typed-JSON v1 model shape (nested arrays), e.g. for compatibility tests. */
export function toV1Model(model) {
  const triples = a => Array.from({ length: a.length / 3 }, (_, i) => [a[i * 3], a[i * 3 + 1], a[i * 3 + 2]]);
  return {
    schema: model.schema, coordinates: { ...model.coordinates, originBlender: [...model.coordinates.originBlender] },
    capabilities: { ...model.capabilities },
    bones: model.bones.map(b => ({ name: b.name, parent: b.parent, restLocal: [...b.restLocal],
      poseLocal: [...b.poseLocal], inverseBind: [...b.inverseBind] })),
    meshes: model.meshes.map(m => ({ name: m.name, positions: triples(m.positions), normals: triples(m.normals),
      normalTriangles: triples(m.normalTriangles), triangles: triples(m.triangles),
      weights: Array.from({ length: m.vertexCount }, (_, v) => {
        const out = [];
        for (let j = m.weightOffsets[v]; j < m.weightOffsets[v + 1]; j++) out.push([m.weightBones[j], m.weightValues[j]]);
        return out;
      }) })),
  };
}
