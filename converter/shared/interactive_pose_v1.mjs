/** Exact server data consumer, shared with Client. No hand kinematics port.
 * Loader must fetch an owner-authorized byte artifact with its expected SHA256.
 * This module verifies bounded canonical bytes and inner state/model bindings.
 * A chosen variant is always local-draft, never an export-ready receipt.
 */
const MAX_BYTES = 64 * 1024 * 1024;
const encoder = new TextEncoder();
const decoder = new TextDecoder('ascii', { fatal: true });
async function sha(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}
function untyped(value) {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    if (Object.keys(value).length === 1 && '$f64' in value) {
      if (!/^[0-9a-f]{16}$/.test(value.$f64)) throw Error('invalid float');
      const bytes = Uint8Array.from(value.$f64.match(/../g), n => parseInt(n, 16));
      const number = new DataView(bytes.buffer).getFloat64(0, false);
      if (!Number.isFinite(number)) throw Error('nonfinite float');
      return number;
    }
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, untyped(v)]));
  }
  if (Array.isArray(value)) return value.map(untyped);
  return value;
}
function deepFreeze(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}
function parse(bytes) {
  if (!(bytes instanceof Uint8Array) || !bytes.byteLength || bytes.byteLength > MAX_BYTES) throw Error('invalid byte budget');
  return untyped(JSON.parse(decoder.decode(bytes)));
}
function wireCanonical(value) {
  if (Array.isArray(value)) return '[' + value.map(wireCanonical).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(k =>
    wireCanonical(k) + ':' + wireCanonical(value[k])).join(',') + '}';
  return JSON.stringify(value).replace(/[\u007f-\uffff]/g, c => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0'));
}
function binding(identity) {
  return Object.fromEntries(Object.entries(identity).filter(([k]) => !['selection', 'hands'].includes(k)));
}
// Equality is structural and independent of object insertion order. Hashing uses
// the original canonical bytes from the server, avoiding JS float formatting.
function same(a, b) {
  if (Object.is(a, b)) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  const ak = Object.keys(a).sort(), bk = Object.keys(b).sort();
  return ak.length === bk.length && ak.every((k, i) => k === bk[i] && same(a[k], b[k]));
}
export async function readSolvedState(bytes, expectedSha256) {
  if (await sha(bytes) !== expectedSha256) throw Error('state hash mismatch');
  const value = parse(bytes);
  if (value.schema !== 'solved-pose-state-v1' || value.model?.schema !== 'interactive-pose-model-v1'
      || value.safety?.finalFileVerified !== false || value.model.capabilities?.localHands !== false)
    throw Error('unsupported solved state');
  return deepFreeze(value);
}
export async function readHandVariants(bytes, expectedSha256) {
  if (await sha(bytes) !== expectedSha256) throw Error('atlas hash mismatch');
  const table = parse(bytes);
  if (table.schema !== 'interactive-hand-variants-v1' || table.capability?.predictionOnly !== true
      || table.capability?.arbitraryPresetCombinations !== false || table.capability?.personalFbx !== false
      || table.capability?.cs3c !== false || Object.keys(table.variants ?? {}).length > 32)
    throw Error('unsupported hand atlas');
  const prepared = {};
  for (const [key, entry] of Object.entries(table.variants)) {
    const state = await readSolvedState(encoder.encode(entry.canonicalState), entry.stateHash);
    if (!same(binding(state.identity), table.binding) || state.identity.hands.librarySha256 !== table.rules.librarySha256
        || JSON.stringify([state.identity.hands.left, state.identity.hands.right]) !== key)
      throw Error('variant binding mismatch');
    if (await sha(encoder.encode(wireCanonical(JSON.parse(entry.canonicalState).model))) !== entry.modelSha256)
      throw Error('variant model hash mismatch');
    prepared[key] = {state, stateHash: entry.stateHash, modelSha256: entry.modelSha256};
  }
  return deepFreeze({...table, prepared});
}
export function selectHandVariant(table, currentIdentity, hands) {
  if (!same(binding(currentIdentity), table.binding)
      || currentIdentity.hands.librarySha256 !== table.rules.librarySha256
      || currentIdentity.hands.rulesVersion !== table.rules.version) throw Error('stale hand atlas');
  const variant = table.prepared[JSON.stringify([hands.left ?? null, hands.right ?? null])];
  if (!variant) return null; // preserve authoritative model and await server.
  return {previewState: 'local-draft', exportState: 'not-requested',
    basisStateHash: variant.stateHash, modelSha256: variant.modelSha256, model: variant.state.model};
}
