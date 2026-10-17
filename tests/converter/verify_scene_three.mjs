// node verify_scene_three.mjs <three package root> <source A> <source B> <verified output dir>
// Run after verify_scene_roundtrip.py. No browser, DOM, or WebGL is required.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [threeRoot, sourceA, sourceB, outputDir] = process.argv.slice(2);
const { FBXLoader } = await import(pathToFileURL(path.resolve(threeRoot, 'examples/jsm/loaders/FBXLoader.js')));
const { Vector3 } = await import(pathToFileURL(path.resolve(threeRoot, 'build/three.module.js')));

function load(filename) {
  const bytes = fs.readFileSync(filename);
  const model = new FBXLoader().parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), '');
  // FBXLoader leaves FBX centimeter coordinates unchanged. Normalize the whole cut once.
  model.scale.setScalar(0.01);
  model.updateMatrixWorld(true);
  return model;
}

function meshes(model) {
  const result = [];
  model.traverse(o => { if (o.isSkinnedMesh) result.push(o); });
  return result;
}

const sources = [load(sourceA), load(sourceB)];
const checks = [];
for (const [name, indices] of [['single', [0]], ['two_people', [0, 1]], ['three_repeated', [0, 0, 1]]]) {
  const model = load(path.join(outputDir, `${name}.fbx`));
  const manifest = JSON.parse(fs.readFileSync(path.join(outputDir, `${name}.json`)));
  const outputMeshes = meshes(model);
  assert.equal(outputMeshes.length, indices.reduce((n, i) => n + meshes(sources[i]).length, 0));
  let maxError = 0;
  for (const [i, sourceIndex] of indices.entries()) {
    const person = manifest.persons[i];
    const prefix = `person_${String(person.person_index).padStart(4, '0')}_`;
    const root = model.getObjectByName(`${prefix}placement`);
    assert.ok(root);
    assert.ok(root.getWorldPosition(new Vector3()).distanceTo(new Vector3(...person.position)) < 0.0001);
    const angle = person.rotation_y_degrees * Math.PI / 180;
    for (const source of meshes(sources[sourceIndex])) {
      const output = model.getObjectByName(prefix + source.name);
      assert.ok(output?.isSkinnedMesh);
      // Blender may include existing unweighted terminal nodes in the skin cluster
      // array. Check the actual bone hierarchy and world geometry instead.
      const sourceBones = [];
      const outputBones = [];
      sources[sourceIndex].traverse(o => { if (o.isBone) sourceBones.push(o.name); });
      root.traverse(o => { if (o.isBone) outputBones.push(o.name); });
      assert.deepEqual(outputBones.sort(), sourceBones.sort());
      assert.equal(output.geometry.attributes.position.count, source.geometry.attributes.position.count);
      for (let vertex = 0; vertex < source.geometry.attributes.position.count; vertex++) {
        const original = source.getVertexPosition(vertex, new Vector3()).applyMatrix4(source.matrixWorld);
        const expected = original.multiplyScalar(person.scale).applyAxisAngle(new Vector3(0, 1, 0), angle)
          .add(new Vector3(...person.position));
        const actual = output.getVertexPosition(vertex, new Vector3()).applyMatrix4(output.matrixWorld);
        maxError = Math.max(maxError, expected.distanceTo(actual));
      }
    }
  }
  assert.ok(maxError < 0.0001, `world geometry error ${maxError}`);
  checks.push({ case: name, persons: indices.length, max_world_error_m: maxError });
}
const report = { ok: true, loader: 'Three.js FBXLoader', version: JSON.parse(fs.readFileSync(path.join(threeRoot, 'package.json'))).version, cases: checks };
fs.writeFileSync(path.join(outputDir, 'three-verification.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
