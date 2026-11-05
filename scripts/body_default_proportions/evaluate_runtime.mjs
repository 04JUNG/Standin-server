// Bundle against the exact client checkout being reviewed, with esbuild aliases
// standin-runtime-format and standin-runtime-evaluate pointing to its TS files.
import fs from 'node:fs/promises';
import { parseShapeRuntime } from 'standin-runtime-format';
import { resolveControls, evaluateShape } from 'standin-runtime-evaluate';

const [runtimePath, casePath, outputPath] = process.argv.slice(2);
const raw = await fs.readFile(runtimePath);
const runtime = parseShapeRuntime(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const input = JSON.parse(await fs.readFile(casePath, 'utf8'));
const controls = resolveControls(runtime, 'BASE', input);
const surface = evaluateShape(runtime, controls, undefined, 'BASE');
await fs.writeFile(outputPath, new Uint8Array(surface.buffer, surface.byteOffset, surface.byteLength));
console.log(JSON.stringify({kind:runtime.kind, source:runtime.header.source, rules:runtime.header.rules,
  preset:'BASE', controls, vertexCount:runtime.header.vertexCount}));
