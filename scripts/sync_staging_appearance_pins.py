"""Generate BFF/client immutable skin pins from the reviewed Converter manifest."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bff', type=Path, required=True)
    p.add_argument('--client', type=Path, required=True)
    a = p.parse_args()
    raw = (ROOT / 'converter/skin_assets.v1.json').read_bytes()
    manifest = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    current = json.loads((ROOT / 'converter/appearance_models.v1.json').read_bytes())
    policy = dict(assetVersion=manifest['revision'], assetManifestSha256=sha, models={}, certifiedTargets=[])
    client = []
    for cid, row in manifest['models'].items():
        model = dict(characterSha256=row['character_sha256'], vertices=row['vertices'], bones=row['bones'],
                     polygonIndicesSha256=row['polygon_indices_sha256'], weightsSha256=row['weights_sha256'],
                     trianglesSha256=row['triangles_sha256'], uvSha256=row['uv']['sha256'], uvBytes=row['uv']['bytes'],
                     previewMesh=row['preview_mesh'], previewResolution=row['preview_resolution'],
                     textures={sid: {k: entry[k] for k in ('sha256', 'bytes')} for sid, entry in row['textures'].items()},
                     previewTextures={sid: {k: entry[k] for k in ('sha256', 'bytes')} for sid, entry in row['preview_textures'].items()})
        policy['models'][cid] = model
        for sid, entry in row['textures'].items():
            preview = row['preview_textures'][sid]
            continuous = cid in current and sid == 'flesh'
            for lo, hi in ([(0, 2)] if continuous else [(0, 0), (2, 2)]):
                policy['certifiedTargets'].append(dict(characterId=cid, characterSha256=model['characterSha256'],
                    scope='full', skinId=sid, minOutlineWidth=lo, maxOutlineWidth=hi))
            client.append(dict(manifestSha256=sha, assetVersion=manifest['revision'], characterId=cid,
                characterSha256=model['characterSha256'], skinId=sid,
                uvSha256=model['uvSha256'], uvBytes=model['uvBytes'], textureSha256=entry['sha256'], textureBytes=entry['bytes'],
                previewTextureSha256=preview['sha256'], previewTextureBytes=preview['bytes'],
                previewMesh=model['previewMesh'], vertices=model['vertices'], bones=model['bones'],
                polygonIndicesSha256=model['polygonIndicesSha256'], weightsSha256=model['weightsSha256'],
                trianglesSha256=model['trianglesSha256'], scope='full', **({'continuousOutline': True} if continuous else {})))
    preface = '/** Generated from Converter skin_assets.v1.json; exact model/UV/material pins. */\n'
    (a.bff / 'src/appearance/skin-pins.ts').write_text(preface + 'import type { SkinPolicy } from "./skin-contract.js";\nexport const approvedSkins: SkinPolicy = ' + json.dumps(policy, indent=2) + ';\n')
    target = a.client / 'src/features/pose-viewer/skin/skinAssetPins.ts'
    old = target.read_text()
    declaration = old[:old.index('export const SKIN_ASSET_PINS')]
    if 'continuousOutline?: boolean' not in declaration:
        declaration = declaration.replace('  scope: "full";', '  scope: "full";\n  continuousOutline?: boolean;')
    target.write_text(declaration + 'export const SKIN_ASSET_PINS: readonly SkinAssetPin[] = ' + json.dumps(client, indent=2) + ';\n')
    print('SYNCHRONIZED', len(client), 'skin choices', sha)


if __name__ == '__main__':
    main()
