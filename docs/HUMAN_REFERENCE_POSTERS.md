# Actual-human reference posters and shared mesh fallback

The original poster API used a joint drawing when assets were absent, and inline/generated candidates never requested a body poster. Those were incomplete human previews. The new path prefers a verified static PNG; missing or pose-only responses use one shared actual default-human mesh template and the exact owned BVH in the browser. It never starts a card-triggered Converter job. Direct BVH review consumers also upgrade a valid pose-only response rather than returning a skeleton as completed preview.

## Reference quality contract

`reference-skinning-v1` uses the approved `standin-master-v2` FBX, SHA `7c648b97a24a3bb4914b6e5d515708c33727979881d92ef916d5726e22301f3d`. The full template has 24,988 vertices; all eight original canonical skin influences are preserved (maximum discarded weight zero). The five anatomical crops are produced offline with the existing skin-region framing code. A raised leg cannot enter an upper-body crop by screen height.

Reference skinning applies the actual source frame-zero world rotations with rest-direction alignment onto the fixed default anatomy. It does **not** run the frozen Converter's correction/IK/hand/contact/final-export pipeline and must be labeled **기본 체형 · 근사 포즈 참고**. Camera is canonical front/orthographic; selected body, shape, hands and search camera are unapplied. Existing frozen posters remain separately accepted. These references never set `actual_mesh_validated`, release, refine or export permissions. Quarantine and exact source checks remain in front of serving.

Poster manifests bind method, source BVH, scope, template hash, approved character, renderer revision and PNG hash. The renderer revision pins the actual client skinning/framing/style source hashes in `config/human_reference_renderer.v1.json`. The paired client has a matching manifest regression test. Renderer changes require updated pins and a fresh bake, not a metadata-only re-label.

## Offline workflow

1. Snapshot the exact deployed library object version, DB and BVH bytes. Keep production and staging libraries separate.
2. In pinned Blender 5.2.0, run `scripts/build_default_body_template.py --character /approved/default.fbx --out /bundle/human-templates` (after Blender's `--`). The character SHA and Blender build are checked. Match the output against `config/default_body_template.v1.json`; review changed pins before release.
3. Run `python scripts/audit_pose_posters.py --library /bundle --out /audit`. It emits the source inventory and per-scope identities without changing the library.
4. From the client repo, run `node scripts/bakeHumanPosters.mjs --library /bundle --inventory /audit/inventory.json --identities /audit/identities.json --templates /bundle/human-templates --out /bundle/pose-posters --renderer-config /server/config/human_reference_renderer.v1.json`. Open the printed loopback UI and start/resume. The local writer allows only inventoried source hashes/scopes, checks the renderer/template pins, requires its session nonce for writes and writes atomically. It makes no Converter, cloud or deployment calls.
5. Repeat the audit with `--require-complete`. It verifies every source/scope/PNG/template hash, image decoding and non-empty foreground. `deploy_pose_library.make_archive` independently validates coverage and packages only expected assets, excluding progress/temp files. A present incomplete poster set fails packaging.

`GET /pose-poster-template/{scope}` serves the pinned raw gzip template with digests and preview-only/method/scope/character headers. The BFF verifies compressed bytes, then uses native HTTP gzip with a decoded-JSON digest so the application does not require the JavaScript DecompressionStream API. Runtime inference requires no Blender for this path.

## Verified local snapshots, 2026-10-19

| Snapshot | Poses | Metadata-tagged mirrors | Verified images |
|---|---:|---:|---:|
| Staging `lib-20261018-8c045abe` | 5,831 | 1,900 | 29,155 |
| Production `lib-20261016-33715cc9` | 5,809 | 1,889 | 29,045 |

Both have zero missing/corrupt images and five pinned templates. Every PNG decoded; minimum foreground fraction was 5.067%. Staging and production differ at 147 production source hashes, so those 735 images were baked separately; source DBs were preserved. These are local full-coverage preparations, not remote rollout claims.

Staging package SHA-256: `5a14eafad393359762ab5beda104eaeb99ed1a306af2a5240d915f883a5f35ba` (884,446,372 bytes). Production-local package: `a8d93e091309d0be8e8d2a622824270a0ec05523db9748b622130bb5009ba32b` (880,242,121 bytes). Archive contents were independently scanned; no progress/temp files were included. No cloud upload, service restart, production write or native app operation was performed.

Visual checks covered original/actual mirror, generated opposite-hand and CMU source poses in all five scopes (25 decoded real meshes). Common-component local fixtures covered normal, near/geometric/composition, actual mirror, inline/generated and missing-poster paths, with five decoded humans, zero Converter calls and no additional calls after body change. Real Node HTTP checks passed for five PNG, five native-gzip template, five owned-source and CORS boundaries. These are isolated local fixtures; deployed/installed staging acceptance remains separate.
