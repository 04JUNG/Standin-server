# Actual final-FBX CSP evidence, 2026-11-06

The current scope is pre-posed individual FBX export → CSP automatic import. `.clip` and posing inside CSP are outside this app feature. Screenshots are from CSP EX5.1.1 build202607291835, new1000×76872dpi canvases, no native outline/culling setting changes. Exact Basic/Realistic property was not read.

- [Neutral0/2 comparison](csp-neutral.png): latest approved male/female final implementation files.
- [Posed0/2 comparison](csp-posed.png): exact final male foot-tap/female crouch; pose and grey body retained with black contours.
- [Observed final-file views](csp-orbit.png): exact final male front/near-side/near-back, female rear-three-quarter. No exact-angle/elevation/all-view guarantee.

Only the test canvas is cropped; images are resized for layout without retouching model pixels. [Source screenshot hashes/crops](evidence.json) let the local originals be audited. No source models, private credentials or unrelated document UI are included.

| Final FBX | SHA-256 |
|---|---|
| Male neutral2 | `02799598d9f60899e83627446c80bf9c6f7f737a47b8b2de7e7f92612200a4e3` |
| Female neutral2 | `07295db35bb6a148f14da4fad91d459ee05d1f5859225b29638b71dce9754e3b` |
| Male foot-tap2 | `81a559fb4279440858a62a73f7abb6c3304039c01095ed3160c772d7ffcaa84d` |
| Female crouch2 | `c47f586d0bebfeaa05f2334338734fe0d0de7b315c2d490d0a3aee5d3e35aa5e` |
| HTTP male full.65 | `0d7eb8befd90ccc3203265c3527e85c2cb6ce4c222552cc2d24af7eb638cd87b` |
| HTTP female full.65 | `e6bd5301075e5c5ce5240b4f081be65cc3e80cbb7a35bde1fd7a8c7b24357f33` |

Both newly generated HTTP .65 files also passed actual initial front-view CSP import with grey surfaces/thin black contours; no pixel-width/all-view claim. Their raw photos and result hashes are in the shared client-owned midpoint-results.json.

Fine dotted native occlusion lines occur near calf/knee/shin; no large black-filled body/obvious white halo/doubled silhouette was seen in these views. Original zero files already have some native contours. Current support is limited to the [exact capability matrix](../../OUTLINE_FBX_CSP.md); these photos do not extend it to other models or arbitrary views.
