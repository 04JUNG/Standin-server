# V5 P1 actual fitting and development probe

Worktree: /private/tmp/standin-refine-v5-20261106/p1 (codex/refine-v5-p1).
Base: develop 160f33d57 + P0 dependency commits. Private run:
/private/tmp/standin-refine-v5-20261106/. P0 and P1 are isolated from the original
dirty checkout. No production/API/solver/retarget files were edited.

## What ran

A small supervised **linear ridge residual baseline**, with 24x24 gray pixel
features and current bone directions, was fit to actual 001/074 rough+B0→GT pairs.
B0 was reconstructed by real RTMPose and geometry Top1 without opening GT.
Targets are four arm bone directions, not full 3D pose or hands. No semantic
vision encoder, P2 learned structural proposals or complete v5 solver is present.
Checkpoints contain learned weights, not a case-ID→FBX lookup table. A separate
inference CLI accepts only checkpoint, rough and source state; GT fields reject.

Both learned candidates were actually saved as FBX and reimported. Direction
error remains below 0.001 degrees, with arm-length drift below 0.0004mm. This is
in-sample fitting sanity, not 100% pose accuracy. Hip-normalized joint NME:

| Training case | Before | Saved candidate |
|---|---:|---:|
| 001 | 0.33312 | 0.20354 |
| 074 | 0.69087 | 0.13866 |

Native before/after/GT were rendered at the same fixed frontal camera and
visually inspected. Arm arrangement improves; hand orientation, torso/head,
shoulder positions and unmodified legs still differ. The frontal camera is a
diagnostic inspection camera and is not certified as the rough's camera.

## Twelve-case development probe

Same checkpoint, with no additional fitting, ran on
001–010, 041, 074. All 12 baselines and 24 auto/oracle outputs were saved and
reimported. The oracle directly supplies **GT 3D arm directions**; it is not a
manual visible-2D oracle and is not automatic recognition performance.

| Four-bone mean angle | Before | After |
|---|---:|---:|
| All 12, including 2 fitted | 51.09° | 45.59° |
| Ten not used for fitting | 52.01° | **54.71°** |
| GT 3D direction oracle, 12 | 51.09° | <0.001° |

Only 4/10 non-fitted cases improve in angle. Their joint NME also worsens
0.80369→0.83170. This is an exploratory development probe, not a sealed holdout.
Even the oracle's joint NME remains 0.34352 (before 0.75507). Matching directions
alone leaves torso/shoulder origins, lengths and body alignment unresolved.

## Interpretation and remaining work

The learning→native-application→save/reimport path works for the trained output
scope. The pixel ridge baseline does not generalize sufficiently. Full-pose
similarity is not achieved. Next engine work needs torso/shoulder anchors and
hand orientation, plus a stronger rough-conditioned learned representation.

P0 artist/scene/exposure grouping and visibility/camera review remain pending;
there is no certified split. The planned manual **2D** target comparison has not
been performed. The 3D oracle must remain labeled separately. Mesh/contact,
protected-branch/world-frame audits and production angle budgets were not
certified; outputs are diagnostic `candidate-not-for-service.fbx` only.

Machine metrics/checkpoint lineage: evaluation/refine_v5/p1-summary.json.
Private visual: /private/tmp/standin-refine-v5-20261106/review.png.
No paid providers, new large model installation, cloud evaluation or deployment.

## Reproduce

P0 downloader/native audit/labels must run first. Use the project's Python 3.12
venv with numpy, Pillow, rtmlib/onnxruntime and existing cached pose models. Run:

```sh
python -m experiments.refine_v5.run_pilot --data PRIVATE_DATA \
  --library ABSOLUTE_LIBRARY_DIR --body READ_ONLY_BASE_FBX \
  --blender /Applications/Blender.app/Contents/MacOS/Blender --out NEW_RUN
```

A fresh output directory is required. GT is used by training/evaluation only.
Blender 5.2.0 LTS and the existing converter are used for actual FBX operations.
A fresh run through the single-command driver reproduced both actual FBX
metrics within 1e-5; those two exports are repeats, not new successful cases.

Tests: 9/9 passed for P0 label/hash/group contracts, checkpoint roundtrip/version rejection,
nonfinite training rejection, and runtime GT-field rejection. Smoke 54/54 passed.
