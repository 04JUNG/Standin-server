> Historical first staging validation. The 45-pose limitation below was subsequently removed; see [on-demand original validation](refine-rc4-on-demand-20261111.md).

# RC4 staging integration verification — 2026-11-11

No pose tuning. All frozen runtime source hashes still match the RC4 manifest.
The three PRs target develop; none was merged during this validation:
- Inference: https://github.com/04JUNG/Standin-server/pull/200
- BFF: https://github.com/04JUNG/Standin-app-server/pull/180
- Desktop: https://github.com/04JUNG/Standin-client/pull/194

## Actual path verified

An isolated `Standin RC4 Validation` macOS app used its own keychain service,
bundle ID and callback scheme. The user completed beta consent. No existing
installation credentials were deleted or changed.

The original 004 PNG was uploaded through the app. The normal search exposed
`rokoko_Typing_UsingMouse_mixamo_00882_mirror` as candidate 2; that unedited library
candidate was selected in the UI. The app called the authenticated BFF, which
sent the stored original image to the actual staging RC4 worker. A fresh
`gpt-6-luna` / `low` analysis produced an accepted **shadow** result.

- Job: `job_0a062d17-e2b6-4cf4-b158-39d0d9c96769`
- Native saved file: `rough-004_standin_pose.fbx`, 1,834,572 bytes
- Saved SHA-256: `a1e057136398b737f9f249d6feac60ca2b79364ba0cb53fcd308ec65a25c4266`
- SHA exactly matches the server's durable owner-scoped release receipt.
- Native review visibly rendered the baked final surface with the returned orthographic camera.
- Native save screen confirmed FBX saved. No legacy re-framing or scene composition was applied to those bytes.
- Camera is a separate receipt, **not embedded in FBX**. Importing FBX alone into an external application does not restore the preview camera. CSP import was not tested here.

An earlier independent real staging 004 HTTP execution also accepted shadow
(95.618 seconds including downloads). For that execution, the final FBX and
display GLB had 24,988 vertices; symmetric surface maximum difference was
9.4834e-7 metres. Returned camera versus client projection differed by at most
2.0265e-11 pixels. These numerical checks refer to the earlier execution, not
the later fresh native run.

## Connection fixes and guards

Legacy search skeleton/camera eligibility had suppressed the RC4 request before
its own extraction. The server now exposes `refineRelease` only while enabled
and for a single target. RC4 separately requires an explicitly selected library
asset, valid original BVH hash, lookup/manual-export permission and ownership.
The private worker still verifies the deployed BVH against the registered source.
Semantic/generated/unbound/multi-target requests do not gain this permission.

The final receipt is persisted with FBX and baked GLB hashes. Preview and save
verify their respective hashes. After selection changes, a stale receipt cannot
be exported. Existing legacy routes retain their eligibility checks.

## Deployment and limitations

Staging inference task revision 138 and BFF revision 136 run RC4 enabled.
The worker is loopback-only within the inference task. Production RC4 remains
disabled; this work did not deploy or change production.

During validation, unrelated GitHub Actions changed staging BFF revision 134 to
135. The deployment guard stopped an overwrite. The latest image (including the
new default-hand fix) was retained as the base for RC4 revision 136. No existing
hand behavior was rolled back.

The concurrent inference deployment also moved staging from revision 136 to 137
with the develop finger-offset fix. Revision 138 retains that newer base and
reapplies only the RC4 HTTP adapter; the private RC4 worker image is unchanged.

Unrelated GitHub Actions also changed production inference 49 to 50 and BFF 58
to 59 during this session. Therefore before/after production task definitions
are **not identical**, though neither production service has RC4 enabled and
this task made no production deployment.

Only 45 registered original poses are currently supported. Unsupported library
assets fail before VLM and do not produce an RC4 correction. This is not a
whole-library release. Body/hand/crop changes are not applied after verification;
the native review states this limitation. Full CSP round-trip and all-library
registration are outside this staging check.

## Checks

- Inference PR CI passed; frozen runtime inventory rechecked.
- BFF PR CI passed, including new RC4 ownership/eligibility checks.
- Client targeted tests: 91 passed, plus 48 HTTP contract tests after updating
  the default-off capability assertion. Native macOS build passed.
- Client full CI passed (frontend, macOS Rust, Windows Rust and final PR gate).
- No PR merge or production activation performed.
