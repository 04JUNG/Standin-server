import { bodyRef, eligible, } from "./model.js";
const object = (v) => v !== null && typeof v === "object" && !Array.isArray(v)
    ? v
    : null;
const sha = (v) => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
/** Validate the consumed sidecar fields without rejecting a successful pose response. */
export function mapBodyRecommendations(cut, catalog) {
    const result = new Map();
    const raw = object(cut.body_matching);
    const fallback = (status, reason) => {
        for (const p of cut.people ?? [])
            result.set(p.index, { status, body: null, reasonCodes: [reason] });
        return result;
    };
    if (!raw || !Object.keys(raw).length)
        return fallback("disabled", "body_disabled");
    if (raw.schema_version !== "body-match.v1" ||
        !["auto", "shadow"].includes(String(raw.mode)) ||
        !["ok", "partial", "unavailable", "not_applicable"].includes(String(raw.status)) ||
        !Array.isArray(raw.people))
        return fallback("unavailable", "body_contract_invalid");
    if (raw.mode === "shadow")
        return fallback("disabled", "body_shadow");
    if (raw.status === "not_applicable")
        return fallback("not_applicable", "body_not_applicable");
    if (raw.status === "unavailable")
        return fallback("unavailable", "body_unavailable");
    const rows = raw.people.map(object);
    if (!sha(raw.input_sha256) ||
        !sha(raw.catalog_sha256) ||
        typeof raw.catalog_version !== "string" ||
        rows.length !== cut.people.length ||
        rows.some((r) => !r) ||
        new Set(rows.map((r) => r?.person_index)).size !== rows.length ||
        rows.some((r) => !cut.people.some((p) => p.index === r?.person_index)))
        return fallback("unavailable", "body_contract_invalid");
    for (const p of cut.people) {
        let rec = {
            status: "unavailable",
            body: null,
            reasonCodes: ["body_contract_invalid"],
        };
        result.set(p.index, rec);
        const row = rows.find((r) => r?.person_index === p.index);
        const bindings = row.pose_bindings;
        if (row.person_id !== `${raw.input_sha256}:p${p.index}` ||
            !Array.isArray(bindings) ||
            bindings.length !== p.candidates.length ||
            bindings.some((v, i) => {
                const b = object(v), c = p.candidates[i];
                return (!b ||
                    b.pose_id !== c.pose_id ||
                    b.view !== c.view ||
                    !(b.pose_sha256 === null || sha(b.pose_sha256)) ||
                    (c.camera &&
                        b.pose_sha256 !== null &&
                        b.pose_sha256 !== c.camera.source_bvh_sha256));
            }))
            continue;
        if (!p.candidates.length) {
            result.set(p.index, {
                status: "not_applicable",
                body: null,
                reasonCodes: ["no_candidates"],
            });
            continue;
        }
        if (row.auto_body_id === null &&
            row.selected_asset === null &&
            row.selection_state === "unavailable") {
            result.set(p.index, {
                status: "unavailable",
                body: null,
                reasonCodes: ["body_unavailable"],
            });
            continue;
        }
        const a = object(row.selected_asset);
        if (!a ||
            row.selection_state !== "selected" ||
            row.applied_body_id !== row.auto_body_id ||
            a.body_id !== row.auto_body_id ||
            ![
                "auto_default",
                "auto_best_effort",
                "auto_presentation_default",
            ].includes(String(row.selection_source)))
            continue;
        if (row.selection_source === "auto_presentation_default") {
            const trace = object(row.presentation_selection);
            const observation = object(row.observations);
            const presentation = object(observation?.presentation);
            if (!presentation ||
                raw.is_mock !== false ||
                observation?.ownership_ambiguous !== false ||
                observation?.provider_error != null ||
                trace?.mode !== "compatible_candidates" ||
                trace.visibility !== "visible" ||
                !["feminine", "masculine"].includes(String(trace.observed)) ||
                presentation?.value !== trace.observed ||
                presentation.visibility !== "visible" ||
                typeof presentation.evidence !== "string" ||
                !presentation.evidence.trim() ||
                !Array.isArray(presentation.cues) ||
                !presentation.cues.some((cue) => cue === "face_design" || cue === "body_contour"))
                continue;
        }
        const asset = catalog.assets.find((v) => v.bodyId === a.body_id &&
            v.bodyVersion === a.body_version &&
            v.assetSha256 === a.asset_sha256 &&
            v.rigVersion === a.rig_version &&
            v.measurementVersion === a.measurement_version);
        rec = {
            status: "unavailable",
            body: null,
            reasonCodes: ["body_asset_unavailable"],
            inputSha256: raw.input_sha256,
            catalogVersion: raw.catalog_version,
            catalogSha256: raw.catalog_sha256,
            selectionSource: row.selection_source,
        };
        if (row.selection_source === "auto_default")
            rec.reasonCodes = ["catalog_default_not_recommendation"];
        else if (asset &&
            eligible(asset, catalog, p.candidates.map((c) => c.pose_id)))
            rec = {
                ...rec,
                status: "available",
                body: bodyRef(asset),
                reasonCodes: row.selection_source === "auto_presentation_default"
                    ? ["presentation_supported_shape_default"]
                    : [],
            };
        result.set(p.index, rec);
    }
    return result;
}
