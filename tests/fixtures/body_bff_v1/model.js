export const defaultPreferences = () => ({
    version: "body-preferences.v1",
    scope: "installation",
    mode: "auto",
    defaultCharacterId: null,
    revision: 0,
});
export const bodyRef = (a) => ({
    bodyId: a.bodyId,
    bodyVersion: a.bodyVersion,
    assetSha256: a.assetSha256,
    rigVersion: a.rigVersion,
    measurementVersion: a.measurementVersion,
    characterId: a.characterId,
});
export const sameBody = (a, b) => JSON.stringify(bodyRef(a)) === JSON.stringify(bodyRef(b));
export function eligible(ref, catalog, poses) {
    return (!!ref &&
        catalog.assets.some((a) => sameBody(a, ref) &&
            poses.every((id) => a.supportedPoseIds.includes(id))));
}
export function snapshot(preferences, catalog) {
    const find = (id) => {
        const a = catalog.assets.find((a) => a.characterId === id);
        return a ? bodyRef(a) : null;
    };
    return {
        preferences: { ...preferences },
        userDefault: find(preferences.defaultCharacterId),
        catalogDefault: find(catalog.defaultCharacterId),
    };
}
export function resolveBody(personIndex, policy, recommendation, catalog, poses, intent = "inherit", manual = null, revision = 0) {
    const state = {
        schemaVersion: "body-selection.v1",
        personIndex,
        intent,
        manualCharacterId: intent === "manual" ? (manual?.characterId ?? null) : null,
        recommendation,
        resolvedBody: null,
        resolvedSource: null,
        resolutionStatus: "needs_selection",
        selectionRevision: revision,
        renderingExecuted: false,
    };
    if (!poses.length)
        return { ...state, resolutionStatus: "not_applicable" };
    const use = (ref, source) => ({
        ...state,
        resolvedBody: ref,
        resolvedSource: source,
        resolutionStatus: eligible(ref, catalog, poses) ? "ready" : "unavailable",
    });
    if (intent === "manual")
        return use(manual, "manual");
    if (intent === "inherit" && policy.preferences.mode === "fixed_default")
        return use(policy.userDefault, "fixed_default");
    if (recommendation.status === "available" &&
        eligible(recommendation.body, catalog, poses))
        return use(recommendation.body, "auto_recommendation");
    // An explicit auto action never silently becomes a default.
    if (intent === "auto")
        return { ...state, resolutionStatus: "unavailable" };
    if (eligible(policy.userDefault, catalog, poses))
        return use(policy.userDefault, "user_default_fallback");
    if (eligible(policy.catalogDefault, catalog, poses))
        return use(policy.catalogDefault, "catalog_default_fallback");
    return state;
}
