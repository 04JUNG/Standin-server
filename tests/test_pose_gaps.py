"""pose_gaps: export 계약·TTL·재검색 충실도·공백 라벨·군집·ID 없는 집계·게이트·개인정보 검사."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_gaps import aggregates as agg
from pose_gaps.__main__ import main as cli
from pose_gaps.analyze import run as run_analysis
from pose_gaps.cluster import cluster_observations, complete_linkage
from pose_gaps.config import GapConfig, load_config
from pose_gaps.eligibility import ineligibility, quantitative_eligible
from pose_gaps.features import mirror, query_feature, sym_distance, sym_distance_matrix
from pose_gaps.gate import Query, queries_from_snapshot, run_gate
from pose_gaps.libraries import Library, load_entries_readonly
from pose_gaps.observations import ObservationError, parse_observation, privacy_problems
from pose_gaps.privacy import load_hash_list, scan
from pose_gaps.pull import PullError, pull
from pose_gaps.replay import FidelityError, check_fidelity, replay
from pose_gaps.report import render
from pose_gaps.ttl import EXPORT_META, OBSERVATIONS, SNAPSHOTS, StaleSnapshotError, latest_fresh, purge
from src.features import normalize_skeleton
from src.repo import build_db, load_entries
from src.schema import LibraryEntry, View

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
CFG = load_config()

# 픽셀 좌표(y 아래). 어깨 중점 (100,70), 엉덩이 중점 (100,150) → 몸통 80px.
STAND = np.array([
    (100, 40), (95, 35), (105, 35), (90, 38), (110, 38),
    (85, 70), (115, 70), (80, 105), (120, 105), (78, 140), (122, 140),
    (90, 150), (110, 150), (90, 200), (110, 200), (90, 250), (110, 250)], dtype=np.float32)


def pose(**moves) -> np.ndarray:
    out = STAND.copy()
    for joint, xy in moves.items():
        out[int(joint[1:])] = xy
    return out


ARMS_UP = pose(j7=(80, 35), j8=(120, 35), j9=(78, 0), j10=(122, 0))
KICK = pose(j14=(150, 165), j16=(195, 160))
# 앉아서 오른손을 턱에 괸 자세. 좌우 비대칭이라 반전하면 다른 자세가 된다.
SIT = pose(j7=(70, 115), j8=(125, 100), j9=(65, 150), j10=(108, 62),
           j13=(50, 140), j14=(150, 140), j15=(40, 200), j16=(160, 200))
INVERTED = pose(j13=(85, 20), j14=(115, 20), j15=(85, -20), j16=(115, -20))


def feature_of(kp) -> np.ndarray:
    return normalize_skeleton(kp, np.ones(17, np.float32))


def make_library(path: Path, poses: dict[str, np.ndarray], version: str) -> Library:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [LibraryEntry(pose_id=pid, view=view, feature=feature_of(kp), tags={},
                            bvh_path=f"data/bvh/{pid}.bvh",
                            meta={"pose_family_id": pid.removesuffix("_mirror")})
               for pid, kp in poses.items() for view in View]
    build_db(entries, str(path))
    return Library.load(path, version=version)


def mirrored_kp(kp: np.ndarray) -> np.ndarray:
    """픽셀 좌표 좌우 반전(관절 이름 교환 포함). x=100 축 기준."""
    perm = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    out = kp[perm].copy()
    out[:, 0] = 200 - out[:, 0]
    return out


def pseudonym(prefix: str, n: int) -> str:
    return f"{prefix}_{n:022d}"


def export_item(kp, library: Library, *, inst: int, obs: int, action="sitting",
                observed_on=date(2026, 9, 20), feedback=None, tamper=0.0, coverage="full",
                version=None, rng=None) -> dict:
    kp = np.asarray(kp, dtype=np.float32)
    if rng is not None:
        kp = kp + rng.normal(0, 1.0, size=kp.shape).astype(np.float32)
    scores = np.full(17, 0.9, np.float32)
    mask = np.ones(17, bool)
    feature = normalize_skeleton(kp, scores, kpt_thr=0.3, valid_mask=mask)
    hits = library.search(feature, mask, top_k=5)
    return {
        "obs": pseudonym("o", obs), "inst": pseudonym("i", inst),
        "observed_on": observed_on.isoformat(),
        "expires_on": (observed_on + timedelta(days=365)).isoformat(),
        "versions": {"pose_library": version or library.version, "feature": 1},
        "cut": {"route": "core", "count_confidence": "high", "person_count": 1,
                "tags": {"shot": "full_half", "action": action, "view": "front",
                         "relationship": "solo"}},
        "person": {
            "keypoints": kp.tolist(), "raw_scores": scores.tolist(),
            "effective_scores": scores.tolist(), "evidence_mask": mask.tolist(),
            "search_mask": mask.tolist(), "coverage_class": coverage,
            "skeleton_state": "valid", "skeleton_source": "full_image", "slot_origin": "vlm",
            "lower_body_observed": True, "confidence": "low", "fallback_mode": "soft",
            "search_scope": "full_body", "distance_metric": "pos",
            "search_stability": "not_required",
            "rank_distance": float(hits[0].distance) + tamper, "confidence_threshold": 0.45,
            "quality_reasons": [], "tags": {"action": action, "view": "front",
                                            "source": "vlm_person"},
            "scope": {"detected": "full", "source": "vlm_person"}},
        "candidates": [{"pose_id": h.pose_id, "view": h.view.value, "rank": r + 1,
                        "distance": float(h.distance), "match_level": "low"}
                       for r, h in enumerate(hits)],
        "behavior": {"selected_rank": None, "exported": False, "selected_refined": None,
                     "job_feedback": feedback},
    }


def write_snapshot(root: Path, items: list[dict], *, pulled=NOW) -> Path:
    snapshot = root / SNAPSHOTS / pulled.strftime("%Y%m%dT%H%M%SZ")
    snapshot.mkdir(parents=True)
    (snapshot / OBSERVATIONS).write_text("".join(json.dumps(i) + "\n" for i in items),
                                         encoding="utf-8")
    (snapshot / EXPORT_META).write_text(json.dumps({"pulled_at": pulled.isoformat()}),
                                        encoding="utf-8")
    return snapshot


@pytest.fixture()
def production(tmp_path) -> Library:
    return make_library(tmp_path / "prod" / "poses.db",
                        {"stand": STAND, "arms_up": ARMS_UP, "kick": KICK,
                         "kick_mirror": mirrored_kp(KICK)}, "lib-20261001-aaaaaaaa")


def gap_items(library: Library, n_inst=4, per_inst=2, seed=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    items, counter = [], 0
    for inst in range(n_inst):
        for k in range(per_inst):
            kp = SIT if (inst + k) % 2 == 0 else mirrored_kp(SIT)   # 왼손·오른손 버전 섞기
            items.append(export_item(kp, library, inst=inst, obs=counter, rng=rng))
            counter += 1
    return items


# ── export 계약 ───────────────────────────────────────────────────────
def test_export_contract_rejects_identifiers_anywhere(production):
    good = export_item(STAND, production, inst=1, obs=1)
    parse_observation(good)
    assert privacy_problems(good) == []

    for mutate in (
        lambda d: d.update(job_id="x"),
        lambda d: d["person"].update(bbox=[0, 0, 1, 1]),
        lambda d: d["person"]["quality_reasons"].append("user_0123456789ab"),
        lambda d: d["behavior"].update(note="installations/x"),
        lambda d: d.update(inst="inst_1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed"),
    ):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(ObservationError, match="privacy"):
            parse_observation(bad)


def test_export_contract_validates_shapes(production):
    item = export_item(STAND, production, inst=1, obs=1)
    item["person"]["keypoints"] = item["person"]["keypoints"][:16]
    with pytest.raises(ObservationError):
        parse_observation(item)


# ── 적격·피처 ─────────────────────────────────────────────────────────
def _coverage_rule(kp, scores):
    """pose_curation/coverage/inference.py·evaluate.py의 정량 조건을 그대로 옮긴 것."""
    valid = scores >= .3
    torso = float(np.linalg.norm(kp[[5, 6]].mean(0) - kp[[11, 12]].mean(0)))
    return bool(valid[[5, 6, 11, 12]].all()) and int(valid[5:].sum()) >= 10 and torso >= 20


def test_quantitative_eligibility_matches_coverage_rule():
    rng = np.random.default_rng(7)
    for _ in range(300):
        kp = STAND + rng.normal(0, 30, size=STAND.shape)
        kp *= rng.uniform(0.05, 1.5)
        scores = rng.uniform(0, 1, 17).astype(np.float32)
        assert quantitative_eligible(kp, scores) == _coverage_rule(kp, scores)


def test_ineligibility_reasons(production):
    base = export_item(STAND, production, inst=1, obs=1)
    assert ineligibility(parse_observation(base)) is None
    cases = {
        ("person", "slot_origin", "rtm_provisional"): "slot_origin",
        ("person", "skeleton_source", "fallback_full_image"): "skeleton_source",
        ("person", "coverage_class", "sparse"): "coverage_class",
        ("person", "search_scope", "upper_body"): "search_scope",
        ("behavior", "job_feedback", "skeleton_wrong"): "skeleton_feedback",
    }
    for (section, key, value), reason in cases.items():
        item = json.loads(json.dumps(base))
        item[section][key] = value
        assert ineligibility(parse_observation(item)) == reason
    item = json.loads(json.dumps(base))
    item["cut"]["tags"]["relationship"] = "hugging"
    assert ineligibility(parse_observation(item)) == "entangled"


def test_query_feature_matches_production_descriptor(production):
    obs = parse_observation(export_item(SIT, production, inst=1, obs=1))
    feature, mask = query_feature(obs)
    expected = normalize_skeleton(obs.keypoints, obs.raw_scores, kpt_thr=0.3,
                                  valid_mask=obs.evidence_mask).reshape(17, 2)
    np.testing.assert_allclose(feature, expected)
    assert mask.all()


def test_mirror_is_an_involution_and_distance_is_mirror_invariant():
    f = feature_of(SIT).reshape(17, 2)
    m = np.ones(17, bool)
    back, back_mask = mirror(*mirror(f, m))
    np.testing.assert_allclose(back, f)
    assert back_mask.all()
    other = feature_of(mirrored_kp(SIT)).reshape(17, 2)
    distance, flipped = sym_distance(f, m, other, m)
    assert distance < 1e-5 and flipped


def test_distance_matrix_agrees_with_pairwise():
    feats = np.stack([feature_of(k).reshape(17, 2) for k in (STAND, SIT, mirrored_kp(SIT), KICK)])
    masks = np.ones((4, 17), bool)
    masks[3, 15] = False
    matrix, _ = sym_distance_matrix(feats, masks)
    for i in range(4):
        for j in range(4):
            if i != j:
                assert abs(matrix[i, j] - sym_distance(feats[i], masks[i], feats[j], masks[j])[0]) < 1e-5


# ── 군집 ──────────────────────────────────────────────────────────────
def test_complete_linkage_respects_eps_and_never_joins_infinite():
    d = np.array([[0, .1, .15, 9], [.1, 0, .12, 9], [.15, .12, 0, 9], [9, 9, 9, 0]], float)
    labels = complete_linkage(d, eps=.2)
    assert labels[0] == labels[1] == labels[2] != labels[3]
    d[3, :3] = d[:3, 3] = np.inf
    assert len(set(complete_linkage(d, eps=100))) == 2
    chain = np.array([[0, .2, .4], [.2, 0, .2], [.4, .2, 0]])
    assert len(set(complete_linkage(chain, eps=.25))) == 2   # 사슬처럼 늘어나지 않는다


def test_cluster_size_counts_installations_not_observations():
    feats = np.stack([feature_of(SIT).reshape(17, 2)] * 5)
    clusters = cluster_observations(feats, np.ones((5, 17), bool), ["a", "a", "a", "b", "b"],
                                    ["full"] * 5, eps=.22)
    assert len(clusters) == 1 and clusters[0].installations == 2 and len(clusters[0].members) == 5


# ── 재검색 충실도·라벨 ─────────────────────────────────────────────────
def test_fidelity_passes_when_replay_matches_and_fails_on_drift(production):
    items = [export_item(kp, production, inst=i, obs=i) for i, kp in
             enumerate([STAND, ARMS_UP, KICK, SIT, mirrored_kp(SIT)])]
    result = check_fidelity([parse_observation(i) for i in items], production, CFG)
    assert result["checked"] == 5 and result["matched"] == 5

    items[0]["person"]["rank_distance"] += 0.01
    with pytest.raises(FidelityError):
        check_fidelity([parse_observation(i) for i in items], production, CFG)

    other = [export_item(STAND, production, inst=9, obs=9, version="v1", tamper=0.5)]
    assert check_fidelity([parse_observation(i) for i in other], production, CFG)["checked"] == 0


def test_labels_cover_gap_filled_and_extraction(production, tmp_path):
    observations = [parse_observation(i) for i in (
        export_item(STAND, production, inst=1, obs=1),
        export_item(SIT, production, inst=2, obs=2),
        export_item(INVERTED, production, inst=3, obs=3),
        export_item(INVERTED, production, inst=4, obs=4, feedback="candidates_irrelevant"),
    )]
    items = replay(observations, production, CFG)
    assert [i.label for i in items] == ["not_gap", "gap_open", "extraction_suspect", "gap_open"]
    assert items[1].prod_distance > CFG.tau_soft
    assert items[1].severity == ("strong" if items[1].prod_distance > CFG.tau_strong else "soft")

    curated = make_library(tmp_path / "curated" / "poses.db",
                           {"stand": STAND, "sit": SIT, "sit_mirror": mirrored_kp(SIT)},
                           "curated-local")
    assert replay(observations[1:2], production, CFG, curated)[0].label == "filled_pending_deploy"


def test_readonly_loader_matches_repo_loader(production):
    db = production.path
    ours = load_entries_readonly(db)
    theirs = load_entries(str(db))
    assert [(e.pose_id, e.view) for e in ours] == [(e.pose_id, e.view) for e in theirs]
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a.feature, b.feature)


# ── 분석·집계 ─────────────────────────────────────────────────────────
def test_analyze_writes_id_free_aggregates_and_report(production, tmp_path):
    root = tmp_path / "gaps"
    items = gap_items(production, n_inst=4, per_inst=2)
    items += [export_item(STAND, production, inst=10 + i, obs=100 + i) for i in range(3)]
    items.append(export_item(SIT, production, inst=20, obs=200, coverage="sparse"))
    snapshot = write_snapshot(root, items)

    analysis = run_analysis(snapshot, production, CFG, root)
    assert analysis["fidelity"]["matched"] == analysis["fidelity"]["checked"] == 11
    assert analysis["summary"]["labels"]["gap_open"] == 8
    assert analysis["summary"]["target_clusters"] == 1

    clusters = agg.load(root)
    assert len(clusters) == 1
    record = clusters[0]
    assert record["support"] == {"installations": 4, "observations": 8}
    assert record["histograms"]["person_action"] == {"sitting": 4}
    text = json.dumps(clusters)
    assert "o_0" not in text and "i_0" not in text and '"obs"' not in text
    assert (snapshot / "analysis.json").is_file()
    html = render(json.loads((snapshot / "analysis.json").read_text(encoding="utf-8")))
    assert "<svg" in html and record["cluster_key"] in html

    # 다음 주기: 같은 자리의 군집은 키를 유지하고 이력만 늘어난다.
    snapshot.rename(snapshot.with_name("old"))
    second = write_snapshot(root, gap_items(production, n_inst=3, per_inst=1, seed=3),
                            pulled=NOW + timedelta(days=1))
    run_analysis(second, production, CFG, root)
    again = agg.load(root)
    assert len(again) == 1 and again[0]["cluster_key"] == record["cluster_key"]
    assert len(again[0]["history"]) == 2


def test_small_clusters_and_rare_values_are_not_kept(production, tmp_path):
    root = tmp_path / "gaps"
    items = gap_items(production, n_inst=2, per_inst=3)
    snapshot = write_snapshot(root, items)
    analysis = run_analysis(snapshot, production, CFG, root)
    assert analysis["summary"]["target_clusters"] == 0 and agg.load(root) == []

    items = gap_items(production, n_inst=4, per_inst=1)
    items[0]["person"]["tags"]["action"] = "reaching"
    snapshot2 = write_snapshot(tmp_path / "gaps2", items)
    run_analysis(snapshot2, production, CFG, tmp_path / "gaps2")
    record = agg.load(tmp_path / "gaps2")[0]
    assert record["histograms"]["person_action"] == {"_other": "<3", "sitting": 3}


def test_centroid_drops_joints_seen_by_too_few_installations(production):
    observations = [parse_observation(i) for i in gap_items(production, n_inst=3, per_inst=1)]
    items = replay(observations, production, CFG)
    items[0].mask[15] = False
    items[1].mask[15] = False
    points, support = agg.centroid(items, [False] * 3, k=3)
    assert points[15] is None and support[15] is None
    assert points[16] is not None and support[16] == 3


def test_assert_id_free_catches_pseudonyms():
    agg.assert_id_free([{"cluster_key": "g-1", "histograms": {"production_top1": {"stand": 3}}}])
    with pytest.raises(ValueError):
        agg.assert_id_free([{"cluster_key": "g-1", "members": [pseudonym("o", 1)]}])


def test_measure_closes_cluster_after_library_fills_it(production, tmp_path):
    root = tmp_path / "gaps"
    run_analysis(write_snapshot(root, gap_items(production)), production, CFG, root)
    filled = make_library(tmp_path / "next" / "poses.db",
                          {"stand": STAND, "arms_up": ARMS_UP, "kick": KICK,
                           "kick_mirror": mirrored_kp(KICK), "sit": SIT,
                           "sit_mirror": mirrored_kp(SIT)}, "lib-20261009-bbbbbbbb")
    observations = [parse_observation(i) for i in gap_items(filled, seed=5)]
    items = replay(observations, filled, CFG)
    records = agg.measure(agg.load(root), items, CFG, filled.version,
                          filled.pose_ids - production.pose_ids, date(2026, 10, 9))
    row = records[0]["history"][-1]
    assert row["fill_rate"] == 1.0 and row["new_pose_top1_share"] == 1.0
    assert records[0]["status"] == "closed"


# ── TTL·pull ──────────────────────────────────────────────────────────
def test_purge_and_freshness(tmp_path, production):
    root = tmp_path / "gaps"
    old = write_snapshot(root, [], pulled=NOW - timedelta(days=20))
    expired = export_item(STAND, production, inst=1, obs=1, observed_on=date(2025, 9, 1))
    live = export_item(STAND, production, inst=2, obs=2)
    recent = write_snapshot(root, [expired, live], pulled=NOW - timedelta(days=8))
    result = purge(root, now=NOW, ttl_days=14)
    assert result == {"removed_snapshots": [old.name], "dropped_rows": 1}
    assert len((recent / OBSERVATIONS).read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(StaleSnapshotError):
        latest_fresh(root, now=NOW, max_age_days=7)
    assert latest_fresh(root, now=NOW, max_age_days=8) == recent


def test_pull_writes_one_snapshot_and_rejects_bad_exports(tmp_path, production):
    root = tmp_path / "gaps"
    write_snapshot(root, [], pulled=NOW - timedelta(days=1))
    pages = {
        None: {"schemaVersion": 1, "exportId": "e1", "window": {"days": 90},
               "generatedAt": NOW.isoformat(), "retention": {"localTtlDays": 14},
               "items": [export_item(STAND, production, inst=1, obs=1)], "nextCursor": "c2"},
        "c2": {"schemaVersion": 1, "exportId": "e1",
               "items": [export_item(SIT, production, inst=2, obs=2)], "nextCursor": None},
    }

    def fetch(url, token):
        assert token == "secret" and "days=90" in url
        return pages["c2" if "cursor=c2" in url else None]

    snapshot = pull("https://bff.example", days=90, root=root, token="secret", now=NOW, fetch=fetch)
    assert len((snapshot / OBSERVATIONS).read_text(encoding="utf-8").splitlines()) == 2
    assert [p.name for p in (root / SNAPSHOTS).iterdir()] == [snapshot.name]
    assert "secret" not in (snapshot / EXPORT_META).read_text(encoding="utf-8")

    pages["c2"]["items"][0]["job_id"] = "x"
    with pytest.raises(PullError, match="계약"):
        pull("https://bff.example", days=90, root=root, token="secret",
             now=NOW + timedelta(hours=1), fetch=fetch)
    assert [p.name for p in (root / SNAPSHOTS).iterdir()] == [snapshot.name]

    pages["c2"]["items"][0].pop("job_id")
    pages["c2"]["exportId"] = "e2"
    with pytest.raises(PullError, match="exportId"):
        pull("https://bff.example", days=90, root=root, token="secret", now=NOW, fetch=fetch)
    with pytest.raises(PullError, match="STANDIN_ADMIN_TOKEN"):
        pull("https://bff.example", days=90, root=root, token="", now=NOW, fetch=fetch)


# ── 게이트 ────────────────────────────────────────────────────────────
def _snapshot_queries(library, extra=()):
    observations = [parse_observation(i) for i in
                    gap_items(library) + [export_item(k, library, inst=50 + n, obs=500 + n)
                                          for n, k in enumerate([STAND, ARMS_UP, KICK, *extra])]]
    return queries_from_snapshot(replay(observations, library, CFG))


def test_gate_passes_when_new_pose_fills_gap_with_mirror_twin(production, tmp_path):
    candidate = make_library(tmp_path / "cand" / "poses.db",
                             {"stand": STAND, "arms_up": ARMS_UP, "kick": KICK,
                              "kick_mirror": mirrored_kp(KICK), "sit": SIT,
                              "sit_mirror": mirrored_kp(SIT)}, "lib-cand")
    report = run_gate(production, candidate, _snapshot_queries(production), CFG)
    assert report["status"] == "passed", report["checks"]
    assert report["summary"]["gap_filled_after"] == report["summary"]["gap_queries"] == 8
    assert report["summary"]["regressed"] == 0


def test_gate_fails_on_orphan_mirror_and_on_dropping_a_good_pose(production, tmp_path):
    orphan = make_library(tmp_path / "orphan" / "poses.db",
                          {"stand": STAND, "arms_up": ARMS_UP, "kick": KICK,
                           "kick_mirror": mirrored_kp(KICK), "sit": SIT}, "lib-orphan")
    report = run_gate(production, orphan, _snapshot_queries(production), CFG)
    checks = {c["name"]: c for c in report["checks"]}
    assert not checks["mirror_closure"]["passed"] and checks["mirror_closure"]["orphans"] == ["sit"]
    assert run_gate(production, orphan, _snapshot_queries(production), CFG,
                    mirror_waivers={"sit"})["status"] == "passed"

    # 서기·발차기를 빼면 원래 high였던 그 쿼리들이 0.25를 넘어간다.
    dropped = make_library(tmp_path / "dropped" / "poses.db",
                           {"arms_up": ARMS_UP, "sit": SIT, "sit_mirror": mirrored_kp(SIT)},
                           "lib-dropped")
    report = run_gate(production, dropped, _snapshot_queries(production), CFG)
    checks = {c["name"]: c for c in report["checks"]}
    assert report["status"] == "failed" and not checks["high_crossings"]["passed"]


def test_gate_flags_hub_that_steals_good_matches(production, tmp_path):
    # 서기 쿼리 여러 개가 잘 맞고 있는데, 서기와 거의 같은 새 포즈가 전부 가져가면 hub다.
    hub = make_library(tmp_path / "hub" / "poses.db",
                       {"stand": STAND, "arms_up": ARMS_UP, "kick": KICK,
                        "kick_mirror": mirrored_kp(KICK), "stand_copy": STAND + 0.01,
                        "stand_copy_mirror": mirrored_kp(STAND)}, "lib-hub")
    queries = [Query(feature_of(STAND + d).reshape(17, 2), np.ones(17, bool), "pos", "coverage")
               for d in np.linspace(0, 0.5, 10)]
    report = run_gate(production, hub, queries, CFG)
    checks = {c["name"]: c for c in report["checks"]}
    assert report["summary"]["regressed"] == 0
    assert not checks["hubness"]["passed"]


def test_gate_checks_bundle_assets(production, tmp_path):
    bundle = tmp_path / "bundle"
    library = make_library(bundle / "poses.db", {"stand": STAND, "stand_mirror": mirrored_kp(STAND)},
                           "lib-assets")
    (bundle / "bvh").mkdir()
    (bundle / "bvh" / "stand.bvh").write_text("x", encoding="utf-8")
    report = run_gate(production, Library.load(bundle, version="lib-assets"),
                      [Query(feature_of(STAND).reshape(17, 2), np.ones(17, bool), "pos", "coverage")], CFG)
    assets = {c["name"]: c for c in report["checks"]}["assets"]
    assert assets["checked"] and not assets["passed"] and assets["missing"] == 1 + 8
    assert library.version == "lib-assets"


# ── 개인정보 검사·설정 ─────────────────────────────────────────────────
def test_privacy_scan_finds_names_contents_and_listed_hashes(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "notes.json").write_text('{"pose": "job_interview_01"}', encoding="utf-8")
    assert scan([clean]) == []

    dirty = tmp_path / "dirty"
    (dirty / "installations" / "inst_1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed").mkdir(parents=True)
    (dirty / "installations" / "inst_1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed" / "a.jpg").write_bytes(b"")
    (dirty / "config.json").write_text('{"reference_roughs": ["user_0123456789ab"]}', encoding="utf-8")
    digest = "ab" * 32
    (dirty / "db.sqlite").write_bytes(b"\x00\x01" + digest[:12].encode() + b"\x00")
    hash_list = tmp_path / "hashes.json"
    hash_list.write_text(json.dumps({"x": digest}), encoding="utf-8")
    labels = {(f.label, f.where) for f in scan([dirty], load_hash_list(hash_list))}
    assert {("installation_id", "name"), ("beta_data_key", "name"), ("user_input_hash", "content"),
            ("listed_user_hash", "content")} <= labels
    assert cli(["privacy-scan", str(dirty), "--strict"]) == 2
    assert cli(["privacy-scan", str(clean), "--strict"]) == 0
    assert cli(["privacy-scan", str(dirty), "--strict", "--exclude", "dirty"]) == 0


def test_config_guards():
    assert CFG.version.startswith("pose_gaps-")
    with pytest.raises(ValueError):
        GapConfig(version="x", k_min_installations=2)
    with pytest.raises(ValueError):
        GapConfig(version="x", tau_fill=0.3, tau_soft=0.25)
