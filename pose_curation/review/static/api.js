export async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : `요청 실패 (${response.status})`);
  }
  return response.json();
}

export function thumbnailUrl(pose, view = "front") {
  const version = pose.thumbnail_versions?.[view] || pose.content_hash;
  return `/api/poses/${pose.key}/thumbnail?view=${view}&v=${version.slice(0, 16)}`;
}

export function saveReview(pose, status, note, visual_checks = []) {
  return request(`/api/poses/${pose.key}/review`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", "X-Pose-Review": "1" },
    body: JSON.stringify({ status, note, visual_checks, content_hash: pose.content_hash }),
  });
}
