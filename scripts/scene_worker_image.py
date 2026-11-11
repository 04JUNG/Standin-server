"""Resolve an immutable scene-worker commit tag without rebuilding it on retries."""

import argparse
import os
from pathlib import Path
import re


REPOSITORY = "standin/scene-worker"


def resolve_image(client, repository, commit):
    if repository != REPOSITORY or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("a full commit SHA and the dedicated repository are required")
    repositories = client.describe_repositories(repositoryNames=[repository])["repositories"]
    if len(repositories) != 1 or repositories[0].get("imageTagMutability") != "IMMUTABLE":
        raise ValueError("scene-worker commit tags must be immutable")
    try:
        reply = client.describe_images(repositoryName=repository, imageIds=[{"imageTag": commit}])
    except Exception as error:
        # Access denial, throttling and network failures are not a cache miss.
        if getattr(error, "response", {}).get("Error", {}).get("Code") == "ImageNotFoundException":
            return None
        raise
    images = reply.get("imageDetails", [])
    if len(images) != 1 or commit not in images[0].get("imageTags", []):
        raise ValueError("unexpected commit image response")
    digest = images[0].get("imageDigest", "")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError("invalid published image digest")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--require", action="store_true")
    args = parser.parse_args()
    import boto3

    digest = resolve_image(boto3.client("ecr"), args.repository, args.commit)
    if args.require and digest is None:
        raise SystemExit("scene-worker commit image has not been published")
    output = f"found={'true' if digest else 'false'}\ndigest={digest or ''}\n"
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(output)
    print("Reusing published commit image" if digest else "Commit image needs its first build")


if __name__ == "__main__":
    main()
