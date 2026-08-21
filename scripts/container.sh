#!/usr/bin/env bash
# Build and run the pinned distrain image (Dockerfile) -- the same image on aurora
# and on rented cloud nodes. One entry point so a rebuild is one command, not four.
#
#   scripts/container.sh build          # (re)build the image
#   scripts/container.sh push           # build + push to a registry (IMAGE must
#                                       # carry the host, e.g. ghcr.io/...)
#   scripts/container.sh test           # run the pytest suite in the container, on GPU
#   scripts/container.sh smoke          # print torch + GPU visibility, then exit
#   scripts/container.sh shell          # interactive bash in the container
#   scripts/container.sh run <cmd...>   # run an arbitrary command, e.g.
#                                       #   scripts/container.sh run torchrun --nproc_per_node=1 -m distrain.train ...
#
# Run/test/shell/smoke bind-mount the working tree at /workspace so rsync'd edits
# are live without a rebuild (the baked venv lives at /opt/venv, untouched by the
# mount). Pass --no-mount to run the code baked into the image instead -- that is the
# reproducible mode for reported results.
#
# Environment overrides:
#   IMAGE=distrain:local     image tag to build/run
#   CUDA_IMAGE=...           base image passed to the build (see Dockerfile)
#   NET_ADMIN=1              add --cap-add=NET_ADMIN (needed for `tc qdisc netem`,
#                            the slow-network experiments; decisions §9)
#   GPUS=all                 value for docker's --gpus
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-distrain:local}"
GPUS="${GPUS:-all}"

DOCKER=(docker)

# Resolve how to talk to the daemon, once, for commands that need it. Falls back to
# sudo if the socket isn't accessible (e.g. before a fresh login has picked up
# 'docker' group membership from setup-docker-nvidia.sh).
ensure_docker() {
    docker info >/dev/null 2>&1 && return 0
    if sudo -n docker info >/dev/null 2>&1 || sudo docker info >/dev/null 2>&1; then
        echo "note: docker socket not accessible as $USER; using sudo." >&2
        echo "      run scripts/setup-docker-nvidia.sh and re-login to drop sudo." >&2
        DOCKER=(sudo docker)
        return 0
    fi
    echo "ERROR: cannot reach the Docker daemon. Run scripts/setup-docker-nvidia.sh first." >&2
    exit 1
}

# BuildKit attaches a provenance attestation by default, which forces the tag to
# be an OCI *index* (a second manifest at platform unknown/unknown). Registry
# clients that only know the Docker media types then get a 404 on the manifest and
# report it as "image does not exist or you don't have permission" -- Prime
# Intellect's image check does exactly this (decisions §20). We build for one
# platform and never consume the attestation, so turning both off costs nothing.
BUILD_FLAGS=(--provenance=false --sbom=false)

build() {
    "${DOCKER[@]}" build \
        "${BUILD_FLAGS[@]}" \
        ${CUDA_IMAGE:+--build-arg "CUDA_IMAGE=$CUDA_IMAGE"} \
        -t "$IMAGE" \
        -f "$REPO/Dockerfile" \
        "$REPO"
    echo "built $IMAGE"
}

# Build and push in one step. `oci-mediatypes=false` is the load-bearing half:
# without it the pushed manifest is still an OCI one even with no attestation,
# and a plain `docker push` of an already-built image copies whatever media type
# the local store holds rather than converting it.
push() {
    case "$IMAGE" in
        *ghcr.io/*|*docker.io/*|*/*/*) ;;
        *) echo "ERROR: IMAGE=$IMAGE has no registry host; refusing to push." >&2; exit 2 ;;
    esac
    "${DOCKER[@]}" buildx build \
        "${BUILD_FLAGS[@]}" \
        ${CUDA_IMAGE:+--build-arg "CUDA_IMAGE=$CUDA_IMAGE"} \
        --output "type=image,name=$IMAGE,oci-mediatypes=false,push=true" \
        -f "$REPO/Dockerfile" \
        "$REPO"
    echo "pushed $IMAGE"
}

# Assemble the common `docker run` flags for GPU work.
run_args() {
    local -n _out=$1        # nameref to the caller's array
    _out=(--rm --gpus "$GPUS" --ipc=host)
    [[ "${NET_ADMIN:-0}" == "1" ]] && _out+=(--cap-add=NET_ADMIN)
    if [[ "${NO_MOUNT:-0}" != "1" ]]; then
        _out+=(-v "$REPO:/workspace" -w /workspace)
    fi
}

require_image() {
    "${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1 || {
        echo "image $IMAGE not found -- run 'scripts/container.sh build' first." >&2
        exit 1
    }
}

cmd="${1:-}"
[[ $# -gt 0 ]] && shift || true

# Strip a leading --no-mount from run/test/shell/smoke arguments.
if [[ "${1:-}" == "--no-mount" ]]; then
    NO_MOUNT=1
    shift
fi

case "$cmd" in
    build|rebuild)
        ensure_docker
        build
        ;;
    push)
        ensure_docker
        push
        ;;
    smoke)
        ensure_docker; require_image
        run_args ARGS
        "${DOCKER[@]}" run "${ARGS[@]}" "$IMAGE"   # image CMD = torch/GPU report
        ;;
    test)
        ensure_docker; require_image
        run_args ARGS
        "${DOCKER[@]}" run "${ARGS[@]}" "$IMAGE" pytest -q "$@"
        ;;
    shell)
        ensure_docker; require_image
        run_args ARGS
        "${DOCKER[@]}" run -it "${ARGS[@]}" "$IMAGE" bash
        ;;
    run)
        [[ $# -gt 0 ]] || { echo "usage: scripts/container.sh run <cmd...>" >&2; exit 2; }
        ensure_docker; require_image
        run_args ARGS
        "${DOCKER[@]}" run "${ARGS[@]}" "$IMAGE" "$@"
        ;;
    ""|-h|--help|help)
        # Print the leading comment block (skip the shebang, stop at the first
        # non-comment line).
        awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
        ;;
    *)
        echo "unknown subcommand: $cmd (try: build push test smoke shell run)" >&2
        exit 2
        ;;
esac
