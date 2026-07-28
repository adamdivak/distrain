# syntax=docker/dockerfile:1

# The one image, run identically on aurora and on rented cloud nodes (brief §3,
# docs/decisions.md §1/§2). An unpinned environment silently invalidates
# cross-provider comparisons, so everything that matters is pinned: the CUDA base,
# uv, and — via pyproject.toml + uv.lock — torch and the rest.
#
# Base is the CUDA *devel* image: it carries nvcc, headers and the system NCCL/IB
# libraries that matter for multi-node all-reduce, plus a toolchain for torch.compile
# / Triton codegen. torch itself is NOT the base image's CUDA — it comes from the
# cu126 pip wheels (pyproject.toml), which bundle their own CUDA runtime, cuDNN and
# NCCL. The base supplies the driver ABI (injected at run time by the NVIDIA
# Container Toolkit) and the toolchain, nothing more.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

# uv, pinned to the host version (docs/decisions.md §2: identical env everywhere).
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*
# iproute2 provides `tc`, needed for the netem slow-network trick (brief §5,
# decisions §9). It does nothing without --cap-add=NET_ADMIN at run time.

# uv config, baked so `python`/`pytest`/`torchrun` Just Work at run time:
#  - the project venv lives at /opt/venv, OUTSIDE /workspace, so bind-mounting a
#    live working tree over /workspace (the iteration path on aurora) never shadows
#    the baked environment.
#  - the uv-managed CPython (pyproject: python-preference = only-managed) is baked
#    into the image at a stable path.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /workspace

# Dependency layer first: install the locked deps without the project itself, so a
# source-only edit doesn't reinstall torch on rebuild. --frozen fails loudly if
# uv.lock is stale relative to pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra dev

# Project layer: the source, then install the distrain package into the same venv.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev

# Baked venv first on PATH -> no `uv run` needed at run time; `uv run --no-sync`
# also resolves to it via UV_PROJECT_ENVIRONMENT.
ENV PATH="/opt/venv/bin:${PATH}"

# Default: prove the GPU is visible through the container. Overridden by whatever
# command scripts/container.sh (or a cloud launcher) passes.
CMD ["python", "-c", "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| device_count', torch.cuda.device_count())"]
