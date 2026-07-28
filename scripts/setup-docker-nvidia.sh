#!/usr/bin/env bash
# One-time host setup so `docker run --gpus all` works on this machine (aurora, and
# any fresh cloud node with a working NVIDIA driver + Docker already installed).
#
# This closes the "NVIDIA Container Toolkit is not installed" gap in README.md, and
# is the prerequisite for scripts/container.sh. It needs root for the apt install and
# the Docker restart, so it uses sudo internally -- run it as your normal user:
#
#   scripts/setup-docker-nvidia.sh
#
# It does NOT install the NVIDIA driver or Docker Engine themselves -- both are
# assumed present (aurora has driver 580.173.02 and Docker 29.x). It is idempotent:
# re-running it is safe.
set -euo pipefail

KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "==> Preconditions"
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi not found -- install the NVIDIA driver first." >&2
    exit 1
}
command -v docker >/dev/null 2>&1 || {
    echo "ERROR: docker not found -- install Docker Engine first." >&2
    exit 1
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "==> Adding the NVIDIA Container Toolkit apt repository"
if [[ ! -f "$KEYRING" ]]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o "$KEYRING"
else
    echo "    keyring already present, skipping"
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=$KEYRING] https://#g" \
    | sudo tee "$LIST" >/dev/null

echo "==> Installing nvidia-container-toolkit"
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "==> Wiring the NVIDIA runtime into Docker and restarting the daemon"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "==> Adding '$USER' to the 'docker' group (so sudo isn't needed per command)"
if id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
    echo "    already a member"
    GROUP_ALREADY=1
else
    sudo usermod -aG docker "$USER"
    GROUP_ALREADY=0
fi

echo "==> Smoke test: GPU visible inside a container"
# Use sudo for this check because the current shell's group membership predates the
# usermod above; a fresh login is needed before docker works without sudo.
if sudo docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi \
        --query-gpu=name --format=csv,noheader; then
    echo "    OK -- the toolkit is working."
else
    echo "ERROR: GPU smoke test failed. Inspect the output above." >&2
    exit 1
fi

echo
echo "Done."
if [[ "${GROUP_ALREADY:-0}" -eq 0 ]]; then
    echo "IMPORTANT: log out and back in (or run 'newgrp docker') before using"
    echo "scripts/container.sh without sudo -- group membership only applies to new logins."
fi
