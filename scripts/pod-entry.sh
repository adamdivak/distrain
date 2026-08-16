#!/usr/bin/env bash
# Entry point for cloud pods that boot the pinned image directly (the image-parity
# path: RunPod "Container Start Command" = this script). The image has no init;
# this supplies the two things a directly-booted pod needs — an sshd to drive the
# session over, and a foreground process so the container stays up.
#
# RunPod injects the account's SSH public keys as $PUBLIC_KEY. Expose TCP port 22
# in the pod config; the console then shows the mapped external port.
set -euo pipefail

mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# SSH sessions do not inherit Docker ENV (sshd resets the environment), so the
# image env -- PATH with /opt/venv/bin first, the UV_* config, the NVIDIA vars the
# container toolkit injected -- is exported where PAM applies it to every session.
env | grep -E '^(PATH|UV_|CUDA|NVIDIA|LD_LIBRARY_PATH)' > /etc/environment

ssh-keygen -A            # per-boot host keys; none are baked in the image
/usr/sbin/sshd           # key-auth only for root (Ubuntu default prohibit-password)

echo "pod-entry: sshd up, idling. torch/GPU check:"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| device_count', torch.cuda.device_count())" || true

exec sleep infinity
