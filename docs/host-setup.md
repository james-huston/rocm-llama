# Host ROCm Setup for AMD GPUs

This guide covers the host-level prerequisites for running `rocm-llama` on an
AMD GPU. The container is self-contained, but the host needs the `amdgpu` kernel
driver loaded and the right device nodes exposed.

## Prerequisites

- **Linux** — ROCm is Linux-only. Ubuntu 24.04+ is known-good.
- **AMD GPU** — Radeon RX 7000/6000 series or Instinct MI series.
- **Docker Engine** with Compose (v2+).

## Step 1: Verify the amdgpu driver is loaded

```bash
# Check if the amdgpu module is loaded
lsmod | grep amdgpu

# If not loaded, load it:
sudo modprobe amdgpu

# Make it persistent across reboots:
echo "amdgpu" | sudo tee /etc/modules-load.d/amdgpu.conf
```

## Step 2: Verify device nodes exist

ROCm needs `/dev/kfd` and `/dev/dri` (or `/dev/dri/card0`, `/dev/dri/renderD128`):

```bash
ls -la /dev/kfd /dev/dri/
```

If these don't exist, your kernel may not have the right firmware or driver
loaded. Check `dmesg | grep -i amdgpu` for errors.

## Step 3: Install ROCm tools (for host-side diagnostics)

You don't need the full ROCm toolkit for the container to work, but it's useful
for host-side diagnostics:

```bash
# Ubuntu 24.04
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.1.2 jammy main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install -y rocm-hiproc rocm-smi-lib rocm-libs
```

After installation, verify:

```bash
rocminfo | grep gfx
rocm-smi
```

You should see your GPU listed with its gfx code (e.g. `gfx1100` for RX 7900 XTX).

## Step 4: Add your user to the right groups

```bash
# Check your group GIDs
getent group render video

# Add your user (replace $USER if needed)
sudo usermod -aG render,video $USER

# Log out and back in for changes to take effect
```

## Step 5: Verify GPU detection

```bash
rocminfo | grep -A5 "Device"
```

You should see your GPU with its name, gfx code, and memory info.

## Troubleshooting

### GPU not detected by rocminfo

- Check `dmesg | grep -i amdgpu` for firmware errors.
- Make sure the `amdgpu` module is loaded: `lsmod | grep amdgpu`.
- Try `HSA_OVERRIDE_GFX_VERSION=10.3.0` for RDNA2 cards not natively supported.

### Docker container can't see the GPU

- Verify `/dev/kfd` and `/dev/dri` exist on the host.
- Check that your user is in the `render` and `video` groups.
- Check that `docker compose ps` shows the container running.
- Inside the container: `rocminfo | grep gfx` should show your GPU.

### `rocm-smi` fails inside container

- The container needs `--device /dev/kfd --device /dev/dri` and `ipc: host`.
- Verify the docker-compose.yml has these settings.

### OOM / segfaults with large models

- Try reducing `ctx` (context size) in your config.
- Dense 24B+ Mistrals may segfault at 128k context on ROCm — keep them at 65536.
- Use `HSA_OVERRIDE_GFX_VERSION` if your GPU isn't natively supported by the
  pinned ROCm version.
