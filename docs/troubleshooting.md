# Troubleshooting

## Common issues and fixes

### Container won't start

**Symptom**: `docker compose up` fails or container exits immediately.

**Check**:
```bash
docker compose logs llama-swap
```

**Common causes**:
- Missing `/dev/kfd` or `/dev/dri` on host
- Wrong group GIDs (check with `getent group render video`)
- ROCm toolkit not installed on host (kernel driver must be loaded)

**Fix**:
```bash
# Verify device nodes
ls -la /dev/kfd /dev/dri/

# Verify groups
getent group render video

# If GIDs differ from defaults (104/39), set them in .env
# RENDER_GID=993
# VIDEO_GID=44
```

### GPU not visible inside container

**Symptom**: `rocminfo` inside container shows no GPUs.

**Check**:
```bash
docker compose exec llama-swap rocminfo | grep gfx
```

**Fix**:
```bash
# Verify host has amdgpu loaded
lsmod | grep amdgpu

# Verify /dev/kfd exists
ls -la /dev/kfd

# Verify user is in render/video groups
groups $USER

# Restart Docker daemon after group changes
sudo systemctl restart docker
```

### `HSA_OVERRIDE_GFX_VERSION` needed

**Symptom**: GPU detected but ROCm reports wrong architecture or fails to load.

**Fix**: Set in `.env`:
```bash
# For RX 6000 series (RDNA2) not natively supported by pinned ROCm version
HSA_OVERRIDE_GFX_VERSION=10.3.0
```

Check your GPU's gfx code:
```bash
rocminfo | grep gfx
```

### Model fails to load (cold start hangs)

**Symptom**: Request to a model hangs for >60s or returns 502.

**Check**:
```bash
docker compose logs llama-swap | tail -50
```

**Common causes**:
- Not enough VRAM for the model
- Corrupted GGUF file
- Model needs a specific chat template

**Fix**:
- Try a smaller quantization (Q5_K_M instead of Q8_0)
- Re-download the GGUF: `make add-model REPO=... FILE=... NAME=...`
- Add the correct template: `TEMPLATE=model-name.jinja` in models.yaml

### Segfaults with large context

**Symptom**: llama-server crashes with SIGSEGV when loading a model with large `ctx`.

**Fix**: Reduce `ctx` in models.yaml. Dense 24B+ Mistrals may segfault at 128k
on ROCm — keep them at 65536:

```yaml
- name: mistral-medium
  ctx: 65536
```

### Image generation fails (sd-server)

**Symptom**: `/v1/images/generations` returns error or blank image.

**Check**:
```bash
docker compose logs llama-swap | grep sd-server
```

**Common causes**:
- Checkpoint file not downloaded or corrupted
- Model resolution mismatch (SDXL needs 1024x1024)

**Fix**:
- Re-download the checkpoint: `make models-apply`
- Verify the checkpoint is a valid safetensors file

### LiteLLM sync fails

**Symptom**: `make sync-litellm` errors or doesn't register models.

**Check**:
```bash
# Verify LiteLLM is running
curl http://localhost:4000/health

# Verify API key
echo $LITELLM_API_KEY
```

**Common causes**:
- Wrong API key scope (admin key vs inference key)
- LiteLLM running on different host (update `MODEL_API_BASE`)
- LiteLLM model drift (models registered manually, not via sync)

**Fix**:
```bash
# Use master key for sync
export LITELLM_MASTER_KEY=sk-...

# If LiteLLM runs on another host, update api_base
make sync-litellm MODEL_API_BASE=http://10.0.0.5:11434/v1

# Force clean rebuild
make sync-litellm RESET=1
```

### Provider gotcha: ollama vs openai

**Symptom**: LiteLLM returns `Ollama_chatException` when proxying to llama-swap.

**Cause**: llama-swap is OpenAI-compatible, not Ollama. Models in LiteLLM must
use the `openai/` provider with an `/v1` `api_base`.

**Fix**: `sync-litellm` registers the right provider automatically. If you
manually added models, use:

```yaml
litellm_params:
  model: openai/model-name
  api_base: http://localhost:11434/v1
```

### Slow performance

**Check**:
```bash
# Verify GPU is being used (not CPU fallback)
docker compose exec llama-swap rocminfo | grep "Profile:"
# Should show FULL, not SMALL or CPU

# Check GPU utilization
rocm-smi --getgpuutil
```

**Fix**:
- Ensure `-ngl 99` is set (all layers on GPU)
- Verify `HIP_VISIBLE_DEVICES` is set correctly
- Check that ROCm is using the right backend (HIP, not CPU)

### Docker build fails

**Symptom**: `docker compose build` fails during llama.cpp compilation.

**Common causes**:
- Out of memory during build
- Network timeout downloading dependencies

**Fix**:
```bash
# Build with more parallelism
docker compose build --build-arg MAKEFLAGS="-j$(nproc)"

# If out of memory, reduce parallelism
docker compose build --build-arg MAKEFLAGS="-j2"

# If network timeout, retry
docker compose build --no-cache
```
