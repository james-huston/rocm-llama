# Upgrading

## Upgrading llama.cpp

Pin to a specific commit for reproducible builds:

```bash
docker compose build --build-arg LLAMA_CPP_REF=b9409
```

Or use `master` for the latest:

```bash
docker compose build --build-arg LLAMA_CPP_REF=master
```

## Upgrading llama-swap

Override the version at build time:

```bash
docker compose build --build-arg LLAMA_SWAP_VERSION=218
```

Check [llama-swap releases](https://github.com/mostlygeek/llama-swap/releases)
for the latest version number.

## Upgrading ROCm base image

The Dockerfile uses `rocm/dev-ubuntu-24.04:${ROCM_IMAGE_TAG}`. Default is
`7.2.1-complete`. To upgrade:

```bash
docker compose build --build-arg ROCM_IMAGE_TAG=7.3.0-complete
```

Check [ROCm Docker images](https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags)
for available tags.

## Full rebuild

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

## After upgrading

1. Run `make test-models` to verify all models still work.
2. Run `make sync-litellm` to re-sync model metadata to LiteLLM.
3. Check `rocm-smi` for any changes in GPU behavior.

## Known upgrade issues

### ROCm version mismatch between llama.cpp and sd-server

Both are built in the same Dockerfile, so they share the same ROCm libraries.
If you upgrade the ROCm base image, both will use the new version.

### llama-swap config regeneration

`make models-apply` regenerates `config/llama-swap.yaml`. If you've made manual
edits to this file, back it up first (it's auto-backed up to `.bak`).

### GPU target rebuild

If you change `GPU_TARGETS` (e.g. from `gfx1100` to `gfx1030`), you need a
full rebuild:

```bash
docker compose build --build-arg GPU_TARGETS=gfx1030 --no-cache
```
