# Improvements

Planned and in-progress improvements for rocm-llama.

## GPU monitoring dashboard

- [ ] Add a lightweight Grafana dashboard for ROCm GPU metrics
- [ ] Expose `rocm-smi` data via a Prometheus endpoint
- [ ] Show per-model GPU utilization in the logs

## Auto-scaling context size

- [ ] Detect model size and automatically pick a safe `ctx` value
- [ ] Warn when `ctx` is likely to cause OOM or segfaults
- [ ] Test context sizes against known-good values per model family

## Multi-GPU support

- [ ] Support `HIP_VISIBLE_DEVICES=0,1` for multi-GPU setups
- [ ] Auto-detect available GPUs and distribute models across them
- [ ] Document multi-GPU docker-compose configuration

## Model quantization presets

- [ ] Add pre-validated quantization presets per model family
- [ ] Auto-select quantization based on available VRAM
- [ ] Benchmark Q4_K_M vs Q5_K_M vs Q8_0 per model on RDNA3

## Web UI integration

- [ ] Pre-configured Open WebUI docker-compose setup
- [ ] Auto-discovery of models in the web UI
- [ ] Image generation UI integration for sd-server models

## Health checks

- [ ] Add Docker healthcheck for llama-server readiness
- [ ] Auto-restart crashed models
- [ ] Alert on repeated model load failures

## Benchmark automation

> Current results are recorded manually in [`../benchmarks.md`](../benchmarks.md)
> (RX 6650 XT, with an Arc Pro B70 cross-stack comparison). This section tracks
> automating their collection.

- [ ] Run `make test-models` with timing and report TPS
- [ ] Store benchmark results in models.yaml for cost derivation
- [ ] Compare performance across ROCm versions

## Docker image optimization

- [ ] Multi-stage build to reduce final image size
- [ ] Cache llama.cpp build artifacts between builds
- [ ] Pre-built images for common GPU targets (gfx1100, gfx1030, gfx942)

## Model recommendation engine

- [ ] Suggest models based on available VRAM
- [ ] Auto-generate models.yaml entries from Hugging Face trending GGUFs
- [ ] One-click "install best model for my GPU"
