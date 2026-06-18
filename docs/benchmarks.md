# Benchmarks

Measured throughput for models served by this stack, with a cross-stack
comparison against the SYCL/Intel sibling
([battlemage-llama](https://github.com/james-huston/battlemage-llama)) running
the **same GGUF files**.

The headline finding: on an 8 GB RDNA2 card, a small **Mixture-of-Experts**
model (LFM2-8B-A1B) decodes ~3.5× faster than a dense 8B at the same VRAM
footprint — and closes most of the gap to a GPU 4× its price, because MoE decode
is bound by *active* parameters (~1.5 B) rather than memory bandwidth.

## Method

- Tool: `llama-bench` (built into each stack's llama.cpp image).
- Metrics: `pp512` = prompt/prefill throughput over a 512-token prompt;
  `tg128` = decode/generation throughput over 128 tokens. Both in tokens/sec,
  mean ± stddev over 3 repetitions.
- Flags: `-ngl 99 -sm none -fa <on|1> -p 512 -n 128 -r 3`. Full GPU offload,
  single device, flash-attention on, FP16 KV cache (no KV quantization).
- One model resident at a time; numbers are steady-state GPU compute, not
  cold-load.

### Test environment

| Stack | GPU | Arch | VRAM | Backend | llama.cpp build |
|-------|-----|------|------|---------|-----------------|
| rocm-llama (this repo) | Radeon RX 6650 XT | RDNA2 / gfx1030 (gfx1032 via `HSA_OVERRIDE_GFX_VERSION=10.3.0`) | 8 GB (8176 MiB) | ROCm/HIP 7.2.1 | `fdc3db9` |
| battlemage-llama | Arc Pro B70 | Xe2 / SYCL | 32 GB (32656 MiB) | oneAPI/SYCL | `da3f990` |

Measured 2026-06-17. The two GPUs share a host and read from the same model
store, so the only variables are the GPU and its backend.

## Results — RX 6650 XT (8 GB, ROCm/HIP)

| Model | Arch | Quant | File size | VRAM used | Prefill (pp512) | Decode (tg128) |
|-------|------|-------|-----------|-----------|-----------------|----------------|
| **LFM2-8B-A1B** | MoE 8.3 B / 1.5 B active | Q5_K_M | 5.51 GiB | ~6.0 GB @ 8K | **2508 t/s** | **157.0 t/s** |
| Gemma 4 E4B (it) | dense, 7.52 B / E4B effective | Q6_K | 6.57 GiB | 4.77 GB @ 32K | 1069 t/s | 50.4 t/s |
| Ministral 3 8B Instruct (2512) | dense 8.49 B | Q4_K_M | 4.83 GiB | ~5.5 GB @ 16K | 697 t/s | 44.7 t/s |
| llama3.1-8b *(reference)* | dense 8 B | Q4_K_M | 4.58 GiB | 7.12 GB @ 16K | ~226 t/s | ~47 t/s |
| gemma-4-12b *(reference, squeeze)* | dense 12 B | Q4_K_M | 7.12 GiB | 8171/8176 MiB @ 2K | — | ~30 t/s |

Notes:
- **gemma-4-12b is a "runs at all" stretch** — fits only at `-c 2048 -fa on`
  with ~5 MiB headroom. This card is fundamentally 8B-class; the models above
  are the comfortable operating range.
- **Gemma 4 E4B uses Per-Layer-Embedding (PLE) offload**: despite the 6.57 GiB
  file, only the ~4 B effective params load to VRAM (PLE weights stay in host
  RAM), so it uses just 4.77 GB even at 32K context — lots of headroom.

## Cross-stack comparison — 6650 XT vs Arc Pro B70

Same GGUFs, same `llama-bench` flags, the two GPUs side by side:

| Model | 6650 XT prefill | B70 prefill | B70 prefill lead | 6650 XT decode | B70 decode | B70 decode lead |
|-------|-----------------|-------------|------------------|----------------|------------|-----------------|
| Ministral 3 8B (dense) | 697 t/s | 2325 t/s | 3.34× | 44.7 t/s | 78.5 t/s | **1.76×** |
| Gemma 4 E4B (dense) | 1069 t/s | 2838 t/s | 2.65× | 50.4 t/s | 68.5 t/s | 1.36× |
| **LFM2-8B-A1B (MoE)** | 2508 t/s | 4146 t/s | 1.65× | 157.0 t/s | 179.1 t/s | **1.14×** |

## Why the MoE "decodes strangely well" here

MoE decode speed scales with **active** parameters, but the **full** expert set
must be VRAM-resident — so *total* params decide whether it fits, and *active*
params decide how fast it runs. LFM2-8B-A1B is 8.3 B total (fits at Q5 in ~6 GB)
but only 1.5 B active per token.

The consequence is visible in the decode-lead column above: the B70's advantage
collapses from **1.76× on dense Ministral to 1.14× on LFM2**. Dense decode is
memory-bandwidth-bound, where the B70's wider, faster memory dominates. MoE
decode touches far fewer weights per token, so it leans on compute and latency
instead — and there the cheaper RDNA2 card nearly keeps pace. **The 6650 XT
punches up hardest on MoE**, which is exactly the workload to favor on a
budget 8 GB card.

The practical upshot for this card:
- **LFM2-8B-A1B is the fast daily driver** — 157 t/s decode, ~3.5× the dense
  8B models at the same footprint, with dense-7B-class quality.
- **Ministral 3 8B** is the no-surprises dense upgrade over llama3.1-8b (same
  envelope, ~equal speed, newer model).
- **Gemma 4 E4B** is the long-context / comfort pick — 32K context fits easily.

### The trap (why the famous big MoEs are *not* here)

The well-known MoEs — Qwen3-30B-A3B, gpt-oss-20B, GLM-4.7-Flash — are 20–106 B
*total*. At any usable quant they're 11–18 GB+, far over 8 GB, forcing
`--n-cpu-moe` offload that drops decode to RAM-bandwidth speeds (~10–20 t/s) and
erases the entire MoE advantage. Those belong on the 32 GB B70. Only
*small-total* MoEs (≤ ~13 B) deliver the effect on this card.

## Reproducing

```bash
# On either stack, inside the container:
/opt/llama-cpp/bin/llama-bench \
  -m /models/lfm2-8b-a1b/Q5_K_M.gguf \
  -dev ROCm0 -ngl 99 -sm none -fa on \
  -p 512 -n 128 -r 3 -o md
# Battlemage equivalent: -dev SYCL0 -fa 1
```
