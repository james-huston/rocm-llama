# rocm-llama

A Docker-based stack for running local LLMs on AMD ROCm GPUs — designed around
**AMD Radeon RX 7900 XTX** (RDNA3) and **Radeon RX 7800/7700 XT**, with support
for the **Radeon RX 6000 series** (RDNA2) and **Instinct MI300X** (CDNA3).
Wraps [llama.cpp](https://github.com/ggml-org/llama.cpp) with the
[llama-swap](https://github.com/mostlygeek/llama-swap) proxy for on-demand model
switching via an OpenAI-compatible API.

Drops in where Ollama would sit on port 11434, keeps the same Ollama-style
model-swap ergonomics, but uses AMD's native ROCm/HIP backend for significantly
better performance on RDNA3/CDNA3 GPUs.

## Why this exists

When I put an AMD GPU into a Linux workstation, the "just works" options were
either slow or impractical:

- **Ollama + Vulkan** runs on AMD, but at lower throughput than native ROCm on
  RDNA3 hardware.
- **Native-host llama.cpp build** works and hits great numbers, but requires
  installing the full ROCm toolkit (~2+ GB), configuring environment variables,
  and generally making a mess of the host system.
- **Pre-built llama.cpp ROCm images** (`ghcr.io/ggml-org/llama.cpp:server-rocm`)
  are useful but don't include llama-swap for model swapping, and are pinned to
  specific ROCm versions that may lag behind GPU support.

This repo is the cleanup: the ROCm + llama.cpp + llama-swap toolchain is
self-contained in a single Docker image. The host stays clean — all it needs is
the `amdgpu` kernel driver and Docker.

## Hardware support

Built and tested for:

- **AMD Radeon RX 7900 XTX** (RDNA3, 20 GB) ✓
- **AMD Radeon RX 7800 XT / 7700 XT** (RDNA3, 16 GB) ✓
- **AMD Radeon RX 6700 XT / 6800 / 6900 XT** (RDNA2, 10–16 GB) ✓ (may need `HSA_OVERRIDE_GFX_VERSION`)
- **AMD Instinct MI300X** (CDNA3, 192 GB) ✓

GPUs other than AMD Radeon/Instinct won't benefit from this setup. For NVIDIA,
use llama.cpp's CUDA builds; for Intel, use the SYCL backend (see
[battlemage-llama](https://github.com/james-huston/battlemage-llama)).

## Prerequisites

On the **host**:

- Linux with a recent kernel that has the `amdgpu` driver (Ubuntu 24.04+ is
  known-good).
- [AMD ROCm installed on the host](docs/host-setup.md) — the `amdgpu` kernel
  module must be loaded, `/dev/kfd` and `/dev/dri` must exist, and `rocminfo`
  should list your GPU.
- Docker Engine with Compose (`docker compose version` >= v2).
- Your user added to the `render` and `video` groups.

Check your render/video GIDs:

```bash
getent group render video
```

If they're not `104` and `39` (Fedora defaults), copy `.env.example` to `.env`
and set `RENDER_GID` / `VIDEO_GID`.

## Quick start

```bash
git clone https://github.com/james-huston/rocm-llama.git
cd rocm-llama

# Adjust GIDs / model paths if needed
cp .env.example .env
$EDITOR .env

# Copy the example config and list your models
cp config/llama-swap.example.yaml config/llama-swap.yaml
./scripts/find-gguf-blobs.sh   # discovers Ollama blob paths on your host

# Paste the blob paths from find-gguf-blobs.sh into config/llama-swap.yaml

# Build and start (first build takes 10-20 min — pulls ROCm base, compiles llama.cpp)
docker compose up -d --build

# Verify the GPU is visible to ROCm inside the container
docker compose exec llama-swap rocminfo | grep gfx
# Expect a line with your GPU's gfx code (e.g. gfx1100 for RX 7900 XTX)

# Smoke test
curl http://localhost:11434/v1/models | jq
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-coder-30b","messages":[{"role":"user","content":"hello"}]}'
```

The first request to a given model takes 10-30 seconds (cold start — llama-swap
spawns llama-server, which mmaps the GGUF and initializes ROCm kernels).
Subsequent requests against the same model are instant. Switching models unloads
the old one and starts the new one.

## Architecture

```
Clients  (LiteLLM, Continue.dev, Open WebUI, scripts, ...)
    │
    ▼  OpenAI-compatible /v1/chat/completions etc.
┌───────────────────────────────────────────────────┐
│  Container: rocm-llama                            │
│                                                   │
│   llama-swap  :11434   ← /config/llama-swap.yaml  │
│       │                                           │
│       │ spawns / stops on demand per request      │
│       ▼                                           │
│   llama-server  :12800+  (HIP/ROCm backend, -ngl 99) │
│       │                                           │
└───────┼───────────────────────────────────────────┘
        ▼  /dev/kfd + /dev/dri passthrough
    AMD Radeon / Instinct GPU
```

## Bring your own GGUFs

You do NOT need Ollama installed. The default `docker-compose.yml` bind-mounts
`/opt/apps/ollama-models` because that's a convenient place to get GGUFs if
you've already pulled them via Ollama — llama.cpp reads Ollama's content-addressed
blobs natively.

If you downloaded GGUFs from Hugging Face directly, point `MODELS_DIR` at
whatever directory holds them (set in `.env`) and adjust the `--model` paths in
your `config/llama-swap.yaml` accordingly. Inside the container, whatever you
bind-mount shows up at `/models/`.

## Managing models declaratively (`models.yaml`)

`models.yaml` (repo root) is the **source of truth** for what we run and where
each GGUF came from. Each entry records the install source (an Ollama blob
`path:`, or a HF `repo:` + `file:`) and the llama-swap runtime params (`ctx`,
`template`, `reasoning` / `reasoning_format`, `temp`, `top_p`). `enabled: false`
keeps a model documented as a candidate without installing or serving it.

Entries can also carry **LiteLLM metadata**: `decode_tps` (measured decode
tok/s) and a `litellm:` block (`description`, `supports_function_calling`,
`supports_tool_choice`, `supports_reasoning`, …), plus a top-level `cost:` block.
`make sync-litellm` turns these into LiteLLM `model_info` — including per-token
costs derived from `decode_tps` (see [Syncing to LiteLLM](#syncing-models-to-a-litellm-proxy)).

```bash
# Edit models.yaml, then:
make models-apply              # download any missing enabled GGUF + regenerate the config
make models-apply DRY_RUN=1    # preview the plan + generated config, write nothing
```

`make models-apply` downloads what's missing and **regenerates**
`config/llama-swap.yaml` from the enabled entries (a `.bak` is kept). That file
is now a generated artifact — edit `models.yaml`, not the config. It also reports
GGUF dirs in `MODELS_DIR` that no enabled model references, so stale downloads are
easy to spot. Follow with `make sync-litellm` and `make test-models`. Requires
PyYAML (`pip install pyyaml`).

## Adding models ad-hoc with `make`

For a quick one-off (without editing `models.yaml`), `make add-model` wraps
`scripts/add-model.sh` to download a GGUF from Hugging Face into `MODELS_DIR` and
*append* a ready-to-run block (ROCm defaults — `-ngl 99 --device HIP0 --jinja`)
to `config/llama-swap.yaml`. Note the next `make models-apply` will overwrite
such appends, so fold anything you want to keep into `models.yaml`:

```bash
# Download + register in one step
make add-model \
    REPO=unsloth/GLM-4.7-Flash-GGUF \
    FILE=GLM-4.7-Flash-Q4_K_XL.gguf \
    NAME=glm-4.7-flash-q4 DIR=glm-4.7-flash OUT=Q4_K_XL.gguf \
    CTX=131072 TEMPLATE=glm-4.7-flash.jinja REASONING=1 TEMP=0.6 TOP_P=0.95

make list-models            # show aliases already in the config
make help                   # all targets + the full variable list
```

| Var | Meaning |
| --- | --- |
| `REPO` / `FILE` | Hugging Face repo id and filename (`FILE` may be a glob/split set — needs the `hf` / `huggingface-cli`; otherwise `curl`/`wget` fetches a single file) |
| `NAME` | model alias (the OpenAI `model` id) |
| `DIR` / `OUT` | subdir under `MODELS_DIR` (default `NAME`) and save-as filename (default the repo filename) |
| `CTX` | `-c` context size — keep dense 24B Mistrals at `65536` (may segfault at 128k on ROCm) |
| `TEMPLATE` | a Jinja file in `templates/` → `--chat-template-file` |
| `REASONING=1`, `TEMP`, `TOP_P`, `EXTRA` | add `--reasoning-format deepseek`, samplers, or extra flags |

Other entrypoints: `make download-model` (fetch only), `make add-config`
(register an already-present GGUF, e.g. an Ollama blob via `MODEL_PATH=`), and
`make find-blobs` (the original Ollama blob mapper). Add `DRY_RUN=1` to preview
the config block without writing it. llama-swap hot-reloads the config, so the
next request to the new alias spawns it — then validate with
`./tests/tool_use/run.sh <NAME>`.

## Syncing models to a LiteLLM proxy

If you front this stack with [LiteLLM](https://github.com/BerriAI/litellm),
`scripts/sync-litellm.py` (`make sync-litellm`) makes LiteLLM mirror what
llama-swap serves. It reads `GET {upstream}/v1/models`, then **adds** missing
models, **deletes** LiteLLM-managed models no longer served, and **re-points**
any whose registration drifted. Models baked into LiteLLM's static `config.yaml`
(not DB-managed) are left untouched.

Each model is registered with the **`openai/` provider** + an `/v1` `api_base`,
and enriched with `model_info` drawn from `models.yaml` (shown in LiteLLM's Model
Hub UI and used for routing/validation):

- **`description`** and the **`supports_function_calling` / `supports_tool_choice` /
  `supports_reasoning` / `supports_vision`** capability flags — from each entry's
  `litellm:` block. (So e.g. a code-only model can be flagged "no tool calling.")
- **`max_input_tokens`** — from the entry's `ctx`.
- **`mode: chat`** (fleet default).
- **`input_cost_per_token` / `output_cost_per_token`** — *derived* per model from
  its `decode_tps` and the manifest's top-level `cost:` block, which prices GPU
  time as electricity × a hardware-amortization multiplier:
  ```yaml
  cost:
    watts: 300          # GPU draw under load (RX 7900 XTX TBP ~355W, MI300X ~750W)
    usd_per_kwh: 0.18   # your electricity rate
    amortization: 3     # capital + wear multiplier over raw electricity
    input_factor: 4     # prefill is ~4x faster than decode -> input = output/4
  ```
  `output $/token = watts/1000 × (1/decode_tps/3600) × usd_per_kwh × amortization`.
  Change a knob and the next sync re-prices the whole fleet (slower models cost
  more per token).

```bash
# Put the admin key in gitignored .env (LITELLM_API_KEY=sk-...), then:
make sync-litellm DRY_RUN=1          # preview the plan, change nothing
make sync-litellm                    # add new, re-point drifted, delete stale
```

| Var / env | Meaning |
| --- | --- |
| `LITELLM_API_KEY` / `LITELLM_MASTER_KEY` | LiteLLM admin key (env or `.env`; never printed). Required. |
| `LITELLM` / `LITELLM_URL` | LiteLLM proxy base URL (default `http://localhost:4000`) |
| `UPSTREAM` / `UPSTREAM_URL` | llama-swap base URL the script reads `/v1/models` from (default `http://localhost:11434`) |
| `API_BASE` / `MODEL_API_BASE` | `api_base` baked into each LiteLLM model. **If LiteLLM runs on another host, this must be routable from there** — use the model server's IP/hostname, not `localhost` (e.g. `http://10.0.0.5:11434/v1`). |
| `NO_DELETE=1` | only add/re-point, never delete |
| `RESET=1` | delete ALL DB-managed models first, then re-add (clean rebuild) |

Run it after `make models-apply` to keep LiteLLM in lockstep. The `model_info`
enrichment (descriptions, flags, costs) needs PyYAML (`pip install pyyaml`);
without it the model list still syncs.

> **Provider gotcha:** llama-swap is OpenAI-compatible, *not* Ollama. Models in
> LiteLLM must use the `openai/` provider with an `/v1` `api_base` — an
> `ollama`/`ollama_chat` provider pointed at port 11434 fails with
> `Ollama_chatException`. `sync-litellm` registers the right provider; see
> [`docs/troubleshooting.md`](docs/troubleshooting.md).

> **Admin vs inference keys:** LiteLLM's `model/*` management API and chat
> inference use different key scopes. A management-only virtual key can sync
> models but gets a 403 on `/v1/chat/completions` — so `make test-models VIA=litellm`
> needs a key allowed to run inference (or the master key), not just the admin key
> used for syncing.

## Smoke-testing models

`scripts/test-models.py` (`make test-models`) cycles through every model an
endpoint advertises, sends a real chat query to each (which forces llama-swap to
cold-load it), and reports pass/fail — handy for confirming a fresh model works
or bisecting which registered models are broken.

```bash
make test-models                       # test every model on llama-swap directly
make test-models VIA=litellm           # test each model through the LiteLLM proxy
make test-models MODELS=glm-4.7-flash-q4   # just one (or a space-separated subset)
```

Because only one model fits in VRAM at a time, it's sequential and each model is
a cold start (expect ~10-30s apiece). A model that loads but returns no visible
text — usually a thinking model that spent the token budget reasoning — is
reported as `WARN`, not `FAIL`; bump `MAX_TOKENS` for those. Exit code is
non-zero if any model fails. Knobs: `VIA`, `BASE`, `API_KEY`, `MAX_TOKENS`,
`TIMEOUT`, `PROMPT` (also Python 3 stdlib only).

## Image generation

llama-swap also serves **text-to-image** on demand: this stack builds
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
`sd-server` with the HIPBLAS backend, and `models.yaml` declares an image model
with `engine: sd-server`. llama-swap swaps it in like any LLM and proxies the
OpenAI `POST /v1/images/generations` endpoint (which the web UI's image section
also drives).

The default `sdxl` entry serves Stable Diffusion XL base 1.0:

```bash
curl -s http://localhost:11434/v1/images/generations \
    -H 'Content-Type: application/json' \
    -d '{"model":"sdxl","prompt":"a red cube on a wooden table, studio lighting","size":"1024x1024"}' \
    | jq -r '.data[0].b64_json' | base64 -d > out.png
```

Add another image model by copying the `sdxl` entry in `models.yaml`
(`engine: sd-server`, point `repo`/`file` at a checkpoint) and running
`make models-apply`.

**Civitai sources** are supported alongside Hugging Face — set `CIVITAI_API_KEY`
in `.env` (token from [civitai.com/user/account](https://civitai.com/user/account))
and use `repo: civitai:<modelVersionId>` on the manifest entry. Find the version
id by opening the model on Civitai and selecting a version — it's **not** the
model id in the user-facing URL `civitai.com/models/<modelId>/…`. `make
add-model REPO=civitai:<id> NAME=… DIR=… OUT=…` works the same way ad-hoc
(downloads the version's `primary` file, with Bearer auth and SHA256
verification).

## Documentation

- [`docs/host-setup.md`](docs/host-setup.md) — host ROCm / kernel setup for AMD GPUs
- [`docs/upgrading.md`](docs/upgrading.md) — bump llama.cpp / llama-swap / ROCm version
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — common issues and diagnostics
- [`docs/improvements/`](docs/improvements/) — planned/in-progress improvements as BMAD stories

## Contributing

Issues and PRs welcome. I'm particularly interested in:

- Testing reports from other AMD GPUs (RX 7600, RX 6700 XT, MI300X, etc.)
- Performance tuning flags that actually help on RDNA3 / RDNA2
- `HSA_OVERRIDE_GFX_VERSION` guidance for older GPUs not natively supported by the pinned ROCm version

## License

MIT. See [LICENSE](LICENSE).

## Credits

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — the inference engine.
- [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap) — the model-swap proxy.
- [ROCm](https://rocm.docs.amd.com) — AMD's open GPU computing platform.
- [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — the image generation engine.
