# rocm-llama — Makefile
#
# Targets:
#   models-apply          download missing enabled GGUFs + regenerate llama-swap config
#   add-model             ad-hoc download + register (without editing models.yaml)
#   download-model        fetch a GGUF from HF only
#   add-config            register an already-present GGUF into llama-swap config
#   find-blobs            discover Ollama blob paths on the host
#   list-models           show aliases currently in the config
#   sync-litellm          mirror llama-swap models into a LiteLLM proxy
#   test-models           smoke-test every model on an endpoint
#   status                GPU + container status
#   help                  this message + all variables

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -ec

# ---------------------------------------------------------------------------
# Variables (override on the command line: make add-model REPO=... NAME=...)
# ---------------------------------------------------------------------------

# --- models-apply / add-model ---
REPO ?=
FILE ?=
NAME ?=
DIR ?= $(NAME)
OUT ?= $(FILE)
CTX ?= 8192
TEMPLATE ?=
REASONING ?=
TEMP ?=
TOP_P ?=
EXTRA ?=
DRY_RUN ?=

# --- sync-litellm ---
LITELLM_API_KEY ?=
LITELLM_MASTER_KEY ?=
LITELLM ?=
LITELLM_URL ?= http://localhost:4000
UPSTREAM ?=
UPSTREAM_URL ?= http://localhost:11435
API_BASE ?=
MODEL_API_BASE ?= http://localhost:11435/v1
# Ownership tag so a shared LiteLLM proxy can host several stacks without their
# syncs deleting each other's models. Defaults to the top-level `stack:` in
# models.yaml when unset; override with `make sync-litellm STACK=rocm`.
STACK ?=
NO_DELETE ?=
RESET ?=

# --- test-models ---
VIA ?=
BASE ?=
API_KEY ?=
MAX_TOKENS ?= 1024
TIMEOUT ?= 120
PROMPT ?= Say one word only: hello

# --- status ---

# --- paths ---
MODELS_DIR ?= /opt/apps/ollama-models
MODELS_YAML ?= models.yaml
CONFIG_DIR ?= config
LLAMA_SWAP_CONFIG ?= $(CONFIG_DIR)/llama-swap.yaml
LLAMA_SWAP_CONFIG_BAK ?= $(LLAMA_SWAP_CONFIG).bak
TEMPLATES_DIR ?= templates
SCRIPTS ?= scripts

.PHONY: help models-apply add-model download-model add-config find-blobs list-models sync-litellm test-models status

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$|^[# ]+# ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | \
		sort; \
	echo; \
	echo "Variables:"; \
	(grep -E '^\w+\s*\?=' $(MAKEFILE_LIST) || true) | \
		awk '{printf "  %-25s = %s\n", $$1, $$3}' | \
		sort

models-apply: ## Download missing enabled GGUFs + regenerate llama-swap config
	@if [ ! -f $(MODELS_YAML) ]; then \
		echo "ERROR: $(MODELS_YAML) not found"; exit 1; \
	fi
	@$(SCRIPTS)/apply-models.py \
		--manifest $(MODELS_YAML) \
		--models-dir "$(MODELS_DIR)" \
		--config $(LLAMA_SWAP_CONFIG) \
		--config-bak $(LLAMA_SWAP_CONFIG_BAK) \
		--templates-dir $(TEMPLATES_DIR) \
		$(if $(DRY_RUN),--dry-run)

add-model: ## Ad-hoc: download + register a GGUF (without editing models.yaml)
	@if [ -z "$(REPO)" ] || [ -z "$(NAME)" ]; then \
		echo "ERROR: REPO and NAME are required"; \
		echo "Usage: make add-model REPO=... NAME=... DIR=... OUT=..."; \
		exit 1; \
	fi
	@$(SCRIPTS)/add-model.sh \
		--repo "$(REPO)" \
		--file "$(FILE)" \
		--name "$(NAME)" \
		--dir "$(DIR)" \
		--out "$(OUT)" \
		--ctx $(CTX) \
		$(if $(TEMPLATE),--template "$(TEMPLATE)") \
		$(if $(REASONING),--reasoning) \
		$(if $(TEMP),--temp "$(TEMP)") \
		$(if $(TOP_P),--top-p "$(TOP_P)") \
		$(if $(EXTRA),--extra "$(EXTRA)") \
		$(if $(DRY_RUN),--dry-run)

download-model: ## Download a GGUF from HF only
	@if [ -z "$(REPO)" ] || [ -z "$(FILE)" ]; then \
		echo "ERROR: REPO and FILE are required"; \
		exit 1; \
	fi
	@$(SCRIPTS)/add-model.sh \
		--repo "$(REPO)" \
		--file "$(FILE)" \
		--name "$(NAME)" \
		--dir "$(DIR)" \
		--out "$(OUT)" \
		--download-only \
		$(if $(DRY_RUN),--dry-run)

add-config: ## Register an already-present GGUF into llama-swap config
	@if [ -z "$(NAME)" ]; then \
		echo "ERROR: NAME is required"; \
		exit 1; \
	fi
	@$(SCRIPTS)/add-model.sh \
		--name "$(NAME)" \
		$(if $(MODEL_PATH),--model-path "$(MODEL_PATH)") \
		--dir "$(DIR)" \
		--out "$(OUT)" \
		--ctx $(CTX) \
		$(if $(TEMPLATE),--template "$(TEMPLATE)") \
		$(if $(REASONING),--reasoning) \
		$(if $(TEMP),--temp "$(TEMP)") \
		$(if $(TOP_P),--top-p "$(TOP_P)") \
		$(if $(EXTRA),--extra "$(EXTRA)") \
		$(if $(DRY_RUN),--dry-run)

find-blobs: ## Discover Ollama blob paths on the host
	@$(SCRIPTS)/find-gguf-blobs.sh

list-models: ## Show aliases currently in the config
	@if [ ! -f $(LLAMA_SWAP_CONFIG) ]; then \
		echo "ERROR: $(LLAMA_SWAP_CONFIG) not found"; exit 1; \
	fi
	@python3 -c "\
	import yaml, sys; \
	cfg = yaml.safe_load(open('$(LLAMA_SWAP_CONFIG)')); \
	servers = cfg.get('llama-servers', cfg.get('llama_servers', {})); \
	for k in sorted(servers.keys()): \
		print(k)"

sync-litellm: ## Mirror llama-swap models into a LiteLLM proxy
	@$(SCRIPTS)/sync-litellm.py \
		--upstream "$(UPSTREAM_URL)" \
		--litellm "$(LITELLM_URL)" \
		$(if $(API_BASE),--api-base "$(API_BASE)") \
		$(if $(STACK),--stack "$(STACK)") \
		$(if $(NO_DELETE),--no-delete) \
		$(if $(RESET),--reset) \
		$(if $(DRY_RUN),--dry-run)

test-models: ## Smoke-test every model on an endpoint
	@$(SCRIPTS)/test-models.py \
		--via "$(VIA)" \
		--base "$(BASE)" \
		$(if $(API_KEY),--api-key "$(API_KEY)") \
		--max-tokens $(MAX_TOKENS) \
		--timeout $(TIMEOUT) \
		--prompt "$(PROMPT)" \
		$(if $(MODELS),--models "$(MODELS)")

status: ## GPU + container status
	@echo "=== Docker containers ===" && \
	docker compose ps 2>/dev/null || echo "docker compose not found"; \
	echo; \
	echo "=== ROCm GPU info (host) ===" && \
	rocminfo 2>/dev/null | grep -E "Name:|gfx|Profile:" || echo "rocminfo not available"; \
	echo; \
	echo "=== ROCm SMI (host) ===" && \
	rocm-smi 2>/dev/null || echo "rocm-smi not available"; \
	echo; \
	echo "=== Container GPU check ===" && \
	docker compose exec llama-swap rocminfo 2>/dev/null | grep -E "Name:|gfx" || echo "container not running"
