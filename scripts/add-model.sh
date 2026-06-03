#!/usr/bin/env bash
# add-model.sh — ad-hoc: download a GGUF from HF/Civitai and append a config block
#
# Usage:
#   make add-model REPO=... NAME=... DIR=... OUT=...
#   ./scripts/add-model.sh --repo unsloth/GLM-4.7-Flash-GGUF --file GLM-4.7-Flash-Q4_K_XL.gguf \
#       --name glm-4.7-flash-q4 --dir glm-4.7-flash --ctx 131072 --reasoning
#
# This appends to config/llama-swap.yaml WITHOUT overwriting existing entries.
# The next `make models-apply` will regenerate the config from models.yaml,
# so fold anything you want to keep into models.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$ROOT_DIR/config/llama-swap.yaml"
TEMPLATES_DIR="$ROOT_DIR/templates"

# Defaults
REPO=""
FILE=""
NAME=""
DIR=""
OUT=""
MODEL_PATH=""
CTX=8192
TEMPLATE=""
REASONING=false
TEMP=""
TOP_P=""
EXTRA=""
DOWNLOAD_ONLY=false
DRY_RUN=false

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo) REPO="$2"; shift 2 ;;
            --file) FILE="$2"; shift 2 ;;
            --name) NAME="$2"; shift 2 ;;
            --dir) DIR="$2"; shift 2 ;;
            --out) OUT="$2"; shift 2 ;;
            --model-path) MODEL_PATH="$2"; shift 2 ;;
            --ctx) CTX="$2"; shift 2 ;;
            --template) TEMPLATE="$2"; shift 2 ;;
            --reasoning) REASONING=true; shift ;;
            --temp) TEMP="$2"; shift 2 ;;
            --top-p) TOP_P="$2"; shift 2 ;;
            --extra) EXTRA="$2"; shift 2 ;;
            --download-only) DOWNLOAD_ONLY=true; shift ;;
            --dry-run) DRY_RUN=true; shift ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

download_hf() {
    local repo="$1"
    local filename="$2"
    local dest="$3"
    local hf_token="${HF_TOKEN:-}"

    mkdir -p "$dest"
    local dest_file="$dest/${OUT:-$filename}"

    if [[ -f "$dest_file" ]]; then
        echo "  File already exists: $dest_file"
        return 0
    fi

    echo "  Downloading $repo/$filename -> $dest_file"
    local url="https://huggingface.co/$repo/resolve/main/$filename"

    if command -v curl &>/dev/null; then
        if [[ -n "$hf_token" ]]; then
            curl -L -o "$dest_file" -H "Authorization: Bearer $hf_token" "$url"
        else
            curl -L -o "$dest_file" "$url"
        fi
    elif command -v wget &>/dev/null; then
        wget -O "$dest_file" "$url"
    else
        echo "  ERROR: curl or wget required" >&2
        return 1
    fi

    echo "  Downloaded: $dest_file"
}

download_civitai() {
    local model_version_id="$1"
    local dest="$2"
    local api_key="${CIVITAI_API_KEY:-}"

    if [[ -z "$api_key" ]]; then
        echo "  ERROR: CIVITAI_API_KEY not set" >&2
        return 1
    fi

    mkdir -p "$dest"
    local dest_file="$dest/${OUT:-civitai_${model_version_id}.gguf}"

    if [[ -f "$dest_file" ]]; then
        echo "  File already exists: $dest_file"
        return 0
    fi

    echo "  Fetching Civitai version $model_version_id"
    local api_url="https://civitai.com/api/v1/model-versions/$model_version_id"

    if command -v curl &>/dev/null; then
        local json
        json=$(curl -s -H "Authorization: Bearer $api_key" "$api_url")
        local file_url
        file_url=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); files=d.get('files',[]); print(files[0].get('downloadUrl') or files[0].get('url'))" 2>/dev/null)
        local file_name
        file_name=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); files=d.get('files',[]); print(files[0].get('name'))" 2>/dev/null)

        if [[ -z "$file_url" ]]; then
            echo "  ERROR: No download URL found" >&2
            return 1
        fi

        dest_file="$dest/${OUT:-$file_name}"
        echo "  Downloading $file_name -> $dest_file"
        curl -L -o "$dest_file" -H "Authorization: Bearer $api_key" "$file_url"
        echo "  Downloaded: $dest_file"
    else
        echo "  ERROR: curl required" >&2
        return 1
    fi
}

append_config() {
    local name="$1"
    local path="$2"
    local engine="${3:-llama-server}"

    local block="  # Added by add-model.sh"
    block+="
  $name:"
    block+="
    model: $path"

    if [[ "$engine" == "llama-server" ]]; then
        block+="
    ctx: $CTX"
        if [[ -n "$TEMPLATE" ]]; then
            block+="
    chat_template_file: /templates/$TEMPLATE"
        fi
        block+="
    temp: ${TEMP:-0.6}"
        block+="
    top_p: ${TOP_P:-0.95}"
        block+="
    ngf: 99"
        block+="
    device: HIP0"
        if [[ "$REASONING" == "true" ]]; then
            block+="
    reasoning_format: deepseek"
        fi
        if [[ -n "$EXTRA" ]]; then
            block+="
    extra: \"$EXTRA\""
        fi
    elif [[ "$engine" == "sd-server" ]]; then
        block+="
    engine: sd-server"
        block+="
    temp: ${TEMP:-0.8}"
        block+="
    max_tokens: 0"
        block+="
    dim: 1024"
        if [[ -n "$EXTRA" ]]; then
            block+="
    extra: \"$EXTRA\""
        fi
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "--- DRY RUN: config block ---"
        echo "$block"
        echo "--- end dry run ---"
    else
        # Append to config file
        mkdir -p "$(dirname "$CONFIG")"
        if [[ ! -f "$CONFIG" ]]; then
            echo "llama-servers: {}" > "$CONFIG"
        fi

        # Use python3 to safely append YAML
        python3 -c "
import yaml, sys

config_path = '$CONFIG'
block = '''$(echo "$block" | sed "s/'/''/g")'''

try:
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
except:
    config = {}

servers = config.get('llama-servers', {})
new_block = yaml.safe_load(block)
if new_block:
    for k, v in new_block.items():
        servers[k] = v
config['llama-servers'] = servers

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(f'Appended config block to {config_path}')
"
    fi
}

main() {
    parse_args "$@"

    if [[ -z "$NAME" ]]; then
        echo "ERROR: --name is required" >&2
        exit 1
    fi

    DIR="${DIR:-$NAME}"
    local dest_dir="$ROOT_DIR/../models/$DIR"
    local models_dir="${MODELS_DIR:-$ROOT_DIR/../models}"

    # Download
    if [[ -n "$REPO" ]]; then
        if [[ "$REPO" == "civitai" ]]; then
            download_civitai "$DIR" "$dest_dir"
        else
            download_hf "$REPO" "$FILE" "$dest_dir"
        fi
    elif [[ -n "$MODEL_PATH" ]]; then
        echo "  Using existing model: $MODEL_PATH"
    else
        echo "ERROR: --repo or --model-path required" >&2
        exit 1
    fi

    # Determine final path
    local final_path
    if [[ -n "$MODEL_PATH" ]]; then
        final_path="$MODEL_PATH"
    else
        final_path="$dest_dir/${OUT:-${FILE:-model.gguf}}"
    fi

    # Append config
    local engine="llama-server"
    if [[ "$DOWNLOAD_ONLY" == "true" ]]; then
        echo "Download complete: $final_path"
        exit 0
    fi

    append_config "$NAME" "$final_path" "$engine"
}

main "$@"
