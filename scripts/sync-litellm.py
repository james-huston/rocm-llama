#!/usr/bin/env python3
"""sync-litellm.py — mirror llama-swap models into a LiteLLM proxy.

Reads /v1/models from the upstream (llama-swap), then adds missing models,
deletes stale ones, and re-points drifted registrations. Models baked into
LiteLLM's static config.yaml are left untouched.

Each model is registered with the openai/ provider + an /v1 api_base, and
enriched with model_info from models.yaml (descriptions, capability flags,
costs derived from decode_tps).

Usage:
    python3 sync-litellm.py --upstream http://localhost:11434 --litellm http://localhost:4000
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

try:
    import yaml
except ImportError:
    yaml = None


def parse_args():
    p = argparse.ArgumentParser(description="Sync models to LiteLLM proxy")
    p.add_argument("--upstream", default="http://localhost:11434",
                   help="llama-swap base URL")
    p.add_argument("--litellm", default="http://localhost:4000",
                   help="LiteLLM proxy base URL")
    p.add_argument("--api-base", default="http://localhost:11434/v1",
                   help="api_base for each model")
    p.add_argument("--no-delete", action="store_true",
                   help="Don't delete stale models")
    p.add_argument("--reset", action="store_true",
                   help="Delete all DB-managed models first")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview without writing")
    p.add_argument("--manifest", default="models.yaml",
                   help="Path to models.yaml for cost/flag enrichment")
    return p.parse_args()


def get_api_key():
    """Get LiteLLM API key from environment or .env file."""
    key = os.environ.get("LITELLM_API_KEY") or os.environ.get("LITELLM_MASTER_KEY")
    if key:
        return key

    # Try .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k in ("LITELLM_API_KEY", "LITELLM_MASTER_KEY"):
                        return v

    print("ERROR: LITELLM_API_KEY or LITELLM_MASTER_KEY not set", file=sys.stderr)
    print("Set it in .env or as an environment variable.", file=sys.stderr)
    sys.exit(1)


def make_request(url, headers=None, data=None, method=None):
    """Make an HTTP request and return parsed JSON."""
    if headers is None:
        headers = {}
    req = urllib.request.Request(url, headers=headers)
    if data:
        req.data = json.dumps(data).encode()
    if method:
        req.get_method = lambda: method

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            if body:
                return json.loads(body)
            return {}
    except urllib.error.URLError as e:
        print(f"ERROR: {url} -> {e}", file=sys.stderr)
        return None


def fetch_upstream_models(upstream, api_key):
    """Fetch /v1/models from llama-swap."""
    url = f"{upstream}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    result = make_request(url, headers=headers)
    if result is None:
        print(f"ERROR: Failed to fetch models from {url}", file=sys.stderr)
        sys.exit(1)
    return [m["id"] for m in result.get("data", [])]


def fetch_litellm_models(litellm_url, api_key):
    """Fetch /v1/models from LiteLLM proxy."""
    url = f"{litellm_url}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    result = make_request(url, headers=headers)
    if result is None:
        print(f"ERROR: Failed to fetch models from {url}", file=sys.stderr)
        sys.exit(1)
    return {m["id"]: m for m in result.get("data", [])}


def load_manifest(manifest_path):
    """Load models.yaml for enrichment data."""
    if yaml is None:
        return {}
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path) as f:
            data = yaml.safe_load(f) or {}
        models = {}
        for entry in data.get("models", []):
            name = entry.get("name")
            if name:
                models[name] = entry
        return models
    except Exception:
        return {}


def derive_costs(model_entry, cost_config):
    """Derive per-token costs from decode_tps and cost config."""
    decode_tps = model_entry.get("decode_tps")
    if not decode_tps:
        return None

    cost = cost_config or {}
    watts = cost.get("watts", 300)
    usd_per_kwh = cost.get("usd_per_kwh", 0.18)
    amortization = cost.get("amortization", 3)
    input_factor = cost.get("input_factor", 4)

    # output $/token = watts/1000 * (1/decode_tps/3600) * usd_per_kwh * amortization
    output_cost = (watts / 1000) * (1 / decode_tps / 3600) * usd_per_kwh * amortization
    input_cost = output_cost * input_factor

    return {
        "input_cost_per_token": round(input_cost, 10),
        "output_cost_per_token": round(output_cost, 10)
    }


def build_model_info(model_entry):
    """Build LiteLLM model_info from a models.yaml entry."""
    if yaml is None:
        return {}

    litellm_block = model_entry.get("litellm", {})
    ctx = model_entry.get("ctx")

    info = {"mode": "chat"}
    if ctx:
        info["max_input_tokens"] = ctx

    if litellm_block:
        if "description" in litellm_block:
            info["description"] = litellm_block["description"]
        for flag in ("supports_function_calling", "supports_tool_choice",
                      "supports_reasoning", "supports_vision"):
            if flag in litellm_block:
                info[flag] = litellm_block[flag]

    return info


def add_model(litellm_url, api_key, name, api_base, model_info, costs):
    """Add a model to LiteLLM."""
    url = f"{litellm_url}/model/new"
    payload = {
        "model_name": name,
        "litellm_params": {
            "model": f"openai/{name}",
            "api_base": api_base,
            "api_key": "not-needed"
        }
    }
    if model_info:
        payload["model_info"] = model_info
    if costs:
        payload["model_info"] = payload.get("model_info", {})
        payload["model_info"].update(costs)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    result = make_request(url, headers=headers, data=payload, method="POST")
    if result is not None:
        print(f"  ADDED: {name}")
        return True
    return False


def delete_model(litellm_url, api_key, model_id):
    """Delete a model from LiteLLM."""
    url = f"{litellm_url}/model/delete"
    payload = {"model_name": model_id}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    result = make_request(url, headers=headers, data=payload, method="POST")
    if result is not None:
        print(f"  DELETED: {model_id}")
        return True
    return False


def update_model(litellm_url, api_key, name, api_base, model_info, costs):
    """Update a model's api_base and model_info in LiteLLM."""
    url = f"{litellm_url}/model/update"
    payload = {"model_name": name}

    litellm_params = {"model": f"openai/{name}", "api_base": api_base}
    payload["litellm_params"] = litellm_params

    if model_info or costs:
        info = model_info.copy() if model_info else {}
        if costs:
            info.update(costs)
        payload["model_info"] = info

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    result = make_request(url, headers=headers, data=payload, method="POST")
    if result is not None:
        print(f"  UPDATED: {name}")
        return True
    return False


def main():
    args = parse_args()
    api_key = get_api_key()

    print(f"Upstream: {args.upstream}")
    print(f"LiteLLM:  {args.litellm}")
    print(f"API base: {args.api_base}")
    print(f"Dry run:  {args.dry_run}")
    print()

    # Fetch model lists
    upstream_models = fetch_upstream_models(args.upstream, api_key)
    litellm_models = fetch_litellm_models(args.litellm, api_key)

    upstream_set = set(upstream_models)
    litellm_set = set(litellm_models.keys())

    to_add = upstream_set - litellm_set
    to_delete = litellm_set - upstream_set
    to_sync = upstream_set & litellm_set

    # Load manifest for enrichment
    manifest = load_manifest(args.manifest)
    cost_config = manifest.get("cost", {})

    # Reset mode: delete everything first
    if args.reset:
        to_delete = litellm_set
        to_sync = set()

    print(f"Models to ADD:   {len(to_add)}")
    print(f"Models to DELETE:{len(to_delete)}")
    print(f"Models to SYNC:  {len(to_sync)}")
    print()

    # Add missing models
    if to_add:
        print("=== Adding models ===")
        for name in sorted(to_add):
            model_entry = manifest.get(name, {})
            model_info = build_model_info(model_entry)
            costs = derive_costs(model_entry, cost_config)
            if args.dry_run:
                print(f"  [DRY RUN] ADD: {name}")
            else:
                add_model(args.litellm, api_key, name, args.api_base, model_info, costs)
        print()

    # Delete stale models
    if to_delete and not args.no_delete:
        print("=== Deleting models ===")
        for name in sorted(to_delete):
            if args.dry_run:
                print(f"  [DRY RUN] DELETE: {name}")
            else:
                delete_model(args.litellm, api_key, name)
        print()

    # Sync existing models (re-point api_base, update info)
    if to_sync:
        print("=== Syncing models ===")
        for name in sorted(to_sync):
            model_entry = manifest.get(name, {})
            model_info = build_model_info(model_entry)
            costs = derive_costs(model_entry, cost_config)

            litellm_model = litellm_models.get(name, {})
            litellm_params = litellm_model.get("litellm_params", {})
            current_api_base = litellm_params.get("api_base", "")

            if current_api_base != args.api_base or (model_info and model_info != litellm_model.get("model_info", {})):
                if args.dry_run:
                    print(f"  [DRY RUN] UPDATE: {name} (api_base: {current_api_base} -> {args.api_base})")
                else:
                    update_model(args.litellm, api_key, name, args.api_base, model_info, costs)
            else:
                print(f"  OK: {name}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
