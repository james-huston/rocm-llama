#!/usr/bin/env python3
"""apply-models.py — process models.yaml manifest.

Reads models.yaml, downloads any missing GGUFs (HF, Civitai, or direct URL),
and regenerates config/llama-swap.yaml from the enabled entries.

Usage:
    python3 apply-models.py --manifest models.yaml --models-dir /models --config config/llama-swap.yaml

Requires: PyYAML (`pip install pyyaml`)
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import urllib.error

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

SCHEMA = ("# yaml-language-server: $schema="
          "https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/config-schema.json")
DEFAULT_HEALTHCHECK = 240
DEFAULT_START_PORT = 12800
DEFAULT_TTL = 600


def parse_args():
    p = argparse.ArgumentParser(description="Apply models.yaml manifest")
    p.add_argument("--manifest", required=True, help="Path to models.yaml")
    p.add_argument("--models-dir", required=True, help="Models directory on host")
    p.add_argument("--config", required=True, help="Output llama-swap config path")
    p.add_argument("--config-bak", default=None, help="Backup config path")
    p.add_argument("--templates-dir", default="templates", help="Templates directory")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing")
    return p.parse_args()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_hf(repo, filename, dest_path, hf_token=None):
    """Download a file from Hugging Face."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    if hf_token:
        url += f"?download=true"

    print(f"  Downloading {repo}/{filename} -> {dest_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "rocm-llama/apply-models"})
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = resp.headers.get("Content-Length")
            if total:
                print(f"  Size: {int(total) / 1024 / 1024:.1f} MB")
            downloaded = 0
            with open(dest_path, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / int(total) * 100
                        print(f"\r  Progress: {pct:.0f}%", end="", flush=True)
            print()
    except urllib.error.URLError as e:
        print(f"  ERROR: Failed to download {url}: {e}", file=sys.stderr)
        return False
    return True


def download_civitai(model_version_id, file_filter, dest_path, api_key):
    """Download from Civitai."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://civitai.com/api/v1/model-versions/{model_version_id}"

    print(f"  Fetching Civitai version {model_version_id}")
    req = urllib.request.Request(url, headers={
        "User-Agent": "rocm-llama/apply-models",
        "Authorization": f"Bearer {api_key}"
    })

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            import json
            data = json.loads(resp.read())
            files = data.get("files", [])
            if not files:
                print(f"  ERROR: No files found for version {model_version_id}", file=sys.stderr)
                return False

            # Filter files
            target = None
            if file_filter:
                for f in files:
                    if file_filter.lower() in f.get("name", "").lower():
                        target = f
                        break
                if not target:
                    print(f"  ERROR: No file matching '{file_filter}' found", file=sys.stderr)
                    return False
            else:
                target = files[0]  # primary

            file_url = target.get("downloadUrl") or target.get("url")
            if not file_url:
                print(f"  ERROR: No download URL for file '{target.get('name')}'", file=sys.stderr)
                return False

            final_name = target.get("name", f"model_version_{model_version_id}.gguf")
            final_dest = os.path.join(os.path.dirname(dest_path), final_name) if dest_path.endswith("/") else dest_path
            os.makedirs(os.path.dirname(final_dest), exist_ok=True)

            print(f"  Downloading {target.get('name')} -> {final_dest}")
            req2 = urllib.request.Request(file_url, headers={
                "User-Agent": "rocm-llama/apply-models",
                "Authorization": f"Bearer {api_key}"
            })
            with urllib.request.urlopen(req2, timeout=600) as resp2:
                total = resp2.headers.get("Content-Length")
                if total:
                    print(f"  Size: {int(total) / 1024 / 1024:.1f} MB")
                downloaded = 0
                with open(final_dest, "wb") as out:
                    while True:
                        chunk = resp2.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / int(total) * 100
                            print(f"\r  Progress: {pct:.0f}%", end="", flush=True)
                print()
            return True
    except urllib.error.URLError as e:
        print(f"  ERROR: Failed to fetch Civitai: {e}", file=sys.stderr)
        return False


def download_url(url, dest_path):
    """Download from a direct URL."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {url} -> {dest_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "rocm-llama/apply-models"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = resp.headers.get("Content-Length")
            if total:
                print(f"  Size: {int(total) / 1024 / 1024:.1f} MB")
            downloaded = 0
            with open(dest_path, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / int(total) * 100
                        print(f"\r  Progress: {pct:.0f}%", end="", flush=True)
            print()
    except urllib.error.URLError as e:
        print(f"  ERROR: Failed to download {url}: {e}", file=sys.stderr)
        return False
    return True


def apply_model(entry, models_dir, templates_dir, dry_run=False):
    """Process a single model entry. Returns (name, dest_path) or None."""
    name = entry.get("name", "unnamed")
    enabled = entry.get("enabled", False)
    engine = entry.get("engine", "llama-server")
    path = entry.get("path")
    repo = entry.get("repo")
    file = entry.get("file")
    url = entry.get("url")
    dir_name = entry.get("dir", name)
    out_name = entry.get("out") or entry.get("file", "")
    hf_token = os.environ.get("HF_TOKEN")
    civitai_key = os.environ.get("CIVITAI_API_KEY")
    civitai_id = entry.get("civitai")
    civitai_file = entry.get("civitai_file")

    if not enabled:
        return None

    if path:
        # Ollama blob path — already on host, just verify
        full_path = path if path.startswith("/") else os.path.join(models_dir, path)
        if not os.path.exists(full_path):
            print(f"  WARNING: path not found: {full_path} (model '{name}' disabled in config)")
            return None
        return (name, full_path)

    # Download needed
    if repo == "civitai" and civitai_id:
        dest = os.path.join(models_dir, dir_name, out_name or f"civitai_{civitai_id}.gguf")
        if not dry_run and not os.path.exists(dest):
            if not civitai_key:
                print(f"  ERROR: CIVITAI_API_KEY not set for Civitai model '{name}'", file=sys.stderr)
                return None
            if not download_civitai(civitai_id, civitai_file, dest, civitai_key):
                return None
        return (name, dest)
    elif repo:
        dest = os.path.join(models_dir, dir_name, out_name or file)
        if not dry_run and not os.path.exists(dest):
            if not download_hf(repo, file, dest, hf_token):
                return None
        return (name, dest)
    elif url:
        dest = os.path.join(models_dir, dir_name, out_name or f"model_{name}.gguf")
        if not dry_run and not os.path.exists(dest):
            if not download_url(url, dest):
                return None
        return (name, dest)
    else:
        print(f"  WARNING: no source for model '{name}' (path/repo/url/civitai required)")
        return None


def generate_block(entry, templates_dir):
    """Render one enabled entry as a llama-swap model block (the real config
    format: `<name>:` -> `cmd: |` -> the llama-server/sd-server argv -> `ttl:`)."""
    name = entry.get("name", "unnamed")
    engine = entry.get("engine", "llama-server")
    path = entry.get("path", "")
    if path and not path.startswith("/"):
        path = f"/models/{path}"

    lines = [f"  # {name} — from models.yaml", f"  {name}:", "    cmd: |"]

    if engine == "sd-server":
        lines += [
            "      /opt/sd-cpp/bin/sd-server",
            f"      --model {path}",
            "      --listen-ip 127.0.0.1",
            "      --listen-port ${PORT}",
        ]
        if entry.get("dim"):
            lines.append(f"      --dim {entry['dim']}")
        if entry.get("extra"):
            lines.append(f"      {entry['extra']}")
        # sd-server has no /health; point llama-swap's probe at one it answers.
        lines.append("    checkEndpoint: /v1/models")
    else:
        lines += [
            "      /opt/llama-cpp/bin/llama-server",
            "      --port ${PORT}",
            f"      --model {path}",
            f"      --alias {name}",
            "      --host 127.0.0.1",
            f"      -ngl {entry.get('ngf', 99)}",
            f"      -c {entry.get('ctx', 8192)}",
        ]
        # Device / split-mode are omitted by default: this image picks the GPU via
        # HIP_VISIBLE_DEVICES, so a single device needs no --device/-sm. Set
        # `device:`/`sm:` in models.yaml only for explicit multi-GPU placement.
        if entry.get("device"):
            lines.append(f"      --device {entry['device']}")
        if entry.get("sm"):
            lines.append(f"      -sm {entry['sm']}")
        lines.append("      --jinja")
        template = entry.get("template")
        if template:
            if not os.path.exists(os.path.join(templates_dir, template)):
                print(f"  WARNING: template not found: templates/{template} (referenced by {name})")
            lines.append(f"      --chat-template-file /templates/{template}")
        if entry.get("reasoning_format"):
            lines.append(f"      --reasoning-format {entry['reasoning_format']}")
        elif entry.get("reasoning"):
            lines.append("      --reasoning-format deepseek")
        # Only emit samplers explicitly set in the manifest — no silent defaults,
        # so a model with no temp/top_p uses the GGUF's own defaults.
        if entry.get("temp") is not None:
            lines.append(f"      --temp {entry['temp']}")
        if entry.get("top_p") is not None:
            lines.append(f"      --top-p {entry['top_p']}")
        if entry.get("extra"):
            lines.append(f"      {entry['extra']}")

    lines.append(f"    ttl: {entry.get('ttl', DEFAULT_TTL)}")
    return "\n".join(lines)


def generate_config(manifest, templates_dir):
    """Build the full llama-swap config text from the manifest's enabled entries."""
    health = manifest.get("healthCheckTimeout", DEFAULT_HEALTHCHECK)
    start = manifest.get("startPort", DEFAULT_START_PORT)
    head = [
        SCHEMA,
        "#",
        "# GENERATED by scripts/apply-models.py from models.yaml — DO NOT EDIT BY HAND.",
        "# Edit models.yaml and run `make models-apply`.",
        "",
        f"healthCheckTimeout: {health}",
        f"startPort: {start}",
        "",
        "models:",
        "",
    ]
    blocks = [generate_block(e, templates_dir)
              for e in manifest.get("models", []) if e.get("enabled", False)]
    return "\n".join(head) + "\n\n".join(blocks) + "\n"


def find_orphaned_dirs(models_dir, models):
    """Find directories in MODELS_DIR not referenced by any enabled model."""
    referenced = set()
    for entry in models:
        if not entry.get("enabled", False):
            continue
        path = entry.get("path", "")
        if path.startswith("/models/"):
            referenced.add(os.path.dirname(path[len("/models/"):]).split("/", 1)[0])
        elif entry.get("dir"):
            referenced.add(entry["dir"])
        elif entry.get("name"):
            referenced.add(entry["name"])

    if not os.path.isdir(models_dir):
        return []

    orphaned = []
    for entry in os.listdir(models_dir):
        full = os.path.join(models_dir, entry)
        if entry in ("blobs", "manifests", "hub", "xet"):
            continue
        if os.path.isdir(full) and entry not in referenced:
            orphaned.append(entry)

    return orphaned


def main():
    args = parse_args()

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    models = manifest.get("models", [])
    templates_dir = args.templates_dir

    # Apply models
    print(f"Processing {args.manifest}...")
    print(f"Models dir: {args.models_dir}")
    print()

    for entry in models:
        name = entry.get("name", "unnamed")
        enabled = entry.get("enabled", False)
        if not enabled:
            print(f"  SKIP (disabled): {name}")
            continue
        apply_model(entry, args.models_dir, templates_dir, args.dry_run)

    # Generate config (real llama-swap format: models -> cmd block -> ttl)
    print()
    config = generate_config(manifest, templates_dir)

    if args.dry_run:
        print("--- DRY RUN: generated config ---")
        print(config)
        print("--- end dry run ---")
    else:
        # Backup existing config
        if os.path.exists(args.config):
            shutil.copy2(args.config, args.config_bak if args.config_bak else f"{args.config}.bak")
            print(f"Backed up {args.config} -> {args.config_bak if args.config_bak else f'{args.config}.bak'}")

        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, "w") as f:
            f.write(config)
        print(f"Wrote {args.config}")

    # Check for orphaned model dirs
    orphaned = find_orphaned_dirs(args.models_dir, models)
    if orphaned:
        print()
        print(f"Orphaned directories in {args.models_dir} (not referenced by any enabled model):")
        for d in orphaned:
            print(f"  {d}")
    else:
        print("No orphaned model directories found.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
