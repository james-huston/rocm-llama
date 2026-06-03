#!/usr/bin/env python3
"""test-models.py — smoke-test every model on an endpoint.

Cycles through every model an endpoint advertises, sends a real chat query
to each (forcing llama-swap to cold-load it), and reports pass/fail.

Usage:
    python3 test-models.py
    python3 test-models.py --via litellm --base http://localhost:4000
    python3 test-models.py --models glm-4.7-flash-q4

Requires: Python 3 stdlib only (urllib, json, argparse)
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


def parse_args():
    p = argparse.ArgumentParser(description="Smoke-test models")
    p.add_argument("--via", default="", choices=["", "litellm"],
                   help="Test through LiteLLM proxy")
    p.add_argument("--base", default="",
                   help="Base URL (overrides --via)")
    p.add_argument("--api-key", default="",
                   help="API key (optional)")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Max output tokens")
    p.add_argument("--timeout", type=int, default=120,
                   help="Request timeout in seconds")
    p.add_argument("--prompt", default="Say one word only: hello",
                   help="Test prompt")
    p.add_argument("--models", default="",
                   help="Space-separated list of models to test (default: all)")
    return p.parse_args()


def get_models(base, api_key):
    """Fetch /v1/models from an endpoint."""
    url = f"{base}/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
    except urllib.error.URLError as e:
        print(f"ERROR: Failed to fetch models from {url}: {e}", file=sys.stderr)
        sys.exit(1)


def test_model(base, api_key, model_name, prompt, max_tokens, timeout):
    """Send a chat completion request to a model."""
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6
    }
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers,
                                 data=json.dumps(payload).encode())

    start = __import__('time').time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = __import__('time').time() - start
            data = json.loads(resp.read())
            content = ""
            if data.get("choices"):
                choice = data["choices"][0]
                content = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason", "")
            else:
                finish_reason = "no_choices"

            # Check for reasoning content (hidden by default in some models)
            reasoning = ""
            if data.get("choices"):
                for key in ("reasoning", "thoughts", "internal_response"):
                    val = data["choices"][0].get(key, "")
                    if val:
                        reasoning = val
                        break

            if not content and not reasoning:
                return ("WARN", f"empty response ({elapsed:.1f}s)", finish_reason)
            return ("PASS", f"got response ({elapsed:.1f}s)", finish_reason)

    except urllib.error.URLError as e:
        elapsed = __import__('time').time() - start
        return ("FAIL", f"{e} ({elapsed:.1f}s)", "")
    except Exception as e:
        elapsed = __import__('time').time() - start
        return ("FAIL", f"{e} ({elapsed:.1f}s)", "")


def main():
    args = parse_args()

    # Determine base URL
    if args.base:
        base = args.base.rstrip("/")
    elif args.via == "litellm":
        base = "http://localhost:4000"
    else:
        base = "http://localhost:11434"

    api_key = args.api_key

    # Get models to test
    if args.models:
        models = args.models.split()
    else:
        models = get_models(base, api_key)

    if not models:
        print("No models found.")
        sys.exit(0)

    print(f"Testing {len(models)} models on {base}...")
    print()

    results = []
    for name in models:
        status, detail, finish_reason = test_model(
            base, api_key, name, args.prompt, args.max_tokens, args.timeout
        )
        results.append((name, status, detail, finish_reason))
        status_str = status
        if status == "PASS":
            status_str = "PASS"
        elif status == "WARN":
            status_str = "WARN"
        else:
            status_str = "FAIL"
        print(f"  {status_str:6s} {name:40s} {detail}")

    # Summary
    print()
    passes = sum(1 for _, s, _, _ in results if s == "PASS")
    warns = sum(1 for _, s, _, _ in results if s == "WARN")
    fails = sum(1 for _, s, _, _ in results if s == "FAIL")
    print(f"Results: {passes} pass, {warns} warn, {fails} fail")

    if fails > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
