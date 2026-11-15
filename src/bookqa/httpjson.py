"""Minimal JSON-over-HTTP client using only the standard library.

Used for OpenAI-compatible servers (Ollama, LM Studio, mlx_lm.server,
llama.cpp, hosted APIs). Avoiding an HTTP dependency keeps installs small.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


def auth_headers(api_key_env: str) -> dict[str, str]:
    if not api_key_env:
        return {}
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"environment variable {api_key_env} is not set")
    return {"Authorization": f"Bearer {key}"}


def post_json(
    url: str, payload: dict, headers: dict[str, str], timeout: float, retries: int = 3
) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST"
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if (e.code < 500 and e.code != 429) or attempt == retries:
                raise RuntimeError(f"POST {url} failed with HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                raise RuntimeError(f"POST {url} failed: {e}") from e
        time.sleep(2**attempt)
    raise AssertionError("unreachable")
