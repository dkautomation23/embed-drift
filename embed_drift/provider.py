# -*- coding: utf-8 -*-
"""Getting vectors out of whatever is serving them.

Two shapes cover almost everything: Ollama's `/api/embed` and the OpenAI-style
`/v1/embeddings` that most hosted providers and every local gateway imitate. A
key is read from the environment when the endpoint wants one and is never
written to a file, never printed, and never passed on a command line where it
would land in a shell history.

One request per probe rather than a batch, on purpose: a batch endpoint that
silently truncates or reorders would corrupt the fingerprint in a way that looks
like drift, and the whole point of this tool is not to cry wolf.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    """The endpoint could not be used. Carries the reason, not a traceback."""


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:200]
        raise ProviderError(f"http {error.code} from {url}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise ProviderError(f"cannot reach {url}: {error}") from None
    except json.JSONDecodeError as error:
        raise ProviderError(f"{url} did not return JSON: {error}") from None


def embed_one(
    text: str,
    *,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    api: str = "ollama",
    api_key_env: str | None = None,
    timeout: float = 120.0,
) -> list[float]:
    """One text, one vector. Raises ProviderError with the reason on failure."""
    headers = {"content-type": "application/json"}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ProviderError(f"environment variable {api_key_env} is not set")
        headers["authorization"] = f"Bearer {key}"

    if api == "ollama":
        body = _post(
            f"{base_url.rstrip('/')}/api/embed",
            {"model": model, "input": text},
            headers,
            timeout,
        )
        vectors = body.get("embeddings")
        if not vectors or not isinstance(vectors, list):
            raise ProviderError(f"no embeddings in response: {str(body)[:160]}")
        return [float(x) for x in vectors[0]]

    if api == "openai":
        body = _post(
            f"{base_url.rstrip('/')}/v1/embeddings",
            {"model": model, "input": text},
            headers,
            timeout,
        )
        data = body.get("data")
        if not data:
            raise ProviderError(f"no data in response: {str(body)[:160]}")
        return [float(x) for x in data[0]["embedding"]]

    raise ProviderError(f"unknown api {api!r}; use 'ollama' or 'openai'")


def embed_all(texts, **kwargs) -> list[list[float]]:
    """Every probe, in order, one request each."""
    return [embed_one(text, **kwargs) for text in texts]
