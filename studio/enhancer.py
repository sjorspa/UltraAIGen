"""LM Studio (OpenAI-compatible) client that turns a short idea into an H3 prompt."""

import base64
import json
import re
from typing import Callable, Iterable

import requests
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QImage

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_FENCE_RE = re.compile(r"^```[a-z]*\s*|\s*```$", re.I)


class EnhancerError(RuntimeError):
    pass


def _native_root(base_url: str) -> str:
    """LM Studio's own REST API lives next to the OpenAI-compatible /v1."""
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def list_models(base_url: str, timeout: float = 5.0) -> list[dict]:
    """[{id, vision, loaded}] for every chat model LM Studio offers (embeddings left out)."""
    try:
        r = requests.get(_native_root(base_url) + "/api/v0/models", timeout=timeout)
        r.raise_for_status()
        return [{"id": m["id"], "vision": m.get("type") == "vlm", "loaded": m.get("state") == "loaded"}
                for m in r.json().get("data", []) if m.get("type") in ("llm", "vlm")]
    except (requests.RequestException, ValueError, KeyError):
        pass  # not LM Studio's native API: fall back to the plain OpenAI listing
    try:
        r = requests.get(base_url.rstrip("/") + "/models", timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise EnhancerError(f"LM Studio not reachable at {base_url}: {e}") from e
    ids = [m.get("id", "") for m in r.json().get("data", [])]
    return [{"id": i, "vision": False, "loaded": False} for i in ids if i and "embed" not in i.lower()]


def unload_all(base_url: str, timeout: float = 30.0) -> list[str]:
    """Free the GPU for video generation by unloading every model LM Studio has loaded."""
    try:
        r = requests.get(_native_root(base_url) + "/api/v0/models", timeout=5)
        r.raise_for_status()
        loaded = [m["id"] for m in r.json().get("data", []) if m.get("state") == "loaded"]
    except (requests.RequestException, ValueError, KeyError):
        return []
    for model_id in loaded:
        try:
            requests.post(_native_root(base_url) + "/api/v1/models/unload", json={"instance_id": model_id}, timeout=timeout)
        except requests.RequestException:
            pass
    return loaded


def image_data_url(path: str, max_side: int = 1024) -> str:
    img = QImage(path)
    if img.isNull():
        raise EnhancerError(f"Cannot read image: {path}")
    if max(img.width(), img.height()) > max_side:
        img = img.scaled(max_side, max_side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "JPEG", 90)
    return "data:image/jpeg;base64," + base64.b64encode(bytes(data)).decode("ascii")


def clean_output(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    return _FENCE_RE.sub("", text).strip()


def enhance(
    base_url: str,
    model: str,
    system_prompt: str,
    idea: str,
    images: Iterable[str] = (),
    temperature: float = 0.7,
    max_tokens: int = 8192,
    on_token: Callable[[str], None] | None = None,
    on_reasoning: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Stream a completion. Reasoning models first stream ``reasoning_content``;
    ``on_reasoning`` gets the running character count of that phase."""
    content: list[dict] | str = idea
    images = list(images)
    if images:
        content = [{"type": "text", "text": idea}]
        for i, path in enumerate(images, 1):
            content.append({"type": "text", "text": f"Image {i}:"})
            content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    try:
        r = requests.post(base_url.rstrip("/") + "/chat/completions", json=payload, stream=True, timeout=(10, 600))
        if r.status_code >= 400:
            raise EnhancerError(f"LM Studio error {r.status_code}: {r.text[:500]}")
        out = []
        reasoning = 0
        finish = None
        # iterate raw bytes: decode_unicode=True would also split on U+2028 & co.,
        # which may legally appear inside a JSON string and would corrupt the event
        for raw_bytes in r.iter_lines():
            if should_stop and should_stop():
                r.close()
                break
            raw = raw_bytes.decode("utf-8", "replace")
            if not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                choice = json.loads(data)["choices"][0]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if thought:
                reasoning += len(thought)
                if on_reasoning:
                    on_reasoning(reasoning)
            piece = delta.get("content") or ""
            if piece:
                out.append(piece)
                if on_token:
                    on_token(piece)
    except requests.RequestException as e:
        raise EnhancerError(f"Connection to LM Studio failed: {e}") from e
    text = clean_output("".join(out))
    if not text and finish == "length" and not (should_stop and should_stop()):
        raise EnhancerError(
            f"{model} used all {max_tokens} tokens"
            + (f" on reasoning ({reasoning} characters of thinking)" if reasoning else "")
            + " before writing the prompt.\n\nRaise 'Max tokens' in Settings, or pick a non-reasoning model.")
    return text
