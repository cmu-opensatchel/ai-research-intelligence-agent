"""Local LLM client (Qwen3-8B via Ollama) used for optional claim verification."""

import json
import logging
from typing import List, Optional

import requests

from .config import AgentConfig
from .models import Passage

logger = logging.getLogger(__name__)

_VALID_VERDICTS = {"supported", "partial", "unsupported", "contradicted"}

_SYSTEM_PROMPT = (
    "You are a strict evidence checker. Given a claim and numbered passages, decide "
    "whether the passages themselves support the claim. Use ONLY the passages; ignore "
    "outside knowledge. A passage that merely mentions the topic does not support the claim. "
    'Reply with JSON only: {"verdict": "supported|partial|unsupported|contradicted", '
    '"confidence": <0.0-1.0>, "supporting": [<passage numbers>], "reason": "<one sentence>"}'
)


class OllamaClient:
    """Minimal Ollama chat client. Never raises on failure; returns None instead."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.cfg = config or AgentConfig()

    def chat_json(self, system: str, user: str) -> Optional[dict]:
        try:
            resp = requests.post(
                f"{self.cfg.ollama_url.rstrip('/')}/api/chat",
                json={
                    "model": self.cfg.ollama_model,
                    "stream": False,
                    "think": False,  # skip Qwen3 reasoning trace; only the verdict is needed
                    "format": "json",
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=self.cfg.ollama_timeout,
            )
            resp.raise_for_status()
            return json.loads(resp.json()["message"]["content"])
        except Exception as e:  # network, HTTP, or JSON errors
            logger.warning("Ollama call failed: %s", e)
            return None

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.cfg.ollama_url.rstrip('/')}/api/tags", timeout=5)
            r.raise_for_status()
            names = {m.get("name") for m in r.json().get("models", [])}
            return self.cfg.ollama_model in names
        except Exception:
            return False

    def judge_claim(self, claim: str, passages: List[Passage]) -> Optional[dict]:
        """Ask the LLM whether passages support the claim. Returns a validated dict or None."""
        if not passages:
            return None
        body = "\n\n".join(
            f"[{i}] ({p.source_title})\n{p.text[:1200]}" for i, p in enumerate(passages, 1)
        )
        raw = self.chat_json(_SYSTEM_PROMPT, f"Claim: {claim}\n\nPassages:\n{body}")
        if not isinstance(raw, dict) or raw.get("verdict") not in _VALID_VERDICTS:
            return None
        try:
            conf = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        supporting = [n for n in raw.get("supporting", []) if isinstance(n, int) and 1 <= n <= len(passages)]
        return {
            "verdict": raw["verdict"],
            "confidence": conf,
            "supporting": supporting,
            "reason": str(raw.get("reason", ""))[:300],
        }
