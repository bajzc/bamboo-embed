"""HyDE — Hypothetical Document Embeddings (§4.1).

The LLM writes a plausible *classical-Chinese* passage answering the modern query;
that pseudo-document is embedded for the dense path, narrowing the modern-query vs
文言文-document gap. The sparse (bigram) path keeps using the raw query.
"""

from __future__ import annotations

from typing import Callable

from ..config import Config

_SYSTEM = "你是精通中國古代典籍的學者，熟悉史書、諸子、詩詞的文言文表達。"
_PROMPT = (
    "根據下面的現代漢語問題，寫一段你認為最可能出現在古籍中、能回答該問題的"
    "「文言文原文」片段（繁體中文，50–150字）。只輸出文言文本身，不要加解釋、"
    "不要標注書名或出處。\n\n問題：{q}\n\n文言文："
)


def generate(cfg: Config, query: str, on_event: Callable[[str, dict], None] | None = None) -> str:
    from openai import OpenAI

    llm = cfg.active_llm()
    # Ollama's OpenAI-compatible endpoint ignores the key but the SDK requires one.
    key = cfg.api_key(llm.api_key_env) or "not-needed"
    client = OpenAI(base_url=llm.base_url, api_key=key)
    stream = client.chat.completions.create(
        model=llm.model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _PROMPT.format(q=query)},
        ],
        max_tokens=cfg.hyde.max_tokens,
        temperature=0.3,
        stream=True,
    )
    parts: list[str] = []
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            parts.append(text)
            if on_event:
                on_event("hyde_delta", {"text": text})
    return "".join(parts).strip()
