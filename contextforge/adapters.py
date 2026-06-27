"""Convert between chat-API request formats and ContextForge Traces.

Lets the proxy be a true drop-in: a client points its SDK ``base_url`` at
ContextForge, we parse the request into a Trace, compile it, and re-serialize it
back into the *same* API format before forwarding upstream.

Supports the Anthropic Messages API and the OpenAI Chat Completions API. Content
blocks (lists) are flattened to text; streaming is downgraded to non-streaming
(documented limitation of v0).
"""

from __future__ import annotations

from typing import Any, Dict

from .types import CompileResult, ContextItem, Trace


def text_of(content: Any) -> str:
    """Flatten string-or-block content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                if blk.get("type") in ("text", None) and "text" in blk:
                    parts.append(blk["text"])
                elif "content" in blk:  # e.g. tool_result blocks
                    parts.append(text_of(blk["content"]))
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n".join(p for p in parts if p)
    return str(content)


# --------------------------------------------------------------------------- #
# Anthropic Messages API
# --------------------------------------------------------------------------- #
def anthropic_to_trace(body: Dict[str, Any]) -> Trace:
    items = []
    sys_text = text_of(body.get("system"))
    if sys_text:
        items.append(ContextItem(content=sys_text, role="system",
                                 pinned=True, id="orig_system"))
    last_user = ""
    for i, m in enumerate(body.get("messages", [])):
        c = text_of(m.get("content", ""))
        role = m.get("role", "user")
        items.append(ContextItem(content=c, role=role, id=f"msg_{i}"))
        if role == "user":
            last_user = c
    return Trace(items=items, task=last_user, model=body.get("model"))


def result_to_anthropic(result: CompileResult, body: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild an Anthropic request. The state header is folded into ``system``;
    the recap footer is appended to the final user turn (the window edge)."""
    system_parts = []
    footer = None
    out = []
    for it in result.items:
        if it.id == "cf_anchor_header":
            system_parts.insert(0, it.content)
            continue
        if it.id == "cf_anchor_footer":
            footer = it.content
            continue
        if it.role == "system":
            system_parts.append(it.content)
            continue
        role = it.role if it.role in ("user", "assistant") else "user"  # tool->user
        out.append({"role": role, "content": it.content})

    # Anthropic requires alternating roles starting with user — merge consecutive.
    merged = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": "(continuing the conversation)"})

    if footer:
        for m in reversed(merged):
            if m["role"] == "user":
                m["content"] += "\n\n" + footer
                break
        else:
            merged.append({"role": "user", "content": footer})

    new = dict(body)
    new["system"] = "\n\n".join(p for p in system_parts if p).strip()
    new["messages"] = merged
    new["stream"] = False
    return new


# --------------------------------------------------------------------------- #
# OpenAI Chat Completions API
# --------------------------------------------------------------------------- #
def openai_to_trace(body: Dict[str, Any]) -> Trace:
    items = []
    last_user = ""
    for i, m in enumerate(body.get("messages", [])):
        c = text_of(m.get("content", ""))
        role = m.get("role", "user")
        items.append(ContextItem(content=c, role=role,
                                 pinned=(role == "system"), id=f"msg_{i}"))
        if role == "user":
            last_user = c
    return Trace(items=items, task=last_user, model=body.get("model"))


def result_to_openai(result: CompileResult, body: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild an OpenAI request. OpenAI allows system messages anywhere, so the
    compiled order (header near front, recap near back) is preserved directly."""
    out = []
    for it in result.items:
        role = it.role
        if role == "tool":
            role = "system"  # represent tool dumps as system context
        if role not in ("system", "user", "assistant"):
            role = "user"
        out.append({"role": role, "content": it.content})
    new = dict(body)
    new["messages"] = out
    new["stream"] = False
    return new


ADAPTERS = {
    "anthropic": (anthropic_to_trace, result_to_anthropic, "/v1/messages"),
    "openai": (openai_to_trace, result_to_openai, "/v1/chat/completions"),
}
