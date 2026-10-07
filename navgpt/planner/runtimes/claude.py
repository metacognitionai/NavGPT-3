"""Claude Agent SDK runtime for NavGPT Planner.

The SDK drives the ``claude`` CLI, which spawns the NavGPT stdio MCP tool server.
Note ``command=sys.executable``: the tool server runs in THIS interpreter,
so its imports (requests, pillow) must resolve here.

Two mechanisms are load-bearing:

1. The MCP connection gate. The CLI starts reasoning before MCP servers
   finish connecting, so we poll ``get_mcp_status()`` until the MCP tool server reports
   ``connected`` and raise otherwise. Without this a broken MCP tool server yields a
   polite "I have no tools available" session that reports success and scores
   SR=0 — a config failure disguised as a navigation failure.

2. The subtype whitelist. ``is_error`` tracks the SESSION, not the navigation
   outcome, so it over-flags. Score by the terminal subtype instead.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
from typing import Any

from ..config import RunConfig
from ..runner import EpisodeContext, EventSink, SessionOutcome, throttle_tag

# Environment that makes the CLI use an API key, a gateway token or a cloud
# provider instead of its stored login.
API_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
                   "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")
LOGIN_FAILURES = ("/login", "not logged in", "invalid api key", "oauth token",
                  "authentication", "unauthorized")


def uses_claude_login() -> bool:
    """True when the CLI under the SDK signs in with its stored login: a Claude
    subscription from /login, or CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`."""
    return not any(os.environ.get(name) for name in API_CREDENTIALS)


def probe_login(model: str) -> dict[str, Any]:
    """One short request through the SDK, checking the Claude login and the model
    before any episode. It uses a few tokens of the account's allowance."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    from ..runner import PreflightError

    hint = ("sign in with the Claude CLI (/login), set CLAUDE_CODE_OAUTH_TOKEN, or set "
            "ANTHROPIC_API_KEY; see the README's Planner sign-in section")

    async def ask():
        options = ClaudeAgentOptions(model=model, max_turns=1, tools=[], setting_sources=[],
                                     strict_mcp_config=True)
        result = None
        async for message in query(prompt="Reply with the single word OK.", options=options):
            if isinstance(message, ResultMessage):
                result = message
        return result

    try:
        # Preflight can run inside the runner's event loop, so ask on a thread.
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            result = pool.submit(asyncio.run, ask()).result(timeout=300)
    except Exception as exc:  # noqa: BLE001 - any failure here means no usable login
        raise PreflightError("Claude sign-in check failed ({}): {}".format(exc, hint)) from exc
    text = str(getattr(result, "result", "") or "")
    if result is None or any(m in text.lower() for m in LOGIN_FAILURES):
        raise PreflightError("Claude sign-in check failed ({!r}): {}".format(text[:200], hint))
    if throttle_tag(text) == "usage_limited":
        return {"checked": True, "ok": False, "auth": "claude login", "model": model,
                "note": "usage limit reached; episodes wait for it to reset"}
    if getattr(result, "is_error", False) and "ok" not in text.lower():
        raise PreflightError("Claude request with model {!r} failed: {}".format(model, text[:300]))
    return {"checked": True, "ok": True, "auth": "claude login", "model": model}


def json_safe(obj: Any) -> Any:
    """Best-effort JSON coercion for SDK dataclasses."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    for attr in ("model_dump", "dict", "_asdict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return json_safe(fn())
            except Exception:  # noqa: BLE001
                break
    if hasattr(obj, "__dict__"):
        return {k: json_safe(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)


def _tool_result_texts(block: Any) -> list[str]:
    """Pull the text parts out of a ToolResultBlock, ignoring images."""
    content = getattr(block, "content", None)
    if isinstance(content, str):
        return [content]
    texts: list[str] = []
    for part in content or []:
        if isinstance(part, dict):
            if part.get("type") == "text" and part.get("text"):
                texts.append(str(part["text"]))
        else:
            text = getattr(part, "text", None)
            if text:
                texts.append(str(text))
    return texts


def _strip_images(obj: Any) -> Any:
    """Replace base64 image payloads with their size, recursively."""
    if isinstance(obj, dict):
        if obj.get("type") == "image" and isinstance(obj.get("source"), dict):
            src = obj["source"]
            return {"type": "image", "source": {
                "type": src.get("type"), "media_type": src.get("media_type"),
                "data": "<{} base64 chars stripped>".format(len(str(src.get("data") or "")))}}
        if "data" in obj and isinstance(obj["data"], str) and (obj.get("format") in ("png", "jpeg", "jpg", "webp") or obj.get("type") == "image"):
            return {**obj, "data": "<{} chars stripped>".format(len(obj["data"]))}
        return {k: _strip_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_images(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:image/"):
        return "<image payload omitted>"
    if isinstance(obj, str) and obj.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(obj)
        except ValueError:
            return obj
        clean = _strip_images(parsed)
        return json.dumps(clean, ensure_ascii=False) if clean != parsed else obj
    return obj


class ClaudePlannerRuntime:
    name = "claude-sdk"

    def __init__(self) -> None:
        self.inherent: dict[str, Any] = {
            "auth": "claude login" if uses_claude_login() else "environment credentials"}

    # ── lifecycle ──

    def prepare(self, cfg: RunConfig) -> None:
        """Once per run, before any episode. Record versions; verify the SDK
        and CLI are importable/present so we fail here rather than mid-episode."""
        try:
            import claude_agent_sdk
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "claude-agent-sdk not importable in {} — pip install claude-agent-sdk"
                .format(sys.executable)
            ) from exc
        self.inherent["sdk_version"] = getattr(claude_agent_sdk, "__version__", "?")
        # Auth is the CLI's: environment credentials when set, otherwise its stored
        # login. Leave it alone and just record which endpoint is in play.
        self.inherent["base_url"] = os.environ.get("ANTHROPIC_BASE_URL", "(default)")

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {"planner_runtime": self.name, "inherent": dict(self.inherent),
                "options": json_safe(self._options(ctx))}

    # ── options ──

    def _options(self, ctx: EpisodeContext) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions
        from ..runner import mcp_launch

        return ClaudeAgentOptions(
            system_prompt=ctx.briefing,
            mcp_servers={
                "env": {
                    "type": "stdio",
                    **mcp_launch(ctx.mcp_env),
                }
            },
            tools=[],  # no built-in tools: vanilla ReAct over the env only
            # No filesystem settings: without this the CLI walks up from cwd and
            # injects any surrounding CLAUDE.md into every session.
            setting_sources=[],
            thinking=ctx.thinking,
            effort=ctx.effort,
            betas=list(ctx.betas),
            # ONLY our MCP tool server — never the user's global MCP config.
            strict_mcp_config=True,
            # The registered tools are the whole surface (built-in tools are off),
            # so no per-tool allow list is needed.
            permission_mode="bypassPermissions",
            # A tool result can carry several images; the default 1 MiB stdout
            # buffer truncates and kills the session mid-parse.
            max_buffer_size=32 * 1024 * 1024,
            max_turns=ctx.max_turns,
            max_budget_usd=ctx.max_budget_usd,
            model=ctx.model or None,
            cwd=str(ctx.workdir),
        )

    # ── the session ──

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeSDKClient,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        # raw.jsonl is the SDK message stream, kept for debugging a session
        # (nothing in the pipeline reads it). By default images are replaced by a
        # placeholder, since base64 images otherwise dominate a run's disk use.
        # NAVGPT_RAW_LOG=full keeps them, NAVGPT_RAW_LOG=0 writes nothing.
        raw_mode = (os.environ.get("NAVGPT_RAW_LOG") or "text").lower()
        raw_path = ctx.workdir / "raw.jsonl"
        result_msg: Any = None

        with raw_path.open("w") as raw:

            def record_raw(message: Any) -> None:
                if raw_mode in ("0", "off", "none"):
                    return
                obj = json_safe(message)
                if raw_mode != "full":
                    obj = _strip_images(obj)
                raw.write(json.dumps({"type": type(message).__name__, "msg": obj},
                                     ensure_ascii=False) + "\n")
                raw.flush()

            async with ClaudeSDKClient(options=self._options(ctx)) as client:
                # The CLI starts reasoning before MCP servers finish connecting;
                # gate the prompt on the MCP tool server reporting 'connected'.
                mcp_status: str | None = None
                for _ in range(60):
                    status = await client.get_mcp_status()
                    entries = status.get("mcpServers", []) if isinstance(status, dict) else []
                    mcp_status = next(
                        (e.get("status") for e in entries if e.get("name") == "env"), None
                    )
                    if mcp_status == "connected" or mcp_status in (
                        "failed", "needs-auth", "disabled",
                    ):
                        break
                    await asyncio.sleep(0.5)
                sink.emit("mcp_status", {"status": mcp_status})
                if mcp_status != "connected":
                    raise RuntimeError(
                        "NavGPT MCP tool server not connected: {} — run `python -m "
                        "navgpt.planner.tools.mcp_tools` to see its traceback".format(mcp_status)
                    )

                await client.query(ctx.first_prompt)
                async for message in client.receive_response():
                    record_raw(message)
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                sink.emit("assistant_text", {"text": block.text})
                            elif isinstance(block, ThinkingBlock):
                                sink.emit("thinking", {"chars": len(block.thinking),
                                                       "text": block.thinking})
                            elif isinstance(block, ToolUseBlock):
                                sink.emit("tool_use", {"id": block.id, "name": block.name,
                                                       "input": block.input})
                    elif isinstance(message, UserMessage):
                        content = message.content
                        blocks = content if isinstance(content, list) else []
                        for block in blocks:
                            if isinstance(block, ToolResultBlock):
                                sink.emit("tool_result", {
                                    "tool_use_id": block.tool_use_id,
                                    "texts": _tool_result_texts(block)})
                    elif isinstance(message, SystemMessage):
                        if getattr(message, "subtype", None) == "init":
                            data = getattr(message, "data", {}) or {}
                            sink.emit("system_init", {"model": data.get("model"),
                                                      "tools": data.get("tools")})
                    elif isinstance(message, ResultMessage):
                        result_msg = message

        # The Agent SDK sets is_error=True even on a clean subtype="success"
        # result — the flag tracks the session, not the navigation outcome, so an
        # episode that called stop and reached the goal can still come back
        # is_error=True. Score by the ENV terminal instead: "success" (normal
        # return, whatever the nav result), "error_max_turns" (clean truncation)
        # and "error_max_budget_usd" (USD fuse tripped — same clean-truncation
        # semantics) are SCORED outcomes; only a genuine execution error is a
        # broken session that propagates. Keeping the fuse subtype out of this
        # whitelist would make the driver retry the most expensive episodes —
        # the opposite of a budget cap.
        subtype = getattr(result_msg, "subtype", None)
        result_text = str(getattr(result_msg, "result", "") or "")
        error = None
        if throttle_tag(result_text):
            # A throttle returns subtype="success" is_error=True with a
            # "temporarily limiting requests" body — tag it retryable so it is
            # never scored as a navigation failure.
            error = throttle_tag(result_text)
        elif (
            getattr(result_msg, "is_error", False)
            and subtype not in ("error_max_turns", "error_max_budget_usd", "success")
        ):
            error = "sdk result {}".format(subtype or "is_error")

        return SessionOutcome(
            usage=json_safe(getattr(result_msg, "usage", None)) or {},
            cost_usd=getattr(result_msg, "total_cost_usd", None),
            turns=sink.planner_turns.count,
            error=error,
            subtype=subtype,
            extra={"duration_ms": getattr(result_msg, "duration_ms", None)},
        )
