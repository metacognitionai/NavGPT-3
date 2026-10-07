"""Codex Python SDK adapter over the same NavGPT stdio MCP tool server."""
from __future__ import annotations

import asyncio
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..runner import (EpisodeContext, EventSink, PreflightError, SessionOutcome,
                      expected_tools, mcp_launch, throttle_tag)
from ..turns import TURN_UNIT


def validate_config(cfg: RunConfig) -> None:
    if cfg.max_budget_usd is not None:
        raise PreflightError("Codex SDK does not report charged USD or enforce a USD cap. "
                             "Set planner.max_budget_usd to null; turn and time limits still apply.")
    if cfg.max_turns < 1:
        raise PreflightError("max_turns must be positive")


def probe_model(cfg: RunConfig) -> dict[str, Any]:
    """Check installed SDK, local login, and advertised model without inference."""
    from openai_codex import Codex, CodexConfig
    validate_config(cfg)
    with Codex(CodexConfig(codex_bin=os.environ.get("NAVGPT_CODEX_BIN") or None)) as client:
        account = client.account()
        if account.account is None:
            raise PreflightError("Codex is not logged in locally. Run codex login on this machine.")
        models = {m.model for m in client.models(include_hidden=True).data}
        if cfg.model not in models:
            raise PreflightError("Codex model {!r} is not advertised by this account; set planner.model to one of {}"
                                 .format(cfg.model, sorted(models)))
    return {"checked": True, "model": cfg.model, "sdk_version": version("openai-codex"),
            "check": "local login and model inventory; inference not probed"}


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<image payload omitted>" if k in ("data", "imageUrl") and isinstance(v, str)
                    and (len(v) > 4096 or value.get("type") in ("image", "base64") or k == "imageUrl") else _scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return "<image payload omitted>"
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        clean = _scrub(parsed)
        return json.dumps(clean, ensure_ascii=False) if clean != parsed else value
    return value




def runtime_config(ctx: EpisodeContext) -> dict[str, Any]:
    """Keep account authentication local, disable unrelated tool providers."""
    # CLI overrides merge by key. Disable every configured external MCP server
    # explicitly, then verify the resulting inventory before submitting a prompt.
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config_path = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    existing = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
    servers = {name: {"enabled": False} for name in existing.get("mcp_servers", {})}
    servers["env"] = {**mcp_launch(ctx.mcp_env), "enabled": True, "required": True,
                      "startup_timeout_sec": 60, "tool_timeout_sec": ctx.timeout,
                      "enabled_tools": sorted(expected_tools(ctx.mcp_env["NAVGPT_CONDITION"], ctx.mcp_env.get("NAVGPT_CAPABILITY_BUNDLE", "full"))),
                      "default_tools_approval_mode": "approve"}
    disabled = ("shell_tool", "unified_exec", "shell_snapshot", "apps", "plugins", "hooks",
                "multi_agent", "multi_agent_v2", "browser_use", "computer_use", "in_app_browser",
                "image_generation", "view_image", "code_mode", "goals",
                "skill_search", "skill_mcp_dependency_install", "tool_suggest", "memories",
                "remote_plugin", "workspace_dependencies")
    # The pinned runtime routes MCP calls through its code-mode host even when
    # code mode itself is disabled. Keep that transport enabled. Preapprove only
    # this explicit simulator tool allowlist; escalation remains deny_all.
    return {"mcp_servers": servers, "features": {**{k: False for k in disabled}, "code_mode_host": True},
            "web_search": "disabled", "project_doc_max_bytes": 0,
            "developer_instructions": "",
            "personality": "none"}


class CodexPlannerRuntime:
    name = "codex-sdk"

    def prepare(self, cfg: RunConfig) -> None:
        import openai_codex  # noqa: F401
        validate_config(cfg)

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        return {"planner_runtime": self.name, "inherent": {
            "sdk_version": version("openai-codex"), "auth": "local Codex account",
            "service_tier": os.environ.get("NAVGPT_CODEX_SERVICE_TIER", "inherited"),
            "cost_usd": "unavailable", "turn_unit": TURN_UNIT}}

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        from openai_codex import AsyncCodex, ApprovalMode, CodexConfig, Sandbox
        from openai_codex.generated.v2_all import ListMcpServerStatusResponse, ReasoningEffort
        # Set startup overrides as well as thread overrides: unrelated global
        # MCP servers must not start even before thread creation.
        settings = runtime_config(ctx)
        if os.environ.get("NAVGPT_CODEX_SERVICE_TIER") == "default":
            settings["service_tier"] = "default"
            settings["features"]["fast_mode"] = False
        def toml(v):
            if isinstance(v, dict):
                return '{' + ', '.join(json.dumps(k) + '=' + toml(x) for k, x in v.items()) + '}'
            return json.dumps(v)
        startup = CodexConfig(codex_bin=os.environ.get("NAVGPT_CODEX_BIN") or None,
            cwd=str(ctx.workdir), config_overrides=tuple(
            k + '=' + toml(v) for k, v in settings.items()))
        usage: dict[str, Any] = {}
        limited = False
        terminal = None
        error = None
        async with AsyncCodex(startup) as client:
            thread = await client.thread_start(model=ctx.model, cwd=str(ctx.workdir),
                base_instructions=ctx.briefing, developer_instructions="", config=settings,
                ephemeral=True, approval_mode=ApprovalMode.deny_all, sandbox=Sandbox.read_only)
            # The SDK has no wrapper for MCP server status yet, so ask its client directly.
            for _ in range(120):
                status = await client._client.request("mcpServerStatus/list",
                    {"threadId": thread.id, "limit": 100}, response_model=ListMcpServerStatusResponse)
                active = {s.name: s for s in status.data if s.tools}
                if set(active) - {"env"}:
                    raise RuntimeError("Unexpected Codex MCP servers: " + ', '.join(sorted(active)))
                actual = set(active["env"].tools) if "env" in active else set()
                # Server inventory keys may be raw or namespace-qualified.
                actual = {name.rsplit('__', 1)[-1] for name in actual}
                want = set(expected_tools(ctx.mcp_env["NAVGPT_CONDITION"], ctx.mcp_env.get("NAVGPT_CAPABILITY_BUNDLE", "full")))
                if actual == want:
                    break
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError("Codex MCP inventory mismatch: {} expected {}".format(sorted(actual), sorted(want)))
            sink.emit("mcp_status", {"status": "connected", "tools": sorted(actual)})
            sink.emit("system_init", {"model": ctx.model, "tools": sorted(actual), "thread_id": thread.id})
            turn = await thread.turn(ctx.first_prompt, effort=ReasoningEffort(ctx.effort) if ctx.effort else None)
            try:
                with (ctx.workdir / "raw.jsonl").open("w") as raw:
                    async for event in turn.stream():
                        payload = event.payload.model_dump(mode="json", by_alias=True) if hasattr(event.payload, "model_dump") else event.payload.params
                        raw.write(json.dumps({"type": event.method, "msg": _scrub(payload)}) + '\n')
                        raw.flush()
                        if event.method in ("item/started", "item/completed"):
                            item = payload.get("item", {})
                            kind = item.get("type")
                            if kind == "mcpToolCall":
                                if event.method == "item/started":
                                    sink.emit("tool_use", {"id": item["id"], "name": "mcp__env__" + item["tool"], "input": item["arguments"]})
                                else:
                                    texts = [p["text"] for p in (item.get("result") or {}).get("content", []) if p.get("type") == "text"]
                                    sink.emit("tool_result", {"tool_use_id": item["id"], "texts": _scrub(texts)})
                                    if item.get("error"):
                                        error = str(item["error"])
                            elif event.method == "item/completed" and kind == "agentMessage":
                                text = item.get("text", "")
                                sink.emit("assistant_text", {"text": text})
                                if throttle_tag(text):
                                    error = throttle_tag(text)
                            elif event.method == "item/completed" and kind == "reasoning":
                                text = '\n'.join(item.get("summary") or [])
                                sink.emit("thinking", {"text": text, "chars": len(text)})
                            elif kind in ("commandExecution", "fileChange", "webSearch", "dynamicToolCall"):
                                await turn.interrupt()
                                raise RuntimeError("Unexpected non-navigation tool: " + str(kind))
                        elif event.method == "thread/tokenUsage/updated":
                            usage = payload["tokenUsage"]["total"]
                        elif event.method == "turn/completed":
                            terminal = payload["turn"]
                        # Enforce the same feedback/response unit we report. Wait
                        # for this response's tools to finish before interrupting;
                        # accounting notifications cannot consume the turn budget.
                        if (event.method == "item/completed" and
                                payload.get("item", {}).get("type") == "mcpToolCall" and
                                sink.planner_turns.count >= ctx.max_turns and
                                not sink.planner_turns.pending and not limited):
                            limited = True
                            await turn.interrupt()
            except asyncio.CancelledError:
                await asyncio.shield(turn.interrupt())
                raise
        truncated = limited and terminal is not None and terminal.get("status") == "interrupted"
        subtype = "error_max_turns" if truncated else "success"
        if not terminal:
            error = error or "Codex stream ended without turn/completed"
        elif terminal.get("status") != "completed" and not truncated:
            error = str(terminal.get("error") or terminal.get("status"))
            subtype = "codex_error"
        if error and throttle_tag(error):
            error = throttle_tag(error)
        normalized = {"input_tokens": max(0, usage.get("inputTokens", 0) - usage.get("cachedInputTokens", 0)),
                      "output_tokens": usage.get("outputTokens", 0),
                      "cache_read_input_tokens": usage.get("cachedInputTokens", 0)}
        return SessionOutcome(turns=sink.planner_turns.count, cost_usd=None, usage=normalized, error=error,
                              subtype=subtype, extra={"thread_id": thread.id,
                              "turn_unit": TURN_UNIT, "codex_usage": usage})
