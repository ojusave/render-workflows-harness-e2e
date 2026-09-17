"""Hosted E2E workflow app for pydantic-ai-harness RenderWorkflows testing.

Deployed as a Render Workflow service; evidence comes back through task
results and platform run records (hosted task runs use isolated instances,
so no shared filesystem).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage
from render import Options, Retry, TaskContext, Workflows
from typing_extensions import TypedDict

from pydantic_ai_harness import RenderWorkflows


BOOT_ID = str(uuid.uuid4())  # unique per worker process: disambiguates instance sharing


class Deps(TypedDict):
    marker: str


def router_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Deterministic router: prompt 'use:<tool>' calls that tool once, then finishes."""
    del info
    returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(json.dumps({'tool_result': returns[-1].content}))])
    prompt = next(str(p.content) for m in messages for p in m.parts if isinstance(p, UserPromptPart))
    if prompt.startswith('use:'):
        return ModelResponse(parts=[ToolCallPart(prompt.removeprefix('use:'), {})])
    return ModelResponse(parts=[TextPart('no-tools')])


app = Workflows()
tools = FunctionToolset[Deps](id='tools')


def resolve_tool_options(_operation_id, _tool, tool_name: str):
    if tool_name == 'always_fail':
        return Options(retry=Retry(max_retries=2, wait_duration_ms=500))
    return None


runtime = RenderWorkflows[Deps](
    app,
    deps_type=Deps,
    model_options=Options(retry=Retry(max_retries=1, wait_duration_ms=500), timeout_seconds=120),
    tool_options=Options(retry=Retry(max_retries=1, wait_duration_ms=500), timeout_seconds=120),
    resolve_tool_options=resolve_tool_options,
)


@tools.tool
async def whoami(ctx: RunContext[Deps]) -> dict[str, object]:
    """Return process/host evidence and add a usage marker (tests effects transfer)."""
    ctx.usage.incr(RunUsage(details={'hosted_effect_marker': 1}))
    return {'pid': os.getpid(), 'host': socket.gethostname(), 'boot_id': BOOT_ID, 'marker': ctx.deps['marker']}


@tools.tool
async def always_fail(ctx: RunContext[Deps]) -> str:
    """Always raise: verifies Render exhausts child-task retries, then fails the run."""
    raise RuntimeError(f'deliberate failure on host {socket.gethostname()} pid {os.getpid()}')


@tools.tool
async def hang(ctx: RunContext[Deps]) -> str:
    """Hang: target for native cancellation."""
    del ctx
    await asyncio.sleep(600)
    return 'never'


agent = Agent[Deps, str](
    FunctionModel(router_model, model_name='hosted-router'),
    name='hosted',
    deps_type=Deps,
    toolsets=[tools],
    capabilities=[runtime],
)


@runtime.task(name='hosted-run', timeout_seconds=900)
async def hosted_run(ctx: TaskContext, prompt: str, deps: Deps) -> dict[str, object]:
    del ctx
    result = await agent.run(prompt, deps=deps)
    return {
        'output': result.output,
        'requests': result.usage.requests,
        'effect_marker': result.usage.details.get('hosted_effect_marker', 0),
        'entry_pid': os.getpid(),
        'entry_host': socket.gethostname(),
        'entry_boot_id': BOOT_ID,
        'marker': deps['marker'],
    }


@runtime.task(name='hosted-huge')
async def hosted_huge(ctx: TaskContext, deps: Deps) -> dict[str, object]:
    """A >4MB model request must be refused by the harness before dispatch."""
    del ctx
    result = await agent.run('x' * (5 * 1024 * 1024), deps=deps)
    return {'output': result.output}


if __name__ == '__main__':
    app.start()
