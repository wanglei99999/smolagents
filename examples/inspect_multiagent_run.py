# 演示如何使用 OpenTelemetry + Arize Phoenix 对多 Agent 运行进行可观测性追踪
# 可观测性（Observability）：记录并可视化每一步的 LLM 调用、工具使用、Token 消耗等信息
#
# 前置依赖：
#   pip install openinference-instrumentation-smolagents arize-phoenix-otel
#   启动 Phoenix UI：python -m phoenix.server.main serve（默认 http://localhost:6006）

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register


# 向 Phoenix 后端注册 OpenTelemetry tracer（追踪器）
# 默认连接本地 Phoenix 服务（http://localhost:6006），可通过环境变量配置远程地址
register()

# 对 smolagents 框架进行自动插桩（Instrumentation）
# 插桩后，所有 Agent 的 LLM 调用、工具执行都会被自动记录并发送到 Phoenix
# skip_dep_check=True：跳过依赖版本检查，避免版本冲突报错
SmolagentsInstrumentor().instrument(skip_dep_check=True)


# ⚠️ 注意：插桩代码必须放在 smolagents 导入之前！
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    ToolCallingAgent,
    VisitWebpageTool,   # 内置工具：访问并提取网页内容
    WebSearchTool,      # 内置工具：执行网络搜索
)


# ============================================================
# 构建多 Agent 系统（Manager + Sub-Agent 架构）
# ============================================================

# 共享同一个模型后端（也可以为不同 Agent 使用不同模型）
model = InferenceClientModel(provider="nebius")

# --- 子 Agent：专门负责网络搜索 ---
# name 和 description 是作为 managed_agent 的必填项
# 主 Agent 会根据 description 决定何时调用这个子 Agent
# return_full_result=True：返回完整的 RunResult 对象（含 token 用量、耗时等）
search_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
    return_full_result=True,
)

# --- 主 Agent：负责规划和协调，将搜索任务委托给子 Agent ---
# tools=[]：主 Agent 本身不直接使用工具，而是通过调用子 Agent 来完成任务
# managed_agents：注册可被调用的子 Agent 列表
# return_full_result=True：返回 RunResult 而非直接返回最终答案字符串
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
    return_full_result=True,
)

# 运行主 Agent，它会自动分解任务并调度子 Agent
run_result = manager_agent.run(
    "If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?"
)

# RunResult 包含丰富的运行元数据
# token_usage：统计整个运行过程中消耗的 prompt/completion token 数量
print("Here is the token usage for the manager agent", run_result.token_usage)
# timing：记录运行的开始时间、结束时间和总耗时
print("Here are the timing informations for the manager agent:", run_result.timing)
