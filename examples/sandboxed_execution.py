# 演示 CodeAgent 的多种沙箱执行环境
# 沙箱（Sandbox）：隔离的代码执行环境，防止 Agent 生成的代码访问宿主机敏感资源
#
# 为什么需要沙箱？
#   CodeAgent 会让 LLM 生成并执行 Python 代码，存在安全风险：
#   - 恶意代码可能读取/删除文件、访问网络、泄露环境变量等
#   - 生产环境必须使用沙箱，学习/测试时可以用默认的本地执行器
#
# 五种沙箱方案对比：
#   blaxel  - Blaxel 云沙箱（需要 Blaxel 账号）
#   docker  - 本地 Docker 容器（需要安装 Docker Desktop）
#   e2b     - E2B 云沙箱（需要 E2B API Key，https://e2b.dev）
#   modal   - Modal 云函数（需要 Modal 账号，https://modal.com）
#   wasm    - WebAssembly 沙箱（完全本地，无需账号，但功能受限）
#
# 前置依赖（按需安装）：
#   pip install smolagents[e2b]     # E2B 沙箱
#   pip install smolagents[docker]  # Docker 沙箱
#   pip install smolagents[modal]   # Modal 沙箱

from smolagents import CodeAgent, InferenceClientModel, WebSearchTool


model = InferenceClientModel()

# ============================================================
# 沙箱用法统一模式：使用 with 语句（上下文管理器）
# ============================================================
# with 语句会在退出时自动调用 agent.cleanup()，释放沙箱资源
# 等价于手动调用：agent = CodeAgent(...); agent.run(...); agent.cleanup()

# --- Blaxel 云沙箱 ---
# 需要：Blaxel 账号 + blaxel CLI 登录（blaxel login）
# 特点：托管云端，自动扩缩容，适合生产部署
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="blaxel") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Blaxel executor result:", output)

# --- Docker 本地沙箱 ---
# 需要：本地安装并运行 Docker Desktop
# 特点：完全本地，无网络依赖，隔离性强，适合企业内网环境
# Agent 代码在独立的 Docker 容器中执行，容器销毁后不留痕迹
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="docker") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Docker executor result:", output)

# --- E2B 云沙箱 ---
# 需要：E2B_API_KEY 环境变量（从 https://e2b.dev 获取）
# 特点：专为 AI 代码执行设计的云沙箱，启动快（~200ms），支持文件系统和网络
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="e2b") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("E2B executor result:", output)

# --- Modal 云函数沙箱 ---
# 需要：Modal 账号 + modal token set（或 MODAL_TOKEN_ID/MODAL_TOKEN_SECRET 环境变量）
# 特点：按需启动的无服务器函数，支持 GPU，适合计算密集型任务
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="modal") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print("Modal executor result:", output)

# --- WebAssembly（Wasm）本地沙箱 ---
# 需要：无（完全本地，基于 Pyodide + Deno 运行时）
# 特点：零配置，最安全（完全隔离），但功能受限：
#   - 不支持大多数 Python 扩展包
#   - 不支持工具调用（tools=[] 必须为空）
#   - 适合纯计算任务（数学、字符串处理等）
with CodeAgent(tools=[], model=model, executor_type="wasm") as agent:
    output = agent.run("Calculate the square root of 125.")
print("Wasm executor result:", output)

# TODO: Wasm 沙箱暂不支持工具，以下代码尚未实现
# with CodeAgent(tools=[VisitWebpageTool()], model=model, executor_type="wasm") as agent:
#     output = agent.run("What is the content of the Wikipedia page at https://en.wikipedia.org/wiki/Intelligent_agent?")
