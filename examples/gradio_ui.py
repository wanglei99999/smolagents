# 演示如何为 Agent 快速搭建一个 Gradio 网页交互界面
# 只需 2 行核心代码，即可将任意 Agent 包装成可视化 Web UI

from smolagents import CodeAgent, GradioUI, InferenceClientModel, WebSearchTool


# 创建一个具备网页搜索能力的 CodeAgent
agent = CodeAgent(
    tools=[WebSearchTool()],   # 内置网页搜索工具，Agent 可自主决定是否调用
    model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="fireworks-ai"),
    verbosity_level=1,         # 日志详细程度：0=静默, 1=基本信息, 2=完整调试
    planning_interval=3,       # 每执行 3 步触发一次规划步骤，适合复杂多步任务
    name="example_agent",      # Agent 名称（作为 managed_agent 时必填）
    description="This is an example agent.",  # Agent 描述（作为 managed_agent 时必填）
    step_callbacks=[],         # 每步执行后的回调函数列表，可用于监控/记录
    stream_outputs=True,       # 流式输出：实时显示 LLM 生成内容，提升交互体验
    # use_structured_outputs_internally=True,  # 启用结构化输出（实验性功能）
)

# 用 GradioUI 包装 Agent，一键启动 Web 界面
# file_upload_folder：允许用户上传文件到此目录，Agent 可访问这些文件
# launch() 默认在 http://127.0.0.1:7860 启动，浏览器会自动打开
GradioUI(agent, file_upload_folder="./data").launch()
