# 演示如何通过 MCP（Model Context Protocol）集成外部工具，并使用结构化输出
#
# MCP（Model Context Protocol）：Anthropic 提出的开放协议，用于标准化 LLM 与外部工具的通信
# 优势：工具可以用任何语言实现，通过标准协议与 Agent 通信，实现工具的跨平台复用
#
# 结构化输出（Structured Output）：工具返回 Pydantic 模型而非字符串
# 优势：LLM 可以精确解析返回值的每个字段，避免字符串解析错误
#
# 本示例流程：
#   1. 在同一进程中内联启动一个 MCP 天气服务（通常 MCP 服务是独立进程）
#   2. 用 MCPClient 连接该服务，获取工具列表
#   3. 将 MCP 工具注入 CodeAgent，Agent 调用时自动处理协议转换
#
# 运行方式（推荐用 uv 隔离依赖）：
#   uv run structured_output_tool.py
#
# 前置依赖：pip install smolagents[mcp,litellm] pydantic mcp

from textwrap import dedent

from mcp import StdioServerParameters   # MCP 服务启动参数（通过标准输入输出通信）

from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, MCPClient  # noqa: F401


def weather_server_script() -> str:
    """
    返回一个内联的 MCP 服务器 Python 脚本字符串。
    实际项目中，MCP 服务通常是独立部署的进程，这里为了演示方便将其内联。
    """
    return dedent(
        '''
        from pydantic import BaseModel, Field
        from mcp.server.fastmcp import FastMCP

        # 创建 MCP 服务实例，命名为 "Weather Service"
        mcp = FastMCP("Weather Service")

        # 使用 Pydantic 定义结构化的返回类型
        # 每个字段都有类型约束和描述，MCP 会将其转换为 JSON Schema
        class WeatherInfo(BaseModel):
            location: str = Field(description="The location name")
            temperature: float = Field(description="Temperature in Celsius")
            conditions: str = Field(description="Weather conditions")
            humidity: int = Field(description="Humidity percentage", ge=0, le=100)  # ge/le 是数值范围约束

        # @mcp.tool 将函数注册为 MCP 工具
        # 返回 Pydantic 模型实例 → 自动序列化为结构化 JSON 返回给 Agent
        @mcp.tool(
            name="get_weather_info",
            description="Get weather information for a location as structured data.",
        )
        def get_weather_info(city: str) -> WeatherInfo:
            """Get weather information for a city."""
            # 演示数据，实际应调用真实天气 API
            return WeatherInfo(
                location=city,
                temperature=22.5,
                conditions="partly cloudy",
                humidity=65
            )

        # 启动 MCP 服务，通过标准输入输出（stdio）与调用方通信
        mcp.run()
        '''
    )


def main() -> None:
    # ============================================================
    # 第一步：配置 LLM 模型
    # ============================================================

    # 备选：使用 HuggingFace 推理 API
    # model = InferenceClientModel()

    # 当前：使用 LiteLLM 调用 Mistral（需要 MISTRAL_API_KEY 环境变量）
    model = LiteLLMModel(
        model_id="mistral/mistral-small-latest",
        # model_id="openai/gpt-4o-mini",  # 备选 OpenAI 模型
    )

    # ============================================================
    # 第二步：启动 MCP 服务并获取工具
    # ============================================================

    # StdioServerParameters：定义如何启动 MCP 服务进程
    # command="python"：用 python 运行
    # args=["-c", script]：通过 -c 参数直接执行脚本字符串（内联方式）
    # 实际项目中通常是：StdioServerParameters(command="python", args=["weather_server.py"])
    serverparams = StdioServerParameters(command="python", args=["-c", weather_server_script()])

    # MCPClient：连接 MCP 服务，自动发现并注册所有工具
    # structured_output=True：启用结构化输出，工具返回的 Pydantic 对象会被保留结构
    #   False（默认）：工具返回值转换为字符串，LLM 需要解析文本
    #   True：工具返回值保留 JSON 结构，LLM 可以直接访问字段
    # 使用 with 语句确保 MCP 服务进程在使用完毕后被正确终止
    with MCPClient(
        serverparams,
        structured_output=True,
    ) as tools:
        # tools 是从 MCP 服务自动发现的工具列表，直接传给 Agent
        agent = CodeAgent(tools=tools, model=model)

        # Agent 会：
        # 1. 调用 get_weather_info(city="Tokyo") 获取结构化天气数据
        # 2. 从返回的 WeatherInfo 对象中读取 temperature 字段（22.5°C）
        # 3. 进行单位换算：22.5 * 9/5 + 32 = 72.5°F
        # 4. 返回最终答案
        agent.run("What is the temperature in Tokyo in Fahrenheit?")


if __name__ == "__main__":
    main()
