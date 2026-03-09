# 演示如何使用 LiteLLMRouterModel 实现多模型负载均衡
# 应用场景：生产环境中提高可用性、分散 API 限流压力、降低单一供应商依赖
#
# 前置依赖：pip install smolagents[litellm]
# 需要设置环境变量：OPENAI_API_KEY、AWS_ACCESS_KEY_ID、AWS_SECRET_ACCESS_KEY、AWS_REGION

import os

from smolagents import CodeAgent, LiteLLMRouterModel, WebSearchTool


# ============================================================
# 配置多模型负载均衡列表
# ============================================================
# 每个条目代表一个可用的模型实例，同一 model_name 下的多个实例会被负载均衡调度
# model_name：逻辑分组名，Agent 使用此名称请求，Router 自动选择具体实例
# litellm_params：LiteLLM 格式的模型参数，支持任意 LiteLLM 兼容的模型

llm_loadbalancer_model_list = [
    {
        # 实例1：OpenAI GPT-4o-mini（成本较低，适合简单任务）
        "model_name": "model-group-1",   # 逻辑组名，同组内的实例会被轮流/随机调用
        "litellm_params": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),  # 从环境变量读取，避免硬编码密钥
        },
    },
    {
        # 实例2：AWS Bedrock 上的 Claude 3 Sonnet（备用，当 OpenAI 限流时自动切换）
        "model_name": "model-group-1",   # 同组名 → 与实例1 共同参与负载均衡
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_region_name": os.getenv("AWS_REGION"),
        },
    },
    # 可以继续添加更多实例，甚至不同的逻辑分组（model-group-2 等）
    # {
    #     "model_name": "model-group-2",
    #     "litellm_params": {
    #         "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    #         "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    #         "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    #         "aws_region_name": os.getenv("AWS_REGION"),
    #     },
    # },
]


# 创建路由模型
# model_id：指定使用哪个逻辑分组（对应上面 model_name 字段）
# model_list：所有可用模型实例的配置列表
# client_kwargs.routing_strategy：调度策略
#   - "simple-shuffle"：随机均匀分配（默认）
#   - "least-busy"：优先选择当前负载最低的实例
#   - "latency-based-routing"：优先选择响应最快的实例
model = LiteLLMRouterModel(
    model_id="model-group-1",
    model_list=llm_loadbalancer_model_list,
    client_kwargs={"routing_strategy": "simple-shuffle"},
)

# 创建 Agent，与普通模型用法完全相同，负载均衡对 Agent 层透明
# return_full_result=True：返回 RunResult 对象，可访问 token_usage、timing、steps 等详细信息
agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True, return_full_result=True)

full_result = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

# full_result 是 RunResult 对象，包含：
#   full_result.output      → 最终答案
#   full_result.state       → "success" 或 "max_steps_error"
#   full_result.steps       → 每一步的详细记录列表
#   full_result.token_usage → Token 消耗统计
#   full_result.timing      → 耗时信息
print(full_result)
