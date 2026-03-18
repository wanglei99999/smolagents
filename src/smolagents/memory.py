# =============================================================================
# memory.py —— Agent 记忆系统
#
# 本文件定义了 smolagents 的记忆模型，即 Agent 在 ReAct 循环中如何记录每一步的状态。
#
# 核心概念：
#   AgentMemory 是 Agent 的"对话历史 + 执行日志"合体，
#   每一步执行完后，Agent 将对应的 MemoryStep 追加到 memory.steps 中。
#   下一步调用 LLM 时，memory.write_memory_to_messages() 将所有步骤序列化为消息列表，
#   作为 LLM 的上下文输入（这就是 Agent 能"记住"之前做了什么的原因）。
#
# 记忆步骤类型（MemoryStep 的子类）：
#
#   SystemPromptStep  ← 系统提示词（每次 run 开始时设置，不重复追加）
#   TaskStep          ← 用户任务描述（一次 run 只有一个）
#   PlanningStep      ← 规划步骤（planning_interval 触发时生成，包含计划文本）
#   ActionStep        ← 行动步骤（ReAct 循环中最核心的步骤，每步一个）
#       - model_input_messages: 本步 LLM 收到的消息列表（完整上下文）
#       - model_output: LLM 的原始输出文本（思考 + 行动代码 / 工具调用）
#       - code_action: 解析出的代码（CodeAgent 专用）
#       - tool_calls: 工具调用列表（ToolCallingAgent 专用）
#       - observations: 执行结果 / 工具输出（写回历史供 LLM 下步参考）
#       - error: 执行错误（若有，LLM 下步会尝试修正）
#       - token_usage: 本步消耗的 Token 数
#   FinalAnswerStep   ← 最终答案（循环结束时生成，不写入对话历史）
#
# 序列化（summary_mode）：
#   summary_mode=True 时，ActionStep.to_messages() 会省略 model_output（减少 token 消耗）
#   summary_mode=True 时，PlanningStep.to_messages() 返回空列表（避免旧计划干扰新计划）
#   summary_mode=True 时，SystemPromptStep.to_messages() 返回空列表
#
# CallbackRegistry：
#   管理步骤回调函数（step_callbacks），每步完成后按类型触发对应的回调。
# =============================================================================

import inspect
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Type

from smolagents.models import ChatMessage, MessageRole, get_dict_from_nested_dataclasses
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


__all__ = ["AgentMemory"]


logger = getLogger(__name__)


@dataclass
class ToolCall:
    """记录 Agent 执行的一次工具调用请求（记忆中的持久化版本）。
    与 models.py 中的 ChatMessageToolCall 的区别：
      - ChatMessageToolCall 是 LLM API 返回的原始格式（用于 API 交互）
      - memory.ToolCall 是写入记忆的序列化版本（用于记录和回放）
    """
    name: str      # 工具名称
    arguments: Any # 工具参数（dict）
    id: str        # 唯一 ID，用于关联 ToolCallingAgent 的并行调用

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    """一步完整的 ReAct 行动记录（记忆中最核心的步骤类型）。
    
    ActionStep 是 ReAct 循环中最重要的数据结构，它记录了一轮完整的 Think-Act-Observe 过程。
    每个 ActionStep 包含：
    1. Think（思考）：model_output - LLM 的推理过程
    2. Act（行动）：tool_calls/code_action - 要执行的工具调用或代码
    3. Observe（观察）：observations - 执行结果
    
    这个类承担双重角色：
    - 运行记录：保存执行历史，用于调试和回放
    - 下一轮输入：通过 to_messages() 转换为 LLM 的上下文
    
    字段说明：
      step_number: 步骤编号（从 1 开始）
      timing: 执行时间统计（开始时间、结束时间、耗时）
      model_input_messages: 本步发给 LLM 的完整消息列表（包含历史）
      model_output_message: LLM 返回的完整消息对象（包含 tool_calls 等元数据）
      model_output: LLM 的原始输出文本（含思考和行动描述）
      code_action: 从 model_output 解析出的 Python 代码（CodeAgent 专用）
      tool_calls: 解析出的工具调用列表（ToolCallingAgent 专用）
      observations: 代码/工具执行后的输出结果字符串（写入下一步的上下文）⭐ ReAct 的关键
      observations_images: 多模态输入的图像（每步都会传入）
      action_output: 代码执行的 Python 返回值（未序列化的原始对象）
      error: 执行错误（LLM 下一步会看到错误并尝试修正）
      token_usage: 本步的 token 使用统计（输入 token 数、输出 token 数）
      is_final_answer: 是否是最后一步（调用了 final_answer 工具/函数）
    """
    # === 基础信息 ===
    step_number: int  # 步骤编号，用于标识这是第几轮 ReAct 循环
    timing: Timing  # 时间统计：start_time, end_time, duration
    
    # === Think（思考）阶段的数据 ===
    model_input_messages: list[ChatMessage] | None = None  # 发给 LLM 的输入（包含历史上下文）
    model_output_message: ChatMessage | None = None  # LLM 返回的完整消息对象
    model_output: str | list[dict[str, Any]] | None = None  # LLM 的文本输出（推理过程）
    
    # === Act（行动）阶段的数据 ===
    tool_calls: list[ToolCall] | None = None  # ToolCallingAgent：解析出的工具调用列表
    code_action: str | None = None  # CodeAgent：解析出的 Python 代码
    
    # === Observe（观察）阶段的数据 ===
    observations: str | None = None  # 执行结果（字符串格式），会被写入下一轮的 LLM 输入
    observations_images: list["PIL.Image.Image"] | None = None  # 多模态：图像输入
    action_output: Any = None  # 原始的 Python 对象（未序列化）
    error: AgentError | None = None  # 执行错误（也是观察的一部分）
    
    # === 元数据 ===
    token_usage: TokenUsage | None = None  # Token 使用统计（用于成本计算）
    is_final_answer: bool = False  # 是否调用了 final_answer（标记循环结束）

    def dict(self):
        """将 ActionStep 序列化为字典（用于保存和传输）。
        
        这个方法手动处理复杂字段的序列化：
        - tool_calls: 转换为字典列表
        - action_output: 使用 make_json_serializable 处理任意 Python 对象
        - model_input_messages: 递归序列化嵌套的 dataclass
        - observations_images: 转换为字节数据
        
        Returns:
            dict: 可 JSON 序列化的字典
        """
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ]
            if self.model_input_messages
            else None,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": make_json_serializable(get_dict_from_nested_dataclasses(self.model_output_message))
            if self.model_output_message
            else None,
            "model_output": self.model_output,
            "code_action": self.code_action,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        """⭐ ReAct 框架的核心方法：将此步骤转换为 LLM 的输入消息。
        
        这个方法实现了 ReAct 的"反馈闭环"：
        1. 将本步的 observations（观察结果）转换为消息
        2. 下一轮 LLM 调用时会看到这些消息
        3. LLM 基于观察结果调整策略
        
        消息顺序（对应 ReAct 的三个阶段）：
        1. Think: model_output（LLM 的推理过程）
        2. Act: tool_calls（工具调用或代码）
        3. Observe: observations（执行结果）或 error（错误信息）
        
        Args:
            summary_mode: 是否使用摘要模式
                - False（默认）：完整输出，包含 model_output
                - True：省略 model_output，只保留工具调用和观察（用于规划时减少 token）
        
        Returns:
            list[ChatMessage]: 消息列表，会被添加到下一轮 LLM 的输入中
        
        Example:
            >>> action_step = ActionStep(
            ...     model_output="我需要搜索天气",
            ...     tool_calls=[ToolCall(name="web_search", ...)],
            ...     observations="Paris: 20°C"
            ... )
            >>> messages = action_step.to_messages()
            >>> # messages 包含：
            >>> # 1. {"role": "assistant", "content": "我需要搜索天气"}
            >>> # 2. {"role": "tool_call", "content": "Calling tools: ..."}
            >>> # 3. {"role": "tool_response", "content": "Observation: Paris: 20°C"}
        """
        messages = []
        
        # === 1. Think（思考）：LLM 的推理过程 ===
        # 在摘要模式下省略，因为推理过程通常很长，且对下一步不是必需的
        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        # === 2. Act（行动）：工具调用或代码 ===
        # 告诉 LLM "我调用了哪些工具"
        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        # === 多模态支持：图像输入 ===
        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        # === 3. Observe（观察）：执行结果 ===
        # ⭐ 这是 ReAct 闭环的关键：将执行结果传递给下一轮 LLM
        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        
        # === 错误也是观察的一部分 ===
        # 让 LLM 看到错误信息，从而能够自我纠错
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    """规划步骤记录（planning_interval 触发时生成）。
    plan: LLM 生成的计划文本（格式："Here are the facts I know and the plan..."）
    注意：to_messages(summary_mode=True) 返回空列表，
    这样在生成新计划时不会受旧计划影响（避免路径依赖）。
    """
    model_input_messages: list[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def dict(self):
        return {
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ],
            "model_output_message": make_json_serializable(
                get_dict_from_nested_dataclasses(self.model_output_message)
            ),
            "plan": self.plan,
            "timing": self.timing.dict(),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]
            ),
            # This second message creates a role change to prevent models models from simply continuing the plan message
        ]


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any


class AgentMemory:
    """Agent 的记忆容器，保存系统提示词以及执行过程中产生的所有步骤。

    该类用于记录 Agent 的运行轨迹，包括任务步骤、行动步骤和规划步骤。
    它支持重置记忆、获取简略版或完整版步骤信息，以及回放整个执行过程。

    Args:
        system_prompt (`str`): Agent 的系统提示词，用于设定行为上下文和基础指令。

    **Attributes**:
        - **system_prompt** (`SystemPromptStep`) -- Agent 的系统提示词步骤。
        - **steps** (`list[TaskStep | ActionStep | PlanningStep]`) -- Agent 已执行的步骤列表，可能包含任务、行动和规划步骤。
    """

    def __init__(self, system_prompt: str):
        self.system_prompt: SystemPromptStep = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        """重置 Agent 的记忆，清空所有步骤，但保留系统提示词。"""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """返回步骤的精简表示，不包含传给模型的输入消息。"""
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """返回步骤的完整表示，包含传给模型的输入消息。"""
        if len(self.steps) == 0:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """以较易读的形式回放 Agent 的执行步骤。

        Args:
            logger (`AgentLogger`): 用于输出回放日志的记录器。
            detailed (`bool`, default `False`): 若为 True，还会展示每一步对应的记忆内容。
                注意：这会显著增加日志长度，建议仅在调试时使用。
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)

    def return_full_code(self) -> str:
        """返回 Agent 各步骤中的所有代码动作，并拼接成一个脚本。"""
        return "\n\n".join(
            [step.code_action for step in self.steps if isinstance(step, ActionStep) and step.code_action is not None]
        )


class CallbackRegistry:
    """用于管理 Agent 每一步执行后要触发的回调函数。

    你可以把它理解成一个“按步骤类型分发事件”的小型注册中心。

    工作方式分为两步：
    1. 通过 `register(step_cls, callback)` 注册回调函数
       表示“当某种步骤执行完成后，要调用这个函数”。
    2. 通过 `callback(memory_step, **kwargs)` 触发回调
       会根据 `memory_step` 的实际类型，把匹配的回调函数逐个执行。

    内部数据结构：
        `_callbacks` 是一个字典，结构如下：

        {
            ActionStep: [callback_a, callback_b],
            PlanningStep: [callback_c],
            MemoryStep: [callback_d],
        }

        含义是：
        - 当完成 `ActionStep` 时，触发 `callback_a` 和 `callback_b`
        - 当完成 `PlanningStep` 时，触发 `callback_c`
        - 当完成任意 `MemoryStep` 子类步骤时，也会触发 `callback_d`

    为什么会触发父类上的回调：
        在触发时，代码会沿着 `memory_step` 的继承链（`__mro__`）向上查找。
        例如某一步是 `ActionStep`，那么不仅会执行注册到 `ActionStep` 上的回调，
        也会执行注册到其父类 `MemoryStep` 上的回调。

    这样设计的好处是：
        - 可以只监听某一种具体步骤
        - 也可以统一监听所有步骤
        - 不需要把日志、监控、UI 更新等附加逻辑硬编码到主流程里
    """

    def __init__(self):
        # 结构：{步骤类型: [该类型对应的回调函数列表]}
        self._callbacks: dict[Type[MemoryStep], list[Callable]] = {}

    def register(self, step_cls: Type[MemoryStep], callback: Callable):
        """为某种步骤类型注册回调函数。

        Args:
            step_cls (Type[MemoryStep]): 要注册回调的步骤类型。
            callback (Callable): 要注册的回调函数。

        Example:
            `register(ActionStep, my_callback)` 表示：
            每当一个 `ActionStep` 执行完成后，都调用一次 `my_callback`。
        """
        if step_cls not in self._callbacks:
            self._callbacks[step_cls] = []
        self._callbacks[step_cls].append(callback)

    def callback(self, memory_step, **kwargs):
        """触发某个步骤类型已注册的回调函数。

        Args:
            memory_step (MemoryStep): 当前完成的步骤对象，将据此查找并执行对应回调。
            **kwargs: 传递给回调函数的额外参数。
                通常会包含当前 agent 实例等上下文信息。

        Notes:
            为了兼容旧版回调：
            如果回调函数只接收一个参数，则仅传入 `memory_step`；
            如果回调函数接收多个参数，则同时传入 `memory_step` 和额外的 `kwargs`。

        执行流程：
            1. 读取 `memory_step` 的实际类型，例如 `ActionStep`
            2. 沿着该类型的继承链向上查找，例如：
               `ActionStep -> MemoryStep -> object`
            3. 对每个类型，取出已注册的回调列表
            4. 将这些回调逐个执行

        Example:
            假设注册了：
            - `register(ActionStep, on_action)`
            - `register(MemoryStep, on_any_step)`

            当传入一个 `ActionStep` 时：
            - `on_action(...)` 会被调用
            - `on_any_step(...)` 也会被调用
        """
        # 兼容旧版只接收一个 step 参数的回调写法
        # __mro__ 表示类的继承链，例如：
        # ActionStep.__mro__ == (ActionStep, MemoryStep, object)
        # 因此这里既会触发注册在具体子类上的回调，也会触发注册在父类上的回调。
        for cls in memory_step.__class__.__mro__:
            for cb in self._callbacks.get(cls, []):
                # 如果回调函数只定义了一个参数，就只传入当前步骤。
                # 如果回调函数还定义了更多参数，则把额外上下文 kwargs 也一起传入。
                cb(memory_step) if len(inspect.signature(cb).parameters) == 1 else cb(memory_step, **kwargs)
