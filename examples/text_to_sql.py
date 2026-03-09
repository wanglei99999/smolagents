# 演示 Text-to-SQL：让 Agent 将自然语言问题转换为 SQL 查询并执行
# 核心思路：将数据库查询能力封装成一个工具，Agent 自主生成并执行 SQL 语句
#
# 前置依赖：pip install smolagents sqlalchemy

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    inspect,   # 用于反射（Reflection）：运行时读取数据库表结构
    text,      # 用于执行原始 SQL 字符串
)


# ============================================================
# 第一步：创建内存数据库并初始化测试数据
# ============================================================

# 使用 SQLite 内存数据库（无需文件，程序结束后数据消失）
# 生产环境可替换为：
#   PostgreSQL: "postgresql://user:password@localhost/dbname"
#   MySQL:      "mysql://user:password@localhost/dbname"
#   SQLite文件: "sqlite:///./mydb.db"
engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# 定义 receipts（收据）表结构
table_name = "receipts"
receipts = Table(
    table_name,
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),          # 收据编号
    Column("customer_name", String(16), primary_key=True),    # 客户姓名（最长16字符）
    Column("price", Float),                                    # 消费金额
    Column("tip", Float),                                      # 小费金额
)
# 在数据库中创建上面定义的所有表
metadata_obj.create_all(engine)

# 插入测试数据
rows = [
    {"receipt_id": 1, "customer_name": "Alan Payne",      "price": 12.06, "tip": 1.20},
    {"receipt_id": 2, "customer_name": "Alex Mason",      "price": 23.86, "tip": 0.24},
    {"receipt_id": 3, "customer_name": "Woodrow Wilson",  "price": 53.43, "tip": 5.43},
    {"receipt_id": 4, "customer_name": "Margaret James",  "price": 21.11, "tip": 1.00},
]
for row in rows:
    stmt = insert(receipts).values(**row)
    with engine.begin() as connection:  # begin() 自动提交事务
        cursor = connection.execute(stmt)

# 通过反射读取表结构，用于构造工具描述（让 LLM 知道表有哪些列）
inspector = inspect(engine)
columns_info = [(col["name"], col["type"]) for col in inspector.get_columns("receipts")]

# 生成表结构描述，将嵌入到工具的 docstring 中，帮助 LLM 生成正确的 SQL
table_description = "Columns:\n" + "\n".join([f"  - {name}: {col_type}" for name, col_type in columns_info])
print(table_description)


# ============================================================
# 第二步：将 SQL 执行能力封装为 Agent 工具
# ============================================================

from smolagents import tool


# 关键设计：在 docstring 中直接嵌入表结构信息
# LLM 会读取这段描述来理解数据库结构，从而生成正确的 SQL
# 注意：这里的表结构描述是静态硬编码的，实际项目中可以动态生成
@tool
def sql_engine(query: str) -> str:
    """
    Allows you to perform SQL queries on the table. Returns a string representation of the result.
    The table is named 'receipts'. Its description is as follows:
        Columns:
        - receipt_id: INTEGER
        - customer_name: VARCHAR(16)
        - price: FLOAT
        - tip: FLOAT

    Args:
        query: The query to perform. This should be correct SQL.
    """
    output = ""
    # 使用 connect() 执行只读查询（不自动提交事务）
    with engine.connect() as con:
        rows = con.execute(text(query))  # text() 将字符串包装为可执行的 SQL 对象
        for row in rows:
            output += "\n" + str(row)   # 将每行结果转为字符串拼接
    return output


# ============================================================
# 第三步：创建 Agent 并运行自然语言查询
# ============================================================

from smolagents import CodeAgent, InferenceClientModel


# 使用较小的模型（8B）即可完成 Text-to-SQL 任务
# CodeAgent 会生成类似以下的 Python 代码：
#   result = sql_engine(query="SELECT customer_name FROM receipts ORDER BY price DESC LIMIT 1")
#   final_answer(result)
agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"),
)

# Agent 将自动：
#   1. 理解问题（找出消费最贵的客户）
#   2. 根据工具描述中的表结构生成正确的 SQL
#   3. 调用 sql_engine 执行 SQL
#   4. 返回查询结果
agent.run("Can you give me the name of the client who got the most expensive receipt?")
