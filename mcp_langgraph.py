import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from llm import llm


# 1. Визначення стану графа
class State(TypedDict):
    messages: Annotated[list, add_messages]


async def main():
    # 1. Ініціалізуємо MultiServerMCPClient без контекстного менеджера (без async with)
    client = MultiServerMCPClient(
        {
            "math_server": {
                "command": ".venv/bin/python",
                "args": ["mcp_server.py"],
                "transport": "stdio",
            }
        }
    )

    # 2. Отримуємо інструменти через await
    tools = await client.get_tools()
    print(f"Завантажено MCP інструментів: {len(tools)}")

    # 3. Прив'язуємо інструменти до LLM
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # 4. Вузли графа
    async def chatbot_node(state: State):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def should_continue(state: State):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # 5. Збірка графа
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", should_continue, ["tools", END])
    builder.add_edge("tools", "chatbot")

    graph = builder.compile()

    # 6. Запуск
    # query = "Знайди невизначений інтеграл від x*cos(x)"
    query = "Знайди похідну від x*sin(x)+cos(x)"
    async for chunk in graph.astream(
        {"messages": [("user", query)]}, stream_mode="values"
    ):
        last_msg = chunk["messages"][-1]
        if last_msg.content:
            print(f"[{last_msg.type.upper()}]: {last_msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
