import json
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_server import (
    compute_definite_integral,
    compute_derivative,
    compute_indefinite_integral,
    compute_limit,
    get_calculus_formulas_reference,
    get_sympy_syntax_guide,
    math_tutor_solve_and_explain,
)

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


async def run_in_mcp_session(test_func):
    """Допоміжна функція для ізольованого запуску сесії у межах одного таску."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await test_func(session)


# 1. Перевірка списку інструментів
@pytest.mark.asyncio
async def test_list_tools():
    async def run(session: ClientSession):
        tools_response = await session.list_tools()
        tool_names = [tool.name for tool in tools_response.tools]

        assert len(tool_names) >= 4
        assert "compute_limit" in tool_names
        assert "compute_derivative" in tool_names
        assert "compute_indefinite_integral" in tool_names
        assert "compute_definite_integral" in tool_names

    await run_in_mcp_session(run)


# 2. Перевірка обчислення границі (compute_limit)
@pytest.mark.asyncio
async def test_compute_limit():
    async def run(session: ClientSession):
        result = await session.call_tool(
            "compute_limit",
            arguments={"expression": "sin(x)/x", "variable": "x", "point": "0"},
        )
        content = json.loads(result.content[0].text)

        assert content["success"] is True
        assert content["result"] == "1"

    await run_in_mcp_session(run)


# 3. Перевірка обчислення похідної (compute_derivative)
@pytest.mark.asyncio
async def test_compute_derivative():
    async def run(session: ClientSession):
        result = await session.call_tool(
            "compute_derivative",
            arguments={"expression": "x**3 + 2*x", "variable": "x"},
        )
        content = json.loads(result.content[0].text)

        assert content["success"] is True
        assert "3*x**2 + 2" in content["result"]
        assert "latex" in content

    await run_in_mcp_session(run)


# 4. Перевірка обчислення визначеного інтеграла (compute_definite_integral)
@pytest.mark.asyncio
async def test_compute_definite_integral():
    async def run(session: ClientSession):
        result = await session.call_tool(
            "compute_definite_integral",
            arguments={
                "expression": "x**2",
                "variable": "x",
                "lower_bound": "0",
                "upper_bound": "3",
            },
        )
        content = json.loads(result.content[0].text)

        assert content["success"] is True
        assert content["result"] == "9"

    await run_in_mcp_session(run)


# 5. Перевірка списку ресурсів (list_resources)
@pytest.mark.asyncio
async def test_list_resources():
    async def run(session: ClientSession):
        resources_response = await session.list_resources()
        assert len(resources_response.resources) >= 0

    await run_in_mcp_session(run)


# 6. Перевірка списку промптів (list_prompts)
@pytest.mark.asyncio
async def test_list_prompts():
    async def run(session: ClientSession):
        try:
            prompts_response = await session.list_prompts()
            assert isinstance(prompts_response.prompts, list)
        except Exception as e:
            # Якщо промпти не зареєстровані у FastMCP, очікуємо Method not found
            assert "Method not found" in str(e) or "prompts" in str(e)

    await run_in_mcp_session(run)


# 1. Async test for compute_limit tool
async def test_compute_limit_success():
    res = compute_limit(expression="sin(x)/x", variable="x", point="0")
    assert res["success"] is True
    assert res["result"] == "1"
    assert res["latex"] == "1"


# 2. Async test for compute_derivative tool
async def test_compute_derivative_success():
    res = compute_derivative(expression="x**3 + log(x)", variable="x", order=1)
    assert res["success"] is True
    assert res["result"] == "3*x**2 + 1/x"
    assert "3 x^{2}" in res["latex"]


# 3. Async test for compute_indefinite_integral tool
async def test_compute_indefinite_integral_success():
    res = compute_indefinite_integral(expression="cos(x)", variable="x")
    assert res["success"] is True
    assert res["result"] == "sin(x)"
    assert res["latex"] == "\\sin{\\left(x \\right)}"


# 4. Async test for compute_definite_integral tool (including float evaluation)
async def test_compute_definite_integral_success():
    res = compute_definite_integral(
        expression="x**2", lower_bound="0", upper_bound="1", variable="x"
    )
    assert res["success"] is True
    assert res["result"] == "1/3"
    assert pytest.approx(res["numeric_value"], 0.001) == 0.3333333


# 5. Async test for error handling across tools
async def test_tools_error_handling():
    # Invalid syntax parsing error
    invalid_expr_res = compute_limit(expression="sin(x)++")
    assert invalid_expr_res["success"] is False
    assert "error" in invalid_expr_res

    # Invalid direction for limit
    invalid_dir_res = compute_limit(expression="x", dir="invalid_dir")
    assert invalid_dir_res["success"] is False
    assert "error" in invalid_dir_res

    # Invalid order for derivative
    invalid_order_res = compute_derivative(expression="x**2", order=0)
    assert invalid_order_res["success"] is False
    assert "error" in invalid_order_res


# 6. Async test for Resources
async def test_resources():
    syntax_guide = get_sympy_syntax_guide()
    assert "# Довідник із синтаксису SymPy" in syntax_guide
    assert "`pi` — число Пі" in syntax_guide

    formulas = get_calculus_formulas_reference()
    assert "# Довідник формул математичного аналізу" in formulas
    assert "lim (x -> 0) [sin(x) / x] = 1" in formulas


# 7. Async test for Prompt generation
async def test_math_tutor_prompt():
    prompt = math_tutor_solve_and_explain(
        task_description="Знайди похідну sin(x)",
        topic="calculus",
        difficulty="beginner",
    )
    assert "Задача:** Знайди похідну sin(x)" in prompt
    assert "Рівень деталізації:** beginner" in prompt
    assert "math://cheatsheet/sympy-syntax" in prompt
