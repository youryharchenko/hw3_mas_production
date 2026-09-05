from typing import Any, Dict, Optional

import sympy as sp
from fastmcp.server import FastMCP

# 1. Ініціалізуємо сервер
mcp = FastMCP("Server")


# 2. Додаємо Tools (Інструменти) з автоматичною валідацією типів


def _parse_expression(expr_str: str) -> sp.Expr:
    """Допоміжна функція для безпечного парсингу рядка у вираз SymPy."""
    if not expr_str or not expr_str.strip():
        raise ValueError("Рядок виразу не може бути порожнім.")
    try:
        # sympify перетворює математичний рядок у вираз SymPy
        parsed = sp.sympify(expr_str)
        return parsed
    except Exception as e:
        raise ValueError(
            f"Не вдалося розпарсити вираз '{expr_str}'. Перевірте синтаксис: {e}"
        )


@mcp.tool()
def compute_limit(
    expression: str,
    variable: str = "x",
    point: str = "0",
    dir: str = "+-",
) -> Dict[str, Any]:
    """Обчислює границю математичної функції (Limit).

    Args:
        expression: Математичний вираз у вигляді рядка (наприклад, "sin(x)/x", "(1 + 1/n)**n").
        variable: Змінна, по якій шукається границя (за замовчуванням "x").
        point: Точка, до якої прагне змінна (наприклад, "0", "oo" для нескінченності, "-oo").
        dir: Напрямок границі: "+-" (двостороння), "+" (справа), "-" (зліва).

    Returns:
        Словник із результатами обчислення у форматах:
        - success (bool): Прапор успішного виконання.
        - result (str): Результат у вигляді рядка SymPy.
        - latex (str): Результат у форматі LaTeX.
        - error (str, optional): Опис помилки у разі невдачі.
    """
    try:
        expr = _parse_expression(expression)
        var_symbol = sp.Symbol(variable)
        point_symbol = _parse_expression(point)

        if dir not in ["+-", "+", "-"]:
            raise ValueError("Напрямок dir повинен бути одним із: '+-', '+', '-'")

        limit_res = sp.limit(expr, var_symbol, point_symbol, dir=dir)

        return {
            "success": True,
            "result": str(limit_res),
            "latex": sp.latex(limit_res),
        }
    except Exception as e:
        return {"success": False, "error": f"Помилка обчислення границі: {str(e)}"}


@mcp.tool()
def compute_derivative(
    expression: str,
    variable: str = "x",
    order: int = 1,
) -> Dict[str, Any]:
    """Обчислює похідну математичної функції (Derivative / Differentiation).

    Args:
        expression: Математичний вираз для диференціювання (наприклад, "x**3 + log(x)").
        variable: Змінна, за якою береться похідна (за замовчуванням "x").
        order: Порядок похідної (ціле додатне число >= 1, за замовчуванням 1).

    Returns:
        Словник із результатами обчислення у форматах:
        - success (bool): Прапор успішного виконання.
        - result (str): Знайдена похідна у вигляді рядка.
        - latex (str): Похідна у форматі LaTeX.
        - error (str, optional): Опис помилки у разі невдачі.
    """
    try:
        if not isinstance(order, int) or order < 1:
            raise ValueError("Порядок похідної 'order' має бути цілим числом >= 1")

        expr = _parse_expression(expression)
        var_symbol = sp.Symbol(variable)

        derivative_res = sp.diff(expr, var_symbol, order)

        return {
            "success": True,
            "result": str(derivative_res),
            "latex": sp.latex(derivative_res),
        }
    except Exception as e:
        return {"success": False, "error": f"Помилка диференціювання: {str(e)}"}


@mcp.tool()
def compute_indefinite_integral(
    expression: str,
    variable: str = "x",
) -> Dict[str, Any]:
    """Обчислює невизначений інтеграл функції (Indefinite Integral / Antiderivative).

    Args:
        expression: Математичний вираз для інтегрування (наприклад, "exp(x) * sin(x)").
        variable: Змінна інтегрування (за замовчуванням "x").

    Returns:
        Словник із результатами обчислення у форматах:
        - success (bool): Прапор успішного виконання.
        - result (str): Первісна функція у вигляді рядка (без константи + C).
        - latex (str): Результат у форматі LaTeX.
        - error (str, optional): Опис помилки у разі невдачі.
    """
    try:
        expr = _parse_expression(expression)
        var_symbol = sp.Symbol(variable)

        integral_res = sp.integrate(expr, var_symbol)

        return {
            "success": True,
            "result": str(integral_res),
            "latex": sp.latex(integral_res),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Помилка обчислення невизначеного інтеграла: {str(e)}",
        }


@mcp.tool()
def compute_definite_integral(
    expression: str,
    lower_bound: str,
    upper_bound: str,
    variable: str = "x",
) -> Dict[str, Any]:
    """Обчислює визначений інтеграл на заданому проміжку (Definite Integral).

    Args:
        expression: Математичний вираз для інтегрування (наприклад, "x**2").
        lower_bound: Нижня межа інтегрування (наприклад, "0", "-oo").
        upper_bound: Верхня межа інтегрування (наприклад, "1", "oo", "pi").
        variable: Змінна інтегрування (за замовчуванням "x").

    Returns:
        Словник із результатами обчислення у форматах:
        - success (bool): Прапор успішного виконання.
        - result (str): Обчислене значення інтеграла у вигляді рядка.
        - numeric_value (float/None): Числове наближене значення, якщо можливо обчислити.
        - latex (str): Результат у форматі LaTeX.
        - error (str, optional): Опис помилки у разі невдачі.
    """
    try:
        expr = _parse_expression(expression)
        var_symbol = sp.Symbol(variable)
        a = _parse_expression(lower_bound)
        b = _parse_expression(upper_bound)

        integral_res = sp.integrate(expr, (var_symbol, a, b))

        # Спробуємо отримати точне числове наближення (float)
        numeric_val = None
        try:
            eval_val = integral_res.evalf()
            if eval_val.is_number and not eval_val.has(sp.I):
                numeric_val = float(eval_val)
        except Exception:
            numeric_val = None

        return {
            "success": True,
            "result": str(integral_res),
            "numeric_value": numeric_val,
            "latex": sp.latex(integral_res),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Помилка обчислення визначеного інтеграла: {str(e)}",
        }


# 3. Додаємо Resources (Ресурси)


@mcp.resource("math://cheatsheet/sympy-syntax")
def get_sympy_syntax_guide() -> str:
    """Довідник із синтаксису SymPy для правильного формування виразів.

    Повертає текстову інструкцію щодо запису математичних констант,
    операторів та спеціальних функцій у вигляді рядків SymPy.
    """
    guide = """
# Довідник із синтаксису SymPy (Cheatsheet)

Для коректної передачі виразів у SymPy інструменти використовуйте наступні правила:

## 1. Основні оператори
- Множення: обов'язково використовувати зірочку `*` (наприклад, `2*x`, а не `2x`).
- Піднесення до степеня: подвійна зірочка `**` (наприклад, `x**2`, а не `x^2`).
- Ділення: знак `/` (наприклад, `1/(x + 1)`).

## 2. Математичні константи
- `pi` — число Пі (3.14159...)
- `E` — число Ейлера / база натурального логарифма e (2.71828...)
- `I` — уявна одиниця (i)
- `oo` — позитивна нескінченність (+∞)
- `-oo` — негативна нескінченність (-∞)

## 3. Вбудовані функції
- Тригонометрія: `sin(x)`, `cos(x)`, `tan(x)`, `cot(x)`
- Обернені тригонометричні: `asin(x)`, `acos(x)`, `atan(x)`
- Показникові та логарифмічні: `exp(x)` (замість e**x), `log(x)` (натуральний логарифм ln), `log(x, 10)` (за основою 10)
- Корені: `sqrt(x)` (квадратний корінь), `x**(1/3)` (кубічний корінь)
- Модуль: `Abs(x)`

## 4. Приклади складних виразів
- Границя: `sin(3*x)/(2*x)`
- Похідна: `exp(-x**2) * log(x + 1)`
- Інтеграл: `1 / sqrt(1 - x**2)`
"""
    return guide.strip()


@mcp.resource("math://reference/calculus-formulas")
def get_calculus_formulas_reference() -> str:
    """Довідник базових формул математичного аналізу.

    Повертає шпаргалку з формулами границь, похідних та інтегралів у форматі Markdown.
    """
    formulas = """
# Довідник формул математичного аналізу

## 1. Замечательные пределы (Чудові границі)
- lim (x -> 0) [sin(x) / x] = 1
- lim (x -> 0) [(1 + x)**(1/x)] = E
- lim (x -> oo) [(1 + 1/x)**x] = E
- lim (x -> 0) [(exp(x) - 1) / x] = 1

## 2. Таблиця похідних (Basic Derivatives)
- d/dx (c) = 0
- d/dx (x**n) = n * x**(n - 1)
- d/dx (exp(x)) = exp(x)
- d/dx (a**x) = a**x * log(a)
- d/dx (log(x)) = 1 / x
- d/dx (sin(x)) = cos(x)
- d/dx (cos(x)) = -sin(x)
- d/dx (tan(x)) = 1 / cos(x)**2
- d/dx (atan(x)) = 1 / (1 + x**2)
- d/dx (asin(x)) = 1 / sqrt(1 - x**2)

## 3. Таблиця невизначених інтегралів (Basic Integrals)
- int (x**n) dx = x**(n + 1) / (n + 1)  [для n != -1]
- int (1 / x) dx = log(Abs(x))
- int (exp(x)) dx = exp(x)
- int (sin(x)) dx = -cos(x)
- int (cos(x)) dx = sin(x)
- int (1 / (1 + x**2)) dx = atan(x)
- int (1 / sqrt(1 - x**2)) dx = asin(x)
"""
    return formulas.strip()


# 4. Додаємо Prompt (Шаблон)


# @mcp.prompt()
def math_tutor_solve_and_explain(
    task_description: str,
    topic: str = "calculus",
    difficulty: Optional[str] = "beginner",
) -> str:
    """Шаблон промпта для покрокового розв'язання математичних задач з поясненнями.

    Args:
        task_description: Текст математичної задачі або вираз (наприклад, "Знайди похідну x*sin(x)").
        topic: Розділ математики (наприклад, "calculus", "limits", "integrals").
        difficulty: Рівень складності пояснення ("beginner", "intermediate", "advanced").

    Returns:
        Системна інструкція для LLM з підключенням довідників та правил використання інструментів.
    """
    prompt = f"""
Ти — професійний викладач вищої математики та репетитор.

Твоя мета: Покроково розв'язати та пояснити математичну задачу користувачу.

---
### Вхідні дані:
- **Тема:** {topic}
- **Рівень деталізації:** {difficulty}
- **Задача:** {task_description}

---
### Правила виконання:
1. **Перевірка синтаксису SymPy:**
   Перед використанням інструментів переглянь довідник `math://cheatsheet/sympy-syntax` для правильного формування рядків виразів (наприклад, використовуй `**` замість `^`, `E` замість `e`, `*` для множення).

2. **Використання інструментів:**
   Обов'язково використовуй відповідні доступні інструменти (`compute_limit`, `compute_derivative`, `compute_indefinite_integral`, `compute_definite_integral`) для отримання точного символьного результату та LaTeX-коду. Не обчислюй складні інтеграли чи похідні в голові!

3. **Структура відповіді:**
   - **Умова:** Чітко запиши математичну задачу у форматі LaTeX.
   - **Теорія:** Коротко нагадай базову формулу з довідника `math://reference/calculus-formulas`, яка застосовується для розв'язку.
   - **Кроки розв'язання:** Покроково розпиши хід розв'язування, пояснюючи проміжні перетворення відповідно до рівня складності '{difficulty}'.
   - **Фінальний результат:** Виведіть точну відповідь, отриману через SymPy інструмент, та її LaTeX-версію.
"""
    return prompt.strip()


if __name__ == "__main__":
    mcp.run(transport="stdio")
