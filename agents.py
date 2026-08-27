import copy
import operator
import re
import sqlite3
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from base_agent import BaseGraphAgent
from kb import search_templates
from llm import llm
from logger import logger
from tools import sympy_solver_tool, verify_math_expression

MAX_ITERATIONS = 2  # Максимальна кількість викликів інструменту
MAX_STEPS = 3


class SuperState(TypedDict):
    messages: Annotated[list[str], operator.add]
    current_agent: str
    topic: str
    plan: list[str]  # список кроків плану
    grade: int
    results: list[str]  # результати виконаних кроків
    current_item_idx: int  # поточний крок виконання плану
    step_count: int
    tool_call_count: int
    task: dict[str, str] | None  # Фінальний результат GeneratedMathProblem
    eval_status: str | None  # PASSED / REJECTED
    feedback: str | None


class RouteDecision(BaseModel):
    """Рішення супервізора, до якого агента надіслати запит."""

    action: Literal["plan", "exec", "eval", "general"] = Field(
        description='Цільовий агент або "general" для нерозпізнаних запитів',
    )
    reasoning: str = Field(description="Коротке пояснення вибору")


tool_allow_list = [search_templates]


class SuperAgent(BaseGraphAgent):
    """Агент керує послідовністю роботи спеціалізованих агентів"""

    def __init__(self, name: str, thread_id: str):
        self.llm = llm
        self.thread_id = thread_id
        self.logger_callback = logger

        config = {
            "recursion_limit": 15,
            "callbacks": [self.logger_callback],
            "configurable": {"thread_id": thread_id},
        }
        super().__init__(name=name, config=config)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(SuperState)

        workflow.add_edge(START, "super")

        workflow.add_node("planner", PlannerAgent("planner", self.thread_id).app)
        workflow.add_node("executor", ExecutorAgent("executor", self.thread_id).app)
        workflow.add_node("evaluator", EvaluatorAgent("evaluator", self.thread_id).app)
        workflow.add_node("super", self._super_node)

        workflow.add_edge("planner", "super")
        workflow.add_edge("executor", "super")
        workflow.add_edge("evaluator", "super")

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)

        return workflow.compile(checkpointer=saver)

    def _super_node(
        self,
        state: SuperState,
    ) -> Command[Literal["super", "planner", END]]:
        """Визначає послідовність викликів агентів."""

        step_count = state.get("step_count", 0)
        plan = state.get("plan", [])
        task = state.get("task", {})
        current_item_idx = state.get("current_item_idx", 0)
        messages = state.get("messages", [])
        if not messages:
            print("Нема запитів до супервізора")
            return Command(goto=END)

        last_message = messages[-1]

        print(f"Super - last_message: {last_message}")

        if step_count > MAX_STEPS:
            print(f"Кількість циклів генерації {step_count} досягла межі")
            return Command(goto=END, update={"step_count": 0})

        prompt = [
            SystemMessage(
                content=(
                    "Ти — супервізор MAS - шкільної програми з матетатики для 3-7 класів..\n"
                    " Маршрутизуй запит:\n"
                    "- початок роботи, розробка плану: action = 'plan' \n"
                    "- якщо план вже є, то робимо виконання пунктів плану: action = 'exec' \n"
                    "- якщо план виконано і задача вже створена, то робимо остаточну перевірку: action = 'eval'\n"
                    "- general: вітання, нерозпізнані запити\n"
                    "Поверни RouteDecision з action та коротким reasoning.\n"
                )
            ),
            HumanMessage(
                content=(
                    f"Тема: {messages[0]},\nКрок:{current_item_idx}\nПлан: \n{' '.join(plan)}, Задача: {task}, Остання дія: {last_message}\n"
                )
            ),
        ]

        response = self.llm.with_structured_output(RouteDecision).invoke(prompt)
        print(f"Super - response: {response}")

        action = response.action
        match action:
            case "plan":
                return Command(
                    goto="planner",
                    update={
                        "current_agent": "planner",
                        "step_count": 0,
                        "topic": messages[0],
                        "messages": [f"Розроби план на тему: {messages[0]}"],
                    },
                )

            case "exec":
                return Command(
                    goto="executor",
                    update={
                        "current_agent": "executor",
                        "step_count": 0,
                        "messages": [
                            f"Виконай {current_item_idx + 1}-й крок плану: {plan[current_item_idx]}"
                        ],
                    },
                )
            case "eval":
                return Command(
                    goto="evaluator",
                    update={
                        "current_agent": "evaluator",
                        "step_count": 0,
                        "messages": [f"Перевір задачу на коректність {task}"],
                    },
                )
            case _:
                return Command(
                    goto="super",
                    update={
                        "messages": [f"Невідомий маршрут: {action}"],
                        "step_count": step_count + 1,
                    },
                )

        # Визначаємо наступний крок

        return Command(goto=END)


class Plan(BaseModel):
    """План виконання задачі."""

    goal: str = Field(description="Головна ціль задачі")
    plan: list[str] = Field(description="Список кроків для досягнення цілі")


class PlannerAgent(BaseGraphAgent):
    """Агент створює покроковий план створення математичної задачі для 3-7 класу."""

    def __init__(self, name: str, thread_id: str):
        self.llm = llm

        self.logger_callback = logger

        config = {
            "recursion_limit": 15,
            "callbacks": [self.logger_callback],
            "configurable": {"thread_id": thread_id},
        }
        super().__init__(name=name, config=config)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(SuperState)

        workflow.add_edge(START, "planner")

        workflow.add_node("planner", self._planner_node)

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)

        return workflow.compile(checkpointer=saver)

    def _planner_node(
        self,
        state: SuperState,
    ) -> Command[Literal["planner", END]]:
        """Розробляє структурований план створення математичної задачі для 3-7 класу."""

        step_count = state.get("step_count", 0)

        messages = state.get("messages", [])
        if not messages:
            print("Нема запитів до планувальника")
            return Command(goto=END)

        last_message = messages[-1]

        print(f"Planner - last_message: {last_message}")

        if step_count > MAX_STEPS:
            print(f"Кількість циклів генерації плвну {step_count} досягла межі")
            return Command(
                goto=END,
                update={
                    "step_count": 0,
                    "current_agent": "super",
                    "messages": [f"Перевищено максимальне число кроків: {step_count}"],
                },
            )

        prompt = [
            SystemMessage(
                content=(
                    "Ти — методист-планувальник. Склади послідовний план з 3 кроків (ОБОВ'ЯЗКОВО 3 КРОКИ) для створення "
                    "корректної математичної задачі з заданої теми та її верифікації.\n"
                    "Кроки повинні включати:\n"
                    "1. Формулювання сюжету (тема, класс) та вибір числових даних. Приклади сюжетів отримуй через інструмент search_templates\n"
                    "2. Складання канонічного рівняння та його точне обчислення через SymPy.\n"
                    "3. Формування підсумкового JSON-документа задачі.\n"
                    "Обов'язково включай в 1 пункт плану тему (topic) та клас (grade) завдання.\n"
                    "ПРИКЛАД\n"
                    "Крок 1: Створити реалістичний і освітньо відповідний сценарій для учнів на основі шаблонів з інформаційної бази. "
                    "Тема (Topic): Арифметика: Проста задача на ділення. Клас (Grade): 4. "
                    "Cюжет, де потрібно рівномірно розподілити певну кількість предметів (наприклад, олівці, цукерки) між заданою кількістю груп або осіб. "
                    "Вибір даних: Вибираються два натуральні числа A (загальна кількість предметів) та B (кількість груп/осіб), де A має бути кратною числу B, щоб уникнути зайвих розрахунків з остатками, які можуть заплутати учня на цьому етапі. "
                    "Приклад даних: A = 72 (олівці); B = 8 (учні)\n"
                    "Крок 2: Складання канонічного рівняння та його точне обчислення через SymPy. "
                    "Перетворити сюжет на математичну модель і провести абсолютно точний розрахунок для верифікації відповіді. "
                    "Формулюється вираз, що відображає ділення A на B. Використовується бібліотека SymPy (Python) для гарантування коректності обчислень. "
                    "Результат: Обчислюється точна відповідь та перевіряється її коректність.\n"
                    "Крок 3: Формування підсумкового JSON-документа задачі. "
                    "Структурувати всі елементи (сюжет, рівняння, відповідь) у стандартизований формат для подальшого використання в освітній платформі. "
                    "Створюється об'єкт JSON, який містить метадані задачі та її компоненти. "
                    "Структура JSON: Документ повинен включати поля: \n"
                    "- topic string\n"
                    "- grade string\n"
                    "- title string\n"
                    "- problem_statement string\n"
                    "- canonical_equation string\n"
                    "- step_by_step_solution string\n"
                    "- canonical_answer string\n"
                )
            ),
            HumanMessage(content=f"Тема: '{state['topic']}', Клас: {state['grade']}"),
        ]

        response = self.llm.with_structured_output(Plan).invoke(prompt)
        print(
            f"Planner - response: \n Goal:{response.goal} \nPlan: {len(response.plan)} \n{response.plan}\n"
        )

        if response.plan:
            return Command(
                goto=END,
                update={
                    "plan": response.plan,
                    "messages": [f"План розроблено на {len(response.plan)} кроків"],
                    "current_item_idx": 0,
                    "current_agent": "super",
                    "step_count": 0,
                },
            )
        else:
            return Command(goto="planner", update={"step_count": step_count + 1})


class GeneratedMathProblem(BaseModel):
    """Фінальна структура математичної задачі."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., description="Тема з шкільної математики")
    grade: int = Field(..., description="Клас")
    title: str = Field(..., description="Коротка назва задачі")
    problem_statement: str = Field(..., description="Текст умови задачі")
    canonical_equation: str = Field(..., description="Математична модель/рівняння")
    step_by_step_solution: str = Field(
        ..., description="Покроковий еталонний розв'язок"
    )
    canonical_answer: str = Field(..., description="Фінальна коротка відповідь")

    @model_validator(mode="after")
    def validate_and_clean_problem(self) -> "GeneratedMathProblem":
        # 1. Очищення від крайових пробілів у всіх текстових полях
        self.topic = self.topic.strip()
        self.title = self.title.strip()
        self.problem_statement = self.problem_statement.strip()
        self.canonical_equation = self.canonical_equation.strip()
        self.step_by_step_solution = self.step_by_step_solution.strip()
        self.canonical_answer = self.canonical_answer.strip()

        # 2. Валідація шкільного класу (1-11)
        if not (1 <= self.grade <= 11):
            raise ValueError(
                f"Клас має бути в межах від 1 до 11, отримано: {self.grade}"
            )

        # 3. Перевірка на мінімальну довжину текстових полів
        if len(self.title) < 3:
            raise ValueError(
                "Назва задачі ('title') занадто коротка (менше 3 символів)."
            )

        if len(self.problem_statement) < 15:
            raise ValueError(
                "Текст умови задачі ('problem_statement') занадто короткий."
            )

        if len(self.step_by_step_solution) < 10:
            raise ValueError(
                "Покроковий розв'язок ('step_by_step_solution') занадто короткий."
            )

        # 4. Перевірка та чистка канонічного рівняння
        if "^" in self.canonical_equation:
            self.canonical_equation = self.canonical_equation.replace("^", "**")

        # Перевіряємо наявність знаку рівності або нерівності у математичній моделі
        valid_operators = ["=", "==", "<", ">", "<=", ">="]
        if not any(op in self.canonical_equation for op in valid_operators):
            raise ValueError(
                f"Канонічне рівняння '{self.canonical_equation}' має містити знак рівності ('=') або нерівності."
            )

        return self

    @field_validator("canonical_equation")
    def validate_equation(cls, v: str) -> str:
        v = v.strip()
        if not v:
            # Якщо модель повернула порожній рядок, можна або дати дефолтне значення,
            # або підставити заглушку замість викидання помилки:
            return "x = 0"
        if "=" not in v and not any(op in v for op in [">", "<", ">=", "<="]):
            # Якщо модель написала "1/3 * 24", перетворюємо на рівняння "x = 1/3 * 24"
            return f"x = {v}"
        return v

    @field_validator("grade", mode="before")
    def parse_grade(cls, v):
        if isinstance(v, str):
            # Витягуємо першу цифру з рядка "5 клас" -> 5
            match = re.search(r"\d+", v)
            if match:
                return int(match.group())
        return v


class ExecutorAgent(BaseGraphAgent):
    """Агент виконує один поточний крок із плану."""

    def __init__(self, name: str, thread_id: str):

        tool_allow_list = [search_templates, sympy_solver_tool, verify_math_expression]

        self.llm = llm
        self.tools = tool_allow_list
        self.logger_callback = logger
        self.tool_node = ToolNode(self.tools)
        config = {
            "recursion_limit": 15,
            "callbacks": [self.logger_callback],
            "configurable": {"thread_id": thread_id},
        }
        super().__init__(name=name, config=config)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(SuperState)

        workflow.add_edge(START, "executor")

        workflow.add_node("executor", self._executor_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge("tools", "executor")

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)

        return workflow.compile(checkpointer=saver)

    def _executor_node(
        self,
        state: SuperState,
    ) -> Command[Literal["tools", END]]:
        """Робить пояснення концепцій."""

        tool_call_count = state.get("tool_call_count", 0)
        last_message = state.get("messages")[-1]
        plan = state["plan"]
        topic = state["topic"]
        grade = state["grade"]
        idx = state["current_item_idx"]
        current_step = plan[idx]

        print(
            f"Executor - last_message: {last_message}, current_step {idx + 1}: {current_step}"
        )

        if (
            tool_call_count > 0
            and last_message.content
            and last_message.status == "success"
        ):
            # return Command(goto=END, update={"tool_call_count": 0})
            return Command(
                goto=END,
                update={
                    "tool_call_count": 0,
                    "current_item_idx": idx + 1,
                    "current_agent": "super",
                    "messages": [
                        f"Результат виконання {idx + 1}-го кроку плану: {last_message.content}"
                    ],
                },
            )

        if tool_call_count > MAX_ITERATIONS:
            print(f"Кількість викликів досягла межі {tool_call_count}")
            return Command(
                goto=END,
                update={
                    "tool_call_count": 0,
                    "messages": [f"Кількість викликів досягла межі {tool_call_count}"],
                },
            )

        messages = state.get("messages", [])

        if not messages:
            print("Нема запитів до консультанта")
            return Command(
                goto=END,
                update={"messages": "Нема запитів до консультанта"},
            )

        if idx == len(plan) - 1:
            prompt = [
                SystemMessage(
                    content=(
                        "На основі попередніх обчислень та розробки сформуй фінальну задачу у JSON.\n"
                        "Використовуй строго структуру JSON."
                        "Згенеруй JSON-об'єкт. Всі поля мають бути на ВЕРХНЬОМУ рівні JSON (без зовнішнього ключа 'task').\n"
                        "Обов'язкові поля:\n"
                        "- topic string\n"
                        "- grade string\n"
                        "- title string\n"
                        "- problem_statement string\n"
                        "- canonical_equation string\n"
                        "- step_by_step_solution string\n"
                        "- canonical_answer string\n"
                    )
                ),
                HumanMessage(
                    content=f"Контекст виконання:\n{past_context}\n\nПоточне завдання: {current_step}"
                ),
            ]
            response = llm.with_structured_output(GeneratedMathProblem).invoke(prompt)
            print(f"Executor - final response: \n {response}\n")
            task_dict = (
                response if isinstance(response, dict) else response.model_dump()
            )

            return Command(
                goto=END,
                update={
                    "task": task_dict,
                    "tool_call_count": 0,
                    "current_item_idx": idx + 1,
                    "current_agent": "super",
                    "messages": "Завершено виконання плану",
                },
            )

        prompt = [
            SystemMessage(
                content=(
                    "Ти — математик-виконавець. Виконай ПОТОЧНИЙ крок плану.\n"
                    "Якщо потрібно шаблони та приклади задач — використовуй пошук search_templates."
                    "Якщо потрібно виконати обчислення чи верифікацію — використовуй доступні SymPy інструменти."
                )
            ),
            HumanMessage(
                content=(
                    f"План:\n{plan}\n\nПОТОЧНИЙ КРОК ({idx + 1}): {current_step}\n"
                    f"Математична задача для {grade} класу на тему '{topic}'."
                )
            ),
        ]

        response = self.llm.bind_tools(self.tools).invoke(prompt)

        # Визначаємо наступний крок
        if response.tool_calls:
            print(f"Executor - response: tool_calls:\n {response.tool_calls}\n")
            return Command(
                goto="tools",
                update={"messages": [response], "tool_call_count": tool_call_count + 1},
            )
        else:
            print(f"Executor - response: \n {response.content}\n")
            return Command(
                goto=END,
                update={
                    "tool_call_count": 0,
                    "current_item_idx": idx + 1,
                    "current_agent": "super",
                    "messages": [
                        f"Результат виконання {idx + 1}-го кроку плану: {response.content}"
                    ],
                },
            )


class EvaluatorAgent(BaseGraphAgent):
    """Агент виконує остаточну перевірку навчального контенту."""

    def __init__(self, name: str, thread_id: str):

        tool_allow_list = [sympy_solver_tool, verify_math_expression]

        self.llm = llm
        self.tools = tool_allow_list
        self.logger_callback = logger
        self.tool_node = ToolNode(self.tools)
        config = {
            "recursion_limit": 15,
            "callbacks": [self.logger_callback],
            "configurable": {"thread_id": thread_id},
        }
        super().__init__(name=name, config=config)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(SuperState)

        workflow.add_edge(START, "evaluator")

        workflow.add_node("evaluator", self._evaluator_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge("tools", "evaluator")

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)

        return workflow.compile(checkpointer=saver)

    def _evaluator_node(
        self,
        state: SuperState,
    ) -> Command[Literal["tools", END]]:
        """Робить пояснення концепцій."""

        tool_call_count = state.get("tool_call_count", 0)
        last_message = state.get("messages")[-1]
        plan = state["plan"]
        topic = state["topic"]
        grade = state["grade"]
        task = state["task"]
        idx = state["current_item_idx"]
        current_step = plan[idx]

        print(f"Executor - last_message: {last_message}, task {task}")

        if (
            tool_call_count > 0
            and last_message.content
            and last_message.status == "success"
        ):
            # return Command(goto=END, update={"tool_call_count": 0})
            return Command(
                goto=END,
                update={
                    "tool_call_count": 0,
                    "current_item_idx": idx + 1,
                    "current_agent": "super",
                    "messages": [
                        f"Результат виконання контролера: {last_message.content}"
                    ],
                },
            )

        if tool_call_count > MAX_ITERATIONS:
            print(f"Кількість викликів досягла межі {tool_call_count}")
            return Command(
                goto=END,
                update={
                    "tool_call_count": 0,
                    "messages": [f"Кількість викликів досягла межі {tool_call_count}"],
                },
            )

        messages = state.get("messages", [])

        if not messages:
            print("Нема запитів до консультанта")
            return Command(
                goto=END,
                update={"messages": "Нема запитів до консультанта"},
            )

        prompt = [
            SystemMessage(
                content=(
                    "Ти — контролер якості освітнього контенту. Виконуй перевірку СУВОРО КРОК ЗА КРОКОМ:\n\n"
                    "КРОК 1: Виклич інструмент для перевірки рівняння та обчислень.\n"
                    "КРОК 2: Оціни результати:\n"
                    "   - Якщо є математичні помилки або некоректні дані -> переходь до КРОКУ 3 зі статусом 'REJECTED'.\n"
                    "   - Якщо все МАТЕМАТИЧНО КОРЕКТНО -> переходь до КРОКУ 3  зі статусом 'PASSED'.\n"
                    # "КРОК 3: ВИКЛИЧ ІНСТРУМЕНТ `save_task` з деталями цієї задачи, щоб зберегти її в LMS. (Це обов'язкова дія при PASSED!).\n"
                    "КРОК 3: Надішли фінальне текстове повідомлення-підсумок у форматі JSON з полями:\n"
                    "  - is_correct_math (boolean)\n"
                    "  - is_clear_text (boolean)\n"
                    "  - feedback (string)\n"
                    "  - status ('PASSED' або 'REJECTED')\n\n"
                    "КРИТИЧНО ВАЖЛИВО: Не повертай підсумковий JSON на Кроках 1–3! Спочатку виконай потрібні tool calls!"
                )
            ),
            HumanMessage(
                content=f"Задача:\n{task}\n\nПоточне завдання: {last_message}"
            ),
        ]

        response = llm.bind_tools(self.tools).invoke(prompt)
        print(f"Evaluator - response: \n {response}\n")

        # Визначаємо наступний крок
        if response.tool_calls:
            print(f"Evaluator - response: tool_calls:\n {response.tool_calls}\n")
            return Command(
                goto="tools",
                update={"messages": [response], "tool_call_count": tool_call_count + 1},
            )
        else:
            print(f"Evaluator - response: \n {response.content}\n")
            return Command(
                goto=END,
                update={
                    "tool_call_count": 0,
                    "current_agent": "super",
                    "messages": [f"Результат виконання перевірки: {response.content}"],
                },
            )


if __name__ == "__main__":
    agent = SuperAgent("super", "th-01")
    inputs = SuperState(
        messages=["Арифметика: Проста задача на ділення"],
        topic="",
        grade=4,
        current_agent="super",
        current_item_idx=0,
        eval_status="",
        feedback="",
        plan=[],
        results=[],
        step_count=0,
        tool_call_count=0,
        task=None,
    )
    result = agent.run(inputs)
    # print(result)
