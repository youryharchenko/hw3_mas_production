import copy
import operator
import sqlite3
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from base_agent import BaseGraphAgent
from kb import search_templates
from llm import llm
from logger import logger

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

    action: Literal["plan", "exec", "general"] = Field(
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
        workflow.add_node("super", self._super_node)

        workflow.add_edge("planner", "super")

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
                    "- якщо план вже є то, робимо виконання пунктів плану: action = 'exec' \n"
                    # "- researcher: довідкові питання — 'як', 'які правила', 'що таке X'\n"
                    "- general: вітання, нерозпізнані запити\n"
                    "Поверни RouteDecision з action та коротким reasoning.\n"
                )
            ),
            HumanMessage(
                content=f"Тема: {messages[0]},\nКрок:{current_item_idx}\nПлан: \n{' '.join(plan)}\n"
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
                    },
                )

            case "exec":
                return Command(goto=END)
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
                    # "messages": [response.plan[0]],
                    "current_item_idx": 0,
                    "current_agent": "super",
                    "step_count": 0,
                },
            )
        else:
            return Command(goto="planner", update={"step_count": step_count + 1})


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
    print(result)
