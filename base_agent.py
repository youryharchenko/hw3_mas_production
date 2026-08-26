from abc import ABC, abstractmethod
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot, interrupt


class BaseGraphAgent(ABC):
    """Базовий абстрактний клас для всіх LangGraph агентів навчальної системи."""

    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.db_path = "agent_state.db"
        # Кожен дочірній клас повинен зібрати свій граф
        self.app: CompiledStateGraph = self._compile_graph()

    @abstractmethod
    def _build_graph(self) -> Any:
        """Абстрактний метод: побудова топології StateGraph (вузли та ребра)."""
        pass

    def _compile_graph(self) -> CompiledStateGraph:
        """Внутрішня компіляція графа з підключенням чекпоїнтера."""
        workflow = self._build_graph()
        # Повертаємо скомпільований граф. Чекпоїнтер передаватиметься при виклику invoke/stream,
        # або підключається тут контекстно.
        return workflow

    # --- УНІФІКОВАНИЙ ПУБЛІЧНИЙ ІНТЕРФЕЙС ---

    def run(self, inputs) -> dict[str, Any]:
        """Запуск або початок нової сесії агента."""
        return self.app.invoke(inputs, config=self.config)

    def hitl(self, hitl_data) -> Any:
        """Припиняє виконання агента для HITL (interrupt)."""
        return interrupt(hitl_data)

    def resume(self, resume_data) -> dict[str, Any]:
        """Відновлення виконання агента після HITL (interrupt)."""
        return self.app.invoke(Command(resume=resume_data), config=self.config)

    def cont(self) -> dict[str, Any]:
        """Продовження виконання агента."""
        return self.app.invoke(None, config=self.config)

    def get_state(self) -> StateSnapshot:
        """Отримання поточного стану та снапшоту сесії."""
        return self.app.get_state(self.config)
