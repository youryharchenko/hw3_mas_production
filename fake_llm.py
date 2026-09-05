from typing import Any, Callable, Optional, Sequence, Type, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class CustomDynamicFakeLLM(BaseChatModel):
    """Кастомна фейкова LLM з власною логікою генерації."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_msg = messages[-1].content.lower()

        print(f"FakeLLM _generate - last_message: {last_msg}")

        # Динамічна логіка залежно від промпту
        if "поточний крок (1)" in last_msg:
            content = "Преший крок плану виконано"
        elif "поточний крок (2)" in last_msg:
            content = "Другий крок плану виконано"
        elif "поточний крок (3)" in last_msg:
            content = "Третій крок плану виконано"
        elif "'canonical_answer': '" in last_msg:
            content = "Перевірку виконано"
        else:
            content = "Нічого не зроблено"

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> "CustomDynamicFakeLLM":
        # Повертаємо той самий екземпляр або його копію з прапорцем bound_tools
        return self

    def with_structured_output(
        self,
        schema: Union[dict[str, Any], Type[BaseModel]],
        **kwargs: Any,
    ) -> Runnable:

        print(f"FakeLLM with_structured_output - schema: {schema.__name__}")

        def _fake_structured_runnable(input_data: Any) -> BaseModel:
            # Тут можна проаналізувати input_data (промпт) і згенерувати потрібний об'єкт

            print(
                f"FakeLLM _fake_structured_runnable - input_data: {str(input_data)[:80]}"
            )

            # prompt_text = str(input_data).lower()
            humman_message = str(input_data[1].content).lower()

            # Динамічна повернення Pydantic-об'єкта залежно від переданої схеми
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                # Якщо схема чекає RouteDecision (з вашого коду):
                if schema.__name__ == "RouteDecision":
                    print(
                        f"FakeLLM _fake_structured_runnable RouteDecision - humman_message: {humman_message}"
                    )

                    if "план: 0" in humman_message:
                        return schema(
                            action="plan",
                            reasoning="Тестова маршрутизація до планувальника",
                        )

                    if "план: 3" in humman_message and "задача: none" in humman_message:
                        return schema(
                            action="exec",
                            reasoning="Тестова маршрутизація до виконавця",
                        )

                    if (
                        "крок: 3" in humman_message
                        and "задача: {" in humman_message
                        and "перевірку виконано" not in humman_message
                    ):
                        return schema(
                            action="eval",
                            reasoning="Тестова маршрутизація до контролера",
                        )

                    if (
                        "крок: 3" in humman_message
                        and "задача: {" in humman_message
                        and "перевірку виконано" in humman_message
                    ):
                        return schema(
                            action="finish",
                            reasoning="Тестова маршрутизація до завершення",
                        )
                # Якщо схема чекає Plan:
                elif schema.__name__ == "Plan":
                    return schema(
                        goal="Тестова ціль",
                        plan=["Крок 1", "Крок 2", "Крок 3"],
                    )
                # Якщо схема чекає Plan:
                elif schema.__name__ == "GeneratedMathProblem":
                    return schema(
                        topic="Арифметика: Проста задача на ділення",
                        grade=4,
                        title="Задача для 4-го классу",
                        problem_statement="Тут має бути текст задачі",
                        canonical_equation="Тут має бути канонічне рівняння",
                        step_by_step_solution="Тут має бути покроковий розв'язок",
                        canonical_answer="Тут має бути канонічна відповідь",
                    )

                # Дефолтний фолбек для будь-якої іншої Pydantic-схеми
                return schema()

            raise ValueError("Непідтримуваний тип схеми")

        # Повертаємо обгортку, яку LangChain сприйме як готовий Runnable
        return RunnableLambda(_fake_structured_runnable)

    @property
    def _llm_type(self) -> str:
        return "custom_fake_llm"


llm = CustomDynamicFakeLLM()
