import json
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel


class TrajectoryLogger(BaseCallbackHandler):
    def __init__(self, filepath: str = "trajectory.json"):
        self.filepath = filepath
        self.step_counter = 0
        self._start_times: dict[UUID, float] = {}
        self._inputs: dict[UUID, Any] = {}
        self._tool_names: dict[UUID, str] = {}
        self.steps = []

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_log(self, log_entry: dict[str, Any]):

        self.steps.append(log_entry)
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(self.steps, ensure_ascii=False, indent=2, default=str)
                + "\n\n"
            )

    # --- Відстеження Chain / Node ---
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Зберігаємо час початку та вхідні дані за run_id
        self._start_times[run_id] = time.perf_counter()
        self._inputs[run_id] = inputs

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._start_times.pop(run_id, time.perf_counter())
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        node_input = self._inputs.pop(run_id, None)
        inputs = self._clean_data(node_input)
        # Визначаємо назву вузла (або використовуємо дефолтну)
        node_name = kwargs.get("name") or "Chain/Node"

        self.step_counter += 1

        log_entry = {
            "step_number": self.step_counter,
            "agent_name": inputs.get("current_agent", "super")
            if hasattr(inputs, "get")
            else "super",
            "event_type": "node_execution",
            "node_name": node_name,
            "timestamp": self._get_timestamp(),
            "duration_ms": duration_ms,
            "input": inputs,
            "output": self._clean_data(outputs),
            # "tool_calls": [],  # Можна витягнути з outputs, якщо вузол повертає AIMessage з tool_calls
        }

        self._write_log(log_entry)

    # --- Відстеження Tool Calls ---
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.perf_counter()
        self._inputs[run_id] = input_str
        # Зберігаємо назву інструменту за run_id
        self._tool_names[run_id] = (
            serialized.get("name", "Tool") if serialized else "Tool"
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._start_times.pop(run_id, time.perf_counter())
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        tool_input = self._inputs.pop(run_id, None)

        tool_name = self._tool_names.pop(run_id, "Tool")

        self.step_counter += 1

        log_entry = {
            "step_number": self.step_counter,
            "event_type": "tool_execution",
            "node_name": tool_name,
            "timestamp": self._get_timestamp(),
            "duration_ms": duration_ms,
            "input": tool_input,
            "output": output,
            "tool_calls": [{"name": tool_name, "args": tool_input}],
        }

        self._write_log(log_entry)

    def _clean_data(self, data: Any) -> Any:
        """Рекурсивно очищає та конвертує об'єкти (включаючи Pydantic V1/V2,

        LangChain Messages) у JSON-сумісні типи.
        """
        # 1. Якщо це Pydantic модель (включаючи вашу Gener)
        if isinstance(data, BaseModel):
            if hasattr(data, "model_dump"):
                # Pydantic V2: mode="json" гарантує серіалізацію вкладених дат/UUID тощо
                return self._clean_data(data.model_dump(mode="json"))
            return self._clean_data(data.dict())  # Pydantic V1

        # 2. Якщо це об'єкт LangChain (наприклад AIMessage, HumanMessage)
        if hasattr(data, "lc_kwargs"):
            # Або стандартний .dict() для LangChain об'єктів
            if hasattr(data, "dict"):
                return self._clean_data(data.dict())

        # 3. Якщо це словник — очищаємо ключі та значення рекурсивно
        if isinstance(data, dict):
            return {
                str(k): self._clean_data(v)
                for k, v in data.items()
                # Ігноруємо приватні поля, якщо вони є у внутрішніх станах
                if not str(k).startswith("_")
            }

        # 4. Якщо це список або кортеж — рекурсивно обробляємо кожен елемент
        if isinstance(data, (list, tuple, set)):
            return [self._clean_data(item) for item in data]

        # 5. Базові JSON-сумісні типи повертаємо як є
        if isinstance(data, (str, int, float, bool, type(None))):
            return data

        # 6. Все інше (кастомні класи, об'єкти без dict, UUID тощо) кастимо в str
        return str(data)


logger = TrajectoryLogger()
