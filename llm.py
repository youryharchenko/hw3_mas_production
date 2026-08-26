# ==========================================
# 2. НАЛАШТУВАННЯ МОДЕЛЕЙ
# ==========================================
# MODEL_NAME = "qwen2.5-coder:7b"
# MODEL_NAME = "qwen3.5:9b"
from langchain_ollama import ChatOllama

MODEL_NAME = "gemma4:e4b"
# MODEL_NAME = "qwen2.5-coder:14b"
OLLAMA_SERVER_IP = "192.168.2.102"

TIMEOUT_SEC = 60  # Загальний тайм-аут у секундах

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.1,
    num_predict=8192,
    # reasoning=False,
    base_url=f"http://{OLLAMA_SERVER_IP}:11434",
    client_kwargs={"timeout": TIMEOUT_SEC},
)
