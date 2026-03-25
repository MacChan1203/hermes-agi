"""Hermes AGI 共有定数。"""

# Mistral / Ollama
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
MISTRAL_API_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MISTRAL_MODEL = "mistral"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
AI_GATEWAY_MODELS_URL = f"{AI_GATEWAY_BASE_URL}/models"
AI_GATEWAY_CHAT_URL = f"{AI_GATEWAY_BASE_URL}/chat/completions"

NOUS_API_BASE_URL = "https://inference-api.nousresearch.com/v1"
NOUS_API_CHAT_URL = f"{NOUS_API_BASE_URL}/chat/completions"

# Groq (無料ティア・OpenAI互換・高速)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
