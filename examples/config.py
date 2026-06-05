from pathlib import Path

# General config
SAMPLE_UUID_COL = "sample_uuid"
LLM_TEMPLATE_DIR = Path("./prompt_templates")


# Config for cleveland heart data

HEART_INPUT_FILE = "./data/cleveland-heart-data.csv"
#HEART_OLLAMA_MODEL = "tinyllama:latest"
HEART_OLLAMA_MODEL = "llama3.1:8b"

# HEART_OLLAMA_MODEL = "qwen2.5:32b-instruct-q4_K_M"
# HEART_OLLAMA_MODEL = "gemma3:27b"


# Config for Medsynth
MEDSYNTH_FULL_FILE = "./data/medsynth_data_full.csv"
MEDSYNTH_OLLAMA_MODEL = "tinyllama:latest"
#MEDSYNTH_OLLAMA_MODEL = "llama3.1:8b"
# MEDSYNTH_OLLAMA_MODEL = "qwen2.5:32b-instruct-q4_K_M"
# MEDSYNTH_OLLAMA_MODEL = "gemma3:27b"