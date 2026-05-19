from pathlib import Path

# General config
SAMPLE_UUID_COL = "sample_uuid"
LLM_TEMPLATE_DIR = Path("./prompt_templates")


# Config for cleveland heart data

HEART_INPUT_FILE = "./data/cleveland-heart-data.csv"
HEART_OLLAMA_MODEL = "tinyllama:latest"
#HEART_OLLAMA_MODEL = "llama3.1:8b"

HEART_OUTPUT_PATH = Path("./results/heart")


# Config for Medsynth