import pandas as pd
import ollama
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "llama3.1:8b"
INPUT_FILE = "./data/MedSynth_huggingface_final.csv"
OUTPUT_DIR = Path("./results/medsynth_ollama")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR = Path("./prompt_templates")

# -----------------------------
# Load input conversations
# -----------------------------
df = pd.read_csv(INPUT_FILE)

# -----------------------------
# Prompt template
# -----------------------------

jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True
)

prompt_template = jinja_env.get_template("medsynth.jinja")


# Helper: run Ollama
def run_llm(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0.2,
            "top_p": 0.9
        }
    )
    return response["message"]["content"].strip()


def parse_output(text: str) -> dict:
    try:
        data = json.loads(text)

        llm_icd10_chapter = data.get("llm_icd10_chapter", "")
        explanation = data.get("explanation", "")

    except json.JSONDecodeError:
        # if invalid JSON
        llm_icd10_chapter = ""
        explanation = ""

    return {"llm_icd10_chapter": llm_icd10_chapter,
        "llm_explanation": explanation,
        "raw_output": text }

# Run LLM over conversations
outputs = []

for idx, row in df.head(10).iterrows():
    print(f"Running LLM on conversation {idx + 1}/{len(df)} ")

    prompt = prompt_template.render(conversation=row['Dialogue'])

    llm_text = run_llm(prompt)
    parsed = parse_output(llm_text)

    outputs.append({
        "id": idx,
        "ICD10": row["ICD10"],
        "llm_icd10_chapter": parsed["llm_icd10_chapter"],
        "llm_explanation": parsed["llm_explanation"],
    })

# Save outputs
out_df = pd.DataFrame(outputs)

out_df.to_csv(OUTPUT_DIR / "llm_outputs_full.csv", index=False)

print("LLM inference complete.")
print(f"Saved to {OUTPUT_DIR}")