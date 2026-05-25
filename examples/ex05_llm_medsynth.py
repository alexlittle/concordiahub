import pandas as pd
import ollama
import json

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from examples import config

TEMPLATE_DIR = Path("./prompt_templates")

# -----------------------------
# Load input conversations
# -----------------------------
df = pd.read_csv(config.MEDSYNTH_FULL_FILE)

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
        model=config.MEDSYNTH_OLLAMA_MODEL,
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

        prediction = data.get("prediction", "")
        explanation = data.get("explanation", "")

    except json.JSONDecodeError:
        # if invalid JSON
        prediction = ""
        explanation = ""

    return {"prediction": prediction,
        "explanation": explanation,}

# Run LLM over conversations
outputs = []

for idx, row in df.head(300).iterrows():
    print(f"Running LLM on conversation {idx + 1}/{len(df)} ")

    prompt = prompt_template.render(conversation=row['Dialogue'])

    llm_text = run_llm(prompt)
    parsed = parse_output(llm_text)

    outputs.append({
        config.SAMPLE_UUID_COL: row[config.SAMPLE_UUID_COL],
        "ICD_chapter": row["ICD_chapter"],
        "prediction": parsed["prediction"],
        "explanation": parsed["explanation"],
    })

# Save outputs
out_df = pd.DataFrame(outputs)

X = df.drop(columns=["ICD_chapter"])
y = df[config.SAMPLE_UUID_COL, "ICD_chapter"]
y_preds = out_df[config.SAMPLE_UUID_COL, "prediction"]
y_exp = out_df[config.SAMPLE_UUID_COL, "explanation"]

X.to_parquet('./cache/medsynth/llm/X.parquet')
y.to_parquet('./cache/medsynth/llm/y.parquet')
y_preds.to_frame().to_parquet('./cache/medsynth/llm/y_preds.parquet')
y_exp.to_frame().to_parquet('./cache/medsynth/llm/y_exp.parquet')
