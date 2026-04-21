import pandas as pd
import ollama
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Config
MODEL_NAME = "llama3.1:8b"
INPUT_FILE = "./data/cleveland-heart-data.csv"
OUTPUT_DIR = Path("./results/heart_ollama")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR = Path("./prompt_templates")

COLUMN_MAPPINGS = {
    'sex': { 0: 'Female', 1: 'Male'},
    'cp': { 1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'},
    'fbs': { 0: 'false', 1: 'true'},
    'restecg': { 0: 'normal', 1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy'},
    'exang': { 0: 'no', 1: 'yes'},
    'slope': { 1: 'upsloping', 2: 'flat', 3: 'downsloping'},
    'thal': { 3: 'normal', 6: 'fixed defect', 7: 'reversible defect'}
}

df = pd.read_csv(INPUT_FILE)

for col, mapping in COLUMN_MAPPINGS.items():
    df[col] = df[col].map(mapping)

print(df.head())

jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True
)

prompt_template = jinja_env.get_template("heart.jinja")

def run_llm(prompt):
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

        score = data.get("score", "")
        explanation = data.get("explanation", "")

    except json.JSONDecodeError:
        # if invalid JSON
        score = ""
        explanation = ""

    return {"llm_score": score,
        "llm_explanation": explanation,
        "raw_output": text }

outputs = []

for idx, row in df.head(5).iterrows():
    print(f"Running LLM on conversation {idx + 1}/{len(df)} ")

    prompt = prompt_template.render(
        age=row['age'],
        sex=row['sex'],
        cp=row['cp'],
        trestbps=row['trestbps'],
        chol=row['chol'],
        fbs=row['fbs'],
        restecg=row['restecg'],
        thalach=row['thalach'],
        exang=row['exang'],
        oldpeak=row['oldpeak'],
        slope=row['slope'],
        ca=row['ca'],
        thal=row['thal'])

    llm_text = run_llm(prompt)
    parsed_output = parse_output(llm_text)

    outputs.append({
        "idx": idx+1,
        "target": row["num"],
        "predicted_score": parsed_output["llm_score"],
        "cor_explanation": parsed_output["llm_explanation"],
    })

out_df = pd.DataFrame(outputs)

out_df.to_csv(OUTPUT_DIR / "llm_outputs_full.csv", index=False)